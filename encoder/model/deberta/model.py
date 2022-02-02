# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Adapted from KEAR https://github.com/microsoft/KEAR/blob/main/model/model.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from transformers import AutoModel
from encoder.utils.settings import proxies, model_cache_dir, huggingface_mirror
from .layers import ChoicePredictor, ChoiceClassifier
from .perturbation import SmartPerturbation
from .sift import SiFTAdversarialLearner, hook_sift_layer
from .loss import LOSS_REGISTRY, LossCriterion


class ModelOutput:
    pass


class Model(nn.Module):
    """
    BERT style classifier, with VAT (Virtual adversarial training)

    1. self.forward(input_ids, attention_mask, token_type_ids, label)
    2. self.predict(input_ids, attention_mask, token_type_ids)
    """

    def __init__(
        self,
        base_type,
        choice_num,
        adversarial_training: bool = False,
        adversarial_training_use_sift: bool = False,
        adversarial_training_config: dict = None,
        choice_predictor_type: str = "predictor",
        choice_predictor_dropout_prob: float = 0.0,
        choice_predictor_dataset_names: List[str] = None,
        regularize_with_bce_loss: bool = False,
        bce_loss_weight: float = 0.02,
    ):
        super(Model, self).__init__()

        if not (
            ("albert" in base_type)
            or ("deberta" in base_type)
            or ("electra" in base_type)
            or ("roberta" in base_type)
        ):
            raise ValueError(f"Model type {base_type} not supported.")

        self.base = AutoModel.from_pretrained(
            base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            return_dict=True,
        )
        if choice_predictor_type not in ("classifier", "predictor", "mixed"):
            raise ValueError(f"Invalid choice_predictor_type {choice_predictor_type}")
        choice_predictor_cls = (
            ChoiceClassifier
            if choice_predictor_type == "classifier"
            else ChoicePredictor
        )
        if choice_predictor_dataset_names is None:
            choice_predictors = {
                "default": choice_predictor_cls(
                    self.base.config.hidden_size,
                    choice_num,
                    choice_predictor_dropout_prob,
                )
            }
        else:
            choice_predictors = {
                dataset: choice_predictor_cls(
                    self.base.config.hidden_size,
                    choice_num,
                    choice_predictor_dropout_prob,
                )
                for dataset in choice_predictor_dataset_names
            }
        if choice_predictor_type == "mixed":
            reranker = {}
            for name in choice_predictors:
                reranker[name + "_reranker"] = ChoiceClassifier(
                    self.base.config.hidden_size,
                    choice_num,
                    choice_predictor_dropout_prob,
                )
            choice_predictors.update(reranker)
        self.choice_predictor_type = choice_predictor_type
        self.choice_predictors = nn.ModuleDict(choice_predictors)
        self.adversarial_training = adversarial_training
        self.adversarial_training_use_sift = adversarial_training_use_sift
        self.adversarial_training_config = adversarial_training_config
        self.regularize_with_bce_loss = regularize_with_bce_loss
        self.bce_loss_weight = bce_loss_weight
        if adversarial_training:
            if adversarial_training_use_sift:
                adv_modules = hook_sift_layer(
                    self,
                    hidden_size=self.hidden_size,
                    learning_rate=adversarial_training_config["adv_step_size"],
                    init_perturbation=adversarial_training_config["adv_noise_var"],
                )
                self.adv_teacher = SiFTAdversarialLearner(self, adv_modules)
            else:
                # Use SMART instead
                cs = adversarial_training_config["adv_loss"]
                lc = LOSS_REGISTRY[LossCriterion[cs]](name=f"Adv Loss func: {cs}")
                self.adv_task_loss_criterion = [lc]
                self.adv_teacher = SmartPerturbation(
                    adversarial_training_config["adv_epsilon"],
                    adversarial_training_config["adv_step_size"],
                    adversarial_training_config["adv_noise_var"],
                    adversarial_training_config["adv_p_norm"],
                    adversarial_training_config["adv_k"],
                    loss_map=self.adv_task_loss_criterion,
                    norm_level=adversarial_training_config["adv_norm_level"],
                )

    def embed_encode(self, input_ids, token_type_ids=None, _attention_mask=None):
        """
        This method is used by Smart (SmartPerturbation)

        input_ids: LongTensor of shape [batch_size, choice_num, seq_length]
        token_type_ids: LongTensor of shape [batch_size, choice_num, seq_length]
        """
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        embedding_output = self.base.embeddings(flat_input_ids, flat_token_type_ids)
        return embedding_output

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        labels,
        dataset_name=None,
        choice_mask=None,
    ):
        """

        Args:
            input_ids: LongTensor of shape [batch_size, choice_num, seq_length]
            attention_mask: FloatTensor of shape [batch_size, choice_num, seq_length]
            token_type_ids: LongTensor of shape [batch_size, choice_num, seq_length]
            labels: LongTensor of shape [batch_size,]
            dataset_name: String indicating which dataset it is from
            choice_mask: FloatTensor of shape [batch_size, choice_num]

        Returns:
            loss: Loss of training.
            right_num: Integer number of right predictions
            logits: Masked (If choice mask is present) logits, FloatTensor of shape [batch_size, choice_num]
            adv_norm: Norm of adversarial training
        """
        if self.choice_predictor_type == "mixed":
            logits, reranked_logits = self._forward(
                input_ids, attention_mask, token_type_ids, with_rerank=True
            )

            if choice_mask is not None:
                clf_logits = choice_mask * -1e7 + logits
                clf_reranked_logits = choice_mask * -1e7 + reranked_logits
            else:
                clf_logits = logits
                clf_reranked_logits = reranked_logits
            rank_loss = F.cross_entropy(
                clf_logits, labels.view(-1), reduction="none"
            ) + F.cross_entropy(clf_reranked_logits, labels.view(-1), reduction="none")
        else:
            logits = self._forward(input_ids, attention_mask, token_type_ids)

            # Masking for multi-task training where choice number is different for each dataset
            if choice_mask is not None:
                clf_logits = choice_mask * -1e7 + logits
            else:
                clf_logits = logits

            rank_loss = F.cross_entropy(clf_logits, labels.view(-1), reduction="none")

        if self.regularize_with_bce_loss:
            target = torch.zeros_like(clf_logits)
            target[range(clf_logits.shape[0]), labels.view(-1)] = 1
            label_loss = F.binary_cross_entropy_with_logits(
                clf_logits, target, reduction="none"
            ).mean(dim=1)
            loss = (
                rank_loss * (1 - self.bce_loss_weight)
                + label_loss * self.bce_loss_weight
            )
        else:
            loss = rank_loss

        if self.adversarial_training and self.training:
            if self.adversarial_training_use_sift:
                adv_loss, adv_norm = self.adv_teacher.loss(
                    logits,
                    self._forward,
                    self.adversarial_training_config["grad_adv_loss"],
                    self.adversarial_training_config["adv_loss"],
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    dataset_name,
                )
                loss = loss + self.adversarial_training_config["adv_alpha"] * adv_loss
            else:
                adv_loss, adv_norm, adv_logits = self.adv_teacher.forward(
                    self,
                    logits,
                    input_ids,
                    attention_mask,
                    token_type_ids,
                    dataset_name,
                )
                if adv_loss is None:
                    adv_loss = torch.zeros_like(loss)
                    adv_norm = adv_loss
                loss = loss + self.adversarial_training_config["adv_alpha"] * adv_loss
        else:
            adv_norm = None

        with torch.no_grad():
            predicts = torch.argmax(clf_logits, dim=1)
            right_num = int(torch.sum(predicts == labels))

        result = ModelOutput()
        result.loss = loss.mean()
        result.right_num = right_num
        result.clf_logits = clf_logits
        result.adv_norm = adv_norm
        return result

    def predict(self, input_ids, attention_mask, token_type_ids, dataset_name=None):
        """
        Args:
            input_ids: LongTensor of shape [batch_size, choice_num, seq_length]
            attention_mask: FloatTensor of shape [batch_size, choice_num, seq_length]
            token_type_ids: LongTensor of shape [batch_size, choice_num, seq_length]
            dataset_name: String indicating which dataset it is from
        """
        if self.choice_predictor_type == "mixed":
            logits, reranked_logits = self._forward(
                input_ids,
                attention_mask,
                token_type_ids,
                dataset_name=dataset_name,
                with_rerank=True,
            )
            score = torch.sigmoid(logits)
            for i, row in enumerate(score):
                if torch.sum(row > 0.1) > 1:
                    logits[i] = reranked_logits[i]
        else:
            logits = self._forward(
                input_ids, attention_mask, token_type_ids, dataset_name=dataset_name
            )
        return logits.detach()

    def _forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        dataset_name=None,
        inputs_embeds=None,
        with_rerank=False,
    ):
        if inputs_embeds is None:
            flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        else:
            flat_input_ids = None

        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        outputs = self.base(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            token_type_ids=flat_token_type_ids,
            inputs_embeds=inputs_embeds,
        )
        dataset_name = dataset_name or "default"

        if with_rerank:
            if self.choice_predictor_type != "mixed":
                raise ValueError("choice_predictor_type is not mixed")
            return (
                self.choice_predictors[dataset_name](outputs.last_hidden_state),
                self.choice_predictors[dataset_name + "_reranker"](
                    outputs.last_hidden_state.detach()
                ),
            )
        else:
            return self.choice_predictors[dataset_name](outputs.last_hidden_state)
