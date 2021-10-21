import torch as t
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BatchEncoding
from .kb_trainer import KBEncoderTrainer
from ..model.ext_vocab import ExtendVocabForQA
from ..dataset.base import EmptyDataset, collate_function_dict_to_batch_encoding
from ..dataset.qa.squad import SQuADDataset
from ..utils.config import QATrainConfig
from ..utils.settings import proxies, model_cache_dir, huggingface_mirror


class QATrainer(pl.LightningModule):
    def __init__(
        self, config: QATrainConfig, stage_result_path="./", is_distributed=False
    ):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.stage_result_path = stage_result_path
        self.is_distributed = is_distributed
        self.kb_encoder = KBEncoderTrainer.load_from_checkpoint(
            config.kb_encoder_path, only_init_model=True
        ).kb_model

        self.qa_model = ExtendVocabForQA(
            base_type=config.base_type,
            extend_config=config.extend_config,
            extend_mode=config.extend_mode,
            **config.base_configs,
        )
        self.qa_tokenizer = AutoTokenizer.from_pretrained(
            config.base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
        )

        initialized_datasets = {}
        if config.train_dataset_path is None:
            self.train_qa_dataset = None
        elif "squad" in config.train_dataset_path:
            if config.train_dataset_path in initialized_datasets:
                self.train_qa_dataset = initialized_datasets[config.train_dataset_path]
            else:
                self.train_qa_dataset = SQuADDataset(
                    dataset_path=config.train_dataset_path, tokenizer=self.qa_tokenizer
                )
                initialized_datasets[config.train_dataset_path] = self.train_qa_dataset
        else:
            raise ValueError(
                f"Unknown QATrainConfig.train_dataset_path: {config.train_dataset_path}"
            )

        if config.validate_dataset_path is None:
            self.validate_qa_dataset = None
        elif "squad" in config.validate_dataset_path:
            if config.validate_dataset_path in initialized_datasets:
                self.validate_qa_dataset = initialized_datasets[
                    config.validate_dataset_path
                ]
            else:
                self.validate_qa_dataset = SQuADDataset(
                    dataset_path=config.validate_dataset_path,
                    tokenizer=self.qa_tokenizer,
                )
                initialized_datasets[
                    config.validate_dataset_path
                ] = self.validate_qa_dataset
        else:
            raise ValueError(
                f"Unknown QATrainConfig.validate_dataset_path: "
                f"{config.validate_dataset_path}"
            )

    @property
    def monitor(self):
        if "squad" in self.config.train_dataset_path:
            # See https://github.com/huggingface/datasets/blob/
            # master/metrics/squad_v2/squad_v2.py
            # "f1" or "exact" for EM score
            return "f1"
        else:
            return "f1"

    def train_dataloader(self):
        if self.train_qa_dataset is not None:
            return DataLoader(
                dataset=self.train_qa_dataset.train_dataset,
                batch_size=self.config.batch_size,
                collate_fn=collate_function_dict_to_batch_encoding,
            )
        else:
            return DataLoader(dataset=EmptyDataset())

    def val_dataloader(self):
        if self.validate_qa_dataset is not None:
            return DataLoader(
                dataset=self.validate_qa_dataset.validate_dataset,
                batch_size=self.config.batch_size,
                collate_fn=collate_function_dict_to_batch_encoding,
            )
        else:
            return DataLoader(dataset=EmptyDataset())

    # noinspection PyTypeChecker
    def training_step(self, batch: BatchEncoding, batch_idx):
        batch.convert_to_tensors("pt")
        with_gradient_num = (
            0
            if not self.config.kb_encoder_trainable
            else self.config.kb_encoder_with_gradient_num
        )
        kb_embeds = self.kb_encoder.compute_sentence_embeds(
            sentence_tokens=batch["input_ids"].to(self.device),
            context_length=self.config.context_length,
            with_gradient_num=with_gradient_num,
        )
        extend_tokens = t.where(
            (batch["input_ids"] != self.qa_tokenizer.cls_token_id)
            & (batch["input_ids"] != self.qa_tokenizer.sep_token_id)
            & (batch["input_ids"] != self.qa_tokenizer.pad_token_id),
            1,
            0,
        )
        out = self.qa_model(
            token_ids=batch["input_ids"].to(self.device),
            extend_embeds=kb_embeds,
            extend_tokens=extend_tokens,
            attention_mask=batch["attention_mask"].to(self.device),
            token_type_ids=batch["token_type_ids"].to(self.device),
            start_positions=batch["start_positions"].to(self.device),
            end_positions=batch["end_positions"].to(self.device),
        )
        return out[0]

    # noinspection PyTypeChecker
    def validation_step(self, batch: BatchEncoding, _batch_idx):
        batch.convert_to_tensors("pt")
        kb_embeds = self.kb_encoder.compute_sentence_embeds(
            sentence_tokens=batch["input_ids"].to(self.device),
            context_length=self.config.context_length,
            with_gradient_num=0,
        )
        extend_tokens = t.where(
            (batch["input_ids"] != self.qa_tokenizer.cls_token_id)
            & (batch["input_ids"] != self.qa_tokenizer.sep_token_id)
            & (batch["input_ids"] != self.qa_tokenizer.pad_token_id),
            1,
            0,
        )
        out = self.qa_model(
            token_ids=batch["input_ids"].to(self.device),
            extend_embeds=kb_embeds,
            extend_tokens=extend_tokens,
            attention_mask=batch["attention_mask"].to(self.device),
            token_type_ids=batch["token_type_ids"].to(self.device),
            start_positions=batch["start_positions"].to(self.device),
            end_positions=batch["end_positions"].to(self.device),
        )
        metrics = self.validate_qa_dataset.validate(batch, out[1], out[2])
        for key, value in metrics.items():
            self.log(key, value, sync_dist=self.is_distributed)

    def configure_optimizers(self):
        optim_cls = getattr(t.optim, self.config.optimizer_class)
        return optim_cls(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.l2_regularization,
        )
