import logging
import torch as t
from tqdm import tqdm
from copy import deepcopy
from transformers import AutoTokenizer
from encoder.model.attr import modify_t5_model_with_attr
from encoder.dataset.concept_net import ConceptNetMatcher


def replace_string_span(string, start, end, new_char):
    return string[:start] + new_char * (end - start) + string[end:]


class IterEnv:
    def __init__(
        self,
        trainer,
        kb_trainer=None,
        attr_steps: int = 1,
        attr_threshold: float = 0.3,
        attr_epoch_interval: int = 1,
        matcher_max_times: int = 300,
        matcher_max_depth: int = 2,
        matcher_max_edges: int = 12,
        matcher_discard_edges_if_similarity_below: float = 0.5,
        matcher_seed: int = -1,
    ):
        """
        Currently only supports T5 model.
        """
        self.trainer = trainer
        self.attr_steps = attr_steps
        self.attr_threshold = attr_threshold
        self.attr_weight = modify_t5_model_with_attr(trainer.model)
        self.attr_map = {}
        self.attr_epoch_interval = attr_epoch_interval
        self.add_hook_to_dataset(trainer.dataset)
        self.add_hook_to_training_epoch(trainer)

        self.kb_trainer = kb_trainer
        self.matcher_max_times = matcher_max_times
        self.matcher_max_depth = matcher_max_depth
        self.matcher_max_edges = matcher_max_edges
        self.matcher_discard_edges_if_similarity_below = (
            matcher_discard_edges_if_similarity_below
        )
        self.matcher_seed = matcher_seed
        # Use the same tokenizer to facilitate adding masks
        self.matcher = ConceptNetMatcher(
            tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")
        )
        self.is_dataset_hook_enabled = True
        logging.info("Iterative training environment initialized")

    def set_dataset_hook(self, status):
        self.is_dataset_hook_enabled = status

    def compute_attr_score(self, sample):
        steps = [i / self.attr_steps for i in range(1, self.attr_steps + 1)]
        with t.no_grad():
            attentions = (
                self.trainer.model(
                    input_ids=sample["sentence"],
                    attention_mask=sample["mask"],
                    labels=sample["answer"],
                    output_attentions=True,
                    return_dict=True,
                ).encoder_attentions[0]
                / self.attr_steps
            )
        attr_per_head = 0
        for step in steps:
            self.attr_weight.set(step)
            result = self.trainer.model(
                input_ids=sample["sentence"],
                attention_mask=sample["mask"],
                labels=sample["answer"],
                output_attentions=True,
                return_dict=True,
            )
            # Or use softmax(logits( instead of loss?
            attr_per_head_grad = t.autograd.grad(
                result.loss, result.encoder_attentions[0]
            )
            attr_per_head += attr_per_head_grad
        # mean by head and mean by row, then softmax
        attr = t.softmax(t.mean(attr_per_head * attentions, dim=(1, 3)), dim=1)
        return attr

    def add_hook_to_dataset(self, trainer_dataset):
        if hasattr(trainer_dataset, "train_dataset"):
            trainer_dataset.train_dataset.set_override_hook(
                self._get_dataset_hook("train")
            )
        if hasattr(trainer_dataset, "validate_dataset"):
            trainer_dataset.validate_dataset.set_override_hook(
                self._get_dataset_hook("validate")
            )
        if hasattr(trainer_dataset, "test_dataset"):
            trainer_dataset.test_dataset.set_override_hook(
                self._get_dataset_hook("test")
            )

    def add_hook_to_training_epoch(self, trainer):
        trainer.training_epoch_end = self._wrap_training_epoch_end_hook(
            trainer.training_epoch_end
        )

    def _get_dataset_hook(self, dataset):
        def hook(result):
            if (
                "sentence" not in result
                or "mask" not in result
                or "answer" not in result
                or "id" not in result
                or len(result["sentence"].shape) != 2
                or len(result["mask"].shape) != 2
                or len(result["answer"].shape) != 2
                or result["sentence"].shape[0] != 1
                or result["mask"].shape[0] != 1
                or result["answer"].shape[0] != 1
            ):
                raise ValueError("Invalid input result")

            if dataset not in self.attr_map or not self.is_dataset_hook_enabled:
                return result
            attr_score = self.attr_map[dataset].get(result["id"], None)
            if attr_score is None:
                attr_mask = result["mask"]
            else:
                # Multiply with mask to skip special characters
                attr_mask = (result["mask"] * attr_score) > self.attr_threshold

            max_seq_length = result["sentence"].shape[1]

            # Prevent tainting original samples
            result = deepcopy(result)

            # 1) First we decode to obtain semi-original sentence S'
            # (with special tokens kept so nothing is lost when we re-encode it)
            # (Space might be incorrect, like [spam] becomes [ spam ]
            #  but that won't affect the result)
            # 2) Then re-encode the sentence to get batch encoding, which can then
            # be used to match tokens with character spans
            sentence = self.trainer.tokenizer.decode(
                result["sentence"], skip_special_tokens=False
            )
            sentence_encoding = self.trainer.tokenizer(
                sentence,
                padding="max_length",
                max_length=max_seq_length,
                truncation=True,
                return_tensors="pt",
            )

            if self.kb_trainer.model is None:
                # For matcher:
                # Use the character span to create a binary mask the same length as
                # the recovered string S', use this mask to tell the matcher which
                # part of the sentence can be searched.
                sentence_mask = "-" * len(sentence)
                insert_position_count = 0
                last_attr_high_position = None
                for i in range(max_seq_length):
                    if attr_mask[0, i] == 1:
                        try:
                            span = sentence_encoding.token_to_chars(0, i)
                        except TypeError:
                            continue
                        sentence_mask = replace_string_span(
                            sentence_mask, span.start, span.end, "+"
                        )

                        if last_attr_high_position is None:
                            last_attr_high_position = span.end
                        else:
                            last_attr_high_position = max(
                                last_attr_high_position, span.end
                            )
                    else:
                        if last_attr_high_position is not None:
                            insert_position_count += 1
                            last_attr_high_position = None

                match = self.matcher.match_by_node_embedding(
                    sentence,
                    target_sentence=sentence,
                    source_mask=sentence_mask,
                    target_mask=sentence_mask,
                    max_times=self.matcher_max_times,
                    max_depth=self.matcher_max_depth,
                    max_edges=min(self.matcher_max_edges, insert_position_count * 2),
                    discard_edges_if_similarity_below=self.matcher_discard_edges_if_similarity_below,
                    seed=self.matcher_seed,
                )
                new_sentence = self.matcher.insert_match(sentence, match)

                encoded_sentence = self.trainer.tokenizer(
                    new_sentence,
                    padding="max_length",
                    max_length=max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                )
                result["sentence"] = encoded_sentence.input_ids
                result["mask"] = encoded_sentence.attention_mask
            else:
                # For KB model
                # 1) For each attribution high score occurrence, insert an
                # <extra_id_{}> token behind it
                # 2) Use KB model to predict the result
                # 3) Combine result with input sentence id tensor
                insert_positions = []
                last_attr_high_position = None
                for i in range(max_seq_length):
                    if attr_mask[0, i] == 1:
                        try:
                            span = sentence_encoding.token_to_chars(0, i)
                        except TypeError:
                            continue
                        if last_attr_high_position is None:
                            last_attr_high_position = span.end
                        else:
                            last_attr_high_position = max(
                                last_attr_high_position, span.end
                            )
                    else:
                        if last_attr_high_position is not None:
                            insert_positions.append(last_attr_high_position)
                            last_attr_high_position = None

                # Insert query markers
                offset = 0
                kb_sentence = deepcopy(sentence)
                for i, insert_pos in enumerate(insert_positions):
                    marker = f"<extra_id_{i}>"
                    pos = insert_pos + offset
                    kb_sentence = kb_sentence[:pos] + marker + kb_sentence[pos:]
                    offset += len(marker)

                encoded_kb_sentence = self.kb_trainer.tokenizer(
                    kb_sentence,
                    padding="max_length",
                    max_length=max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                )
                out = self.kb_trainer.model.generate(
                    encoded_kb_sentence.input_ids.to(self.kb_trainer.device),
                    max_length=max_seq_length,
                    attention_mask=encoded_kb_sentence.attention_mask.to(
                        self.kb_trainer.device
                    ),
                    early_stopping=True,
                )
                answer = self.kb_trainer.tokenizer.decode(out, skip_special_tokens=True)

                kb_answer_search_offset = 0
                sentence_insert_offset = 0
                new_sentence = deepcopy(sentence)
                for i, insert_pos in enumerate(insert_positions):
                    marker = f"<extra_id_{i}>"
                    start = answer.find(marker, kb_answer_search_offset)
                    if start != -1:
                        # extract knowledge from answer
                        end = answer.find("<", start + len(marker))
                        kb_answer_search_offset = end
                        knowledge = answer[start:end]

                        # insert knowledge into sentence
                        pos = insert_pos + new_sentence
                        new_sentence = (
                            new_sentence[:pos] + knowledge + new_sentence[pos:]
                        )
                        sentence_insert_offset += len(knowledge)
                    else:
                        break

                encoded_sentence = self.trainer.tokenizer(
                    new_sentence,
                    padding="max_length",
                    max_length=max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                )
                result["sentence"] = encoded_sentence.input_ids
                result["mask"] = encoded_sentence.attention_mask

        return hook

    def _wrap_training_epoch_end_hook(self, func):
        def wrapped(outputs):
            result = func(outputs)
            if self.trainer.current_epoch % self.attr_epoch_interval == 0:
                trainer_dataset = self.trainer.dataset
                self.trainer.model.zero_grad()
                if hasattr(trainer_dataset, "train_dataset"):
                    logging.info("Updating attribute score for the train dataset")
                    train_map = self.attr_map["train"] = {}
                    with tqdm(
                        total=len(trainer_dataset.train_dataset),
                        desc="Processed samples",
                        unit=" samples",
                    ) as progress_bar:
                        for i in range(len(trainer_dataset.train_dataset)):
                            sample = trainer_dataset.train_dataset[i]
                            train_map[sample["id"]] = self.compute_attr_score(sample)
                            progress_bar.update(1)
                if hasattr(trainer_dataset, "validate_dataset"):
                    logging.info("Updating attribute score for the validate dataset")
                    validate_map = self.attr_map["validate"] = {}
                    with tqdm(
                        total=len(trainer_dataset.train_dataset),
                        desc="Processed samples",
                        unit=" samples",
                    ) as progress_bar:
                        for i in range(len(trainer_dataset.validate_dataset)):
                            sample = trainer_dataset.validate_dataset[i]
                            validate_map[sample["id"]] = self.compute_attr_score(sample)
                            progress_bar.update(1)
                if hasattr(trainer_dataset, "test_dataset"):
                    logging.info("Updating attribute score for the test dataset")
                    test_map = self.attr_map["test"] = {}
                    with tqdm(
                        total=len(trainer_dataset.train_dataset),
                        desc="Processed samples",
                        unit=" samples",
                    ) as progress_bar:
                        for i in range(len(trainer_dataset.test_dataset)):
                            sample = trainer_dataset.test_dataset[i]
                            test_map[sample["id"]] = self.compute_attr_score(sample)
                            progress_bar.update(1)
            return result

        return wrapped
