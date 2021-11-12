import os
import logging
import torch as t
from tqdm import tqdm
from copy import deepcopy
from torch.distributed import is_initialized, get_rank
from transformers import AutoTokenizer
from encoder.model.attr import modify_t5_model_with_attr
from encoder.dataset.base import collate_function_dict_to_batch_encoding
from encoder.dataset.concept_net import ConceptNetMatcher
from encoder.utils.inspect import save_inspect_data


def replace_string_span(string, start, end, new_char):
    return string[:start] + new_char * (end - start) + string[end:]


def chunk(num, chunk_size):
    for i in range(0, num, chunk_size):
        yield [j for j in range(i, min(i + chunk_size, num))]


def fix_special_tokens(sentence: str):
    sentence = sentence.replace("< / s >", "</s>")
    sentence = sentence.replace("< pad >", "<pad>")
    sentence = sentence.replace(" <pad> ", "<pad>")
    return sentence


def print_sample_with_score(tokenizer, sample, attr_score):
    decoded_sentences = [
        [tokenizer.decode(token) for token in sentence]
        for sentence in sample["sentence"].cpu().tolist()
    ]
    sentence_scores = attr_score.cpu().tolist()
    print("")
    for st, st_sc in zip(decoded_sentences, sentence_scores):
        result = []
        for token, score in zip(st, st_sc):
            result.append(f"{token}:{score:.3f}")
        print("")
        print(f"[{' '.join(result)}]")
        print("")


# class IterEnv:
#     def __init__(
#         self,
#         trainer,
#         kb_trainer=None,
#         attr_file_path: str = None,
#         attr_steps: int = 1,
#         attr_threshold: float = 0.35,
#         attr_warmup_epochs: int = 0,
#         attr_epoch_interval: int = 1,
#         attr_process_batch_size: int = 8,
#         matcher_max_times: int = 300,
#         matcher_max_depth: int = 2,
#         matcher_max_edges: int = 12,
#         matcher_discard_edges_if_similarity_below: float = 0.4,
#         matcher_seed: int = -1,
#     ):
#         """
#         Currently only supports T5 model.
#         """
#         self.trainer = trainer
#         self.attr_file_path = attr_file_path
#
#         if attr_file_path is not None:
#             self.attr_map_from_file = t.load(self.attr_file_path)
#             logging.info(f"Loading attr_map from [{self.attr_file_path}]")
#
#         self.attr_steps = attr_steps
#         self.attr_threshold = attr_threshold
#         # Only replace forward methods, no modification with parameters, etc.
#         # Therefore modifications here are not saved in checkpoint
#         self.attr_weight = modify_t5_model_with_attr(trainer.model)
#         self.attr_warmup_epochs = attr_warmup_epochs
#         self.attr_epoch_interval = attr_epoch_interval
#         self.attr_process_batch_size = attr_process_batch_size
#         self.add_hook_to_dataset(trainer.dataset)
#         self.add_hook_to_training_epoch(trainer)
#
#         self.kb_trainer = kb_trainer
#         self.matcher_max_times = matcher_max_times
#         self.matcher_max_depth = matcher_max_depth
#         self.matcher_max_edges = matcher_max_edges
#         self.matcher_discard_edges_if_similarity_below = (
#             matcher_discard_edges_if_similarity_below
#         )
#         self.matcher_seed = matcher_seed
#         # Use the same tokenizer to facilitate adding masks
#         self.matcher = ConceptNetMatcher(
#             tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")
#         )
#         self.is_dataset_hook_enabled = True
#         logging.info("Iterative training environment initialized")
#
#     @classmethod
#     def patch_checkpoint_hook(cls, trainer_class):
#         logging.warning(
#             f"Patching save/load checkpoint hooks of class {trainer_class}, this effect is GLOBAL."
#         )
#         org_on_load_checkpoint = trainer_class.on_load_checkpoint
#         org_on_save_checkpoint = trainer_class.on_save_checkpoint
#
#         def on_load_checkpoint(self, checkpoint):
#             self._iter_env_attr_map = checkpoint["_iter_env_attr_map"]
#             logging.info("Loaded attr map from checkpoint")
#             logging.info(f"Train attr number {len(self._iter_env_attr_map['train'])}")
#             logging.info(
#                 f"Validate attr number {len(self._iter_env_attr_map['validate'])}"
#             )
#             logging.info(f"Test attr number {len(self._iter_env_attr_map['test'])}")
#             org_on_load_checkpoint(self, checkpoint)
#
#         def on_save_checkpoint(self, checkpoint):
#             checkpoint["_iter_env_attr_map"] = self._iter_env_attr_map
#             logging.info("Added attr map to checkpoint")
#             org_on_save_checkpoint(self, checkpoint)
#
#         trainer_class.on_load_checkpoint = on_load_checkpoint
#         trainer_class.on_save_checkpoint = on_save_checkpoint
#
#     def add_hook_to_dataset(self, trainer_dataset):
#         dataset_class = type(trainer_dataset)
#         logging.warning(
#             f"Adding post-processing hooks to class {dataset_class}, this effect is GLOBAL."
#         )
#         for dataset_name in ("train", "validate", "test"):
#             # replace the fget function in sub dataset property if it exists
#             if hasattr(dataset_class, f"{dataset_name}_dataset"):
#                 self._wrap_dataset_getter(dataset_class, dataset_name)
#
#     def add_hook_to_training_epoch(self, trainer):
#         logging.warning(
#             f"Adding training end hook to trainer instance, this effect is local."
#         )
#         trainer.training_epoch_end = self._wrap_training_epoch_end_hook(
#             trainer.training_epoch_end
#         )
#
#     def compute_attr_score(self, batch):
#         steps = [0] + [i / self.attr_steps for i in range(1, self.attr_steps + 1)]
#         device = getattr(self.trainer, "real_device", None) or self.trainer.device
#         decoder_input_ids = t.full(
#             [
#                 batch["sentence"].shape[0],
#                 getattr(self.trainer.config, "generate_length", 16),
#             ],
#             self.trainer.tokenizer.pad_token_id,
#             dtype=t.long,
#             device=device,
#         )
#         with t.no_grad():
#             attentions = (
#                 self.trainer.model(
#                     input_ids=batch["sentence"].to(device),
#                     attention_mask=batch["mask"].to(device),
#                     decoder_input_ids=decoder_input_ids,
#                     output_attentions=True,
#                     return_dict=True,
#                 ).encoder_attentions[0]
#                 / self.attr_steps
#             )
#         attr_per_head = 0
#         for step in steps:
#             self.attr_weight.set(step)
#             result = self.trainer.model(
#                 input_ids=batch["sentence"].to(device),
#                 attention_mask=batch["mask"].to(device),
#                 decoder_input_ids=decoder_input_ids,
#                 output_attentions=True,
#                 return_dict=True,
#             )
#             # Computes grad of
#             # the probability of the vocab position selected by the model
#             # to the input
#             label = t.argmax(result.logits, dim=2, keepdim=True)
#             # Attention from layer 0
#             attr_per_head_grad = t.autograd.grad(
#                 t.gather(t.softmax(result.logits, dim=2), dim=2, index=label).sum(),
#                 result.encoder_attentions[0],
#             )[0]
#             attr_per_head = attr_per_head + attr_per_head_grad
#
#         # restore to normal
#         self.attr_weight.set(1)
#
#         # sum by head, then normalize to range of [-1, 1] by dividing the largest attribute score in dim (1, 2)
#         # shape [batch_size, input_seq_length, input_seq_length]
#         attr = t.sum(attr_per_head * attentions, dim=1)
#         normalized_attr = (
#             attr
#             / t.max(t.max(t.abs(attr), dim=1, keepdim=True)[0], dim=2, keepdim=True)[
#                 0
#             ].detach()
#         )
#
#         # remove self attention (diagonal values, attention to word itself)
#         # and values below 0 (negative contribution to selected vocab position)
#         positive_attr = (
#             normalized_attr
#             * (1 - t.eye(attr.shape[-1], device=attr.device).unsqueeze(0))
#             * (normalized_attr > 0).detach()
#         )
#         per_token_attr = positive_attr.sum(dim=2)
#         # shape [batch_size, input_seq_length]
#         return per_token_attr
#
#     def _get_dataset_hook(self, dataset_name):
#         def hook(result):
#             if (
#                 "sentence" not in result
#                 or "mask" not in result
#                 or "id" not in result
#                 or len(result["sentence"].shape) != 2
#                 or len(result["mask"].shape) != 2
#                 or result["sentence"].shape[0] != 1
#                 or result["mask"].shape[0] != 1
#             ):
#                 raise ValueError("Invalid input result")
#
#             attr_map = self._create_or_use_trainer_attr_map()
#
#             attr_score = attr_map[dataset_name].get(result["id"], None)
#
#             max_seq_length = result["sentence"].shape[1]
#
#             # Prevent tainting original samples
#             result = deepcopy(result)
#
#             if dataset_name == "train":
#                 match = self.matcher.match_by_node_embedding(
#                     result["text_choices"],
#                     target_sentence=result["text_question"],
#                     max_times=self.matcher_max_times,
#                     max_depth=self.matcher_max_depth,
#                     max_edges=self.matcher_max_edges,
#                     seed=self.matcher_seed,
#                     discard_edges_if_similarity_below=self.matcher_discard_edges_if_similarity_below,
#                 )
#
#                 new_choices = self.matcher.insert_match(result["text_choices"], match)
#                 encoded_sentence = self.trainer.tokenizer(
#                     result["text_question"].lower(),
#                     new_choices,
#                     padding="max_length",
#                     max_length=max_seq_length,
#                     truncation=True,
#                     return_tensors="pt",
#                 )
#                 result["sentence"] = encoded_sentence.input_ids
#                 result["mask"] = encoded_sentence.attention_mask
#                 return result
#             else:
#                 if not self.is_dataset_hook_enabled or attr_score is None:
#                     return result
#                 else:
#                     # Multiply with mask to skip special characters
#                     attr_mask = (
#                         result["mask"] * attr_score.to(result["mask"].device)
#                     ) >= self.attr_threshold
#
#                 # 1) First we decode to obtain semi-original sentence S'
#                 # (with special tokens kept so nothing is lost when we re-encode it)
#                 # (Space might be incorrect, like [spam] becomes [ spam ]
#                 #  but that won't affect the result)
#                 # 2) Then re-encode the sentence to get batch encoding, which can then
#                 # be used to match tokens with character spans
#                 sentence = self.trainer.tokenizer.decode(
#                     result["sentence"][0], skip_special_tokens=False
#                 )
#                 sentence_encoding = self.trainer.tokenizer(
#                     sentence,
#                     padding="max_length",
#                     max_length=max_seq_length,
#                     truncation=True,
#                     return_tensors="pt",
#                 )
#
#                 if self.kb_trainer is None:
#                     # For matcher:
#                     # Use the character span to create a binary mask the same length as
#                     # the recovered string S', use this mask to tell the matcher which
#                     # part of the sentence can be searched.
#                     sentence_mask = "-" * len(sentence)
#                     insert_position_count = 0
#                     last_attr_high_position = None
#                     for i in range(max_seq_length):
#                         if attr_mask[0, i] == 1:
#                             try:
#                                 span = sentence_encoding.token_to_chars(0, i)
#                             except TypeError:
#                                 continue
#                             sentence_mask = replace_string_span(
#                                 sentence_mask, span.start, span.end, "+"
#                             )
#
#                             if last_attr_high_position is None:
#                                 last_attr_high_position = span.end
#                             else:
#                                 last_attr_high_position = max(
#                                     last_attr_high_position, span.end
#                                 )
#                         else:
#                             if last_attr_high_position is not None:
#                                 insert_position_count += 1
#                                 last_attr_high_position = None
#
#                     first_part = sentence.find("</s>")
#
#                     keep_first_part_mask = replace_string_span(
#                         sentence_mask, first_part, len(sentence), "-"
#                     )
#                     keep_second_part_mask = replace_string_span(
#                         "-" * len(sentence),
#                         first_part + len("</s>"),
#                         sentence.find("</s>", first_part + len("</s>")),
#                         "+",
#                     )
#                     # match1 = self.matcher.match_by_node_embedding(
#                     #     sentence,
#                     #     target_sentence=sentence,
#                     #     source_mask=keep_first_part_mask,
#                     #     target_mask=keep_second_part_mask,
#                     #     max_times=self.matcher_max_times,
#                     #     max_depth=self.matcher_max_depth,
#                     #     max_edges=min(
#                     #         self.matcher_max_edges,
#                     #         insert_position_count * self.matcher_max_depth,
#                     #     ),
#                     #     discard_edges_if_similarity_below=self.matcher_discard_edges_if_similarity_below,
#                     #     seed=self.matcher_seed,
#                     # )
#
#                     match2 = self.matcher.match_by_node_embedding(
#                         sentence,
#                         target_sentence=sentence,
#                         source_mask=keep_second_part_mask,
#                         target_mask=keep_first_part_mask,
#                         max_times=self.matcher_max_times,
#                         max_depth=self.matcher_max_depth,
#                         max_edges=min(
#                             self.matcher_max_edges,
#                             insert_position_count * self.matcher_max_depth,
#                         ),
#                         discard_edges_if_similarity_below=self.matcher_discard_edges_if_similarity_below,
#                         seed=self.matcher_seed,
#                     )
#
#                     # new_sentence = fix_special_tokens(
#                     #     self.matcher.insert_match(
#                     #         sentence, self.matcher.unify_match([match1, match2])
#                     #     )
#                     # )
#
#                     new_sentence = fix_special_tokens(
#                         self.matcher.insert_match(sentence, match2)
#                     )
#
#                     encoded_sentence = self.trainer.tokenizer(
#                         new_sentence,
#                         padding="max_length",
#                         max_length=max_seq_length,
#                         truncation=True,
#                         return_tensors="pt",
#                     )
#                     result["sentence"] = encoded_sentence.input_ids
#                     result["mask"] = encoded_sentence.attention_mask
#                 else:
#                     # For KB model
#                     # 1) For each attribution high score occurrence, insert an
#                     # <extra_id_{}> token behind it
#                     # 2) Use KB model to predict the result
#                     # 3) Combine result with input sentence id tensor
#                     insert_positions = []
#                     last_attr_high_position = None
#                     for i in range(max_seq_length):
#                         if attr_mask[0, i] == 1:
#                             try:
#                                 span = sentence_encoding.token_to_chars(0, i)
#                             except TypeError:
#                                 continue
#                             if last_attr_high_position is None:
#                                 last_attr_high_position = span.end
#                             else:
#                                 last_attr_high_position = max(
#                                     last_attr_high_position, span.end
#                                 )
#                         else:
#                             if last_attr_high_position is not None:
#                                 insert_positions.append(last_attr_high_position)
#                                 last_attr_high_position = None
#
#                     # Insert query markers
#                     offset = 0
#                     kb_sentence = deepcopy(sentence)
#                     for i, insert_pos in enumerate(insert_positions):
#                         marker = f"<extra_id_{i}>"
#                         pos = insert_pos + offset
#                         kb_sentence = kb_sentence[:pos] + marker + kb_sentence[pos:]
#                         offset += len(marker)
#
#                     encoded_kb_sentence = self.kb_trainer.tokenizer(
#                         kb_sentence,
#                         padding="max_length",
#                         max_length=max_seq_length,
#                         truncation=True,
#                         return_tensors="pt",
#                     )
#                     out = self.kb_trainer.model.generate(
#                         encoded_kb_sentence.input_ids.to(self.kb_trainer.device),
#                         max_length=max_seq_length,
#                         attention_mask=encoded_kb_sentence.attention_mask.to(
#                             self.kb_trainer.device
#                         ),
#                         early_stopping=True,
#                     )
#                     answer = self.kb_trainer.tokenizer.decode(
#                         out, skip_special_tokens=True
#                     )
#
#                     kb_answer_search_offset = 0
#                     sentence_insert_offset = 0
#                     new_sentence = deepcopy(sentence)
#                     for i, insert_pos in enumerate(insert_positions):
#                         marker = f"<extra_id_{i}>"
#                         start = answer.find(marker, kb_answer_search_offset)
#                         if start != -1:
#                             # extract knowledge from answer
#                             end = answer.find("<", start + len(marker))
#                             kb_answer_search_offset = end
#                             knowledge = answer[start:end]
#
#                             # insert knowledge into sentence
#                             pos = insert_pos + new_sentence
#                             new_sentence = (
#                                 new_sentence[:pos] + knowledge + new_sentence[pos:]
#                             )
#                             sentence_insert_offset += len(knowledge)
#                         else:
#                             break
#
#                     encoded_sentence = self.trainer.tokenizer(
#                         new_sentence,
#                         padding="max_length",
#                         max_length=max_seq_length,
#                         truncation=True,
#                         return_tensors="pt",
#                     )
#                     result["sentence"] = encoded_sentence.input_ids
#                     result["mask"] = encoded_sentence.attention_mask
#                 return result
#
#         return hook
#
#     def _wrap_dataset_getter(self, dataset_class, dataset_name):
#         org_sub_dataset_prop = getattr(dataset_class, f"{dataset_name}_dataset")
#
#         def new_dataset_getter(dataset_self):
#             dataset = org_sub_dataset_prop.fget(dataset_self)
#             dataset.set_override_hook(self._get_dataset_hook(dataset_name))
#             return dataset
#
#         setattr(
#             dataset_class, f"{dataset_name}_dataset", property(new_dataset_getter),
#         )
#
#     def _wrap_training_epoch_end_hook(self, func):
#         def wrapped(outputs):
#             result = func(outputs)
#
#             attr_map = self._create_or_use_trainer_attr_map()
#             if self.attr_file_path is not None:
#                 logging.info(
#                     f"Skipping attr_map update for current epoch {self.trainer.current_epoch} "
#                     f"because attr_file_path is specified"
#                 )
#                 return result
#             if self.trainer.current_epoch < self.attr_warmup_epochs - 1:
#                 logging.info(
#                     f"Skipping attr_map update for current epoch {self.trainer.current_epoch} "
#                     f"because warm up requires {self.attr_warmup_epochs} epochs"
#                 )
#                 return result
#             if (
#                 self.trainer.current_epoch - self.attr_warmup_epochs + 1
#             ) % self.attr_epoch_interval == 0:
#                 logging.info("Computing attribute scores")
#                 # disable matching for attr computation
#                 self.is_dataset_hook_enabled = False
#                 logging.info("Dataset hook disabled")
#                 trainer_dataset = self.trainer.dataset
#                 # for dataset_name in ("train", "validate", "test"):
#                 for dataset_name in ("validate", "test"):
#                     if hasattr(trainer_dataset, f"{dataset_name}_dataset"):
#                         logging.info(
#                             f"Updating attribute score for the {dataset_name} dataset"
#                         )
#                         attr_sub_map = attr_map[dataset_name] = {}
#                         sub_dataset = getattr(
#                             trainer_dataset, f"{dataset_name}_dataset"
#                         )
#                         with tqdm(
#                             total=len(sub_dataset),
#                             desc="Processed samples",
#                             unit=" samples",
#                         ) as progress_bar:
#                             for ck in chunk(
#                                 len(sub_dataset), self.attr_process_batch_size
#                             ):
#                                 batch = collate_function_dict_to_batch_encoding(
#                                     [sub_dataset[i] for i in ck]
#                                 )
#                                 attr_score = self.compute_attr_score(batch)
#                                 for id, a_score in zip(batch["id"], attr_score):
#                                     attr_sub_map[str(id)] = a_score.cpu()
#
#                                 if ck[-1] < 32 and (
#                                     not is_initialized() or get_rank() == 0
#                                 ):
#                                     print_sample_with_score(
#                                         self.trainer.tokenizer, batch, attr_score
#                                     )
#                                 progress_bar.update(len(ck))
#                 # Restore matching
#                 self.is_dataset_hook_enabled = True
#                 logging.info("Dataset hook enabled")
#                 save_inspect_data(attr_map, "attr_map")
#             return result
#
#         return wrapped
#
#     def _create_or_use_trainer_attr_map(self):
#         if self.attr_file_path is not None:
#             setattr(self.trainer, "_iter_env_attr_map", self.attr_map_from_file)
#         else:
#             if not hasattr(self.trainer, "_iter_env_attr_map"):
#                 setattr(
#                     self.trainer,
#                     "_iter_env_attr_map",
#                     {"train": {}, "validate": {}, "test": {}},
#                 )
#         return getattr(self.trainer, "_iter_env_attr_map")


class IterEnv:
    def __init__(
        self,
        trainer,
        kb_trainer=None,
        attr_file_path: str = None,
        attr_steps: int = 1,
        attr_threshold: float = 0.35,
        attr_warmup_epochs: int = 0,
        attr_epoch_interval: int = 1,
        attr_process_batch_size: int = 8,
        attr_previous_ratio: float = 0.8,
        matcher_max_times: int = 300,
        matcher_max_depth: int = 2,
        matcher_max_edges: int = 12,
        matcher_discard_edges_if_similarity_below: float = 0.4,
        matcher_seed: int = -1,
    ):
        """
        Currently only supports T5 model.
        """
        self.trainer = trainer
        self.attr_file_path = attr_file_path

        if attr_file_path is not None:
            self.attr_map_from_file = t.load(self.attr_file_path)
            logging.info(f"Loading attr_map from [{self.attr_file_path}]")

        self.attr_steps = attr_steps
        self.attr_threshold = attr_threshold
        # Only replace forward methods, no modification with parameters, etc.
        # Therefore modifications here are not saved in checkpoint
        self.attr_weight = modify_t5_model_with_attr(trainer.model)
        self.attr_warmup_epochs = attr_warmup_epochs
        self.attr_epoch_interval = attr_epoch_interval
        self.attr_process_batch_size = attr_process_batch_size
        self.attr_previous_ratio = attr_previous_ratio
        self.add_hook_to_training_step(trainer)
        self.add_hook_to_training_epoch(trainer)

        trainer.dataset.use_matcher = False
        # IMPORTANT
        trainer.dataset.insert_answers_at_end = True

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

    @classmethod
    def patch_checkpoint_hook(cls, trainer_class):
        logging.warning(
            f"Patching save/load checkpoint hooks of class {trainer_class}, this effect is GLOBAL."
        )
        org_on_load_checkpoint = trainer_class.on_load_checkpoint
        org_on_save_checkpoint = trainer_class.on_save_checkpoint

        def on_load_checkpoint(self, checkpoint):
            self._iter_env_attr_map = checkpoint["_iter_env_attr_map"]
            logging.info("Loaded attr map from checkpoint")
            logging.info(f"Train attr number {len(self._iter_env_attr_map['train'])}")
            logging.info(
                f"Validate attr number {len(self._iter_env_attr_map['validate'])}"
            )
            logging.info(f"Test attr number {len(self._iter_env_attr_map['test'])}")
            org_on_load_checkpoint(self, checkpoint)

        def on_save_checkpoint(self, checkpoint):
            checkpoint["_iter_env_attr_map"] = self._iter_env_attr_map
            logging.info("Added attr map to checkpoint")
            org_on_save_checkpoint(self, checkpoint)

        trainer_class.on_load_checkpoint = on_load_checkpoint
        trainer_class.on_save_checkpoint = on_save_checkpoint

    def add_hook_to_training_step(self, trainer):
        logging.warning(
            f"Adding training step hook to trainer instance, this effect is local."
        )
        trainer.training_step = self._wrap_training_step_hook(trainer.training_step)

    def add_hook_to_training_epoch(self, trainer):
        logging.warning(
            f"Adding training end hook to trainer instance, this effect is local."
        )
        trainer.training_epoch_end = self._wrap_training_epoch_end_hook(
            trainer.training_epoch_end
        )

    def compute_attr_score(
        self, batch, use_original=False, return_raw=False, create_graph=False
    ):
        steps = [0] + [i / self.attr_steps for i in range(1, self.attr_steps + 1)]
        device = getattr(self.trainer, "real_device", None) or self.trainer.device
        if use_original:
            sentence_key = "org_sentence"
            mask_key = "org_mask"
        else:
            sentence_key = "sentence"
            mask_key = "mask"
        decoder_input_ids = t.full(
            [
                batch[sentence_key].shape[0],
                getattr(self.trainer.config, "generate_length", 16),
            ],
            self.trainer.tokenizer.pad_token_id,
            dtype=t.long,
            device=device,
        )
        with t.no_grad():
            attentions = (
                self.trainer.model(
                    input_ids=batch[sentence_key].to(device),
                    attention_mask=batch[mask_key].to(device),
                    decoder_input_ids=decoder_input_ids,
                    output_attentions=True,
                    return_dict=True,
                ).encoder_attentions[0]
                / self.attr_steps
            )
        attr_per_head = 0
        for step in steps:
            self.attr_weight.set(step)
            result = self.trainer.model(
                input_ids=batch[sentence_key].to(device),
                attention_mask=batch[mask_key].to(device),
                decoder_input_ids=decoder_input_ids,
                output_attentions=True,
                return_dict=True,
            )
            # Computes grad of
            # the probability of the vocab position selected by the model
            # to the input
            label = t.argmax(result.logits, dim=2, keepdim=True)
            # Attention from layer 0
            attr_per_head_grad = t.autograd.grad(
                t.gather(t.softmax(result.logits, dim=2), dim=2, index=label).sum(),
                result.encoder_attentions[0],
                create_graph=create_graph,
            )[0]
            attr_per_head = attr_per_head + attr_per_head_grad

        # restore to normal
        self.attr_weight.set(1)

        # sum by head, then normalize to range of [-1, 1] by dividing the largest attribute score in dim (1, 2)
        # shape [batch_size, input_seq_length, input_seq_length]
        attr = t.sum(attr_per_head * attentions, dim=1)
        if return_raw:
            return attr

        normalized_attr = (
            attr
            / t.max(t.max(t.abs(attr), dim=1, keepdim=True)[0], dim=2, keepdim=True)[
                0
            ].detach()
        )

        # remove self attention (diagonal values, attention to word itself)
        # and values below 0 (negative contribution to selected vocab position)
        positive_attr = (
            normalized_attr
            * (1 - t.eye(attr.shape[-1], device=attr.device).unsqueeze(0))
            * (normalized_attr > 0).detach()
        )
        per_token_attr = positive_attr.sum(dim=2)
        # shape [batch_size, input_seq_length]
        return per_token_attr

    def _wrap_training_step_hook(self, func):
        def wrapped(batch, batch_idx):
            loss = func(batch, batch_idx)
            if self.trainer.current_epoch >= self.attr_warmup_epochs:
                attr_map = self._create_or_use_trainer_attr_map()
                attr = self.compute_attr_score(
                    batch, use_original=True, return_raw=True, create_graph=True
                )
                attr_loss = attr - t.cat(
                    [attr_map["train"][id].unsqueeze(0) for id in batch["id"]], dim=0
                ).to(attr.device)
                attr_loss = t.abs(attr_loss).sum()
                attr_loss.backward()
            return loss

        return wrapped

    def _wrap_training_epoch_end_hook(self, func):
        def wrapped(outputs):
            result = func(outputs)

            attr_map = self._create_or_use_trainer_attr_map()
            trainer_dataset = self.trainer.dataset
            if (
                self.trainer.current_epoch >= self.attr_warmup_epochs - 1
                and (self.trainer.current_epoch - self.attr_warmup_epochs + 1)
                % self.attr_epoch_interval
                == 0
            ):
                self.trainer.dataset.use_matcher = False
                logging.info(f"Updating attribute score for the validate dataset")
                attr_sub_map = attr_map["validate"] = {}
                sub_dataset = getattr(trainer_dataset, "validate_dataset")
                with tqdm(
                    total=len(sub_dataset), desc="Processed samples", unit=" samples",
                ) as progress_bar:
                    for ck in chunk(len(sub_dataset), self.attr_process_batch_size):
                        batch = collate_function_dict_to_batch_encoding(
                            [sub_dataset[i] for i in ck]
                        )
                        attr_score = self.compute_attr_score(batch)
                        for id, a_score in zip(batch["id"], attr_score):
                            attr_sub_map[str(id)] = a_score.cpu()

                        if ck[-1] < 32 and (not is_initialized() or get_rank() == 0):
                            print_sample_with_score(
                                self.trainer.tokenizer, batch, attr_score
                            )
                        progress_bar.update(len(ck))

                logging.info(f"Updating attribute score for the train dataset")
                attr_sub_map = attr_map["train"]
                sub_dataset = getattr(trainer_dataset, "train_dataset")
                is_first = len(attr_sub_map) == 0
                with tqdm(
                    total=len(sub_dataset), desc="Processed samples", unit=" samples",
                ) as progress_bar:
                    for ck in chunk(len(sub_dataset), self.attr_process_batch_size):
                        batch = collate_function_dict_to_batch_encoding(
                            [sub_dataset[i] for i in ck]
                        )
                        attr_score = self.compute_attr_score(batch, return_raw=True)
                        for id, a_score, mask in zip(
                            batch["id"], attr_score, batch["mask"]
                        ):
                            if is_first:
                                attr_sub_map[str(id)] = a_score.cpu()
                            else:
                                attr_sub_map[str(id)] = (
                                    a_score.cpu() * (1 - self.attr_previous_ratio)
                                    + attr_sub_map[str(id)] * self.attr_previous_ratio
                                )
                        progress_bar.update(len(ck))

                # after warming up, every epoch will have the sample with matched answers
                self.trainer.dataset.use_matcher = True

            save_inspect_data(
                {"validate": attr_map["validate"]},
                f"attr_map_epoch={self.trainer.current_epoch}",
            )
            return result

        return wrapped

    def _create_or_use_trainer_attr_map(self):
        if not hasattr(self.trainer, "_iter_env_attr_map"):
            setattr(
                self.trainer,
                "_iter_env_attr_map",
                {"train": {}, "validate": {}, "test": {}},
            )
        return getattr(self.trainer, "_iter_env_attr_map")
