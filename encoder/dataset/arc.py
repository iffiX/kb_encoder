import os
import copy
import json
import pickle
import logging
import nltk
import numpy as np
import torch as t
from typing import List, Dict
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, PreTrainedTokenizerBase, BatchEncoding
from encoder.dataset.matcher.arc import ARCMatcher
from encoder.utils.settings import (
    preprocess_cache_dir,
    proxies,
    model_cache_dir,
    huggingface_mirror,
    local_files_only,
)
from encoder.utils.file import open_file_with_create_directories
from .base import StaticIterableDataset
from .download import ARC, OpenBookQA


class ARCDataset:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 300,
        use_matcher: bool = False,
        matcher_mode: str = "embedding",
        matcher_seed: int = -1,
        matcher_config: dict = None,
    ):
        self.tokenizer = tokenizer
        # Word piece is stabler for matching purpose
        self.matcher_tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            local_files_only=local_files_only,
        )
        self.max_seq_length = max_seq_length
        self.use_matcher = use_matcher
        self.matcher_mode = matcher_mode
        self.matcher_seed = matcher_seed
        self.matcher_config = matcher_config

        self.matcher = ARCMatcher(tokenizer=self.matcher_tokenizer)
        self.arc = ARC().require()
        self.openbook_qa = OpenBookQA().require()

        archive_path = os.path.join(preprocess_cache_dir, "arc.data")
        if not os.path.exists(archive_path):
            self.train_data = (
                self.parse_data(self.arc.train_challenge_path, "train_challenge")
                + self.parse_data(self.arc.train_easy_path, "train_easy")
                + self.parse_data(self.arc.validate_easy_path, "validate_easy")
                + self.parse_data(self.arc.test_easy_path, "test_easy")
                + self.parse_openbook_qa_data(self.openbook_qa.train_path)
                + self.parse_openbook_qa_data(self.openbook_qa.validate_path)
                + self.parse_openbook_qa_data(self.openbook_qa.test_path)
            )
            self.validate_data = self.parse_data(
                self.arc.validate_challenge_path, "validate_challenge"
            )
            self.test_data = self.parse_data(
                self.arc.test_challenge_path, "test_challenge"
            )
            self.save(archive_path)
        else:
            with open_file_with_create_directories(archive_path, "rb") as file:
                data = pickle.load(file)
                self.train_data = data["train"]
                self.validate_data = data["validate"]
                self.test_data = data["test"]

        self.set_corpus()

    @property
    def train_dataset(self):
        return StaticIterableDataset(len(self.train_data), self.generator, ("train",),)

    @property
    def validate_dataset(self):
        return StaticIterableDataset(
            len(self.validate_data), self.generator, ("validate",)
        )

    @property
    def test_dataset(self):
        return StaticIterableDataset(len(self.test_data), self.generator, ("test",))

    def generator(self, index: int, split: str):
        if split == "train":
            data = self.train_data[index]
        elif split == "validate":
            data = self.validate_data[index]
        else:
            data = self.test_data[index]

        if self.use_matcher:
            # prevent any modification to data, also prevent checkpoint storing
            # data to gpu by moving
            data = copy.deepcopy(data)
            if self.matcher_mode == "embedding":
                target = " @ ".join(sorted(list(set(data["target"]))))
                target_mask = []
                for c in target:
                    if c == "@":
                        target_mask.append("-")
                    else:
                        target_mask.append("+")
                target_mask = "".join(target_mask)

                match = self.matcher.match_by_node_embedding(
                    data["text_question"],
                    target_sentence=target,
                    target_mask=target_mask,
                    seed=self.matcher_seed,
                    max_times=self.matcher_config["question_match_max_times"],
                    max_depth=self.matcher_config["question_match_max_depth"],
                    edge_top_k=self.matcher_config["question_match_edge_top_k"],
                    source_context_range=self.matcher_config[
                        "question_match_source_context_range"
                    ],
                )
                selection = self.matcher.select_paths(
                    match,
                    max_edges=self.matcher_config["question_select_max_edges"],
                    discard_edges_if_rank_below=self.matcher_config[
                        "question_select_discard_edges_if_rank_below"
                    ],
                )
                new_question = self.matcher.insert_selection_at_end_preserve_case(
                    data["text_question"], selection
                )

                new_choices = []
                for choice in data["choices"]:
                    match = self.matcher.match_by_node_embedding(
                        choice,
                        target_sentence=target,
                        target_mask=target_mask,
                        seed=self.matcher_seed,
                        max_times=self.matcher_config["choices_match_max_times"],
                        max_depth=self.matcher_config["choices_match_max_depth"],
                        edge_top_k=self.matcher_config["choices_match_edge_top_k"],
                        source_context_range=self.matcher_config[
                            "choices_match_source_context_range"
                        ],
                    )
                    selection = self.matcher.select_paths(
                        match,
                        max_edges=self.matcher_config["choices_select_max_edges"],
                        discard_edges_if_rank_below=self.matcher_config[
                            "choices_select_discard_edges_if_rank_below"
                        ],
                    )

                    new_choices.append(
                        self.matcher.insert_selection_at_end_preserve_case(
                            choice, selection,
                        )
                    )
            elif self.matcher_mode == "none":
                new_question = data["text_question"]
                new_choices = data["choices"]
            else:
                raise ValueError(f"Invalid match mode {self.matcher_mode}")

            sentences, masks, type_ids = [], [], []
            for choice in new_choices:
                encoded_sentence = self.tokenizer(
                    new_question,
                    choice,
                    padding="max_length",
                    max_length=self.max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                )
                sentences.append(encoded_sentence.input_ids)
                masks.append(encoded_sentence.attention_mask)
                type_ids.append(encoded_sentence.token_type_ids)
            data["sentence"] = t.stack(sentences, dim=1)
            data["mask"] = t.stack(masks, dim=1)
            data["type_ids"] = t.stack(type_ids, dim=1)
        else:
            sentences, masks, type_ids = [], [], []
            for choice in data["choices"]:
                encoded_sentence = self.tokenizer(
                    data["text_question"],
                    choice,
                    padding="max_length",
                    max_length=self.max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                )
                sentences.append(encoded_sentence.input_ids)
                masks.append(encoded_sentence.attention_mask)
                type_ids.append(encoded_sentence.token_type_ids)
            data["sentence"] = t.stack(sentences, dim=1)
            data["mask"] = t.stack(masks, dim=1)
            data["type_ids"] = t.stack(type_ids, dim=1)
        return data

    def validate_logits(self, batch: BatchEncoding, logits: t.Tensor):
        """
        For use with a classifier model
        """
        logits = logits.cpu() - batch["choice_mask"] * 1e7
        score = t.sigmoid(logits).numpy()
        labels = np.argmax(logits.numpy(), axis=1)
        ref_labels = batch["label"].cpu().numpy()

        logit_error_type_count = [0, 0, 0, 0, 0, 0]
        logit_correct_type_count = [0, 0, 0, 0, 0, 0]

        for i in range(len(labels)):
            answer = labels[i]
            ref_answer = ref_labels[i]
            tokens = batch["sentence"][i]

            if tokens.dim() > 1:
                sentences = [
                    self.tokenizer.decode(to, skip_special_tokens=True) for to in tokens
                ]

                if answer != ref_answer:
                    for j, sentence in enumerate(sentences):
                        print(f"sentence {j}: [{sentence}] \n")
            else:
                sentence = self.tokenizer.decode(tokens, skip_special_tokens=True)
                if answer != ref_answer:
                    print(f"sentence: [{sentence}] \n")

            if answer != ref_answer:
                print(
                    f"answer: [{answer}] \n"
                    f"ref_answer: [{ref_answer}] \n"
                    f"logits: [{score[i].tolist()}]"
                )
                postive_count = np.sum(score[i] > 0.1)
                logit_error_type_count[postive_count] += 1
            else:
                postive_count = np.sum(score[i] > 0.1)
                logit_correct_type_count[postive_count] += 1
        print(f"Logit error types: {logit_error_type_count}")
        print(f"Logit correct types: {logit_correct_type_count}")
        return {"accuracy": float(np.sum(labels == ref_labels)) / len(labels)}

    def generate_test_result_logits(self, logits: t.Tensor, directory: str):
        choice_mask = t.cat([d["choice_mask"] for d in self.test_data], dim=0)
        logits = (logits.cpu() - choice_mask * 1e7).numpy()
        labels = np.argmax(logits, axis=1).tolist()
        with open_file_with_create_directories(
            os.path.join(directory, "arc.csv"), "w"
        ) as file:
            if len(labels) != len(self.test_data):
                raise ValueError(
                    f"Label size {len(labels)} does not match "
                    f"test size {len(self.test_data)}"
                )
            for label, preprocessed in zip(labels, self.test_data):
                file.write(f"{preprocessed['id']},{preprocessed['choice_labels']}\n")

    def set_corpus(self):
        corpus = []
        for data in self.train_data:
            corpus.append(
                self.matcher.tokenizer.encode(
                    data["text_question"] + " " + data["text_choices"],
                    add_special_tokens=False,
                )
            )
        print("Corpus loaded, begin setting")
        self.matcher.matcher.set_corpus(corpus)

    def set_search_targets(self, search_target: Dict[str, List[str]]):
        wnl = WordNetLemmatizer()
        for split_data in (self.train_data, self.validate_data, self.test_data):
            for data in split_data:
                if len(data["original_split"]) == 0:
                    continue
                key = f"arc_{data['original_split']}_{data['original_index']}"
                if key not in search_target:
                    raise ValueError(f"Entry {key} not found in search data")

                tokens = nltk.word_tokenize(" ".join(search_target[key][:2]))
                allowed_tokens = []
                tagged = nltk.pos_tag(tokens)
                for token, pos in tagged:
                    if pos.startswith("NN"):
                        allowed_tokens.append(wnl.lemmatize(token))

                data["target"] = allowed_tokens

    def parse_data(self, path, original_split):
        data = []
        logging.info(f"Parsing {path}")
        with open_file_with_create_directories(path, "r") as file:
            for idx, line in enumerate(file):
                entry = json.loads(line)
                choices = [ch["text"] for ch in entry["question"]["choices"]]
                choice_labels = [ch["label"] for ch in entry["question"]["choices"]]
                if len(choices) < 5:
                    diff = 5 - len(choices)
                    choices += [""] * diff
                    choice_labels += [""] * diff
                choice_mask = t.FloatTensor(
                    [[0 if len(ch) > 0 else 1 for ch in choices]]
                )
                preprocessed = {
                    "text_question": entry["question"]["stem"],
                    "text_choices": self.generate_choice_str(choices),
                    "target": [],
                    "choices": choices,
                    "choice_labels": choice_labels,
                    "choice_mask": choice_mask,
                    "id": entry["id"],
                    "original_split": original_split,
                    "original_index": idx,
                }
                if "answerKey" in entry:
                    # For BERT, ALBERT, ROBERTA, use label instead, which is an integer
                    label = [
                        i
                        for i, ch in enumerate(entry["question"]["choices"])
                        if ch["label"] == entry["answerKey"]
                    ][0]
                    preprocessed["label"] = label
                    preprocessed["text_answer"] = choices[label]
                data.append(preprocessed)
        return data

    def parse_openbook_qa_data(self, path):
        data = []
        logging.info(f"Parsing {path}")
        wnl = WordNetLemmatizer()
        with open_file_with_create_directories(path, "r") as file:
            for line in file:
                entry = json.loads(line)
                text_choices = self.generate_choice_str(
                    [ch["text"] for ch in entry["question"]["choices"]] + [""]
                )

                choices = [
                    f"{ch['text'].lower().strip(',')}"
                    for ch in entry["question"]["choices"]
                ]

                tokens = nltk.word_tokenize(entry["fact1"])
                allowed_tokens = []
                tagged = nltk.pos_tag(tokens)
                for token, pos in tagged:
                    if pos.startswith("NN"):
                        allowed_tokens.append(wnl.lemmatize(token))

                preprocessed = {
                    "text_question": entry["question"]["stem"].lower() + "?",
                    "text_choices": text_choices,
                    "target": allowed_tokens,
                    "choices": choices + [""],
                    "choice_labels": ["A", "B", "C", "D", ""],
                    "choice_mask": t.FloatTensor([[0, 0, 0, 0, 1]]),
                    "id": entry["id"],
                    "original_split": "",
                    "original_index": -1,
                }
                if "answerKey" in entry:
                    # For BERT, ALBERT, ROBERTA, use label instead, which is an integer
                    label = [
                        i
                        for i, ch in enumerate(entry["question"]["choices"])
                        if ch["label"] == entry["answerKey"]
                    ][0]
                    preprocessed["label"] = label
                    preprocessed["text_answer"] = choices[label]
                data.append(preprocessed)
        return data

    def save(self, archive_path):
        with open_file_with_create_directories(archive_path, "wb") as file:
            pickle.dump(
                {
                    "train": self.train_data,
                    "validate": self.validate_data,
                    "test": self.test_data,
                },
                file,
            )

    def generate_choice_str(self, choices: List[str]):
        return ", ".join(choices)