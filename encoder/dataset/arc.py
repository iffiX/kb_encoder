import os
import re
import copy
import json
import tqdm
import pickle
import difflib
import logging
import nltk
import multiprocessing
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


class ARCDatasetParallelContext:
    dataset: "ARCDataset" = None


class ARCDataset:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 300,
        use_matcher: bool = False,
        matcher_mode: str = "embedding",
        matcher_seed: int = -1,
        matcher_config: dict = None,
        match_closest_when_no_equal: bool = True,
        output_mode: str = "single",
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
        self.match_closest_when_no_equal = match_closest_when_no_equal
        self.output_mode = output_mode

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
        if self.output_mode == "single":
            if self.use_matcher:
                annotation = self.generate_t5_annotation(data)
                encoded_sentence = self.tokenizer(
                    self.normalize_t5_input(
                        data["text_question"]
                        + " \\n "
                        + data["text_choices"].replace("\n", " ")
                        + annotation
                    ),
                    padding="max_length",
                    max_length=self.max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                )
                data["sentence"] = encoded_sentence.input_ids
                data["mask"] = encoded_sentence.attention_mask
            else:
                encoded_sentence = self.tokenizer(
                    self.normalize_t5_input(
                        data["text_question"]
                        + " \\n "
                        + data["text_choices"].replace("\n", " ")
                    ),
                    padding="max_length",
                    max_length=self.max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                )
                data["sentence"] = encoded_sentence.input_ids
                data["mask"] = encoded_sentence.attention_mask
            answer = self.tokenizer.encode(
                self.normalize_t5_input(data["text_answer"]),
                padding="max_length",
                max_length=16,
                truncation=True,
                return_tensors="pt",
            )
            # Use -100 to focus on training the answer part, rather than pad
            # tokens
            answer.masked_fill_(answer == self.tokenizer.pad_token_id, -100)
            data["answer"] = answer
        else:
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

    def generate_t5_annotation(self, data):
        # prevent any modification to data, also prevent checkpoint storing
        # data to gpu by moving
        data = copy.deepcopy(data)
        if self.matcher_mode == "embedding":
            if len(data["target"]) == 0:
                raise ValueError(f"Target not set for data with id {data['id']}")
            target = " @ ".join(sorted(list(set(data["target"]))))
            target_mask = []
            for c in target:
                if c == "@":
                    target_mask.append("-")
                else:
                    target_mask.append("+")
            target_mask = "".join(target_mask)

            fact_annotation = f"(facts) {', '.join(data['facts'])}"

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
                source_context_weight=1,
            )
            selection = self.matcher.select_paths(
                match,
                max_edges=self.matcher_config["question_select_max_edges"],
                discard_edges_if_rank_below=self.matcher_config[
                    "question_select_discard_edges_if_rank_below"
                ],
            )

            question_annotation = self.matcher.insert_selection_at_end_preserve_case(
                "(question) ", selection, begin="", end="",
            )

            choice_annotations = []
            for label, choice in zip(self.generate_labels(), data["choices"]):
                if len(choice) > 0:
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
                        source_context_weight=1,
                    )
                    selection = self.matcher.select_paths(
                        match,
                        max_edges=self.matcher_config["choices_select_max_edges"],
                        discard_edges_if_rank_below=self.matcher_config[
                            "choices_select_discard_edges_if_rank_below"
                        ],
                    )

                    choice_annotations.append(
                        self.matcher.insert_selection_at_end_preserve_case(
                            f"({choice}) ", selection, begin="", end=""
                        )
                    )
            annotation = (
                fact_annotation
                + " "
                + question_annotation
                + " "
                + " ".join(choice_annotations)
            )
        elif self.matcher_mode == "none":
            annotation = ""
        else:
            raise ValueError(f"Invalid match mode {self.matcher_mode}")

        annotation = " \\n " + annotation if len(annotation) > 0 else annotation
        return annotation

    def generate_all_t5_data(self):
        train_data, val_data, test_data, val_original_data, test_original_data = (
            [],
            [],
            [],
            [],
            [],
        )
        with multiprocessing.Pool(
            initializer=self.initialize_pool,
            initargs=(self.matcher_mode, self.matcher_seed, self.matcher_config),
        ) as pool:
            for split, target, source in (
                ("train", train_data, self.train_data),
                ("validate", val_data, self.validate_data),
                ("test", test_data, self.test_data),
                ("validate_original", val_original_data, self.validate_data),
                ("test_original", test_original_data, self.test_data),
            ):
                print(f"Processing {split}")
                with tqdm.tqdm(total=len(source)) as pbar:
                    for result in pool.imap_unordered(
                        self.generate_t5_input,
                        [(split, i) for i in range(len(source))],
                    ):
                        pbar.update()
                        target.append(result)

        for path, data in (
            ("arc_train_for_t5.json", train_data),
            ("arc_validate_for_t5.json", val_data),
            ("arc_test_for_t5.json", test_data),
            ("arc_validate_original_for_t5.json", val_original_data),
            ("arc_test_original_for_t5.json", test_original_data),
        ):
            with open(os.path.join(preprocess_cache_dir, path), "w") as file:
                json.dump(data, file, indent=2)

    @staticmethod
    def initialize_pool(matcher_mode, matcher_seed, matcher_config):
        ARCDatasetParallelContext.dataset = ARCDataset(
            None,
            matcher_mode=matcher_mode,
            matcher_seed=matcher_seed,
            matcher_config=matcher_config,
        )
        with open(os.path.join(preprocess_cache_dir, "arc_targets.json"), "r") as file:
            ARCDatasetParallelContext.dataset.set_search_targets(json.load(file))

    @staticmethod
    def generate_t5_input(args):
        split, index = args
        if split == "train":
            data = ARCDatasetParallelContext.dataset.train_data[index]
        elif split in ("validate", "validate_original"):
            data = ARCDatasetParallelContext.dataset.validate_data[index]
        else:
            data = ARCDatasetParallelContext.dataset.test_data[index]
        annotation = ""
        if "original" not in split:
            annotation = ARCDatasetParallelContext.dataset.generate_t5_annotation(data)
        return {
            "inputs": ARCDataset.normalize_t5_input(
                data["text_question"]
                + " \\n "
                + data["text_choices"].replace("\n", " ")
                + annotation
            ),
            "targets": ARCDataset.normalize_t5_input(data["text_answer"]),
            "choices": data["choices"],
            "label": data["label"],
            "id": data["id"],
        }

    @staticmethod
    def normalize_t5_input(text):
        text = text.lower()
        text = re.sub(r"'(.*)'", r"\1", text)
        return text

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

    def validate_tokens(self, batch: BatchEncoding, tokens: t.Tensor):
        total = tokens.shape[0]
        correct = 0
        approximately_correct = 0
        missing = 0
        answers = {}
        labels = self.generate_labels()
        for i in range(tokens.shape[0]):
            answer = self.tokenizer.decode(tokens[i], skip_special_tokens=True)
            ref_answer = self.normalize_t5_input(batch["text_answer"][i])
            sentence = self.tokenizer.decode(
                batch["sentence"][i], skip_special_tokens=True
            )
            answers[batch["id"][i]] = False
            if answer == ref_answer:
                correct += 1
                answers[batch["id"][i]] = True
            elif answer not in labels:
                if self.match_closest_when_no_equal:
                    # Gestalt Pattern Matching
                    # https://en.wikipedia.org/wiki/Gestalt_Pattern_Matching
                    possible_matches = difflib.get_close_matches(
                        answer,
                        [self.normalize_t5_input(c) for c in batch["choices"][i]],
                        n=1,
                    )
                    if len(possible_matches) == 0:
                        missing += 1

                    elif possible_matches[0] == ref_answer:
                        approximately_correct += 1
                        correct += 1
                        answers[batch["id"][i]] = True
                else:
                    missing += 1
            print(
                f"sentence: [{sentence}] \n"
                f"answer: [{answer}] \n"
                f"ref_answer: [{ref_answer}]"
            )

        print(f"Missing ratio {float(missing) / total}")
        if self.match_closest_when_no_equal:
            print(f"Approximately correct ratio {float(approximately_correct) / total}")

        return {"accuracy": float(correct) / total}

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
                file.write(
                    f"{preprocessed['id']},{preprocessed['choice_labels'][label]}\n"
                )

    def generate_test_result_tokens(self, tokens: t.Tensor, directory: str):
        missing = 0
        with open_file_with_create_directories(
            os.path.join(directory, "arc.csv"), "w"
        ) as file:
            if tokens.shape[0] != len(self.test_data):
                raise ValueError(
                    f"Token size {tokens.shape[0]} does not match "
                    f"test size {len(self.test_data)}"
                )
            for answer_tokens, preprocessed in zip(tokens, self.test_data):
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                choices = [self.normalize_t5_input(c) for c in preprocessed["choices"]]
                for i, choice in enumerate(choices):
                    if answer == choice:
                        file.write(
                            f"{preprocessed['id']},{preprocessed['choice_labels'][i]}\n"
                        )
                        break
                else:
                    is_missing = True
                    if self.match_closest_when_no_equal:
                        # Gestalt Pattern Matching
                        # https://en.wikipedia.org/wiki/Gestalt_Pattern_Matching
                        possible_matches = difflib.get_close_matches(
                            answer, choices, n=1
                        )
                        if not len(possible_matches) == 0:
                            print(
                                f"Using answer {possible_matches[0]} for output {answer}, "
                                f"question: {preprocessed['text_question']}, "
                                f"choices: {preprocessed['choices']}"
                            )
                            is_missing = False
                            file.write(
                                f"{preprocessed['id']},"
                                f"{preprocessed['choice_labels'][choices.index(possible_matches[0])]}\n"
                            )
                    if is_missing:
                        missing += 1
                        print(
                            f"Missing answer, answer: {answer}, using first label as answer."
                        )
                    file.write(
                        f"{preprocessed['id']},{preprocessed['choice_labels'][0]}"
                    )
        print(f"Missing ratio {float(missing) / len(self.test_data)}")

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

                tokens = nltk.word_tokenize(
                    " ".join(search_target[key][:3]) + " " + data["text_question"]
                )
                allowed_tokens = []
                tagged = nltk.pos_tag(tokens)
                for token, pos in tagged:
                    if pos.startswith("NN"):
                        allowed_tokens.append(wnl.lemmatize(token))

                data["target"] = allowed_tokens if len(allowed_tokens) > 0 else [""]
                data["facts"] = search_target[key]

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
                    "facts": [],
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

                choices = [ch["text"] for ch in entry["question"]["choices"]]

                text_choices = self.generate_choice_str(choices + [""])

                tokens = nltk.word_tokenize(entry["fact1"])
                allowed_tokens = []
                tagged = nltk.pos_tag(tokens)
                for token, pos in tagged:
                    if pos.startswith("NN"):
                        allowed_tokens.append(wnl.lemmatize(token))

                if len(allowed_tokens) < 3:
                    for token, pos in tagged:
                        if pos.startswith("JJ"):
                            allowed_tokens.append(wnl.lemmatize(token))
                if len(allowed_tokens) == 0:
                    allowed_tokens.append("")

                preprocessed = {
                    "text_question": entry["question"]["stem"].lower() + "?",
                    "text_choices": text_choices,
                    "target": allowed_tokens,
                    "facts": [entry["fact1"]],
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
        result = ""
        for label, choice in zip(self.generate_labels(), choices):
            if len(choice) > 0:
                result += label + " " + choice + " "
        return result

    def generate_labels(self):
        labels = []
        for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            labels.append(f"({char})")
        return labels

    def __reduce__(self):
        return (
            ARCDataset,
            (
                self.tokenizer,
                self.max_seq_length,
                self.use_matcher,
                self.matcher_mode,
                self.matcher_seed,
                self.matcher_config,
                self.match_closest_when_no_equal,
                self.output_mode,
            ),
        )
