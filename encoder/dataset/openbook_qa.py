import os
import copy
import json
import pickle
import difflib
import logging
import numpy as np
import torch as t
from typing import List
from transformers import AutoTokenizer, PreTrainedTokenizerBase, BatchEncoding
from encoder.dataset.matcher.openbook_qa import OpenBookQAMatcher
from encoder.utils.settings import (
    dataset_cache_dir,
    preprocess_cache_dir,
)
from encoder.utils.file import (
    open_file_with_create_directories,
    download_to,
    decompress_zip,
)
from encoder.utils.inspect import save_inspect_data
from .base import StaticIterableDataset


class OpenBookQADataset:
    OPENBOOK_QA_URL = (
        "https://ai2-public-datasets.s3.amazonaws.com/open-book-qa/"
        "OpenBookQA-V1-Sep2018.zip"
    )

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 128,
        generate_length: int = 16,
        use_matcher: bool = False,
        matcher_mode: str = "embedding",
        matcher_seed: int = -1,
        matcher_config: dict = None,
        include_option_label_in_sentence: bool = False,
        include_option_label_in_answer_and_choices: bool = False,
        use_option_label_as_answer_and_choices: bool = False,
        insert_answers_at_end: bool = False,
        match_closest_when_no_equal: bool = True,
        regenerate: bool = True,
    ):
        self.tokenizer = tokenizer
        # Word piece is stabler for matching purpose
        self.matcher_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_seq_length = max_seq_length
        self.generate_length = generate_length
        self.use_matcher = use_matcher
        self.matcher_mode = matcher_mode
        self.matcher_seed = matcher_seed
        self.counter = 0
        self.matcher_config = matcher_config
        self.include_option_label_in_sentence = include_option_label_in_sentence
        self.include_option_label_in_answer_and_choices = (
            include_option_label_in_answer_and_choices
        )
        self.insert_answers_at_end = insert_answers_at_end
        self.use_option_label_as_answer_and_choices = (
            use_option_label_as_answer_and_choices
        )
        self.match_closest_when_no_equal = match_closest_when_no_equal
        self.regenerate = regenerate
        self.matcher = OpenBookQAMatcher(tokenizer=self.matcher_tokenizer)

        openbook_qa_path = os.path.join(dataset_cache_dir, "openbook_qa")
        train_path = os.path.join(
            openbook_qa_path,
            "OpenBookQA-V1-Sep2018",
            "Data",
            "Additional",
            "train_complete.jsonl",
        )
        validate_path = os.path.join(
            openbook_qa_path,
            "OpenBookQA-V1-Sep2018",
            "Data",
            "Additional",
            "dev_complete.jsonl",
        )
        test_path = os.path.join(
            openbook_qa_path,
            "OpenBookQA-V1-Sep2018",
            "Data",
            "Additional",
            "test_complete.jsonl",
        )
        archive_path = os.path.join(preprocess_cache_dir, "openbook_qa.data")
        if not os.path.exists(openbook_qa_path):
            if not os.path.exists(str(openbook_qa_path) + ".zip"):
                logging.info("Downloading OpenBook QA")
                download_to(self.OPENBOOK_QA_URL, str(openbook_qa_path) + ".zip")
            logging.info("Decompressing")
            decompress_zip(str(openbook_qa_path) + ".zip", openbook_qa_path)

        if not os.path.exists(archive_path):
            self.train_data = self.parse_data(train_path)
            self.validate_data = self.parse_data(validate_path)
            self.test_data = self.parse_data(test_path)
            self.save(archive_path)
        else:
            with open_file_with_create_directories(archive_path, "rb") as file:
                data = pickle.load(file)
            if (
                data["max_seq_length"] != self.max_seq_length
                or data["generate_length"] != self.generate_length
                or data["include_option_label_in_answer_and_choices"]
                != self.include_option_label_in_answer_and_choices
                or data["include_option_label_in_sentence"]
                != self.include_option_label_in_sentence
                or data["use_option_label_as_answer_and_choices"]
                != self.use_option_label_as_answer_and_choices
            ):
                if regenerate:
                    logging.info(
                        "Configuration mismatch, regenerating OpenBook QA dataset."
                    )
                    self.train_data = self.parse_data(train_path)
                    self.validate_data = self.parse_data(validate_path)
                    self.test_data = self.parse_data(test_path)
                    self.save(archive_path)
                else:
                    raise ValueError("Configuration mismatch")
            else:
                self.train_data = data["train"]
                self.validate_data = data["validate"]
                self.test_data = data["test"]

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
        self.counter = (self.counter + 1) % 100000000
        seed = (self.matcher_seed + self.counter) % 100000000
        if self.use_matcher:
            # prevent any modification to data, also prevent checkpoint storing
            # data to gpu by moving
            data = copy.deepcopy(data)

            if self.matcher_mode == "embedding":
                matcher_config = self.matcher_config or {
                    "max_times": 300,
                    "max_depth": 1,
                    "max_edges": 8,
                    "discard_edges_if_similarity_below": 0.45,
                }
                match = self.matcher.match_by_node_embedding(
                    data["text_choices"],
                    target_sentence=data["text_question"],
                    seed=seed,
                    **matcher_config,
                )
            elif self.matcher_mode == "token":
                matcher_config = self.matcher_config or {
                    "max_times": 300,
                    "max_depth": 1,
                    "max_edges": 8,
                }
                match = self.matcher.match_by_token(
                    data["text_choices"],
                    target_sentence=data["text_question"],
                    seed=seed,
                    **matcher_config,
                )
            else:
                raise ValueError(f"Invalid match mode {self.matcher_mode}")

            new_choices = self.matcher.insert_match(
                data["text_choices"], match, insert_at_end=self.insert_answers_at_end
            )
            encoded_sentence = self.tokenizer(
                data["text_question"],
                new_choices,
                padding="max_length",
                max_length=self.max_seq_length,
                truncation=True,
                return_tensors="pt",
            )
            data["org_sentence"] = data["sentence"]
            data["org_mask"] = data["mask"]
            data["sentence"] = encoded_sentence.input_ids
            data["mask"] = encoded_sentence.attention_mask

        return data

    def validate_logits(self, batch: BatchEncoding, logits: t.Tensor):
        """
        For use with a classifier model
        """
        logits = logits.cpu().numpy()
        labels = np.argmax(logits, axis=1)
        ref_labels = batch["label"].cpu().numpy()
        return {"accuracy": float(np.sum(labels == ref_labels)) / labels.shape[0]}

    def validate_tokens(self, batch: BatchEncoding, tokens: t.Tensor):
        """
        For use with a classifier model
        """
        total = tokens.shape[0]
        correct = 0
        approximately_correct = 0
        missing = 0
        answers = {}
        for i in range(tokens.shape[0]):
            answer = self.tokenizer.decode(tokens[i], skip_special_tokens=True)
            ref_answer_tensor = batch["answer"][i]
            ref_answer_tensor.masked_fill_(
                ref_answer_tensor == -100, self.tokenizer.pad_token_id
            )
            ref_answer = self.tokenizer.decode(
                ref_answer_tensor, skip_special_tokens=True
            )
            sentence = self.tokenizer.decode(
                batch["sentence"][i], skip_special_tokens=True
            )
            print(
                f"sentence: [{sentence}] \n"
                f"answer: [{answer}] \n"
                f"ref_answer: [{ref_answer}]"
            )
            answers[batch["id"][i]] = False
            if answer == ref_answer:
                correct += 1
                answers[batch["id"][i]] = True
            elif answer not in batch["choices"][i]:
                if self.match_closest_when_no_equal:
                    # Gestalt Pattern Matching
                    # https://en.wikipedia.org/wiki/Gestalt_Pattern_Matching
                    possible_matches = difflib.get_close_matches(
                        answer, batch["choices"][i], n=1
                    )
                    if len(possible_matches) == 0:
                        missing += 1

                    elif possible_matches[0] == ref_answer:
                        approximately_correct += 1
                        correct += 1
                        answers[batch["id"][i]] = True
                else:
                    missing += 1

        print(f"Missing ratio {float(missing) / total}")
        if self.match_closest_when_no_equal:
            print(f"Approximately correct ratio {float(approximately_correct) / total}")

        save_inspect_data(answers, "openbook_qa_val_answers")
        return {"accuracy": float(correct) / total}

    def generate_test_results_logits(self, logits: t.Tensor, directory: str):
        logits = logits.cpu().numpy()
        labels = np.argmax(logits, axis=1).tolist()
        with open_file_with_create_directories(
            os.path.join(directory, "openbook_qa.csv"), "w"
        ) as file:
            if len(labels) != len(self.test_data):
                raise ValueError(
                    f"Label size {len(labels)} does not match "
                    f"test size {len(self.test_data)}"
                )
            answer_keys = ["A", "B", "C", "D", "E"]
            for label, preprocessed in zip(labels, self.test_data):
                file.write(f"{preprocessed['id']},{answer_keys[label]}")

    def generate_test_results_tokens(self, tokens: t.Tensor, directory: str):
        missing = 0
        with open_file_with_create_directories(
            os.path.join(directory, "openbook_qa.csv"), "w"
        ) as file:
            if tokens.shape[0] != len(self.test_data):
                raise ValueError(
                    f"Token size {tokens.shape[0]} does not match "
                    f"test size {len(self.test_data)}"
                )
            answer_keys = ["A", "B", "C", "D", "E"]
            for answer_tokens, preprocessed in zip(tokens, self.test_data):
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                for i, choice in enumerate(preprocessed["choices"]):
                    if answer == choice:
                        file.write(f"{preprocessed['id']},{answer_keys[i]}")
                        break
                else:
                    missing += 1
                    print(
                        f"Missing answer, choices: {preprocessed['choices']}, "
                        f"answer: {answer}, using default A as answer."
                    )
                    file.write(f"{preprocessed['id']},A")
        print(f"Missing ratio {float(missing)/len(self.test_data)}")

    def parse_data(self, path):
        data = []
        sep = None
        if (
            hasattr(self.tokenizer, "sep_token")
            and self.tokenizer.sep_token is not None
        ):
            # BERT, ALBERT, ROBERTA
            sep = self.tokenizer.sep_token
        elif (
            hasattr(self.tokenizer, "eos_token")
            and self.tokenizer.eos_token is not None
        ):
            sep = self.tokenizer.eos_token
        with open_file_with_create_directories(path, "r") as file:
            for line in file:
                entry = json.loads(line)
                if self.include_option_label_in_sentence:
                    sentence_choices = self.generate_choice_str(
                        [ch["text"] for ch in entry["question"]["choices"]]
                    )
                else:
                    sentence_choices = ", ".join(
                        ch["text"] for ch in entry["question"]["choices"]
                    )

                if sep is not None:
                    org_sentence = (
                        entry["fact1"].capitalize()
                        + ". "
                        + entry["question"]["stem"]
                        + "?"
                        + f" {sep} "
                        + sentence_choices
                    )
                else:
                    org_sentence = (
                        entry["fact1"].capitalize()
                        + ". "
                        + entry["question"]["stem"]
                        + "?"
                        + f"  "
                        + sentence_choices
                    )
                encoded_sentence = self.tokenizer(
                    org_sentence,
                    padding="max_length",
                    max_length=self.max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                )

                if self.include_option_label_in_answer_and_choices:
                    choices = [
                        f"({ch['label'].lower()}) {ch['text'].lower()}"
                        for ch in entry["question"]["choices"]
                    ]
                elif self.use_option_label_as_answer_and_choices:
                    choices = [
                        f"({ch['label'].lower()})"
                        for ch in entry["question"]["choices"]
                    ]
                else:
                    choices = [
                        ch["text"].lower() for ch in entry["question"]["choices"]
                    ]
                preprocessed = {
                    "sentence": encoded_sentence.input_ids,
                    "mask": encoded_sentence.attention_mask,
                    "text_question": entry["fact1"].capitalize()
                    + ". "
                    + entry["question"]["stem"]
                    + "?",
                    "text_choices": sentence_choices,
                    "choices": choices,
                    "id": entry["id"],
                }
                if "answerKey" in entry:
                    # For BERT, ALBERT, ROBERTA, use label instead, which is an integer
                    preprocessed["label"] = [
                        i
                        for i, ch in enumerate(entry["question"]["choices"])
                        if ch["label"] == entry["answerKey"]
                    ][0]

                    # For T5, use answer
                    if self.include_option_label_in_answer_and_choices:
                        answer = [
                            f"({ch['label'].lower()}) {ch['text'].lower()}"
                            for ch in entry["question"]["choices"]
                            if ch["label"] == entry["answerKey"]
                        ][0]
                    elif self.use_option_label_as_answer_and_choices:
                        answer = [
                            f"({ch['label'].lower()})"
                            for ch in entry["question"]["choices"]
                            if ch["label"] == entry["answerKey"]
                        ][0]
                    else:
                        answer = [
                            ch["text"].lower()
                            for ch in entry["question"]["choices"]
                            if ch["label"] == entry["answerKey"]
                        ][0]
                    answer = self.tokenizer.encode(
                        answer,
                        padding="max_length",
                        max_length=self.generate_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    # Use -100 to focus on training the answer part, rather than pad
                    # tokens
                    answer.masked_fill_(answer == self.tokenizer.pad_token_id, -100)
                    preprocessed["answer"] = answer

                    # DEPRECATED, prepared for match by token, rank focus and exclude
                    preprocessed["answer_match"] = [
                        ch["text"]
                        for ch in entry["question"]["choices"]
                        if ch["label"] == entry["answerKey"]
                    ]

                    # DEPRECATED, prepared for match by token, rank focus and exclude
                    preprocessed["false_answers_match"] = [
                        ch["text"]
                        for ch in entry["question"]["choices"]
                        if ch["label"] != entry["answerKey"]
                    ]

                data.append(preprocessed)
        return data

    def save(self, archive_path):
        with open_file_with_create_directories(archive_path, "wb") as file:
            pickle.dump(
                {
                    "train": self.train_data,
                    "validate": self.validate_data,
                    "test": self.test_data,
                    "max_seq_length": self.max_seq_length,
                    "generate_length": self.generate_length,
                    "include_option_label_in_sentence": self.include_option_label_in_sentence,
                    "include_option_label_in_answer_and_choices": self.include_option_label_in_answer_and_choices,
                    "use_option_label_as_answer_and_choices": self.use_option_label_as_answer_and_choices,
                },
                file,
            )

    @staticmethod
    def generate_choice_str(choices: List[str]):
        result = ""
        options = ["(A)", "(B)", "(C)", "(D)", "(E)"]
        for option, choice in zip(options, choices):
            result += option + " " + choice + " "
        return result
