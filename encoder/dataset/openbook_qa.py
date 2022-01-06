import os
import copy
import json
import pickle
import difflib
import logging
import nltk
import torch as t
from typing import List
from nltk.stem import WordNetLemmatizer
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
        max_seq_length: int = 300,
        generate_length: int = 32,
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
        # Note: fact is not directly used in train/test/validation
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
                wnl = WordNetLemmatizer()

                tokens = nltk.word_tokenize(data["fact"])
                allowed_tokens = []
                for token, pos in nltk.pos_tag(tokens):
                    if pos.startswith("NN"):
                        allowed_tokens.append(wnl.lemmatize(token))
                target = (
                    " ".join(sorted(list(set(allowed_tokens))))
                    + " "
                    + data["text_question"]
                )

                match = self.matcher.match_by_node_embedding(
                    data["text_question"],
                    target_sentence=target,
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

                new_question = self.matcher.insert_selection(
                    data["text_question"], selection, insert_at_end=True,
                )

                # new_question = data["text_question"] + f" ({data['fact']}) "

                choice_mask = "+" * len(data["text_choices"])
                for choice in ("[A]", "[B]", "[C]", "[D]"):
                    start_pos = data["text_choices"].find(choice)
                    if start_pos != -1:
                        choice_mask = (
                            choice_mask[:start_pos]
                            + "-" * len(choice)
                            + choice_mask[start_pos + len(choice) :]
                        )

                match = self.matcher.match_by_node_embedding(
                    data["text_choices"],
                    target_sentence=target,
                    source_mask=choice_mask,
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

                new_choices = self.matcher.insert_selection(
                    data["text_choices"], selection,
                )
                for choice in ("a", "b", "c", "d"):
                    new_choices = new_choices.replace(
                        f"[ {choice} ]", f"[{choice.upper()}]"
                    )
            elif self.matcher_mode == "none":
                new_question = data["text_question"]
                new_choices = data["text_choices"]
            else:
                raise ValueError(f"Invalid match mode {self.matcher_mode}")

            encoded_sentence = self.tokenizer(
                new_question,
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

    def validate(self, batch: BatchEncoding, tokens: t.Tensor):
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
            print(
                f"sentence: [{sentence}] \n"
                f"answer: [{answer}] \n"
                f"ref_answer: [{ref_answer}]"
            )

        print(f"Missing ratio {float(missing) / total}")
        if self.match_closest_when_no_equal:
            print(f"Approximately correct ratio {float(approximately_correct) / total}")

        save_inspect_data(answers, "openbook_qa_val_answers")
        return {"accuracy": float(correct) / total}

    def generate_test_results(self, tokens: t.Tensor, directory: str):
        missing = 0
        with open_file_with_create_directories(
            os.path.join(directory, "openbook_qa.csv"), "w"
        ) as file:
            if tokens.shape[0] != len(self.test_data):
                raise ValueError(
                    f"Token size {tokens.shape[0]} does not match "
                    f"test size {len(self.test_data)}"
                )
            answer_keys = ["A", "B", "C", "D"]
            for answer_tokens, preprocessed in zip(tokens, self.test_data):
                answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
                for i, choice in enumerate(preprocessed["choices"]):
                    if answer == choice:
                        file.write(f"{preprocessed['id']},{answer_keys[i]}\n")
                        break
                else:
                    missing += 1
                    print(
                        f"Missing answer, choices: {preprocessed['choices']}, "
                        f"answer: {answer}, using default A as answer."
                    )
                    file.write(f"{preprocessed['id']},A")
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
                        entry["question"]["stem"] + "?" + f" {sep} " + sentence_choices
                    )
                else:
                    org_sentence = (
                        entry["question"]["stem"] + "?" + f"  " + sentence_choices
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
                        f"[{ch['label'].upper()}] {ch['text'].lower().strip(',')}"
                        for ch in entry["question"]["choices"]
                    ]
                elif self.use_option_label_as_answer_and_choices:
                    choices = [
                        f"[{ch['label'].upper()}]"
                        for ch in entry["question"]["choices"]
                    ]
                else:
                    choices = [
                        ch["text"].lower().strip(",")
                        for ch in entry["question"]["choices"]
                    ]
                preprocessed = {
                    "sentence": encoded_sentence.input_ids,
                    "mask": encoded_sentence.attention_mask,
                    "text_question": entry["question"]["stem"] + "?",
                    "text_choices": sentence_choices,
                    "fact": entry["fact1"],
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
                            f"[{ch['label'].upper()}] {ch['text'].lower().strip(',')}"
                            for ch in entry["question"]["choices"]
                            if ch["label"] == entry["answerKey"]
                        ][0]
                    elif self.use_option_label_as_answer_and_choices:
                        answer = [
                            f"[{ch['label'].upper()}]"
                            for ch in entry["question"]["choices"]
                            if ch["label"] == entry["answerKey"]
                        ][0]
                    else:
                        answer = [
                            ch["text"].lower().strip(",")
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
        options = ["[A]", "[B]", "[C]", "[D]"]
        for option, choice in zip(options, choices):
            result += option + " " + choice + " "
        return result
