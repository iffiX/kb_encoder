import os
import re
import copy
import json
import pickle
import difflib
import logging
import nltk
import numpy as np
import torch as t
import random
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
from .utils import num2word
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
        output_mode: str = "single",
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

        if output_mode not in ("single", "splitted"):
            raise ValueError(f"Invalid output_mode {output_mode}")
        self.output_mode = output_mode
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
                data["include_option_label_in_answer_and_choices"]
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

        if output_mode == "splitted":
            self.normalize_training_data()

        if use_matcher:
            self.add_associated_fact_training_data()
        self.add_arithmetic_training_data()
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
                # prevent any modification to data, also prevent checkpoint storing
                # data to gpu by moving
                data = copy.deepcopy(data)
                if self.matcher_mode == "embedding":
                    wnl = WordNetLemmatizer()

                    tokens = nltk.word_tokenize(data["fact"])
                    allowed_tokens = []
                    tagged = nltk.pos_tag(tokens)
                    for token, pos in tagged:
                        if (
                            pos.startswith("NN")
                            # or pos.startswith("JJ")
                            # or (
                            #     pos.startswith("VB")
                            #     and token not in self.matcher.VERB_FILTER_SET
                            # )
                        ):
                            allowed_tokens.append(wnl.lemmatize(token))
                    if len(allowed_tokens) < 3:
                        for token, pos in tagged:
                            if pos.startswith("JJ"):
                                allowed_tokens.append(wnl.lemmatize(token))

                    # target = (
                    #     " ".join(sorted(list(set(allowed_tokens))))
                    #     + " "
                    #     + data["text_question"]
                    # )

                    target = " @ ".join(sorted(list(set(allowed_tokens))))
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
                        target_mask=target_mask,
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
                data["sentence"] = encoded_sentence.input_ids
                data["mask"] = encoded_sentence.attention_mask
            else:
                encoded_sentence = self.tokenizer(
                    data["text_question"],
                    data["text_choices"],
                    padding="max_length",
                    max_length=self.max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                )
                data["sentence"] = encoded_sentence.input_ids
                data["mask"] = encoded_sentence.attention_mask
            answer = self.tokenizer.encode(
                data["text_answer"],
                padding="max_length",
                max_length=self.generate_length,
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
                    wnl = WordNetLemmatizer()

                    tokens = nltk.word_tokenize(data["fact"])
                    allowed_tokens = []
                    tagged = nltk.pos_tag(tokens)
                    for token, pos in tagged:
                        if (
                            pos.startswith("NN")
                            # or pos.startswith("JJ")
                            # or (
                            #     pos.startswith("VB")
                            #     and token not in self.matcher.VERB_FILTER_SET
                            # )
                        ):
                            allowed_tokens.append(wnl.lemmatize(token))

                    if len(allowed_tokens) < 3:
                        for token, pos in tagged:
                            if pos.startswith("JJ"):
                                allowed_tokens.append(wnl.lemmatize(token))

                    # target = (
                    #     " ".join(sorted(list(set(allowed_tokens))))
                    #     + " "
                    #     + data["text_question"]
                    # )

                    target = " @ ".join(sorted(list(set(allowed_tokens))))
                    target_mask = []
                    for c in target:
                        if c == "@":
                            target_mask.append("-")
                        else:
                            target_mask.append("+")
                    target_mask = "".join(target_mask)

                    if "exact_fact" not in data["id"]:
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

                        new_question = self.matcher.insert_selection(
                            data["text_question"], selection, insert_at_end=True,
                        )
                    else:
                        new_question = data["text_question"] + f" ({data['fact']}) "

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
                            self.matcher.insert_selection(choice, selection,)
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
        score = t.sigmoid(logits.cpu()).numpy()
        logits = logits.cpu().numpy()
        # labels = np.argmax(logits, axis=1)
        labels = self.apply_logical_ops_to_logits(batch["choices"], logits)
        ref_labels = batch["label"].cpu().numpy()

        logit_error_type_count = [0, 0, 0, 0, 0]
        logit_correct_type_count = [0, 0, 0, 0, 0]

        for i in range(len(labels)):
            answer = ["A", "B", "C", "D"][labels[i]]
            ref_answer = ["A", "B", "C", "D"][batch["label"][i]]
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

    def generate_test_result_logits(self, logits: t.Tensor, directory: str):
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
            answer_keys = ["A", "B", "C", "D"]
            for label, preprocessed in zip(labels, self.test_data):
                file.write(f"{preprocessed['id']},{answer_keys[label]}\n")

    def generate_test_result_tokens(self, tokens: t.Tensor, directory: str):
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
        logging.info(f"Parsing {path}")
        with open_file_with_create_directories(path, "r") as file:
            for line in file:
                entry = json.loads(line)
                text_choices = self.generate_choice_str(
                    [ch["text"] for ch in entry["question"]["choices"]]
                )

                choices = [
                    f"{ch['text'].lower().strip(',')}"
                    for ch in entry["question"]["choices"]
                ]

                preprocessed = {
                    "text_question": entry["question"]["stem"].lower() + "?",
                    "text_choices": text_choices,
                    "fact": entry["fact1"],
                    "choices": choices,
                    "id": entry["id"],
                }
                if "answerKey" in entry:
                    # For BERT, ALBERT, ROBERTA, use label instead, which is an integer
                    label = [
                        i
                        for i, ch in enumerate(entry["question"]["choices"])
                        if ch["label"] == entry["answerKey"]
                    ][0]
                    preprocessed["label"] = label
                    preprocessed["text_answer"] = self.generate_text_answer(
                        label, choices[label]
                    )
                data.append(preprocessed)
        return data

    def save(self, archive_path):
        with open_file_with_create_directories(archive_path, "wb") as file:
            pickle.dump(
                {
                    "train": self.train_data,
                    "validate": self.validate_data,
                    "test": self.test_data,
                    "include_option_label_in_sentence": self.include_option_label_in_sentence,
                    "include_option_label_in_answer_and_choices": self.include_option_label_in_answer_and_choices,
                    "use_option_label_as_answer_and_choices": self.use_option_label_as_answer_and_choices,
                },
                file,
            )

    def add_associated_fact_training_data(self):
        new_train_data = copy.deepcopy(self.train_data)
        for td in new_train_data:
            td["id"] += "_exact_fact"
        self.train_data = self.train_data + new_train_data

    def add_arithmetic_training_data(self):
        generator = random.Random(42)
        # TIME ARITHMETIC
        question_template = "{month_1} is {time_diff} {direction}?"
        answer_template = "{month_2}"
        months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        for i in range(100):
            month_1, month_2 = generator.choices(list(range(12)), k=2)
            direction = generator.choice(["before", "after"])
            if direction == "before":
                time_diff = (month_2 + 12 - month_1) % 12
            else:
                time_diff = (month_1 + 12 - month_2) % 12
            unit, unit_scale = generator.choice(
                ##[("month", 1), ("week", 4), ("day", 30)]
                [("month", 1)]
            )
            wrong_months = months[:month_2] + months[month_2 + 1 :]
            generator.shuffle(wrong_months)
            choices = [
                answer_template.format(month_2=months[month_2]),
                answer_template.format(month_2=wrong_months[0]),
                answer_template.format(month_2=wrong_months[1]),
                answer_template.format(month_2=wrong_months[2]),
            ]
            sample = {
                "text_question": question_template.format(
                    month_1=months[month_1],
                    time_diff=f"{num2word(time_diff * unit_scale)} "
                    f"{unit}{'s' if time_diff * unit_scale > 1 else ''}",
                    direction=direction,
                ),
                "text_choices": self.generate_choice_str(choices),
                "fact": "month week day",
                "choices": choices,
                "id": f"gen-ar-time-{i}",
                "label": 0,
                "text_answer": self.generate_text_answer(0, choices[0]),
            }
            self.train_data.append(sample)
        logging.info("Added 100 samples of time arithmetic")

        # TIME CONVERSION
        time_units = ["day", "week", "month", "season", "year"]
        time_unit_names = {
            "day": ["day"],  # ["day", "24-hour period", "working day", "solar day"],
            "week": ["week"],  # ["week", "weekend"],
            "month": [
                "month"
            ],  # ["month", "calendar page", "full moon phase", "lunar month"],
            "season": ["season"],  # ["season", "3-month period", "quarter of year"],
            "year": ["year"],
        }
        time_unit_conversion = [
            [-1, 7, 30, 90, 365],
            [-1, -1, 4, 12, 52],
            [-1, -1, -1, 3, 12],
            [-1, -1, -1, -1, 4],
            [-1, -1, -1, -1, -1],
        ]
        question_template = (
            "How many {time_unit_1}s are there in {time_number_2} {time_unit_2}?"
        )
        answer_template = "{time_number_1}"
        for i in range(100):
            time_unit_1 = generator.choice(time_units[:4])
            time_unit_1_name = generator.choice(time_unit_names[time_unit_1])
            time_unit_2 = generator.choice(
                time_units[time_units.index(time_unit_1) + 1 :]
            )
            time_unit_2_name = generator.choice(time_unit_names[time_unit_2])

            time_number_2 = generator.randint(1, 3)
            time_number_1 = (
                time_number_2
                * time_unit_conversion[time_units.index(time_unit_1)][
                    time_units.index(time_unit_2)
                ]
            )
            wrong_numbers = list(
                set(range(1, time_number_1 * 2)).difference({time_number_1})
            )
            generator.shuffle(wrong_numbers)
            choices = [
                answer_template.format(time_number_1=num2word(time_number_1)),
                answer_template.format(time_number_1=num2word(wrong_numbers[0])),
                answer_template.format(time_number_1=num2word(wrong_numbers[1])),
                answer_template.format(time_number_1=num2word(wrong_numbers[2])),
            ]
            sample = {
                "text_question": question_template.format(
                    time_unit_1=time_unit_1_name,
                    time_number_2=num2word(time_number_2),
                    time_unit_2=f"{time_unit_2_name}{'s' if time_number_2 > 1 else ''}",
                ),
                "text_choices": self.generate_choice_str(choices),
                "fact": "month week day",
                "choices": choices,
                "id": f"gen-ar-timeconv-{i}",
                "label": 0,
                "text_answer": self.generate_text_answer(0, choices[0]),
            }
            self.train_data.append(sample)
        logging.info("Added 100 samples of time conversion")

    def normalize_training_data(self):
        append_train_data = []
        delete_train_data = set()
        data = self.train_data
        train_data_num = len(data)
        available_choices = list(
            set([ch for train_data in data for ch in train_data["choices"]]).difference(
                {"all of these", "none of these"}
            )
        )
        generator = random.Random(42)
        for train_idx, train_data in enumerate(data):
            correct_choice = train_data["choices"][train_data["label"]].lower()
            if correct_choice == "all of these":
                # randomly choose 3 choices from other samples
                choice_num = len(train_data["choices"])
                for choice_idx in range(choice_num):
                    if choice_idx != train_data["label"]:
                        new_train_data = copy.deepcopy(train_data)
                        new_train_data["label"] = choice_idx
                        for other_choice_idx in range(choice_num):
                            if other_choice_idx != choice_idx:
                                new_train_data["choices"][
                                    other_choice_idx
                                ] = generator.choice(available_choices)
                        append_train_data.append(new_train_data)
                delete_train_data.add(train_idx)
            elif correct_choice == "none of these":
                delete_train_data.add(train_idx)
        new_data = [
            train_data
            for train_idx, train_data in enumerate(data)
            if train_idx not in delete_train_data
        ] + append_train_data
        logging.info(
            f"Appended {len(append_train_data)} samples, Deleted {len(delete_train_data)} samples"
        )
        self.train_data = new_data

    @staticmethod
    def apply_logical_ops_to_logits(choices: List[List[str]], logits: np.ndarray):
        labels = []
        for ch, lo in zip(choices, logits):
            ch = [c.lower() for c in ch]
            all_of_index, none_of_index, multiple_index = None, None, None
            multiple_choices = None
            if "all of these" in ch:
                all_of_index = ch.index("all of these")
            if "none of these" in ch:
                none_of_index = ch.index("none of these")
            for idx, c in enumerate(ch):
                match_result = re.match(r"([abcd]) and ([abcd])", c)
                if match_result is not None:
                    multiple_index = idx
                    multiple_choices = [
                        ord(match_result.group(1)) - ord("a"),
                        ord(match_result.group(2)) - ord("a"),
                    ]
            if any(
                [
                    all_of_index is not None,
                    # none_of_index is not None,
                    multiple_index is not None,
                ]
            ):
                normal_choices = list(
                    {0, 1, 2, 3}.difference(
                        {all_of_index, none_of_index, multiple_index}
                    )
                )
                if all_of_index is not None and np.all(lo[normal_choices] > 0.1):
                    labels.append(all_of_index)
                elif none_of_index is not None and np.all(lo[normal_choices] < 0.1):
                    labels.append(none_of_index)
                elif multiple_index is not None and np.all(lo[multiple_choices] > 0.1):
                    labels.append(multiple_index)
                else:
                    labels.append(normal_choices[np.argmax(lo[normal_choices])])
            else:
                labels.append(np.argmax(lo))
        return np.array(labels)

    def generate_choice_str(self, choices: List[str]):
        if self.include_option_label_in_sentence:
            result = ""
            options = ["[A]", "[B]", "[C]", "[D]"]
            for option, choice in zip(options, choices):
                result += option + " " + choice + " "
            return result
        else:
            return ", ".join(choices)

    def generate_text_answer(self, label, choice):
        options = ["[A]", "[B]", "[C]", "[D]"]
        if self.include_option_label_in_answer_and_choices:
            text_answer = f"{options[label]} {choice.lower().strip(',')}"
        elif self.use_option_label_as_answer_and_choices:
            text_answer = f"{options[label]}"
        else:
            text_answer = choice.lower().strip(",")
        return text_answer
