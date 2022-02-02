import os
import copy
import json
import pickle
import random
import difflib
import logging
import nltk
import torch as t
from typing import List
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from nltk.tokenize.treebank import TreebankWordDetokenizer
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


def pool_initializer():
    matcher_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    matcher = OpenBookQAMatcher(tokenizer=matcher_tokenizer)
    OpenBookQAWithSearchDataset.matcher = matcher


class OpenBookQAWithSearchDataset:
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
        include_prefix: bool = False,
        include_option_label_in_sentence: bool = False,
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
        self.include_prefix = include_prefix
        self.include_option_label_in_sentence = include_option_label_in_sentence
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
        archive_path = os.path.join(
            preprocess_cache_dir, "openbook_qa_with_search.data"
        )
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
            # self.rank_search_target_of_train()
            self.save(archive_path)
        else:
            with open_file_with_create_directories(archive_path, "rb") as file:
                data = pickle.load(file)
            if (
                data["include_option_label_in_sentence"]
                != self.include_option_label_in_sentence
                or data["use_option_label_as_answer_and_choices"]
                != self.use_option_label_as_answer_and_choices
            ):
                if regenerate:
                    logging.info(
                        "Configuration mismatch, regenerating OpenBook QA (With search) dataset."
                    )
                    self.train_data = self.parse_data(train_path)
                    self.validate_data = self.parse_data(validate_path)
                    self.test_data = self.parse_data(test_path)
                    # self.rank_search_target_of_train()
                    self.save(archive_path)
                else:
                    raise ValueError("Configuration mismatch")
            else:
                self.train_data = data["train"]
                self.validate_data = data["validate"]
                self.test_data = data["test"]
        self.validate_search_targets = {}
        self.test_search_targets = {}
        self.set_corpus()

    @property
    def train_qa_dataset(self):
        return StaticIterableDataset(
            len(self.train_data), self.qa_generator, ("train",),
        )

    @property
    def validate_qa_dataset(self):
        return StaticIterableDataset(
            len(self.validate_data) * 2, self.qa_generator, ("validate",)
        )

    @property
    def test_qa_dataset(self):
        return StaticIterableDataset(len(self.test_data), self.qa_generator, ("test",))

    @property
    def train_search_dataset(self):
        return StaticIterableDataset(
            len(self.train_data), self.search_generator, ("train",),
        )

    @property
    def validate_search_dataset(self):
        return StaticIterableDataset(
            len(self.validate_data), self.search_generator, ("validate",)
        )

    @property
    def test_search_dataset(self):
        return StaticIterableDataset(
            len(self.test_data), self.search_generator, ("test",)
        )

    def qa_generator(self, index: int, split: str):
        if split == "train":
            data = self.train_data[index]
            search_targets = {}
        elif split == "validate":
            data = self.validate_data[index % len(self.validate_data)]
            search_targets = self.validate_search_targets
        elif split == "test":
            data = self.test_data[index]
            search_targets = self.test_search_targets
        else:
            raise ValueError(f"Invalid split: {split}")
        if self.use_matcher:
            # prevent any modification to data, also prevent checkpoint storing
            # data to gpu by moving
            data = copy.deepcopy(data)
            if self.matcher_mode == "embedding":
                if split == "validate" and index >= len(self.validate_data):
                    data["id"] = data["id"] + "_real"

                if split == "train" or (
                    split == "validate" and index >= len(self.validate_data)
                ):
                    target = " ".join(data["target"]) + " " + data["text_question"]
                else:
                    if data["id"] not in search_targets:
                        raise ValueError("Set search targets first")
                    target = search_targets[data["id"]] + " " + data["text_question"]
                match = self.matcher.match_by_node_embedding(
                    data["text_question"],
                    target_sentence=target,
                    seed=self.matcher_seed,
                    max_times=self.matcher_config["question_match_max_times"],
                    max_depth=self.matcher_config["question_match_max_depth"],
                    edge_top_k=self.matcher_config["question_match_edge_top_k"],
                )
                selection = self.matcher.select_paths(
                    match, max_edges=self.matcher_config["question_select_max_edges"]
                )
                new_question = self.matcher.insert_selection(
                    data["text_question"], selection, insert_at_end=True,
                )

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
                )
                selection = self.matcher.select_paths(
                    match, max_edges=self.matcher_config["choices_select_max_edges"]
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
                new_question if not self.include_prefix else "predict: " + new_question,
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

    def search_generator(self, index: int, split: str):
        if split == "train":
            data = self.train_data[index]
        elif split == "validate":
            data = self.validate_data[index]
        elif split == "test":
            data = self.test_data[index]
        else:
            raise ValueError(f"Invalid split: {split}")

        # data = copy.deepcopy(data)
        #
        # match = self.matcher.match_by_node_embedding(
        #     data["text_question"],
        #     target_sentence=data["text_question"],
        #     seed=self.matcher_seed,
        #     max_depth=1,
        #     max_times=1000,
        #     edge_top_k=10,
        # )
        # selection = self.matcher.select_paths(match, max_edges=3)
        # question = self.matcher.insert_selection(
        #     data["text_question"], selection, insert_at_end=True,
        # )
        #
        # if self.include_prefix:
        #     question = "search: " + question
        #
        # choice_mask = "+" * len(data["text_choices"])
        # for choice in ("[A]", "[B]", "[C]", "[D]"):
        #     start_pos = data["text_choices"].find(choice)
        #     if start_pos != -1:
        #         choice_mask = (
        #             choice_mask[:start_pos]
        #             + "-" * len(choice)
        #             + choice_mask[start_pos + len(choice) :]
        #         )
        #
        # match = self.matcher.match_by_node_embedding(
        #     data["text_choices"],
        #     target_sentence=data["text_question"],
        #     source_mask=choice_mask,
        #     seed=self.matcher_seed,
        #     max_depth=1,
        #     max_times=1000,
        # )
        # selection = self.matcher.select_paths(match, max_edges=6)
        # choices = self.matcher.insert_selection(data["text_choices"], selection,)
        # for choice in ("a", "b", "c", "d"):
        #     choices = choices.replace(f"[ {choice} ]", f"[{choice.upper()}]")

        question = (
            "search: " + data["text_question"]
            if self.include_prefix
            else data["text_question"]
        )
        choices = data["text_choices"]

        encoded_sentence = self.tokenizer(
            question,
            choices,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )
        answer = self.tokenizer.encode(
            " ".join(data["target"]),
            padding="max_length",
            max_length=32,
            truncation=True,
            return_tensors="pt",
        )
        # Use -100 to focus on training the answer part, rather than pad
        # tokens
        answer.masked_fill_(answer == self.tokenizer.pad_token_id, -100)

        data["sentence"] = encoded_sentence.input_ids
        data["mask"] = encoded_sentence.attention_mask
        data["answer"] = answer
        return data

    def validate_qa(self, batch: BatchEncoding, tokens: t.Tensor):
        total = tokens.shape[0]
        correct = 0
        approximately_correct = 0
        missing = 0
        is_answer_correct = {}
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
            is_answer_correct[batch["id"][i]] = False
            if answer == ref_answer:
                correct += 1
                is_answer_correct[batch["id"][i]] = True
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
                        is_answer_correct[batch["id"][i]] = True
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

        base_ids = [k for k in is_answer_correct.keys() if "_real" not in k]
        for bid in base_ids:
            if not is_answer_correct[bid] and is_answer_correct[bid + "_real"]:
                i, i_real = sorted(
                    [
                        i
                        for i, id in enumerate(batch["id"])
                        if id in (bid, bid + "_real")
                    ]
                )
                ref_answer_tensor = batch["answer"][i]
                ref_answer_tensor.masked_fill_(
                    ref_answer_tensor == -100, self.tokenizer.pad_token_id
                )
                ref_answer = self.tokenizer.decode(
                    ref_answer_tensor, skip_special_tokens=True
                )
                sentence_base = self.tokenizer.decode(
                    batch["sentence"][i], skip_special_tokens=True
                )
                sentence_real = self.tokenizer.decode(
                    batch["sentence"][i_real], skip_special_tokens=True
                )
                answer_base = self.tokenizer.decode(tokens[i], skip_special_tokens=True)
                answer_real = self.tokenizer.decode(
                    tokens[i_real], skip_special_tokens=True
                )
                print(
                    f"real sentence: [{sentence_real}] \n"
                    f"real answer: [{answer_real}] \n"
                    f"real target: [{' '.join(batch['target'][i])}] \n"
                    f"base sentence: [{sentence_base}] \n"
                    f"base answer: [{answer_base}] \n"
                    f"base target: [{self.validate_search_targets[bid]}] \n"
                    f"ref_answer: [{ref_answer}]"
                )

        save_inspect_data(is_answer_correct, "openbook_qa_val_answers")
        return {"accuracy": float(correct) / total}

    def validate_search(self, batch: BatchEncoding, tokens: t.Tensor):
        total = tokens.shape[0]
        total_f1 = 0
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
                batch["sentence"][i], skip_special_tokens=True,
            )

            keywords = set(nltk.word_tokenize(answer.lower()))
            ref_keywords = set(nltk.word_tokenize(ref_answer.lower()))
            intersection = ref_keywords.intersection(keywords)

            if len(ref_keywords) == 0:
                f1 = 1
            else:
                precision = len(intersection) / (len(keywords) + 1e-6)
                recall = len(intersection) / (len(ref_keywords) + 1e-6)
                f1 = (2 * precision * recall) / (precision + recall + 1e-6)

            total_f1 += f1
            print(
                f"sentence: [{sentence}] \n"
                f"keywords: [{keywords}] \n"
                f"ref_keywords: [{ref_keywords}] \n"
                f"f1: {f1}"
            )

        return {"f1": total_f1 / total}

    def test_qa(self, batch: BatchEncoding, tokens: t.Tensor):
        total = tokens.shape[0]
        correct = 0
        approximately_correct = 0
        missing = 0
        is_answer_correct = {}
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
            is_answer_correct[batch["id"][i]] = False
            if answer == ref_answer:
                correct += 1
                is_answer_correct[batch["id"][i]] = True
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
                        is_answer_correct[batch["id"][i]] = True
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
        print(f"Missing ratio {float(missing)/len(self.test_data)}")

    def set_search_target(self, tokens: t.Tensor, split: str, id):
        if split == "validate":
            split_data = self.validate_data
            search_targets = self.validate_search_targets
        elif split == "test":
            split_data = self.test_data
            search_targets = self.test_search_targets
        else:
            raise ValueError(f"Invalid split: {split}")
        if tokens.ndim != 1:
            raise ValueError("Token tensor must have a dimension number of 1")
        found = any(d["id"] == id for d in split_data)
        if not found:
            raise ValueError(f"Id {id} not found in split {split}")
        raw_search_target = self.tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"Search target of [{id}]: {raw_search_target}")
        search_target = sorted(list(set(nltk.word_tokenize(raw_search_target.lower()))))
        search_targets[id] = " ".join(search_target)

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

    def rank_search_target_of_train(self):
        print("Ranking target words of training set")
        for data in self.train_data:
            source = data["text_question"] + " " + data["text_choices"]
            wnl = WordNetLemmatizer()
            tokens = [wnl.lemmatize(t).lower() for t in nltk.word_tokenize(source)]
            new_target = []
            for keyphrase in data["target"]:
                keyphrase = keyphrase.lower()
                index = tokens.index(keyphrase) if keyphrase in tokens else 1000000
                new_target.append((keyphrase, index))
            data["target"] = [x[0] for x in sorted(new_target, key=lambda x: x[1])]
        # with Pool(initializer=pool_initializer, processes=cpu_count() - 1) as p:
        #     simplified_train_data = [
        #         {
        #             "target": d["target"],
        #             "text_question": d["text_question"],
        #             "text_choices": d["text_choices"],
        #         }
        #         for d in self.train_data
        #     ]
        #     new_targets = list(
        #         tqdm(
        #             p.imap(
        #                 self.get_ranked_search_target_of_train, simplified_train_data
        #             ),
        #             total=len(simplified_train_data),
        #         )
        #     )
        #     for data, new_target in zip(self.train_data, new_targets):
        #         data["target"] = new_target

    @classmethod
    def get_ranked_search_target_of_train(cls, data):
        target = data["target"].copy()
        new_target = []
        base_knowledge = cls.get_selected_knowledge(data, data["target"])
        # Only rank top 2
        while target and len(new_target) < 2:
            knowledge_list = [
                cls.get_selected_knowledge(data, new_target + [t]) for t in target
            ]
            score_list = [
                cls.knowledge_score(base_knowledge, kn) for kn in knowledge_list
            ]
            most_important_target = max(
                [(t, sc) for t, sc in zip(target, score_list)], key=lambda x: x[1],
            )[0]
            new_target.append(most_important_target)
            target.pop(target.index(most_important_target))
        new_target += target
        return new_target

    @classmethod
    def get_selected_knowledge(cls, data, target):
        search_target = " ".join(target) + " " + data["text_question"]
        knowledge = []
        match = cls.matcher.match_by_node_embedding(
            data["text_question"],
            target_sentence=search_target,
            seed=1394823,
            max_times=1000,
            max_depth=3,
        )
        selection = cls.matcher.select_paths(match, max_edges=3)
        knowledge += cls.matcher.selection_to_list_of_strings(selection)

        choice_mask = "+" * len(data["text_choices"])
        for choice in ("[A]", "[B]", "[C]", "[D]"):
            start_pos = data["text_choices"].find(choice)
            if start_pos != -1:
                choice_mask = (
                    choice_mask[:start_pos]
                    + "-" * len(choice)
                    + choice_mask[start_pos + len(choice) :]
                )
        match = cls.matcher.match_by_node_embedding(
            data["text_choices"],
            target_sentence=search_target,
            source_mask=choice_mask,
            seed=1394823,
            max_depth=2,
            max_times=1000,
        )
        selection = cls.matcher.select_paths(match, max_edges=8)
        knowledge += cls.matcher.selection_to_list_of_strings(selection)
        return knowledge

    @staticmethod
    def knowledge_score(ref_knowledge: List[str], knowledge: List[str]):
        knowledge = set(knowledge)
        ref_knowledge = set(ref_knowledge)
        intersection = ref_knowledge.intersection(knowledge)

        if len(ref_knowledge) == 0:
            f1 = 1
        else:
            precision = len(intersection) / (len(knowledge) + 1e-6)
            recall = len(intersection) / (len(ref_knowledge) + 1e-6)
            f1 = (2 * precision * recall) / (precision + recall + 1e-6)
        return f1

    def parse_data(self, path):
        data = []
        sep = None
        if (
            hasattr(self.tokenizer, "sep_token")
            and self.tokenizer.sep_token is not None
        ):
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
                    "predict: " + org_sentence,
                    padding="max_length",
                    max_length=self.max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                )

                if self.use_option_label_as_answer_and_choices:
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
                    "target": self.get_gold_search_target(entry["fact1"]),
                    "choices": choices,
                    "id": entry["id"],
                }
                if "answerKey" in entry:
                    if self.use_option_label_as_answer_and_choices:
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

    def get_gold_search_target(self, fact: str):
        tokens = nltk.word_tokenize(fact.lower())
        wnl = WordNetLemmatizer()
        allowed_tokens = []
        for token, pos in nltk.pos_tag(tokens):
            if pos.startswith("NN"):
                allowed_tokens.append(wnl.lemmatize(token))
        search_target = sorted(list(set(allowed_tokens)))
        return search_target

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
