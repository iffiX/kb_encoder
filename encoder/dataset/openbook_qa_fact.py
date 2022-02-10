import os
import copy
import random

import nltk
import torch as t
import numpy as np
from nltk.stem import WordNetLemmatizer
from transformers import PreTrainedTokenizerBase, BatchEncoding
from encoder.utils.settings import dataset_cache_dir
from .base import StaticIterableDataset
from .openbook_qa import OpenBookQADataset


class OpenBookQAFactDataset(OpenBookQADataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        search_tokenizer: PreTrainedTokenizerBase,
        search_negative_samples: int = 4,
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
        super(OpenBookQAFactDataset, self).__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            generate_length=generate_length,
            use_matcher=use_matcher,
            matcher_mode=matcher_mode,
            matcher_seed=matcher_seed,
            matcher_config=matcher_config,
            include_option_label_in_sentence=include_option_label_in_sentence,
            include_option_label_in_answer_and_choices=include_option_label_in_answer_and_choices,
            use_option_label_as_answer_and_choices=use_option_label_as_answer_and_choices,
            insert_answers_at_end=insert_answers_at_end,
            match_closest_when_no_equal=match_closest_when_no_equal,
            regenerate=regenerate,
            output_mode=output_mode,
        )
        self.search_tokenizer = search_tokenizer
        self.search_negative_samples = search_negative_samples
        self.fact_list = self.get_fact_list()

    @property
    def train_search_dataset(self):
        return StaticIterableDataset(
            len(self.original_train_data), self.search_generator, ("train",),
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

    def generator(self, index: int, split: str):
        if split == "validate":
            data = self.validate_data[index]
        elif split == "test":
            data = self.test_data[index]
        else:
            data = None

        if data is not None and len(data["target"]) == 0:
            raise ValueError(f"Set search targets for data {split}-{data['id']} first")
        return super(OpenBookQAFactDataset, self).generator(index, split)

    def search_generator(self, index: int, split: str):
        if split == "train":
            data = self.original_train_data[index]
        elif split == "validate":
            data = self.validate_data[index]
        elif split == "test":
            data = self.test_data[index]
        else:
            raise ValueError(f"Invalid split: {split}")

        data = copy.deepcopy(data)

        sentences, masks, type_ids = [], [], []
        if split == "train":
            correct_choice = random.randint(0, self.search_negative_samples)
            available_fact_idx = list(
                set(range(len(self.fact_list))).difference(
                    {self.fact_list.index(data["fact"])}
                )
            )
            for i in range(self.search_negative_samples + 1):
                fact = (
                    data["fact"]
                    if i == correct_choice
                    else self.fact_list[random.choice(available_fact_idx)]
                )
                encoded_sentence = self.tokenizer(
                    data["text_question"] + f" ({fact}) ",
                    data["text_choices"],
                    padding="max_length",
                    max_length=90,
                    truncation=True,
                    return_tensors="pt",
                )
                sentences.append(encoded_sentence.input_ids)
                masks.append(encoded_sentence.attention_mask)
                type_ids.append(encoded_sentence.token_type_ids)
        else:
            correct_choice = self.fact_list.index(data["fact"])
            for fact in self.fact_list:
                encoded_sentence = self.tokenizer(
                    data["text_question"] + f" ({fact}) ",
                    data["text_choices"],
                    padding="max_length",
                    max_length=90,
                    truncation=True,
                    return_tensors="pt",
                )
                sentences.append(encoded_sentence.input_ids)
                masks.append(encoded_sentence.attention_mask)
                type_ids.append(encoded_sentence.token_type_ids)
        data["sentence"] = t.stack(sentences, dim=1)
        data["mask"] = t.stack(masks, dim=1)
        data["type_ids"] = t.stack(type_ids, dim=1)
        data["answer"] = correct_choice
        return data

    def validate_search(self, batch: BatchEncoding, choice: t.Tensor):
        total = choice.shape[0]
        total_correct = 0
        for i in range(choice.shape[0]):
            fact_idx = choice[i].item()
            ref_fact_idx = self.fact_list.index(batch["fact"][i])
            fact = batch["fact"][i]
            sentence = (
                batch["text_question"][i]
                + f" ({fact}) "
                + " "
                + batch["text_choices"][i]
            )
            total_correct += ref_fact_idx == fact_idx
            print(
                f"sentence: [{sentence}] \n"
                f"fact: [{self.fact_list[fact_idx]}] \n"
                f"ref_fact: [{self.fact_list[ref_fact_idx]}] \n"
            )

        return {"accuracy": total_correct / total}

    def set_search_target(self, fact_idx: int, split: str, id):
        if split == "validate":
            split_data = self.validate_data
        elif split == "test":
            split_data = self.test_data
        else:
            raise ValueError(f"Invalid split: {split}")
        found_data = [d for d in split_data if d["id"] == id]
        if len(found_data) != 1:
            raise ValueError(f"Id {id} not found in split {split}")
        search_target = self.get_gold_search_target(self.fact_list[fact_idx])
        print(f"Search target of [{split}-{id}]: {search_target}")
        search_target = search_target
        # prevent raising an exception since sometimes the target may be empty
        if len(search_target) == 0:
            search_target.append("")

        found_data[0]["target"] = search_target

    def get_gold_search_target(self, fact: str):
        tokens = nltk.word_tokenize(fact.lower())
        wnl = WordNetLemmatizer()
        allowed_tokens = []
        tagged = nltk.pos_tag(tokens)
        for token, pos in tagged:
            if pos.startswith("NN"):
                allowed_tokens.append(wnl.lemmatize(token))
        if len(allowed_tokens) < 3:
            for token, pos in tagged:
                if pos.startswith("JJ"):
                    allowed_tokens.append(wnl.lemmatize(token))
        search_target = sorted(list(set(allowed_tokens)))
        return search_target

    def get_fact_list(self):
        openbook_qa_path = os.path.join(
            dataset_cache_dir,
            "openbook_qa",
            "OpenBookQA-V1-Sep2018",
            "Data",
            "Main",
            "openbook.txt",
        )
        fact_list = []
        with open(openbook_qa_path, "r") as file:
            for line in file:
                fact_list.append(
                    line.strip("\n").strip(".").strip('"').strip("'").strip(",")
                )
        return fact_list
