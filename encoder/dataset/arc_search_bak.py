import os
import copy
import json
import pickle
import logging
import random
import torch as t
import nltk
from typing import List
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, PreTrainedTokenizerBase, BatchEncoding
from encoder.utils.settings import (
    preprocess_cache_dir,
    proxies,
    model_cache_dir,
    huggingface_mirror,
    local_files_only,
)
from encoder.utils.file import open_file_with_create_directories
from .base import StaticIterableDataset
from .download import OpenBookQA, QASC, ARC


class KeywordNode:
    def __init__(self):
        self.children = []
        self.covered_indexes = []


class KeywordTree:
    def __init__(self, corpus: List[str], minimum_branch_size: int = 10):
        self.root = self.buildTree(KeywordNode(), corpus)
        self.corpus = corpus
        self.minimum_branch_size = minimum_branch_size
        # corpus_words: List[List[str]]
        # word_index: Dict[str, List[int]]
        self.corpus_words, self.word_index = self.buildWordIndex(
            list(range(len(corpus)))
        )

    def buildTree(self, parent, corpus_indexes):
        word_ranks = self.rankWords(corpus_indexes)
        parent.covered_indexes = corpus_indexes
        for word, _covered_size, covered_indexes in word_ranks:
            if word is not None:
                parent.children.append(self.buildTree(KeywordNode(), covered_indexes))
        return parent

    def rankWords(self, corpus_indexes):
        ranks = []
        corpus_indexes = set(corpus_indexes)
        while len(corpus_indexes) > 0:
            coverage = []
            for word, covered_indexes in self.word_index.items():
                covered_sub_indexes = set(covered_indexes).intersection(corpus_indexes)
                coverage.append((word, len(covered_sub_indexes), covered_sub_indexes))
            best_sub_cover = max(coverage, key=lambda c: c[1])
            if best_sub_cover[1] >= self.minimum_branch_size:
                ranks.append(best_sub_cover)
                corpus_indexes = corpus_indexes.difference(best_sub_cover[2])
            else:
                break
        # Use a "miscellaneous" node to cover remaining sentences
        if len(corpus_indexes) > 0:
            ranks.append((None, len(corpus_indexes), corpus_indexes))
        return ranks

    def buildWordIndex(self, corpus):
        wnl = WordNetLemmatizer()
        corpus_words = []
        word_index = {}
        for idx, sentence in enumerate(corpus):
            words = []
            tokens = nltk.word_tokenize(sentence)
            tagged = nltk.pos_tag(tokens)
            for token, pos in tagged:
                if pos.startswith("NN"):
                    words.append(wnl.lemmatize(token))
            corpus_words.append(words)
            for word in words:
                if word in word_index:
                    word_index[word].append(idx)
                else:
                    word_index[word] = [idx]
        return corpus_words, word_index


class NegativeFactSampler:
    def __init__(self, facts, target_fact, seed=42):
        self.facts = facts
        self.target_fact = target_fact
        self.generator = random.Random(seed)
        self.available_fact_idx = list(
            set(range(len(self.facts))).difference({self.facts.index(target_fact)})
        )

    def __call__(self, exclude_indexes=None):
        available_fact_idx = self.available_fact_idx
        if exclude_indexes is not None:
            available_fact_idx = list(
                set(available_fact_idx).difference(set(exclude_indexes))
            )
        return self.facts[self.generator.choice(available_fact_idx)]


class ARCSearchDataset:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        search_negative_samples: int = 4,
        max_seq_length: int = 300,
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
        self.search_negative_samples = search_negative_samples
        self.max_seq_length = max_seq_length
        self.openbook_qa = OpenBookQA().require()
        self.arc = ARC().require()
        self.qasc = QASC().require()

        archive_path = os.path.join(preprocess_cache_dir, "arc_search.data")
        if not os.path.exists(archive_path):
            self.facts = self.generate_facts()
            (
                self.train_data,
                self.unannotated_train_data,
                self.search_data,
            ) = self.generate_train_data()
            self.validate_data = self.generate_validate_data()
            self.save(archive_path)
        else:
            with open_file_with_create_directories(archive_path, "rb") as file:
                data = pickle.load(file)
                self.facts = data["facts"]
                self.train_data = data["train"]
                self.unannotated_train_data = data["unannotated_train"]
                self.search_data = data["search"]
                self.validate_data = data["validate"]
        self.negative_fact_gen = {
            fact: NegativeFactSampler(self.facts, fact) for fact in self.facts
        }

    @property
    def train_dataset(self):
        return StaticIterableDataset(len(self.train_data), self.generator, ("train",),)

    @property
    def unannotated_train_dataset(self):
        return StaticIterableDataset(
            len(self.unannotated_train_data), self.generator, ("unannotated_train",),
        )

    @property
    def validate_dataset(self):
        return StaticIterableDataset(
            len(self.validate_data), self.generator, ("validate",)
        )

    @property
    def search_dataset(self):
        return StaticIterableDataset(
            len(self.search_data), self.generator, ("search",),
        )

    def generator(self, index: int, split: str):
        if split == "train":
            data = self.train_data[index]
        elif split == "unannotated_train":
            data = self.unannotated_train_data[index]
        elif split == "validate":
            data = self.validate_data[index]
        elif split == "search":
            data = self.search_data[index]
        else:
            raise ValueError(f"Invalid split: {split}")

        data = copy.deepcopy(data)

        sentences, masks, type_ids = [], [], []
        correct_choice = -1
        if split == "train":
            correct_choice = random.randint(0, self.search_negative_samples)
            exclude_indexes = None
            # if len(data["facts"]) > 1:
            #     exclude_indexes = [self.facts.index(f) for f in data["facts"][1:]]

            for i in range(self.search_negative_samples + 1):
                fact = (
                    data["facts"][0]
                    if i == correct_choice
                    else self.negative_fact_gen[data["facts"][0]](exclude_indexes)
                )
                encoded_sentence = self.tokenizer(
                    data["question"],
                    fact,
                    padding="max_length",
                    max_length=self.max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                )
                sentences.append(encoded_sentence.input_ids)
                masks.append(encoded_sentence.attention_mask)
                type_ids.append(encoded_sentence.token_type_ids)
        else:
            if split == "validate":
                correct_choice = self.facts.index(data["facts"][0])
            for fact in self.facts:
                encoded_sentence = self.tokenizer(
                    data["question"],
                    fact,
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
        if split in ("train", "validate"):
            data["answer"] = correct_choice
        return data

    def validate_search(self, batch: BatchEncoding, choice: t.Tensor):
        total = choice.shape[0]
        total_correct = 0
        for i in range(choice.shape[0]):
            fact = self.facts[choice[i].item()]
            sentence = batch["question"][i] + f" ({batch['facts'][i][0]}) "
            total_correct += fact in batch["facts"][i]
            print(
                f"sentence: [{sentence}] \n"
                f"fact: [{fact}] \n"
                f"ref_fact: [{'|'.join(batch['facts'][i])}] \n"
            )

        return {"accuracy": total_correct / total}

    def generate_train_data(self):
        data = []
        unannotated_data = []
        search_data = []
        logging.info("Generating train data")
        with open(self.openbook_qa.train_path, "r") as file:
            for idx, line in enumerate(file):
                entry = json.loads(line)
                fact = entry["fact1"].strip(".").lower()
                preprocessed = {
                    "question": entry["question"]["stem"],
                    "facts": [fact],
                    "id": f"openbook_qa_train_{idx}",
                }
                data.append(preprocessed)
        with open(self.qasc.train_path, "r") as file:
            for idx, line in enumerate(file):
                entry = json.loads(line)
                fact1 = entry["fact1"].strip(".").lower()
                fact2 = entry["fact2"].strip(".").lower()
                preprocessed = {
                    "question": entry["question"]["stem"],
                    "facts": [fact1, fact2],
                    "id": f"qasc_train_{idx}",
                }
                data.append(preprocessed)
        for name, path in (
            ("arc_train_challenge", self.arc.train_challenge_path),
            ("arc_validate_challenge", self.arc.validate_challenge_path),
            ("arc_test_challenge", self.arc.test_challenge_path),
            ("arc_train_easy", self.arc.train_easy_path),
            ("arc_validate_easy", self.arc.validate_easy_path),
            ("arc_test_easy", self.arc.test_easy_path),
            ("qasc_test", self.qasc.test_path),
        ):
            with open(path, "r") as file:
                for idx, line in enumerate(file):
                    entry = json.loads(line)
                    preprocessed = {
                        "question": entry["question"]["stem"],
                        "facts": [],
                        "id": f"{name}_{idx}",
                    }
                    unannotated_data.append(preprocessed)
                    if "arc" in name:
                        search_data.append(preprocessed)
        return data, unannotated_data, search_data

    def generate_validate_data(self):
        data = []
        logging.info("Generating validate data")
        for name, path in (
            ("openbook_qa_validate", self.openbook_qa.validate_path),
            ("openbook_qa_test", self.openbook_qa.test_path),
        ):
            with open(path, "r") as file:
                for idx, line in enumerate(file):
                    entry = json.loads(line)
                    fact = entry["fact1"].strip(".").lower()
                    preprocessed = {
                        "question": entry["question"]["stem"],
                        "facts": [fact],
                        "id": f"{name}_{idx}",
                    }
                    data.append(preprocessed)

        with open(self.qasc.validate_path, "r") as file:
            for idx, line in enumerate(file):
                entry = json.loads(line)
                fact1 = entry["fact1"].strip(".").lower()
                fact2 = entry["fact2"].strip(".").lower()
                preprocessed = {
                    "question": entry["question"]["stem"],
                    "facts": [fact1, fact2],
                    "id": f"qasc_validate_{idx}",
                }
                data.append(preprocessed)
        return data

    def generate_facts(self):
        facts = set()
        for path in (
            self.qasc.train_path,
            self.qasc.validate_path,
        ):
            with open(path, "r") as file:
                for line in file:
                    entry = json.loads(line)
                    facts.add(entry["fact1"].strip(".").lower())
                    # facts.add(entry["fact2"].strip(".").lower())
        for path in (
            self.openbook_qa.facts_path,
            # self.openbook_qa.crowd_source_facts_path,
        ):
            with open(path, "r") as file:
                for line in file:
                    line = line.strip("\n").strip('"').strip(".")
                    if len(line) < 3:
                        continue
                    facts.add(line.lower())
        logging.info(f"Generated {len(facts)} facts")
        return list(facts)

    def save(self, archive_path):
        with open_file_with_create_directories(archive_path, "wb") as file:
            pickle.dump(
                {
                    "facts": self.facts,
                    "train": self.train_data,
                    "unannotated_train": self.unannotated_train_data,
                    "search": self.search_data,
                    "validate": self.validate_data,
                },
                file,
            )

    def annotate_train_data(
        self, batch: BatchEncoding, choice: t.Tensor, confidence: t.Tensor
    ):
        for i in range(choice.shape[0]):
            if confidence[i].item():
                fact = self.facts[choice[i].item()]
                annotated_data, idx = [
                    (d, idx)
                    for idx, d in enumerate(self.unannotated_train_data)
                    if d["id"] == batch["id"][i]
                ][0]
                annotated_data["facts"] = [fact]
                self.train_data.append(self.unannotated_train_data.pop(idx))
        logging.info(f"Annotated {confidence.to(dtype=t.int32).sum()} train_data")

    def save_search_targets(self, choice: t.Tensor):
        target_dict = {}
        if choice.shape[0] != len(self.search_data):
            raise ValueError(
                f"Choice size {choice.shape[0]} does not match "
                f"search target size {len(self.search_data)}"
            )
        for i in range(choice.shape[0]):
            target_dict[self.search_data[i]["id"]] = self.facts[choice[i].item()]
        with open(
            os.path.join(preprocess_cache_dir, f"arc_targets.json"), "w",
        ) as file:
            json.dump(target_dict, file)
