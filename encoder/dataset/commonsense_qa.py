import os
import copy
import json
import pickle
import difflib
import logging
import numpy as np
import torch as t
from typing import List, Union
from transformers import AutoTokenizer, PreTrainedTokenizerBase, BatchEncoding
from encoder.dataset.concept_net import ConceptNetMatcher
from encoder.utils.settings import dataset_cache_dir, preprocess_cache_dir
from encoder.utils.file import open_file_with_create_directories, download_to
from .base import StaticIterableDataset


class CommonsenseQADataset:
    TRAIN_URL = "https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl"
    VALIDATE_URL = "https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl"
    TEST_URL = "https://s3.amazonaws.com/commensenseqa/test_rand_split_no_answers.jsonl"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 128,
        generate_length: int = 16,
        use_matcher: bool = False,
        matcher_seed: int = -1,
        include_option_label_in_sentence: bool = False,
        include_option_label_in_answer_and_choices: bool = False,
        use_option_label_as_answer_and_choices: bool = False,
        match_closest_when_no_equal: bool = True,
    ):
        self.tokenizer = tokenizer
        # Word piece is stabler for matching purpose
        self.matcher_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.max_seq_length = max_seq_length
        self.generate_length = generate_length
        self.use_matcher = use_matcher
        self.matcher_seed = matcher_seed
        self.include_option_label_in_sentence = include_option_label_in_sentence
        self.include_option_label_in_answer_and_choices = (
            include_option_label_in_answer_and_choices
        )
        self.use_option_label_as_answer_and_choices = (
            use_option_label_as_answer_and_choices
        )
        self.match_closest_when_no_equal = match_closest_when_no_equal

        if use_matcher:
            self.matcher = ConceptNetMatcher(tokenizer=self.matcher_tokenizer)

        base = os.path.join(dataset_cache_dir, "commonsense_qa")
        train_path = os.path.join(base, "train.jsonl")
        validate_path = os.path.join(base, "validate.jsonl")
        test_path = os.path.join(base, "test.jsonl")
        archive_path = os.path.join(preprocess_cache_dir, "commonsense_qa.data")
        if not os.path.exists(train_path):
            logging.info("Downloading commonsense qa train dataset.")
            download_to(self.TRAIN_URL, train_path)

        if not os.path.exists(validate_path):
            logging.info("Downloading commonsense qa validate dataset.")
            download_to(self.VALIDATE_URL, validate_path)

        if not os.path.exists(test_path):
            logging.info("Downloading commonsense qa test dataset.")
            download_to(self.TEST_URL, test_path)

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
                logging.info(
                    "Configuration mismatch, regenerating commonsense qa dataset."
                )
                self.train_data = self.parse_data(train_path)
                self.validate_data = self.parse_data(validate_path)
                self.test_data = self.parse_data(test_path)
                self.save(archive_path)
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
        if self.use_matcher:
            # data = copy.deepcopy(data)

            # if split == "train":
            #     match = self.matcher.match(
            #         data["text_sentence"],
            #         similarity_exclude=similarity_exclude + data["false_answers_match"],
            #         rank_focus=data["answer_match"] + data["question_match"],
            #         rank_exclude=data["false_answers_match"],
            #         max_times=300,
            #         max_depth=2,
            #         max_edges=4,
            #     )
            # else:
            #     match = self.matcher.match(
            #         data["text_sentence"],
            #         similarity_exclude=similarity_exclude,
            #         rank_focus=data["question_match"],
            #         max_times=300,
            #         max_depth=2,
            #         max_edges=4,
            #     )

            # # BERT tokenizer doesn't honor T5 eos token / Roberta sep token
            # # and choice brackets, fix it.
            # # Have no effect for BERT, ALBERT.
            # new_sentence = new_sentence.replace("< / s >", "</s>")
            # if self.include_option_label_in_sentence:
            #     new_sentence = new_sentence.replace("( a )", "(a)")
            #     new_sentence = new_sentence.replace("( b )", "(b)")
            #     new_sentence = new_sentence.replace("( c )", "(c)")
            #     new_sentence = new_sentence.replace("( d )", "(d)")
            #     new_sentence = new_sentence.replace("( e )", "(e)")
            #
            # # Note that the sentence is now uncased after being processed
            # # by BERT tokenizer
            # data["sentence"] = self.tokenizer(
            #     new_sentence,
            #     padding="max_length",
            #     max_length=self.max_seq_length,
            #     truncation=True,
            #     return_tensors="pt",
            # ).input_ids

            if "matched" not in data:
                data["matched"] = True

                match = self.matcher.match_by_node_embedding(
                    data["text_choices"],
                    target_sentence=data["text_question"],
                    max_times=300,
                    max_depth=3,
                    max_edges=12,
                    seed=self.matcher_seed,
                    discard_edges_if_similarity_below=0.5,
                )

                # match = self.matcher.match_by_token(
                #     data["text_choices"],
                #     target_sentence=data["text_question"],
                #     max_times=300,
                #     max_depth=2,
                #     max_edges=12,
                #     seed=self.matcher_seed,
                #     # rank_focus=data["question_match"],
                # )

                # match = self.matcher.match(
                #     data["text_choices"], target_sentence=data["text_question"]
                # )
                new_choices = self.matcher.insert_match(data["text_choices"], match)
                encoded_sentence = self.tokenizer(
                    data["text_question"],
                    new_choices,
                    padding="max_length",
                    max_length=self.max_seq_length,
                    truncation=True,
                    return_tensors="pt",
                )
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
            if answer == ref_answer:
                correct += 1
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
                else:
                    missing += 1
        print(f"Missing ratio {float(missing) / total}")
        if self.match_closest_when_no_equal:
            print(f"Approximately correct ratio {float(approximately_correct) / total}")
        return {"accuracy": float(correct) / total}

    def generate_test_results_logits(self, logits: t.Tensor, directory: str):
        logits = logits.cpu().numpy()
        labels = np.argmax(logits, axis=1).tolist()
        with open_file_with_create_directories(
            os.path.join(directory, "commonsense_qa.jsonl"), "w"
        ) as file:
            if len(labels) != len(self.test_data):
                raise ValueError(
                    f"Label size {len(labels)} does not match "
                    f"test size {len(self.test_data)}"
                )
            answer_keys = ["A", "B", "C", "D", "E"]
            for label, preprocessed in zip(labels, self.test_data):
                file.write(
                    json.dumps(
                        {"id": preprocessed["id"], "answerKey": answer_keys[label]}
                    )
                )

    def generate_test_results_tokens(self, tokens: t.Tensor, directory: str):
        missing = 0
        with open_file_with_create_directories(
            os.path.join(directory, "commonsense_qa.jsonl"), "w"
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
                        file.write(
                            json.dumps(
                                {"id": preprocessed["id"], "answerKey": answer_keys[i]}
                            )
                        )
                        break
                else:
                    missing += 1
                    print(
                        f"Missing answer, choices: {preprocessed['choices']}, "
                        f"answer: {answer}, using default A as answer."
                    )
                    file.write(json.dumps({"id": preprocessed["id"], "answerKey": "A"}))
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
                        entry["question"]["stem"] + f" {sep} " + sentence_choices
                    )
                else:
                    org_sentence = entry["question"]["stem"] + " " + sentence_choices
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
                    "text_question": entry["question"]["stem"],
                    "text_choices": sentence_choices,
                    # DEPRECATED, prepared for match by token, rank focus and exclude
                    "question_match": [entry["question"]["question_concept"]],
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
                    answer.qrw4rtmasked_fill_(
                        answer == self.tokenizer.pad_token_id, -100
                    )
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

    def __reduce__(self):
        return (
            CommonsenseQADataset,
            (
                self.tokenizer,
                self.max_seq_length,
                self.generate_length,
                self.use_matcher,
            ),
        )
