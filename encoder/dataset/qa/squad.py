import os
import logging
import pickle
import torch as t
from .base import QADataset, StaticMapDataset
from typing import List, Dict, Any
from transformers import PreTrainedTokenizerBase, BatchEncoding
from datasets import load_dataset, load_metric, DownloadConfig
from encoder.utils.file import open_file_with_create_directories
from encoder.utils.settings import (
    dataset_cache_dir,
    metrics_cache_dir,
    preprocess_cache_dir,
    proxies,
)


class SQuADDataset(QADataset):
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, dataset_path: str = "squad",
    ):
        huggingface_path = str(os.path.join(dataset_cache_dir, "huggingface"))

        self.tokenizer = tokenizer
        self.dataset_path = dataset_path

        self.dataset = load_dataset(
            path=dataset_path,
            cache_dir=huggingface_path,
            download_config=DownloadConfig(proxies=proxies),
        )

        # squad v2 works for squad and squad v2 and any custom squad datasets
        self.metric = load_metric(
            "squad_v2",
            cache_dir=metrics_cache_dir,
            download_config=DownloadConfig(proxies=proxies),
        )

        self._train = None
        self._validate = None
        self._train_raw = None
        self._validate_raw = None

        if os.path.exists(
            os.path.join(preprocess_cache_dir, f"squad_{dataset_path}.cache")
        ):
            logging.info(
                f"Found states cache for Squad[{dataset_path}], skipping generation."
            )
            self.restore(
                os.path.join(preprocess_cache_dir, f"squad_{dataset_path}.cache")
            )
        else:
            logging.info(
                f"States cache for Squad[{dataset_path}] not found, generating."
            )
            self._train, self._train_raw = self.preprocess(split="train")
            self._validate, self._validate_raw = self.preprocess(split="validation")
            self.save(os.path.join(preprocess_cache_dir, f"squad_{dataset_path}.cache"))

    @property
    def train_dataset(self):
        return StaticMapDataset(self._train)

    @property
    def validate_dataset(self):
        return StaticMapDataset(self._validate)

    def validate(
        self,
        batch: BatchEncoding,
        start_logits: t.FloatTensor,
        end_logits: t.FloatTensor,
    ):
        predictions = []
        references = []
        for index, input_ids, start_l, end_l in zip(
            batch["sample-index"], batch["input_ids"], start_logits, end_logits
        ):
            raw = self._validate_raw[index]
            # get the most likely beginning of answer with the argmax of the score
            answer_start = t.argmax(start_l)
            answer_end = t.argmax(end_l) + 1
            answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
            )
            for i in range(len(raw["answers"]["answer_start"])):
                predictions.append(
                    {
                        "prediction_text": answer,
                        "id": raw["id"],
                        "no_answer_probability": 0.0,
                    }
                )
                references.append(
                    {
                        "id": raw["id"],
                        "answers": [
                            {"text": text, "answer_start": answer_start}
                            for text, answer_start in zip(
                                raw["answers"]["text"], raw["answers"]["answer_start"]
                            )
                        ],
                    }
                )

        return self.metric.compute(predictions=predictions, references=references)

    def preprocess(self, split="train"):
        logging.info(f"SQuADDataset begin pre-processing split {split}")
        # flatten answers in the dataset
        contexts = []
        questions = []
        answers = []
        indexes = []
        raw = []
        for idx, item in enumerate(self.dataset[split]):
            raw.append(item)
            for text, answer_start in zip(
                item["answers"]["text"], item["answers"]["answer_start"]
            ):
                indexes.append(idx)
                contexts.append(item["context"])
                questions.append(item["question"])
                answers.append({"text": text, "answer_start": answer_start})

        # add end idx to answers
        self.add_end_idx(answers, contexts)

        # compute encoding [cls] context [sep] answer [sep]
        encodings = self.tokenizer(contexts, questions, truncation=True, padding=True)

        # update token start / end positions
        self.add_token_positions(encodings, self.tokenizer, answers)

        # update index
        encodings.update({"sample-index": indexes})

        logging.info(f"SQuADDataset finished pre-processing split {split}")
        return encodings, raw

    @staticmethod
    def add_end_idx(answers: List[Dict[str, Any]], contexts: List[str]):
        for answer, context in zip(answers, contexts):
            gold_text = answer["text"]
            start_idx = answer["answer_start"]
            end_idx = start_idx + len(gold_text)

            # sometimes squad answers are off by a character or two – fix this
            if context[start_idx:end_idx] == gold_text:
                answer["answer_end"] = end_idx
            elif context[start_idx - 1 : end_idx - 1] == gold_text:
                # When the gold label is off by one character
                answer["answer_start"] = start_idx - 1
                answer["answer_end"] = end_idx - 1
            elif context[start_idx - 2 : end_idx - 2] == gold_text:
                # When the gold label is off by two characters
                answer["answer_start"] = start_idx - 2
                answer["answer_end"] = end_idx - 2

    @staticmethod
    def add_token_positions(
        encodings: BatchEncoding,
        tokenizer: PreTrainedTokenizerBase,
        answers: List[Dict[str, Any]],
    ):
        start_positions = []
        end_positions = []
        for i, answer in enumerate(answers):
            start_positions.append(encodings.char_to_token(i, answer["answer_start"]))
            end_positions.append(encodings.char_to_token(i, answer["answer_end"] - 1))

            # if start position is None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = tokenizer.model_max_length

        encodings.update(
            {"start_positions": start_positions, "end_positions": end_positions}
        )

    def restore(self, path):
        save = pickle.load(open(path, "rb"))
        for k, v in save.items():
            setattr(self, k, v)

    def save(self, path):
        with open_file_with_create_directories(path, "wb") as file:
            pickle.dump(
                {
                    "_train": self._train,
                    "_validate": self._validate,
                    "_train_raw": self._train_raw,
                    "_validate_raw": self._validate_raw,
                },
                file,
                protocol=4,
            )
