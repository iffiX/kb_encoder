import os
import random
import string
import torch as t
from typing import List
from datasets import (
    load_dataset,
    DownloadConfig,
)
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from torch.utils.data import get_worker_info
from transformers import AutoTokenizer, PreTrainedTokenizerBase, BatchEncoding
from encoder.dataset.concept_net import ConceptNetMatcher
from encoder.dataset.base import DynamicIterableDataset
from encoder.utils.settings import (
    dataset_cache_dir,
    proxies,
)


class C4KBDataset:
    STOP_WORDS = set(stopwords.words("english") + list(string.punctuation))
    DEFAULT_SIMILARITY_EXCLUDE = [
        "the",
        "a",
        "an",
        "to",
        "of",
        "for",
        "is",
        "are",
    ]

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        matcher_max_times: int = 300,
        matcher_max_depth: int = 2,
        matcher_max_edges: int = 6,
        matcher_similarity_exclude: List[str] = None,
        matcher_seed: int = -1,
        c4_seed: int = 2628807,
        max_seq_length: int = 128,
    ):
        """
        Currently only supports T5 model.
        """
        huggingface_path = str(os.path.join(dataset_cache_dir, "huggingface"))

        self.tokenizer = tokenizer
        self.matcher_max_times = matcher_max_times
        self.matcher_max_depth = matcher_max_depth
        self.matcher_max_edges = matcher_max_edges
        self.matcher_similarity_exclude = (
            matcher_similarity_exclude or self.DEFAULT_SIMILARITY_EXCLUDE
        )
        self.matcher_seed = matcher_seed
        self.c4_seed = c4_seed
        self.max_seq_length = max_seq_length
        self.matcher_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.matcher = ConceptNetMatcher(tokenizer=self.matcher_tokenizer)

        # Load dataset in stream mode since C4 is too large
        self.dataset = load_dataset(
            path="c4",
            name="en",
            cache_dir=huggingface_path,
            download_config=DownloadConfig(proxies=proxies),
            streaming=True,
        )
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
        else:
            worker_id = 0

        self.rnd = random.Random(c4_seed + worker_id)
        self.train_data = iter(self.dataset["train"].shuffle(seed=c4_seed + worker_id))
        self.validate_data = iter(
            self.dataset["validation"].shuffle(seed=c4_seed + worker_id)
        )

    @property
    def train_dataset(self):
        return DynamicIterableDataset(self.generator, ("train",),)

    @property
    def validate_dataset(self):
        return DynamicIterableDataset(self.generator, ("validate",))

    def validate(self, batch: BatchEncoding, tokens: t.Tensor):
        total = tokens.shape[0]
        em_total = 0
        for i in range(tokens.shape[0]):
            answer = self.tokenizer.decode(tokens[i])
            offset = 0
            em_per_sample = 0
            print(
                f"sentence: [{self.tokenizer.decode(batch['input'][i], skip_special_tokens=True)}] \n"
                f"answer: [{answer}] \n"
                f"ref_answer: [{batch['knowledge_list'][i]}]"
            )
            for j in range(
                min(self.matcher_max_edges, len(batch["knowledge_list"][i]))
            ):
                start_marker = f"<extra_id_{j}>"
                start = answer.find(start_marker, offset)
                if start != -1:
                    end = answer.find("<", start + len(start_marker))
                    offset = end
                    knowledge = self.get_word_set(answer[start:end])
                    ref_knowledge = self.get_word_set(batch["knowledge_list"][i][j])
                    shared_words = knowledge.intersection(ref_knowledge)
                    precision = len(shared_words) / len(knowledge)
                    recall = len(shared_words) / len(ref_knowledge)
                    em_per_sample += 2 * precision * recall / (precision + recall)
                else:
                    break
            em_per_sample /= len(batch["knowledge_list"][i])
            em_total += em_per_sample
        return {"EM": em_total / total}

    def get_word_set(self, sentence: str):
        return set(
            i for i in word_tokenize(sentence.lower()) if i not in self.STOP_WORDS
        )

    def generator(self, split: str):
        if split == "train":
            sample = next(self.train_data)
        else:
            sample = next(self.validate_data)
        sentences = sent_tokenize(sample["text"], language="english")

        # Rank sentence by their approximate length distance to the optimal input length
        # T5 usually has 1/4 the number of tokens of total text length
        selected_sentence = sorted(
            sentences, key=lambda x: abs(len(x) / 4 - self.max_seq_length)
        )[0]

        match = self.matcher.match(
            selected_sentence,
            similarity_exclude=self.matcher_similarity_exclude,
            max_times=self.matcher_max_times,
            max_depth=self.matcher_max_depth,
            max_edges=self.matcher_max_edges,
        )

        input = ""
        predict_target = ""

        sentence_remain, s_match = self.matcher.match_to_string(
            selected_sentence, match
        )
        knowledge_combined_list = []
        for i, (sentence_piece, knowledge_sequences) in enumerate(s_match):
            knowledge_combined = ", ".join(knowledge_sequences)
            input += sentence_piece + f" <extra_id_{i}> "
            predict_target += f"<extra_id_{i}> {knowledge_combined} "
            knowledge_combined_list.append(knowledge_combined)
        input += sentence_remain

        encoded_input = self.tokenizer(
            input,
            padding="max_length",
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt",
        )
        encoded_predict_target = self.tokenizer.encode(
            predict_target,
            padding="max_length",
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt",
        )
        # Use -100 to focus on training the answer part, rather than pad
        # tokens
        encoded_predict_target.masked_fill_(
            encoded_predict_target == self.tokenizer.pad_token_id, -100
        )
        preprocessed = {
            "input": encoded_input.input_ids,
            "mask": encoded_input.attention_mask,
            "predict_target": encoded_predict_target,
            "knowledge_list": knowledge_combined_list,
            "id": hash(sample["text"]),
        }
        return preprocessed

    def __reduce__(self):
        return (
            C4KBDataset,
            (
                self.tokenizer,
                self.matcher_max_times,
                self.matcher_max_depth,
                self.matcher_max_edges,
                self.matcher_seed,
                self.c4_seed,
                self.max_seq_length,
            ),
        )
