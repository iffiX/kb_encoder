import copy
import nltk
import torch as t
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, PreTrainedTokenizerBase, BatchEncoding
from .base import StaticIterableDataset
from .openbook_qa import OpenBookQADataset


class OpenBookQAWithSearchDataset(OpenBookQADataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        search_tokenizer: PreTrainedTokenizerBase,
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
        super(OpenBookQAWithSearchDataset, self).__init__(
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

    def generator(self, index: int, split: str):
        if split == "validate":
            data = self.validate_data[index]
        elif split == "test":
            data = self.test_data[index]
        else:
            data = None

        if data is not None and len(data["target"]) == 0:
            raise ValueError(f"Set search targets for data {split}-{data['id']} first")
        return super(OpenBookQAWithSearchDataset, self).generator(index, split)

    def search_generator(self, index: int, split: str):
        if split == "train":
            data = self.train_data[index]
        elif split == "validate":
            data = self.validate_data[index]
        elif split == "test":
            data = self.test_data[index]
        else:
            raise ValueError(f"Invalid split: {split}")

        data = copy.deepcopy(data)

        match = self.matcher.match_by_node_embedding(
            data["text_question"],
            target_sentence=data["text_question"],
            seed=self.matcher_seed,
            max_depth=1,
            max_times=1000,
            edge_top_k=10,
        )
        selection = self.matcher.select_paths(match, max_edges=8)
        question = self.matcher.insert_selection(
            data["text_question"], selection, insert_at_end=True,
        )

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

        # question = data["text_question"]
        choices = data["text_choices"]

        encoded_sentence = self.search_tokenizer(
            question,
            choices,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )
        answer = self.search_tokenizer.encode(
            " ".join(self.get_gold_search_target(data["fact"])),
            padding="max_length",
            max_length=32,
            truncation=True,
            return_tensors="pt",
        )
        # Use -100 to focus on training the answer part, rather than pad
        # tokens
        answer.masked_fill_(answer == self.search_tokenizer.pad_token_id, -100)

        data["sentence"] = encoded_sentence.input_ids
        data["mask"] = encoded_sentence.attention_mask
        data["answer"] = answer
        return data

    def validate_search(self, batch: BatchEncoding, tokens: t.Tensor):
        total = tokens.shape[0]
        total_f1 = 0
        for i in range(tokens.shape[0]):
            answer = self.search_tokenizer.decode(tokens[i], skip_special_tokens=True)
            ref_answer_tensor = batch["answer"][i]
            ref_answer_tensor.masked_fill_(
                ref_answer_tensor == -100, self.search_tokenizer.pad_token_id
            )
            ref_answer = self.search_tokenizer.decode(
                ref_answer_tensor, skip_special_tokens=True
            )
            sentence = self.search_tokenizer.decode(
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

    def set_search_target(self, tokens: t.Tensor, split: str, id):
        if split == "validate":
            split_data = self.validate_data
        elif split == "test":
            split_data = self.test_data
        else:
            raise ValueError(f"Invalid split: {split}")
        if tokens.ndim != 1:
            raise ValueError("Token tensor must have a dimension number of 1")
        found_data = [d for d in split_data if d["id"] == id]
        if len(found_data) != 1:
            raise ValueError(f"Id {id} not found in split {split}")
        raw_search_target = self.search_tokenizer.decode(
            tokens, skip_special_tokens=True
        )
        print(f"Search target of [{split}-{id}]: {raw_search_target}")
        search_target = sorted(list(set(nltk.word_tokenize(raw_search_target.lower()))))
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
