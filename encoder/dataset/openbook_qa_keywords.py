import copy
import nltk
from nltk.stem import WordNetLemmatizer
from transformers import PreTrainedTokenizerBase, BatchEncoding
from .base import StaticIterableDataset
from .openbook_qa import OpenBookQADataset


class OpenBookQAKeywordsDataset(OpenBookQADataset):
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
        super(OpenBookQAKeywordsDataset, self).__init__(
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
            len(self.original_train_data), self.search_generator, ("train",),
        )

    @property
    def validate_search_dataset(self):
        return StaticIterableDataset(
            len(self.original_validate_data), self.search_generator, ("validate",)
        )

    @property
    def test_search_dataset(self):
        return StaticIterableDataset(
            len(self.original_test_data), self.search_generator, ("test",)
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
        return super(OpenBookQAKeywordsDataset, self).generator(index, split)

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

        # match = self.matcher.match_by_node_embedding(
        #     data["text_question"],
        #     target_sentence=data["text_question"],
        #     seed=self.matcher_seed,
        #     max_depth=1,
        #     max_times=1000,
        #     edge_top_k=10,
        # )
        # selection = self.matcher.select_paths(match, max_edges=8)
        # question = self.matcher.insert_selection(
        #     data["text_question"], selection, insert_at_end=True,
        # )

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

        question = data["text_question"]
        choices = data["text_choices"]

        encoded_sentence = self.search_tokenizer(
            question,
            choices,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        data["sentence"] = encoded_sentence.input_ids
        data["mask"] = encoded_sentence.attention_mask
        data["type_ids"] = encoded_sentence.token_type_ids
        data["answer"] = self.get_gold_search_target(data["fact"])
        return data

    def validate_search(self, batch: BatchEncoding, keywords_list):
        total = len(keywords_list)
        total_f1 = 0
        for i in range(len(keywords_list)):
            answer = keywords_list[i]
            ref_answer = batch["answer"][i]

            sentence = self.search_tokenizer.decode(
                batch["sentence"][i], skip_special_tokens=True,
            )

            keywords = set(answer)
            ref_keywords = set(ref_answer)
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

    def set_search_target(self, keywords, split: str, id):
        if split == "validate":
            split_data = self.validate_data
        elif split == "test":
            split_data = self.test_data
        else:
            raise ValueError(f"Invalid split: {split}")

        found_data = [d for d in split_data if d["id"] == id]
        if len(found_data) != 1:
            raise ValueError(f"Id {id} not found in split {split}")

        print(f"Search target of [{split}-{id}]: {keywords}")
        search_target = keywords
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
