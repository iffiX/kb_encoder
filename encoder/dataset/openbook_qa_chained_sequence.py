import copy
import nltk
from nltk.stem import WordNetLemmatizer
from transformers import PreTrainedTokenizerBase
from .openbook_qa import OpenBookQADataset


class OpenBookQAChainedSequenceDataset(OpenBookQADataset):
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
        super(OpenBookQAChainedSequenceDataset, self).__init__(
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
        )

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

                if len(data["target"]) > 0:
                    # For supporting manually setting search targets
                    allowed_tokens = data["target"]
                else:
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

            new_input = f"{new_question} [CLS] {' [CLS] '.join(new_choices)}"
            encoded_sentence = self.tokenizer(
                new_input,
                padding="max_length",
                max_length=self.max_seq_length * 4,
                truncation=True,
                return_tensors="pt",
            )
            data["sentence"] = encoded_sentence.input_ids
            data["mask"] = encoded_sentence.attention_mask
            data["type_ids"] = encoded_sentence.token_type_ids
        else:
            new_input = (
                f"{data['text_question']} [CLS] {' [CLS] '.join(data['choices'])}"
            )
            encoded_sentence = self.tokenizer(
                new_input,
                padding="max_length",
                max_length=self.max_seq_length * 4,
                truncation=True,
                return_tensors="pt",
            )
            data["sentence"] = encoded_sentence.input_ids
            data["mask"] = encoded_sentence.attention_mask
            data["type_ids"] = encoded_sentence.token_type_ids
        return data
