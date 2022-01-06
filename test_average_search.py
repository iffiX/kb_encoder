import logging
import os
import json
import nltk
from nltk.stem import WordNetLemmatizer
import multiprocessing
from transformers import AutoTokenizer
from encoder.dataset.openbook_qa import OpenBookQADataset


tokenizer = AutoTokenizer.from_pretrained("t5-base")
dataset = OpenBookQADataset(
    tokenizer=tokenizer,
    max_seq_length=512,
    generate_length=16,
    use_matcher=True,
    matcher_mode="embedding",
    matcher_seed=697474,
    matcher_config={
        "max_times": 1000,
        "max_depth": 3,
        "max_edges": 3,
        "edge_top_k": 20,
        "stop_searching_edge_if_similarity_below": 0,
        "discard_edges_if_rank_below": 0,
    },
    include_option_label_in_sentence=True,
    use_option_label_as_answer_and_choices=True,
)


def get_gold_search_target(fact: str, question: str):
    tokens = nltk.word_tokenize(fact.lower())
    wnl = WordNetLemmatizer()
    allowed_tokens = []
    for token, pos in nltk.pos_tag(tokens):
        # if (
        #     pos.startswith("NN")
        #     or pos.startswith("JJ")
        #     or (pos.startswith("VB") and token not in dataset.matcher.VERB_FILTER_SET)
        # ):
        if pos.startswith("NN") or (
            pos.startswith("VB") and token not in dataset.matcher.VERB_FILTER_SET
        ):
            # if pos.startswith("JJ"):
            # if pos.startswith("VB") and token not in dataset.matcher.VERB_FILTER_SET:
            allowed_tokens.append(wnl.lemmatize(token))
    search_target = (
        set(allowed_tokens)
        .difference(
            set([wnl.lemmatize(x) for x in nltk.word_tokenize(question.lower())])
        )
        .difference({"something", "thing", "example"})
    )
    return search_target


if __name__ == "__main__":
    total = 0
    for data in dataset.train_data:
        # total += len(get_gold_search_target(data["fact"], data["text_question"]))
        result = get_gold_search_target(data["fact"], data["text_question"])
        total += len(result)
        print(sorted(list(result)))
    print(f"Train average length: {total / len(dataset.train_data)}")

    total = 0
    for data in dataset.validate_data:
        total += len(get_gold_search_target(data["fact"], data["text_question"]))
    print(f"Validate average length: {total / len(dataset.validate_data)}")

    total = 0
    for data in dataset.test_data:
        total += len(get_gold_search_target(data["fact"], data["text_question"]))
    print(f"Test average length: {total / len(dataset.test_data)}")
