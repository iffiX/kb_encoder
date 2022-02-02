import logging
import os
import nltk
from nltk.stem import WordNetLemmatizer
import json
import multiprocessing
from transformers import AutoTokenizer
from encoder.dataset.openbook_qa import OpenBookQADataset


tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
dataset = OpenBookQADataset(
    tokenizer=tokenizer,
    max_seq_length=150,
    use_matcher=True,
    matcher_mode="embedding",
    matcher_seed=697474,
    matcher_config={
        "question_match_max_times": 1000,
        "question_match_max_depth": 3,
        "question_match_edge_top_k": 10,
        "question_match_source_context_range": 1,
        "question_select_max_edges": 3,
        "question_select_discard_edges_if_rank_below": "auto",
        "choices_match_max_times": 1000,
        "choices_match_max_depth": 1,
        "choices_match_edge_top_k": 10,
        "choices_match_source_context_range": 1,
        "choices_select_max_edges": 2,
        "choices_select_discard_edges_if_rank_below": "auto",
    },
    output_mode="splitted",
)


def test_train(i):
    print(i)
    result = dataset.generator(i, "train")
    fact = result["fact"].lower().replace(" ,", ",").replace(" '", "'")
    sentences = [
        tokenizer.decode(result["sentence"][0, i], skip_special_tokens=True)
        .lower()
        .replace(" - ", "-")
        for i in range(4)
    ]
    if fact not in sentences[0]:
        print(fact)
        for j, sentence in enumerate(sentences):
            print(f"sentence {j}: [{sentence}] \n")
        print("\n")
        return False
    return True


def test_validate(i):
    print(i)
    result = dataset.generator(i, "validate")
    fact = result["fact"].lower().replace(" ,", ",").replace(" '", "'")
    sentences = [
        tokenizer.decode(result["sentence"][0, i], skip_special_tokens=True)
        .lower()
        .replace(" - ", "-")
        for i in range(4)
    ]
    if fact not in sentences[0]:
        print(fact)
        for j, sentence in enumerate(sentences):
            print(f"sentence {j}: [{sentence}] \n")
        print("\n")
        return False
    return True


def test_test(i):
    print(i)
    result = dataset.generator(i, "test")

    fact = result["fact"].lower().replace(" ,", ",").replace(" '", "'")
    sentences = [
        tokenizer.decode(result["sentence"][0, i], skip_special_tokens=True)
        .lower()
        .replace(" - ", "-")
        for i in range(4)
    ]
    if fact not in sentences[0]:
        print(fact)
        for j, sentence in enumerate(sentences):
            print(f"sentence {j}: [{sentence}] \n")
        print("\n")
        return False
    return True


if __name__ == "__main__":
    # dataset.validate_data[0]["fact"] = "resource money"
    result = dataset.generator(1, "validate")
    for i in range(4):
        sentence = tokenizer.decode(
            result["sentence"][0, i], skip_special_tokens=True
        ).lower()
        print(sentence)

    # with multiprocessing.Pool(processes=14) as pool:
    #     results = pool.map(test_train, list(range(len(dataset.train_dataset))))
    #     print(f"Acc: {sum(results) / len(dataset.train_dataset)}")

    # results = map(test_train, list(range(len(dataset.train_dataset))))
    # print(f"Acc: {sum(results) / len(dataset.train_dataset)}")

    # with multiprocessing.Pool(processes=14) as pool:
    #     results = pool.map(test_validate, list(range(len(dataset.validate_dataset))))
    #     print(f"Acc: {sum(results) / len(dataset.validate_dataset)}")
    #
    # with multiprocessing.Pool(processes=14) as pool:
    #     results = pool.map(test_test, list(range(len(dataset.test_dataset))))
    #     print(f"Acc: {sum(results) / len(dataset.test_dataset)}")

    # results = map(test_test, list(range(len(dataset.test_dataset))))
    # print(f"Acc: {sum(results) / len(dataset.test_dataset)}")
