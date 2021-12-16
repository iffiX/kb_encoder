import logging
import os
import nltk
from nltk.stem import WordNetLemmatizer
import json
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
        "edge_beam_width": 20,
        "stop_searching_edge_if_similarity_below": 0,
        "discard_edges_if_rank_below": 0.3,
    },
    include_option_label_in_sentence=True,
    use_option_label_as_answer_and_choices=True,
)

ranking = {}
with open(
    "/home/iffi/Projects/OpenBookQA/data/OpenBookQA-V1-Sep2018/"
    "Data/Main/ranked_knowledge/openbook/full.jsonl.ranking.json",
    "r",
) as file:
    for line in file:
        object = json.loads(line)
        ranking[object["id"]] = [x[0] for x in object["ext_fact_global_ids"]]

knowledge = []
with open(
    "/home/iffi/Projects/OpenBookQA/data/OpenBookQA-V1-Sep2018/"
    "Data/Main/ranked_knowledge/openbook/knowledge.json",
    "r",
) as file:
    for line in file:
        knowledge.append(json.loads(line)["SCIENCE-FACT"])


def test_train(i):
    print(i)
    result = dataset.generator(i, "train")
    fact = result["fact"].lower().replace(" ,", ",").replace(" '", "'")
    sentence = (
        tokenizer.decode(result["sentence"][0], skip_special_tokens=True)
        .lower()
        .replace(" - ", "-")
    )
    if fact not in sentence:
        print(fact)
        print(sentence)
        return False
    return True


def test_validate(i):
    print(i)
    result = dataset.generator(i, "validate")
    fact = result["fact"].lower().replace(" ,", ",").replace(" '", "'")
    sentence = (
        tokenizer.decode(result["sentence"][0], skip_special_tokens=True)
        .lower()
        .replace(" - ", "-")
    )
    if fact not in sentence:
        print(fact)
        print(sentence)
    if fact not in sentence:
        return False
    return True


def test_test(i):
    print(i)
    result = dataset.generator(i, "test")

    fact = result["fact"].lower().replace(" ,", ",").replace(" '", "'")
    sentence = (
        tokenizer.decode(result["sentence"][0], skip_special_tokens=True)
        .lower()
        .replace(" - ", "-")
    )
    if fact not in sentence:
        print(fact)
        print(sentence)
    if fact not in sentence:
        return False
    return True

    # fact = result["fact"]
    # r = any(fact in xx for xx in result["x"])
    # return r

    # fact = result["fact"]
    # r = any(fact in knowledge[xx] for xx in ranking[result["id"] + "__q"][:10])
    # if not r:
    #     print(f"fact: {fact}")
    #     print([knowledge[xx] for xx in ranking[result["id"] + "__q"][:10]])
    #     print()
    # return r


if __name__ == "__main__":
    dataset.validate_data[496]["fact"] = "gravity"
    result = dataset.generator(450, "validate")
    sentence = tokenizer.decode(result["sentence"][0], skip_special_tokens=True).lower()
    print(sentence)
    # if fact not in sentence:
    #     print(fact)
    #     print(sentence)

    # with multiprocessing.Pool(processes=14) as pool:
    #     results = pool.map(test_train, list(range(len(dataset.train_dataset))))
    #     print(f"Acc: {sum(results) / len(dataset.train_dataset)}")

    # results = map(test_train, list(range(len(dataset.train_dataset))))
    # print(f"Acc: {sum(results) / len(dataset.train_dataset)}")

    # with multiprocessing.Pool(processes=14) as pool:
    #     results = pool.map(test_validate, list(range(len(dataset.validate_dataset))))
    #     print(f"Acc: {sum(results) / len(dataset.validate_dataset)}")

    # with multiprocessing.Pool(processes=14) as pool:
    #     results = pool.map(test_test, list(range(len(dataset.test_dataset))))
    #     print(f"Acc: {sum(results) / len(dataset.test_dataset)}")

    # results = map(test_test, list(range(len(dataset.test_dataset))))
    # print(f"Acc: {sum(results) / len(dataset.test_dataset)}")