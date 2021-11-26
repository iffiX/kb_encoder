import logging
import os
import multiprocessing
from transformers import AutoTokenizer
from encoder.dataset.openbook_qa import OpenBookQADataset


tokenizer = AutoTokenizer.from_pretrained("t5-base")
dataset = OpenBookQADataset(
    tokenizer=tokenizer,
    max_seq_length=200,
    generate_length=16,
    use_matcher=True,
    matcher_mode="embedding",
    matcher_seed=697474,
    matcher_config={
        "max_times": 1000,
        "max_depth": 2,
        "max_edges": 6,
        "discard_edges_if_similarity_below": 0.45,
        "discard_edges_if_rank_below": 0.7,
    },
)


def test_train(i):
    print(i)
    result = dataset.generator(i, "train")
    fact = result["fact"].lower()
    sentence = tokenizer.decode(result["sentence"][0], skip_special_tokens=True).lower()
    if fact not in sentence:
        print(fact)
        print(sentence)
        return False
    return True


def test_validate(i):
    print(i)
    result = dataset.generator(i, "validate")
    fact = result["fact"].lower().replace(" ,", ",").replace(" '", "'")
    sentence = tokenizer.decode(result["sentence"][0], skip_special_tokens=True).lower()
    if fact not in sentence:
        print(fact)
        print(sentence)
        return False
    return True


def test_test(i):
    print(i)
    result = dataset.generator(i, "test")
    fact = result["fact"].lower()
    sentence = tokenizer.decode(result["sentence"][0], skip_special_tokens=True).lower()
    if fact not in sentence:
        print(fact)
        print(sentence)
        return False
    return True


if __name__ == "__main__":
    # result = dataset.generator(33, "validate")
    # with multiprocessing.Pool(processes=8) as pool:
    #     results = pool.map(test_train, list(range(len(dataset.train_dataset))))
    #     print(f"Acc: {sum(results) / len(dataset.train_dataset)}")

    with multiprocessing.Pool(processes=8) as pool:
        results = pool.map(test_validate, list(range(len(dataset.validate_dataset))))
        print(f"Acc: {sum(results) / len(dataset.validate_dataset)}")

    # with multiprocessing.Pool(processes=8) as pool:
    #     results = pool.map(test_test, list(range(len(dataset.test_dataset))))
    #     print(f"Acc: {sum(results) / len(dataset.test_dataset)}")
