import os
import json
import logging
import numpy as np
from transformers import AutoTokenizer
from encoder.dataset.concept_net import ConceptNetMatcher
from encoder.utils.settings import dataset_cache_dir
from encoder.utils.file import open_file_with_create_directories, download_to


def parse_data(path):
    data = []
    answer_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    with open_file_with_create_directories(path, "r") as file:
        for line in file:
            entry = json.loads(line)
            question = entry["question"]["stem"]
            choices = "| ".join(ch["text"] for ch in entry["question"]["choices"])
            answer = answer_map[entry["answerKey"]]
            data.append([question, choices, answer])
    return data


def test_overlap(data, matcher):
    correct = 0
    for i in range(len(data)):
        match = matcher.match_by_node_embedding(
            data[i][1],  # choices
            target_sentence=data[i][0],  # question
            max_times=300,
            max_depth=3,
            max_edges=12,
            seed=-1,
            discard_edges_if_similarity_below=0.5,
        )

        # match = self.matcher.match_by_token(
        #     data[1],
        #     target_sentence=data[0],
        #     max_times=300,
        #     max_depth=2,
        #     max_edges=12,
        #     seed=-1,
        #     # rank_focus=data["question_match"],
        # )
        new_choices_text = matcher.insert_match(data[i][1], match)
        new_choices = new_choices_text.split("|")
        knowledge_counts = []
        for choice in new_choices:
            knowledge_count = 0
            if "(" in choice:
                count = choice.count(",")
                knowledge_count = 1 if count == 0 else count
            knowledge_counts.append(knowledge_count)
        ans = np.argmax(knowledge_counts)
        if ans == data[i][2]:
            correct += 1
        else:
            print(
                f"a: {ans}, ra: {data[i][2]}, sent: [{data[i][0]} {new_choices_text}]"
            )
    return correct / len(data) * 100


if __name__ == "__main__":
    TRAIN_URL = "https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl"
    VALIDATE_URL = "https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl"

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    matcher = ConceptNetMatcher(tokenizer=tokenizer)

    base = os.path.join(dataset_cache_dir, "commonsense_qa")
    train_path = os.path.join(base, "train.jsonl")
    validate_path = os.path.join(base, "validate.jsonl")
    if not os.path.exists(train_path):
        logging.info("Downloading commonsense qa train dataset.")
        download_to(TRAIN_URL, train_path)

    if not os.path.exists(validate_path):
        logging.info("Downloading commonsense qa validate dataset.")
        download_to(VALIDATE_URL, validate_path)

    train_data = parse_data(train_path)
    validate_data = parse_data(validate_path)

    print("train:")
    tr = test_overlap(train_data, matcher)
    print("validate:")
    va = test_overlap(validate_data, matcher)
    print(f"Train correct: {tr:.2f}%")
    print(f"Validate correct: {va:.2f}%")
