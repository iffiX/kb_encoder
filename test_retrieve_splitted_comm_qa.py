import logging
import os
import nltk
from nltk.stem import WordNetLemmatizer
import json
import multiprocessing
from transformers import AutoTokenizer
from encoder.dataset.commonsense_qa import CommonsenseQADataset
from encoder.utils.settings import preprocess_cache_dir

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
dataset = CommonsenseQADataset(
    tokenizer=tokenizer,
    max_seq_length=200,
    use_matcher=True,
    matcher_mode="embedding",
    matcher_seed=697474,
    matcher_config={
        "question_match_max_times": 1000,
        "question_match_max_depth": 2,
        "question_match_edge_top_k": 10,
        "question_match_source_context_range": 0,
        "question_select_max_edges": 4,
        "question_select_discard_edges_if_rank_below": 0.4,
        "choices_match_max_times": 1000,
        "choices_match_max_depth": 1,
        "choices_match_edge_top_k": 10,
        "choices_match_source_context_range": 0,
        "choices_select_max_edges": 4,
        "choices_select_discard_edges_if_rank_below": 0,
    },
    output_mode="splitted",
)


def load_targets():
    targets = []
    for file_name in (
        "commonsense_qa_val_targets.json",
        "commonsense_qa_test_targets.json",
    ):
        with open(os.path.join(preprocess_cache_dir, file_name), "r") as file:
            targets.append(json.load(file))
    return targets


if __name__ == "__main__":
    targets = load_targets()
    for split, target_dict in (("validate", targets[0]), ("test", targets[1])):
        for id, target in target_dict.items():
            dataset.set_search_target(target, split, id)

    # print(
    #     dataset.question_matcher.find_closest_concept(
    #         "rejection", ["working hard", "frustration", "defeat", "stress"],
    #     )
    # )
    result = dataset.generator(1193, "validate")
    for i in range(5):
        sentence = tokenizer.decode(
            result["sentence"][0, i], skip_special_tokens=True
        ).lower()
        print(sentence)
