import logging
import os
import nltk
from nltk.stem import WordNetLemmatizer
import json
import multiprocessing
from transformers import AutoTokenizer
from encoder.utils.settings import preprocess_cache_dir
from encoder.dataset.commonsense_qa import CommonsenseQADataset


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
        "question_select_max_edges": 6,
        "question_select_discard_edges_if_rank_below": 0.6,
        "choices_match_max_times": 1000,
        "choices_match_max_depth": 1,
        "choices_match_edge_top_k": 10,
        "choices_match_source_context_range": 0,
        "choices_select_max_edges": 6,
        "choices_select_discard_edges_if_rank_below": 0,
    },
    output_mode="splitted",
)


if __name__ == "__main__":
    with open(
        os.path.join(preprocess_cache_dir, "commonsense_qa_val_targets.json"), "r"
    ) as file:
        val_targets = json.load(file)
    count = 0
    for sample in dataset.validate_data:
        if sample["choices"][sample["label"]] in val_targets[sample["id"]]:
            count += 1
        else:
            print(
                f"{sample['text_question']} / {sample['choices'][sample['label']]} / {val_targets[sample['id']]}"
            )
    print(count / len(dataset.validate_data))
