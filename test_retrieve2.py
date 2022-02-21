import logging
import os
import nltk
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer
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
        "question_match_max_depth": 1,
        "question_match_edge_top_k": 10,
        "question_match_source_context_range": 1,
        "question_select_max_edges": 0,
        "question_select_discard_edges_if_rank_below": 0,
        "choices_match_max_times": 1000,
        "choices_match_max_depth": 1,
        "choices_match_edge_top_k": 10,
        "choices_match_source_context_range": 0,
        "choices_select_max_edges": 6,
        "choices_select_discard_edges_if_rank_below": "auto",
    },
    output_mode="splitted",
)

if __name__ == "__main__":
    result = dataset.generator(681, "validate")
    for i in range(5):
        sentence = tokenizer.decode(
            result["sentence"][0, i], skip_special_tokens=True
        ).lower()
        print(sentence)
        print()
