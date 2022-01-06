import logging
import os
import nltk
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer
from encoder.dataset.openbook_qa_with_search import OpenBookQAWithSearchDataset


tokenizer = AutoTokenizer.from_pretrained("t5-base")
dataset = OpenBookQAWithSearchDataset(
    tokenizer=tokenizer,
    max_seq_length=300,
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
    },
    include_option_label_in_sentence=True,
    use_option_label_as_answer_and_choices=True,
)

# if __name__ == "__main__":
#     for name, sub_data in (("val", dataset.validate_data), ("test", dataset.test_data)):
#         question_length = sum(
#             len(nltk.word_tokenize(data["text_question"])) for data in sub_data
#         ) / len(sub_data)
#         print(f"{name}: {question_length}")


# if __name__ == "__main__":
#     for name, sub_data in (("val", dataset.validate_data), ("test", dataset.test_data)):
#         choice_length = sum(
#             len(nltk.word_tokenize(data["text_choices"])) for data in sub_data
#         ) / len(sub_data)
#         print(f"{name}: {choice_length}")


if __name__ == "__main__":
    for name, sub_data in (("val", dataset.validate_data), ("test", dataset.test_data)):
        choice_length = sum(
            len(nltk.word_tokenize(data["text_choices"])) for data in sub_data
        ) / len(sub_data)
        print(f"{name}: {choice_length}")
