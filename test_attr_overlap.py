import torch as t
from transformers import AutoTokenizer
from encoder.dataset.commonsense_qa import CommonsenseQADataset
from encoder.dataset.concept_net import ConceptNetMatcher

# BASE results after training for 8 epochs, 1e-4 lr start, 20 steps
# ATTR_DATA_PATH = (
#     "/home/muhan/data/workspace/kb_encoder/data/inspect/attr_map_base_epoch=8.data"
# )
#
# VALIDATION_RESULT_PATH = "/home/muhan/data/workspace/kb_encoder/data/inspect/commensense_qa_val_answers_base_epoch=8.data"

# 20 steps
# ATTR_DATA_PATH = "/home/muhan/data/workspace/kb_encoder/data/inspect/attr_map_large_epoch=8.data"
#
# VALIDATION_RESULT_PATH = (
#     "/home/muhan/data/workspace/kb_encoder/data/inspect/commensense_qa_val_answers_large_epoch=8.data"
# )

# 2 steps
ATTR_DATA_PATH = "/home/muhan/data/workspace/kb_encoder/data/inspect/attr_map_base_epoch=8_steps=2.data"

VALIDATION_RESULT_PATH = "/home/muhan/data/workspace/kb_encoder/data/inspect/commensense_qa_val_answers_base_epoch=8_steps=2.data"

ATTR_THRESHOLD = 0.35

matcher = ConceptNetMatcher(
    tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")
)


def get_sentence_with_score(tokenizer, sentence, attr_score):
    decoded_sentence = [tokenizer.decode(token) for token in sentence]
    sentence_scores = attr_score.tolist()
    result = []
    for token, score in zip(decoded_sentence, sentence_scores):
        result.append(f"{token}:{score:.3f}")
    return " ".join(result)


def test_attr_overlap(tokenizer, key, sub_dataset, attr_sub_map, val_data=None):
    valid = 0
    missing = 0
    below_threshold = 0
    val_incorrect_same = 0
    for sample in sub_dataset:
        sentence = sample["sentence"][0]
        id = sample["id"]
        if id not in attr_sub_map:
            print(f"Missing id {id}")
            continue
        attr = attr_sub_map[id]
        org_sentence = tokenizer.decode(sentence)
        encoding = tokenizer(org_sentence)
        match_words = sample[key][0].split(" ")
        best_attr = -1
        for word in match_words:
            char_pos = org_sentence.lower().find(word.lower())
            if char_pos == -1:
                missing += 1
                continue
            token_pos = set(
                encoding.char_to_token(p) for p in range(char_pos, char_pos + len(word))
            ).difference({None})
            if len(token_pos) > 0:
                attr_values = {tp: float(attr[tp]) for tp in token_pos}
                best_attr = max(best_attr, t.max(t.tensor(list(attr_values.values()))))
            else:
                missing += 1
                continue
        if best_attr < ATTR_THRESHOLD:
            below_threshold += 1
            if val_data is not None and not val_data[id]:
                val_incorrect_same += 1
            # print(
            #     f"No match, best attr {best_attr}, {key} [{' '.join(match_words)}], "
            #     f"attrs [{get_sentence_with_score(tokenizer, sentence, attr)}]"
            # )
            # match = matcher.match_by_node_embedding(
            #     sample["text_choices"],
            #     target_sentence=sample["text_question"],
            #     max_times=300,
            #     max_depth=2,
            #     max_edges=16,
            #     seed=600,
            #     discard_edges_if_similarity_below=0.45,
            # )
            # print(org_sentence)
            # print(matcher.insert_match(sample["text_choices"], match))
        else:
            valid += 1
    print(f"Missing {missing / len(sub_dataset)}")
    print(f"Accuracy of {key}: {valid / len(sub_dataset)}")
    if val_data is not None:
        print(
            f"Below threshold and wrong prediction overlap: {val_incorrect_same / below_threshold}"
        )


if __name__ == "__main__":
    attr_map = t.load(ATTR_DATA_PATH)
    val_data = t.load(VALIDATION_RESULT_PATH)
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    dataset = CommonsenseQADataset(
        tokenizer=tokenizer,
        max_seq_length=200,
        generate_length=16,
        use_matcher=False,
        regenerate=False,
    )
    print("Train:")
    test_attr_overlap(
        tokenizer, "question_match", dataset.train_dataset, attr_map["train"]
    )
    test_attr_overlap(
        tokenizer, "answer_match", dataset.train_dataset, attr_map["train"]
    )
    print("Validate:")
    test_attr_overlap(
        tokenizer,
        "question_match",
        dataset.validate_dataset,
        attr_map["validate"],
        val_data=val_data,
    )
    test_attr_overlap(
        tokenizer, "answer_match", dataset.validate_dataset, attr_map["validate"]
    )
    # print("Test:")
    # test_attr_overlap(
    #     tokenizer, "question_match", dataset.validate_dataset, attr_map["test"]
    # )
