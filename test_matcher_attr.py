import torch as t
from transformers import AutoTokenizer
from encoder.dataset.commonsense_qa import CommonsenseQADataset
from encoder.dataset.concept_net import ConceptNetMatcher

# BASE samples after training for 8 epochs, 1e-4 lr start, 20 steps
# ATTR_DATA_PATH = (
#     "/home/muhan/data/workspace/kb_encoder/data/inspect/attr_map_base_epoch=8.data"
# )

# 20 steps
# ATTR_DATA_PATH = "/home/muhan/data/workspace/kb_encoder/data/inspect/attr_map_large_epoch=8.data"

# 2 steps
ATTR_DATA_PATH = "/home/muhan/data/workspace/kb_encoder/data/inspect/attr_map_base_epoch=8_steps=2.data"

ATTR_THRESHOLD = 0.35

matcher = ConceptNetMatcher(
    tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")
)


def replace_string_span(string, start, end, new_char):
    return string[:start] + new_char * (end - start) + string[end:]


def fix_special_tokens(sentence: str):
    sentence = sentence.replace("< / s >", "</s>")
    sentence = sentence.replace("< pad >", "<pad>")
    sentence = sentence.replace(" <pad> ", "<pad>")
    return sentence


def test_matcher_attr(tokenizer, sub_dataset, attr_sub_map):
    for sample in sub_dataset:
        attr_score = attr_sub_map[sample["id"]]
        attr_mask = (
            sample["mask"] * attr_score.to(sample["mask"].device) >= ATTR_THRESHOLD
        )

        max_seq_length = sample["sentence"].shape[1]

        # 1) First we decode to obtain semi-original sentence S'
        # (with special tokens kept so nothing is lost when we re-encode it)
        # (Space might be incorrect, like [spam] becomes [ spam ]
        #  but that won't affect the sample)
        # 2) Then re-encode the sentence to get batch encoding, which can then
        # be used to match tokens with character spans
        sentence = tokenizer.decode(sample["sentence"][0], skip_special_tokens=False)
        sentence_encoding = tokenizer(
            sentence,
            padding="max_length",
            max_length=max_seq_length,
            truncation=True,
            return_tensors="pt",
        )

        # For matcher:
        # Use the character span to create a binary mask the same length as
        # the recovered string S', use this mask to tell the matcher which
        # part of the sentence can be searched.
        sentence_mask = "-" * len(sentence)
        insert_position_count = 0
        last_attr_high_position = None
        for i in range(max_seq_length):
            if attr_mask[0, i] == 1:
                try:
                    span = sentence_encoding.token_to_chars(0, i)
                except TypeError:
                    continue
                sentence_mask = replace_string_span(
                    sentence_mask, span.start, span.end, "+"
                )

                if last_attr_high_position is None:
                    last_attr_high_position = span.end
                else:
                    last_attr_high_position = max(last_attr_high_position, span.end)
            else:
                if last_attr_high_position is not None:
                    insert_position_count += 1
                    last_attr_high_position = None

        first_part = sentence.find("</s>")

        keep_first_part_mask = replace_string_span(
            sentence_mask, first_part, len(sentence), "-"
        )
        # keep_second_part_mask = replace_string_span(
        #     sentence_mask, 0, first_part + len("</s>"), "-"
        # )
        keep_second_part_mask = replace_string_span(
            "-" * len(sentence),
            first_part + len("</s>"),
            sentence.find("</s>", first_part + len("</s>")),
            "+",
        )
        # match1 = matcher.match_by_node_embedding(
        #     sentence,
        #     target_sentence=sentence,
        #     source_mask=keep_first_part_mask,
        #     target_mask=keep_second_part_mask,
        #     max_times=300,
        #     max_depth=2,
        #     max_edges=min(
        #         16,
        #         insert_position_count * 2,
        #     ),
        #     discard_edges_if_similarity_below=0.45,
        #     seed=1481652,
        # )

        match2 = matcher.match_by_node_embedding(
            sentence,
            target_sentence=sentence,
            source_mask=keep_second_part_mask,
            target_mask=keep_first_part_mask,
            max_times=300,
            max_depth=2,
            max_edges=min(16, insert_position_count * 2,),
            discard_edges_if_similarity_below=0.45,
            seed=1481652,
        )

        # new_sentence = fix_special_tokens(
        #     matcher.insert_match(
        #         sentence, matcher.unify_match([match1, match2])
        #     )
        # )

        new_sentence = fix_special_tokens(matcher.insert_match(sentence, match2))

        ref_match = matcher.match_by_node_embedding(
            sample["text_choices"],
            target_sentence=sample["text_question"],
            max_times=300,
            max_depth=2,
            max_edges=16,
            seed=1481652,
            discard_edges_if_similarity_below=0.45,
        )

        new_choices = matcher.insert_match(sample["text_choices"], ref_match)

        print(
            f"Original sentence: [{sentence}] \n"
            f"New sentence: [{new_sentence}] \n"
            f"Ref sentence: [{sample['text_question'] + ' </s> ' + new_choices}] \n"
        )


if __name__ == "__main__":
    attr_map = t.load(ATTR_DATA_PATH)
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    dataset = CommonsenseQADataset(
        tokenizer=tokenizer,
        max_seq_length=200,
        generate_length=16,
        use_matcher=False,
        regenerate=False,
    )

    test_matcher_attr(tokenizer, dataset.validate_dataset, attr_map["validate"])
