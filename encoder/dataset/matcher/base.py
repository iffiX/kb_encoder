import os
import nltk
import logging
from typing import List, Dict, Tuple
from transformers import PreTrainedTokenizerBase
from encoder.dataset.matcher import KnowledgeMatcher
from encoder.utils.settings import preprocess_cache_dir


class BaseMatcher:
    ASSERTION_URL = (
        "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/"
        "conceptnet-assertions-5.7.0.csv.gz"
    )
    NUMBERBATCH_URL = (
        "https://conceptnet.s3.amazonaws.com/downloads/2019/"
        "numberbatch/numberbatch-en-19.08.txt.gz"
    )
    VERB_FILTER_SET = {
        "do",
        "did",
        "does",
        "done",
        "have",
        "having",
        "has",
        "had",
        "be",
        "am",
        "is",
        "are",
        "being",
        "was",
        "were",
    }

    def __init__(self, tokenizer: PreTrainedTokenizerBase, matcher: KnowledgeMatcher):
        self.tokenizer = tokenizer
        self.matcher = matcher

    def match_composite_nodes(self, target_sentence, target_mask: str = ""):
        target_tokens, _target_mask = self.tokenize_and_mask(
            target_sentence, target_mask
        )
        return self.matcher.match_composite_nodes(target_tokens, _target_mask)

    def match_by_node_embedding(
        self,
        source_sentence: str,
        target_sentence: str = "",
        source_mask: str = "",
        target_mask: str = "",
        max_times: int = 1000,
        max_depth: int = 3,
        seed: int = -1,
        edge_beam_width: int = -1,
        trim_path: bool = True,
        stop_searching_edge_if_similarity_below: float = 0,
    ):
        """
        Returns:
            A match result object that can be unified or be used to select paths
        """
        source_tokens, _source_mask = self.tokenize_and_mask(
            source_sentence, source_mask
        )
        if len(target_sentence) == 0:
            target_tokens, _target_mask = source_tokens, _source_mask
        else:
            target_tokens, _target_mask = self.tokenize_and_mask(
                target_sentence, target_mask
            )

        result = self.matcher.match_by_node_embedding(
            source_sentence=source_tokens,
            target_sentence=target_tokens,
            source_mask=_source_mask,
            target_mask=_target_mask,
            max_times=max_times,
            max_depth=max_depth,
            seed=seed,
            edge_beam_width=edge_beam_width,
            trim_path=trim_path,
            stop_searching_edge_if_similarity_below=stop_searching_edge_if_similarity_below,
        )
        return result

    def unify_match(self, matches: list):
        """
        Unify several match result into one.
        """
        return self.matcher.join_match_results(matches)

    def select_paths(
        self, match, max_edges: int = 10, discard_edges_if_rank_below: float = 0
    ) -> Dict[int, Tuple[int, List[List[int]], List[float]]]:
        return self.matcher.select_paths(match, max_edges, discard_edges_if_rank_below)

    def selection_to_list_of_strings(
        self, selection: Dict[int, Tuple[int, List[List[int]], List[float]]],
    ) -> List[str]:
        """
        Returns List of knowledge sequences
        """
        knowledge_tokens = list(
            v for _, (__, v, ___) in selection.items()
        )  # type: List[List[List[int]]]
        knowledge = []
        for kt_list in knowledge_tokens:
            for kt in kt_list:
                knowledge.append(self.tokenizer.decode(kt))
        return knowledge

    def insert_selection(
        self,
        sentence: str,
        selection: Dict[int, Tuple[int, List[List[int]], List[float]]],
        begin: str = "(",
        sep: str = ",",
        end: str = ")",
        insert_at_end: bool = False,
        include_weights: bool = False,
    ) -> str:
        """
        If the triples are directly inserted at the end, accuracy
        reduced by more than 50%
        """
        sentence_tokens, _ = self.tokenize_and_mask(sentence)
        if len(selection) == 0:
            return self.tokenizer.decode(sentence_tokens)
        begin_tokens = self.tokenizer.encode(begin, add_special_tokens=False)
        end_tokens = self.tokenizer.encode(end, add_special_tokens=False)
        sep_tokens = self.tokenizer.encode(sep, add_special_tokens=False)
        new_matches = {}
        for pos, (_, edges, weights) in selection.items():
            new_edges = []
            for i, (edge, weight) in enumerate(zip(edges, weights)):
                if not insert_at_end and i == 0:
                    new_edges += begin_tokens
                if include_weights:
                    new_edges += edge + self.tokenizer.encode(
                        f"{weight:.1f}", add_special_tokens=False
                    )
                else:
                    new_edges += edge
                if not insert_at_end and i == len(edges) - 1:
                    new_edges += end_tokens
                else:
                    new_edges += sep_tokens
            new_matches[pos] = new_edges
        sorted_selection = list(
            (k, v) for k, v in new_matches.items()
        )  # type: List[Tuple[int, List[int]]]
        sorted_selection = sorted(sorted_selection, key=lambda x: x[0])
        if insert_at_end:
            sentence_tokens += begin_tokens
            for ss in sorted_selection:
                sentence_tokens = sentence_tokens + ss[1]
            sentence_tokens += end_tokens
        else:
            offset = 0
            for ss in sorted_selection:
                pos = ss[0] + offset
                sentence_tokens = sentence_tokens[:pos] + ss[1] + sentence_tokens[pos:]
                offset += len(ss[1])
        return self.tokenizer.decode(sentence_tokens)

    def tokenize_and_mask(self, sentence: str, sentence_mask: str = ""):
        """
        Args:
            sentence: A sentence to be tagged by Part of Speech (POS)
            sentence_mask: A string same length as sentence, comprised of "+" and "-", where
                "+" indicates allowing the word overlapped with that position to be matched
                and "-" indicates the position is disallowed.
        Returns:
            A list of token ids.
            A list of POS mask.
        """
        use_mask = False
        if len(sentence_mask) != 0:
            mask_characters = set(sentence_mask)
            if not mask_characters.issubset({"+", "-"}):
                raise ValueError(
                    f"Sentence mask should only be comprised of '+' "
                    f"and '-',"
                    f" but got {sentence_mask}"
                )
            elif len(sentence) != len(sentence_mask):
                raise ValueError(
                    f"Sentence mask should be the same length as the " f"sentence."
                )
            use_mask = True

        tokens = nltk.word_tokenize(sentence)

        offset = 0
        masks = []
        ids = []
        allowed_tokens = []
        for token, pos in self.safe_pos_tag(tokens):
            token_position = sentence.find(token, offset)
            offset = token_position + len(token)

            # Relaxed matching, If any part is not masked, allow searchin for that part
            ids.append(self.tokenizer.encode(token, add_special_tokens=False))
            if (not use_mask or "+" in set(sentence_mask[token_position:offset])) and (
                pos.startswith("NN")
                or pos.startswith("JJ")
                or pos.startswith("RB")
                or (pos.startswith("VB") and token.lower() not in self.VERB_FILTER_SET)
                or pos.startswith("CD")
            ):
                allowed_tokens.append(token)
                # noun, adjective, adverb, verb
                masks.append([1] * len(ids[-1]))
            else:
                masks.append([0] * len(ids[-1]))
        logging.debug(
            f"Tokens allowed for matching: {allowed_tokens} from sentence {sentence}"
        )
        return [i for iid in ids for i in iid], [m for mask in masks for m in mask]

    @staticmethod
    def safe_pos_tag(tokens):
        # In case the input sentence is a decoded sentence with special characters
        # used in tokenizers
        cleaned_tokens_with_index = [
            (i, token)
            for i, token in enumerate(tokens)
            if len(set(token).intersection({"<", ">", "/", "]", "["})) == 0
        ]
        pos_result = nltk.pos_tag([ct[1] for ct in cleaned_tokens_with_index])
        result = [[token, ","] for token in tokens]
        for (i, _), (token, pos) in zip(cleaned_tokens_with_index, pos_result):
            _, individual_pos = nltk.pos_tag([token])[0]
            if individual_pos.startswith("NN") or individual_pos.startswith("VB"):
                result[i][1] = individual_pos
            else:
                result[i][1] = pos
        return [tuple(r) for r in result]
