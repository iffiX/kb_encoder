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

    def __init__(self, tokenizer: PreTrainedTokenizerBase, matcher: KnowledgeMatcher):
        self.tokenizer = tokenizer
        self.matcher = matcher

    def match_by_node(
        self,
        source_sentence: str,
        target_sentence: str = "",
        source_mask: str = "",
        target_mask: str = "",
        max_times: int = 1000,
        max_depth: int = 3,
        max_edges: int = 3,
        seed: int = -1,
        edge_beam_width: int = -1,
        trim_path: bool = True,
        discard_edges_if_similarity_below: float = 0,
        discard_edges_if_rank_below: float = 0,
    ) -> Dict[int, Tuple[int, List[List[int]], List[float]]]:
        """
        Returns:
            Dict[end position, Tuple[start position, vector of knowledge sequences, vector of weights]]
            end position is the index of the word behind the match
            start position is the index of the first word in the match
        """
        if not self.matcher.kb.is_landmark_inited():
            landmark_path = str(
                os.path.join(preprocess_cache_dir, "conceptnet-landmark.cache")
            )
            self.matcher.kb.init_landmarks(
                seed_num=100,
                landmark_num=100,
                seed=41379823,
                landmark_path=landmark_path,
            )

        source_tokens, _source_mask = self.tokenize_and_mask(
            source_sentence, source_mask
        )
        if len(target_sentence) == 0:
            target_tokens, _target_mask = source_tokens, _source_mask
        else:
            target_tokens, _target_mask = self.tokenize_and_mask(
                target_sentence, target_mask
            )

        result = self.matcher.match_by_node(
            source_sentence=source_tokens,
            target_sentence=target_tokens,
            source_mask=_source_mask,
            target_mask=_target_mask,
            max_times=max_times,
            max_depth=max_depth,
            max_edges=max_edges,
            seed=seed,
            edge_beam_width=edge_beam_width,
            trim_path=trim_path,
            discard_edges_if_similarity_below=discard_edges_if_similarity_below,
            discard_edges_if_rank_below=discard_edges_if_rank_below,
        )
        return result

    def match_by_node_embedding(
        self,
        source_sentence: str,
        target_sentence: str = "",
        source_mask: str = "",
        target_mask: str = "",
        max_times: int = 1000,
        max_depth: int = 3,
        max_edges: int = 3,
        seed: int = -1,
        edge_beam_width: int = -1,
        trim_path: bool = True,
        discard_edges_if_similarity_below: float = 0.5,
        discard_edges_if_rank_below: float = 0,
    ) -> Dict[int, Tuple[int, List[List[int]], List[float]]]:
        """
        Returns:
            Dict[end position, Tuple[start position, vector of knowledge sequences, vector of weights]]
            end position is the index of the word behind the match
            start position is the index of the first word in the match
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
            max_edges=max_edges,
            seed=seed,
            edge_beam_width=edge_beam_width,
            trim_path=trim_path,
            discard_edges_if_similarity_below=discard_edges_if_similarity_below,
            discard_edges_if_rank_below=discard_edges_if_rank_below,
        )
        return result

    def match_by_token(
        self,
        source_sentence: str,
        target_sentence: str = "",
        source_mask: str = "",
        target_mask: str = "",
        max_times: int = 1000,
        max_depth: int = 3,
        max_edges: int = 3,
        seed: int = -1,
        edge_beam_width: int = -1,
        trim_path: bool = True,
        discard_edges_if_similarity_below: float = 0,
        discard_edges_if_rank_below: float = 0,
        rank_focus: List[str] = None,
        rank_exclude: List[str] = None,
    ) -> Dict[int, Tuple[int, List[List[int]], List[float]]]:
        """
        Returns:
            Dict[end position, Tuple[start position, vector of knowledge sequences, vector of weights]]
            end position is the index of the word behind the match
            start position is the index of the first word in the match
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

        result = self.matcher.match_by_token(
            source_sentence=source_tokens,
            target_sentence=target_tokens,
            source_mask=_source_mask,
            target_mask=_target_mask,
            max_times=max_times,
            max_depth=max_depth,
            max_edges=max_edges,
            seed=seed,
            edge_beam_width=edge_beam_width,
            trim_path=trim_path,
            discard_edges_if_similarity_below=discard_edges_if_similarity_below,
            discard_edges_if_rank_below=discard_edges_if_rank_below,
            rank_focus=self.tokenizer(rank_focus, add_special_tokens=False).input_ids
            if rank_focus
            else [],
            rank_exclude=self.tokenizer(
                rank_exclude, add_special_tokens=False
            ).input_ids
            if rank_exclude
            else [],
        )
        return result

    def match_to_string(
        self, sentence: str, match: Dict[int, Tuple[int, List[List[int]], List[float]]]
    ) -> Tuple[str, List[Tuple[str, List[str]]]]:
        """
        Returns:
            Remaining sentence piece and
            A list of tuples of:
                1. Sentence piece before knowledge sequences to be inserted,
                2. List of knowledge sequences
        """
        sentence_tokens, _ = self.tokenize_and_mask(sentence)
        sorted_matches = list(
            (k, v) for k, (_, v, __) in match.items()
        )  # type: List[Tuple[int, List[str]]]
        sorted_matches = sorted(sorted_matches, key=lambda x: x[0])
        start = 0
        result = []
        for m in sorted_matches:
            result.append(
                (
                    self.tokenizer.decode(sentence_tokens[start : m[0]]),
                    [self.tokenizer.decode(seq) for seq in m[1]],
                )
            )
            start = m[0]
        return self.tokenizer.decode(sentence_tokens[start:]), result

    @staticmethod
    def unify_match(
        matches: List[Dict[int, Tuple[int, List[List[int]], List[float]]]]
    ) -> Dict[int, Tuple[int, List[List[int]], List[float]]]:
        """
        Unify several match result into one.
        """
        result = {}
        existed_knowledge = set()
        for m in matches:
            for end_pos, (start_pos, knowledge_list, weights) in m.items():
                if end_pos not in result:
                    result[end_pos] = (start_pos, [], [])
                if result[end_pos][0] != start_pos:
                    raise ValueError("Start position mismatch")
                for knowledge, weight in zip(knowledge_list, weights):
                    if tuple(knowledge) not in existed_knowledge:
                        result[end_pos][1].append(knowledge)
                        result[end_pos][2].append(weight)
                        existed_knowledge.add(tuple(knowledge))
        return result

    def insert_match(
        self,
        sentence: str,
        match: Dict[int, Tuple[int, List[List[int]], List[float]]],
        match_begin: str = "(",
        match_sep: str = ",",
        match_end: str = ")",
        insert_at_end: bool = False,
        include_weights: bool = False,
    ) -> str:
        """
        If the triples are directly inserted at the end, accuracy
        reduced by more than 50%
        """
        sentence_tokens, _ = self.tokenize_and_mask(sentence)
        begin_tokens = self.tokenizer.encode(match_begin, add_special_tokens=False)
        end_tokens = self.tokenizer.encode(match_end, add_special_tokens=False)
        sep_tokens = self.tokenizer.encode(match_sep, add_special_tokens=False)
        new_matches = {}
        for pos, (_, edges, weights) in match.items():
            new_edges = []
            for i, (edge, weight) in enumerate(zip(edges, weights)):
                if i == 0:
                    new_edges += begin_tokens
                if include_weights:
                    new_edges += edge + self.tokenizer.encode(
                        f"{weight:.1f}", add_special_tokens=False
                    )
                else:
                    new_edges += edge
                if i == len(edges) - 1:
                    new_edges += end_tokens
                else:
                    new_edges += sep_tokens
            new_matches[pos] = new_edges
        sorted_matches = list(
            (k, v) for k, v in new_matches.items()
        )  # type: List[Tuple[int, List[int]]]
        sorted_matches = sorted(sorted_matches, key=lambda x: x[0])
        if insert_at_end:
            for m in sorted_matches:
                sentence_tokens = sentence_tokens + m[1]
        else:
            offset = 0
            for m in sorted_matches:
                pos = m[0] + offset
                sentence_tokens = sentence_tokens[:pos] + m[1] + sentence_tokens[pos:]
                offset += len(m[1])
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
        filter_set = {
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
                or (pos.startswith("RB") and token.lower().endswith("ly"))
                or (pos.startswith("VB") and token.lower() not in filter_set)
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
            if individual_pos.startswith("NN"):
                result[i][1] = individual_pos
            else:
                result[i][1] = pos
        return [tuple(r) for r in result]
