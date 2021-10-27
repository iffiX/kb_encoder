import re
import os
import sys
import nltk
import cmake
import shutil
import logging
import subprocess
import importlib.util
from checksumdir import dirhash
from typing import List, Dict, Tuple, Union

from statsmodels.sandbox.stats.runs import Runs
from transformers import PreTrainedTokenizerBase
from encoder.utils.settings import dataset_cache_dir, preprocess_cache_dir
from encoder.utils.file import download_to, decompress_gz

# Set this if gcc is not the default compiler
# os.environ["CC"] = "/usr/bin/gcc-7"
# os.environ["CXX"] = "/usr/bin/g++-7"

_dir_path = str(os.path.dirname(os.path.abspath(__file__)))
_src_path = str(os.path.join(_dir_path, "matcher_src"))
_build_path = str(os.path.join(_dir_path, "build"))
sys.path.append(_src_path)

md5hash = dirhash(
    _src_path,
    "md5",
    excluded_extensions=["txt", "so"],
    excluded_files=["cmake-build-debug", "idea"],
)
build = True
if os.path.exists(os.path.join(_build_path, "hash.txt")):
    with open(os.path.join(_build_path, "hash.txt"), "r") as file:
        build = file.read() != md5hash
if build:
    shutil.rmtree(_build_path)
    os.makedirs(_build_path)
    subprocess.call(
        [
            os.path.join(cmake.CMAKE_BIN_DIR, "cmake"),
            "-S",
            _src_path,
            "-B",
            _build_path,
            "-DCMAKE_BUILD_TYPE=Release",
        ]
    )
    subprocess.call(["make", "-C", _build_path, "clean"])
    if subprocess.call(["make", "-C", _build_path, "-j4"]) != 0:
        raise RuntimeError("Make failed")
    subprocess.call(["make", "-C", _build_path, "install"])
    with open(os.path.join(_build_path, "hash.txt"), "w") as file:
        file.write(md5hash)

matcher = importlib.import_module("matcher")


class ConceptNetTokenizer:
    def __init__(self):
        pass


class ConceptNetMatcher:
    ASSERTION_URL = (
        "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/"
        "conceptnet-assertions-5.7.0.csv.gz"
    )
    NUMBERBATCH_URL = (
        "https://conceptnet.s3.amazonaws.com/downloads/2019/"
        "numberbatch/numberbatch-en-19.08.txt.gz"
    )

    def __init__(self, tokenizer: Union[PreTrainedTokenizerBase, ConceptNetTokenizer]):
        self.tokenizer = tokenizer
        nltk.download("stopwords")
        nltk.download("punkt")
        nltk.download("averaged_perceptron_tagger")

        dataset_cache_dir = preprocess_cache_dir = "/data"

        assertion_path = str(
            os.path.join(dataset_cache_dir, "conceptnet-assertions.csv")
        )
        numberbatch_path = str(
            os.path.join(dataset_cache_dir, "conceptnet-numberbatch.txt")
        )
        archive_path = str(
            os.path.join(preprocess_cache_dir, "conceptnet-archive.data")
        )
        embedding_path = str(
            os.path.join(preprocess_cache_dir, "conceptnet-embedding.hdf5")
        )
        landmark_path = str(
            os.path.join(preprocess_cache_dir, "conceptnet-landmark.cache")
        )

        for task, data_path, url in (
            ("assertions", assertion_path, self.ASSERTION_URL),
            ("numberbatch", numberbatch_path, self.NUMBERBATCH_URL),
        ):
            if not os.path.exists(data_path):
                if not os.path.exists(str(data_path) + ".gz"):
                    logging.info(f"Downloading concept net {task}")
                    download_to(url, str(data_path) + ".gz")
                logging.info("Decompressing")
                decompress_gz(str(data_path) + ".gz", data_path)

        if not os.path.exists(archive_path):
            logging.info("Processing concept net")
            reader = matcher.ConceptNetReader().read(
                asserion_path=assertion_path,
                weight_path=numberbatch_path,
                weight_hdf5_path=embedding_path,
                simplify_with_int8=True,
            )
            reader.tokenized_nodes = tokenizer(
                reader.nodes, add_special_tokens=False
            ).input_ids
            relationships = [
                " ".join([string.lower() for string in re.findall("[A-Z][a-z]*", rel)])
                for rel in reader.relationships
            ]
            reader.tokenized_relationships = tokenizer(
                relationships, add_special_tokens=False
            ).input_ids
            reader.tokenized_edge_annotations = tokenizer(
                [edge[4] for edge in reader.edges], add_special_tokens=False
            ).input_ids
            net_matcher = matcher.KnowledgeMatcher(reader)
            logging.info("Saving preprocessed concept net data as archive")
            net_matcher.save(archive_path)
            self.net_matcher = net_matcher
        else:
            self.net_matcher = matcher.KnowledgeMatcher(archive_path)

        self.net_matcher.kb.init_landmarks(
            seed_num=100, landmark_num=100, seed=41379823, landmark_path=landmark_path
        )

    def match_by_node(
        self,
        source_sentence: str,
        target_sentence: str = "",
        max_times: int = 1000,
        max_depth: int = 3,
        max_edges: int = 3,
        seed: int = -1,
        discard_edges_if_similarity_below: float = 0,
        discard_edges_if_rank_below: float = 0,
    ) -> Dict[int, Tuple[int, List[List[int]]]]:
        """
        Returns:
            Dict[end position, Tuple[start position, vector of knowledge sequences]]
            end position is the index of the word behind the match
            start position is the index of the first word in the match
        """
        source_tokens, source_mask = self.tokenize_and_mask(source_sentence)
        if len(target_sentence) == 0:
            target_tokens, target_mask = source_tokens, source_mask
        else:
            target_tokens, target_mask = self.tokenize_and_mask(target_sentence)

        result = self.net_matcher.match_by_node(
            source_sentence=source_tokens,
            target_sentence=target_tokens,
            source_mask=source_mask,
            target_mask=target_mask,
            max_times=max_times,
            max_depth=max_depth,
            max_edges=max_edges,
            seed=seed,
            discard_edges_if_similarity_below=discard_edges_if_similarity_below,
            discard_edges_if_rank_below=discard_edges_if_rank_below,
        )
        return result

    def match_by_node_embedding(
        self,
        source_sentence: str,
        target_sentence: str = "",
        max_times: int = 1000,
        max_depth: int = 3,
        max_edges: int = 3,
        seed: int = -1,
        discard_edges_if_similarity_below: float = 0.5,
        discard_edges_if_rank_below: float = 0,
    ) -> Dict[int, Tuple[int, List[List[int]]]]:
        """
        Returns:
            Dict[end position, Tuple[start position, vector of knowledge sequences]]
            end position is the index of the word behind the match
            start position is the index of the first word in the match
        """
        source_tokens, source_mask = self.tokenize_and_mask(source_sentence)
        if len(target_sentence) == 0:
            target_tokens, target_mask = source_tokens, source_mask
        else:
            target_tokens, target_mask = self.tokenize_and_mask(target_sentence)

        result = self.net_matcher.match_by_node_embedding(
            source_sentence=source_tokens,
            target_sentence=target_tokens,
            source_mask=source_mask,
            target_mask=target_mask,
            max_times=max_times,
            max_depth=max_depth,
            max_edges=max_edges,
            seed=seed,
            discard_edges_if_similarity_below=discard_edges_if_similarity_below,
            discard_edges_if_rank_below=discard_edges_if_rank_below,
        )
        return result

    def match_by_token(
        self,
        source_sentence: str,
        target_sentence: str = "",
        max_times: int = 1000,
        max_depth: int = 3,
        max_edges: int = 3,
        seed: int = -1,
        discard_edges_if_similarity_below: float = 0,
        discard_edges_if_rank_below: float = 0,
        rank_focus: List[str] = None,
        rank_exclude: List[str] = None,
    ) -> Dict[int, Tuple[int, List[List[int]]]]:
        """
        Returns:
            Dict[end position, Tuple[start position, vector of knowledge sequences]]
            end position is the index of the word behind the match
            start position is the index of the first word in the match
        """
        source_tokens, source_mask = self.tokenize_and_mask(source_sentence)
        if len(target_sentence) == 0:
            target_tokens, target_mask = source_tokens, source_mask
        else:
            target_tokens, target_mask = self.tokenize_and_mask(target_sentence)

        result = self.net_matcher.match_by_token(
            source_sentence=source_tokens,
            target_sentence=target_tokens,
            source_mask=source_mask,
            target_mask=target_mask,
            max_times=max_times,
            max_depth=max_depth,
            max_edges=max_edges,
            seed=seed,
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
        self, sentence: str, matches: Dict[int, Tuple[int, List[List[int]]]]
    ) -> Tuple[str, List[Tuple[str, List[str]]]]:
        """
        Returns:
            Remaining sentence piece and
            A list of tuples of:
                1. Sentence piece before knowledge sequences to be inserted,
                2. List of knowledge sequences
        """
        sentence_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
        sorted_matches = list(
            (k, v) for k, (_, v) in matches.items()
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

    def insert_match(
        self,
        sentence: str,
        matches: Dict[int, Tuple[int, List[List[int]]]],
        match_begin: str = "(",
        match_sep: str = ",",
        match_end: str = ")",
        insert_at_end: bool = False,
    ) -> str:
        """
        If the triples are directly inserted at the end, accuracy
        reduced by more than 50%
        """
        sentence_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
        begin_tokens = self.tokenizer.encode(match_begin, add_special_tokens=False)
        end_tokens = self.tokenizer.encode(match_end, add_special_tokens=False)
        sep_tokens = self.tokenizer.encode(match_sep, add_special_tokens=False)
        new_matches = {}
        for pos, (_, edges) in matches.items():
            new_edges = []
            for i, edge in enumerate(edges):
                if i == 0:
                    new_edges += begin_tokens
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

    def tokenize_and_mask(self, sentence: str):
        """
        Args:
            sentence: A sentence to be tagged by Part of Speech (POS)
        Returns:
            A list of token ids.
            A list of POS mask.
        """
        tokens = nltk.word_tokenize(sentence.lower())
        masks = []
        ids = []
        for token, pos in nltk.pos_tag(tokens):
            ids.append(self.tokenizer.encode(token, add_special_tokens=False))
            if "NN" in pos or "JJ" in pos or "RB" in pos or "VB" in pos:
                # noun, adjective, adverb, verb
                masks.append([1] * len(ids[-1]))
            else:
                masks.append([0] * len(ids[-1]))
        return [i for iid in ids for i in iid], [m for mask in masks for m in mask]

    def __reduce__(self):
        return ConceptNetMatcher, (self.tokenizer,)
