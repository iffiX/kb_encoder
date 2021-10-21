import re
import os
import sys
import logging
import cppimport
from typing import List, Dict, Tuple
from transformers import PreTrainedTokenizerBase
from encoder.utils.settings import dataset_cache_dir, preprocess_cache_dir
from encoder.utils.file import download_to, decompress_gz

# Set this if gcc is not the default compiler
# os.environ["CC"] = "/usr/bin/gcc-7"
# os.environ["CXX"] = "/usr/bin/g++-7"

_dir_path = os.path.dirname(os.path.abspath(__file__))
_src_path = str(os.path.join(_dir_path, "matcher_src/init.cpp"))
sys.path.append(_dir_path)

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root_logger.addHandler(handler)
matcher = cppimport.imp_from_filepath(_src_path, "matcher_src.matcher")
root_logger.setLevel(logging.INFO)

matcher.set_omp_max_threads(16)


class ConceptNetMatcher:
    URL = (
        "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/"
        "conceptnet-assertions-5.7.0.csv.gz"
    )

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

        data_path = os.path.join(dataset_cache_dir, "conceptnet-assertions.csv")
        archive_path = os.path.join(preprocess_cache_dir, "conceptnet-archive.data")
        if not os.path.exists(data_path):
            if not os.path.exists(str(data_path) + ".gz"):
                logging.info("Downloading concept net assertions")
                download_to(self.URL, str(data_path) + ".gz")
            logging.info("Decompressing")
            decompress_gz(str(data_path) + ".gz", data_path)

        if not os.path.exists(archive_path):
            logging.info("Processing concept net")
            reader = matcher.ConceptNetReader(data_path)
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
            net_matcher = matcher.ConceptNetMatcher(reader)
            logging.info("Saving preprocessed concept net data as archive")
            net_matcher.save(archive_path)
            self.net_matcher = net_matcher
        else:
            self.net_matcher = matcher.ConceptNetMatcher(archive_path)

    def match(
        self,
        sentence: str,
        max_times: int = 1000,
        max_depth: int = 3,
        max_edges: int = 3,
        seed: int = -1,
        similarity_exclude: List[str] = None,
        rank_focus: List[str] = None,
        rank_exclude: List[str] = None,
    ) -> Dict[int, Tuple[int, List[List[int]]]]:
        """
        Returns:
            Dict[end position, Tuple[start position, vector of knowledge sequences]]
            end position is the index of the word behind the match
            start position is the index of the first word in the match
        """
        similarity_exclude = similarity_exclude or [
            "the",
            "a",
            "an",
            "to",
            "of",
            "for",
            "is",
            "are",
        ]
        # print(
        #     f"sentence: {sentence} "
        #     f"rank_focus: {rank_focus} "
        #     f"rank_exclude: {rank_exclude}"
        # )
        # print(
        #     f"sentence_tokens: "
        #     f"{self.tokenizer.encode(sentence, add_special_tokens=False)}\n"
        #     f"rank_focus_tokens: "
        #     f"{self.tokenizer(rank_focus, add_special_tokens=False).input_ids}\n"
        #     f"rank_exclude_tokens: "
        #     f"{self.tokenizer(rank_exclude, add_special_tokens=False).input_ids}\n\n"
        # )
        result = self.net_matcher.match(
            self.tokenizer.encode(sentence, add_special_tokens=False),
            max_times=max_times,
            max_depth=max_depth,
            max_edges=max_edges,
            seed=seed,
            rank_focus=self.tokenizer(rank_focus, add_special_tokens=False).input_ids
            if rank_focus
            else [],
            rank_exclude=self.tokenizer(
                rank_exclude, add_special_tokens=False
            ).input_ids
            if rank_exclude
            else [],
            similarity_exclude=self.tokenizer(
                similarity_exclude, add_special_tokens=False
            ).input_ids,
        )
        return result

    def insert_matches(
        self,
        sentence: str,
        matches: Dict[int, Tuple[int, List[List[int]]]],
        match_begin: str = "(",
        match_sep: str = ",",
        match_end: str = ")",
    ):
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
        offset = 0
        for m in sorted_matches:
            pos = m[0] + offset
            sentence_tokens = sentence_tokens[:pos] + m[1] + sentence_tokens[pos:]
            offset += len(m[1])
        return self.tokenizer.decode(sentence_tokens)

    def __reduce__(self):
        return ConceptNetMatcher, (self.tokenizer,)
