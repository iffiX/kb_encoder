import re
import os
import nltk
import logging
from transformers import PreTrainedTokenizerBase
from encoder.dataset.matcher import ConceptNetReader, KnowledgeMatcher
from encoder.dataset.matcher.base import BaseMatcher
from encoder.utils.settings import dataset_cache_dir, preprocess_cache_dir
from encoder.utils.file import download_to, decompress_gz, decompress_zip


class OpenBookQAMatcher(BaseMatcher):
    ASSERTION_URL = (
        "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/"
        "conceptnet-assertions-5.7.0.csv.gz"
    )
    NUMBERBATCH_URL = (
        "https://conceptnet.s3.amazonaws.com/downloads/2019/"
        "numberbatch/numberbatch-en-19.08.txt.gz"
    )
    OPENBOOK_QA_URL = (
        "https://ai2-public-datasets.s3.amazonaws.com/open-book-qa/"
        "OpenBookQA-V1-Sep2018.zip"
    )

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        nltk.download("stopwords")
        nltk.download("punkt")
        nltk.download("averaged_perceptron_tagger")

        assertion_path = str(
            os.path.join(dataset_cache_dir, "conceptnet-assertions.csv")
        )
        numberbatch_path = str(
            os.path.join(dataset_cache_dir, "conceptnet-numberbatch.txt")
        )
        openbook_qa_path = str(os.path.join(dataset_cache_dir, "openbook_qa"))
        embedding_path = str(
            os.path.join(preprocess_cache_dir, "conceptnet-embedding.hdf5")
        )
        archive_path = str(
            os.path.join(preprocess_cache_dir, "conceptnet-archive.data")
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

        if not os.path.exists(openbook_qa_path):
            if not os.path.exists(str(openbook_qa_path) + ".zip"):
                logging.info("Downloading OpenBook QA")
                download_to(self.OPENBOOK_QA_URL, str(openbook_qa_path) + ".zip")
            logging.info("Decompressing")
            decompress_zip(str(openbook_qa_path) + ".zip", openbook_qa_path)

        if not os.path.exists(archive_path):
            logging.info("Processing concept net")
            reader = ConceptNetReader().read(
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
            matcher = KnowledgeMatcher(reader)
            logging.info("Saving preprocessed concept net data as archive")
            matcher.save(archive_path)
        else:
            matcher = KnowledgeMatcher(archive_path)

        # Disable relations of similar word forms
        matcher.kb.disable_edges_of_relationships(
            [
                "DerivedFrom",
                "EtymologicallyDerivedFrom",
                "EtymologicallyRelatedTo",
                "FormOf",
            ]
        )
        self.add_openbook_qa_knowledge()
        # Add knowledge from openbook QA as composite nodes
        super(OpenBookQAMatcher, self).__init__(tokenizer, matcher)

    def add_openbook_qa_knowledge(self):
        openbook_qa_path = os.path.join(
            dataset_cache_dir, "openbook_qa", "OpenBookQA-V1-Sep2018", "Data"
        )

        crowd_source_facts_path = os.path.join(
            openbook_qa_path, "Additional", "crowdsourced-facts.txt"
        )
        openbook_qa_facts_path = os.path.join(openbook_qa_path, "Main", "openbook.txt")

        facts = []
        for path in (crowd_source_facts_path, openbook_qa_facts_path):
            with open(path, "r") as file:
                for line in file:
                    if line.startswith('"'):
                        line = line[1:-1]
                self.matcher.add_composite_node(
                    line,
                    "RelatedTo",
                    self.tokenizer.encode(line, add_special_tokens=False),
                )

    def __reduce__(self):
        return OpenBookQAMatcher, (self.tokenizer,)