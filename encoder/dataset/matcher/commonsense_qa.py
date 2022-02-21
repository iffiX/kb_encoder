import re
import os
import nltk
import json
import logging
import datasets
from nltk.corpus import wordnet
from transformers import PreTrainedTokenizerBase
from encoder.dataset.matcher import ConceptNetReader, KnowledgeMatcher
from encoder.dataset.matcher.base import BaseMatcher
from encoder.utils.settings import dataset_cache_dir, preprocess_cache_dir
from encoder.utils.file import download_to, decompress_gz


class CommonsenseQAMatcher(BaseMatcher):
    ASSERTION_URL = (
        "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/"
        "conceptnet-assertions-5.7.0.csv.gz"
    )
    NUMBERBATCH_URL = (
        "https://conceptnet.s3.amazonaws.com/downloads/2019/"
        "numberbatch/numberbatch-en-19.08.txt.gz"
    )
    TRAIN_URL = "https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl"

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        assertion_path = str(
            os.path.join(dataset_cache_dir, "conceptnet-assertions.csv")
        )
        numberbatch_path = str(
            os.path.join(dataset_cache_dir, "conceptnet-numberbatch.txt")
        )
        embedding_path = str(
            os.path.join(preprocess_cache_dir, "conceptnet-embedding.hdf5")
        )
        archive_path = str(
            os.path.join(preprocess_cache_dir, "conceptnet-archive.data")
        )
        train_path = os.path.join(dataset_cache_dir, "commonsense_qa", "train.jsonl")

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

        if not os.path.exists(train_path):
            logging.info("Downloading commonsense qa train dataset.")
            download_to(self.TRAIN_URL, train_path)

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
        super(CommonsenseQAMatcher, self).__init__(tokenizer, matcher)
        # Disable relations of similar word forms
        matcher.kb.disable_edges_of_relationships(
            [
                "DerivedFrom",
                "EtymologicallyDerivedFrom",
                "EtymologicallyRelatedTo",
                # "RelatedTo",
                # "FormOf",
            ]
        )
        self.matcher.kb.disable_edges_with_weight_below(1)
        super(CommonsenseQAMatcher, self).__init__(tokenizer, matcher)
        self.add_commonsense_qa_train_dataset()
        # self.add_wordnet_definition()
        # self.add_openbook_qa_knowledge()
        # self.add_generics_kb()

    def add_commonsense_qa_train_dataset(self):
        logging.info("Adding Commonsense QA train dataset")
        commonsense_qa_train_dataset_path = os.path.join(
            dataset_cache_dir, "commonsense_qa", "train.jsonl"
        )
        count = 0
        with open(commonsense_qa_train_dataset_path, "r") as file:
            for line in file:
                sample = json.loads(line)
                correct_choice = [
                    c["text"]
                    for c in sample["question"]["choices"]
                    if c["label"] == sample["answerKey"]
                ][0]
                line = sample["question"]["stem"] + " " + correct_choice
                if line.count(".") >= 3:
                    continue
                count += 1
                ids, mask = self.tokenize_and_mask(line)
                self.matcher.kb.add_composite_node(line, "RelatedTo", ids, mask)
        logging.info(f"Added {count} composite nodes")

    # def add_generics_kb(self):
    #     logging.info("Adding generics kb")
    #     path = str(os.path.join(dataset_cache_dir, "generics_kb"))
    #     os.makedirs(path, exist_ok=True)
    #     if not os.path.exists(os.path.join(path, "GenericsKB-Best.tsv")):
    #         logging.info("Skipping loading generics_kb because file is not loaded")
    #         logging.info(
    #             f"Please download GenericsKB-Best.tsv "
    #             f"from https://drive.google.com/drive/folders/1vqfVXhJXJWuiiXbUa4rZjOgQoJvwZUoT "
    #             f"to path {os.path.join(path, 'GenericsKB-Best.tsv')}"
    #         )
    #     gkb = datasets.load_dataset("generics_kb", "generics_kb_best", data_dir=path,)
    #     added = set()
    #     for entry in gkb["train"]:
    #         if (
    #             "ConceptNet" in entry["source"]
    #             or "WordNet" in entry["source"]
    #             or entry["generic_sentence"].count(" ") < 3
    #         ):
    #             continue
    #         knowledge = (
    #             entry["generic_sentence"]
    #             .strip(".")
    #             .replace("(", ",")
    #             .replace(")", ",")
    #             .replace(";", ",")
    #             .replace('"', " ")
    #             .lower()
    #         )
    #         if knowledge not in added:
    #             added.add(knowledge)
    #             self.matcher.kb.add_composite_node(
    #                 knowledge,
    #                 "RelatedTo",
    #                 self.tokenizer.encode(knowledge, add_special_tokens=False),
    #             )
    #     logging.info(f"Added {len(added)} composite nodes")

    # def add_openbook_qa_knowledge(self):
    #     logging.info("Adding OpenBook QA knowledge")
    #     openbook_qa_path = os.path.join(
    #         dataset_cache_dir, "openbook_qa", "OpenBookQA-V1-Sep2018", "Data"
    #     )
    #     openbook_qa_facts_path = os.path.join(openbook_qa_path, "Main", "openbook.txt")
    #     crowd_source_facts_path = os.path.join(
    #         openbook_qa_path, "Additional", "crowdsourced-facts.txt"
    #     )
    #     qasc_additional_path = os.path.join(preprocess_cache_dir, "qasc_additional.txt")
    #     manual_additional_path = os.path.join(
    #         preprocess_cache_dir, "manual_additional.txt"
    #     )
    #     count = 0
    #     for path in (
    #         openbook_qa_facts_path,
    #         crowd_source_facts_path,
    #         # qasc_additional_path,
    #         manual_additional_path,
    #     ):
    #         with open(path, "r") as file:
    #             for line in file:
    #                 line = line.strip("\n").strip(".").strip('"').strip("'").strip(",")
    #                 if len(line) < 3:
    #                     continue
    #                 count += 1
    #                 ids, mask = self.tokenize_and_mask(line)
    #                 self.matcher.kb.add_composite_node(line, "RelatedTo", ids, mask)
    #     logging.info(f"Added {count} composite nodes")

    def add_wordnet_definition(self):
        logging.info("Adding wordnet definition")
        added = set()
        for ss in wordnet.all_synsets():
            s = [ln.replace("_", " ").lower() for ln in ss.lemma_names()]
            if s[0].count(" ") > 0:
                continue
            definition = (
                ss.definition()
                .replace("(", ",")
                .replace(")", ",")
                .replace(";", ",")
                .replace('"', " ")
                .lower()
            )
            # knowledge = f"{','.join(s)} is defined as {definition}"
            knowledge = f"{s[0]} is defined as {definition}"

            if len(knowledge) > 100:
                # if trim_index = -1 this will also work, but not trimming anything
                trim_index = knowledge.find(" ", 100)
                knowledge = knowledge[:trim_index]
            if knowledge not in added:
                added.add(knowledge)
                # Only allow definition part to be matched
                sentence_mask = "+" * len(s[0]) + "-" * (len(knowledge) - len(s[0]))
                ids, mask = self.tokenize_and_mask(knowledge, sentence_mask)
                self.matcher.kb.add_composite_node(knowledge, "RelatedTo", ids, mask)
        logging.info(f"Added {len(added)} composite nodes")

    def __reduce__(self):
        return CommonsenseQAMatcher, (self.tokenizer,)
