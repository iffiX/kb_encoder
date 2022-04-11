import re
import os
import json
import logging
from transformers import PreTrainedTokenizerBase
from encoder.dataset.download import ConceptNet, OpenBookQA, ARC, QASC
from encoder.dataset.matcher import ConceptNetReader, KnowledgeMatcher
from encoder.dataset.matcher.base import BaseMatcher
from encoder.utils.settings import preprocess_cache_dir


class ARCMatcher(BaseMatcher):
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        archive_path = str(
            os.path.join(preprocess_cache_dir, "conceptnet-archive.data")
        )
        self.concept_net = ConceptNet().require()
        self.openbook_qa = OpenBookQA().require()
        self.arc = ARC().require()
        self.qasc = QASC().require()

        if not os.path.exists(archive_path):
            logging.info("Processing concept net")
            reader = ConceptNetReader().read(
                asserion_path=self.concept_net.assertion_path,
                weight_path=self.concept_net.numberbatch_path,
                weight_hdf5_path=self.concept_net.embedding_path,
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
            logging.info("Initializing KnowledgeMatcher")
            matcher = KnowledgeMatcher(archive_path)
        super(ARCMatcher, self).__init__(tokenizer, matcher)

        matcher.kb.disable_edges_of_relationships(
            [
                "DerivedFrom",
                "EtymologicallyDerivedFrom",
                "EtymologicallyRelatedTo",
                "RelatedTo",
                "FormOf",
                "DefinedAs",
            ]
        )
        self.matcher.kb.disable_edges_with_weight_below(1)

        # Add knowledge as composite nodes
        self.add_additional_knowledge()

    def add_additional_knowledge(self):
        logging.info("Adding additional knowledge")
        count = 0
        for fact in self.generate_facts():
            fact = fact.strip("\n").strip(".").strip('"').strip("'").strip(",")
            if len(fact) < 3:
                continue
            count += 1
            ids, mask = self.tokenize_and_mask(fact)
            self.matcher.kb.add_composite_node(fact, "RelatedTo", ids, mask)
        logging.info(f"Added {count} composite nodes")

    def generate_facts(self):
        facts = set()
        for path in (
            self.qasc.train_path,
            self.qasc.validate_path,
        ):
            with open(path, "r") as file:
                for line in file:
                    entry = json.loads(line)
                    facts.add(entry["fact1"].strip(".").lower())
                    facts.add(entry["fact2"].strip(".").lower())
        for path in (
            self.openbook_qa.facts_path,
            self.openbook_qa.crowd_source_facts_path,
        ):
            with open(path, "r") as file:
                for line in file:
                    line = line.strip("\n").strip('"').strip(".")
                    if len(line) < 3:
                        continue
                    facts.add(line.lower())
        logging.info(f"Generated {len(facts)} facts")
        return sorted(list(facts))

    def __reduce__(self):
        return ARCMatcher, (self.tokenizer,)
