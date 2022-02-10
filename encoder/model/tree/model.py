import logging
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import graphviz as g
from typing import List, Tuple
from transformers import AutoModel
from encoder.utils.settings import (
    proxies,
    model_cache_dir,
    huggingface_mirror,
    local_files_only,
)


class Node:
    def __init__(self, keyword, idf, covered_document_idx, is_end=False):
        self.keyword = keyword
        self.idf = idf
        self.covered_document_idx = covered_document_idx
        self.children = None
        self.is_end = is_end


class KeywordTree(nn.Module):
    def __init__(self, keywords_list: List[List[str]], hidden_size: int):
        super(KeywordTree, self).__init__()
        self.documents = list(
            self.normalize_keywords(keywords) for keywords in keywords_list
        )
        self.documents_set = set(self.documents)
        root = Node(
            keyword=None, idf=0, covered_document_idx=set(range(len(self.documents))),
        )
        # Note: returned nodes does not include root
        self.root, self.nodes = self.make_tree(root, self.documents)
        self.nodes = [self.root] + self.nodes
        logging.info(f"Created tree with {len(self.nodes) - 1} keyword nodes")
        self.embedding = nn.Embedding(
            num_embeddings=len(self.nodes), embedding_dim=hidden_size
        )

    def forward(self, outputs, keywords_list: List[List[str]]):
        """
        Args:
            outputs: Hidden states from previous BERT style models.
                Shape [batch_size, seq_length, hidden_size]
            keywords_list: List of keywords to be trained, list size if batch_size.
        """
        for keywords in keywords_list:
            if self.normalize_keywords(keywords) not in self.documents_set:
                raise ValueError(
                    f"Normalized Keywords {self.normalize_keywords(keywords)}, "
                    f"from {keywords}, "
                    f"not found"
                )
        hidden = outputs[:, 0, :]
        total_loss = 0
        for h, keywords in zip(hidden, keywords_list):
            root = self.root
            keywords = self.normalize_keywords(keywords)
            document_idx = self.documents.index(keywords)
            loss = 0
            steps = 0
            while root.children is not None:
                next_root = [
                    child
                    for child in root.children
                    if document_idx in child.covered_document_idx
                ][0]
                embedding_offset = self.nodes.index(root.children[0])
                children_embeddings = self.get_embedding(
                    list(range(embedding_offset, embedding_offset + len(root.children)))
                )
                loss = loss + F.cross_entropy(
                    t.matmul(children_embeddings, h).unsqueeze(0),
                    t.LongTensor([self.nodes.index(next_root) - embedding_offset]).to(
                        children_embeddings.device
                    ),
                )
                steps += 1
                root = next_root
            loss = loss / steps
            total_loss = total_loss + loss
        return total_loss / hidden.shape[0]

    def predict(self, outputs):
        """
        Args:
            outputs: Hidden states from previous BERT style models.
                Shape [batch_size, seq_length, hidden_size]
        """
        hidden = outputs[:, 0, :]
        keywords_list = []
        for h in hidden:
            root = self.root
            keywords = []
            with t.no_grad():
                while root.children is not None:
                    if len(root.children) == 1:
                        next_root = root.children[0]
                    else:
                        embedding_offset = self.nodes.index(root.children[0])
                        children_embeddings = self.get_embedding(
                            list(
                                range(
                                    embedding_offset,
                                    embedding_offset + len(root.children),
                                )
                            )
                        )
                        next_root_relative_idx = t.argmax(
                            t.matmul(children_embeddings, h)
                        ).item()
                        next_root = root.children[next_root_relative_idx]
                    if next_root.keyword is not None:
                        # skip end node
                        keywords.append(next_root.keyword)
                    root = next_root
                keywords_list.append(keywords)
        return keywords_list

    def normalize_keywords(self, keywords: List[str]):
        return tuple(sorted(list(set(keywords))))

    def get_embedding(self, positions):
        return self.embedding(t.LongTensor(positions).to(self.embedding.weight.device))

    def make_tree(self, root, documents: List[Tuple[str]]):
        nodes = []
        children = []
        children_covered_documents = []
        not_covered_documents = documents
        all_covered_document_idx = set(
            [idx for idx, doc in enumerate(documents) if doc is not None and not doc]
        )
        while any(not_covered_documents):
            (
                most_common_keyword,
                idf,
                covered_documents,
                covered_document_idx,
                not_covered_documents,
                not_covered_document_idx,
            ) = self.cover_documents_with_most_common_keyword(not_covered_documents)

            children.append(
                Node(
                    keyword=most_common_keyword,
                    idf=idf,
                    covered_document_idx=covered_document_idx,
                )
            )
            children_covered_documents.append(covered_documents)
        nodes += children

        for child, child_covered_documents in zip(children, children_covered_documents):
            _, child_added_nodes = self.make_tree(child, child_covered_documents)
            nodes += child_added_nodes

        if all_covered_document_idx:
            # special end node
            end_node = [
                Node(
                    keyword=None,
                    idf=0,
                    covered_document_idx=all_covered_document_idx,
                    is_end=True,
                )
            ]
            children = end_node + children
            nodes = end_node + nodes
        root.children = children if len(children) > 0 else None
        return root, nodes

    def cover_documents_with_most_common_keyword(self, documents: List[Tuple[str]]):
        covered_documents = []
        not_covered_documents = []
        covered_document_idx = set()
        not_covered_document_idx = set()
        most_common_keyword, idf = self.find_most_common_keyword(documents)
        if most_common_keyword is None:
            raise ValueError("No common keyword found")
        for idx, doc in enumerate(documents):
            if doc and most_common_keyword in doc:
                covered_document_idx.add(idx)
                new_doc = tuple(
                    keyword for keyword in doc if keyword != most_common_keyword
                )
                covered_documents.append(new_doc)
                not_covered_documents.append(None)
            else:
                covered_documents.append(None)
                not_covered_documents.append(doc)
                if doc:
                    not_covered_document_idx.add(idx)

        return (
            most_common_keyword,
            idf,
            covered_documents,
            covered_document_idx,
            not_covered_documents,
            not_covered_document_idx,
        )

    def find_most_common_keyword(self, documents: List[Tuple[str]]):
        frequency = {}
        for doc in documents:
            if doc is not None:
                for keyword in doc:
                    if keyword not in frequency:
                        frequency[keyword] = 0
                    frequency[keyword] += 1
        keyword_by_frequency = sorted(
            list(frequency.items()), key=lambda x: x[1], reverse=True
        )
        if len(keyword_by_frequency) > 0:
            return (
                keyword_by_frequency[0][0],
                np.log(len(documents) / (1 + keyword_by_frequency[0][1])),
            )
        else:
            return None, 0

    def visualize_tree(self):
        graph = ["digraph BST {", "graph [ dpi = 72 ]; ", 'node [fontname="Arial" ];']
        for idx, node in enumerate(self.nodes):
            shape = "box"
            if node.keyword is None and not node.is_end:
                shape = "diamond"
                label = '""'
            elif node.is_end:
                label = '"$"'
            else:
                label = f'"{node.keyword}"'
            graph.append(f"node{idx} [shape={shape}, label={label}];")
        for idx, node in enumerate(self.nodes):
            connected = []
            if node.children is not None:
                for child in node.children:
                    connected.append(f"node{self.nodes.index(child)}")
            if connected:
                graph.append(f"node{idx} -> {{ {' '.join(connected)} }};")
        graph.append("}")
        src = g.Source("\n".join(graph), filename="visualize.gv", format="svg")
        src.view()


class Model(nn.Module):
    def __init__(self, base_type, keywords_list: List[List[str]]):
        super(Model, self).__init__()

        if not (
            ("albert" in base_type)
            or ("deberta" in base_type)
            or ("electra" in base_type)
            or ("roberta" in base_type)
        ):
            raise ValueError(f"Model type {base_type} not supported.")

        self.base = AutoModel.from_pretrained(
            base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            return_dict=True,
            local_files_only=local_files_only,
        )

        self.tree = KeywordTree(
            keywords_list=keywords_list, hidden_size=self.base.config.hidden_size
        )

    def forward(
        self, input_ids, attention_mask, token_type_ids, keywords_list: List[List[str]]
    ):
        """
        Args:
            input_ids: LongTensor of shape [batch_size, choice_num, seq_length]
            attention_mask: FloatTensor of shape [batch_size, choice_num, seq_length]
            token_type_ids: LongTensor of shape [batch_size, choice_num, seq_length]
            keywords_list: List of keywords to be trained, list size if batch_size.
        """
        return self.tree(
            self._forward(input_ids, attention_mask, token_type_ids), keywords_list
        )

    def predict(self, input_ids, attention_mask, token_type_ids):
        """
        Args:
            input_ids: LongTensor of shape [batch_size, choice_num, seq_length]
            attention_mask: FloatTensor of shape [batch_size, choice_num, seq_length]
            token_type_ids: LongTensor of shape [batch_size, choice_num, seq_length]
        """
        return self.tree.predict(
            self._forward(input_ids, attention_mask, token_type_ids)
        )

    def _forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs.last_hidden_state
