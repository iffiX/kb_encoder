import logging
import numpy as np
import torch as t
import torch.nn as nn
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
    def __init__(
        self, keyword, idf, is_intermediate, covered_document_idx, is_root=False
    ):
        self.keyword = keyword
        self.idf = idf
        self.is_root = is_root
        self.is_intermediate = is_intermediate
        self.covered_document_idx = covered_document_idx
        self.left = None
        self.right = None


class KeywordTree(nn.Module):
    def __init__(self, keywords_list: List[List[str]], hidden_size: int):
        super(KeywordTree, self).__init__()
        self.documents = list(
            self.normalize_keywords(keywords) for keywords in keywords_list
        )
        self.documents_set = set(self.documents)
        root = Node(
            keyword=None,
            idf=0,
            is_root=True,
            is_intermediate=False,
            covered_document_idx=set(range(len(self.documents))),
        )
        self.root, self.nodes = self.make_tree(root, self.documents)
        keyword_count = sum(1 if n.keyword is not None else 0 for n in self.nodes)
        intermediate_count = sum(1 if n.is_intermediate else 0 for n in self.nodes)
        logging.info(
            f"Created tree with {keyword_count} keyword nodes, "
            f"{intermediate_count} intermediate nodes"
        )
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
            while True:
                (
                    root,
                    embedding,
                    is_left,
                    is_end,
                ) = self.get_next_node_and_current_embedding(root, document_idx)
                if is_left:
                    loss = loss + t.log(t.sigmoid(t.dot(embedding, h)) + 1e-7)
                else:
                    loss = loss + t.log(t.sigmoid(-1 * t.dot(embedding, h)) + 1e-7)
                # if root.is_intermediate:
                #     print(f"Moving to <intermediate>")
                # elif is_end:
                #     print("Moving to <end>")
                # else:
                #     print(f"Moving to {root.keyword}")
                steps += 1
                if is_end:
                    break
            loss = loss / steps
            total_loss = total_loss + loss
            # print()
        # Maximize "loss"
        return -1 * total_loss / hidden.shape[0]

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
            embedding = self.get_embedding(self.nodes.index(root))
            keywords = []
            with t.no_grad():
                while True:
                    left_score = t.sigmoid(t.dot(embedding, h)).item()
                    if left_score > 0.5 or root.right is None:
                        root = root.left
                    else:
                        root = root.right
                    embedding = self.get_embedding(self.nodes.index(root))

                    if not root.is_intermediate and root.keyword is None:
                        # end node
                        keywords_list.append(keywords)
                        break
                    elif root.keyword is not None:
                        keywords.append(root.keyword)

        return keywords_list

    def get_next_node_and_current_embedding(self, root: Node, document_idx: int):
        """
        Returns:
            next root, embedding of current root, is moving left, is ending
        """
        embedding = self.get_embedding(self.nodes.index(root))
        if root.is_intermediate:
            # Move along intermediate nodes until a valid keyword node
            # which covers the document index is reached
            is_left = document_idx in root.left.covered_document_idx
            next_node = root.left if is_left else root.right
            if not is_left and root.right is None:
                raise ValueError("Reaching the end of intermediate nodes")
            return next_node, embedding, is_left, False
        else:
            if (
                root.right is not None
                and document_idx in root.right.covered_document_idx
            ):
                # There are uncovered keywords left,
                # move from a keyword node to next intermediate node
                return root.right, embedding, False, False
            else:
                # Reach an end node
                return root.left, embedding, True, True

    def normalize_keywords(self, keywords: List[str]):
        return tuple(sorted(list(set(keywords))))

    def get_embedding(self, position):
        return self.embedding(
            t.LongTensor([position]).to(self.embedding.weight.device)
        ).squeeze(0)

    def make_tree(self, root, documents: List[Tuple[str]]):
        nodes = [root]
        if root.is_root or root.keyword is not None:
            all_covered_document_idx = set(
                idx
                for idx in root.covered_document_idx
                if documents[idx] is not None and len(documents[idx]) == 0
            )

            # For the root node, or the keyword node
            # left is always a special end node
            root.left = Node(
                keyword=None,
                idf=0,
                is_intermediate=False,
                covered_document_idx=all_covered_document_idx,
            )
            nodes += [root.left]

            # right is an intermediate node if there are uncovered documents
            # otherwise it is None
            if any(documents):
                root.right, right_added_nodes = self.make_tree(
                    Node(
                        keyword=None,
                        idf=0,
                        is_intermediate=True,
                        covered_document_idx=root.covered_document_idx.difference(
                            all_covered_document_idx
                        ),
                    ),
                    documents,
                )
                nodes += right_added_nodes
        else:
            (
                most_common_keyword,
                idf,
                covered_documents,
                covered_document_idx,
                not_covered_documents,
                not_covered_document_idx,
            ) = self.cover_documents_with_most_common_keyword(documents)

            # For the intermediate node
            # left is always a keyword node
            root.left, left_added_nodes = self.make_tree(
                Node(
                    keyword=most_common_keyword,
                    idf=idf,
                    is_intermediate=False,
                    covered_document_idx=covered_document_idx,
                ),
                covered_documents,
            )
            nodes += left_added_nodes

            # right is an intermediate node if there are uncovered documents
            # otherwise it is None
            if not_covered_document_idx:
                root.right, right_added_nodes = self.make_tree(
                    Node(
                        keyword=None,
                        idf=0,
                        is_intermediate=True,
                        covered_document_idx=not_covered_document_idx,
                    ),
                    not_covered_documents,
                )
                nodes += right_added_nodes

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
            if node.is_root:
                shape = "diamond"
            elif node.is_intermediate:
                shape = "oval"
            if not node.is_intermediate:
                if node.keyword is None:
                    label = '"$"'
                else:
                    label = f'"{node.keyword}"'
            else:
                label = '""'
            graph.append(f"node{idx} [shape={shape}, label={label}];")
        for idx, node in enumerate(self.nodes):
            connected = []
            if node.left is not None:
                connected.append(f"node{self.nodes.index(node.left)}")
            if node.right is not None:
                connected.append(f"node{self.nodes.index(node.right)}")
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
