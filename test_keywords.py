import os
import nltk
import torch as t
from encoder.utils.settings import dataset_cache_dir
from encoder.model.tree.model import KeywordTree
from nltk.stem import WordNetLemmatizer


def generate_keywords_list():
    openbook_qa_path = os.path.join(
        dataset_cache_dir,
        "openbook_qa",
        "OpenBookQA-V1-Sep2018",
        "Data",
        "Main",
        "openbook.txt",
    )
    keywords_list = []
    wnl = WordNetLemmatizer()
    with open(openbook_qa_path, "r") as file:
        for line in file:
            fact = line.strip("\n").strip(".").strip('"').strip("'").strip(",")
            tokens = nltk.word_tokenize(fact.lower())
            allowed_tokens = []
            tagged = nltk.pos_tag(tokens)
            for token, pos in tagged:
                if pos.startswith("NN"):
                    allowed_tokens.append(wnl.lemmatize(token))
            if len(allowed_tokens) < 3:
                for token, pos in tagged:
                    if pos.startswith("JJ"):
                        allowed_tokens.append(wnl.lemmatize(token))
            keywords_list.append(sorted(list(set(allowed_tokens))))
    return keywords_list


if __name__ == "__main__":
    tree = KeywordTree(generate_keywords_list(), 512)
    # tree.visualize_tree()
    # tree(
    #     t.rand(2, 2, 512),
    #     [
    #         ["cycle", "earth", "source", "energy", "sun"],
    #         ["animal", "food", "health", "impact"],
    #     ],
    # )

    print(tree.predict(t.rand(2, 2, 512)))
