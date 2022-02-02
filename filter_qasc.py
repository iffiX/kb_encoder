import os
import json
import logging
import multiprocessing
import nltk
import numpy as np
from typing import List
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from multiprocessing import cpu_count
from encoder.utils.file import decompress_zip, decompress_tar_gz, download_to
from encoder.utils.settings import dataset_cache_dir, preprocess_cache_dir

TOP_K = 10
MIN_SCORE = 0.35
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

TOKEN_INDEX_DICT = None  # type: dict
TOKEN_OCCURRENCE_DICT = None  # type: dict
# first key is token number in obqa, value is a list of qasc indexes of lines with this token
CORPUS_SIZE = None  # type: int
CORPUS_RAW_LENGTH = None  # type: List[int]


def prepare_openbook_qa():
    OPENBOOK_QA_URL = (
        "https://ai2-public-datasets.s3.amazonaws.com/open-book-qa/"
        "OpenBookQA-V1-Sep2018.zip"
    )
    openbook_qa_path = str(os.path.join(dataset_cache_dir, "openbook_qa"))
    if not os.path.exists(openbook_qa_path):
        if not os.path.exists(str(openbook_qa_path) + ".zip"):
            logging.info("Downloading OpenBook QA")
            download_to(OPENBOOK_QA_URL, str(openbook_qa_path) + ".zip")
        logging.info("Decompressing")
        decompress_zip(str(openbook_qa_path) + ".zip", openbook_qa_path)


def prepare_qasc():
    QASC_URL = (
        "https://s3-us-west-2.amazonaws.com/data.allenai.org/downloads/"
        "qasc/qasc_corpus.tar.gz"
    )
    qasc_corpus_path = str(os.path.join(dataset_cache_dir, "qasc_corpus"))
    if not os.path.exists(qasc_corpus_path):
        if not os.path.exists(str(qasc_corpus_path) + ".tar.gz"):
            logging.info("Downloading QASC")
            download_to(QASC_URL, str(qasc_corpus_path) + ".tar.gz")
        logging.info("Decompressing")
        decompress_tar_gz(str(qasc_corpus_path) + ".tar.gz", qasc_corpus_path)


def sample_to_line(sample):
    sample = json.loads(sample)
    line = f"{sample['question']['stem']}, {', '.join(ch['text'] for ch in sample['question']['choices'])}"
    return line


def line_to_valid_tokens(line):
    tokens = list(wnl.lemmatize(x).lower() for x in nltk.word_tokenize(line))
    valid_tokens = []
    for token, pos in nltk.pos_tag(tokens):
        if (
            pos.startswith("NN")
            or pos.startswith("JJ")
            or pos.startswith("RB")
            or (pos.startswith("VB") and token.lower() not in VERB_FILTER_SET)
            or pos.startswith("CD")
        ):
            valid_tokens.append(token)
    return valid_tokens, line


def line_to_token_index_set(line):
    tokens = list(wnl.lemmatize(x).lower() for x in nltk.word_tokenize(line))
    return (
        len(tokens),
        {TOKEN_INDEX_DICT[token] for token in tokens if token in TOKEN_INDEX_DICT},
    )


def token_index_set_to_token_occurrence_dict(token_dict_size, token_index_set):
    token_occurrence_dict = {i: [] for i in range(token_dict_size)}
    for corpus_index, ti in enumerate(token_index_set):
        for tti in ti:
            token_occurrence_dict[tti].append(corpus_index)
    return token_occurrence_dict


def filter_top_K_knowledge(token_indexes):
    occurrence = np.zeros([CORPUS_SIZE], dtype=np.int32)
    for ti in token_indexes:
        occurrence[TOKEN_OCCURRENCE_DICT[ti]] += 1
    # F1 score
    precision = occurrence.astype(np.float32) / np.array(
        CORPUS_RAW_LENGTH, dtype=np.float32
    )
    recall = occurrence.astype(np.float32) / len(token_indexes)
    score = precision * recall * 5 / (4 * precision + recall + 1e-4)
    corpus_indexes = np.argpartition(score, -TOP_K)[-TOP_K:]
    ordered_indexes = corpus_indexes[np.argsort(-score[corpus_indexes])]
    ordered_scores = score[ordered_indexes]
    ordered_indexes = ordered_indexes[ordered_scores > MIN_SCORE]
    ordered_scores = ordered_scores[ordered_scores > MIN_SCORE]
    return ordered_indexes.tolist(), ordered_scores.tolist()


def init_global_variable(name, value):
    if isinstance(name, str):
        globals()[name] = value
    else:
        for n, v in zip(name, value):
            globals()[n] = v


if __name__ == "__main__":
    prepare_openbook_qa()
    prepare_qasc()

    openbook_qa_path = os.path.join(dataset_cache_dir, "openbook_qa")
    qasc_corpus_path = os.path.join(dataset_cache_dir, "qasc_corpus")
    # Note: fact is not directly used in train/test/validation
    train_path = os.path.join(
        openbook_qa_path,
        "OpenBookQA-V1-Sep2018",
        "Data",
        "Additional",
        "train_complete.jsonl",
    )
    validate_path = os.path.join(
        openbook_qa_path,
        "OpenBookQA-V1-Sep2018",
        "Data",
        "Additional",
        "dev_complete.jsonl",
    )
    test_path = os.path.join(
        openbook_qa_path,
        "OpenBookQA-V1-Sep2018",
        "Data",
        "Additional",
        "test_complete.jsonl",
    )

    wnl = WordNetLemmatizer()

    obqa_corpus = []
    logging.info("Reading OpenBook QA corpus")
    for path in (train_path, validate_path, test_path):
        num_lines = sum(1 for line in open(path, "r"))
        with open(path, "r") as file:
            for i, sample in enumerate(tqdm(file, total=num_lines)):
                line = sample_to_line(sample)
                obqa_corpus.append(line_to_valid_tokens(line))

    token_list = sorted(
        list({valid_token for sample in obqa_corpus for valid_token in sample[0]})
    )
    token_index_dict = {token: i for i, token in enumerate(token_list)}
    obqa_token_indexes = [
        [token_index_dict[valid_token] for valid_token in sample[0]]
        for sample in obqa_corpus
    ]  # List[List[int]]

    logging.info("Reading QASC corpus")
    path = os.path.join(qasc_corpus_path, "QASC_Corpus", "QASC_Corpus.txt")
    with open(path, "r") as file:
        qasc_raw = list(file)

    with multiprocessing.Pool(
        processes=cpu_count() - 1,
        initializer=init_global_variable,
        initargs=("TOKEN_INDEX_DICT", token_index_dict),
    ) as pool:
        result = list(
            tqdm(
                pool.imap(line_to_token_index_set, qasc_raw, chunksize=256),
                total=len(qasc_raw),
            )
        )
        qasc_token_index_set = [r[1] for r in result]
        qasc_raw_length = [r[0] for r in result]

    token_occurrence_dict = token_index_set_to_token_occurrence_dict(
        len(token_index_dict), qasc_token_index_set
    )

    with multiprocessing.Pool(
        processes=cpu_count() - 1,
        initializer=init_global_variable,
        initargs=(
            ("TOKEN_OCCURRENCE_DICT", "CORPUS_SIZE", "CORPUS_RAW_LENGTH"),
            (token_occurrence_dict, len(qasc_token_index_set), qasc_raw_length),
        ),
    ) as pool:
        result = list(
            tqdm(
                pool.imap(filter_top_K_knowledge, obqa_token_indexes),
                total=len(obqa_token_indexes),
            )
        )

    with open(os.path.join(preprocess_cache_dir, "qasc_additional.txt"), "w") as file:
        for top_k in result:
            for index in top_k[0]:
                file.write(f"{qasc_raw[index]}")

    with open(os.path.join(preprocess_cache_dir, "qasc_debug.txt"), "w") as file:
        for obqa_sample, top_k in zip(obqa_corpus, result):
            file.write(f"sample: {obqa_sample[1]}, tokens: {obqa_sample[0]}\n")
            for i, (index, score) in enumerate(zip(top_k[0], top_k[1])):
                file.write(f"{i}. {qasc_raw[index]} score={score}\n")
            file.write("\n")
