from transformers import PreTrainedTokenizerBase, BatchEncoding
from encoder.dataset.concept_net import ConceptNetMatcher
from encoder.utils.file import open_file_with_create_directories
from encoder.utils.settings import (
    dataset_cache_dir,
    metrics_cache_dir,
    preprocess_cache_dir,
    proxies,
)


class C4KBDataset:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int = 128,
        max_train_samples: int = None,
        max_validate_samples: int = None,
    ):
        self.tokenizer = tokenizer
