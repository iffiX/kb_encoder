import itertools
import warnings
import torch as t
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.distributed import all_gather_object, get_world_size, get_rank
from transformers import (
    AutoTokenizer,
    BatchEncoding,
)
from pytorch_lightning.utilities import rank_zero_only
from .utils import (
    collate_and_filter_outputs,
    set_worker_sharing_strategy,
    make_scheduler,
)
from encoder.model.model import ChainedSequenceModel
from encoder.dataset.base import collate_function_dict_to_batch_encoding
from encoder.dataset.openbook_qa_chained_sequence import (
    OpenBookQAChainedSequenceDataset,
)
from encoder.utils.config import OpenBookQAChainedSequenceTrainConfig, fix_missing
from encoder.utils.settings import (
    proxies,
    model_cache_dir,
    huggingface_mirror,
    local_files_only,
)
from encoder.utils.adafactor import Adafactor


class OpenBookQAChainedSequenceTrainer(pl.LightningModule):
    def __init__(
        self,
        config: OpenBookQAChainedSequenceTrainConfig,
        stage_result_path="./",
        is_distributed=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        warnings.filterwarnings("ignore")

        fix_missing(config)
        self.config = config
        self.stage_result_path = stage_result_path
        self.is_distributed = is_distributed

        self.tokenizer = AutoTokenizer.from_pretrained(
            "t5-base" if config.base_type.startswith("t5") else config.base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            local_files_only=local_files_only,
        )
        self.dataset = OpenBookQAChainedSequenceDataset(
            tokenizer=self.tokenizer,
            max_seq_length=config.max_seq_length,
            generate_length=config.generate_length,
            use_matcher=config.use_matcher,
            matcher_mode=config.matcher_mode,
            matcher_seed=config.seed,
            matcher_config=config.matcher_config,
            include_option_label_in_sentence=config.include_option_label_in_sentence,
            include_option_label_in_answer_and_choices=config.include_option_label_in_answer_and_choices,
            use_option_label_as_answer_and_choices=config.use_option_label_as_answer_and_choices,
            match_closest_when_no_equal=config.match_closest_when_no_equal,
        )

        model_configs = config.model_configs or {}
        self.model = ChainedSequenceModel(config.base_type, 4, **model_configs)
        self._real_device = None

    @property
    def monitor(self):
        return "test_accuracy"

    @property
    def monitor_mode(self):
        return "max"

    @property
    def real_device(self):
        return self._real_device or self.device

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset.train_dataset,
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
            batch_size=self.config.batch_size,
            collate_fn=collate_function_dict_to_batch_encoding,
            worker_init_fn=set_worker_sharing_strategy,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                dataset=self.dataset.validate_dataset,
                num_workers=self.config.load_worker_num,
                prefetch_factor=self.config.load_prefetch_per_worker,
                batch_size=self.config.batch_size,
                collate_fn=collate_function_dict_to_batch_encoding,
                worker_init_fn=set_worker_sharing_strategy,
            ),
            DataLoader(
                dataset=self.dataset.test_dataset,
                num_workers=self.config.load_worker_num,
                prefetch_factor=self.config.load_prefetch_per_worker,
                batch_size=self.config.batch_size,
                collate_fn=collate_function_dict_to_batch_encoding,
                worker_init_fn=set_worker_sharing_strategy,
            ),
        ]

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset.test_dataset,
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
            batch_size=self.config.batch_size,
            collate_fn=collate_function_dict_to_batch_encoding,
            worker_init_fn=set_worker_sharing_strategy,
        )

    # noinspection PyTypeChecker
    def training_step(self, batch: BatchEncoding, batch_idx):
        if self.config.base_type.startswith("t5"):
            # answer shape [batch_size, sequence_length]
            out = self.model(
                input_ids=batch["sentence"].to(self.device),
                attention_mask=batch["mask"].to(self.device),
                labels=batch["answer"].to(self.device),
            )
        else:
            out = self.model(
                input_ids=batch["sentence"].to(self.device),
                attention_mask=batch["mask"].to(self.device),
                token_type_ids=batch["type_ids"].to(self.device),
                labels=batch["label"].to(self.device),
            )
        return out.loss

    # noinspection PyTypeChecker
    def validation_step(self, batch: BatchEncoding, _batch_idx, _dataloader_idx):
        return {
            "batch": batch.to("cpu"),
            "result": self.model.predict(
                input_ids=batch["sentence"].to(self.device),
                attention_mask=batch["mask"].to(self.device),
                token_type_ids=batch["type_ids"].to(self.device),
            ).cpu(),
        }

    def validation_epoch_end(self, outputs):
        if self.is_distributed:
            t.cuda.set_device(self.device)
            gathered_outputs = [None] * get_world_size()
            all_gather_object(gathered_outputs, outputs)
            gathered_outputs = list(itertools.chain.from_iterable(gathered_outputs))
            self.validate_on_every_process(gathered_outputs)
        else:
            self.validate_on_every_process(outputs)

    def validate_on_every_process(self, outputs):
        for prefix, dataloader_idx in (("val", 0), ("test", 1)):
            batch, result = collate_and_filter_outputs(outputs[dataloader_idx])
            metrics = self.dataset.validate_logits(batch, result)
            for key, value in metrics.items():
                self.log(f"{prefix}_{key}", value, prog_bar=True, sync_dist=True)
            if not self.is_distributed or get_rank() == 0:
                print(f"Validation on {prefix} result:")
                for key, value in metrics.items():
                    print(f"{prefix}_{key}: {value}")

    def test_step(self, batch: BatchEncoding, _batch_idx):
        return {
            "batch": batch.to("cpu"),
            "result": self.model.predict(
                input_ids=batch["sentence"].to(self.device),
                attention_mask=batch["mask"].to(self.device),
                token_type_ids=batch["type_ids"].to(self.device),
            ).cpu(),
        }

    def test_epoch_end(self, outputs):
        if self.is_distributed:
            t.cuda.set_device(self.device)
            gathered_outputs = [None] * get_world_size()
            all_gather_object(gathered_outputs, outputs)
            gathered_outputs = list(itertools.chain.from_iterable(gathered_outputs))
            self.test_on_main_process(gathered_outputs)
        else:
            self.test_on_main_process(outputs)

    @rank_zero_only
    def test_on_main_process(self, outputs):
        _, result = collate_and_filter_outputs(outputs)
        self.dataset.generate_test_result_logits(result, self.stage_result_path)

    def configure_optimizers(self):
        if self.config.optimizer_class == "Adafactor":
            optim = Adafactor(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.l2_regularization,
            )
        else:
            optim_cls = getattr(t.optim, self.config.optimizer_class)
            optim = optim_cls(
                self.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.l2_regularization,
            )
        training_steps = (
            len(self.dataset.train_dataset)
            * self.config.epochs
            // self.config.accumulate_grad_batches
        )
        sch = make_scheduler(
            optim,
            self.config.scheduler_warmup_proportion,
            training_steps,
            self.config.scheduler_cycles,
        )
        return (
            [optim],
            [
                {
                    # REQUIRED: The scheduler instance
                    "scheduler": sch,
                    "monitor": self.monitor,
                }
            ],
        )
