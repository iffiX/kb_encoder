import itertools
import warnings
import torch as t
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.distributed import all_gather_object, get_world_size, get_rank
from transformers import T5ForConditionalGeneration, T5TokenizerFast, BatchEncoding
from pytorch_lightning.utilities import rank_zero_only
from .utils import collate_and_filter_outputs, set_worker_sharing_strategy
from encoder.dataset.base import collate_function_dict_to_batch_encoding
from encoder.dataset.openbook_qa import OpenBookQADataset
from encoder.utils.config import OpenBookQATrainConfig, fix_missing
from encoder.utils.settings import proxies, model_cache_dir, huggingface_mirror
from encoder.utils.adafactor import Adafactor


class OpenBookQATrainer(pl.LightningModule):
    def __init__(
        self,
        config: OpenBookQATrainConfig,
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

        self.tokenizer = T5TokenizerFast.from_pretrained(
            config.base_type if config.base_type.startswith("t5") else "t5-base",
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
        )
        self.dataset = OpenBookQADataset(
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
        self.model = T5ForConditionalGeneration.from_pretrained(
            config.base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            return_dict=True,
        )
        self.real_device = None

    @property
    def monitor(self):
        return "test_accuracy"

    @property
    def monitor_mode(self):
        return "max"

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

    def on_fit_start(self):
        if self.config.device_map is not None:
            if self.is_distributed:
                raise ValueError(
                    "Parallelize T5 model is incompatible with distributed training."
                )
            start_device_id = [k for k, v in self.config.device_map.items() if 0 in v][
                0
            ]
            # replace device property
            self.real_device = f"cuda:{start_device_id}"
            self.model.parallelize(self.config.device_map)
        else:
            self.real_device = None

    # noinspection PyTypeChecker
    def training_step(self, batch: BatchEncoding, batch_idx):
        # answer shape [batch_size, sequence_length]
        out = self.model(
            input_ids=batch["sentence"].to(self.real_device or self.device),
            attention_mask=batch["mask"].to(self.real_device or self.device),
            labels=batch["answer"].to(self.real_device or self.device),
        )
        return out.loss

    # noinspection PyTypeChecker
    def validation_step(self, batch: BatchEncoding, _batch_idx, _dataloader_idx):
        out = self.model.generate(
            batch["sentence"].to(self.real_device or self.device),
            max_length=self.config.generate_length,
            attention_mask=batch["mask"].to(self.real_device or self.device),
            early_stopping=True,
        )
        result = t.full(
            [out.shape[0], self.config.generate_length], self.tokenizer.pad_token_id
        )
        result[:, : out.shape[1]] = out.cpu()
        batch = batch.to("cpu")
        return {
            "batch": batch,
            "tokens": result,
        }

    def validation_epoch_end(self, outputs):
        if self.is_distributed:
            t.cuda.set_device(self.real_device or self.device)
            gathered_outputs = [None] * get_world_size()
            all_gather_object(gathered_outputs, outputs)
            gathered_outputs = list(itertools.chain.from_iterable(gathered_outputs))
            self.validate_on_every_process(gathered_outputs)
        else:
            self.validate_on_every_process(outputs)

    def validate_on_every_process(self, outputs):
        for prefix, dataloader_idx in (("val", 0), ("test", 1)):
            batch, tokens = collate_and_filter_outputs(outputs[dataloader_idx])
            metrics = self.dataset.validate(batch, tokens)
            for key, value in metrics.items():
                self.log(f"{prefix}_{key}", value, prog_bar=True, sync_dist=True)
            if not self.is_distributed or get_rank() == 0:
                print(f"Validation on {prefix} result:")
                for key, value in metrics.items():
                    print(f"{prefix}_{key}: {value}")

    def test_step(self, batch: BatchEncoding, _batch_idx):
        out = self.model.generate(
            batch["sentence"].to(self.real_device or self.device),
            max_length=self.config.generate_length,
            attention_mask=batch["mask"].to(self.real_device or self.device),
            early_stopping=True,
        )
        result = t.full(
            [out.shape[0], self.config.generate_length], self.tokenizer.pad_token_id
        )
        result[:, : out.shape[1]] = out.cpu()
        batch = batch.to("cpu")
        return {
            "batch": batch,
            "tokens": result,
        }

    def test_epoch_end(self, outputs):
        if self.is_distributed:
            t.cuda.set_device(self.real_device or self.device)
            gathered_outputs = [None] * get_world_size()
            all_gather_object(gathered_outputs, outputs)
            gathered_outputs = list(itertools.chain.from_iterable(gathered_outputs))
            self.test_on_main_process(gathered_outputs)
        else:
            self.test_on_main_process(outputs)

    @rank_zero_only
    def test_on_main_process(self, outputs):
        _, tokens = collate_and_filter_outputs(outputs)
        self.dataset.generate_test_results(tokens, self.stage_result_path)

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
        sch = t.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="max", factor=0.3, patience=0, min_lr=3e-5, verbose=True
        )
        return (
            [optim],
            [
                {
                    # REQUIRED: The scheduler instance
                    "scheduler": sch,
                    "monitor": "test_accuracy",
                }
            ],
        )
