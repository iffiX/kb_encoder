import itertools
import warnings
import torch as t
import pytorch_lightning as pl
from typing import Dict
from torch.utils.data import DataLoader
from torch.distributed import all_gather_object, get_world_size, get_rank
from transformers import T5ForConditionalGeneration, T5TokenizerFast, BatchEncoding
from pytorch_lightning.utilities import rank_zero_only
from .utils import collate_and_filter_outputs, set_worker_sharing_strategy
from .openbook_qa_trainer import OpenBookQATrainer
from encoder.dataset.base import collate_function_dict_to_batch_encoding
from encoder.dataset.openbook_qa_with_search import OpenBookQAWithSearchDataset
from encoder.utils.config import OpenBookQAWithSearchTrainConfig, fix_missing
from encoder.utils.settings import proxies, model_cache_dir, huggingface_mirror
from encoder.utils.adafactor import Adafactor


class OpenBookQAWithSearchTrainer(pl.LightningModule):
    def __init__(
        self,
        config: OpenBookQAWithSearchTrainConfig,
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
        self.dataset = OpenBookQAWithSearchDataset(
            tokenizer=self.tokenizer,
            max_seq_length=config.max_seq_length,
            generate_length=config.generate_length,
            use_matcher=config.use_matcher,
            matcher_mode=config.matcher_mode,
            matcher_seed=config.seed,
            matcher_config=config.matcher_config,
            include_option_label_in_sentence=config.include_option_label_in_sentence,
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
        self.qa_model = OpenBookQATrainer.load_from_checkpoint(
            config.qa_checkpoint_path
        ).model
        self.real_device = None

    @property
    def monitor(self):
        return "qa_accuracy"

    @property
    def monitor_mode(self):
        return "max"

    def train_dataloader(self):
        search_loader = DataLoader(
            dataset=self.dataset.train_search_dataset,
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
            batch_size=self.config.batch_size,
            collate_fn=collate_function_dict_to_batch_encoding,
            worker_init_fn=set_worker_sharing_strategy,
        )
        return search_loader

    def val_dataloader(self):
        qa_loader = DataLoader(
            dataset=self.dataset.validate_qa_dataset,
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
            batch_size=self.config.batch_size,
            collate_fn=collate_function_dict_to_batch_encoding,
            worker_init_fn=set_worker_sharing_strategy,
        )
        search_loader = DataLoader(
            dataset=self.dataset.validate_search_dataset,
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
            batch_size=self.config.batch_size,
            collate_fn=collate_function_dict_to_batch_encoding,
            worker_init_fn=set_worker_sharing_strategy,
        )
        return [search_loader, qa_loader]

    def test_dataloader(self):
        qa_loader = DataLoader(
            dataset=self.dataset.test_qa_dataset,
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
            batch_size=self.config.batch_size,
            collate_fn=collate_function_dict_to_batch_encoding,
            worker_init_fn=set_worker_sharing_strategy,
        )
        search_loader = DataLoader(
            dataset=self.dataset.test_search_dataset,
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
            batch_size=self.config.batch_size,
            collate_fn=collate_function_dict_to_batch_encoding,
            worker_init_fn=set_worker_sharing_strategy,
        )
        return [search_loader, qa_loader]

    def setup(self, stage=None):
        print("Updating qa checkpoint")
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
        self.log("search_loss", out.loss, prog_bar=True, on_step=True)
        return out.loss

    # noinspection PyTypeChecker
    def validation_step(self, batch: BatchEncoding, _batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            out = self.model.generate(
                batch["sentence"].to(self.real_device or self.device),
                max_length=self.config.generate_length,
                attention_mask=batch["mask"].to(self.real_device or self.device),
                early_stopping=True,
                num_beams=5,
            )
        else:
            out = self.qa_model.generate(
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
        if dataloader_idx == 0:
            for res, id in zip(result, batch["id"]):
                self.dataset.set_search_target(res, "validate", id)
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
        search_batch, search_tokens = collate_and_filter_outputs(outputs[0])
        search_metrics = self.dataset.validate_search(search_batch, search_tokens)
        qa_batch, qa_tokens = collate_and_filter_outputs(outputs[1])
        qa_metrics = self.dataset.validate_qa(qa_batch, qa_tokens)

        for key, value in search_metrics.items():
            self.log("search_" + key, value, prog_bar=True, sync_dist=True)
        for key, value in qa_metrics.items():
            self.log("qa_" + key, value, prog_bar=True, sync_dist=True)
        if not self.is_distributed or get_rank() == 0:
            print("Validation result:")
            for key, value in search_metrics.items():
                print(f"search_{key}: {value}")
            for key, value in qa_metrics.items():
                print(f"qa_{key}: {value}")

    def configure_optimizers(self):
        if self.config.optimizer_class == "Adafactor":
            optim = Adafactor(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.l2_regularization,
            )
        else:
            optim_cls = getattr(t.optim, self.config.optimizer_class)
            optim = optim_cls(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.l2_regularization,
            )
        return optim
        # sch = t.optim.lr_scheduler.ReduceLROnPlateau(
        #     optim, mode="max", factor=0.3, patience=0, min_lr=3e-5, verbose=True
        # )
        # return (
        #     [optim],
        #     [
        #         {
        #             # REQUIRED: The scheduler instance
        #             "scheduler": sch,
        #             "monitor": "qa_accuracy",
        #         }
        #     ],
        # )
