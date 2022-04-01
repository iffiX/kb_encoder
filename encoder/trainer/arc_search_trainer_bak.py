import copy
import itertools
import warnings
import torch as t
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.distributed import all_gather_object, get_world_size, get_rank
from transformers import AutoTokenizer, BatchEncoding
from encoder.model.model import Model
from encoder.dataset.base import collate_function_dict_to_batch_encoding
from encoder.dataset.arc_search import ARCSearchDataset
from encoder.utils.config import ARCSearchTrainConfig, fix_missing
from encoder.utils.settings import (
    proxies,
    model_cache_dir,
    huggingface_mirror,
    local_files_only,
)
from encoder.utils.adafactor import Adafactor
from .utils import (
    collate_and_filter_outputs,
    set_worker_sharing_strategy,
    make_scheduler,
)


class ARCSearchTrainer(pl.LightningModule):
    def __init__(
        self,
        config: ARCSearchTrainConfig,
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
            config.base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            local_files_only=local_files_only,
        )

        self.dataset = ARCSearchDataset(
            tokenizer=self.tokenizer,
            search_negative_samples=config.search_negative_samples,
            max_seq_length=config.max_seq_length,
        )
        model_configs = config.model_configs or {}
        self.model = Model(
            config.base_type, config.search_negative_samples + 1, **model_configs
        )
        self.original_state_dict = copy.deepcopy(self.model.state_dict())
        self.best_accuracy = 0

    @property
    def monitor(self):
        return "accuracy"

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
        validate_loader = DataLoader(
            dataset=self.dataset.validate_dataset,
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
            batch_size=1,
            collate_fn=collate_function_dict_to_batch_encoding,
            worker_init_fn=set_worker_sharing_strategy,
        )
        unannotated_train_loader = DataLoader(
            dataset=self.dataset.unannotated_train_dataset,
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
            batch_size=1,
            collate_fn=collate_function_dict_to_batch_encoding,
            worker_init_fn=set_worker_sharing_strategy,
        )
        search_loader = DataLoader(
            dataset=self.dataset.search_dataset,
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
            batch_size=1,
            collate_fn=collate_function_dict_to_batch_encoding,
            worker_init_fn=set_worker_sharing_strategy,
        )
        return [validate_loader, unannotated_train_loader, search_loader]

    # noinspection PyTypeChecker
    def training_step(self, batch: BatchEncoding, batch_idx):
        # answer shape [batch_size, sequence_length]
        out = self.model(
            input_ids=batch["sentence"].to(self.device),
            attention_mask=batch["mask"].to(self.device),
            token_type_ids=batch["type_ids"].to(self.device),
            labels=batch["answer"].to(self.device),
        )
        self.log("search_loss", out.loss.item(), prog_bar=True, on_step=True)
        return out.loss

    # noinspection PyTypeChecker
    def validation_step(self, batch: BatchEncoding, batch_idx, dataloader_idx):
        if dataloader_idx == 1:
            if (
                self.current_epoch + 1
            ) % self.config.epochs_per_self_learn == 0 and not self.is_restored:
                if batch_idx == 0:
                    print("\nAnnotating remaining train dataset")
            else:
                if batch_idx == 0:
                    print("\nSkipping annotation")
                return None

        choice_num = self.model.choice_predictors["default"].choice_num
        self.model.choice_predictors["default"].choice_num = 1
        logits_list = []
        for b in range(0, batch["sentence"].shape[0]):
            sub_logits_list = []
            for i in range(0, batch["sentence"].shape[1], 32):
                sub_logits = self.model.predict(
                    input_ids=batch["sentence"][b : b + 1, i : i + 32].to(self.device),
                    attention_mask=batch["mask"][b : b + 1, i : i + 32].to(self.device),
                    token_type_ids=batch["type_ids"][b : b + 1, i : i + 32].to(
                        self.device
                    ),
                )
                sub_logits_list.append(sub_logits.view(1, -1))
            logits_list.append(t.cat(sub_logits_list, dim=1))
        logits = t.cat(logits_list)
        choice = t.argmax(logits, dim=1).item()
        is_confident = 1 if t.softmax(logits, dim=1)[0, choice].item() > 0.5 else 0
        self.model.choice_predictors["default"].choice_num = choice_num
        return {"batch": batch, "result": t.LongTensor([[choice, is_confident]])}

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
        validate_batch, validate_result = collate_and_filter_outputs(outputs[0])
        metrics = self.dataset.validate_search(validate_batch, validate_result[:, 0])
        for key, value in metrics.items():
            self.log(key, value, prog_bar=True, sync_dist=True)
        if not self.is_distributed or get_rank() == 0:
            print("Validation result:")
            for key, value in metrics.items():
                print(f"{key}: {value}")

        if len(outputs[1]) > 0:
            (annotate_batch, annotate_result) = collate_and_filter_outputs(outputs[1])
            self.dataset.annotate_train_data(
                annotate_batch, annotate_result[:, 0], annotate_result[:, 1]
            )

        if metrics["accuracy"] > self.best_accuracy:
            self.best_accuracy = metrics["accuracy"]
            search_batch, search_result = collate_and_filter_outputs(outputs[2])
            if not self.is_distributed or get_rank() == 0:
                self.dataset.save_search_targets(search_result[:, 0])

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
        training_steps = (
            len(self.dataset.train_dataset)
            * self.config.epochs_per_self_learn
            // (self.config.batch_size * self.config.accumulate_grad_batches)
        )
        sch = make_scheduler(
            optim,
            self.config.scheduler_warmup_proportion,
            training_steps,
            self.config.scheduler_cycles,
            allow_continue=True,
        )
        return (
            [optim],
            [
                {
                    # REQUIRED: The scheduler instance
                    "scheduler": sch,
                    "interval": "step",
                    "monitor": self.monitor,
                }
            ],
        )

    def on_load_checkpoint(self, _checkpoint):
        self.is_restored = True
