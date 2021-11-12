import itertools
import warnings
import torch as t
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.distributed import all_gather_object, get_world_size, get_rank
from transformers import T5ForConditionalGeneration, T5TokenizerFast, BatchEncoding
from ..dataset.base import collate_function_dict_to_batch_encoding, dict_iter
from ..dataset.c4kb import C4KBDataset
from ..utils.config import C4KBTrainConfig
from ..utils.settings import proxies, model_cache_dir, huggingface_mirror
from ..utils.adafactor import Adafactor


class C4KBTrainer(pl.LightningModule):
    def __init__(
        self, config: C4KBTrainConfig, stage_result_path="./", is_distributed=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        warnings.filterwarnings("ignore")

        self.config = config
        self.stage_result_path = stage_result_path
        self.is_distributed = is_distributed

        self.model = T5ForConditionalGeneration.from_pretrained(
            config.base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
            return_dict=True,
        )

        self.tokenizer = T5TokenizerFast.from_pretrained(
            config.base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
        )

        self.dataset = C4KBDataset(
            tokenizer=self.tokenizer,
            matcher_max_times=config.matcher_max_times,
            matcher_max_depth=config.matcher_max_depth,
            matcher_max_edges=config.matcher_max_edges,
            matcher_seed=config.seed,
            c4_seed=config.seed + 2628,
            max_seq_length=config.max_seq_length,
        )

    @property
    def monitor(self):
        return "EM"

    @property
    def monitor_mode(self):
        return "max"

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset.train_dataset,
            batch_size=self.config.batch_size,
            collate_fn=collate_function_dict_to_batch_encoding,
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset.validate_dataset,
            batch_size=self.config.batch_size,
            collate_fn=collate_function_dict_to_batch_encoding,
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
        )

    def on_train_start(self):
        if self.config.device_map is not None:
            if self.is_distributed:
                raise ValueError(
                    "Parallelize T5 model is incompatible with distributed training."
                )
            self.model.parallelize(self.config.device_map)

    # noinspection PyTypeChecker
    def training_step(self, batch: BatchEncoding, batch_idx):
        input_ids = batch["sentence"].to(self.device)
        out = self.model(
            input_ids=input_ids, attention_mask=batch["mask"], labels=batch["answer"],
        )
        return out.loss

        # noinspection PyTypeChecker

    def validation_step(self, batch: BatchEncoding, _batch_idx):
        out = self.model.generate(
            batch["sentence"].to(self.device),
            max_length=self.config.max_seq_length,
            attention_mask=batch["mask"],
            early_stopping=True,
        )
        result = t.full(
            [out.shape[0], self.config.max_seq_length], self.tokenizer.pad_token_id
        )
        result[:, : out.shape[1]] = out.cpu()
        batch = batch.to("cpu")
        return {
            "batch": batch,
            "tokens": result,
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
        batch, tokens = self.collate_and_filter_outputs(outputs)
        metrics = self.dataset.validate(batch, tokens)
        for key, value in metrics.items():
            self.log(key, value, prog_bar=True, sync_dist=True)
        if not self.is_distributed or get_rank() == 0:
            print("Validation result:")
            for key, value in metrics.items():
                print(f"{key}: {value}")

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
        return optim

    @staticmethod
    def collate_and_filter_outputs(outputs):
        batch = collate_function_dict_to_batch_encoding([o["batch"] for o in outputs])
        tokens = t.cat([o["tokens"] for o in outputs], dim=0)
        list_of_results = [
            (b["id"][0], b, to.unsqueeze(0)) for b, to in zip(dict_iter(batch), tokens)
        ]
        # filter duplicates brought by resetting dataset
        existed = {}
        filtered = []
        for lr in list_of_results:
            if lr[0] not in existed:
                filtered.append(lr)
                existed[lr[0]] = True
        list_of_results = filtered
        tokens = t.cat([lr[2] for lr in list_of_results], dim=0)
        batch = collate_function_dict_to_batch_encoding(
            [lr[1] for lr in list_of_results]
        )
        return batch, tokens
