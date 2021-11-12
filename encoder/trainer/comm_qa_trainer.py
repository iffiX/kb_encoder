import itertools
import warnings
import torch as t
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.distributed import all_gather_object, get_world_size, get_rank
from transformers import T5ForConditionalGeneration, T5TokenizerFast, BatchEncoding
from pytorch_lightning.utilities import rank_zero_only
from ..dataset.base import collate_function_dict_to_batch_encoding, dict_iter
from encoder.dataset.commonsense_qa import CommonsenseQADataset
from ..utils.config import CommonsenseQATrainConfig
from ..utils.settings import proxies, model_cache_dir, huggingface_mirror
from ..utils.adafactor import Adafactor


def set_worker_sharing_strategy(worker_id: int) -> None:
    t.multiprocessing.set_sharing_strategy("file_system")


class CommonsenseQATrainer(pl.LightningModule):
    def __init__(
        self,
        config: CommonsenseQATrainConfig,
        stage_result_path="./",
        is_distributed=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        warnings.filterwarnings("ignore")

        self.config = config
        self.stage_result_path = stage_result_path
        self.is_distributed = is_distributed

        self.tokenizer = T5TokenizerFast.from_pretrained(
            config.base_type,
            cache_dir=model_cache_dir,
            proxies=proxies,
            mirror=huggingface_mirror,
        )
        self.dataset = CommonsenseQADataset(
            tokenizer=self.tokenizer,
            max_seq_length=config.max_seq_length,
            generate_length=config.generate_length,
            use_matcher=config.use_matcher,
            matcher_seed=config.seed,
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
        self.real_device = "cpu"

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
        return DataLoader(
            dataset=self.dataset.validate_dataset,
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
            batch_size=self.config.batch_size,
            collate_fn=collate_function_dict_to_batch_encoding,
            worker_init_fn=set_worker_sharing_strategy,
        )

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
        input_ids = batch["sentence"].to(self.real_device or self.device)
        out = self.model(
            input_ids=input_ids,
            attention_mask=batch["mask"].to(self.real_device or self.device),
            labels=batch["answer"].to(self.real_device or self.device),
        )
        # for i in range(out.logits.shape[0]):
        #     print(self.tokenizer.decode(input_ids[i]))
        #     ref_answer_tensor = batch["answer"][i]
        #     ref_answer_tensor.masked_fill_(
        #         ref_answer_tensor == -100, self.tokenizer.pad_token_id
        #     )
        #     print(self.tokenizer.decode(ref_answer_tensor))
        #     print(self.tokenizer.decode(t.argmax(out.logits[i], dim=-1)))
        #     print()
        return out.loss

    # noinspection PyTypeChecker
    def validation_step(self, batch: BatchEncoding, _batch_idx):
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
        batch, tokens = self.collate_and_filter_outputs(outputs)
        metrics = self.dataset.validate_tokens(batch, tokens)
        for key, value in metrics.items():
            self.log(key, value, prog_bar=True, sync_dist=True)
        if not self.is_distributed or get_rank() == 0:
            print("Validation result:")
            for key, value in metrics.items():
                print(f"{key}: {value}")

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
        _, tokens = self.collate_and_filter_outputs(outputs)
        self.dataset.generate_test_results_tokens(tokens, self.stage_result_path)

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
            optim, mode="max", factor=0.3, patience=0, min_lr=1e-6, verbose=True
        )
        return (
            [optim],
            [
                {
                    # REQUIRED: The scheduler instance
                    "scheduler": sch,
                    "monitor": "accuracy",
                }
            ],
        )

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
