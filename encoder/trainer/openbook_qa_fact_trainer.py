import itertools
import warnings
import torch as t
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.distributed import all_gather_object, get_world_size, get_rank
from transformers import AutoTokenizer, BatchEncoding
from pytorch_lightning.utilities import rank_zero_only
from .utils import collate_and_filter_outputs, set_worker_sharing_strategy
from .openbook_qa_trainer import OpenBookQATrainer
from encoder.model.deberta.model import Model
from encoder.dataset.base import collate_function_dict_to_batch_encoding
from encoder.dataset.openbook_qa_fact import OpenBookQAFactDataset
from encoder.utils.config import OpenBookQAFactTrainConfig, fix_missing
from encoder.utils.settings import (
    proxies,
    model_cache_dir,
    huggingface_mirror,
    local_files_only,
)
from encoder.utils.adafactor import Adafactor


class OpenBookQAFactTrainer(pl.LightningModule):
    def __init__(
        self,
        config: OpenBookQAFactTrainConfig,
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
        qa_trainer = OpenBookQATrainer.load_from_checkpoint(config.qa_checkpoint_path)
        qa_config = qa_trainer.config
        self.dataset = OpenBookQAFactDataset(
            tokenizer=qa_trainer.tokenizer,
            search_tokenizer=self.tokenizer,
            search_negative_samples=config.search_negative_samples,
            max_seq_length=qa_config.max_seq_length,
            generate_length=qa_config.generate_length,
            use_matcher=qa_config.use_matcher,
            matcher_mode=qa_config.matcher_mode,
            matcher_seed=qa_config.seed,
            matcher_config=qa_config.matcher_config,
            include_option_label_in_sentence=qa_config.include_option_label_in_sentence,
            include_option_label_in_answer_and_choices=qa_config.include_option_label_in_answer_and_choices,
            use_option_label_as_answer_and_choices=qa_config.use_option_label_as_answer_and_choices,
            match_closest_when_no_equal=qa_config.match_closest_when_no_equal,
            output_mode="single"
            if qa_config.base_type.startswith("t5")
            else "splitted",
        )
        model_configs = config.model_configs or {}
        self.model = Model(
            config.base_type, config.search_negative_samples + 1, **model_configs
        )
        self.qa_model_type = qa_config.base_type
        self.qa_model = qa_trainer.model
        self._real_device = None

    @property
    def monitor(self):
        return "test_qa_accuracy"

    @property
    def monitor_mode(self):
        return "max"

    @property
    def real_device(self):
        return self._real_device or self.device

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
        # qa_loader_val = DataLoader(
        #     dataset=self.dataset.validate_dataset,
        #     num_workers=self.config.load_worker_num,
        #     prefetch_factor=self.config.load_prefetch_per_worker,
        #     batch_size=self.config.batch_size,
        #     collate_fn=collate_function_dict_to_batch_encoding,
        #     worker_init_fn=set_worker_sharing_strategy,
        # )
        # search_loader_val = DataLoader(
        #     dataset=self.dataset.validate_search_dataset,
        #     num_workers=self.config.load_worker_num,
        #     prefetch_factor=self.config.load_prefetch_per_worker,
        #     batch_size=self.config.batch_size,
        #     collate_fn=collate_function_dict_to_batch_encoding,
        #     worker_init_fn=set_worker_sharing_strategy,
        # )
        qa_loader_test = DataLoader(
            dataset=self.dataset.test_dataset,
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
            batch_size=self.config.batch_size,
            collate_fn=collate_function_dict_to_batch_encoding,
            worker_init_fn=set_worker_sharing_strategy,
        )
        search_loader_test = DataLoader(
            dataset=self.dataset.test_search_dataset,
            num_workers=self.config.load_worker_num,
            prefetch_factor=self.config.load_prefetch_per_worker,
            batch_size=self.config.batch_size,
            collate_fn=collate_function_dict_to_batch_encoding,
            worker_init_fn=set_worker_sharing_strategy,
        )
        # return [search_loader_val, search_loader_test, qa_loader_val, qa_loader_test]
        return [search_loader_test, qa_loader_test]

    def test_dataloader(self):
        qa_loader = DataLoader(
            dataset=self.dataset.test_dataset,
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
            self._real_device = f"cuda:{start_device_id}"
            self.model.parallelize(self.config.device_map)
        else:
            self._real_device = None

    # noinspection PyTypeChecker
    def training_step(self, batch: BatchEncoding, batch_idx):
        # answer shape [batch_size, sequence_length]
        out = self.model(
            input_ids=batch["sentence"].to(self.real_device),
            attention_mask=batch["mask"].to(self.real_device),
            token_type_ids=batch["type_ids"].to(self.real_device),
            labels=batch["answer"].to(self.real_device),
        )
        self.log("search_loss", out.loss.item(), prog_bar=True, on_step=True)
        return out.loss

    # noinspection PyTypeChecker
    def validation_step(self, batch: BatchEncoding, _batch_idx, dataloader_idx):
        # if dataloader_idx in (0, 1):
        if dataloader_idx == 0:
            choice_num = self.model.choice_predictors["default"].choice_num
            self.model.choice_predictors["default"].choice_num = 1
            logits_list = []
            for b in range(0, batch["sentence"].shape[0]):
                sub_logits_list = []
                for i in range(0, batch["sentence"].shape[1], 32):
                    logits = self.model.predict(
                        input_ids=batch["sentence"][b : b + 1, i : i + 32].to(
                            self.real_device
                        ),
                        attention_mask=batch["mask"][b : b + 1, i : i + 32].to(
                            self.real_device
                        ),
                        token_type_ids=batch["type_ids"][b : b + 1, i : i + 32].to(
                            self.real_device
                        ),
                    )
                    sub_logits_list.append(logits.view(1, -1))
                logits_list.append(t.cat(sub_logits_list, dim=1))
            choice = t.argmax(t.cat(logits_list), dim=1)
            self.model.choice_predictors["default"].choice_num = choice_num
            for ch, id in zip(choice, batch["id"]):
                # self.dataset.set_search_target(
                #     ch, "test" if dataloader_idx == 1 else "validate", id,
                # )
                self.dataset.set_search_target(ch, "test", id)
            return {
                "batch": batch,
                "result": choice,
            }
        else:
            if self.qa_model_type.startswith("t5"):
                out = self.qa_model.generate(
                    batch["sentence"].to(self.real_device),
                    max_length=self.config.generate_length,
                    attention_mask=batch["mask"].to(self.real_device),
                    early_stopping=True,
                )
                result = t.full(
                    [out.shape[0], self.config.generate_length],
                    self.tokenizer.pad_token_id,
                )
                result[:, : out.shape[1]] = out.cpu()
                batch = batch.to("cpu")
                return {
                    "batch": batch,
                    "result": result,
                }
            else:
                return {
                    "batch": batch.to("cpu"),
                    "result": self.qa_model.predict(
                        input_ids=batch["sentence"].to(self.real_device),
                        attention_mask=batch["mask"].to(self.real_device),
                        token_type_ids=batch["type_ids"].to(self.real_device),
                    ).cpu(),
                }

    def validation_epoch_end(self, outputs):
        if self.is_distributed:
            t.cuda.set_device(self.real_device)
            gathered_outputs = [None] * get_world_size()
            all_gather_object(gathered_outputs, outputs)
            gathered_outputs = list(itertools.chain.from_iterable(gathered_outputs))
            self.validate_on_every_process(gathered_outputs)
        else:
            self.validate_on_every_process(outputs)

    def validate_on_every_process(self, outputs):
        # for prefix, dataloader_idx in (("val", 0), ("test", 1)):
        for prefix, dataloader_idx in (("test", 0),):
            search_batch, search_keywords_list = collate_and_filter_outputs(
                outputs[dataloader_idx]
            )
            search_metrics = self.dataset.validate_search(
                search_batch, search_keywords_list
            )
            for key, value in search_metrics.items():
                self.log(f"{prefix}_search_{key}", value, prog_bar=True, sync_dist=True)

            # qa_batch, qa_result = collate_and_filter_outputs(
            #     outputs[dataloader_idx + 2]
            # )
            qa_batch, qa_result = collate_and_filter_outputs(
                outputs[dataloader_idx + 1]
            )

            if self.qa_model_type.startswith("t5"):
                qa_metrics = self.dataset.validate_tokens(qa_batch, qa_result)
            else:
                qa_metrics = self.dataset.validate_logits(qa_batch, qa_result)

            for key, value in qa_metrics.items():
                self.log(f"{prefix}_qa_{key}", value, prog_bar=True, sync_dist=True)

            if not self.is_distributed or get_rank() == 0:
                print("Validation result:")
                for key, value in search_metrics.items():
                    print(f"{prefix}_search_{key}: {value}")
                for key, value in qa_metrics.items():
                    print(f"{prefix}_qa_{key}: {value}")

    def test_step(self, batch: BatchEncoding, _batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            choice_num = self.model.choice_predictors["default"].choice_num
            self.model.choice_predictors["default"].choice_num = 1
            logits_list = []
            for b in range(0, batch["sentence"].shape[0]):
                sub_logits_list = []
                for i in range(0, batch["sentence"].shape[1], 32):
                    logits = self.model.predict(
                        input_ids=batch["sentence"][b : b + 1, i : i + 32].to(
                            self.real_device
                        ),
                        attention_mask=batch["mask"][b : b + 1, i : i + 32].to(
                            self.real_device
                        ),
                        token_type_ids=batch["type_ids"][b : b + 1, i : i + 32].to(
                            self.real_device
                        ),
                    )
                    sub_logits_list.append(logits.view(1, -1))
                logits_list.append(t.cat(sub_logits_list, dim=1))
            choice = t.argmax(t.cat(logits_list), dim=1)
            self.model.choice_predictors["default"].choice_num = choice_num
            for ch, id in zip(choice, batch["id"]):
                self.dataset.set_search_target(ch, "test", id)
        else:
            if self.qa_model_type.startswith("t5"):
                out = self.qa_model.generate(
                    batch["sentence"].to(self.real_device),
                    max_length=self.config.generate_length,
                    attention_mask=batch["mask"].to(self.real_device),
                    early_stopping=True,
                )
                result = t.full(
                    [out.shape[0], self.config.generate_length],
                    self.tokenizer.pad_token_id,
                )
                result[:, : out.shape[1]] = out.cpu()
                batch = batch.to("cpu")
                return {
                    "batch": batch,
                    "result": result,
                }
            else:
                return {
                    "batch": batch.to("cpu"),
                    "result": self.qa_model.predict(
                        input_ids=batch["sentence"].to(self.real_device),
                        attention_mask=batch["mask"].to(self.real_device),
                        token_type_ids=batch["type_ids"].to(self.real_device),
                    ).cpu(),
                }

    def test_epoch_end(self, outputs):
        if self.is_distributed:
            t.cuda.set_device(self.real_device)
            gathered_outputs = [None] * get_world_size()
            all_gather_object(gathered_outputs, outputs)
            gathered_outputs = list(itertools.chain.from_iterable(gathered_outputs))
            self.test_on_main_process(gathered_outputs)
        else:
            self.test_on_main_process(outputs)

    @rank_zero_only
    def test_on_main_process(self, outputs):
        _, result = collate_and_filter_outputs(outputs[1])
        if self.qa_model_type.startswith("t5"):
            self.dataset.generate_test_result_tokens(result, self.stage_result_path)
        else:
            self.dataset.generate_test_result_logits(result, self.stage_result_path)

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
