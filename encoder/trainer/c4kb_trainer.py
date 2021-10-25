import torch as t
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, BatchEncoding
from ..dataset.base import collate_function_dict_to_batch_encoding
from ..dataset.c4_kb import C4KBDataset
from ..utils.config import C4KBTrainConfig
from ..utils.settings import proxies, model_cache_dir


class C4KBTrainer(pl.LightningModule):
    def __init__(
        self,
        config: C4KBTrainConfig,
        stage_result_path="./",
        is_distributed=False,
        only_init_model=False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.stage_result_path = stage_result_path
        self.is_distributed = is_distributed
        self.kb_model = KBMaskedLMEncoder(
            relation_size=config.relation_size,
            base_type=config.base_type,
            relation_mode=config.relation_mode,
            mlp_hidden_size=config.mlp_hidden_size,
            **config.base_configs,
        )
        if not only_init_model:
            self.kb_tokenizer = AutoTokenizer.from_pretrained(
                config.base_type, cache_dir=model_cache_dir, proxies=proxies,
            )

            if config.dataset == "KDWD":
                self.dataset = KDWDBertDataset(
                    relation_size=config.relation_size,
                    context_length=config.context_length,
                    sequence_length=config.max_seq_length,
                    tokenizer=self.kb_tokenizer,
                    **config.dataset_config.dict(),
                )
            else:
                raise ValueError(
                    f"Unknown KBEncoderTrainConfig.dataset: {config.dataset}"
                )
            if config.task not in ("entity", "relation", "entity+relation"):
                raise ValueError(f"Unknown KBEncoderTrainConfig.task: {config.task}")

    @property
    def monitor(self):
        if self.config.task == "entity":
            return "mlm_loss"
        elif self.config.task == "relation":
            if self.config.dataset == "KDWD":
                return "total_loss"
            else:
                raise ValueError("Unknown dataset.")
        else:
            if self.config.dataset == "KDWD":
                return "total_loss"
            else:
                raise ValueError("Unknown dataset.")

    @property
    def monitor_mode(self):
        return "min"

    def train_dataloader(self):
        if self.config.task == "entity":
            return DataLoader(
                dataset=self.dataset.train_entity_encode_dataset,
                batch_size=self.config.batch_size,
                collate_fn=collate_function_dict_to_batch_encoding,
                num_workers=self.config.load_worker_num,
                prefetch_factor=self.config.load_prefetch_per_worker,
            )
        elif self.config.task == "relation":
            return DataLoader(
                dataset=self.dataset.train_relation_encode_dataset,
                batch_size=self.config.batch_size,
                collate_fn=collate_function_dict_to_batch_encoding,
                num_workers=self.config.load_worker_num,
                prefetch_factor=self.config.load_prefetch_per_worker,
            )
        else:
            return CombinedLoader(
                [
                    DataLoader(
                        dataset=d,
                        batch_size=self.config.batch_size,
                        collate_fn=collate_function_dict_to_batch_encoding,
                        num_workers=self.config.load_worker_num,
                        prefetch_factor=self.config.load_prefetch_per_worker,
                    )
                    for d in (
                        self.dataset.train_entity_encode_dataset,
                        self.dataset.train_relation_encode_dataset,
                    )
                ],
                "max_size_cycle",
            )

    def val_dataloader(self):
        if self.config.task == "entity":
            return DataLoader(
                dataset=self.dataset.validate_entity_encode_dataset,
                batch_size=self.config.batch_size,
                collate_fn=collate_function_dict_to_batch_encoding,
                num_workers=self.config.load_worker_num,
                prefetch_factor=self.config.load_prefetch_per_worker,
            )
        elif self.config.task == "relation":
            return DataLoader(
                dataset=self.dataset.validate_relation_encode_dataset,
                batch_size=self.config.batch_size,
                collate_fn=collate_function_dict_to_batch_encoding,
                num_workers=self.config.load_worker_num,
                prefetch_factor=self.config.load_prefetch_per_worker,
            )
        else:
            return CombinedLoader(
                [
                    DataLoader(
                        dataset=d,
                        batch_size=self.config.batch_size,
                        collate_fn=collate_function_dict_to_batch_encoding,
                        num_workers=self.config.load_worker_num,
                        prefetch_factor=self.config.load_prefetch_per_worker,
                    )
                    for d in (
                        self.dataset.validate_entity_encode_dataset,
                        self.dataset.validate_relation_encode_dataset,
                    )
                ],
                "min_size",
                # "max_size_cycle",
            )

    # noinspection PyTypeChecker
    def training_step(self, batch: BatchEncoding, batch_idx):
        if self.config.task == "entity":
            # Masked Language model training
            out = self.kb_model(
                tokens=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                token_type_ids=batch["token_type_ids"].to(self.device),
                labels=batch["labels"].to(self.device),
            )
            self.log("train_loss", out[1], sync_dist=self.is_distributed)
            return out[1]
        elif self.config.task == "relation":
            # Relation encoding training
            # Make sure that your model is trained on MLM first
            relation_logits = self.kb_model.compute_relation(
                tokens1=batch["input_ids_1"].to(self.device),
                tokens2=batch["input_ids_2"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                token_type_ids=batch["token_type_ids"].to(self.device),
            )
            result = self.dataset.get_loss(batch, relation_logits)
            self.log("train_loss", result[0] + result[1], sync_dist=self.is_distributed)
            return result[0] + result[1]
        else:
            # entity + relation
            out = self.kb_model(
                tokens=batch[0]["input_ids"].to(self.device),
                attention_mask=batch[0]["attention_mask"].to(self.device),
                token_type_ids=batch[0]["token_type_ids"].to(self.device),
                labels=batch[0]["labels"].to(self.device),
            )
            relation_logits = self.kb_model.compute_relation(
                tokens1=batch[1]["input_ids_1"].to(self.device),
                tokens2=batch[1]["input_ids_2"].to(self.device),
                attention_mask=batch[1]["attention_mask"].to(self.device),
                token_type_ids=batch[1]["token_type_ids"].to(self.device),
            )
            result = self.dataset.get_loss(batch[1], relation_logits)
            self.log(
                "train_loss",
                out[1] + result[0] + result[1],
                sync_dist=self.is_distributed,
                prog_bar=True,
            )
            self.log(
                "entity_loss", out[1], sync_dist=self.is_distributed, prog_bar=True,
            )
            self.log(
                "relation_loss",
                result[0] + result[1],
                sync_dist=self.is_distributed,
                prog_bar=True,
            )
            return out[1] + result[0] + result[1]

    # noinspection PyTypeChecker
    def validation_step(self, batch: BatchEncoding, _batch_idx):
        if self.config.task == "entity":
            # Masked Language model validation
            out = self.kb_model(
                tokens=batch["input_ids"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                token_type_ids=batch["token_type_ids"].to(self.device),
                labels=batch["labels"].to(self.device),
            )
            metrics = {"mlm_loss": out[1]}
        elif self.config.task == "relation":
            # Relation encoding training
            # Make sure that your model is trained on MLM first
            relation_logits = self.kb_model.compute_relation(
                tokens1=batch["input_ids_1"].to(self.device),
                tokens2=batch["input_ids_2"].to(self.device),
                attention_mask=batch["attention_mask"].to(self.device),
                token_type_ids=batch["token_type_ids"].to(self.device),
            )
            metrics = self.dataset.validate_relation_encode(batch, relation_logits)
        else:
            out = self.kb_model(
                tokens=batch[0]["input_ids"].to(self.device),
                attention_mask=batch[0]["attention_mask"].to(self.device),
                token_type_ids=batch[0]["token_type_ids"].to(self.device),
                labels=batch[0]["labels"].to(self.device),
            )
            relation_logits = self.kb_model.compute_relation(
                tokens1=batch[1]["input_ids_1"].to(self.device),
                tokens2=batch[1]["input_ids_2"].to(self.device),
                attention_mask=batch[1]["attention_mask"].to(self.device),
                token_type_ids=batch[1]["token_type_ids"].to(self.device),
            )
            result = self.dataset.get_loss(batch[1], relation_logits)
            metrics = {
                "total_loss": out[1] + result[0] + result[1],
                "mlm_loss": out[1],
                "direction_loss": result[0],
                "relation_loss": result[1],
            }

        for key, value in metrics.items():
            self.log(key, value, sync_dist=self.is_distributed)

    def configure_optimizers(self):
        optim_cls = getattr(t.optim, self.config.optimizer_class)
        return optim_cls(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.l2_regularization,
        )
