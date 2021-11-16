import warnings
import torch as t
import pytorch_lightning as pl
from collections import Counter
from torch.utils.data import DataLoader
from transformers import BatchEncoding
from .utils import collate_and_filter_outputs
from .commonsense_qa_trainer import CommonsenseQATrainer
from .openbook_qa_trainer import OpenBookQATrainer
from encoder.dataset.base import EmptyDataset
from encoder.utils.config import EnsembleTrainConfig


class EnsembleTrainer(pl.LightningModule):
    def __init__(
        self,
        config: EnsembleTrainConfig,
        stage_result_path="./",
        _is_distributed=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        warnings.filterwarnings("ignore")

        self.config = config
        self.stage_result_path = stage_result_path

        if len(config.checkpoints) == 0:
            raise ValueError("Checkpoint must be non-empty")

        self.trainers = [
            self.stage_name_to_checkpoint(config.task_trainer_stage, chk)
            for chk in config.checkpoints
        ]

    @property
    def monitor(self):
        return self.trainers[0].monitor

    @property
    def monitor_mode(self):
        return self.trainers[0].monitor_mode

    def train_dataloader(self):
        return DataLoader(dataset=EmptyDataset())

    def val_dataloader(self):
        return self.trainers[0].val_dataloader

    def test_dataloader(self):
        return self.trainers[0].test_dataloader

    # noinspection PyTypeChecker
    def training_step(self, batch: BatchEncoding, batch_idx):
        return None

    # noinspection PyTypeChecker
    def validation_step(self, batch: BatchEncoding, batch_idx):
        results = [
            trainer.validation_step(batch, batch_idx) for trainer in self.trainers
        ]
        return {
            "batch": results[0]["batch"],
            "tokens": t.cat(
                [result["tokens"].unsqueeze(1) for result in results], dim=1
            ),
        }

    def validation_epoch_end(self, outputs):
        batch, tokens = self.select_answer(outputs)
        metrics = self.trainers[0].dataset.validate_tokens(batch, tokens)

    def test_step(self, batch: BatchEncoding, batch_idx):
        results = [
            trainer.validation_step(batch, batch_idx) for trainer in self.trainers
        ]
        return {
            "batch": results[0]["batch"],
            "tokens": t.cat(
                [result["tokens"].unsqueeze(1) for result in results], dim=1
            ),
        }

    def test_epoch_end(self, outputs):
        batch, tokens = self.select_answer(outputs)
        self.trainers[0].dataset.generate_test_results_tokens(
            tokens, self.stage_result_path
        )

    def configure_optimizers(self):
        return None

    def select_answer(self, outputs):
        batch, tokens = collate_and_filter_outputs(outputs)
        total = len(batch["id"])
        answer_num = len(self.trainers)
        selected_tokens = []
        for i in range(total):
            answers = [
                (j, self.tokenizer.decode(tokens[i, j], skip_special_tokens=True))
                for j in range(answer_num)
            ]
            answer = Counter([a[1] for a in answers]).most_common(1)[0]
            selected_tokens.append(
                [tokens[i, a[0]].unsqueeze(0) for a in answers if a[1] == answer][0]
            )
        return batch, t.cat(selected_tokens, dim=0)

    @classmethod
    def stage_name_to_checkpoint(cls, stage: str, checkpoint_path: str):
        stage_name_to_trainer_map = {
            "commonsense_qa": CommonsenseQATrainer,
            "openbook_qa": OpenBookQATrainer,
        }
        if stage in stage_name_to_trainer_map:
            return stage_name_to_trainer_map[stage].load_from_checkpoint(
                checkpoint_path, "cpu"
            )
        else:
            raise ValueError(f"Unknown stage {stage}.")
