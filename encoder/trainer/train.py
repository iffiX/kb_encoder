import re
import os
import logging
import pytorch_lightning as pl
from ..utils.config import *
from .qa_trainer import QATrainer
from .glue_trainer import GLUETrainer
from .commonsense_qa_trainer import CommonsenseQATrainer
from .openbook_qa_trainer import OpenBookQATrainer
from .ensemble_trainer import EnsembleTrainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

stage_name_to_trainer_map = {
    "qa": QATrainer,
    "glue": GLUETrainer,
    "commonsense_qa": CommonsenseQATrainer,
    "openbook_qa": OpenBookQATrainer,
    "ensemble": EnsembleTrainer,
}


def find_checkpoint(
    checkpoint_path: str, monitor: str = None, monitor_mode: str = None
):
    available_files = []
    if monitor is None or monitor_mode is None:
        logging.info("Finding last available checkpoint")
        for file in os.listdir(checkpoint_path):
            if file.endswith(".ckpt"):
                available_files.append(file)
        sorted_by_time = sorted(available_files, key=lambda f: os.stat(f).st_mtime)
        if len(sorted_by_time) == 0:
            return None
        checkpoint = sorted_by_time[-1]
    else:
        logging.info(
            f"Finding checkpoint with monitor={monitor}, monitor_mode={monitor_mode}"
        )
        for file in os.listdir(checkpoint_path):
            if re.search(f"{monitor}=([+-]?([0-9]*[.])?[0-9]+)", file) is not None:
                available_files.append(file)
        sorted_by_epoch = sorted(
            available_files,
            key=lambda f: float(
                re.search(f"{monitor}=([+-]?([0-9]*[.])?[0-9]+)", f)[1]
            ),
        )
        if len(sorted_by_epoch) == 0:
            return None
        if monitor_mode == "max":
            checkpoint = sorted_by_epoch[-1]
        else:
            checkpoint = sorted_by_epoch[0]

    return os.path.join(checkpoint_path, checkpoint)


def stage_name_to_trainer(
    stage: str, stage_config, stage_result_path: str, is_distributed: bool
):
    if stage in stage_name_to_trainer_map:
        return stage_name_to_trainer_map[stage](
            stage_config, stage_result_path, is_distributed=is_distributed
        )
    else:
        raise ValueError(f"Unknown stage {stage}.")


def stage_name_to_checkpoint(stage: str, checkpoint_path: str):
    if stage in stage_name_to_trainer_map:
        return stage_name_to_trainer_map[stage].load_from_checkpoint(checkpoint_path)
    else:
        raise ValueError(f"Unknown stage {stage}.")


def _train(
    config,
    stage_config,
    stage_trainer,
    is_distributed: bool,
    checkpoint_path: str,
    log_path: str,
):
    # create directories, or reuse
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    save_config(config, os.path.join(config.working_directory, "config.json"))
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename="{epoch:02d}-"
        + stage_trainer.monitor
        + "-{"
        + stage_trainer.monitor
        + ":.2f}",
        save_top_k=1 if stage_config.save else 0,
        save_last=stage_config.save_last,
        monitor=stage_trainer.monitor,
        mode=stage_trainer.monitor_mode,
        verbose=True,
    )
    early_stopping = EarlyStopping(
        monitor=stage_trainer.monitor,
        mode=stage_trainer.monitor_mode,
        patience=config.early_stopping_patience,
        verbose=True,
    )
    t_logger = TensorBoardLogger(log_path)

    checkpoint = None
    if hasattr(stage_config, "load") and stage_config.load:
        checkpoint = find_checkpoint(
            checkpoint_path, stage_trainer.monitor, stage_trainer.monitor_mode
        )
        if checkpoint is None:
            logging.info("Failed to find a valid checkpoint, using original weights.")
        else:
            logging.info(f"Using checkpoint {checkpoint}")
    else:
        logging.info("Not loading, using original weights.")

    trainer = pl.Trainer(
        gpus=str(config.gpus[0])
        if isinstance(config.gpus, list) and len(config.gpus) == 1
        else config.gpus,
        accelerator="ddp" if is_distributed else None,
        plugins=[DDPPlugin(find_unused_parameters=True)] if is_distributed else None,
        callbacks=[checkpoint_callback, early_stopping],
        logger=[t_logger],
        limit_train_batches=getattr(stage_config, "train_steps", None) or 1.0,
        limit_val_batches=getattr(stage_config, "validate_steps", None) or 1.0,
        max_epochs=stage_config.epochs,
        # # For iterable datasets, to validate after each epoch,
        # # set check interval equal to number of training steps.
        # val_check_interval=stage_config.train_steps,
        accumulate_grad_batches=stage_config.accumulate_grad_batches,
        resume_from_checkpoint=checkpoint,
        deterministic=True,
    )

    trainer.fit(stage_trainer)


def train(config: Config, stage_index: int, only_test: bool = False):
    # t.multiprocessing.set_start_method("spawn", force=True)
    # execute stages
    stage = config.stages[stage_index]

    is_distributed = (isinstance(config.gpus, list) and len(config.gpus) > 1) or (
        isinstance(config.gpus, int) and config.gpus > 1
    )
    stage_config = config.configs[stage_index]
    seed_everything(stage_config.seed, workers=True)
    if stage_config.load_worker_num > 0:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    checkpoint_path = os.path.join(
        config.working_directory, str(stage_index), "checkpoint"
    )
    log_path = os.path.join(config.working_directory, str(stage_index), "log")
    stage_result_path = os.path.join(
        config.working_directory, str(stage_index), "result"
    )

    if not only_test:
        stage_trainer = stage_name_to_trainer(
            stage, stage_config, stage_result_path, is_distributed
        )

        logging.info("Training.")
        _train(
            config=config,
            stage_config=stage_config,
            stage_trainer=stage_trainer,
            is_distributed=is_distributed,
            checkpoint_path=checkpoint_path,
            log_path=log_path,
        )
    else:
        logging.info("Testing.")

        checkpoint = find_checkpoint(checkpoint_path)
        if checkpoint is None:
            raise RuntimeError("Cannot find a valid checkpoint for testing.")
        else:
            logging.info(f"Using checkpoint {checkpoint}")

        # model` must be provided to `trainer.test()` when it hasn't been passed
        # in a previous run.
        # the ckpt_path in test will be ignored in this case.
        # and must perform manual load
        stage_trainer = stage_name_to_checkpoint(stage, checkpoint)

        trainer = pl.Trainer(
            gpus=config.gpus,
            accelerator="ddp" if len(config.gpus) > 1 else None,
            plugins=[DDPPlugin(find_unused_parameters=True)],
            deterministic=True,
        )
        trainer.test(stage_trainer)
