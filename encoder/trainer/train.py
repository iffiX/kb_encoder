import os
import sys
import logging
import torch as t
import pytorch_lightning as pl
from ..utils.config import *
from .c4kb_trainer import C4KBTrainer
from .qa_trainer import QATrainer
from .glue_trainer import GLUETrainer
from .comm_qa_trainer import CommonsenseQATrainer
from .iter_env import IterEnv
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin


def find_checkpoint(checkpoint_path: str):
    # temp for finding best checkpoint, only works if save k=1
    sorted_by_epoch = sorted(
        os.listdir(checkpoint_path), key=lambda x: int(x.split("-")[0].strip("epoch="))
    )
    if len(sorted_by_epoch) == 0:
        return None, None
    checkpoint = sorted_by_epoch[-1]
    epoch = int(checkpoint.split("-")[0].strip("epoch="))
    return os.path.join(checkpoint_path, checkpoint), epoch


def stage_name_to_trainer(stage: str, stage_config, stage_result_path, is_distributed):
    map = {
        "c4kb": C4KBTrainer,
        "qa": QATrainer,
        "glue": GLUETrainer,
        "commonsense_qa": CommonsenseQATrainer,
    }

    if stage in map:
        return map[stage](
            stage_config, stage_result_path, is_distributed=is_distributed
        )
    else:
        raise ValueError(f"Unknown stage {stage}.")


def stage_name_to_checkpoint(stage: str, checkpoint):
    map = {
        "c4kb": C4KBTrainer,
        "qa": QATrainer,
        "glue": GLUETrainer,
        "commonsense_qa": CommonsenseQATrainer,
    }

    if stage in map:
        return map[stage].load_from_checkpoint(checkpoint)
    else:
        raise ValueError(f"Unknown stage {stage}.")


def _train(
    config,
    stage,
    stage_config,
    stage_trainer,
    checkpoint_path: str,
    log_path: str,
    only_test: bool,
):
    is_distributed = (isinstance(config.gpus, list) and len(config.gpus) > 1) or (
        isinstance(config.gpus, int) and config.gpus > 1
    )
    if not only_test:
        logging.info("Training.")

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
            save_last=False,
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
            checkpoint, _epoch = find_checkpoint(checkpoint_path)
            if checkpoint is None:
                logging.info(
                    "Failed to find a valid checkpoint, using original weights."
                )
            else:
                logging.info(f"Using checkpoint {checkpoint}")
        else:
            logging.info("Not loading, using original weights.")

        trainer = pl.Trainer(
            gpus=str(config.gpus[0])
            if isinstance(config.gpus, list) and len(config.gpus) == 1
            else config.gpus,
            accelerator="ddp" if is_distributed else None,
            plugins=[DDPPlugin(find_unused_parameters=True)]
            if is_distributed
            else None,
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
    else:
        logging.info("Testing.")

        checkpoint, _epoch = find_checkpoint(checkpoint_path)
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

    if trainer.global_rank != 0:
        sys.exit(0)


def train_iter(config: Config, stage_index: int, only_test: bool = False):
    logging.info("Iterative training mode")
    is_distributed = (isinstance(config.gpus, list) and len(config.gpus) > 1) or (
        isinstance(config.gpus, int) and config.gpus > 1
    )
    iter_config = config.configs[stage_index]
    if iter_config.kb_trainer_stage == "c4kb":
        kb_trainer = C4KBTrainer.load_from_checkpoint(iter_config, only_init_model=True)
    elif iter_config.kb_trainer_stage == "none":
        kb_trainer = None
    else:
        raise ValueError(f"Unknown kb stage {iter_config.kb_trainer_stage}.")

    stage = iter_config.task_trainer_stage
    stage_config = stage_name_to_config(stage, iter_config.task_trainer_config)
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
    stage_trainer = stage_name_to_trainer(
        stage, stage_config, stage_result_path, is_distributed
    )

    # Modify dataset and add post processing hook
    iter_env = IterEnv(
        stage_trainer,
        kb_trainer=kb_trainer,
        attr_steps=iter_config.attr_steps,
        attr_threshold=iter_config.attr_threshold,
        attr_epoch_interval=iter_config.attr_epoch_interval,
        matcher_max_times=iter_config.matcher_max_times,
        matcher_max_depth=iter_config.matcher_max_depth,
        matcher_max_edges=iter_config.matcher_max_edges,
        matcher_discard_edges_if_similarity_below=iter_config.matcher_discard_edges_if_similarity_below,
        matcher_seed=iter_config.matcher_seed,
    )
    # Run epoch
    # if epoch % refresh_attr_score_interval == 0
    # -> Compute attr score for each token in the sample
    # -> Use matcher to find matches by providing attr score mask
    #    (Or Use kb model to find matches)
    # -> Compute new input "sentence", "mask"
    # -> Train model with new input
    _train(
        config=config,
        stage=stage,
        stage_config=stage_config,
        stage_trainer=stage_trainer,
        checkpoint_path=checkpoint_path,
        log_path=log_path,
        only_test=only_test,
    )


def train(config: Config, stage_index: int, only_test: bool = False):
    # t.multiprocessing.set_start_method("spawn", force=True)
    # execute stages
    stage = config.stages[stage_index]
    if stage == "iter":
        train_iter(config, stage_index, only_test)
    else:
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
        stage_trainer = stage_name_to_trainer(
            stage, stage_config, stage_result_path, is_distributed
        )
        _train(
            config=config,
            stage=stage,
            stage_config=stage_config,
            stage_trainer=stage_trainer,
            checkpoint_path=checkpoint_path,
            log_path=log_path,
            only_test=only_test,
        )
