import json
from pydantic import BaseModel
from pprint import pprint
from typing import *


class QATrainConfig(BaseModel):
    load: bool = False
    seed: int = 0
    save: bool = True
    save_last: bool = False
    epochs: int = 100
    train_steps: Optional[int] = None
    validate_steps: Optional[int] = None
    batch_size: int = 2
    accumulate_grad_batches: int = 32

    optimizer_class: str = "Adam"
    learning_rate: float = 5e-5
    l2_regularization: float = 0
    context_length: int = 200

    base_type: str = "bert-base-uncased"
    extend_config: Optional[Dict[str, Any]]
    extend_mode: str = "ratio_mix"
    base_configs: Dict[str, Any] = {}

    kb_encoder_path: str = ""
    kb_encoder_trainable: bool = False
    kb_encoder_with_gradient_num: int = 1
    # "squad", "squad_v2", "nq", etc.
    train_dataset_path: Optional[str] = "squad"
    validate_dataset_path: Optional[str] = "squad"


class CommonsenseQATrainConfig(BaseModel):
    load: bool = False
    seed: int = 0
    save: bool = True
    save_last: bool = False
    epochs: int = 5
    train_steps: Optional[int] = None
    validate_steps: Optional[int] = None
    batch_size: int = 2
    accumulate_grad_batches: int = 32

    optimizer_class: str = "Adam"
    learning_rate: float = 5e-5
    l2_regularization: float = 0

    base_type: str = "t5-large"
    max_seq_length: int = 128
    generate_length: int = 20
    device_map: Optional[Dict[int, List[int]]] = None
    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2

    use_matcher: bool = True
    matcher_mode: str = "embedding"
    include_option_label_in_sentence: bool = False
    include_option_label_in_answer_and_choices: bool = False
    use_option_label_as_answer_and_choices: bool = False
    match_closest_when_no_equal: bool = True


class OpenBookQATrainConfig(BaseModel):
    load: bool = False
    seed: int = 0
    save: bool = True
    save_last: bool = False
    epochs: int = 5
    train_steps: Optional[int] = None
    validate_steps: Optional[int] = None
    batch_size: int = 2
    accumulate_grad_batches: int = 32

    optimizer_class: str = "Adam"
    learning_rate: float = 5e-5
    l2_regularization: float = 0

    base_type: str = "t5-large"
    max_seq_length: int = 128
    generate_length: int = 20
    device_map: Optional[Dict[int, List[int]]] = None
    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2

    use_matcher: bool = True
    matcher_mode: str = "embedding"
    include_option_label_in_sentence: bool = False
    include_option_label_in_answer_and_choices: bool = False
    use_option_label_as_answer_and_choices: bool = False
    match_closest_when_no_equal: bool = True


class GLUETrainConfig(BaseModel):
    task: str = "cola"
    load: bool = False
    seed: int = 0
    save: bool = True
    save_last: bool = False
    epochs: int = 3
    train_steps: Optional[int] = None
    batch_size: int = 2
    accumulate_grad_batches: int = 32
    storage_precision: int = 32
    max_train_samples: Optional[int] = None
    max_validate_samples: Optional[int] = None
    max_test_samples: Optional[int] = None

    optimizer_class: str = "Adam"
    learning_rate: float = 2e-5
    l2_regularization: float = 0

    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2

    base_type: str = "bert-base-uncased"
    max_seq_length: int = 128
    extend_config: Optional[Dict[str, Any]]
    extend_mode: str = "ratio_mix"
    base_configs: Dict[str, Any] = {}

    kb_encoder_path: str = ""
    kb_encoder_context_length: int = 32
    kb_encoder_max_seq_length: int = 64
    kb_process_gpus: List[int] = [0]
    kb_process_batch_size_per_gpu: int = 32


class EnsembleTrainConfig(BaseModel):
    task_trainer_stage: str
    checkpoints: List[str]


class Config(BaseModel):
    # Cuda ids of GPUs
    gpus: Optional[Union[int, List[int]]] = [0]

    # Maximum validation epochs allowed before stopping
    # when monitored metric is not decreasing
    early_stopping_patience: int = 100

    # Path to the working directory
    # sub-stages will be created as 0, 1, ... subdirectories
    working_directory: str = "./train"

    # example: ["kb_encoder", "qa"]
    # config in configs must match items in stages
    stages: List[str] = []
    configs: List[
        Union[
            QATrainConfig,
            CommonsenseQATrainConfig,
            OpenBookQATrainConfig,
            GLUETrainConfig,
            EnsembleTrainConfig,
        ]
    ] = []


def stage_name_to_config(name: str, config_dict: dict = None):
    stage_name_to_config_map = {
        "qa": QATrainConfig,
        "glue": GLUETrainConfig,
        "commonsense_qa": CommonsenseQATrainConfig,
        "openbook_qa": OpenBookQATrainConfig,
        "ensemble": EnsembleTrainConfig,
    }
    if name in stage_name_to_config_map:
        config_dict = config_dict or {}
        return stage_name_to_config_map[name](**config_dict)
    else:
        raise ValueError(f"Unknown stage {name}.")


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        config_dict = json.load(f)
        config = Config(
            gpus=config_dict["gpus"],
            early_stopping_patience=config_dict["early_stopping_patience"],
            working_directory=config_dict["working_directory"],
        )
        for s, c in zip(config_dict["stages"], config_dict["configs"]):
            config.stages.append(s)
            config.configs.append(stage_name_to_config(s, c))
        return config


def generate_config(stages: List[str], path: str, print_config: bool = True):
    config = Config()
    for stage in stages:
        config.stages.append(stage)
        config.configs.append(stage_name_to_config(stage))

    if print_config:
        pprint(config.dict())
    else:
        save_config(config, path)
        print(f"Config saved to {path}")


def save_config(config: Config, path: str):
    with open(path, "w") as f:
        json.dump(config.dict(), f, indent=4, sort_keys=True)
