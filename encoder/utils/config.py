import json
from pydantic import BaseModel
from pprint import pprint
from typing import *


class C4KBTrainConfig(BaseModel):
    load: bool = False
    seed: int = 0
    save: bool = True
    epochs: int = 100
    train_steps: int = 100000
    validate_steps: int = 1000
    batch_size: int = 2
    accumulate_grad_batches: int = 32

    optimizer_class: str = "Adam"
    learning_rate: float = 5e-5
    l2_regularization: float = 0

    base_type: str = "t5-large"
    max_seq_length: int = 128
    matcher_max_times: int = 300
    matcher_max_depth: int = 2
    matcher_max_edges: int = 6
    matcher_similarity_exclude: List[str] = None

    device_map: Optional[Dict[int, List[int]]] = None
    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2


class QATrainConfig(BaseModel):
    load: bool = False
    seed: int = 0
    save: bool = True
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
    use_matcher: bool = True
    device_map: Optional[Dict[int, List[int]]] = None
    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2

    include_option_label_in_sentence: bool = False
    include_option_label_in_answer_and_choices: bool = False
    use_option_label_as_answer_and_choices: bool = False
    match_closest_when_no_equal: bool = True


class GLUETrainConfig(BaseModel):
    task: str = "cola"
    load: bool = False
    seed: int = 0
    save: bool = True
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


class Config(BaseModel):
    # Cuda ids of GPUs
    gpus: Optional[List[int]] = [0]

    # Maximum validation epochs allowed before stopping
    # when monitored metric is not decreasing
    early_stopping_patience: int = 100

    # Path to the working directory
    # sub-stages will be created as 0, 1, ... subdirectories
    working_directory: str = "./train"

    # example: ["kb_encoder", "qa"]
    # config in configs must match items in pipeline
    pipeline: List[str] = []
    configs: List[
        Union[
            QATrainConfig, C4KBTrainConfig, CommonsenseQATrainConfig, GLUETrainConfig,
        ]
    ] = []


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        config_dict = json.load(f)
        config = Config(
            gpus=config_dict["gpus"],
            early_stopping_patience=config_dict["early_stopping_patience"],
            working_directory=config_dict["working_directory"],
        )
        for p, c in zip(config_dict["pipeline"], config_dict["configs"]):
            config.pipeline.append(p)
            if p == "c4kb":
                config.configs.append(C4KBTrainConfig(**c))
            elif p == "qa":
                config.configs.append(QATrainConfig(**c))
            elif p == "glue":
                config.configs.append(GLUETrainConfig(**c))
            elif p == "commonsense_qa":
                config.configs.append(CommonsenseQATrainConfig(**c))
            else:
                raise ValueError(f"Unknown stage {p}.")
        return config


def generate_config(stages: List[str], path: str, print_config: bool = True):
    config = Config()
    for stage in stages:
        if stage == "qa":
            config.pipeline.append(stage)
            config.configs.append(QATrainConfig())
        elif stage == "c4kb":
            config.pipeline.append(stage)
            config.configs.append(C4KBTrainConfig)
        elif stage == "glue":
            config.pipeline.append(stage)
            config.configs.append(GLUETrainConfig())
        elif stage == "commonsense_qa":
            config.pipeline.append(stage)
            config.configs.append(CommonsenseQATrainConfig())
        else:
            raise ValueError(f"Unknown stage {stage}")

    if print_config:
        pprint(config.dict())
    else:
        save_config(config, path)
        print(f"Config saved to {path}")


def save_config(config: Config, path: str):
    with open(path, "w") as f:
        json.dump(config.dict(), f, indent=4, sort_keys=True)
