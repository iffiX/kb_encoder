import json
from pydantic import BaseModel
from pprint import pprint
from typing import *


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
    scheduler_warmup_proportion: float = 0
    scheduler_cycles: int = 1

    base_type: str = "t5-large"
    model_configs: Optional[dict] = None
    max_seq_length: int = 128
    generate_length: int = 20
    device_map: Optional[Dict[int, List[int]]] = None
    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2

    use_matcher: bool = True
    matcher_mode: str = "embedding"
    matcher_config: Optional[dict] = None
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
    scheduler_warmup_proportion: float = 0
    scheduler_cycles: int = 1

    base_type: str = "t5-large"
    model_configs: Optional[dict] = None
    max_seq_length: int = 128
    generate_length: int = 20
    device_map: Optional[Dict[int, List[int]]] = None
    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2

    use_matcher: bool = True
    matcher_mode: str = "embedding"
    matcher_config: Optional[dict] = None
    include_option_label_in_sentence: bool = False
    include_option_label_in_answer_and_choices: bool = False
    use_option_label_as_answer_and_choices: bool = False
    match_closest_when_no_equal: bool = True


class OpenBookQAWithSearchTrainConfig(BaseModel):
    load: bool = False
    seed: int = 0
    save: bool = True
    save_last: bool = False
    epochs: int = 5
    qa_checkpoint_path: str = ""
    search_warmup_epochs: int = 0
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
    matcher_config: Optional[dict] = None
    include_option_label_in_sentence: bool = False
    include_option_label_in_answer_and_choices: bool = False
    use_option_label_as_answer_and_choices: bool = False
    match_closest_when_no_equal: bool = True


class OpenBookQAKeywordsTrainConfig(BaseModel):
    load: bool = False
    seed: int = 0
    save: bool = True
    save_last: bool = False
    epochs: int = 5
    qa_checkpoint_path: str = ""
    search_warmup_epochs: int = 0
    train_steps: Optional[int] = None
    validate_steps: Optional[int] = None
    batch_size: int = 2
    accumulate_grad_batches: int = 32

    optimizer_class: str = "Adam"
    learning_rate: float = 5e-5
    tree_learning_rate: float = 5e-5
    l2_regularization: float = 0

    base_type: str = "t5-large"
    max_seq_length: int = 128
    generate_length: int = 20
    device_map: Optional[Dict[int, List[int]]] = None
    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2

    use_matcher: bool = True
    matcher_mode: str = "embedding"
    matcher_config: Optional[dict] = None
    include_option_label_in_sentence: bool = False
    include_option_label_in_answer_and_choices: bool = False
    use_option_label_as_answer_and_choices: bool = False
    match_closest_when_no_equal: bool = True


class OpenBookQAFactTrainConfig(BaseModel):
    load: bool = False
    seed: int = 0
    save: bool = True
    save_last: bool = False
    epochs: int = 5
    qa_checkpoint_path: str = ""
    search_warmup_epochs: int = 0
    search_negative_samples: int = 4
    train_steps: Optional[int] = None
    validate_steps: Optional[int] = None
    batch_size: int = 2
    accumulate_grad_batches: int = 32

    optimizer_class: str = "Adam"
    learning_rate: float = 5e-5
    l2_regularization: float = 0

    base_type: str = "t5-large"
    model_configs: Optional[dict] = None
    max_seq_length: int = 128
    generate_length: int = 20
    device_map: Optional[Dict[int, List[int]]] = None
    load_worker_num: Optional[int] = 0
    load_prefetch_per_worker: Optional[int] = 2

    use_matcher: bool = True
    matcher_mode: str = "embedding"
    matcher_config: Optional[dict] = None
    include_option_label_in_sentence: bool = False
    include_option_label_in_answer_and_choices: bool = False
    use_option_label_as_answer_and_choices: bool = False
    match_closest_when_no_equal: bool = True


class EnsembleTrainConfig(BaseModel):
    task_trainer_stage: str
    checkpoints: List[str]
    matcher_modes_list: List[List[str]]
    matcher_seeds_list: List[List[int]]
    matcher_configs_list: List[List[dict]]


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
            CommonsenseQATrainConfig,
            OpenBookQATrainConfig,
            OpenBookQAWithSearchTrainConfig,
            OpenBookQAKeywordsTrainConfig,
            OpenBookQAFactTrainConfig,
            EnsembleTrainConfig,
        ]
    ] = []


def stage_name_to_config(name: str, config_dict: dict = None):
    stage_name_to_config_map = {
        "commonsense_qa": CommonsenseQATrainConfig,
        "openbook_qa": OpenBookQATrainConfig,
        "openbook_qa_with_search": OpenBookQAWithSearchTrainConfig,
        "openbook_qa_keywords": OpenBookQAKeywordsTrainConfig,
        "openbook_qa_fact": OpenBookQAFactTrainConfig,
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
            gpus=config_dict.get("gpus", 0),
            early_stopping_patience=config_dict.get("early_stopping_patience", 100),
            working_directory=config_dict["working_directory"],
        )
        for s, c in zip(config_dict["stages"], config_dict["configs"]):
            config.stages.append(s)
            config.configs.append(stage_name_to_config(s, c))
        return config


def fix_missing(config):
    default = type(config)()
    for k, v in default.__dict__.items():
        if not hasattr(config, k):
            setattr(config, k, v)
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
