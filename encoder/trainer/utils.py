import torch as t
from transformers import (
    get_constant_schedule,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)
from ..dataset.base import collate_function_dict_to_batch_encoding, dict_iter


def set_worker_sharing_strategy(_worker_id: int) -> None:
    t.multiprocessing.set_sharing_strategy("file_system")


def collate_and_filter_outputs(outputs):
    batch = collate_function_dict_to_batch_encoding([o["batch"] for o in outputs])
    if t.is_tensor(outputs[0]["result"]):
        results = t.cat([o["result"] for o in outputs], dim=0)
        list_of_results = [
            (b["id"][0], b, to.unsqueeze(0)) for b, to in zip(dict_iter(batch), results)
        ]
    else:
        results = [r for o in outputs for r in o["result"]]
        list_of_results = [
            (b["id"][0], b, li) for b, li in zip(dict_iter(batch), results)
        ]
    # filter duplicates brought by resetting dataset
    existed = {}
    filtered = []
    for lr in list_of_results:
        if lr[0] not in existed:
            filtered.append(lr)
            existed[lr[0]] = True
    list_of_results = filtered
    if t.is_tensor(list_of_results[0][2]):
        results = t.cat([lr[2] for lr in list_of_results], dim=0)
    else:
        results = [lr[2] for lr in list_of_results]
    batch = collate_function_dict_to_batch_encoding([lr[1] for lr in list_of_results])
    return batch, results


def make_scheduler(optimizer, warmup_proportion, training_steps, num_cycles):
    if warmup_proportion <= 0:
        return get_constant_schedule(optimizer)
    else:
        return get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_proportion * training_steps,
            num_training_steps=training_steps,
            num_cycles=num_cycles,
        )
