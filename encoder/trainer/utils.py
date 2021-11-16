import torch as t
from ..dataset.base import collate_function_dict_to_batch_encoding, dict_iter


def set_worker_sharing_strategy(_worker_id: int) -> None:
    t.multiprocessing.set_sharing_strategy("file_system")


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
    batch = collate_function_dict_to_batch_encoding([lr[1] for lr in list_of_results])
    return batch, tokens
