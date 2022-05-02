from typing import Set
from encoder.dataset.annotator.classes import (
    Scalar,
    Variable,
)


def no_tag_or_not_eq(tag_name: str, tag_value: Scalar):
    def requirement(var: Variable) -> bool:
        return tag_name not in var.tags or var.tags[tag_name] != tag_value

    return [requirement]


def tag_eq(tag_name: str, tag_value: Scalar):
    def requirement(var: Variable) -> bool:
        return tag_name in var.tags and var.tags[tag_name] == tag_value

    return [requirement]


def tag_in(tag_name: str, tag_values: Set[Scalar]):
    def requirement(var: Variable) -> bool:
        return tag_name in var.tags and var.tags[tag_name] in tag_values

    return [requirement]


def all_tag_eq(tag_name: str):
    def requirement(*vars: Variable) -> bool:
        if all(tag_name in var.tags for var in vars):
            return len(set(var.tags[tag_name] for var in vars)) == 1
        return False

    return [requirement]


def all_tag_func_eq(tag_name: str, eq_func):
    def requirement(*vars: Variable) -> bool:
        if all(tag_name in var.tags for var in vars):
            return all(
                eq_func(vars[idx].tags[tag_name], vars[idx + 1].tags[tag_name])
                for idx in range(len(vars) - 1)
            )
        return False

    return [requirement]
