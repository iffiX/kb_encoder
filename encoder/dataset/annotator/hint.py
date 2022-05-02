import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Union
from .classes import Scalar


class HintBase(ABC):
    def predict_next_step_function_boundary(
        self,
        prev_steps_and_outputs: Dict[Tuple[str, ...], str],
        function_tags: Dict[str, Dict[Scalar, List[str]]],
        variable_tags: Dict[str, Dict[Scalar, List[str]]],
    ) -> Union[List[str], None]:
        return None

    @abstractmethod
    def predict_next_step_combination_probability(
        self,
        prev_steps_and_outputs: Dict[Tuple[str, ...], str],
        next_steps: List[Tuple[str, ...]],
        function_tags: Dict[str, Dict[Scalar, List[str]]],
        variable_tags: Dict[str, Dict[Scalar, List[str]]],
    ) -> np.ndarray:
        pass


class UniformHint(HintBase):
    def predict_next_step_combination_probability(
        self, prev_steps_and_outputs, next_steps, function_tags, variable_tags,
    ):
        return np.ones([len(next_steps)]) / len(next_steps)


class BasicCategoricalHint(UniformHint):
    def __init__(self, hint_words):
        self.hint_words = set(hint_words)

    def predict_next_step_function_boundary(
        self,
        prev_steps_and_outputs: Dict[Tuple[str, ...], str],
        function_tags: Dict[str, Dict[Scalar, List[str]]],
        variable_tags: Dict[str, Dict[Scalar, List[str]]],
    ) -> Union[List[str], None]:
        functions = []
        if (
            len(
                {"acceleration", "speed", "time", "distance", "momentum",}.intersection(
                    self.hint_words
                )
            )
            > 0
        ):
            functions += function_tags["category"]["physics_kinematics"]
        if len({"voltage", "power"}.intersection(self.hint_words)) > 0:
            functions += function_tags["category"]["physics_electric"]
        if "particle" in self.hint_words:
            functions += function_tags["category"]["physics_particle"]
        else:
            pass
        functions += function_tags["category"]["general"]
        return functions if len(functions) > 0 else None
