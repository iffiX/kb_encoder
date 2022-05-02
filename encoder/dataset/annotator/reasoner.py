import numpy as np
from itertools import product
from typing import List, Tuple, Callable
from .classes import Context, Function, Variable
from .hint import HintBase


class Reasoner:
    def __init__(
        self,
        hint: HintBase,
        stopper: Callable[[List[Variable]], bool],
        context: Context,
    ):
        self.hint = hint
        self.stopper = stopper
        self.context = context
        self.prev_steps = (
            {}
        )  # key: (function_name, arg1_name, arg2_name, ...) value: output variable name
        self.function_tags = self.collect_tags(source="functions")

    def reason(self, max_steps: int = 100, deterministic: bool = False, seed: int = 42):
        rng = np.random.default_rng(seed)
        prev_variable_value_set = self.serialize_vars_for_comparison()
        for i in range(max_steps):
            if self.stopper(list(self.context.variables.values())):
                break
            variable_tags = self.collect_tags(source="variables")

            function_boundary = self.hint.predict_next_step_function_boundary(
                self.prev_steps, self.function_tags, variable_tags
            ) or [function.name for function in self.context.functions.values()]

            valid_steps = self.generate_valid_steps(function_boundary)
            valid_step_descriptions = [(ss.name for ss in step) for step in valid_steps]

            # remove existing steps
            valid_step_descriptions = list(
                set(valid_step_descriptions).difference(set(self.prev_steps.keys()))
            )

            prob = self.hint.predict_next_step_combination_probability(
                self.prev_steps,
                valid_step_descriptions,
                self.function_tags,
                variable_tags,
            )

            for i in range(len(valid_steps)):
                # Find a step that won't create a cycle
                # eg: speed * time = distance, then distance / time = speed will
                # create a cycle
                if deterministic:
                    next_step_idx = np.argmax(prob)
                else:
                    next_step_idx = rng.choice(list(range(len(valid_steps))), p=prob)

                next_step = valid_steps[
                    next_step_idx
                ]  # type: Tuple[Function, Variable, ...]
                # Execute step
                # (returned variable is added to the context during its creation in function)
                result = next_step[0](*next_step[1:])
                serialized_result_value = self.serialize_variable(result)
                if serialized_result_value not in prev_variable_value_set:
                    prev_variable_value_set.add(serialized_result_value)
                    break
                else:
                    prob[next_step_idx] = 0
                    sum = prob.sum()
                    if sum == 0:
                        # print("Unable to find a non-cyclic step, stopping early")
                        return
                    prob = prob / prob.sum()
                    self.context.delete_variable(result.name)
            else:
                break

    def generate_valid_steps(self, function_boundary):
        valid_steps = []
        for function_name in function_boundary:
            function = self.context.function(function_name)
            valid_variable_for_each_pos = [
                list(self.context.variables.values()) for _ in range(function.arg_num)
            ]

            # Find single argument requirement first
            for arg_pos, requirement in function.inputs_requirements.items():
                if len(arg_pos) == 1:
                    valid_variable_for_each_pos[arg_pos[0]] = [
                        v
                        for v in valid_variable_for_each_pos[arg_pos[0]]
                        if requirement(v)
                    ]

            # Find valid combinations of multi-argument requirement
            valid_variable_combinations = list(product(*valid_variable_for_each_pos))
            multi_arg_requirements = [
                (arg_pos, requirement)
                for arg_pos, requirement in function.inputs_requirements.items()
                if len(arg_pos) > 1
            ]
            for arg_pos, requirement in multi_arg_requirements:
                valid_variable_combinations = [
                    combination
                    for combination in valid_variable_combinations
                    if requirement(*[combination[pos] for pos in arg_pos])
                ]
            for combination in valid_variable_combinations:
                valid_steps.append((function, *combination))
        return valid_steps

    def collect_tags(self, source):
        tags = {}
        for obj in getattr(self.context, source).values():
            for key, value in obj.tags.items():
                if key not in tags:
                    tags[key] = {}
                if value not in tags[key]:
                    tags[key][value] = [obj.name]
                else:
                    existing_scalar_type = type(next(iter(tags[key].keys())))
                    if not isinstance(value, existing_scalar_type):
                        raise ValueError(
                            f"Detected different value type {type(value)} with value {value} "
                            f"for {source} tag {key},"
                            f"existing value type is {existing_scalar_type}"
                        )
                    tags[key][value].append(obj.name)
        return tags

    def serialize_vars_for_comparison(self):
        variable_value_set = set()
        for variable in self.context.variables.values():
            variable_value_set.add(self.serialize_variable(variable))
        return variable_value_set

    @staticmethod
    def serialize_variable(variable):
        tag_values = sorted(list(variable.tags.items()), key=lambda x: x[0])
        value = tuple([variable.value] + [xx for x in tag_values for xx in x])
        return value
