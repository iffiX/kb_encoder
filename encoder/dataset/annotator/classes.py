from functools import lru_cache
from typing import List, Dict, Tuple, Union, Callable

Scalar = Union[str, float, int]


class Context:
    def __init__(self):
        self.variables = {}  # type: Dict[str, Variable]
        self.requirements = {}  # type: Dict[str, Requirement]
        self.functions = {}  # type: Dict[str, Function]
        self.variable_counter = 0

    def init(self):
        for function in self.functions.values():
            function.resolve_requirements()

    def variable(
        self, name: str = None, value=None, tags: Dict[str, Scalar] = None
    ) -> "Variable":
        if name is None:
            name = f"@var{self.variable_counter}"
            self.variable_counter += 1
        elif name.startswith("@var"):
            raise ValueError(
                "Manually created variables must not have a name starting with @var"
            )
        if name not in self.variables:
            if tags is None:
                raise ValueError("Variable not fully specified")
            variable = Variable(name, value, tags, self)
            self.variables[name] = variable
        return self.variables[name]

    def requirement(
        self,
        name: str,
        requirements: Union[
            List[Callable[["Variable"], bool]], List[Callable[..., bool]]
        ] = None,
    ) -> "Requirement":
        if name not in self.requirements:
            if requirements is None:
                raise ValueError("Requirement not fully specified")
            requirement = Requirement(name, requirements, self)
            self.requirements[name] = requirement
        return self.requirements[name]

    def function(
        self,
        name: str,
        arg_num: int = None,
        inputs_requirements: Dict[Tuple[int], "Requirement"] = None,
        output_entails_requirements: Union[
            List["Requirement"], Callable[["Requirement"], bool]
        ] = None,
        func: Callable[[List["Variable"], "Context"], "Variable"] = None,
        tags: Dict[str, Scalar] = None,
    ):
        if name not in self.functions:
            if tags is None:
                raise ValueError("Function not fully specified")
            function = Function(
                name,
                arg_num,
                inputs_requirements,
                output_entails_requirements,
                func,
                tags,
                self,
            )
            self.functions[name] = function
        return self.functions[name]

    def delete_variable(self, name):
        self.variables.pop(name)


class Scope:
    def __init__(self, prefix, scope_or_context: Union[Context, "Scope"]):
        self.prefix = prefix
        self.prev_scope = scope_or_context

    def variable(
        self, name: str = None, value=None, tags: Dict[str, Scalar] = None
    ) -> "Variable":
        return self.prev_scope.variable(f"{self.prefix}_{name}", value, tags)

    def requirement(
        self,
        name: str,
        requirements: Union[
            List[Callable[["Variable"], bool]], List[Callable[..., bool]]
        ] = None,
    ) -> "Requirement":
        return self.prev_scope.requirement(f"{self.prefix}_{name}", requirements)

    def function(
        self,
        name: str,
        arg_num: int = None,
        inputs_requirements: Dict[Tuple[int], "Requirement"] = None,
        output_entails_requirements: Union[
            List["Requirement"], Callable[["Requirement"], bool]
        ] = None,
        func: Callable[[List["Variable"], "Context"], "Variable"] = None,
        tags: Dict[str, Scalar] = None,
    ):
        return self.prev_scope.function(
            f"{self.prefix}_{name}",
            arg_num,
            inputs_requirements,
            output_entails_requirements,
            func,
            tags,
        )


class Variable:
    def __init__(self, name: str, value, tags: Dict[str, Scalar], ctx: Context):
        self.name = name
        self.value = value
        self.tags = tags
        self.ctx = ctx
        self.entails_requirements = set()

    def __hash__(self):
        # For supporting LRU cache
        return id(self)

    def __eq__(self, var):
        return id(self) == id(var)

    def is_equivalent(self, var):
        return self.value == var.value and self.tags == var.tags


class Requirement:
    def __init__(
        self,
        name: str,
        requirements: Union[
            List[Callable[[Variable], bool]], List[Callable[..., bool]]
        ],
        ctx: Context,
    ):
        self.name = name
        self.requirements = requirements
        self.ctx = ctx
        self.__call__ = lru_cache(maxsize=1024)(self.__call__)

    def __call__(self, *variables: Variable):
        if len(variables) == 1:
            return all(
                id(self) in variables[0].entails_requirements
                or requirement(variables[0])
                for requirement in self.requirements
            )
        else:
            return all(requirement(*variables) for requirement in self.requirements)


class Function:
    def __init__(
        self,
        name: str,
        arg_num: int,
        inputs_requirements: Dict[Tuple[int], "Requirement"],
        output_entails_requirements: Union[
            List[Requirement], Callable[[Requirement], bool]
        ],
        func: Callable[[List[Variable], Context], Variable],
        tags: Dict[str, Scalar],
        ctx: Context,
    ):
        self.name = name
        self.arg_num = arg_num
        self.inputs_requirements = inputs_requirements
        self.output_entails_requirements = output_entails_requirements
        self.func = func
        self.tags = tags or {}
        self.ctx = ctx

    def resolve_requirements(self):
        output_entails_requirements = set()
        if isinstance(self.output_entails_requirements, list):
            for out_req in self.output_entails_requirements:
                output_entails_requirements.add(id(out_req))
        else:
            for req in self.ctx.requirements.values():
                if self.output_entails_requirements(req):
                    output_entails_requirements.add(id(req))
        self.output_entails_requirements = output_entails_requirements

    def __call__(self, *variables):
        if all(
            in_req(*(variables[pos] for pos in arg_pos))
            for arg_pos, in_req in self.inputs_requirements.items()
        ):
            output = self.func(variables, self.ctx)
            output.entails_requirements = self.output_entails_requirements
            return output
        else:
            raise ValueError("Input requirement not satisfied")
