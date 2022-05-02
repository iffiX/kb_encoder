from typing import List
from encoder.dataset.annotator.classes import Context, Scope, Variable
from .helpers import tag_eq, all_tag_func_eq, no_tag_or_not_eq
from .physics import add_physics_knowledge
from .physics_particle import add_physics_particle_knowledge


def _func_equal(vars: List[Variable], ctx: Context):
    target, source = vars
    new_tags = source.tags.copy()
    new_tags["object"] = target.tags["object"]
    return ctx.variable(None, value=source.value, tags=new_tags)


def create_core_context():
    ctx = Context()
    add_physics_knowledge(ctx)
    add_physics_particle_knowledge(ctx)
    add_general_knowledge(ctx)
    ctx.init()
    return ctx


def add_general_knowledge(ctx: Context):
    sc = Scope("general", ctx)
    sc.requirement("is_grounded", no_tag_or_not_eq("is_ungrounded", True))
    sc.requirement("is_ungrounded", tag_eq("is_ungrounded", True))
    sc.requirement("is_target_of", [lambda a, b: a.tags["source"] == b.tags["object"]])
    sc.requirement("same_unit_type", all_tag_func_eq("unit", lambda a, b: a[0] == b[0]))
    sc.function(
        "func_equal",
        arg_num=2,
        inputs_requirements={
            (0,): sc.requirement("is_ungrounded"),
            (1,): sc.requirement("is_grounded"),
            (0, 1): sc.requirement("is_target_of"),
            (0, 1): sc.requirement("same_unit_type"),
        },
        output_entails_requirements=[],
        tags={"category": "general"},
        func=_func_equal,
    )
