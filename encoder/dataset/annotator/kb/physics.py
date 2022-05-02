from itertools import product
from typing import List, Tuple
from encoder.dataset.annotator.classes import (
    Context,
    Scope,
    Variable,
)
from .helpers import tag_eq, tag_in, all_tag_eq


def convert_time(in_value, in_unit: str, out_unit: str):
    conversion = {"s": 1, "min": 60, "h": 3600, "d": 86400}
    return in_value * conversion[in_unit] / conversion[out_unit]


def convert_distance(in_value, in_unit: str, out_unit: str):
    conversion = {"mm": 1e-3, "cm": 1e-2, "m": 1, "km": 1e3, "au": 1.495978707e11}
    return in_value * conversion[in_unit] / conversion[out_unit]


def convert_speed(in_value, in_unit: Tuple[str, str], out_unit: Tuple[str, str]):
    dist_conversion = {"mm": 1e-3, "cm": 1e-2, "m": 1, "km": 1e3, "au": 1.495978707e11}
    time_conversion = {"s": 1, "min": 60, "h": 3600, "d": 86400}

    return (
        in_value
        * (dist_conversion[in_unit[0]] / dist_conversion[out_unit[0]])
        * (time_conversion[out_unit[1]] / time_conversion[in_unit[1]])
    )


def distance_units():
    return set(product(["dist"], ["mm", "cm", "m", "km", "au"]))


def time_units():
    return set(product(["time"], ["s", "min", "h", "d"]))


def speed_units():
    return set(product(["spd"], ["mm", "cm", "m", "km", "au"], ["s", "min", "h", "d"]))


def _kinematics_acceleration_units():
    return set(product(["acc"], ["mm", "cm", "m", "km", "au"], ["s", "min", "h", "d"]))


def _kinematics_linear_speed(vars: List[Variable], ctx: Context):
    dist, time = vars
    return ctx.variable(
        None,
        value=dist.value / time.value,
        tags={
            "unit": ("spd", dist.tags["unit"][1], time.tags["unit"][1]),
            "object": dist.tags["object"],
        },
    )


def _kinematics_linear_time(vars: List[Variable], ctx: Context):
    dist, speed = vars
    return ctx.variable(
        None,
        value=convert_distance(dist.value, dist.tags["unit"][1], speed.tags["unit"][1])
        / speed.value,
        tags={"unit": ("time", speed.tags["unit"][2]), "object": dist.tags["object"],},
    )


def _kinematics_linear_distance(vars: List[Variable], ctx: Context):
    time, speed = vars
    return ctx.variable(
        None,
        value=speed.value
        * convert_time(time.value, time.tags["unit"][1], speed.tags["unit"][2]),
        tags={"unit": ("dist", speed.tags["unit"][1]), "object": speed.tags["object"],},
    )


def _kinematics_linear_acceleration(vars: List[Variable], ctx: Context):
    speed, time = vars
    return ctx.variable(
        None,
        value=speed.value
        / convert_time(time.value, time.tags["unit"][1], speed.tags["unit"][2]),
        tags={
            "unit": ("acc", speed.tags["unit"][1], speed.tags["unit"][2]),
            "object": speed.tags["object"],
        },
    )


def _kinematics_linear_acceleration_time(vars: List[Variable], ctx: Context):
    speed, acceleration = vars
    return ctx.variable(
        None,
        value=convert_speed(
            speed.value, speed.tags["unit"][1:], acceleration.tags["unit"][1:]
        )
        / acceleration.value,
        tags={
            "unit": ("time", acceleration.tags["unit"][2]),
            "object": acceleration.tags["object"],
        },
    )


def _kinematics_linear_accelerate_to_speed(vars: List[Variable], ctx: Context):
    time, acceleration = vars
    return ctx.variable(
        None,
        value=convert_time(
            time.value, time.tags["unit"][1], acceleration.tags["unit"][2]
        )
        * acceleration.value,
        tags={
            "unit": ("spd", acceleration.tags["unit"][1], acceleration.tags["unit"][2]),
            "object": acceleration.tags["object"],
        },
    )


def _kinematics_momentum(vars: List[Variable], ctx: Context):
    mass, speed = vars
    return ctx.variable(
        None,
        value=mass.value * speed.value,
        tags={"unit": ("mmt", "kg", "m", "s"), "object": mass.tags["object"],},
    )


def _electric_voltage(vars: List[Variable], ctx: Context):
    current, resistance = vars
    return ctx.variable(
        None,
        value=current.value * resistance.value,
        tags={"unit": ("volt", "v"), "object": current.tags["object"]},
    )


def _electric_power(vars: List[Variable], ctx: Context):
    energy, time = vars
    return ctx.variable(
        None,
        value=energy.value / time.value,
        tags={"unit": ("pwr", "w"), "object": energy.tags["object"]},
    )


def add_physics_knowledge(ctx: Context):
    add_kinematics_knowledge(ctx)
    add_electric_knowledge(ctx)


def add_kinematics_knowledge(ctx: Context):
    """
    In this simple kinematics model, all objects are assumed to start with
    zero absolute motion to simplify rule definitions.
    """
    sc = Scope("physics_kinematics", ctx)
    # Declare shared requirements
    sc.requirement("distance_unit", tag_in("unit", distance_units()))
    sc.requirement("time_unit", tag_in("unit", time_units()))
    sc.requirement("speed_unit", tag_in("unit", speed_units()))
    sc.requirement(
        "acceleration_unit", tag_in("unit", _kinematics_acceleration_units()),
    )
    sc.requirement("standard_speed_unit", tag_eq("unit", ("spd", "m", "s")))
    sc.requirement("standard_mass_unit", tag_eq("unit", ("mass", "kg")))
    sc.requirement("standard_momentum_unit", tag_eq("unit", ("mmt", "kg", "m", "s")))
    sc.requirement("same_object", all_tag_eq("object"))
    # Function linear_speed
    # Given distance d and time t, compute speed v = d / t
    sc.function(
        "func_linear_speed",
        arg_num=2,
        inputs_requirements={
            (0,): sc.requirement("distance_unit"),
            (1,): sc.requirement("time_unit"),
            (0, 1): sc.requirement("same_object"),
        },
        output_entails_requirements=[sc.requirement("speed_unit")],
        tags={"category": "physics_kinematics", "solve_question_type": "linear_speed"},
        func=_kinematics_linear_speed,
    )

    # Function linear_time
    # Given distance d and speed v, compute time t = d / v
    sc.function(
        "func_linear_time",
        arg_num=2,
        inputs_requirements={
            (0,): sc.requirement("distance_unit"),
            (1,): sc.requirement("speed_unit"),
            (0, 1): sc.requirement("same_object"),
        },
        output_entails_requirements=[sc.requirement("time_unit")],
        tags={"category": "physics_kinematics", "solve_question_type": "linear_time"},
        func=_kinematics_linear_time,
    )

    # Function linear_distance
    # Given speed v and time t, compute distance d = v * t
    sc.function(
        "func_linear_distance",
        arg_num=2,
        inputs_requirements={
            (0,): sc.requirement("time_unit"),
            (1,): sc.requirement("speed_unit"),
            (0, 1): sc.requirement("same_object"),
        },
        output_entails_requirements=[sc.requirement("distance_unit")],
        tags={
            "category": "physics_kinematics",
            "solve_question_type": "linear_distance",
        },
        func=_kinematics_linear_distance,
    )

    # Function linear_acceleration
    # Given speed v and time t, compute acceleration a = v / t
    sc.function(
        "func_linear_acceleration",
        arg_num=2,
        inputs_requirements={
            (0,): sc.requirement("speed_unit"),
            (1,): sc.requirement("time_unit"),
            (0, 1): sc.requirement("same_object"),
        },
        output_entails_requirements=[sc.requirement("acceleration_unit")],
        tags={
            "category": "physics_kinematics",
            "solve_question_type": "linear_acceleration",
        },
        func=_kinematics_linear_acceleration,
    )

    # Function linear_accelerate_time
    # Given speed v and acceleration a, compute acceleration time t = v / a
    sc.function(
        "func_linear_acceleration_time",
        arg_num=2,
        inputs_requirements={
            (0,): sc.requirement("speed_unit"),
            (1,): sc.requirement("acceleration_unit"),
            (0, 1): sc.requirement("same_object"),
        },
        output_entails_requirements=[sc.requirement("time_unit")],
        tags={
            "category": "physics_kinematics",
            "solve_question_type": "linear_acceleration_time",
        },
        func=_kinematics_linear_acceleration_time,
    )

    # Function linear_accelerate_to_speed
    # Given time t and acceleration a, compute speed v = a * t
    sc.function(
        "func_linear_accelerate_to_speed",
        arg_num=2,
        inputs_requirements={
            (0,): sc.requirement("time_unit"),
            (1,): sc.requirement("acceleration_unit"),
            (0, 1): sc.requirement("same_object"),
        },
        output_entails_requirements=[sc.requirement("speed_unit")],
        tags={
            "category": "physics_kinematics",
            "solve_question_type": "linear_accelerate_to_speed",
        },
        func=_kinematics_linear_accelerate_to_speed,
    )
    sc.function(
        "func_momentum",
        arg_num=2,
        inputs_requirements={
            (0,): sc.requirement("standard_mass_unit"),
            (1,): sc.requirement("standard_speed_unit"),
            (0, 1): sc.requirement("same_object"),
        },
        output_entails_requirements=[sc.requirement("standard_momentum_unit")],
        tags={"category": "physics_kinematics", "solve_question_type": "momentum"},
        func=_kinematics_momentum,
    )


def add_electric_knowledge(ctx: Context):
    sc = Scope("electric", ctx)
    sc.requirement("voltage_unit", tag_eq("unit", ("volt", "v")))
    sc.requirement("current_unit", tag_eq("unit", ("curr", "a")))
    sc.requirement("resistance_unit", tag_eq("unit", ("resist", "o")))
    sc.requirement("energy_unit", tag_eq("unit", ("enrgy", "j")))
    sc.requirement("power_unit", tag_eq("unit", ("pwr", "w")))
    sc.requirement("time_unit", tag_eq("unit", ("time", "s")))
    sc.requirement("same_object", all_tag_eq("object"))
    sc.function(
        "func_voltage",
        arg_num=2,
        inputs_requirements={
            (0,): sc.requirement("current_unit"),
            (1,): sc.requirement("resistance_unit"),
            (0, 1): sc.requirement("same_object"),
        },
        output_entails_requirements=[sc.requirement("voltage_unit")],
        tags={"category": "physics_electric", "solve_question_type": "voltage"},
        func=_electric_voltage,
    )
    sc.function(
        "func_power",
        arg_num=2,
        inputs_requirements={
            (0,): sc.requirement("energy_unit"),
            (1,): sc.requirement("time_unit"),
            (0, 1): sc.requirement("same_object"),
        },
        output_entails_requirements=[sc.requirement("power_unit")],
        tags={"category": "physics_electric", "solve_question_type": "power"},
        func=_electric_power,
    )
