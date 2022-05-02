import periodictable as pt
from typing import List, Tuple
from encoder.dataset.annotator.classes import (
    Context,
    Scope,
    Variable,
)
from .helpers import tag_eq, all_tag_eq


def _mass_from_proton_and_neutron(vars: List[Variable], ctx: Context):
    proton, neutron = vars
    return ctx.variable(
        None,
        value=proton.value + neutron.value,
        tags={"property": "mass", "object": proton.tags["object"]},
    )


def _nucleus_particle_number_from_mass(vars: List[Variable], ctx: Context):
    mass = vars[0]
    return ctx.variable(
        None,
        value=mass.value,
        tags={"property": "nucleus_particle_number", "object": mass.tags["object"]},
    )


def _proton_and_electron_diff(vars: List[Variable], ctx: Context):
    proton, electron = vars
    return ctx.variable(
        None,
        value=proton.value - electron.value,
        tags={
            "property": "num_diff",
            "diff_objects": ("proton", "electron"),
            "object": proton.tags["object"],
        },
    )


def _charge_from_proton_and_electron_diff(vars: List[Variable], ctx: Context):
    diff = vars[0]
    return ctx.variable(
        None,
        value=diff.value,
        tags={"property": "charge", "object": diff.tags["object"]},
    )


def _neutron_from_proton_and_mass(vars: List[Variable], ctx: Context):
    proton, mass = vars
    return ctx.variable(
        None,
        value=mass.value - proton.value,
        tags={"particle_type": "neutron", "object": proton.tags["object"]},
    )


def _electron_from_proton(vars: List[Variable], ctx: Context):
    proton = vars[0]
    return ctx.variable(
        None,
        value=proton.value,
        tags={"particle_type": "electron", "object": proton.tags["object"]},
    )


def _new_atom_from_proton_diff_and_old_atom(vars: List[Variable], ctx: Context):
    proton_diff, old_atom = vars
    return ctx.variable(
        None,
        value=pt.elements[
            pt.elements.name(old_atom.value).number + proton_diff.value
        ].name,
        tags={"particle_type": "atom", "object": old_atom.tags["object"] + "_new_atom"},
    )


def add_physics_particle_knowledge(ctx: Context):
    sc = Scope("physics_particle", ctx)
    # Declare shared requirements
    sc.requirement("atom", tag_eq("particle_type", "atom"))
    sc.requirement("proton", tag_eq("particle_type", "proton"))
    sc.requirement("neutron", tag_eq("particle_type", "neutron"))
    sc.requirement("electron", tag_eq("particle_type", "electron"))
    sc.requirement("same_object", all_tag_eq("object"))
    sc.requirement("mass", tag_eq("property", "mass"))
    sc.requirement(
        "nucleus_particle_number", tag_eq("property", "nucleus_particle_number")
    )
    sc.requirement("charge", tag_eq("property", "charge"))
    sc.requirement("num_diff", tag_eq("property", "num_diff"))
    sc.requirement("proton_diff", tag_eq("diff_objects", ("proton", "proton")))
    sc.requirement(
        "proton_and_electron_diff", tag_eq("diff_objects", ("proton", "electron"))
    )
    sc.function(
        "func_mass_from_component",
        arg_num=2,
        inputs_requirements={
            (0,): sc.requirement("proton"),
            (1,): sc.requirement("neutron"),
            (0, 1): sc.requirement("same_object"),
        },
        output_entails_requirements=[sc.requirement("mass")],
        tags={"category": "physics_particle"},
        func=_mass_from_proton_and_neutron,
    )
    sc.function(
        "func_nucleus_particle_number_from_mass",
        arg_num=1,
        inputs_requirements={(0,): sc.requirement("mass")},
        output_entails_requirements=[sc.requirement("nucleus_particle_number")],
        tags={"category": "physics_particle"},
        func=_nucleus_particle_number_from_mass,
    )
    sc.function(
        "func_proton_and_electron_diff",
        arg_num=2,
        inputs_requirements={
            (0,): sc.requirement("proton"),
            (1,): sc.requirement("electron"),
            (0, 1): sc.requirement("same_object"),
        },
        output_entails_requirements=[
            sc.requirement("num_diff"),
            sc.requirement("proton_and_electron_diff"),
        ],
        tags={"category": "physics_particle"},
        func=_proton_and_electron_diff,
    )
    sc.function(
        "func_charge_from_proton_and_electron_diff",
        arg_num=1,
        inputs_requirements={
            (0,): sc.requirement("num_diff"),
            (0,): sc.requirement("proton_and_electron_diff"),
        },
        output_entails_requirements=[sc.requirement("charge")],
        tags={"category": "physics_particle"},
        func=_charge_from_proton_and_electron_diff,
    )
    sc.function(
        "func_neutron_from_proton_and_mass",
        arg_num=2,
        inputs_requirements={
            (0,): sc.requirement("proton"),
            (1,): sc.requirement("mass"),
            (0, 1): sc.requirement("same_object"),
        },
        output_entails_requirements=[sc.requirement("neutron")],
        tags={"category": "physics_particle"},
        func=_neutron_from_proton_and_mass,
    )
    sc.function(
        "func_electron_from_proton",
        arg_num=2,
        inputs_requirements={
            (0,): sc.requirement("proton"),
            (1,): sc.requirement("atom"),
            (0, 1): sc.requirement("same_object"),
        },
        output_entails_requirements=[sc.requirement("electron")],
        tags={"category": "physics_particle"},
        func=_electron_from_proton,
    )
    sc.function(
        "func_new_atom_from_proton_diff_and_old_atom",
        arg_num=2,
        inputs_requirements={
            (0,): sc.requirement("num_diff"),
            (0,): sc.requirement("proton_diff"),
            (1,): sc.requirement("atom"),
            (0, 1): sc.requirement("same_object"),
        },
        output_entails_requirements=[sc.requirement("atom")],
        tags={"category": "physics_particle"},
        func=_new_atom_from_proton_diff_and_old_atom,
    )
