import re
import nltk
import chemparse
import periodictable as pt
from itertools import product


def is_astronomy_size_order(question, _choices):
    question_query = nltk.sent_tokenize(question)[-1].lower()
    return (
        "smallest to largest" in question_query
        or "largest to smallest" in question_query
    )


def annotate_astronomy_size_order(question, choices):
    question_query = nltk.sent_tokenize(question)[-1].lower()
    size = {
        "moon": 0,
        "earth": 1,
        "jupiter": 2,
        "sun": 3,
        "solar": 4,
        "galaxy": 5,
    }
    entities_set = set(
        sub_choice for choice in choices for sub_choice in choice.split(", ")
    )
    entities_with_size = []
    for entity in entities_set:
        for size_name in size:
            if size_name in entity:
                entities_with_size.append((entity, size[size_name]))
                break
    entities_with_size = sorted(
        entities_with_size,
        key=lambda x: x[1],
        reverse="largest to smallest" in question_query,
    )
    order = (
        "largest to smallest"
        if "largest to smallest" in question_query
        else "smallest to largest"
    )
    return [
        f"From {order} is {', '.join([entity[0] for entity in entities_with_size])}."
    ]


def extract_chemical_equation(input):
    if "->" not in input:
        return None
    tokens = [t for t in input.split(" ") if len(t) > 0]
    possible_components = [[], []]
    center_idx = tokens.index("->")
    prev_connector = "->"
    for idx in range(center_idx - 1, -1, -1):
        if len(prev_connector) > 0 and re.match(
            r"([0-9]?)([a-zA-Z0-9_{}]+)", tokens[idx]
        ):
            match = re.match(r"([0-9]?)([a-zA-Z0-9_{}]+)", tokens[idx]).groups()
            num = match[0]
            chemical = match[-1]
            if num == "":
                num = 1
            else:
                num = int(num)
            atoms = chemparse.parse_formula(
                chemical.replace("{", "").replace("}", "").replace("_", "")
            )
            possible_components[0].append((num, chemical, atoms))
            prev_connector = ""
        elif tokens[idx] == "+":
            prev_connector = "+"
        elif tokens[idx].lower() == "\\box":
            possible_components[0][-1] = (
                None,
                possible_components[0][-1][1],
                possible_components[0][-1][2],
            )
        else:
            break
    possible_components[0].reverse()
    prev_connector = "->"
    for idx in range(center_idx + 1, len(tokens)):
        if len(prev_connector) > 0 and re.match(
            r"([0-9]?)([a-zA-Z0-9_{}]+)", tokens[idx]
        ):
            match = re.match(r"([0-9]?)([a-zA-Z0-9_{}]+)", tokens[idx]).groups()
            num = match[0]
            chemical = match[-1]
            if num == "":
                num = 1
            else:
                num = int(num)
            atoms = chemparse.parse_formula(
                chemical.replace("{", "").replace("}", "").replace("_", "")
            )
            possible_components[1].append((num, chemical, atoms))
            prev_connector = ""
        elif tokens[idx] == "+":
            prev_connector = "+"
        elif tokens[idx] == "\\box":
            possible_components[1][-1] = (
                None,
                possible_components[1][-1][1],
                possible_components[1][-1][2],
            )
        else:
            break
    return possible_components


def check_chemical_equation_validity(chemical_equation):
    left_atoms, right_atoms = {}, {}
    for left_molecule in chemical_equation[0]:
        for atom, atom_num in left_molecule[2].items():
            left_atoms[atom] = left_atoms.get(atom, 0) + atom_num * left_molecule[0]
    for right_molecule in chemical_equation[1]:
        for atom, atom_num in right_molecule[2].items():
            right_atoms[atom] = right_atoms.get(atom, 0) + atom_num * right_molecule[0]
    return left_atoms == right_atoms


def is_chemistry_equation_conservation_of_mass(question, choices):
    return "conservation of mass" in question.lower() and any(
        "{" in choice for choice in choices
    )


def annotate_chemistry_equation_conservation_of_mass(question, choices):
    result = []
    for choice in choices:
        if check_chemical_equation_validity(extract_chemical_equation(choice)):
            result.append(f"Equation {choice} is balanced")
    return result


def is_chemistry_balance_equation(question, choices):
    question_query = nltk.sent_tokenize(question)[-1].lower()
    return re.search("balance.+equation", question_query) and all(
        re.match(r"[0-9]+", choice) for choice in choices
    )


def annotate_chemistry_balance_equation(question, choices):
    result = []
    question = question.replace("CH_{4}", "2CH_{4}")
    for choice in choices:
        equation = extract_chemical_equation(question)
        for part in (0, 1):
            for idx, molecule in enumerate(equation[part]):
                if molecule[0] is None:
                    equation[part][idx] = (int(choice), molecule[1], molecule[2])
        if check_chemical_equation_validity(equation):
            result.append(f"Need {choice} molecules to balance the equation.")
    return result


def check_chemical_molecule_validity(chemical):
    charge = {
        "H": [1],
        "Na": [1],
        "Mg": [2],
        "Al": [3],
        "Si": [-4, 4],
        "O": [-2],
        "P": [-3, 3, 5],
        "S": [-2, 2, 4, 6],
        "Cl": [-1, 1, 3, 5, 7],
        "K": [1],
        "Ca": [2],
        "Fe": [2, 3, 6],
        "Zn": [2],
        "Br": [-1, 1, 3, 5],
    }
    molecule_number = {
        "H": [2],
        "Na": [1],
        "Mg": [1],
        "Al": [1],
        "Si": [1],
        "P": [1],
        "O": [2, 3],
        "S": [1],
        "Cl": [2],
        "K": [1],
        "Ca": [1],
        "Fe": [1],
        "Zn": [2],
        "Br": [2],
    }
    atoms = chemparse.parse_formula(
        chemical.replace("{", "").replace("}", "").replace("_", "")
    )
    if len(atoms) > 2:
        raise ValueError("Not checkable")
    if len(atoms) == 1:
        atom = list(atoms.keys())[0]
        return atoms[atom] in molecule_number[atom]
    elif len(atoms) == 2:
        atom1, atom2 = atoms.keys()
        for atom1_charge, atom2_charge in product(charge[atom1], charge[atom2]):
            if atoms[atom1] * atom1_charge + atoms[atom2] * atom2_charge == 0:
                return True
    return False


def is_chemistry_correct_molecule(question, choices):
    return (
        "formula correctly represents" in question.lower()
        or "expected product" in question.lower()
    ) and any("{" in choice for choice in choices)


def annotate_chemistry_correct_molecule(question, choices):
    result = []
    for choice in choices:
        if check_chemical_molecule_validity(choice):
            result.append(f"Formula {choice} is correct.")
    return result


def is_chemistry_molecule_elements(question, choices):
    return "how many elements" in question.lower()


def annotate_chemistry_molecule_elements(question, choices):
    tokens = question.split(" ")
    result = []
    for token in tokens:
        if "(" in token:
            token = token.strip("?").strip("!").strip(".")
            formula = chemparse.parse_formula(token)
            if token not in formula:
                result.append(f"There are {len(formula)} elements in compound {token}.")
    return result


def is_chemistry_molecule_mass(question, choices):
    question_query = nltk.sent_tokenize(question)[-1].lower()
    return "mass of" in question_query and "molecule" in question_query


def annotate_chemistry_molecule_mass(question, choices):
    mass = {
        "water": 18,
        "oxygen": 32,
        "nitrogen": 28,
    }
    question_query = nltk.sent_tokenize(question)[-1].lower()
    molecule = re.search("a ([a-zA-Z]+) molecule", question_query).groups()[0]

    return [f"The mass of the {molecule} molecule is {mass[molecule]}."]


def is_biology_sex_cell_chromosome_number(question, _choices):
    question_query = nltk.sent_tokenize(question)[-1].lower()
    return "how many chromosomes" in question_query and (
        "sex cell" in question_query
        or "sperm cell" in question_query
        or "egg cell" in question_query
    )


def annotate_biology_sex_cell_chromosome_number(question, _choices):
    match = re.findall(r"((\d{1,3}(,\d{3})*|\d+)(\.\d+)?) chromosomes", question)
    if len(match) == 0:
        if "human" in question:
            return ["There are 23 chromosomes in a human sex cell."]
    elif len(match) == 1:
        return [f"There are {int(match[0][0]) / 2} chromosomes in the sex cell."]
    return []


def can_dedicated_annotate(question, choices):
    return (
        is_astronomy_size_order(question, choices)
        or is_chemistry_equation_conservation_of_mass(question, choices)
        or is_chemistry_balance_equation(question, choices)
        or is_chemistry_correct_molecule(question, choices)
        or is_chemistry_molecule_mass(question, choices)
        or is_chemistry_molecule_elements(question, choices)
        or is_biology_sex_cell_chromosome_number(question, choices)
    )


def dedicated_annotate(question, choices):
    if is_astronomy_size_order(question, choices):
        return annotate_astronomy_size_order(question, choices)
    elif is_chemistry_equation_conservation_of_mass(question, choices):
        return annotate_chemistry_equation_conservation_of_mass(question, choices)
    elif is_chemistry_balance_equation(question, choices):
        return annotate_chemistry_balance_equation(question, choices)
    elif is_chemistry_correct_molecule(question, choices):
        return annotate_chemistry_correct_molecule(question, choices)
    elif is_chemistry_molecule_mass(question, choices):
        return annotate_chemistry_molecule_mass(question, choices)
    elif is_chemistry_molecule_elements(question, choices):
        return annotate_chemistry_molecule_elements(question, choices)
    elif is_biology_sex_cell_chromosome_number(question, choices):
        return annotate_biology_sex_cell_chromosome_number(question, choices)
    else:
        raise ValueError("Not an annotateble question")
