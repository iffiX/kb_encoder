import re


def close_eq(float1, float2):
    return -1e-6 < float1 - float2 < 1e-6


def can_mine_numerical_relation(question, choices):
    return (
        re.match("^((\d{1,3}(,\d{3})*|\d+)(\.\d+)?)( [\w/]+)?$", choices[0]) is not None
        and 1 <= len(re.findall("((\d{1,3}(,\d{3})*|\d+)(\.\d+)?)", question) or []) < 3
    )


def mine_numerical_relation(question, choices):
    question_matches = re.findall("((\d{1,3}(,\d{3})*|\d+)(\.\d+)?)", question)
    choice_matches = re.findall("((\d{1,3}(,\d{3})*|\d+)(\.\d+)?)", " ".join(choices))
    question_values = [float(match[0].replace(",", "")) for match in question_matches]
    choice_values = [float(match[0].replace(",", "")) for match in choice_matches]
    result = []
    if len(question_values) == 1:
        for ch, ch_val in zip(choices, choice_values):
            if question_values[0] == ch_val:
                result = [f"{question_matches[0][0]} = {ch_val:g} in {ch}."]
    else:
        result = []
        for arg0, arg1, operation in (
            (question_values[0], question_values[1], "+"),
            (question_values[0], question_values[1], "-"),
            (question_values[0], question_values[1], "*"),
            (question_values[0], question_values[1], "/"),
            (question_values[1], question_values[0], "-"),
            (question_values[1], question_values[0], "/"),
        ):
            if operation == "+":
                value = arg0 + arg1
            elif operation == "-":
                value = arg0 - arg1
            elif operation == "*":
                value = arg0 * arg1
            else:
                value = arg0 / arg1
            for ch, ch_val in zip(choices, choice_values):
                if close_eq(value, ch_val):
                    result.append(
                        f"{arg0:g} {operation} {arg1:g} = {ch_val:g} in {ch}."
                    )
    return result
