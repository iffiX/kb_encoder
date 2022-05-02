import re
import nltk
from typing import List
from .hint import BasicCategoricalHint
from .reasoner import Reasoner
from .kb.core import create_core_context
from .kb.physics import convert_time, convert_speed


class NumericalAnnotator:
    def __init__(self):
        pass

    def should_annotate(self, question: str, choices: List[str]):
        question = self.normalize_text_to_integer(question)
        return (
            self.is_physics_kinematics_linear_speed(question, choices)
            or self.is_physics_kinematics_linear_time(question, choices)
            or self.is_physics_kinematics_linear_acceleration(question, choices)
            or self.is_physics_kinematics_linear_distance_rescale(question, choices)
            or self.is_physics_general_binary_computation(question, choices)
            or self.is_physics_particle(question, choices)
        )

    def annotate(
        self,
        question: str,
        choices: List[str],
        max_steps: int = 100,
        deterministic: bool = False,
        seed: int = 42,
    ):
        question = self.normalize_text_to_integer(question)
        context = create_core_context()
        if self.is_physics_kinematics_linear_speed(question, choices):
            (
                hint_words,
                initial_vars,
                stopper,
                extracter,
            ) = self.parse_physics_kinematics_linear_speed(question, choices)
        elif self.is_physics_kinematics_linear_time(question, choices):
            (
                hint_words,
                initial_vars,
                stopper,
                extracter,
            ) = self.parse_physics_kinematics_linear_time(question, choices)
        elif self.is_physics_kinematics_linear_acceleration(question, choices):
            (
                hint_words,
                initial_vars,
                stopper,
                extracter,
            ) = self.parse_physics_kinematics_linear_acceleration(question, choices)
        elif self.is_physics_kinematics_linear_distance_rescale(question, choices):
            (
                hint_words,
                initial_vars,
                stopper,
                extracter,
            ) = self.parse_physics_kinematics_linear_distance_rescale(question, choices)
        elif self.is_physics_general_binary_computation(question, choices):
            (
                hint_words,
                initial_vars,
                stopper,
                extracter,
            ) = self.parse_physics_general_binary_computation(question, choices)
        elif self.is_physics_particle(question, choices):
            (
                hint_words,
                initial_vars,
                stopper,
                extracter,
            ) = self.parse_physics_particle(question, choices)
        else:
            raise ValueError("Not an annotatable question")
        for var in initial_vars:
            context.variable(var["name"], var["value"], var["tags"])
        reasoner = Reasoner(BasicCategoricalHint(hint_words), stopper, context)
        reasoner.reason(max_steps=max_steps, deterministic=deterministic, seed=seed)
        return extracter(list(context.variables.values()))

    def normalize_text_to_integer(self, question):
        numwords = {
            "half": "0.5",
            "zero": "0",
            "first": "1",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
            "eleven": "11",
            "twelve": "12",
            "thirteen": "13",
            "fourteen": "14",
            "fifteen": "15",
            "sixteen": "16",
            "seventeen": "17",
            "eighteen": "18",
            "nineteen": "19",
            "twenty": "20",
            "thirty": "30",
            "forty": "40",
            "fifty": "50",
            "sixty": "60",
            "seventy": "70",
            "eighty": "80",
            "ninety": "90",
            "hundred": "100",
            "thousand": "1000",
            "million": "1000000",
            "billion": "1000000000",
            "trillion": "1000000000000",
        }
        for numword, replacement in numwords.items():
            question = re.sub(
                numword + " ", replacement + " ", question, flags=re.IGNORECASE
            )
        return question

    def extract_time_length(self, input):
        time_map = {
            "s": ("time", "s"),
            "h": ("time", "h"),
            "se": ("time", "s"),
            "mi": ("time", "min"),
            "ho": ("time", "h"),
            "hr": ("time", "h"),
            "da": ("time", "d"),
            "we": ("time", "w"),
            "mo": ("time", "mon"),
            "ye": ("time", "y"),
        }
        time_matches = re.findall(
            r"((\d{1,3}(,\d{3})*|\d+)(\.\d+)?)"
            r"[^a-zA-Z]{0,1}"
            r"(seconds?|min(ute|utes|s)?|hours?|days?|weeks?|months|years?|s|h|hr)"
            r"([ \.!?,;]|$)",
            input,
            flags=re.IGNORECASE,
        )
        return [
            (
                float(time_match[0].replace(",", "")),
                time_map[
                    time_match[-3][0].lower()
                    if len(time_match[-3]) == 1
                    else time_match[-3][:2].lower()
                ],
            )
            for time_match in time_matches
        ]

    def extract_distance(self, input):
        input = input.replace("meters per", "").replace("kilometers per", "")
        distance_map = {
            "m": ("dist", "m"),
            "km": ("dist", "km"),
            "mm": ("dist", "mm"),
            "mi": ("dist", "mm"),
            "me": ("dist", "m"),
            "ki": ("dist", "km"),
        }
        distance_matches = re.findall(
            r"((\d{1,3}(,\d{3})*|\d+)(\.\d+)?)"
            r"[^a-zA-Z]{0,1}"
            r"(millimeters?|meters?|kilometers?|m|mm|km)"
            r"([ \.!?,;]|$)",
            input,
            flags=re.IGNORECASE,
        )
        return [
            (
                float(distance_match[0].replace(",", "")),
                distance_map[distance_match[-2][:2].lower()],
            )
            for distance_match in distance_matches
        ]

    def extract_speed(self, input):
        speed_map = {
            "meters per second": ("spd", "m", "s"),
            "meters per hour": ("spd", "m", "h"),
            "kilometers per hour": ("spd", "km", "h"),
            "m/s": ("spd", "m", "s"),
            "m/h": ("spd", "m", "h"),
            "km/h": ("spd", "km", "h"),
            "km/hr": ("spd", "km", "h"),
        }
        speed_matches = re.findall(
            r"((\d{1,3}(,\d{3})*|\d+)(\.\d+)?)"
            r"[^a-zA-Z]{0,1}"
            r"(meters per second|meters per hour|kilometers per hour|m/s|m/h|km/h|km/hr)"
            r"([ \.!?,;]|$)",
            input,
            flags=re.IGNORECASE,
        )
        return [
            (
                float(speed_match[0].replace(",", "")),
                speed_map[speed_match[-2].lower()],
            )
            for speed_match in speed_matches
        ]

    def extract_physics_general_binary_variables(self, question):
        current_match = re.search(
            r"current of ((\d{1,3}(,\d{3})*|\d+)(\.\d+)?)",
            question,
            flags=re.IGNORECASE,
        )
        resistance_match = re.search(
            r"resistance of ((\d{1,3}(,\d{3})*|\d+)(\.\d+)?)",
            question,
            flags=re.IGNORECASE,
        )
        general_match = re.findall(
            r"((\d{1,3}(,\d{3})*|\d+)(\.\d+)?)[^a-zA-Z]{0,1}(seconds?|kg|j|m/s)[ \.!?,;]",
            question,
            flags=re.IGNORECASE,
        )
        result = []
        if current_match is not None:
            result.append(
                ("current", float(current_match.groups()[0].replace(",", "")))
            )
        if resistance_match is not None:
            result.append(
                ("resistance", float(resistance_match.groups()[0].replace(",", "")))
            )
        if len(general_match) > 0:
            category_map = {"s": "time", "k": "mass", "j": "energy", "m": "speed"}
            existed_matches = set()
            for match in general_match:
                value = match[0] + " " + match[-1]
                if value not in existed_matches:
                    existed_matches.add(value)
                    result.append(
                        (
                            category_map[match[-1][0].lower()],
                            float(match[0].replace(",", "")),
                        )
                    )
        if len(result) == 2:
            if (
                {result[0][0], result[1][0]} == {"current", "resistance"}
                or {result[0][0], result[1][0]} == {"mass", "speed"}
                or {result[0][0], result[1][0]} == {"energy", "time"}
            ):
                return result
        return None

    def extract_particle_properties(self, input):
        result = {}
        match = re.findall(
            r"atomic number of ((\d{1,3}(,\d{3})*|\d+)(\.\d+)?)",
            input,
            flags=re.IGNORECASE,
        )
        if match:
            if len(match) == 1:
                result["proton"] = int(match[0][0])
            else:
                return {}
        match = re.findall(
            r"mass number of ((\d{1,3}(,\d{3})*|\d+)(\.\d+)?)",
            input,
            flags=re.IGNORECASE,
        )
        if match:
            if len(match) == 1:
                result["mass"] = int(match[0][0])
            else:
                return {}
        match = re.findall(
            r"((\d{1,3}(,\d{3})*|\d+)(\.\d+)?)"
            r"[^a-zA-Z]{0,1}"
            r"(protons|neutrons|electrons)",
            input,
            flags=re.IGNORECASE,
        )
        if len(match) > 0:
            for m in match:
                if m[-1].lower().startswith("proton"):
                    if "proton" in result:
                        return {}
                    result["proton"] = int(m[0])
                elif m[-1].lower().startswith("neutron"):
                    result["neutron"] = int(m[0])
                elif m[-1].lower().startswith("electron"):
                    result["electron"] = int(m[0])

        match = re.findall(
            r"((\d{1,3}(,\d{3})*|\d+)(\.\d+)?) more "
            r"(protons?|electrons?) than "
            r"(protons?|electrons?)",
            input,
            flags=re.IGNORECASE,
        )
        if match:
            if len(match) == 1:
                if match[0][-2].lower().startswith("electron") and match[0][
                    -1
                ].lower().startswith("proton"):
                    result["proton_and_electron_diff"] = -int(match[0][0])
                elif match[0][-2].lower().startswith("electron") and match[0][
                    -1
                ].lower().startswith("proton"):
                    result["proton_and_electron_diff"] = int(match[0][0])
            else:
                return {}

        match = re.findall(
            r"((\d{1,3}(,\d{3})*|\d+)(\.\d+)?) more "
            r"protons? than "
            r"[\w ]*atom of ([a-z]+)",
            input,
            flags=re.IGNORECASE,
        )
        if match:
            if len(match) == 1:
                result["proton_diff"] = (int(match[0][0]), match[0][-1])
            else:
                return {}
        return result

    def is_physics_kinematics_linear_speed(self, question, _choices):
        sentences = nltk.sent_tokenize(question)
        return (
            re.search(r"average speed.*\?", sentences[-1], flags=re.IGNORECASE,)
            is not None
            and len(self.extract_distance(question)) == 1
            and len(self.extract_time_length(question)) == 1
        )

    def parse_physics_kinematics_linear_speed(self, question, choices):
        distance_match = self.extract_distance(question)[0]
        time_match = self.extract_time_length(question)[0]
        distance = {
            "name": "distance",
            "value": distance_match[0],
            "tags": {"unit": distance_match[1], "object": "object1"},
        }
        time = {
            "name": "time",
            "value": time_match[0],
            "tags": {"unit": time_match[1], "object": "object1"},
        }

        choice_speeds = []
        for choice in choices:
            choice_speeds.append(self.extract_speed(choice)[0])

        def stopper(vars):
            for var in vars:
                if var.tags["unit"][0] == "spd":
                    return True
            return False

        def extracter(vars):
            for var in vars:
                if var.tags["unit"][0] == "spd":
                    converted_units = set()
                    result = []
                    for choice_speed in choice_speeds:
                        if choice_speed[1] not in converted_units:
                            converted_units.add(choice_speed[1])
                            new_speed = convert_speed(
                                var.value, var.tags["unit"][1:], choice_speed[1][1:],
                            )
                            if len(result) == 0:
                                result.append(
                                    f"Speed is {new_speed:g} {choice_speed[1][1]}/{choice_speed[1][2]}."
                                )
                            else:
                                result.append(
                                    f"Speed is also {new_speed:g} {choice_speed[1][1]}/{choice_speed[1][2]}."
                                )
                    return result

        return ["speed"], (distance, time), stopper, extracter

    def is_physics_kinematics_linear_time(self, question, _choices):
        sentences = nltk.sent_tokenize(question)
        return (
            re.search(
                r"(how long|how much time).*\?", sentences[-1], flags=re.IGNORECASE,
            )
            is not None
            and len(self.extract_distance(question)) == 1
            and len(self.extract_speed(question)) == 1
        )

    def parse_physics_kinematics_linear_time(self, question, choices):
        distance_match = self.extract_distance(question)[0]
        speed_match = self.extract_speed(question)[0]
        distance = {
            "name": "distance",
            "value": distance_match[0],
            "tags": {"unit": distance_match[1], "object": "object1",},
        }
        speed = {
            "name": "speed",
            "value": speed_match[0],
            "tags": {"unit": speed_match[1], "object": "object1",},
        }

        choice_times = []
        for choice in choices:
            choice_times.append(self.extract_time_length(choice)[0])

        def stopper(vars):
            for var in vars:
                if var.tags["unit"][0] == "time":
                    return True
            return False

        def extracter(vars):
            for var in vars:
                if var.tags["unit"][0] == "time":
                    converted_units = set()
                    result = []
                    for choice_time in choice_times:
                        if choice_time[1] not in converted_units:
                            converted_units.add(choice_time[1])
                            new_time = convert_time(
                                var.value, var.tags["unit"][1], choice_time[1][1],
                            )
                            if len(result) == 0:
                                result.append(
                                    f"Time is {new_time:g} {choice_time[1][1]}."
                                )
                            else:
                                result.append(
                                    f"Time is also {new_time:g} {choice_time[1][1]}."
                                )
                    return result

        return ["time"], (distance, speed), stopper, extracter

    def is_physics_kinematics_linear_acceleration(self, question, _choices):
        speeds_match = self.extract_speed(question)
        return (
            len(speeds_match) == 2
            and speeds_match[0][1] == speeds_match[1][1]  # same unit
            and len(self.extract_time_length(question)) == 1
        )

    def parse_physics_kinematics_linear_acceleration(self, question, _choices):
        speeds_match = self.extract_speed(question)
        time_match = self.extract_time_length(question)[0]
        speed_diff = {
            "name": "speed_diff",
            "value": speeds_match[1][0] - speeds_match[0][0],
            "tags": {"unit": speeds_match[0][1], "object": "object1"},
        }
        time = {
            "name": "time",
            "value": time_match[0],
            "tags": {"unit": time_match[1], "object": "object1"},
        }

        def stopper(vars):
            for var in vars:
                if var.tags["unit"][0] == "acc":
                    return True
            return False

        def extracter(vars):
            for var in vars:
                if var.tags["unit"][0] == "acc":
                    result = [
                        f"Acceleration is {var.value:g} {var.tags['unit'][1]}/{var.tags['unit'][2]}^2."
                    ]
                    return result

        return ["acceleration"], (speed_diff, time), stopper, extracter

    def is_physics_kinematics_linear_distance_rescale(self, question, _choices):
        return (
            len(self.extract_distance(question)) == 1
            and len(self.extract_time_length(question)) == 2
        )

    def parse_physics_kinematics_linear_distance_rescale(self, question, _choices):
        distance_match = self.extract_distance(question)[0]
        times_match = self.extract_time_length(question)
        dist1 = {
            "name": "dist1",
            "value": distance_match[0],
            "tags": {"unit": distance_match[1], "object": "object1"},
        }
        time1 = {
            "name": "time1",
            "value": times_match[0][0],
            "tags": {"unit": times_match[0][1], "object": "object1"},
        }

        time2 = {
            "name": "time2",
            "value": times_match[1][0],
            "tags": {"unit": times_match[1][1], "object": "object2",},
        }

        speed2_placeholder = {
            "name": "speed2_placeholder",
            "value": None,
            "tags": {
                "unit": ("spd", None, None),
                "object": "object2",
                "source": "object1",
                "is_ungrounded": True,
            },
        }

        def stopper(vars):
            for var in vars:
                if var.tags["unit"][0] == "dist" and var.tags["object"] == "object2":
                    return True
            return False

        def extracter(vars):
            for var in vars:
                if var.tags["unit"][0] == "dist" and var.tags["object"] == "object2":
                    result = [
                        f"New traveled distance is {var.value:g} {var.tags['unit'][1]}."
                    ]
                    return result

        return (
            ["distance"],
            (dist1, time1, time2, speed2_placeholder),
            stopper,
            extracter,
        )

    def is_physics_general_binary_computation(self, question, _choices):
        return self.extract_physics_general_binary_variables(question) is not None

    def parse_physics_general_binary_computation(self, question, _choices):
        match_1, match_2 = self.extract_physics_general_binary_variables(question)
        unit_map = {
            "mass": ("mass", "kg"),
            "speed": ("spd", "m", "s"),
            "energy": ("enrgy", "j"),
            "time": ("time", "s"),
            "current": ("curr", "a"),
            "resistance": ("resist", "o"),
        }
        variable1 = {
            "name": "variable1",
            "value": float(match_1[1]),
            "tags": {"unit": unit_map[match_1[0]], "object": "object1",},
        }
        variable2 = {
            "name": "variable2",
            "value": float(match_2[1]),
            "tags": {"unit": unit_map[match_2[0]], "object": "object1",},
        }

        def stopper(vars):
            return len(vars) == 3

        def extracter(vars):
            result_value = [
                var.value for var in vars if var.name not in ("variable1", "variable2")
            ][0]
            if {match_1[0], match_2[0]} == {"current", "resistance"}:
                return [f"Voltage is {result_value:g} V"]
            elif {match_1[0], match_2[0]} == {"mass", "speed"}:
                return [f"Momentum is {result_value:g} kg x m/s"]
            elif {match_1[0], match_2[0]} == {"energy", "time"}:
                return [f"Power is {result_value:g} W"]

        if {match_1[0], match_2[0]} == {"current", "resistance"}:
            hint = ["voltage"]
        elif {match_1[0], match_2[0]} == {"mass", "speed"}:
            hint = ["momentum"]
        elif {match_1[0], match_2[0]} == {"energy", "time"}:
            hint = ["power"]
        else:
            raise ValueError(
                f"'{question}' is not annotatable by physics_general_binary_computation"
            )
        return hint, (variable1, variable2), stopper, extracter

    def is_physics_particle(self, question, _choices):
        return len(self.extract_particle_properties(question)) > 0

    def parse_physics_particle(self, question, choices):
        query = set()
        question_query = nltk.sent_tokenize(question)[-1].lower()
        if question_query.endswith("?"):
            if "how many protons" in question_query:
                query.add("component")
            elif "how many subatomic particles" in question_query:
                query.add("subatomic particles")
            elif "of which element" in question_query:
                query.add("atom")
            elif "mass" in question_query:
                query.add("mass")
        choice_query = " ".join(choices)
        if "charge" in choice_query:
            query.add("charge")
        elif "mass" in choice_query:
            query.add("mass")

        particle_properties = self.extract_particle_properties(question)
        variables = []
        for key, value in particle_properties.items():
            if key in ("proton", "neutron", "electron"):
                variables.append(
                    {
                        "name": key,
                        "value": value,
                        "tags": {"object": "object1", "particle_type": key},
                    }
                )
            elif key == "mass":
                variables.append(
                    {
                        "name": "mass",
                        "value": value,
                        "tags": {"object": "object1", "property": "mass"},
                    }
                )
            elif key == "proton_and_electron_diff":
                variables.append(
                    {
                        "name": "proton_and_electron_diff",
                        "value": value,
                        "tags": {
                            "object": "object1",
                            "property": "num_diff",
                            "diff_objects": ("proton", "electron"),
                        },
                    }
                )
            elif key == "proton_diff":
                variables.append(
                    {
                        "name": "proton_and_electron_diff",
                        "value": value[0],
                        "tags": {
                            "object": "object1",
                            "property": "num_diff",
                            "diff_objects": ("proton", "proton"),
                        },
                    }
                )
                variables.append(
                    {
                        "name": "atom1",
                        "value": value[1],
                        "tags": {"object": "object1", "particle_type": "atom"},
                    }
                )

        if "proton" in particle_properties and "electron" not in particle_properties:
            # Assume a neutral atom
            variables.append(
                {
                    "name": "atom1-placeholder",
                    "value": None,
                    "tags": {"object": "object1", "particle_type": "atom"},
                }
            )

        starting_variable_names = set(var["name"] for var in variables)

        def stopper(vars):
            # rely on auto stopping
            return False

        def extracter(vars):
            result = []
            for var in vars:
                if "particle_type" in var.tags:
                    if var.tags["particle_type"] != "atom":
                        if "component" in query:
                            result.append(
                                f"There are {var.value} "
                                f"{var.tags['particle_type']}{'s' if var.value > 1 else ''} "
                                f"in the atom."
                            )
                    else:
                        if "atom" in query and var.name not in starting_variable_names:
                            result.append(f"The element is {var.value}.")
                elif "property" in var.tags:
                    if var.tags["property"] == "mass" and "mass" in query:
                        result.append(f"The mass number is {var.value}.")
                    elif var.tags["property"] == "charge" and "charge" in query:
                        if var.value > 0:
                            sign = "+"
                        elif var.value < 0:
                            sign = "-"
                        else:
                            sign = ""
                        result.append(f"The charge is {sign}{var.value}.")
                    elif (
                        var.tags["property"] == "nucleus_particle_number"
                        and "subatomic particles" in query
                    ):
                        result.append(
                            f"There are {var.value} subatomic particles in the nucleus."
                        )
            return sorted(list(set(result)))

        return ["particle"], tuple(variables), stopper, extracter
