import traceback
from .numerical_annotator import NumericalAnnotator
from .simple_relation_miner import can_mine_numerical_relation, mine_numerical_relation
from .dedicated_annotators import can_dedicated_annotate, dedicated_annotate


class Annotator:
    def __init__(self):
        self.numerical_annotator = NumericalAnnotator()

    def annotate(self, question, choices, quiet=True):
        choices = [choice for choice in choices if len(choice) > 0]
        try:
            if self.numerical_annotator.should_annotate(question, choices):
                return " \\n (annotation) " + " ".join(
                    self.numerical_annotator.annotate(question, choices,)
                )
        except Exception:
            if not quiet:
                print(
                    f"Exception occurred while using numerical annotator on question {question}"
                )
                print(traceback.format_exc())
                print("Fall back to alternate solutions")

        try:
            if can_dedicated_annotate(question, choices):
                facts = dedicated_annotate(question, choices)
                if len(facts) > 0:
                    return " \\n (annotation) " + " ".join(facts)
        except Exception:
            if not quiet:
                print(
                    f"Exception occurred while using dedicated annotator on question {question}"
                )
                print(traceback.format_exc())
                print("Fall back to alternate solutions")

        try:
            if can_mine_numerical_relation(question, choices):
                facts = mine_numerical_relation(question, choices)
                if len(facts) > 0:
                    return " \\n (annotation) " + " ".join(facts)
        except Exception:
            if not quiet:
                print(
                    f"Exception occurred while using numerical miner on question {question}"
                )
                print(traceback.format_exc())
                print("Fall back to alternate solutions")
        return None
