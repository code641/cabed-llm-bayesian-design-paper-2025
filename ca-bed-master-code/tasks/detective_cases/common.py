import re
from textwrap import dedent

from tasks.detective_cases.data import DetectiveCasesInstance


def parse_question(hypothesis_space: list[str], question: str) -> tuple[str, str]:
    suspect_match = re.match(r"\[(.*?)\]\s*(.*)", question)
    assert suspect_match, f"Bad question: {question}"
    suspect_name, actual_question = suspect_match.groups()
    if suspect_name in hypothesis_space:
        return suspect_name, actual_question

    # allowed_chars = set(string.ascii_letters)

    # def sanitise(text: str) -> str:
    #     return "".join(c for c in text.lower() if c in allowed_chars).strip()

    # sanitised_suspect_name = sanitise(suspect_name)
    # for hypo in hypothesis_space:
    #     if sanitise(hypo) == sanitised_suspect_name:
    #         return hypo, actual_question

    raise RuntimeError(f"Unrecognised: {suspect_name}")


def get_case_background(instance: DetectiveCasesInstance) -> str:
    return dedent(f"""\
        Time: {instance["time"]}
        Location: {instance["location"]}
        Victim:
        - Name: {instance["victim"]["name"]}
        - Introduction: {instance["victim"]["introduction"]}
        - Cause of Death: {instance["victim"]["cause_of_death"]}
        - Murder Weapon: {instance["victim"]["murder_weapon"]}
    """)
