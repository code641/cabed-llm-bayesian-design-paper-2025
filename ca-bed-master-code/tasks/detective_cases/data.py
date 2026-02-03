import json
import os
from pathlib import Path
import random
from typing import NotRequired, TypedDict, cast


class VictimInformation(TypedDict):
    name: str
    introduction: str
    cause_of_death: str
    murder_weapon: str


class SuspectOverview(TypedDict):
    name: str
    introduction: str


class SuspectInformation(TypedDict):
    name: str
    introduction: str
    # There is always a character unaware of the murder, who is definitely not the murderer
    is_murderer: NotRequired[bool]
    story: str
    task: str


class InitialInformation(TypedDict):
    time: str
    location: str
    victim: VictimInformation
    suspect: list[SuspectOverview]


class DetectiveCasesInstance(TypedDict):
    time: str
    location: str
    victim: VictimInformation
    suspects: list[SuspectInformation]
    initial_information: InitialInformation


def load_all_data() -> list[DetectiveCasesInstance]:
    data_path = Path(os.path.dirname(__file__), "ARBenchTest.json")
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Murderer is always index 0
    random.seed(42)
    for instance in data:
        random.shuffle(instance["suspects"])

    return cast(list[DetectiveCasesInstance], data)


if __name__ == "__main__":
    data = load_all_data()

    for inst in data:
        print(
            next(
                i
                for i in range(len(inst["suspects"]))
                if inst["suspects"][i].get("is_murderer", False)
            )
        )
