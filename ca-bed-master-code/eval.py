from datetime import datetime, timedelta
from pathlib import Path
from typing import TypedDict

from history import RunRecord

import json


class RunEval(TypedDict):
    top1: bool
    top3: bool
    start_time: datetime
    end_time: datetime
    conversation_length: int
    questioner_input_tokens: int
    questioner_output_tokens: int
    answerer_input_tokens: int
    answerer_output_tokens: int


class GroupEval(TypedDict):
    num_runs: int
    top1: float
    top3: float
    duration: timedelta
    mean_conversation_length: float
    mean_conversation_length_in_successful_cases: float
    questioner_input_tokens: int
    questioner_output_tokens: int
    answerer_input_tokens: int
    answerer_output_tokens: int


def get_run_eval(run_history: RunRecord) -> RunEval:
    guesses = sorted(
        run_history.final_belief_state.keys(),
        key=run_history.final_belief_state.__getitem__,
        reverse=True,
    )
    top3_guesses = guesses[:3]
    top1_guesses = guesses[:1]

    top1 = run_history.expected_answer in top1_guesses
    top3 = run_history.expected_answer in top3_guesses

    conversation_length = len(run_history.final_path) // 2
    return {
        "top1": top1,
        "top3": top3,
        "start_time": run_history.start_time,
        "end_time": run_history.end_time,
        "conversation_length": conversation_length,
        "questioner_input_tokens": run_history.questioner_session.total_input_tokens,
        "questioner_output_tokens": run_history.questioner_session.total_output_tokens,
        "answerer_input_tokens": run_history.answerer_session.total_input_tokens,
        "answerer_output_tokens": run_history.answerer_session.total_output_tokens,
    }


def get_group_eval(run_evals: list[RunEval]) -> GroupEval:
    top1 = sum(run_eval["top1"] for run_eval in run_evals) / len(run_evals)
    top3 = sum(run_eval["top3"] for run_eval in run_evals) / len(run_evals)
    mean_conversation_length = sum(
        run_eval["conversation_length"] for run_eval in run_evals
    ) / len(run_evals)
    conv_length_successful = [
        run_eval["conversation_length"] for run_eval in run_evals if run_eval["top1"]
    ]
    mean_conversation_length_in_successful_cases = sum(conv_length_successful) / len(
        conv_length_successful
    )

    questioner_input_tokens = sum(
        run_eval["questioner_input_tokens"] for run_eval in run_evals
    )
    questioner_output_tokens = sum(
        run_eval["questioner_output_tokens"] for run_eval in run_evals
    )
    answerer_input_tokens = sum(
        run_eval["answerer_input_tokens"] for run_eval in run_evals
    )
    answerer_output_tokens = sum(
        run_eval["answerer_output_tokens"] for run_eval in run_evals
    )

    duration = max(run_eval["end_time"] for run_eval in run_evals) - min(
        run_eval["start_time"] for run_eval in run_evals
    )

    return {
        "num_runs": len(run_evals),
        "top1": top1,
        "top3": top3,
        "duration": duration,
        "mean_conversation_length": mean_conversation_length,
        "mean_conversation_length_in_successful_cases": mean_conversation_length_in_successful_cases,
        "questioner_input_tokens": questioner_input_tokens,
        "questioner_output_tokens": questioner_output_tokens,
        "answerer_input_tokens": answerer_input_tokens,
        "answerer_output_tokens": answerer_output_tokens,
    }


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    from history import deserialise_run_record
    from tqdm import tqdm
    import polars as pl

    parser = argparse.ArgumentParser(prog="Experiment evaluator")
    parser.add_argument(
        "-p",
        "--paths",
        nargs="+",
        type=Path,
        required=True,
        help="List of directories to evaluate",
    )
    parser.add_argument(
        "--questioner-input-price",
        type=float,
        default=0.28,
        help="Questioner input token price per 1M tokens",
    )
    parser.add_argument(
        "--questioner-output-price",
        type=float,
        default=0.42,
        help="Questioner output token price per 1M tokens",
    )
    parser.add_argument(
        "--answerer-input-price",
        type=float,
        default=0.28,
        help="Answerer input token price per 1M tokens",
    )
    parser.add_argument(
        "--answerer-output-price",
        type=float,
        default=0.42,
        help="Answerer output token price per 1M tokens",
    )
    args = parser.parse_args()

    paths: list[Path] = args.paths
    questioner_input_price: float = args.questioner_input_price
    questioner_output_price: float = args.questioner_output_price
    answerer_input_price: float = args.answerer_input_price
    answerer_output_price: float = args.answerer_output_price

    results = []
    for dir_path in paths:
        run_evals: list[RunEval] = []
        for i, path in enumerate(
            tqdm(dir_path.rglob("*run.json"), desc=f"Loading {dir_path.name}")
        ):
            with path.open("r", encoding="utf-8") as f:
                run_record = deserialise_run_record(json.load(f))
            run_evals.append(get_run_eval(run_record))

        group_eval = get_group_eval(run_evals)
        cost = {
            "questioner_input_price": (questioner_input_price / 1_000_000)
            * group_eval["questioner_input_tokens"],
            "questioner_output_price": (questioner_output_price / 1_000_000)
            * group_eval["questioner_output_tokens"],
            "answerer_input_price": (answerer_input_price / 1_000_000)
            * group_eval["answerer_input_tokens"],
            "answerer_output_price": (answerer_output_price / 1_000_000)
            * group_eval["answerer_output_tokens"],
        }
        total_cost = sum(cost.values())

        results.append(
            {
                "experiment": dir_path.as_posix().split("/")[-1],
                "num_runs": group_eval["num_runs"],
                "top1": group_eval["top1"],
                "top3": group_eval["top3"],
                "mean_conversation_length": group_eval["mean_conversation_length"],
                "mean_conversation_length_in_successful_cases": group_eval[
                    "mean_conversation_length_in_successful_cases"
                ],
                "total_cost": total_cost,
                "duration": group_eval["duration"],
            }
        )

    df = pl.DataFrame(results)

    print("\n" + "=" * 100)
    print("EXPERIMENT COMPARISON TABLE")
    print("=" * 100)

    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(df)
