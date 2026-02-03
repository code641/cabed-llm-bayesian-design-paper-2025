import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import cast

import direct_prompting_method
from history import save_question_clustering, serialise_run_record
import method
from models import LLMRequestSession
from question_clustering import QuestionClustering
from tasks.detective_cases.data import load_all_data as load_detective_data
from tasks.detective_cases.uot import DetectiveCasesUoT
from tasks.detective_cases.bayesian import DetectiveCasesBayesian
from tasks.detective_cases.bayesian_multi import (
    DetectiveCasesBayesianWithMultibranching,
)
from tasks.detective_cases.direct import DetectiveCasesDirect
from tasks.twenty_questions.data import COMMON
from tasks.twenty_questions.uot import TwentyQuestionsUoT
from tasks.twenty_questions.bayesian import TwentyQuestionsBayesian
from tasks.twenty_questions.bayesian_multi import (
    TwentyQuestionsBayesianWithMultibranching,
)
from tasks.twenty_questions.direct import TwentyQuestionsDirect
from tasks.direct_prompting_task import DirectPromptingTask
from tasks.tree_task import TreeTask

logger = logging.getLogger("Main")
type Task = DirectPromptingTask | TreeTask


def create_task_instance(task_name: str, item, args) -> Task:
    q_session = LLMRequestSession(args.questioner_model)
    a_session = LLMRequestSession(args.answerer_model)

    # === DETECTIVE CASES ===
    if task_name == "detective_direct":
        return DetectiveCasesDirect(
            questioner_session=q_session,
            answerer_session=a_session,
            instance=item,
            max_conversation_depth=args.conversation_depth,
        )

    elif task_name == "detective_uot":
        return DetectiveCasesUoT(
            questioner_session=q_session,
            answerer_session=a_session,
            instance=item,
            max_question_nodes=args.max_question_nodes,
            max_lookahead_depth=args.max_lookahead_depth,
            max_conversation_depth=args.conversation_depth,
            confidence_threshold=args.confidence_threshold,
            estimator_confidence=args.estimator_confidence,
        )

    elif task_name == "detective_bayesian":
        return DetectiveCasesBayesian(
            questioner_session=q_session,
            answerer_session=a_session,
            instance=item,
            max_question_nodes=args.max_question_nodes,
            max_lookahead_depth=args.max_lookahead_depth,
            max_conversation_depth=args.conversation_depth,
            confidence_threshold=args.confidence_threshold,
            estimator_confidence=args.estimator_confidence,
        )

    elif task_name == "detective_bayesian_multi":
        return DetectiveCasesBayesianWithMultibranching(
            questioner_session=q_session,
            answerer_session=a_session,
            instance=item,
            max_question_nodes=args.max_question_nodes,
            max_lookahead_depth=args.max_lookahead_depth,
            max_conversation_depth=args.conversation_depth,
            confidence_threshold=args.confidence_threshold,
            estimator_confidence=args.estimator_confidence,
        )

    # === TWENTY QUESTIONS ===
    if task_name == "twentyq_direct":
        return TwentyQuestionsDirect(
            questioner_session=q_session,
            answerer_session=a_session,
            task_answer=item,
            max_conversation_depth=args.conversation_depth,
            hypothesis_space=COMMON,
        )

    elif task_name == "twentyq_uot":
        return TwentyQuestionsUoT(
            questioner_session=q_session,
            answerer_session=a_session,
            task_answer=item,
            max_question_nodes=args.max_question_nodes,
            max_lookahead_depth=args.max_lookahead_depth,
            max_conversation_depth=args.conversation_depth,
            confidence_threshold=args.confidence_threshold,
            estimator_confidence=args.estimator_confidence,
            hypothesis_space=COMMON,
        )

    elif task_name == "twentyq_bayesian":
        return TwentyQuestionsBayesian(
            questioner_session=q_session,
            answerer_session=a_session,
            task_answer=item,
            max_question_nodes=args.max_question_nodes,
            max_lookahead_depth=args.max_lookahead_depth,
            max_conversation_depth=args.conversation_depth,
            confidence_threshold=args.confidence_threshold,
            estimator_confidence=args.estimator_confidence,
            hypothesis_space=COMMON,
        )

    elif task_name == "twentyq_bayesian_multi":
        return TwentyQuestionsBayesianWithMultibranching(
            questioner_session=q_session,
            answerer_session=a_session,
            task_answer=item,
            max_question_nodes=args.max_question_nodes,
            max_lookahead_depth=args.max_lookahead_depth,
            max_conversation_depth=args.conversation_depth,
            confidence_threshold=args.confidence_threshold,
            estimator_confidence=args.estimator_confidence,
            hypothesis_space=COMMON,
        )

    raise ValueError(f"Unknown task: {task_name}")


async def main(args: argparse.Namespace) -> None:
    if args.task.startswith("detective_"):
        dataset = load_detective_data()
    else:
        dataset = COMMON

    shared_clustering = QuestionClustering(args.clustering_threshold)

    tasks = [
        create_task_instance(args.task, item, args)
        for item in dataset[args.start_idx : args.end_idx]
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.max_concurrent)

    logger.info(f"Running task '{args.task}'")
    logger.info(
        f"Questioner: {args.questioner_model} | Answerer: {args.answerer_model}"
    )

    if args.task.endswith("direct"):
        tasks = cast(list[DirectPromptingTask], tasks)
        await asyncio.gather(
            *[
                run_direct_prompting_task(i, task, output_dir, semaphore)
                for i, task in enumerate(tasks, start=args.start_idx)
            ]
        )
    else:
        tasks = cast(list[TreeTask], tasks)
        await asyncio.gather(
            *[
                run_tree_based_task(
                    idx=i,
                    task=task,
                    output_dir=output_dir,
                    semaphore=semaphore,
                    sharpness_constant=args.sharpness_constant,
                    min_probability=args.min_probability,
                    question_clustering=(
                        shared_clustering
                        if args.shared_cluster
                        else QuestionClustering(args.clustering_threshold)
                    ),
                )
                for i, task in enumerate(tasks, start=args.start_idx)
            ]
        )

    logger.info("All runs completed successfully!")


async def run_tree_based_task(
    idx: int,
    task: TreeTask,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
    sharpness_constant: float,
    min_probability: float,
    question_clustering: QuestionClustering,
) -> None:
    async with semaphore:
        run_record = await method.run_task(
            task, question_clustering, sharpness_constant, min_probability
        )

        save_question_clustering(
            question_clustering,
            output_dir / f"{idx}_cluster.json",
            output_dir / f"{idx}_cluster.voy",
        )
        with (output_dir / f"{idx}_run.json").open("w", encoding="utf-8") as f:
            json.dump(serialise_run_record(run_record), f)
        logger.info(f"[{idx}] Completed run.")


async def run_direct_prompting_task(
    idx: int, task: DirectPromptingTask, output_dir: Path, semaphore: asyncio.Semaphore
) -> None:
    async with semaphore:
        run_record = await direct_prompting_method.run_task(task)
        with (output_dir / f"{idx}_run.json").open("w", encoding="utf-8") as f:
            json.dump(serialise_run_record(run_record), f)
        logger.info(f"[{idx}] Completed direct run.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run experiments for DetectiveCases and TwentyQuestions."
    )
    subparsers = parser.add_subparsers(dest="task", required=True)

    # ========== COMMON ARGS ==========
    def add_shared_args(p):
        p.add_argument("--questioner_model", default="deepseek-chat")
        p.add_argument("--answerer_model", default="deepseek-reasoner")
        p.add_argument("--start_idx", type=int, default=0)
        p.add_argument("--end_idx", type=int, default=10)
        p.add_argument("--conversation_depth", type=int, default=20)
        p.add_argument("--max_concurrent", type=int, default=6)
        p.add_argument("--clustering_threshold", type=float, default=1.0)
        p.add_argument("--shared_cluster", action="store_true")
        p.add_argument(
            "--output_dir",
            default=f"logs/{datetime.now().strftime('%Y%m%d%H%M%S')}",
        )
        p.add_argument("--sharpness_constant", type=float, default=0.4)
        p.add_argument("--min_probability", type=float, default=1 / 25_000)

    # ========== TREE ARGS ==========
    def add_tree_args(p):
        p.add_argument("--max_question_nodes", type=int, default=2)
        p.add_argument("--max_lookahead_depth", type=int, default=3)
        p.add_argument("--confidence_threshold", type=float, default=0.8)
        p.add_argument("--estimator_confidence", type=float, default=0.7)

    # ========== DETECTIVE CASES ==========
    for name in [
        "detective_direct",
        "detective_uot",
        "detective_bayesian",
        "detective_bayesian_multi",
    ]:
        p = subparsers.add_parser(name, help=f"Run {name.replace('_', ' ').title()}")
        add_shared_args(p)
        if name != "detective_direct":
            add_tree_args(p)

    # ========== TWENTY QUESTIONS ==========
    for name in [
        "twentyq_direct",
        "twentyq_uot",
        "twentyq_bayesian",
        "twentyq_bayesian_multi",
    ]:
        p = subparsers.add_parser(name, help=f"Run {name.replace('_', ' ').title()}")
        add_shared_args(p)
        if name != "twentyq_direct":
            add_tree_args(p)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(args.output_dir) / "logs.log", encoding="utf-8"),
        ],
        force=True,
    )

    asyncio.run(main(args))
