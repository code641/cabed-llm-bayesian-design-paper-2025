from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Literal, TypedDict
from models import LLMRequestSession
from node import EvidenceNode, QuestionNode
from question_clustering import Cluster, QuestionClustering
from voyager import Index


class SerialisedQuestionNode(TypedDict):
    type: Literal["question"]
    question: str
    possible_answers: list[str]
    children: list["SerialisedEvidenceNode"]


class SerialisedEvidenceNode(TypedDict):
    type: Literal["evidence"]
    answer: str
    belief_state: dict[str, float]
    marginal_likelihood: float
    children: list["SerialisedQuestionNode"]


def serialise_question_node(node: QuestionNode) -> SerialisedQuestionNode:
    return {
        "type": "question",
        "question": node.question,
        "possible_answers": node.possible_answers,
        "children": [serialise_evidence_node(child) for child in node.children],
    }


def serialise_evidence_node(node: EvidenceNode) -> SerialisedEvidenceNode:
    return {
        "type": "evidence",
        "answer": node.answer,
        "belief_state": node.belief_state,
        "marginal_likelihood": node.marginal_likelihood,
        "children": [serialise_question_node(child) for child in node.children],
    }


def deserialise_question_node(
    node: SerialisedQuestionNode, parent: EvidenceNode
) -> QuestionNode:
    deserialised_node = QuestionNode(
        question=node["question"],
        possible_answers=node["possible_answers"],
        parent=parent,
    )
    deserialised_node.children = [
        deserialise_evidence_node(child, parent=deserialised_node)
        for child in node["children"]
    ]
    return deserialised_node


def deserialise_evidence_node(
    node: SerialisedEvidenceNode, parent: QuestionNode | None = None
) -> EvidenceNode:
    deserialised_node = EvidenceNode(
        answer=node["answer"],
        belief_state=node["belief_state"],
        marginal_likelihood=node["marginal_likelihood"],
        parent=parent,
    )
    deserialised_node.children = [
        deserialise_question_node(child, parent=deserialised_node)
        for child in node["children"]
    ]
    return deserialised_node


@dataclass
class RunRecord:
    task_info: str
    questioner_session: LLMRequestSession
    answerer_session: LLMRequestSession
    expected_answer: str
    start_time: datetime
    end_time: datetime
    final_path: list[str]
    final_belief_state: dict[str, float]
    serialised_tree: SerialisedEvidenceNode | None


def serialise_run_record(run_record: RunRecord) -> dict:
    return {
        "task_info": run_record.task_info,
        "expected_answer": run_record.expected_answer,
        "questioner_session": {
            "model_key": run_record.questioner_session.model_key,
            "total_input_tokens": run_record.questioner_session.total_input_tokens,
            "total_output_tokens": run_record.questioner_session.total_output_tokens,
        },
        "answerer_session": {
            "model_key": run_record.answerer_session.model_key,
            "total_input_tokens": run_record.answerer_session.total_input_tokens,
            "total_output_tokens": run_record.answerer_session.total_output_tokens,
        },
        "start_time": run_record.start_time.isoformat(),
        "end_time": run_record.end_time.isoformat(),
        "final_path": run_record.final_path,
        "final_belief_state": run_record.final_belief_state,
        "serialised_tree": run_record.serialised_tree,
    }


def deserialise_run_record(
    run_record: dict,
    include_tree: bool = False,
) -> RunRecord:
    return RunRecord(
        task_info=run_record["task_info"],
        questioner_session=LLMRequestSession(
            model_key=run_record["questioner_session"]["model_key"],
            total_input_tokens=run_record["questioner_session"]["total_input_tokens"],
            total_output_tokens=run_record["questioner_session"]["total_output_tokens"],
        ),
        answerer_session=LLMRequestSession(
            model_key=run_record["answerer_session"]["model_key"],
            total_input_tokens=run_record["answerer_session"]["total_input_tokens"],
            total_output_tokens=run_record["answerer_session"]["total_output_tokens"],
        ),
        expected_answer=run_record["expected_answer"],
        start_time=datetime.fromisoformat(run_record["start_time"]),
        end_time=datetime.fromisoformat(run_record["end_time"]),
        serialised_tree=run_record["serialised_tree"] if include_tree else None,
        final_path=run_record["final_path"],
        final_belief_state=run_record["final_belief_state"],
    )


def save_question_clustering(
    clustering: QuestionClustering, json_path: Path, voyager_path: Path
) -> None:
    serialised_clusters = {
        key: {
            "questions": cluster.questions,
            "likelihoods": cluster.likelihoods,
        }
        for key, cluster in clustering.clusters.items()
    }

    json_cluster = {"clusters": serialised_clusters, "threshold": clustering.threshold}

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_cluster, f)

    clustering.index.save(str(voyager_path))


def load_question_clustering(json_path: Path, voyager_path: Path) -> QuestionClustering:
    clustering = QuestionClustering(threshold=-1)

    with json_path.open("r", encoding="utf-8") as f:
        qc_dict = json.load(f)

    clustering.threshold = qc_dict["threshold"]

    for key, cluster_dict in qc_dict["clusters"].items():
        cluster = Cluster(
            questions=cluster_dict["questions"],
            likelihoods=cluster_dict["likelihoods"],
        )
        clustering.clusters[key] = cluster

    with voyager_path.open("rb") as f:
        clustering.index = Index.load(f)

    return clustering
