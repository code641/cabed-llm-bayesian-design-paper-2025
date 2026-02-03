from abc import ABC, abstractmethod
from dataclasses import dataclass
import re

import numpy as np
from globals import SENTENCE_TRANSFORMER

from models import LLMRequestSession
from node import EvidenceNode, QuestionNode


class TreeTask(ABC):
    questioner_session: LLMRequestSession
    answerer_session: LLMRequestSession
    task_answer: str
    max_question_nodes: int
    max_lookahead_depth: int
    max_conversation_depth: int
    confidence_threshold: float
    estimator_confidence: float
    hypothesis_space: list[str]

    def __init__(
        self,
        questioner_session: LLMRequestSession,
        answerer_session: LLMRequestSession,
        task_answer: str,
        max_question_nodes: int,
        max_lookahead_depth: int,
        max_conversation_depth: int,
        confidence_threshold: float,
        estimator_confidence: float,
        hypothesis_space: list[str],
    ):
        self.questioner_session = questioner_session
        self.answerer_session = answerer_session
        self.task_answer = task_answer
        self.max_question_nodes = max_question_nodes
        self.max_lookahead_depth = max_lookahead_depth
        self.max_conversation_depth = max_conversation_depth
        self.confidence_threshold = confidence_threshold
        self.estimator_confidence = estimator_confidence
        self.hypothesis_space = hypothesis_space

    @abstractmethod
    async def create_initial_belief_state(self) -> dict[str, float]:
        pass

    @abstractmethod
    async def create_questions(
        self, current_node: EvidenceNode
    ) -> dict[str, list[str]]:
        pass

    @abstractmethod
    async def get_likelihoods(
        self, question: str, answers: list[str], hypotheses: list[str]
    ) -> dict[str, dict[str, float]]:
        pass

    @abstractmethod
    async def get_answer(self, current_node: QuestionNode) -> EvidenceNode:
        pass


def normalise_logprobs(logprobs_dict: dict[str, float]) -> dict[str, float]:
    """
    Converts a dictionary of logprobs into a normalised
    probability distribution.
    """
    keys = list(logprobs_dict.keys())
    logprobs_array = np.array(list(logprobs_dict.values()))

    max_logprob = np.max(logprobs_array)

    # If all logprobs are near 0, then return uniform
    if max_logprob == -np.inf:
        num_keys = len(keys)
        if num_keys == 0:
            return {}
        return {key: 1.0 / num_keys for key in keys}

    shifted_logprobs = logprobs_array - max_logprob
    exp_logprobs = np.exp(shifted_logprobs)
    sum_exp_logprobs = np.sum(exp_logprobs)

    normalised_probs = exp_logprobs / sum_exp_logprobs

    return {key: float(prob) for key, prob in zip(keys, normalised_probs)}


@dataclass
class Question:
    question: str
    possible_answers: list[str]


def parse_multi_questions(output: str) -> list[Question]:
    output.replace("\\n", "\n")

    questions: list[Question] = []

    # Regex to find lines starting with optional whitespace, one or more digits,
    # a period, more whitespace, and then captures the rest of the line.
    question_pattern = re.compile(r"^\s*\d+\.\s+(.*)")

    for line in output.splitlines():
        match = question_pattern.match(line)
        if match:
            # group(1) contains the captured question text
            question_text = match.group(1).strip()
            question_text, *possible_answers = question_text.split("|")
            questions.append(
                Question(question=question_text, possible_answers=possible_answers)
            )

    return questions


def parse_binary_questions(output: str) -> list[Question]:
    output.replace("\\n", "\n")

    questions: list[Question] = []

    # Regex to find lines starting with optional whitespace, one or more digits,
    # a period, more whitespace, and then captures the rest of the line.
    question_pattern = re.compile(r"^\s*\d+\.\s+(.*)")

    for line in output.splitlines():
        match = question_pattern.match(line)
        if match:
            # group(1) contains the captured question text
            question_text = match.group(1).strip()
            questions.append(
                Question(question=question_text, possible_answers=["Yes", "No"])
            )

    return questions


@dataclass
class Likelihood:
    hypothesis: str
    likelihoods: list[float]


def parse_categorical_likelihoods(
    output: str, possible_answers: list[str]
) -> list[Likelihood]:
    output = output.replace("\\n", "\n")

    label_to_items: dict[str, list[str]] = {}

    # Match lines like "Label: item1, item2"
    line_pattern = re.compile(r"^\s*([^:]+):\s*(.*)$")

    for line in output.splitlines():
        if match := line_pattern.match(line):
            label = match.group(1).strip()
            items = [s.strip() for s in match.group(2).split(",") if s.strip()]
            label_to_items[label] = items

    all_items = sorted({item for items in label_to_items.values() for item in items})

    likelihoods: list[Likelihood] = []

    for item in all_items:
        vector = [
            1.0 if item in label_to_items[label] else 1e-5 for label in possible_answers
        ]
        likelihoods.append(Likelihood(hypothesis=item, likelihoods=vector))

    return likelihoods


def parse_answer(output: str, question_node: QuestionNode) -> EvidenceNode:
    llm_answer = output.strip().lower()

    # First try exact match (case-insensitive)
    for child in question_node.children:
        if child.answer.strip().lower() == llm_answer:
            return child

    # Fall back to semantic similarity
    candidate_answers = [c.answer.strip() for c in question_node.children]
    answer_embeddings = SENTENCE_TRANSFORMER.encode(
        candidate_answers, convert_to_tensor=True, normalize_embeddings=True
    )
    output_embedding = SENTENCE_TRANSFORMER.encode(
        [llm_answer], convert_to_tensor=True, normalize_embeddings=True
    )
    similarities = SENTENCE_TRANSFORMER.similarity(
        output_embedding, answer_embeddings
    ).squeeze(0)
    best_idx = int(similarities.argmax().item())
    best_child = question_node.children[best_idx]

    return best_child
