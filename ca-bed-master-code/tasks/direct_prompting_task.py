from abc import ABC, abstractmethod
from dataclasses import dataclass

from models import LLMRequestSession
from node import EvidenceNode


@dataclass
class Prediction:
    value: str


@dataclass
class Question:
    value: str


class DirectPromptingTask(ABC):
    questioner_session: LLMRequestSession
    answerer_session: LLMRequestSession
    task_answer: str
    max_conversation_depth: int
    hypothesis_space: list[str]

    def __init__(
        self,
        questioner_session: LLMRequestSession,
        answerer_session: LLMRequestSession,
        task_answer: str,
        max_conversation_depth: int,
        hypothesis_space: list[str],
    ):
        self.questioner_session = questioner_session
        self.answerer_session = answerer_session
        self.task_answer = task_answer
        self.max_conversation_depth = max_conversation_depth
        self.hypothesis_space = hypothesis_space

    @abstractmethod
    async def query_questioner(
        self, current_node: EvidenceNode
    ) -> Question | Prediction:
        pass

    @abstractmethod
    async def query_answerer(self, question: str) -> str:
        pass
