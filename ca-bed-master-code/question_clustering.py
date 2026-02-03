import asyncio
from dataclasses import dataclass, field
import logging

from voyager import Index, Space

from globals import SENTENCE_TRANSFORMER

logger = logging.getLogger("Question Clustering")


@dataclass
class Cluster:
    questions: dict[str, int]
    # hypothesis -> answer -> likelihood
    likelihoods: dict[str, dict[str, float]] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=lambda: asyncio.Lock())

    def get_hypotheses(self) -> list[str]:
        return list(self.likelihoods.keys())

    def get_answers(self) -> list[str]:
        if not self.likelihoods:
            return []

        it = iter(self.likelihoods.values())
        first_keys = set(next(it).keys())

        assert all(set(answers.keys()) == first_keys for answers in it), (
            "Hypotheses in cluster do not have consistent answers!"
        )

        return list(first_keys)

    def get_likelihoods_for_answer(self, answer: str) -> dict[str, float]:
        return {
            hypo: hypo_likelihoods[answer]
            for hypo, hypo_likelihoods in self.likelihoods.items()
        }


class QuestionClustering:
    index: Index
    clusters: dict[str, Cluster]
    threshold: float

    def __init__(self, threshold: float):
        logger.info(f"Setting up question cluster with threshold '{threshold}'")
        self.index = Index(Space.Cosine, num_dimensions=768)
        self.threshold = threshold
        self.clusters = {}

    def get_cluster(self, question: str) -> Cluster:
        embedding = SENTENCE_TRANSFORMER.encode(
            question, convert_to_numpy=True, normalize_embeddings=False
        )
        neighbours, distances = (
            self.index.query(embedding, k=1) if len(self.clusters) >= 1 else ([], [])
        )

        if len(neighbours) > 0 and 1 - distances[0] >= self.threshold:
            best_cluster = self.clusters[str(neighbours[0])]
            logger.info(
                f"Cluster found for '{question}', with similarity {1 - distances[0]}!"
            )
            best_cluster.questions[question] = (
                best_cluster.questions.get(question, 0) + 1
            )
            return best_cluster

        logger.info(f"Cluster not found for '{question}'. Creating new cluster...")
        idx = str(self.index.add_item(embedding))
        new_cluster = Cluster(
            {question: 1},
        )
        self.clusters[idx] = new_cluster
        return new_cluster
