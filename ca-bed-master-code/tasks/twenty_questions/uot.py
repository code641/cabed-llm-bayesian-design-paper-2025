from textwrap import dedent
from typing import override

from models import LLMRequestSession, get_response
from node import EvidenceNode, QuestionNode, get_conversation_history
from tasks.tree_task import (
    TreeTask,
    parse_answer,
    parse_binary_questions,
    parse_categorical_likelihoods,
)


class TwentyQuestionsUoT(TreeTask):
    def __init__(
        self,
        questioner_session: LLMRequestSession,
        answerer_session: LLMRequestSession,
        task_answer: str,
        max_question_nodes: int,
        max_lookahead_depth: int,
        max_conversation_depth: int,
        estimator_confidence: float,
        confidence_threshold: float,
        hypothesis_space: list[str],
    ):
        super().__init__(
            questioner_session=questioner_session,
            answerer_session=answerer_session,
            task_answer=task_answer,
            max_question_nodes=max_question_nodes,
            max_lookahead_depth=max_lookahead_depth,
            max_conversation_depth=max_conversation_depth,
            confidence_threshold=confidence_threshold,
            hypothesis_space=hypothesis_space,
            estimator_confidence=estimator_confidence,
        )

    def __str__(self) -> str:
        return (
            "Twenty Questions (UoT): "
            f"{self.questioner_session.model_key=} "
            f"{self.answerer_session.model_key=} "
            f"{self.max_question_nodes=} "
            f"{self.max_lookahead_depth=} "
            f"{self.max_conversation_depth=} "
            f"{self.confidence_threshold=} "
            f"{self.estimator_confidence=} "
        )

    @override
    async def create_initial_belief_state(self) -> dict[str, float]:
        return {
            entity: 1 / len(self.hypothesis_space) for entity in self.hypothesis_space
        }

    def _build_question_prompt(self, current_node: EvidenceNode) -> str:
        parts = []
        parts.append(
            dedent("""
                You are an expert player of the 20 Questions game. Your goal is to guess a secret entity, X. I will be impersonating the secret entity, X.
                You will ask me up to 20 questions which start with 'Is X' and can only be answered by 'Yes' or 'No', and I will answer each one truthfully based on being X.
            """).strip()
        )

        reduced_hypotheses = "\n".join(
            f"- {item}" for item in current_node.belief_state.keys()
        )
        parts.append(
            dedent(f"""
                The secret entity X is one of these:
                {reduced_hypotheses}
                """).strip()
        )

        history = get_conversation_history(current_node)
        if history:
            history_formatted = "\n".join(
                f"{idx}. Q: {q}; A: {a}" for idx, (q, a) in enumerate(history, start=1)
            )
            parts.append(
                dedent(f"""\
                The game has proceeded as follows:
                {history_formatted}
                """).strip()
            )

        parts.append(
            dedent(f"""
                Your task is to generate {self.max_question_nodes} *excellent* yes/no questions to ask next.
                The best questions are those that will help distinguish between these likely possibilities.
                Format your response in this structure:
                1. <Question 1>
                2. <Question 2>
                ...
                n. <Question n>
                """).strip()
        )
        return "\n\n".join(parts)

    @override
    async def create_questions(
        self, current_node: EvidenceNode
    ) -> dict[str, list[str]]:
        prompt = self._build_question_prompt(current_node)
        output = await get_response(
            messages=[{"role": "user", "content": prompt}],
            session=self.questioner_session,
        )
        questions = parse_binary_questions(output)
        return {question.question: question.possible_answers for question in questions}

    @override
    async def get_likelihoods(
        self, question: str, answers: list[str], hypotheses: list[str]
    ) -> dict[str, dict[str, float]]:
        hypotheses_formatted = "\n".join(f"- {h}" for h in hypotheses)
        prompt = dedent(f"""
            You are playing a game of 20 Questions.
            
            ### Possible entities
            {hypotheses_formatted}

            ### Question
            "{question}"

            ### Task
            - Interpret the question and possible answers.  
            - For each entity, assume they are the mystery entity and decide whether the answerer would most likely say 'Yes' or 'No'.  
            - Assign each entity to exactly one of 'Yes' or 'No' (no omissions, no duplicates).  
            - Use the entity names exactly as given.
            - Display the answers exactly in the order as given.

            ### Response Format

            Yes: Suspect_1, Suspect_2, ...
            No: Suspect_3, Suspect_4, ...

            ### Example

            Yes: Dog, Cookie
            No: Frog

            Do not include commentary or explanations. Return only the formatted response.
            """).strip()

        # Query LLM
        output = await get_response(
            messages=[{"role": "user", "content": prompt}],
            session=self.questioner_session,
        )
        likelihoods = parse_categorical_likelihoods(output, possible_answers=answers)
        return {
            likelihood.hypothesis: {
                ans: prob
                for ans, prob in zip(answers, likelihood.likelihoods, strict=True)
            }
            for likelihood in likelihoods
        }

    @override
    async def get_answer(self, current_node: QuestionNode) -> EvidenceNode:
        prompt = dedent(f"""\
            You are a player of the 20 Questions game. Your goal is to impersonate the secret entity, X. X is {self.task_answer}.
            I will ask up to 20 questions and you should answer each one truthfully based on being X.

            ### Instructions
            - Answer truthfully based on what X is.  
            - You must ONLY respond with either 'Yes' or 'No', matching it EXACTLY.
            - Do not add extra text or commentary. Return exactly one of the options.

            ### Question
            "{current_node.question}"
            """).strip()

        output = await get_response(
            messages=[{"role": "user", "content": prompt}],
            session=self.answerer_session,
        )
        return parse_answer(output, current_node)
