import asyncio
from textwrap import dedent
from typing import override

from models import (
    LLMRequestSession,
    get_response,
    get_top_logprobs_for_messages,
)
from node import EvidenceNode, QuestionNode, get_conversation_history
from tasks.tree_task import (
    TreeTask,
    normalise_logprobs,
    parse_answer,
    parse_binary_questions,
)


class TwentyQuestionsBayesian(TreeTask):
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
            estimator_confidence=estimator_confidence,
            confidence_threshold=confidence_threshold,
            hypothesis_space=hypothesis_space,
        )

    def __str__(self) -> str:
        return (
            "Twenty Questions (Bayesian): "
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
        tasks = []
        for hypothesis_name in hypotheses:
            if hypothesis_name not in self.hypothesis_space:
                continue

            tasks.append(
                self._get_likelihood_for_one_item(
                    item=hypothesis_name,
                    question=question,
                    target_answers=answers,
                )
            )

        results = await asyncio.gather(*tasks)
        return {hypo_name: probs for hypo_name, probs in results}

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

    async def _get_likelihood_for_one_item(
        self,
        item: str,
        question: str,
        target_answers: list[str],
    ) -> tuple[str, dict[str, float]]:
        answer_list_str = "\n".join(
            f"{idx}. {answer}" for idx, answer in enumerate(target_answers, start=1)
        )

        user_prompt = dedent(f"""\
            You are playing a game of 20 Questions.
            ---
            ### Conditional Assumption
            For the purpose of this question, **assume {item} is the secret entity.**
            ---

            ### Scenario
            You asked the following question:
            "{question}"

            ### Possible Answers
            {answer_list_str}

            ### Task
            Given that {item} is the secret entity, which answer did the answerer give?
            Respond with the number for the answer only.

            The answer was number:""").strip()

        messages_for_api = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": " "},  # Force start token
        ]

        top_logprobs_list = await get_top_logprobs_for_messages(
            messages_for_api, self.questioner_session
        )

        if not top_logprobs_list:
            num_answers = len(target_answers)
            return (
                item,
                {answer: 1.0 / num_answers for answer in target_answers},
            )

        logprob_lookup = {lp.token: lp.logprob for lp in top_logprobs_list}

        raw_logprobs = {}
        for i, answer in enumerate(target_answers, start=1):
            target_token_str = f"{i}"
            logprob_with_space = logprob_lookup.get(
                f" {target_token_str}", -float("inf")
            )
            logprob_without_space = logprob_lookup.get(target_token_str, -float("inf"))
            logprob = max(logprob_with_space, logprob_without_space)
            raw_logprobs[answer] = logprob

        normalised_probs = normalise_logprobs(raw_logprobs)
        return (item, normalised_probs)
