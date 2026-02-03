import asyncio
from textwrap import dedent
from typing import override

from models import (
    LLMRequestSession,
    get_response,
    get_top_logprobs_for_messages,
)
from node import EvidenceNode, QuestionNode, get_conversation_history
from tasks.detective_cases.common import get_case_background, parse_question
from tasks.detective_cases.data import DetectiveCasesInstance
from tasks.tree_task import (
    TreeTask,
    normalise_logprobs,
    parse_answer,
    parse_multi_questions,
)


class DetectiveCasesBayesianWithMultibranching(TreeTask):
    instance: DetectiveCasesInstance
    background_info: str
    suspects_info: str

    def __init__(
        self,
        questioner_session: LLMRequestSession,
        answerer_session: LLMRequestSession,
        instance: DetectiveCasesInstance,
        max_question_nodes: int,
        max_lookahead_depth: int,
        max_conversation_depth: int,
        estimator_confidence: float,
        confidence_threshold: float,
    ):
        self.instance = instance

        self.background_info = get_case_background(self.instance)
        self.suspects_info = "\n".join(
            dedent(f"""\
            - Suspect {idx + 1}:
                - Name: {suspect["name"]}
                - Introduction: {suspect["introduction"]}
            """).strip()
            for idx, suspect in enumerate(self.instance["suspects"])
        )

        super().__init__(
            questioner_session=questioner_session,
            answerer_session=answerer_session,
            task_answer=next(
                suspect["name"]
                for suspect in self.instance["suspects"]
                if suspect.get("is_murderer", False)
            ),
            max_question_nodes=max_question_nodes,
            max_lookahead_depth=max_lookahead_depth,
            max_conversation_depth=max_conversation_depth,
            confidence_threshold=confidence_threshold,
            estimator_confidence=estimator_confidence,
            hypothesis_space=[suspect["name"] for suspect in self.instance["suspects"]],
        )

    def __str__(self) -> str:
        return (
            "Detective Cases (Bayesian + Multibranching): "
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
            suspect: 1 / len(self.hypothesis_space) for suspect in self.hypothesis_space
        }

    def _build_question_prompt(self, current_node: EvidenceNode) -> str:
        parts = []
        parts.append(
            dedent(f"""\
            You are a detective investigating a murder. You can ask up to {self.max_conversation_depth} questions.

            ### Case Background
            {self.background_info}
            """).strip()
        )

        parts.append(
            dedent(f"""\
            The investigation focuses on {len(self.hypothesis_space)} suspects:
            {self.suspects_info}
            """).strip()
        )

        history = get_conversation_history(current_node)
        if history:
            history_formatted = "\n".join(
                f"{idx}. Q: {q}; A: {a}" for idx, (q, a) in enumerate(history, start=1)
            )
            parts.append(
                dedent(f"""\
                These are the questions you've already asked so far:
                {history_formatted}
                """).strip()
            )

        parts.append(
            dedent(f"""\
            ### Task
            Generate {self.max_question_nodes} excellent interrogation questions.  
            - Each question must be explicitly directed to a specific suspect.  
            - Format the question as: "[Suspect Name] Question text".
            - Provide a realistic set of possible answers for that suspect.  
            - Focus on questions that help distinguish between suspects (motive, alibi, opportunity, access to weapon).

            ### Response Format
            One line per question:
            1. <Question 1>|Answer1|Answer2|Answer3
            2. <Question 2>|Answer1|Answer2
            ...
            n. <Question n>|Answer1|Answer2|Answer3|...|AnswerK

            ### Example
            1. [Mr. Jones] Where were you at the time of the murder?|In the kitchen|In the garden|With the victim  
            2. [Dr. Otto] Did you have access to the murder weapon?|Yes|No
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
        questions = parse_multi_questions(output)
        return {question.question: question.possible_answers for question in questions}

    @override
    async def get_likelihoods(
        self, question: str, answers: list[str], hypotheses: list[str]
    ) -> dict[str, dict[str, float]]:
        answerer_name, actual_question = parse_question(self.hypothesis_space, question)

        tasks = []
        for hypothesis_name in hypotheses:
            if hypothesis_name not in self.hypothesis_space:
                continue

            tasks.append(
                self._get_likelihood_for_one_suspect(
                    hypothesis_name=hypothesis_name,
                    answerer_name=answerer_name,
                    actual_question=actual_question,
                    target_answers=answers,
                )
            )

        results = await asyncio.gather(*tasks)
        return {hypo_name: probs for hypo_name, probs in results}

    @override
    async def get_answer(self, current_node: QuestionNode) -> EvidenceNode:
        suspect_name, question = parse_question(
            self.hypothesis_space, current_node.question
        )

        suspect = next(
            (s for s in self.instance["suspects"] if s["name"] == suspect_name),
            None,
        )
        assert suspect is not None, f"Suspect '{suspect_name}' not found in case data"
        answers = [child.answer for child in current_node.children]

        prompt = dedent(f"""\
            You are roleplaying as a suspect in a murder investigation.

            ### Suspect
            - Name: {suspect["name"]}
            - Task: {suspect["task"]}
            - Story: {suspect["story"]}

            ### Instructions
            - Answer the detective's question in character as {suspect_name}.
            - Stay consistent with your task and story.
            - You may lie, evade, or tell the truth depending on what seems natural for this suspect.
            - You must ONLY respond with one of the following option, matching it EXACTLY: {answers}
            - Do not add extra text or commentary. Return exactly one of the options.

            ### Detective's Question
            "{question}"
            """).strip()

        output = await get_response(
            messages=[{"role": "user", "content": prompt}],
            session=self.answerer_session,
        )
        return parse_answer(output, current_node)

    async def _get_likelihood_for_one_suspect(
        self,
        hypothesis_name: str,
        answerer_name: str,
        actual_question: str,
        target_answers: list[str],
    ) -> tuple[str, dict[str, float]]:
        answer_list_str = "\n".join(
            f"{idx}. {answer}" for idx, answer in enumerate(target_answers, start=1)
        )

        user_prompt = dedent(f"""\
            You are a detective investigating a murder case.

            ### Case Background
            {self.background_info}

            ### Suspects
            {self.suspects_info}

            ---
            ### Conditional Assumption
            For the purpose of this question, **assume {hypothesis_name} is the murderer.**
            ---

            ### Scenario
            You asked {answerer_name} the following question:
            "{actual_question}"

            ### Possible Answers
            {answer_list_str}

            ### Task
            Given that {hypothesis_name} is the murderer, which answer did {answerer_name} give?
            Respond with the number for the answer only.

            {answerer_name}'s answer was number:""").strip()

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
                hypothesis_name,
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
        return (hypothesis_name, normalised_probs)
