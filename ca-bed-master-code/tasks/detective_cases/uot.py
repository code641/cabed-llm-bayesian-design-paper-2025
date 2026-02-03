from textwrap import dedent
from typing import override
from models import LLMRequestSession, get_response
from node import EvidenceNode, QuestionNode, get_conversation_history
from tasks.detective_cases.common import get_case_background, parse_question
from tasks.detective_cases.data import DetectiveCasesInstance
from tasks.tree_task import (
    TreeTask,
    parse_answer,
    parse_binary_questions,
    parse_categorical_likelihoods,
)


class DetectiveCasesUoT(TreeTask):
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
            "Detective Cases (UoT): "
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
            Generate {self.max_question_nodes} excellent yes/no interrogation questions.
            - Each question must be explicitly directed to a specific suspect.
            - Format the question as: "[Suspect Name] Question text".
            - Each question can only answered by 'Yes' or 'No'
            - Focus on questions that help distinguish between suspects (motive, alibi, opportunity, access to weapon).

            ### Response Format
            One line per question:
            1. <Question 1>
            ...
            n. <Question n>

            ### Example
            1. [Mr. Jones] Were you outside at 12:00PM? 
            2. [Dr. Otto] Did you have access to the murder weapon?
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
        answerer_name, actual_question = parse_question(self.hypothesis_space, question)

        suspects = [s for s in self.instance["suspects"] if s["name"] in hypotheses]
        assert len(suspects) > 0, f"No matching suspect found in question: {question}"

        suspects_info = "\n".join(
            dedent(f"""\
            - Suspect {idx + 1}:
                - Name: {suspect["name"]}
                - Introduction: {suspect["introduction"]}
            """).strip()
            for idx, suspect in enumerate(self.instance["suspects"])
        )

        # Prompt
        prompt = dedent(f"""\
            You are a detective investigating a murder case.

            ### Case Background
            {get_case_background(self.instance)}

            ### Suspects
            {suspects_info}

            ### Question to {answerer_name}
            "{actual_question}"

            ### {len(answers)} Possible Answers
            {answers}

            ### Task
            - Interpret the question and possible answers.  
            - For each suspect, assume they are the murderer and decide whether {answerer_name} would most likely say 'Yes' or 'No'.  
            - Assign each suspect to exactly one of 'Yes' or 'No' (no omissions, no duplicates).  
            - Use the suspect names exactly as given.
            - Display the answers exactly in the order as given.

            ### Response Format

            Yes: Suspect_1, Suspect_2, ...
            No: Suspect_3, Suspect_4, ...

            ### Example

            Yes: Ms. Alice, Dr. Bob
            No: Mr. Charlie

            Do not include commentary or explanations. Return only the formatted response.
        """)

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
        suspect_name, question = parse_question(
            self.hypothesis_space, current_node.question
        )

        suspect = next(
            (s for s in self.instance["suspects"] if s["name"] == suspect_name),
            None,
        )
        assert suspect is not None, f"Suspect '{suspect_name}' not found in case data"

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
            - You must ONLY respond with either 'Yes' or 'No', matching it EXACTLY.
            - Do not add extra text or commentary. Return exactly one of the options.

            ### Detective's Question
            "{question}"
            """).strip()

        output = await get_response(
            messages=[{"role": "user", "content": prompt}],
            session=self.answerer_session,
        )
        return parse_answer(output, current_node)
