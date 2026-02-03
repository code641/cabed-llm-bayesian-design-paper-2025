import re
from textwrap import dedent
from typing import override

from models import LLMRequestSession, get_response
from node import EvidenceNode, get_conversation_history
from tasks.detective_cases.common import get_case_background, parse_question
from tasks.detective_cases.data import DetectiveCasesInstance
from tasks.direct_prompting_task import (
    DirectPromptingTask,
    Prediction,
    Question,
)


class DetectiveCasesDirect(DirectPromptingTask):
    instance: DetectiveCasesInstance

    def __init__(
        self,
        questioner_session: LLMRequestSession,
        answerer_session: LLMRequestSession,
        instance: DetectiveCasesInstance,
        max_conversation_depth: int,
    ):
        self.instance = instance
        super().__init__(
            questioner_session=questioner_session,
            answerer_session=answerer_session,
            task_answer=next(
                suspect["name"]
                for suspect in self.instance["suspects"]
                if suspect.get("is_murderer", False)
            ),
            max_conversation_depth=max_conversation_depth,
            hypothesis_space=[suspect["name"] for suspect in self.instance["suspects"]],
        )

    def __str__(self) -> str:
        return f"Detective Cases (Direct): {self.task_answer=} {self.max_conversation_depth=} {self.hypothesis_space=}"

    @override
    async def query_questioner(
        self, current_node: EvidenceNode
    ) -> Question | Prediction:
        parts = []

        # Case background
        parts.append(
            dedent(f"""\
            You are a detective investigating a murder.  

            ### Case Background
            {get_case_background(self.instance)}
            """).strip()
        )

        # Suspects info
        suspects_info_parts = []
        for idx, suspect in enumerate(self.instance["suspects"], start=1):
            suspects_info_parts.append(
                dedent(f"""\
                - Suspect {idx}:
                    - Name: {suspect["name"]}
                    - Introduction: {suspect["introduction"]}
                """).strip()
            )
        suspects_info = "\n".join(suspects_info_parts)
        parts.append(
            dedent(f"""\
            The investigation focuses on {len(self.hypothesis_space)} suspects:
            {suspects_info}
            """).strip()
        )

        # Conversation history
        history = get_conversation_history(current_node)
        if history:
            history_formatted = "\n".join(f"- Q: {q}; A: {a}" for q, a in history)
            parts.append(
                dedent(f"""\
                These are the questions you've already asked so far:
                {history_formatted}
                """).strip()
            )

        # Instructions
        parts.append(
            dedent("""\
            ### Task
            Your goal is to identify the correct culprit.
            You can either ask a question to a specific suspect to gather more information,
            or you can make a prediction. If you ask a question, it MUST only be answerable
            by either a 'Yes' or 'No'.
                   
            ### Response Format
            If you are confident enough to make a prediction, output:
            [PREDICTION]: <Exact suspect name>
                   
            E.g., [PREDICTION]: Dr. Rose

            Otherwise, if you need more information, output:
            [QUESTION]: [Suspect Name] <Question text>
            
            E.g., [QUESTION]: [Professor Karpov] Where were you at 12:00PM?
            """)
        )

        # Targetting prompt
        if len(history) >= self.max_conversation_depth - 3:
            parts.append(
                dedent("""
                Now you should make predicitions instead of asking questions
                """).strip()
            )

        # Query LLM
        prompt = "\n\n".join(parts)
        output = await get_response(
            messages=[{"role": "user", "content": prompt}],
            session=self.questioner_session,
        )

        # Parse LLM
        question_match = re.search(r"\[QUESTION\]:\s*(.*)", output, re.IGNORECASE)
        prediction_match = re.search(
            r"\[(PREDICTION|ANSWER|PREDECTION)\]:\s*(.*)", output, re.IGNORECASE
        )

        if question_match and parse_question(
            self.hypothesis_space, question_match.group(1).strip()
        ):
            return Question(question_match.group(1).strip())
        elif prediction_match:
            return Prediction(prediction_match.group(2).strip())
        else:
            raise RuntimeError(f"Response does not match expected structure, {output}")

    @override
    async def query_answerer(self, question: str) -> str:
        suspect_name, actual_question = parse_question(self.hypothesis_space, question)

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

            ### Detective's Question
            "{actual_question}"
        """)

        return await get_response(
            messages=[{"role": "user", "content": prompt}],
            session=self.answerer_session,
        )
