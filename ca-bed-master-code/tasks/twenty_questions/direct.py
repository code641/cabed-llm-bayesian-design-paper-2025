import re
from textwrap import dedent
from typing import override

from models import LLMRequestSession, get_response
from node import EvidenceNode, get_conversation_history
from tasks.direct_prompting_task import (
    DirectPromptingTask,
    Prediction,
    Question,
)


class TwentyQuestionsDirect(DirectPromptingTask):
    def __init__(
        self,
        questioner_session: LLMRequestSession,
        answerer_session: LLMRequestSession,
        task_answer: str,
        max_conversation_depth: int,
        hypothesis_space: list[str],
    ):
        super().__init__(
            questioner_session=questioner_session,
            answerer_session=answerer_session,
            task_answer=task_answer,
            max_conversation_depth=max_conversation_depth,
            hypothesis_space=hypothesis_space,
        )

    def __str__(self) -> str:
        return f"Twenty Questions (Direct): {self.task_answer=} {self.max_conversation_depth=} {self.hypothesis_space=}"

    @override
    async def query_questioner(
        self, current_node: EvidenceNode
    ) -> Question | Prediction:
        parts = []

        # Prologue
        possible_items = "\n".join(f"- {h}" for h in self.hypothesis_space)
        parts.append(
            dedent(f"""\
            You are an expert player of the 20 Questions game. Your goal is to guess a secret entity, X. I will be impersonating the secret entity, X.
            The secret entity could be one of the following:
            {possible_items}
            
            You can either ask questions that starts with 'Is X' and can only be answered by 'Yes' or 'No', or you can make a prediction of what X is.
            
            If you are confident enough to make a prediction, output:
            [PREDICTION]: <This should ONLY be the exact name of the entity from the list of possible entities>

            Otherwise, if you need more information, output:
            [QUESTION]: <Your yes/no question here>
            """)
            .format(possible_items=possible_items)
            .strip()
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

        if question_match:
            return Question(question_match.group(1).strip())
        elif prediction_match:
            return Prediction(prediction_match.group(2).strip())
        else:
            raise RuntimeError(f"Response does not match expected structure, {output}")

    @override
    async def query_answerer(self, question: str) -> str:
        prompt = dedent(f"""\
            You are a player of the 20 Questions game. Your goal is to impersonate the secret entity, X. X is {self.task_answer}.
            I will ask up to 20 questions and you should answer each one truthfully based on being X.

            ### Instructions
            - Answer truthfully based on what X is.
            - You must ONLY respond with either 'Yes' or 'No', matching it EXACTLY.
            - Do not add extra text or commentary. Return exactly one of the options.

            ### Question
            "{question}"
            """).strip()

        return await get_response(
            messages=[{"role": "user", "content": prompt}],
            session=self.answerer_session,
        )
