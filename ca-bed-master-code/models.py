from dataclasses import dataclass
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_token_logprob import TopLogprob
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


load_dotenv()
api_key = os.getenv("DEEPSEEK_KEY")
api_base_url = "https://api.deepseek.com"

CLIENT = AsyncOpenAI(api_key=api_key, base_url=api_base_url)


@dataclass(slots=True)
class LLMRequestSession:
    model_key: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0


@retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=3, max=60))
async def get_top_logprobs_for_messages(
    messages: list[dict[str, str]],
    session: LLMRequestSession,
) -> list[TopLogprob]:
    response = await CLIENT.chat.completions.create(
        model=session.model_key,
        messages=messages,  # type: ignore
        stream=False,
        temperature=1.0,
        logprobs=True,
        top_logprobs=20,
        max_tokens=1,
    )  # type: ignore

    prompt_tokens = response.usage.prompt_tokens  # type: ignore
    completion_tokens = response.usage.completion_tokens  # type: ignore
    session.total_input_tokens += prompt_tokens
    session.total_output_tokens += completion_tokens

    if response.choices[0].logprobs and response.choices[0].logprobs.content:
        return response.choices[0].logprobs.content[0].top_logprobs
    return []


@retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=3, max=60))
async def get_response(
    messages: list[dict[str, str]],
    session: LLMRequestSession,
) -> str:
    response = await CLIENT.chat.completions.create(
        model=session.model_key,
        messages=messages,  # type: ignore
        stream=False,
        temperature=1.0,
    )  # type: ignore

    prompt_tokens = response.usage.prompt_tokens  # type: ignore
    completion_tokens = response.usage.completion_tokens  # type: ignore
    session.total_input_tokens += prompt_tokens
    session.total_output_tokens += completion_tokens

    return response.choices[0].message.content  # type: ignore
