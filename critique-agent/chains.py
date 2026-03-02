from pydantic import BaseModel, Field
from typing import List

from langchain_openai import ChatOpenAI

from .config import Settings
from .prompts import build_generation_prompt, build_reflection_prompt


class Critique(BaseModel):
    score: int = Field(ge=1, le=10)
    weaknesses: List[str] = Field(min_length=1)
    improvements: List[str] = Field(min_length=1)


def build_chains(settings: Settings):
    llm = ChatOpenAI(model=settings.model, temperature=settings.temperature)

    generation_prompt = build_generation_prompt()
    reflection_prompt = build_reflection_prompt()

    generation_chain = generation_prompt | llm

    # Structured output for deterministic parsing
    reflection_chain = reflection_prompt | llm.with_structured_output(Critique)

    return generation_chain, reflection_chain
