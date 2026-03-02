from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GraphState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    score: int
    weaknesses: List[str]
    improvements: List[str]
    iteration: int
    max_iterations: int
    final_post: str
