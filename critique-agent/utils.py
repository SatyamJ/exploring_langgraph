import json
from typing import Any, Dict
from langchain_core.messages import BaseMessage


def pretty_print_messages(messages: list[BaseMessage]) -> None:
    for i, msg in enumerate(messages, 1):
        print(f"\n[{i}] {msg.type.upper()}")
        print("-" * 70)
        print(msg.content)
        print("-" * 70)


def to_serializable_state(state: Dict[str, Any]) -> Dict[str, Any]:
    # Convert BaseMessage objects to dicts for json.dumps
    serializable = dict(state)
    if "messages" in serializable:
        serializable["messages"] = [m.dict() for m in serializable["messages"]]
    return serializable


def print_json_state(state: Dict[str, Any]) -> None:
    print(json.dumps(to_serializable_state(state), indent=2))
