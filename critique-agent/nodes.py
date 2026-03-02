from langchain_core.messages import HumanMessage

from state import GraphState
from chains import Critique


def generation_node(state: GraphState, generation_chain) -> dict:
    messages = state.get("messages", [])

    ai_msg = generation_chain.invoke({"messages": messages})

    return {
        "messages": [ai_msg],
        "final_post": ai_msg.content,
    }


def critique_node(state: GraphState, reflection_chain) -> dict:
    messages = state.get("messages", [])

    critique: Critique = reflection_chain.invoke({"messages": messages})


    critique_text = (
            f"Score: {critique.score}/10\n\n"
            f"Weaknesses:\n- " + "\n- ".join(critique.weaknesses) + "\n\n"
            f"Improvements:\n- " + "\n- ".join(critique.improvements)
    )

    return {
        "score": critique.score,
        "weaknesses": critique.weaknesses,
        "improvements": critique.improvements,
        "messages": [HumanMessage(content=critique_text)],
    }


def bump_iteration_node(state: GraphState) -> dict:
    return {"iteration": state.get("iteration", 0) + 1}


def should_continue(state: GraphState) -> str:
    score = state.get("score", 0)
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    if score >= 7:
        return "end"
    if iteration >= max_iterations:
        return "end"
    return "continue"
