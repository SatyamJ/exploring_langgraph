from langgraph.graph import StateGraph, END

from state import GraphState
from nodes import generation_node, critique_node, bump_iteration_node, should_continue


def build_graph(generation_chain, reflection_chain):
    builder = StateGraph(GraphState)


    builder.add_node("generate", lambda s: generation_node(s, generation_chain))
    builder.add_node("critique", lambda s: critique_node(s, reflection_chain))
    builder.add_node("bump_iter", bump_iteration_node)

    builder.set_entry_point("generate")
    builder.add_edge("generate", "critique")
    builder.add_edge("critique", "bump_iter")

    builder.add_conditional_edges(
        "bump_iter",
        should_continue,
        {
            "continue": "generate",
            "end": END,
        },
    )

    return builder.compile()
