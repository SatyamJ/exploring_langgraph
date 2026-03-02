from langchain_core.messages import HumanMessage

from config import get_settings
from chains import build_chains
from graph import build_graph
from utils import pretty_print_messages


def main():
    settings = get_settings()
    generation_chain, reflection_chain = build_chains(settings)
    graph = build_graph(generation_chain, reflection_chain)

    topic = "Why state machines beat prompt chaining for production LLM agents (with a LangGraph example)."

    input_state = {
        "messages": [HumanMessage(content=f"Write a LinkedIn post about: {topic}")],
        "iteration": 0,
        "max_iterations": settings.max_iterations,
    }

    response = graph.invoke(input_state)


    pretty_print_messages(response["messages"])

    print("\n\n=== FINAL SCORE ===")
    print(response.get("score"))

    print("\n=== FINAL POST ===")
    print(response.get("final_post"))


if __name__ == "__main__":
    main()
