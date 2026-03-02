from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
import os

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chains import generation_chain, reflection_chain

class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

CRITIQUE = "critique"
GENERATE = "generate"

def generation_node(state: GraphState):
    return {
        "messages": [generation_chain.invoke({"messages": state["messages"]})]
    }

def critique_node(state: GraphState):
    res = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


graphBuilder = StateGraph(state_schema=GraphState)
graphBuilder.add_node(GENERATE, generation_node)
graphBuilder.add_node(CRITIQUE, critique_node)
graphBuilder.set_entry_point(GENERATE)

def should_continue(state: GraphState):
    if len(state["messages"]) > 6:
        return END
    return CRITIQUE

graphBuilder.add_conditional_edges(source=GENERATE, path=should_continue, path_map={END: END, CRITIQUE: CRITIQUE})
graphBuilder.add_edge(CRITIQUE, GENERATE)

graph = graphBuilder.compile()
print(graph.get_graph().draw_mermaid())



def main():
    print("Hello from critique-agent!")
    print(os.environ['OPENAI_API_KEY'])



if __name__ == "__main__":
    main()
