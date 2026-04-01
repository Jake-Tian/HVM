import operator
import pickle
from pathlib import Path
from typing import Literal
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from utils.prompts import prompt_agent, prompt_answer_with_search_results_final

from utils.search import (
    general_search,
    search_before,
    search_after,
    search_first,
    search_last,
    search_object,
    set_graph,
    get_bound_graph,
    get_bound_query,
    get_bound_choices,
)
from classes.hetero_graph import HeteroGraph


model = init_chat_model("gpt-4o-mini")
tools = [general_search, search_before, search_after, search_first, search_last, search_object]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)
MAX_SEARCH_ROUNDS = 5


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


def build_graph_reason_human_prompt(current_round: int, max_rounds: int = MAX_SEARCH_ROUNDS) -> str:
    graph = get_bound_graph()
    query = get_bound_query()
    choices = get_bound_choices()

    prompt = f"This is round {current_round} of {max_rounds}."

    graph_summary = graph.graph_summary() if graph is not None else ""
    prompt += "\n" + graph_summary + "\n"

    if current_round >= max_rounds:
        prompt += prompt_answer_with_search_results_final
    else:
        prompt += prompt_agent

    prompt += f"\nQuestion: {query if query is not None else ''}"
    prompt += f"\nChoices: {choices if choices is not None else ''}"
    return prompt


def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""
    current_round = state.get("llm_calls", 0) + 1
    llm_input = [
        SystemMessage(content="You are a graph QA assistant."),
        HumanMessage(content=build_graph_reason_human_prompt(current_round=current_round, max_rounds=MAX_SEARCH_ROUNDS)),
    ] + state["messages"]
    # print("llm_input: ", llm_input)

    # Hard guard: final round must answer, so disable tool-calling.
    if current_round >= MAX_SEARCH_ROUNDS:
        response = model.invoke(llm_input)
    else:
        response = model_with_tools.invoke(llm_input)

    return {
        "messages": [response],
        "llm_calls": current_round
    }


def tool_node(state: dict):
    """Performs the tool call with a safeguard to prevent crashes on failure."""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_name = tool_call["name"]
        try:
            tool = tools_by_name[tool_name]
            observation = tool.invoke(tool_call["args"])
            result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
        except Exception as e:
            error_msg = f"Error calling tool '{tool_name}': {str(e)}. Please check your arguments and try again."
            result.append(ToolMessage(content=error_msg, tool_call_id=tool_call["id"]))
            
    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END


# Build workflow
def build_agent():

    agent_builder = StateGraph(MessagesState)

    # Add nodes
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)

    # Add edges to connect nodes
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        ["tool_node", END]
    )
    agent_builder.add_edge("tool_node", "llm_call")

    # Compile the agent
    agent = agent_builder.compile()

    return agent