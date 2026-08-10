"""Five-tool LangGraph planner, executor, verifier, and answer workflow."""

import operator
import os
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from classes.hetero_graph import HeteroGraph
from reasoning.tools import get_tools
from reasoning.prompts import (
    prompt_answer_with_search_results_final,
    prompt_final_answer,
    prompt_planner_strategy,
    prompt_planner_system,
)
from reasoning.runtime import (
    build_graph_stats,
    execute_tool_calls,
    last_tool_calls,
    response_tokens,
    verifier_hints,
)
from utils.llm_gpt import MODEL


DEFAULT_BUDGET = 6
_API_KEY = os.environ.get("OPENAI_API_KEY")
_BASE_URL = os.environ.get("OPENAI_BASE_URL")


class AgentState(TypedDict):
    question: str
    messages: Annotated[list[AnyMessage], operator.add]
    clip_history: Annotated[list[int], operator.add]
    tool_call_history: Annotated[list[dict], operator.add]
    budget: int
    total_tokens: Annotated[int, operator.add]


def _llm():
    kwargs = {
        "model": MODEL,
        "api_key": _API_KEY,
        "use_responses_api": True,
    }
    if _BASE_URL:
        kwargs["base_url"] = _BASE_URL
    return ChatOpenAI(**kwargs)


def _ensure_completion_call(response, call_id):
    if not getattr(response, "tool_calls", None):
        response.tool_calls = [{
            "name": "complete_task",
            "args": {"ready": True},
            "id": call_id,
        }]
    return response


def build_agent(graph: HeteroGraph, video_name: str, budget: int = DEFAULT_BUDGET):
    """Compile the five-tool reasoning graph for one video."""
    tools = get_tools(graph, video_name)
    tools_by_name = {tool.name: tool for tool in tools}
    llm = _llm()
    llm_with_tools = llm.bind_tools(tools)
    graph_stats = build_graph_stats(graph)

    def planner(state: AgentState):
        current_budget = state.get("budget", budget)
        is_first_round = not state.get("tool_call_history", [])
        force_final = current_budget <= 1

        if is_first_round:
            strategy = (
                "**FIRST ROUND**: You MUST use `general_search` FIRST and allocate "
                "the FULL budget (total k=50) to get a comprehensive view of the video. "
                "(If appearance is irrelevant, set k_appearance to 0 and redistribute "
                "its budget to k_low_level or k_conversations.)\n\n"
                + prompt_planner_strategy
            )
        elif force_final:
            strategy = prompt_answer_with_search_results_final
        else:
            strategy = prompt_planner_strategy

        messages = [
            SystemMessage(content=prompt_planner_system.format(
                budget=current_budget,
                graph_stats=graph_stats,
            )),
            *state.get("messages", []),
            HumanMessage(content=strategy),
        ]
        model = llm if force_final else llm_with_tools
        response = model.invoke(messages)
        response = _ensure_completion_call(
            response,
            "force_complete" if force_final else "auto_complete",
        )

        print(f"[Planner] (budget={current_budget}) Thought: {response.content}")
        for tool_call in response.tool_calls:
            print(f"[Planner] Calling: {tool_call['name']} with {tool_call['args']}")

        return {
            "messages": [response],
            "budget": current_budget - 1,
            "total_tokens": response_tokens(response),
        }

    def executor(state: AgentState):
        return execute_tool_calls(
            getattr(state["messages"][-1], "tool_calls", []) or [],
            tools_by_name,
            state.get("clip_history", []),
        )

    def verifier(state: AgentState):
        hints = verifier_hints(state)
        if hints:
            print(f"[Verifier] Hint: {hints[0].content}")
            return {"messages": hints, "total_tokens": 0}
        return {"total_tokens": 0}

    def final_answer(state: AgentState):
        response = llm.invoke([
            SystemMessage(content=prompt_final_answer),
            *state.get("messages", []),
        ])
        print(f"[Final Answer] Output: {response.content}")
        return {
            "messages": [response],
            "total_tokens": response_tokens(response),
        }

    def route_after_executor(
        state: AgentState,
    ) -> Literal["verifier", "final_answer"]:
        tool_calls = last_tool_calls(state.get("messages", []))
        if tool_calls and any(
            tool_call["name"] == "complete_task" for tool_call in tool_calls
        ):
            return "final_answer"
        return "verifier"

    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner)
    workflow.add_node("executor", executor)
    workflow.add_node("verifier", verifier)
    workflow.add_node("final_answer", final_answer)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_conditional_edges("executor", route_after_executor)
    workflow.add_edge("verifier", "planner")
    workflow.add_edge("final_answer", END)
    return workflow.compile()
