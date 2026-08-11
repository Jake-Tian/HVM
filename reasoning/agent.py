import operator
import os
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from reasoning.tools import get_tools
from utils.llm_gpt import MODEL
from reasoning.prompts import prompt_planner_system, prompt_planner_strategy, prompt_final_answer
from reasoning.runtime import (
    execute_tool_calls,
    last_tool_call_names,
    response_tokens,
    verifier_hints,
)


def _llm():
    # Use Chat Completions (not Responses API) so multi-turn history does not
    # re-send reasoning item IDs (rs_*) that OpenAI-compatible proxies often drop.
    kwargs = {
        "model": MODEL,
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "use_responses_api": False,
    }
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


class AgentState(TypedDict):
    question: str
    messages: Annotated[list[AnyMessage], operator.add]
    findings: Annotated[list[str], operator.add]
    clip_history: Annotated[list[int], operator.add]
    budget: int
    total_tokens: Annotated[int, operator.add]

def build_agent(graph, video_name):
    tools = get_tools(graph, video_name)
    tools_by_name = {tool.name: tool for tool in tools}
    llm = _llm().bind_tools(tools)

    def planner(state: AgentState):
        budget = state.get("budget", 5)
        if budget <= 0:
            force_msg = AIMessage(
                content="Budget exhausted. Moving to final answer.",
                tool_calls=[{"name": "complete_task", "args": {"ready": True}, "id": "force_complete"}]
            )
            return {"messages": [force_msg], "budget": budget}
            
        # 1. Fundamental Identity and Hard Rules
        sys_prompt = SystemMessage(content=prompt_planner_system.format(budget=budget))

        # 2. Operational Strategy (The 'Tail' Instruction)
        is_first_round = len(state["messages"]) <= 1
        current_strategy = prompt_planner_strategy
        if is_first_round:
            current_strategy = "**FIRST ROUND**: You MUST use `general_search` FIRST and allocate the FULL budget (Total k=50) to get a comprehensive view of the video. (If appearance is irrelevant, set k_appearance to 0 and distribute its budget to k_action or k_conversation).\n\n" + current_strategy

        strategy_reminder = HumanMessage(content=current_strategy)
        
        # Invoke LLM with System Prompt + History + Strategy Reminder
        response = llm.invoke([sys_prompt] + state["messages"] + [strategy_reminder])
        
        # Enforce tool calling to prevent wandering
        if not response.tool_calls:
            response.tool_calls = [{"name": "complete_task", "args": {"ready": True}, "id": "auto_complete"}]
            
        print(f"[Planner] Thought: {response.content}")
        if response.tool_calls:
            print(f"[Planner] Calling: {response.tool_calls[0]['name']} with {response.tool_calls[0]['args']}")

        return {
            "messages": [response],
            "budget": budget - 1,
            "total_tokens": response_tokens(response),
        }

    def executor(state: AgentState):
        return execute_tool_calls(
            state["messages"][-1],
            tools_by_name,
            state.get("clip_history", []),
        )

    def verifier(state: AgentState):
        hints = verifier_hints(state["messages"][-1])

        if hints:
            print(f"[Verifier] Hint: {hints[0].content}")
            return {"messages": hints, "total_tokens": 0}
        return {"total_tokens": 0}

    def final_answer(state: AgentState):
        sys_prompt = SystemMessage(content=prompt_final_answer)
        final_llm = _llm()
        response = final_llm.invoke([sys_prompt] + state["messages"])
        print(f"[Final Answer] Output: {response.content}")

        return {"messages": [response], "total_tokens": response_tokens(response)}

    def route_after_executor(state: AgentState) -> Literal["verifier", "final_answer"]:
        tool_call_names = last_tool_call_names(state["messages"])
        if "complete_task" in tool_call_names:
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
