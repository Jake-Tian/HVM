import operator
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from utils.agent_tools import execute_general_search, execute_search_temporal_context, execute_watch_video_clip
from utils.prompts import prompt_planner_system, prompt_planner_strategy, prompt_final_answer

class AgentState(TypedDict):
    question: str
    messages: Annotated[list[AnyMessage], operator.add]
    findings: Annotated[list[str], operator.add]
    clip_history: Annotated[list[int], operator.add]
    budget: int
    total_tokens: Annotated[int, operator.add]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "general_search",
            "description": "General semantic search that covers conversation, action, and OCR information. Modify the k_budgets to specify how many results to retrieve for each category. Use this first to get an overview of events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The natural language query to search for."},
                    "k_action": {"type": "integer", "description": "Budget for action/behavior search (0-30)."},
                    "k_conversation": {"type": "integer", "description": "Budget for conversation search (0-30)."},
                    "k_ocr": {"type": "integer", "description": "Budget for OCR search (0-30)."},
                    "k_high_level": {"type": "integer", "description": "Budget for high-level attributes/relationships search (0-10)."},
                    "k_appearance": {"type": "integer", "description": "Budget for character appearance search (0-8)."}
                },
                "required": ["query", "k_action", "k_conversation", "k_ocr", "k_high_level", "k_appearance"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_temporal_context",
            "description": "Search what happened in and around a specific video clip (window=1). Use this to see events right before or after a known clip.",
            "parameters": {
                "type": "object",
                "properties": {
                    "clip_id": {"type": "integer", "description": "The clip ID to investigate."}
                },
                "required": ["clip_id"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "watch_video_clip",
            "description": "Watch the raw video frames of a specific clip. Use this ONLY to verify visual ground truth when the text graph is insufficient or ambiguous. It is expensive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "clip_id": {"type": "integer", "description": "The clip ID to watch."},
                    "focus": {"type": "string", "description": "A specific question or focus area to look for in the video."}
                },
                "required": ["clip_id", "focus"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "complete_task",
            "description": "Call this tool when you have enough information to answer the question, or when you exhaust your budget.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ready": {"type": "boolean", "description": "True if ready to answer."}
                },
                "required": ["ready"],
                "additionalProperties": False
            }
        }
    }
]

def build_agent(graph, video_name):
    llm = ChatOpenAI(model="gpt-4o", temperature=0).bind_tools(TOOLS)

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

        tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens = response.usage_metadata.get("total_tokens", 0)

        return {"messages": [response], "budget": budget - 1, "total_tokens": tokens}

    def executor(state: AgentState):
        last_message = state["messages"][-1]
        tool_results = []
        new_history = []
        total_tokens = 0
        
        for tool_call in last_message.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            obs = ""
            tokens = 0
            try:
                if name == "general_search":
                    obs, tokens = execute_general_search(
                        graph, 
                        args.get("query"), 
                        args.get("k_action", 25), 
                        args.get("k_conversation", 15), 
                        args.get("k_ocr", 5),
                        args.get("k_high_level", 5),
                        args.get("k_appearance", 0)
                    )
                elif name == "search_temporal_context":
                    obs, tokens = execute_search_temporal_context(graph, args.get("clip_id"))
                elif name == "watch_video_clip":
                    clip_id = args.get("clip_id")
                    if clip_id in state.get("clip_history", []):
                        obs = f"Warning: You already watched clip {clip_id}. Avoid redundant re-watching unless the focus is fundamentally different."
                        tokens = 0
                    else:
                        new_history.append(clip_id)
                        obs, tokens = execute_watch_video_clip(video_name, clip_id, args.get("focus"))
                elif name == "complete_task":
                    obs = "Task marked complete. Proceeding to final verification."
                    tokens = 0
                else:
                    obs = f"Unknown tool: {name}"
                    tokens = 0
            except Exception as e:
                obs = f"Tool execution failed: {e}"
                tokens = 0
            
            total_tokens += tokens
            tool_results.append(ToolMessage(content=str(obs), tool_call_id=tool_call["id"]))
            
        return {"messages": tool_results, "clip_history": new_history, "total_tokens": total_tokens}

    def verifier(state: AgentState):
        last_message = state["messages"][-1]
        content = last_message.content
        hints = []
        
        if "No relevant information found" in content or content.strip() == "":
            hints.append(SystemMessage(content="System Hint: The last search returned no useful information. Try searching for a broader entity, changing the keywords, or allocating budget differently."))
        elif "Warning: You already watched clip" in content:
            hints.append(SystemMessage(content="System Hint: Redundant tool call detected. Use the evidence you already have, or try a different clip or tool."))
            
        if hints:
            print(f"[Verifier] Hint: {hints[0].content}")
            return {"messages": hints, "total_tokens": 0}
        return {"total_tokens": 0}

    def final_answer(state: AgentState):
        sys_prompt = SystemMessage(content=prompt_final_answer)
        final_llm = ChatOpenAI(model="gpt-4o", temperature=0)
        response = final_llm.invoke([sys_prompt] + state["messages"])
        print(f"[Final Answer] Output: {response.content}")

        tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens = response.usage_metadata.get("total_tokens", 0)

        return {"messages": [response], "total_tokens": tokens}

    def route_after_executor(state: AgentState) -> Literal["verifier", "final_answer"]:
        tool_call_names = []
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    tool_call_names = [tc["name"] for tc in msg.tool_calls]
                break
                
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
