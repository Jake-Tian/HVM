import operator
from typing import Literal
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END

from utils.general import extract_choice_from_content
from utils.search_tools import get_tools
from classes.hetero_graph import HeteroGraph

# Hardcode the model as in HVM-Ego
model = init_chat_model("gpt-4o-mini")
MAX_SEARCH_ROUNDS = 5
MAX_RETRIES_PER_ROUND = 1

from utils.prompts import prompt_agent, prompt_answer_with_search_results_final

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    retry_count: int

def build_graph_reason_human_prompt(current_round: int, max_rounds: int, query: str, choices: dict, graph: HeteroGraph) -> str:
    prompt = f"This is round {current_round} of {max_rounds}."

    # Graph summary logic
    main_char = graph.get_main_character()
    main_char_info = f"Main Character of the video: {main_char}" if main_char else "Main Character: None explicitly defined"
    
    char_list = list(graph.characters.keys())
    char_info = f"Characters in graph: {', '.join(char_list) if char_list else 'None'}"
    
    high_level_count = sum(1 for e in graph.edges.values() if e.clip_id == 0)
    low_level_count = len(graph.edges) - high_level_count
    
    edge_info = f"Total Edges: {len(graph.edges)} (High-level: {high_level_count}, Low-level: {low_level_count})"
    conv_info = f"Total Conversations: {len(graph.conversations)}"
    
    stats_summary = f"--- Graph Stats ---\n{main_char_info}\n{char_info}\n{edge_info}\n{conv_info}\n-------------------"

    if hasattr(graph, 'graph_summary'):
        graph_summary = stats_summary + "\n" + graph.graph_summary()
    else:
        # Fallback if HeteroGraph in Hippo doesn't have graph_summary
        # Let's extract some high level summary
        high_level = []
        for edge in graph.edges.values():
            if edge.clip_id == 0 and edge.scene in ["high-level", "appearance"]:
                target_str = edge.target if edge.target is not None else ""
                high_level.append(f"{edge.source}, {edge.content}, {target_str}".strip(", "))
        graph_summary = stats_summary + "\nGraph Summary:\n" + "\n".join(high_level[:100]) # top 100

    prompt += "\n" + graph_summary + "\n"

    if current_round >= max_rounds:
        prompt += prompt_answer_with_search_results_final
    else:
        prompt += prompt_agent

    prompt += f"\nQuestion: {query if query is not None else ''}"
    prompt += f"\nChoices: {choices if choices is not None else ''}"
    return prompt


def build_agent(graph: HeteroGraph, query: str, choices: dict, video_name: str):
    tools = get_tools(graph, video_name, query)
    tools_by_name = {tool.name: tool for tool in tools}
    model_with_tools = model.bind_tools(tools)
    
    def llm_call(state: dict):
        current_round = state.get("llm_calls", 0) + 1
        retry_count = state.get("retry_count", 0)
        prompt_content = build_graph_reason_human_prompt(current_round, MAX_SEARCH_ROUNDS, query, choices, graph)
        
        messages_to_send = state["messages"]
        if retry_count > 0:
            messages_to_send = messages_to_send + [
                HumanMessage(content="You said information is insufficient but did not call a search tool. You MUST call a tool (e.g., general_search) to continue searching, or provide a final answer (A/B/C/D) if you are ready.")
            ]

        llm_input = [
            SystemMessage(content="You are a graph QA assistant."),
            HumanMessage(content=prompt_content),
        ] + messages_to_send

        if current_round >= MAX_SEARCH_ROUNDS:
            response = model.invoke(llm_input)
        else:
            response = model_with_tools.invoke(llm_input)

        return {
            "messages": [response],
            "llm_calls": current_round
        }

    def tool_node(state: dict):
        result = []
        for tool_call in state["messages"][-1].tool_calls:
            tool_name = tool_call["name"]
            try:
                tool_fn = tools_by_name[tool_name]
                observation = tool_fn.invoke(tool_call["args"])
                result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
            except Exception as e:
                error_msg = f"Error calling tool '{tool_name}': {str(e)}. Please check your arguments and try again."
                result.append(ToolMessage(content=error_msg, tool_call_id=tool_call["id"]))
                
        return {"messages": result, "retry_count": 0}

    def guard_node(state: dict):
        return {"retry_count": state.get("retry_count", 0) + 1}

    def should_continue(state: MessagesState) -> Literal["guard_node", "tool_node", END]:
        messages = state["messages"]
        last_message = messages[-1]

        if last_message.tool_calls:
            return "tool_node"

        current_round = state.get("llm_calls", 0)
        retry_count = state.get("retry_count", 0)
        
        if current_round < MAX_SEARCH_ROUNDS:
            choice = extract_choice_from_content(last_message.content)
            if len(choice) != 1 and retry_count < MAX_RETRIES_PER_ROUND:
                return "guard_node"

        return END

    agent_builder = StateGraph(MessagesState)

    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)
    agent_builder.add_node("guard_node", guard_node)

    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        {
            "tool_node": "tool_node",
            "guard_node": "guard_node",
            END: END
        }
    )
    agent_builder.add_edge("tool_node", "llm_call")
    agent_builder.add_edge("guard_node", "llm_call")

    agent = agent_builder.compile()
    return agent
