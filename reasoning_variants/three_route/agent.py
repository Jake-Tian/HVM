"""LangGraph planner/executor/verifier/final_answer agent for HVM.

Architecture (synthesized from HVM-web / HVM-Hippo / HVM-Ego):

  START -> planner -> executor -+-> verifier -> planner  (loop)
                               +-> final_answer -> END    (when complete_task called)

  - budget-driven (default 6), not a fixed round count. Each planner call
    decrements budget; when budget <= 1 the planner is forced to stop calling
    tools and reason over collected evidence (Hippo-style final prompt).
  - First round forces a broad `general_search` (web-style).
  - Executor dedups `watch_video_clip` against `clip_history` (web-style).
  - Verifier injects system hints for: empty results, redundant clip watching,
    repeated identical tool calls (Ego-style), and temporal questions that have
    not used search_temporal_context.
  - A lightweight question router selects isolated location/action-frequency
    strategies; general questions retain the default strategy.
  - final_answer is a separate free-text LLM call (HVM is open-ended QA, not MCQ).

The model is GPT-5 mini, matching the reasoning adapters used by the HVM pipeline.
"""

import operator
import os
import re
import sys
from typing import TypedDict, Annotated, Literal

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


def _report_llm_failure(model, exc):
    """Print LLM API failures to stderr (visible even when stdout is quiet-logged)."""
    try:
        sys.stderr.write(f"[LLM ERROR] model={model}: {exc}\n")
        sys.stderr.flush()
    except Exception:
        pass

from reasoning_variants.three_route.frequency_memory import (
    format_action_frequency_memory,
    update_action_frequency_memory,
)
from reasoning_variants.three_route.tools import get_tools
from utils.llm_gpt import MODEL
from utils.token_usage import (
    empty_usage,
    merge_stage_usage,
    merge_usage,
    usage_from_response,
    usage_total,
)
from utils.prompts import (
    prompt_planner_system,
    prompt_planner_strategy,
    prompt_planner_strategy_location,
    prompt_planner_strategy_event_location,
    prompt_planner_strategy_action_frequency,
    prompt_answer_with_search_results_final,
    prompt_answer_with_search_results_final_location,
    prompt_answer_with_search_results_final_action_frequency,
    prompt_final_answer,
    prompt_final_answer_location,
    prompt_final_answer_action_frequency,
)
from classes.hetero_graph import HeteroGraph

_ACTIVE_MODEL = MODEL
_ACTIVE_PROVIDER = "openai"
_ACTIVE_API_KEY = os.environ.get("OPENAI_API_KEY")
_ACTIVE_BASE_URL = os.environ.get("OPENAI_BASE_URL")

DEFAULT_BUDGET = 6

_TEMPORAL_KEYWORDS = ("before", "after", "first", "last", "earliest", "latest",
                      "next", "previous", "then", "sequence")


class AgentState(TypedDict):
    question: str
    messages: Annotated[list[AnyMessage], operator.add]
    findings: Annotated[list[str], operator.add]
    clip_history: Annotated[list[int], operator.add]
    tool_call_history: Annotated[list[dict], operator.add]
    location_candidate_clips: Annotated[list[int], operator.add]
    location_object_name: str
    location_intent: str
    budget: int
    action_frequency_memory: dict
    total_tokens: Annotated[int, operator.add]
    token_details: Annotated[dict, merge_stage_usage]


def _build_graph_stats(graph: HeteroGraph) -> str:
    """A dense 'map' of the graph injected into the planner system prompt.

    Mirrors HVM-Hippo's build_graph_reason_human_prompt stats block so the
    planner knows what entities / clips / conversations exist to navigate.
    """
    char_list = list(graph.characters.keys())
    obj_count = len(graph.objects)

    high_level_count = sum(1 for e in graph.edges.values() if e.clip_id == 0)
    low_level_count = len(graph.edges) - high_level_count

    max_clip = 0
    for e in graph.edges.values():
        if e.clip_id is not None and e.clip_id > max_clip:
            max_clip = e.clip_id

    scenes = set()
    for e in graph.edges.values():
        if e.scene:
            scenes.add(e.scene)

    return (
        "--- Graph Stats ---\n"
        f"Characters: {', '.join(char_list) if char_list else 'None'}\n"
        f"Total Object Nodes: {obj_count}\n"
        f"Total Edges: {len(graph.edges)} (High-level: {high_level_count}, Low-level: {low_level_count})\n"
        f"Total Clips: {max_clip}\n"
        f"Total Conversations: {len(graph.conversations)}\n"
        f"Scenes: {', '.join(sorted(scenes)) if scenes else 'None'}\n"
        "Graph Construction Info:\n"
        "- The video is divided into 30-second clips (Clip 1 = 0-30s, Clip 2 = 30-60s, etc.).\n"
        "- Character nodes are enclosed in angle brackets (e.g., <robot>, <Alice>).\n"
        "- Object nodes are plain text (e.g., coffee, table).\n"
        "- High-level edges (clip_id=0) represent overall attributes and relationships.\n"
        "- Low-level edges represent specific actions/states occurring in specific clips.\n"
        "-------------------"
    )


def _extract_last_tool_calls(messages):
    """Return the tool_call list from the most recent AIMessage that has any."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            return msg.tool_calls
    return None


def _has_consecutive_general_searches(history, count=2):
    """Return whether the latest tool calls are all general searches."""
    return (
        len(history) >= count
        and all(item.get("name") == "general_search" for item in history[-count:])
    )


def classify_location_mode(question: str) -> Literal["object", "event"]:
    """Separate event-scene questions from object-state location questions."""
    text = " ".join(str(question or "").lower().split())
    if re.search(
        r"\bwhere\s+(?:did|does|do)\b.*\b"
        r"(do|work|study|sleep|eat|play|meet|sit|stand|live|happen|occur)\b",
        text,
    ):
        return "event"
    return "object"


def _location_query_contract(question: str) -> str:
    mode = classify_location_mode(question)
    return (
        "Location query contract:\n"
        f"- mode: {mode}-location\n"
        "- preserve the complete target phrase and temporal condition from the question\n"
        "- select the state requested by the question, not merely the latest mention"
    )


def _focus_preserves_location_object(object_name, focus):
    """Require the locked object's identity terms without policing locations."""
    object_text = " ".join(str(object_name or "").lower().replace("-", " ").split())
    focus_text = " ".join(str(focus or "").lower().replace("-", " ").split())
    if not object_text or not focus_text:
        return False
    object_words = re.findall(r"[a-z0-9]+", object_text)
    required = {
        word for word in object_words
        if len(word) > 2 and word not in {"the", "for", "from", "with"}
    }
    return all(re.search(rf"\b{re.escape(term)}\b", focus_text) for term in required)


def _extract_rewatch_clips(content):
    match = re.search(r"Suggested rewatch clips:\s*([^\n]+)", str(content))
    if not match or match.group(1).strip().lower() == "none":
        return []
    return [int(value) for value in re.findall(r"\d+", match.group(1))]


_LOCATION_ONLY_TOOLS = {"search_object_events", "watch_video_clip"}
_FREQUENCY_ONLY_TOOLS = {"search_action_evidence"}


def _tools_for_route(tools, route):
    """Expose route-specific tools only to the route that owns them."""
    if route == "location":
        hidden = _FREQUENCY_ONLY_TOOLS
    elif route == "action_frequency":
        hidden = _LOCATION_ONLY_TOOLS
    else:
        hidden = _LOCATION_ONLY_TOOLS | _FREQUENCY_ONLY_TOOLS
    return [tool for tool in tools if tool.name not in hidden]


def classify_question_route(
    question: str,
) -> Literal["location", "action_frequency", "general"]:
    """Route only explicit location/action-frequency forms."""
    text = " ".join(str(question or "").lower().split())
    if re.search(
        r"\b(how many times|number of times|how often|how frequently|frequency of)\b",
        text,
    ):
        return "action_frequency"
    if re.search(r"\b(where|whereabouts|location of)\b", text):
        return "location"
    if re.search(r"\bshould\b.*\b(be\s+)?(put|placed|stored|hung|laced)\b", text):
        return "location"
    if re.search(
        r"\b(is|are)\b.*\b(on|in|at|inside|beside|behind)\b.*\bor\b.*",
        text,
    ):
        return "location"
    return "general"


def build_agent(
    graph: HeteroGraph,
    video_name: str,
    budget: int = DEFAULT_BUDGET,
):
    if not _ACTIVE_API_KEY:
        raise RuntimeError(
            "An OpenAI API key is required for GPT reasoning."
        )

    tools = get_tools(graph, video_name)
    tools_by_name = {t.name: t for t in tools}

    llm_kwargs = {
        "model": _ACTIVE_MODEL,
        "api_key": _ACTIVE_API_KEY,
        "use_responses_api": True,
    }
    if _ACTIVE_BASE_URL:
        llm_kwargs["base_url"] = _ACTIVE_BASE_URL
    llm = ChatOpenAI(**llm_kwargs)
    llm_with_tools = {
        route: llm.bind_tools(_tools_for_route(tools, route))
        for route in ("general", "location", "action_frequency")
    }

    graph_stats = _build_graph_stats(graph)

    # ------------------------------------------------------------------
    # planner
    # ------------------------------------------------------------------
    def planner(state: AgentState):
        current_budget = state.get("budget", budget)
        is_first_round = len(state.get("tool_call_history", [])) == 0
        force_final = current_budget <= 1
        route = classify_question_route(state.get("question", ""))

        location_mode = classify_location_mode(state.get("question", ""))
        location_strategy = (
            prompt_planner_strategy_event_location
            if location_mode == "event"
            else prompt_planner_strategy_location
        )
        route_strategy = {
            "location": location_strategy,
            "action_frequency": prompt_planner_strategy_action_frequency,
            "general": prompt_planner_strategy,
        }[route]
        if route == "location":
            route_strategy += "\n\n" + _location_query_contract(
                state.get("question", "")
            )

        if force_final:
            route_final = {
                "location": prompt_answer_with_search_results_final_location,
                "action_frequency": prompt_answer_with_search_results_final_action_frequency,
                "general": "",
            }[route]
            if route == "location":
                route_final += "\n\n" + _location_query_contract(
                    state.get("question", "")
                )
            strategy = prompt_answer_with_search_results_final
            if route_final:
                strategy += "\n\n" + route_final
        elif is_first_round:
            strategy = (
                "**FIRST ROUND**: You MUST use `general_search` FIRST and allocate "
                "the FULL budget (total k=50) to get a comprehensive view of the video. "
                "(If appearance is irrelevant, set k_appearance to 0 and redistribute "
                "its budget to k_low_level or k_conversations.)\n\n"
                + route_strategy
            )
        else:
            strategy = route_strategy

        sys_prompt = SystemMessage(
            content=prompt_planner_system.format(budget=current_budget, graph_stats=graph_stats)
        )
        strategy_reminder = HumanMessage(content=strategy)

        memory_messages = []
        if route == "action_frequency":
            memory_messages.append(HumanMessage(
                content=(
                    "Action-frequency intermediate memory:\n"
                    + format_action_frequency_memory(
                        state.get("action_frequency_memory", {})
                    )
                )
            ))
        messages_to_send = (
            [sys_prompt]
            + state.get("messages", [])
            + memory_messages
            + [strategy_reminder]
        )

        try:
            if force_final:
                # No tools bound: force the model to reason and call complete_task.
                response = llm.invoke(messages_to_send)
                if not getattr(response, "tool_calls", None):
                    response.tool_calls = [{
                        "name": "complete_task",
                        "args": {"ready": True},
                        "id": "force_complete",
                    }]
            else:
                response = llm_with_tools[route].invoke(messages_to_send)
                if not getattr(response, "tool_calls", None):
                    response.tool_calls = [{
                        "name": "complete_task",
                        "args": {"ready": True},
                        "id": "auto_complete",
                    }]
        except Exception as e:
            _report_llm_failure(_ACTIVE_MODEL, e)
            raise

        print(f"[Planner] (budget={current_budget}) Thought: {response.content}")
        if getattr(response, "tool_calls", None):
            for tc in response.tool_calls:
                print(f"[Planner] Calling: {tc['name']} with {tc['args']}")

        usage = usage_from_response(response, _ACTIVE_MODEL, _ACTIVE_PROVIDER)

        return {
            "messages": [response],
            "budget": current_budget - 1,
            "total_tokens": usage_total(usage),
            "token_details": {"planner": usage},
        }

    # ------------------------------------------------------------------
    # executor
    # ------------------------------------------------------------------
    def executor(state: AgentState):
        last_message = state["messages"][-1]
        tool_results = []
        new_clip_history = []
        new_tool_history = []
        new_location_candidate_clips = []
        location_object_update = None
        location_intent_update = None
        tool_usage = empty_usage()
        token_details = {}

        for tool_call in getattr(last_message, "tool_calls", []) or []:
            name = tool_call["name"]
            args = tool_call["args"]
            new_tool_history.append({"name": name, "args": args})

            try:
                if name not in tools_by_name:
                    obs = f"Unknown tool: {name}"
                    tokens = 0
                elif (name in _LOCATION_ONLY_TOOLS
                      and classify_question_route(state.get("question", "")) != "location"):
                    obs = (
                        f"Tool unavailable: {name} is restricted to location questions."
                    )
                    tokens = 0
                elif (name in _FREQUENCY_ONLY_TOOLS
                      and classify_question_route(state.get("question", "")) != "action_frequency"):
                    obs = (
                        f"Tool unavailable: {name} is restricted to "
                        "action-frequency questions."
                    )
                    tokens = 0
                elif (
                    name in _LOCATION_ONLY_TOOLS
                    and classify_location_mode(state.get("question", "")) == "event"
                ):
                    obs = (
                        f"Tool unavailable: {name} is not used for event-location "
                        "questions. Use graph and temporal evidence."
                    )
                    tokens = 0
                elif (
                    name == "search_object_events"
                    and state.get("location_object_name")
                ):
                    obs = (
                        "Tool unavailable: the object timeline has already been "
                        "built. Use temporal context or a validated rewatch clip."
                    )
                    tokens = 0
                elif (
                    name == "watch_video_clip"
                    and not state.get("location_object_name")
                ):
                    obs = (
                        "Tool unavailable: call `search_object_events` first and "
                        "rewatch one of its candidate event clips."
                    )
                    tokens = 0
                elif name == "watch_video_clip" and args.get("clip_id") not in set(
                    state.get("location_candidate_clips", [])
                ):
                    obs = (
                        "Tool unavailable: rewatch clip must come from `Suggested "
                        "rewatch clips` in the object timeline."
                    )
                    tokens = 0
                elif name == "watch_video_clip" and not any(
                    item.get("name") == "search_temporal_context"
                    and abs(item.get("args", {}).get("clip_id", -9999) - args.get("clip_id", 0)) <= 1
                    for item in state.get("tool_call_history", [])
                ):
                    obs = (
                        "Tool unavailable: verify this candidate with "
                        "`search_temporal_context` before video rewatch."
                    )
                    tokens = 0
                elif name == "watch_video_clip" and not _focus_preserves_location_object(
                    state.get("location_object_name", ""), args.get("focus", "")
                ):
                    obs = (
                        "Tool unavailable: rewatch focus must preserve the exact "
                        "target object and its identity modifiers."
                    )
                    tokens = 0
                elif name == "watch_video_clip":
                    clip_id = args.get("clip_id")
                    if clip_id in state.get("clip_history", []):
                        obs = (
                            f"Warning: You already watched clip {clip_id}. Avoid redundant "
                            "re-watching unless the focus is fundamentally different; use the "
                            "evidence you already have or try a different clip/tool."
                        )
                        tokens = 0
                    else:
                        new_clip_history.append(clip_id)
                        res = tools_by_name[name].invoke(args)
                        obs, tokens = _unpack_tool_result(res)
                elif name == "search_object_events":
                    res = tools_by_name[name].invoke(args)
                    obs, tokens = _unpack_tool_result(res)
                    if "Canonical object:" in str(obs):
                        location_object_update = args.get("object_name", "")
                        location_intent_update = args.get("intent", "")
                        new_location_candidate_clips.extend(
                            _extract_rewatch_clips(obs)
                        )
                else:
                    res = tools_by_name[name].invoke(args)
                    obs, tokens = _unpack_tool_result(res)
            except Exception as e:
                obs = f"Tool execution failed: {e}. Check your arguments and try again."
                tokens = 0

            tool_usage = merge_usage(tool_usage, tokens)
            token_details = merge_stage_usage(
                token_details, {f"tool:{name}": tokens}
            )
            tool_results.append(
                ToolMessage(content=str(obs), tool_call_id=tool_call["id"])
            )

        result = {
            "messages": tool_results,
            "clip_history": new_clip_history,
            "tool_call_history": new_tool_history,
            "location_candidate_clips": new_location_candidate_clips,
            "total_tokens": usage_total(tool_usage),
            "token_details": token_details,
        }
        if location_object_update is not None:
            result["location_object_name"] = location_object_update
            result["location_intent"] = location_intent_update
        return result

    # ------------------------------------------------------------------
    # verifier
    # ------------------------------------------------------------------
    def verifier(state: AgentState):
        last_message = state["messages"][-1]
        content = last_message.content if hasattr(last_message, "content") else ""
        hints = []
        memory_update = None
        memory_usage = empty_usage()

        # 1. Empty / failed results.
        if ("No relevant information found" in str(content)
                or "No temporal information found" in str(content)
                or "No raw action evidence found" in str(content)
                or "No episodic memory found" in str(content)
                or "No matching actions found" in str(content)
                or str(content).strip() == ""
                or "Tool execution failed" in str(content)):
            hints.append(SystemMessage(
                content="System Hint: The last tool call returned no useful information. "
                        "Try a broader entity, different keywords, a different k-allocation, "
                        "or a different tool."
            ))

        # 2. Redundant clip watching (the executor already warned; reinforce).
        if "Warning: You already watched clip" in str(content):
            hints.append(SystemMessage(
                content="System Hint: Redundant tool call detected. Use the evidence you "
                        "already have, or try a different clip or tool."
            ))

        if "call `search_object_events` first" in str(content):
            hints.append(SystemMessage(
                content="System Hint: Video rewatch is a final visual resolver, not a "
                        "broad retrieval tool. Build the exact object's event timeline "
                        "first, then rewatch one suggested clip with a narrowly specified "
                        "spatial focus."
            ))

        # 3. Repeated identical tool calls (Ego-style).
        history = state.get("tool_call_history", [])
        if len(history) >= 2:
            last_tc = history[-1]
            prev_tc = history[-2]
            if (last_tc.get("name") == prev_tc.get("name")
                    and last_tc.get("args") == prev_tc.get("args")):
                hints.append(SystemMessage(
                    content="System Hint: You just repeated the exact same tool call. This is "
                            "meaningless. You MUST change your search strategy, use different "
                            "query keywords or k-allocation, or use a different tool to gather "
                            "new information."
                ))
            if _has_consecutive_general_searches(history):
                route = classify_question_route(state.get("question", ""))
                next_tool = (
                    "`search_object_events`, `search_temporal_context`, or "
                    "`watch_video_clip`"
                    if route == "location"
                    else "`search_temporal_context`"
                )
                hints.append(SystemMessage(
                    content="System Hint: You have already used two consecutive "
                            "`general_search` calls. Stop rephrasing the search. "
                            "Select the strongest candidate clip and use "
                            f"{next_tool}."
                ))

        # 4. Question-type nudges.
        q_lower = state.get("question", "").lower()
        used_tools = {h.get("name") for h in history}

        if any(kw in q_lower for kw in _TEMPORAL_KEYWORDS):
            if "search_temporal_context" not in used_tools and "general_search" in used_tools:
                hints.append(SystemMessage(
                    content="System Hint: This looks like a temporal-order question. Once you "
                            "have a candidate clip_id, use `search_temporal_context` to verify "
                            "what happened immediately before or after it."
                ))

        route = classify_question_route(state.get("question", ""))
        last_tool = history[-1].get("name") if history else None
        evidence_is_usable = not any(
            marker in str(content)
            for marker in (
                "No relevant information found",
                "No temporal information found",
                "No raw action evidence found",
                "No episodic memory found",
                "Tool execution failed",
            )
        )
        if (
            route == "action_frequency"
            and last_tool in {
                "general_search", "search_temporal_context", "search_action_evidence"
            }
            and evidence_is_usable
        ):
            try:
                memory_update, memory_usage = update_action_frequency_memory(
                    question=state.get("question", ""),
                    current_memory=state.get("action_frequency_memory", {}),
                    new_evidence=str(content),
                )
                print(
                    "[Action Memory]\n"
                    + format_action_frequency_memory(memory_update)
                )
            except Exception as exc:
                hints.append(SystemMessage(
                    content=(
                        "System Hint: The action-frequency intermediate memory "
                        f"could not be updated: {exc}. Continue using graph evidence."
                    )
                ))

        result = {
            "total_tokens": usage_total(memory_usage),
        }
        if memory_update is not None:
            result["action_frequency_memory"] = memory_update
            result["token_details"] = {"action_frequency_memory": memory_usage}
        if hints:
            print(f"[Verifier] Hint: {hints[0].content}")
            result["messages"] = hints
        return result

    # ------------------------------------------------------------------
    # final_answer
    # ------------------------------------------------------------------
    def final_answer(state: AgentState):
        route = classify_question_route(state.get("question", ""))
        route_final = {
            "location": prompt_final_answer_location,
            "action_frequency": prompt_final_answer_action_frequency,
            "general": "",
        }[route]
        final_prompt = prompt_final_answer
        if route_final:
            final_prompt += "\n\n" + route_final
        sys_prompt = SystemMessage(content=final_prompt)
        memory_messages = []
        if route == "action_frequency":
            memory_messages.append(HumanMessage(
                content=(
                    "Use this action-frequency intermediate memory as a working "
                    "summary and reconcile it with explicit evidence:\n"
                    + format_action_frequency_memory(
                        state.get("action_frequency_memory", {})
                    )
                )
            ))
        messages_to_send = [sys_prompt] + state.get("messages", []) + memory_messages
        try:
            response = llm.invoke(messages_to_send)
        except Exception as e:
            _report_llm_failure(_ACTIVE_MODEL, e)
            raise
        print(f"[Final Answer] Output: {response.content}")

        usage = usage_from_response(response, _ACTIVE_MODEL, _ACTIVE_PROVIDER)

        return {
            "messages": [response],
            "total_tokens": usage_total(usage),
            "token_details": {"final_answer": usage},
        }

    # ------------------------------------------------------------------
    # routing
    # ------------------------------------------------------------------
    def route_after_executor(state: AgentState) -> Literal["verifier", "final_answer"]:
        tool_calls = _extract_last_tool_calls(state.get("messages", []))
        if tool_calls and any(tc["name"] == "complete_task" for tc in tool_calls):
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


def _unpack_tool_result(res):
    """Normalize a tool's return value into (observation, tokens)."""
    if isinstance(res, tuple) and len(res) == 2:
        obs, tokens = res
        return obs, (tokens or 0)
    return str(res), 0
