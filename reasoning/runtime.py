"""Small runtime helpers used by the five-tool LangGraph workflow."""

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage

from utils.token_usage import usage_total


TEMPORAL_KEYWORDS = (
    "before", "after", "first", "last", "earliest", "latest",
    "next", "previous", "then", "sequence",
)


def build_graph_stats(graph):
    high_level_count = sum(edge.clip_id == 0 for edge in graph.edges.values())
    max_clip = max(
        (
            edge.clip_id
            for edge in graph.edges.values()
            if edge.clip_id is not None
        ),
        default=0,
    )
    scenes = sorted({edge.scene for edge in graph.edges.values() if edge.scene})
    characters = list(graph.characters)

    return (
        "--- Graph Stats ---\n"
        f"Characters: {', '.join(characters) if characters else 'None'}\n"
        f"Total Object Nodes: {len(graph.objects)}\n"
        f"Total Edges: {len(graph.edges)} "
        f"(High-level: {high_level_count}, "
        f"Low-level: {len(graph.edges) - high_level_count})\n"
        f"Total Clips: {max_clip}\n"
        f"Total Conversations: {len(graph.conversations)}\n"
        f"Scenes: {', '.join(scenes) if scenes else 'None'}\n"
        "Graph Construction Info:\n"
        "- The video is divided into 30-second clips "
        "(Clip 1 = 0-30s, Clip 2 = 30-60s, etc.).\n"
        "- Character nodes are enclosed in angle brackets "
        "(e.g., <robot>, <Alice>).\n"
        "- Object nodes are plain text (e.g., coffee, table).\n"
        "- High-level edges (clip_id=0) represent overall attributes "
        "and relationships.\n"
        "- Low-level edges represent specific actions/states occurring "
        "in specific clips.\n"
        "-------------------"
    )


def last_tool_calls(messages):
    for message in reversed(messages):
        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            return message.tool_calls
    return None


def response_tokens(response):
    return usage_total(getattr(response, "usage_metadata", None))


def unpack_tool_result(result):
    if isinstance(result, tuple) and len(result) == 2:
        observation, tokens = result
        return observation, usage_total(tokens)
    return str(result), 0


def execute_tool_calls(tool_calls, tools_by_name, watched_clips):
    messages = []
    new_clips = []
    history = []
    total_tokens = 0

    for tool_call in tool_calls or []:
        name = tool_call["name"]
        args = tool_call["args"]
        history.append({"name": name, "args": args})

        try:
            if name not in tools_by_name:
                observation, tokens = f"Unknown tool: {name}", 0
            elif name == "watch_video_clip" and args.get("clip_id") in watched_clips:
                clip_id = args.get("clip_id")
                observation = (
                    f"Warning: You already watched clip {clip_id}. Avoid redundant "
                    "re-watching unless the focus is fundamentally different; use "
                    "the evidence you already have or try a different clip/tool."
                )
                tokens = 0
            else:
                if name == "watch_video_clip":
                    new_clips.append(args.get("clip_id"))
                result = tools_by_name[name].invoke(args)
                observation, tokens = unpack_tool_result(result)
        except Exception as error:
            observation = (
                f"Tool execution failed: {error}. Check your arguments and try again."
            )
            tokens = 0

        total_tokens += tokens
        messages.append(ToolMessage(
            content=str(observation),
            tool_call_id=tool_call["id"],
        ))

    return {
        "messages": messages,
        "clip_history": new_clips,
        "tool_call_history": history,
        "total_tokens": total_tokens,
    }


def verifier_hints(state):
    content = str(getattr(state["messages"][-1], "content", ""))
    hints = []

    if (
        "No relevant information found" in content
        or "No temporal information found" in content
        or "No matching actions found" in content
        or not content.strip()
        or "Tool execution failed" in content
    ):
        hints.append(SystemMessage(
            content="System Hint: The last tool call returned no useful information. "
                    "Try a broader entity, different keywords, a different k-allocation, "
                    "or a different tool."
        ))

    if "Warning: You already watched clip" in content:
        hints.append(SystemMessage(
            content="System Hint: Redundant tool call detected. Use the evidence you "
                    "already have, or try a different clip or tool."
        ))

    history = state.get("tool_call_history", [])
    if len(history) >= 2:
        current, previous = history[-1], history[-2]
        if current.get("name") == previous.get("name") and current.get("args") == previous.get("args"):
            hints.append(SystemMessage(
                content="System Hint: You just repeated the exact same tool call. This is "
                        "meaningless. You MUST change your search strategy, use different "
                        "query keywords or k-allocation, or use a different tool to gather "
                        "new information."
            ))

    question = state.get("question", "").lower()
    used_tools = {call.get("name") for call in history}
    if any(keyword in question for keyword in TEMPORAL_KEYWORDS):
        if "search_temporal_context" not in used_tools and "general_search" in used_tools:
            hints.append(SystemMessage(
                content="System Hint: This looks like a temporal-order question. Once you "
                        "have a candidate clip_id, use `search_temporal_context` to verify "
                        "what happened immediately before or after it."
            ))

    return hints
