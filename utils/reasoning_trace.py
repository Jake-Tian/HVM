"""Helpers for serializing LangGraph tool calls and their observations."""

from langchain_core.messages import AIMessage, ToolMessage


def build_tool_rounds(messages):
    """Pair each tool call with the ToolMessage produced for that call."""
    outputs_by_call_id = {
        message.tool_call_id: message.content
        for message in messages
        if isinstance(message, ToolMessage)
    }

    rounds = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for tool_call in getattr(message, "tool_calls", None) or []:
            output = outputs_by_call_id.get(tool_call.get("id"))
            rounds.append({
                "tool": tool_call["name"],
                "args": tool_call["args"],
                "output": output,
                "output_chars": len(str(output)) if output is not None else 0,
            })
    return rounds
