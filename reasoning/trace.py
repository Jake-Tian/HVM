from langchain_core.messages import AIMessage, ToolMessage


def build_tool_rounds(messages):
    tool_outputs = {
        message.tool_call_id: message.content
        for message in messages
        if isinstance(message, ToolMessage)
    }
    return [
        {
            "tool": tool_call["name"],
            "args": tool_call["args"],
            "output": tool_outputs.get(tool_call.get("id"), ""),
        }
        for message in messages
        if isinstance(message, AIMessage) and message.tool_calls
        for tool_call in message.tool_calls
    ]
