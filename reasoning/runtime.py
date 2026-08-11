from langchain_core.messages import AIMessage, SystemMessage, ToolMessage


def response_tokens(response):
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        return response.usage_metadata.get("total_tokens", 0)
    return 0


def execute_tool_calls(last_message, tools_by_name, clip_history):
    tool_results = []
    new_history = []
    total_tokens = 0

    for tool_call in last_message.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]
        obs = ""
        tokens = 0
        try:
            if name in tools_by_name:
                tool_fn = tools_by_name[name]
                if name == "watch_video_clip":
                    clip_id = args.get("clip_id")
                    if clip_id in clip_history:
                        obs = f"Warning: You already watched clip {clip_id}. Avoid redundant re-watching unless the focus is fundamentally different."
                        tokens = 0
                    else:
                        new_history.append(clip_id)
                        res = tool_fn.invoke(args)
                        obs, tokens = res
                else:
                    res = tool_fn.invoke(args)
                    if isinstance(res, tuple) and len(res) == 2:
                        obs, tokens = res
                    else:
                        obs = str(res)
                        tokens = 0
            else:
                obs = f"Unknown tool: {name}"
                tokens = 0
        except Exception as e:
            obs = f"Tool execution failed: {e}. Check your arguments and try again."
            tokens = 0

        total_tokens += tokens
        tool_results.append(ToolMessage(content=str(obs), tool_call_id=tool_call["id"]))

    return {
        "messages": tool_results,
        "clip_history": new_history,
        "total_tokens": total_tokens,
    }


def verifier_hints(last_message):
    content = last_message.content if hasattr(last_message, "content") else ""
    hints = []

    if ("No relevant information found" in str(content)
            or "No temporal information found" in str(content)
            or str(content).strip() == ""
            or "Tool execution failed" in str(content)):
        hints.append(SystemMessage(
            content="System Hint: The last tool call returned no useful information. "
                    "Try a broader entity, different keywords, a different k-allocation, "
                    "or a different tool."
        ))

    if "Warning: You already watched clip" in str(content):
        hints.append(SystemMessage(
            content="System Hint: Redundant tool call detected. Use the evidence you "
                    "already have, or try a different clip or tool."
        ))

    return hints


def last_tool_call_names(messages):
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            if message.tool_calls:
                return [tool_call["name"] for tool_call in message.tool_calls]
            break
    return []
