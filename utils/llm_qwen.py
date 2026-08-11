"""Text LLM via Qwen's OpenAI-compatible endpoint."""

import os

from openai import OpenAI

MODEL = "qwen3.5-flash"
_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _client():
    return OpenAI(
        api_key=os.environ.get("DASHSCOPE_API_KEY"),
        base_url=os.environ.get("DASHSCOPE_BASE_URL", _BASE_URL),
    )


def generate_text_response(prompt, text_format=None):
    client = _client()
    kwargs = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "extra_body": {"enable_thinking": False},
    }
    if text_format is None:
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content, response.usage.total_tokens

    response = client.chat.completions.parse(
        **kwargs,
        response_format=text_format,
    )
    return response.choices[0].message.parsed, response.usage.total_tokens
