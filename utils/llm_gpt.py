"""Text LLM via OpenAI GPT."""

import os

from openai import OpenAI

MODEL = "gpt-5.6-luna"


def _client():
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )


def generate_text_response(prompt, text_format=None):
    client = _client()
    kwargs = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }
    if text_format is None:
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content, response.usage.total_tokens

    response = client.chat.completions.parse(
        **kwargs,
        response_format=text_format,
    )
    return response.choices[0].message.parsed, response.usage.total_tokens
