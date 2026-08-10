"""Text LLM via OpenAI GPT.

API key and base URL are read from OPENAI_API_KEY / OPENAI_BASE_URL env vars
by default.
"""

import os
import sys
from openai import OpenAI
from utils.token_usage import usage_from_response

MODEL = "gpt-5.6-luna"


def _client(api_key=None, base_url=None):
    return OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
    )


def _report_llm_failure(model, exc):
    """Print a concise, terminal-visible error for an LLM API failure.

    stderr is used (not stdout) so the message shows up in the terminal even
    when the caller has redirected stdout to a quiet log file. The exception is
    re-raised so callers can decide whether to retry or abort.
    """
    try:
        sys.stderr.write(f"[LLM ERROR] model={model}: {exc}\n")
        sys.stderr.flush()
    except Exception:
        pass


def generate_text_response(prompt, text_format=None, model=None,
                            api_key=None, base_url=None, temperature=None):
    """Call a GPT text LLM. Returns (content, token_usage).

    If text_format is given (a pydantic schema), returns (parsed, total_tokens).
    """
    model = model or MODEL

    client = _client(api_key, base_url)
    create_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None:
        create_kwargs["temperature"] = temperature

    try:
        if text_format is None:
            response = client.chat.completions.create(**create_kwargs)
            return (
                response.choices[0].message.content,
                usage_from_response(response, model, "openai"),
            )
        response = client.chat.completions.parse(
            **create_kwargs, response_format=text_format,
        )
        return (
            response.choices[0].message.parsed,
            usage_from_response(response, model, "openai"),
        )
    except Exception as e:
        _report_llm_failure(model, e)
        raise
