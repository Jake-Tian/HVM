"""Token usage monitor for reasoning pipeline (LLM and MLLM calls)."""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class TokenUsage:
    """Token usage for a single API call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, prompt: int = 0, completion: int = 0, total: int = 0) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


class TokenMonitor:
    """Accumulates token usage across reasoning pipeline."""

    def __init__(self) -> None:
        self.text_llm = TokenUsage()  # GPT-4o-mini (parse query, semantic eval, evaluator)
        self.vision_llm = TokenUsage()  # Gemini flash (video watching)
        self._video_watch_calls = 0  # Number of video clip API calls

    def add_text_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: Optional[int] = None,
    ) -> None:
        total = total_tokens if total_tokens is not None else prompt_tokens + completion_tokens
        self.text_llm.add(prompt_tokens, completion_tokens, total)

    def add_vision_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: Optional[int] = None,
    ) -> None:
        total = total_tokens if total_tokens is not None else prompt_tokens + completion_tokens
        self.vision_llm.add(prompt_tokens, completion_tokens, total)
        self._video_watch_calls += 1

    @property
    def video_watch_calls(self) -> int:
        return self._video_watch_calls

    def total_tokens(self) -> int:
        return self.text_llm.total_tokens + self.vision_llm.total_tokens

    def to_dict(self) -> dict:
        return {
            "text_llm": self.text_llm.to_dict(),
            "vision_llm": self.vision_llm.to_dict(),
            "video_watch_calls": self._video_watch_calls,
            "total_tokens": self.total_tokens(),
        }

    def summary(self) -> str:
        lines = [
            "Token usage:",
            f"  Text LLM:   {self.text_llm.prompt_tokens} prompt + {self.text_llm.completion_tokens} completion = {self.text_llm.total_tokens} total",
            f"  Vision LLM: {self.vision_llm.prompt_tokens} prompt + {self.vision_llm.completion_tokens} completion = {self.vision_llm.total_tokens} total",
            f"  Video watch API calls: {self._video_watch_calls}",
            f"  Total tokens: {self.total_tokens()}",
        ]
        return "\n".join(lines)
