"""Simple total-token accounting shared by model adapters and pipeline stages."""

from __future__ import annotations


class TokenUsage(dict):
    """JSON-serializable total token count compatible with legacy int casts."""

    def __int__(self):
        return usage_total(self)

    def __bool__(self):
        return bool(int(self))

    def __add__(self, other):
        return int(self) + usage_total(other)

    def __radd__(self, other):
        return usage_total(other) + int(self)

    def __lt__(self, other):
        return int(self) < usage_total(other)

    def __le__(self, other):
        return int(self) <= usage_total(other)

    def __gt__(self, other):
        return int(self) > usage_total(other)

    def __ge__(self, other):
        return int(self) >= usage_total(other)


def empty_usage() -> TokenUsage:
    return TokenUsage({"total": 0})


def _get(value, *names, default=0):
    for name in names:
        if isinstance(value, dict) and name in value:
            return value.get(name, default)
        if hasattr(value, name):
            return getattr(value, name)
    return default


def usage_total(value) -> int:
    """Return a total from new totals, legacy usage dicts, or plain integers."""
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)

    total = int(_get(value, "total", "total_tokens", default=0) or 0)
    if total:
        return total

    input_tokens = int(_get(
        value, "input", "input_tokens", "prompt_tokens", default=0
    ) or 0)
    output_tokens = int(_get(
        value, "output", "output_tokens", "completion_tokens", default=0
    ) or 0)
    return input_tokens + output_tokens


def normalize_usage(value) -> TokenUsage:
    return TokenUsage({"total": usage_total(value)})


def usage_from_response(response, model: str, provider: str) -> TokenUsage:
    """Extract only the total token count from model response metadata."""
    del model, provider
    metadata = getattr(response, "usage_metadata", None)
    raw_usage = metadata or getattr(response, "usage", None) or {}
    return normalize_usage(raw_usage)


def merge_usage(*values) -> TokenUsage:
    return TokenUsage({"total": sum(usage_total(value) for value in values)})


def merge_stage_usage(left: dict | None, right: dict | None) -> dict:
    merged = {
        stage: normalize_usage(usage)
        for stage, usage in (left or {}).items()
    }
    for stage, usage in (right or {}).items():
        merged[stage] = merge_usage(merged.get(stage), usage)
    return merged


def add_stage_usage(stages: dict, stage: str, usage):
    stages[stage] = merge_usage(stages.get(stage), usage)


def build_token_summary(stages: dict) -> dict:
    """Return flat per-stage totals plus one overall total."""
    summary = {
        stage: usage_total(usage)
        for stage, usage in stages.items()
    }
    summary["total"] = sum(summary.values())
    return summary
