"""Structured intermediate memory for action-frequency reasoning."""

import json
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ActionEvent(BaseModel):
    event_id: str
    clip_ids: list[int] = Field(default_factory=list)
    description: str
    status: Literal["candidate", "confirmed", "rejected", "merged"]
    occurrence_count: int = Field(default=1, ge=0)
    merged_into: Optional[str] = None


class ActionFrequencyMemory(BaseModel):
    counting_unit: str
    events: list[ActionEvent] = Field(default_factory=list)
    total_confirmed: int = 0


_UPDATE_PROMPT = """Maintain a compact event ledger for an action-frequency question.

Question:
{question}

Current ledger:
{current_memory}

New evidence:
{new_evidence}

Rules:
- First define the counting unit as one completed occurrence of the target action.
- Distinguish an action episode from the requested counting unit. One continuous
  episode may contain multiple completed occurrences, such as printing six posters.
- Set occurrence_count to the explicit or directly supported number of completed
  counting units inside an event. Do not assume one event always equals one occurrence.
- Do not count preparatory sub-actions, object handling, or repeated captions as
  separate occurrences.
- Merge the same continuous event described in the same or adjacent clips.
- Create a new event only when the evidence shows a reset or a new completed episode,
  such as a new actor, a clear stop and restart, or a later independent occurrence.
- Preserve earlier supported events unless new evidence directly contradicts them.
- Use candidate when an occurrence is plausible but incomplete, confirmed when a
  complete occurrence is directly supported, rejected for irrelevant evidence, and
  merged when an entry duplicates another event.
- Keep descriptions short and evidence-based. Do not infer events that are absent.

Return the complete updated ledger."""


def _as_dict(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    raise TypeError(f"Unsupported action-frequency memory: {type(value)!r}")


def normalize_action_frequency_memory(memory):
    data = _as_dict(memory)
    events = []
    for event in data.get("events") or []:
        event_data = _as_dict(event)
        event_data["clip_ids"] = sorted(set(event_data.get("clip_ids") or []))
        try:
            event_data["occurrence_count"] = max(
                0, int(event_data.get("occurrence_count", 1))
            )
        except (TypeError, ValueError):
            event_data["occurrence_count"] = 1
        events.append(event_data)
    data["events"] = events
    data["total_confirmed"] = sum(
        event.get("occurrence_count", 1)
        for event in events
        if event.get("status") == "confirmed"
    )
    return data


def format_action_frequency_memory(memory):
    if not memory:
        return "No action events recorded yet."
    data = normalize_action_frequency_memory(memory)
    lines = [f"Counting unit: {data.get('counting_unit') or 'not established'}"]
    for event in data["events"]:
        clip_ids = event.get("clip_ids") or []
        if len(clip_ids) == 1:
            clips = f"clip {clip_ids[0]}"
        elif clip_ids:
            clips = f"clips {clip_ids[0]}-{clip_ids[-1]}"
        else:
            clips = "clip unknown"
        status = event.get("status", "candidate")
        if status == "merged" and event.get("merged_into"):
            status = f"merged into {event['merged_into']}"
        count = event.get("occurrence_count", 1)
        if event.get("status") == "confirmed":
            status += f" x{count}"
        lines.append(
            f"{event.get('event_id', '?')} | {clips} | {status} | "
            f"{event.get('description', '')}"
        )
    lines.append(f"Total confirmed: {data['total_confirmed']}")
    return "\n".join(lines)


def update_action_frequency_memory(
    question,
    current_memory,
    new_evidence,
    generate=None,
):
    """Update and return ``(memory_dict, token_usage)`` from new graph evidence."""
    if generate is None:
        from utils.llm_gpt import generate_text_response

        generate = generate_text_response

    prompt = _UPDATE_PROMPT.format(
        question=question,
        current_memory=json.dumps(current_memory or {}, ensure_ascii=False),
        new_evidence=new_evidence,
    )
    response, usage = generate(prompt, ActionFrequencyMemory)
    return normalize_action_frequency_memory(response), usage
