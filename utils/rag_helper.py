from __future__ import annotations

from pathlib import Path

from utils.search import (
    evidence_linker,
    general_search,
    search_after,
    search_before,
    search_behavior,
    search_conversation,
    search_first,
    search_last,
)


def normalize_option_label(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    if text in {"A", "B", "C", "D"}:
        return text
    # Handle patterns like "A.", "A)", "Option A"
    if text[0] in {"A", "B", "C", "D"}:
        return text[0]
    return None


def evaluate_multiple_choice_answer(
    predicted_option: str | None,
    ground_truth_option: str | None,
    options: dict[str, str],
    predicted_text: str | None = None,
) -> bool:
    """
    Evaluate multiple-choice correctness without calling an LLM.
    Priority:
    1) compare normalized option labels directly
    2) fallback: compare predicted text to option text for the ground-truth label
    """
    pred = normalize_option_label(predicted_option)
    gt = normalize_option_label(ground_truth_option)

    if pred is not None and gt is not None:
        return pred == gt

    if gt is None or predicted_text is None:
        return False

    gt_text = (options.get(gt) or "").strip().lower()
    pred_text = str(predicted_text).strip().lower()
    if not gt_text or not pred_text:
        return False
    return pred_text == gt_text


def format_options(options: dict[str, str]) -> str:
    return (
        f"A. {options.get('A', '')}\n"
        f"B. {options.get('B', '')}\n"
        f"C. {options.get('C', '')}\n"
        f"D. {options.get('D', '')}"
    )


def timestamp_to_hhmm(timestamp: str) -> str:
    ts = (timestamp or "").strip()
    if len(ts) >= 4 and ts[:4].isdigit():
        return f"[{ts[:2]}:{ts[2:4]}]"
    return "[??:??]"


def organize_results_temporal(
    behavior_hits: list[list[str]],
    conversation_hits: list[list[str]],
) -> str:
    """
    Sort behavior and conversation separately by timestamp.
    Return two sections:
      behavior: ...
      conversation: ...
    """
    behavior_lines: list[tuple[str, str]] = []
    for row in behavior_hits:
        if len(row) < 2:
            continue
        ts, content = row[0], row[1]
        behavior_lines.append((ts, f"{timestamp_to_hhmm(ts)} {content}"))

    conversation_lines: list[tuple[str, str]] = []
    for row in conversation_hits:
        if len(row) < 3:
            continue
        ts, speaker, content = row[0], row[1], row[2]
        conversation_lines.append((ts, f"{timestamp_to_hhmm(ts)} {speaker}: {content}"))

    behavior_lines.sort(key=lambda x: x[0])
    conversation_lines.sort(key=lambda x: x[0])

    behavior_block = (
        "\n".join(item[1] for item in behavior_lines) if behavior_lines else "(none)"
    )
    conversation_block = (
        "\n".join(item[1] for item in conversation_lines) if conversation_lines else "(none)"
    )

    return (
        "behavior:\n"
        f"{behavior_block}\n\n"
        "conversation:\n"
        f"{conversation_block}"
    )


def organize_results_for_llm(
    behavior_hits: list[list[str]],
    conversation_hits: list[list[str]],
    retrieval_method: str = "default",
    target: str | None = None,
) -> str:
    """
    Format retrieved results for the reasoning prompt.
    - Default: chronological (same as organize_results_temporal).
    - search_before/search_after: include target line.
    - search_before: reverse temporal order in the active source block,
      with target line pinned at the top.
    """
    if retrieval_method not in {"search_before", "search_after"}:
        return organize_results_temporal(behavior_hits, conversation_hits)

    def _build_behavior_lines(rows: list[list[str]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for row in rows:
            if len(row) < 2:
                continue
            ts, content = row[0], row[1]
            out.append((ts, f"{timestamp_to_hhmm(ts)} {content}"))
        return out

    def _build_conversation_lines(rows: list[list[str]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for row in rows:
            if len(row) < 3:
                continue
            ts, speaker, content = row[0], row[1], row[2]
            out.append((ts, f"{timestamp_to_hhmm(ts)} {speaker}: {content}"))
        return out

    behavior_lines = _build_behavior_lines(behavior_hits)
    conversation_lines = _build_conversation_lines(conversation_hits)

    behavior_lines.sort(key=lambda x: x[0])
    conversation_lines.sort(key=lambda x: x[0])

    # For search_before, show nearest previous evidence first.
    if retrieval_method == "search_before":
        if behavior_lines and not conversation_lines:
            behavior_lines = list(reversed(behavior_lines))
        if conversation_lines and not behavior_lines:
            conversation_lines = list(reversed(conversation_lines))

    target_line = f"[TARGET] {target}" if target else None

    if behavior_lines:
        behavior_body = "\n".join(item[1] for item in behavior_lines)
        if target_line:
            behavior_body = f"{target_line}\n{behavior_body}"
    else:
        behavior_body = "(none)"

    if conversation_lines:
        conversation_body = "\n".join(item[1] for item in conversation_lines)
        if target_line:
            conversation_body = f"{target_line}\n{conversation_body}"
    else:
        conversation_body = "(none)"

    return (
        "behavior:\n"
        f"{behavior_body}\n\n"
        "conversation:\n"
        f"{conversation_body}"
    )


def safe_allocation(k_behavior: int | None, k_conversation: int | None) -> tuple[int, int]:
    kb = int(k_behavior) if isinstance(k_behavior, int) else 25
    kc = int(k_conversation) if isinstance(k_conversation, int) else 25
    if kb < 0:
        kb = 0
    if kc < 0:
        kc = 0
    total = kb + kc
    if total <= 50:
        return kb, kc
    if total == 0:
        return 25, 25
    # rescale down to max 50 while keeping integer
    kb = int(round(kb * 50 / total))
    kc = 50 - kb
    return kb, kc


def harmonize_allocation_with_total(
    k_behavior: int | None,
    k_conversation: int | None,
    total_search_k: int | None,
) -> tuple[int, int]:
    """
    Harmonize behavior/conversation allocation with an optional target total.
    Enforces:
      - non-negative integers
      - total <= 50
      - if total_search_k is provided, k_behavior + k_conversation == total_search_k
    """
    kb = int(k_behavior) if isinstance(k_behavior, int) else 25
    kc = int(k_conversation) if isinstance(k_conversation, int) else 25
    kb = max(0, kb)
    kc = max(0, kc)

    if isinstance(total_search_k, int):
        target = max(1, min(50, total_search_k))
    else:
        target = min(50, kb + kc)
        if target <= 0:
            target = 50

    current = kb + kc
    if current <= 0:
        kb = target // 2
        kc = target - kb
        return kb, kc

    # Scale to target total.
    kb = int(round(kb * target / current))
    kb = max(0, min(target, kb))
    kc = target - kb
    return kb, kc


def split_tool_search_output(tool_output: dict) -> tuple[list[list[str]], list[list[str]]]:
    if not isinstance(tool_output, dict):
        return [], []
    # general_search / evidence_linker / search_first / search_last output
    if "behavior" in tool_output or "conversation" in tool_output:
        behavior_hits = tool_output.get("behavior") or []
        conversation_hits = tool_output.get("conversation") or []
        return behavior_hits, conversation_hits
    # search_before / search_after output
    source = tool_output.get("source")
    results = tool_output.get("results") or []
    if source == "behavior":
        return results, []
    if source == "conversation":
        return [], results
    return [], []


def run_search_for_round(
    round_id: int,
    search_question: str,
    k_behavior: int,
    k_conversation: int,
    speaker_strict: list[str] | None,
    behavior_dir: str | Path,
    conversation_dir: str | Path,
    planned_tool_name: str | None,
    planned_target: str | None,
) -> tuple[list[list[str]], list[list[str]], str]:
    """
    Execute retrieval for one round.
    - Round 1 and 5: normal search_behavior/search_conversation.
    - Rounds 2-4: prefer tool search when valid; fallback to normal search.
    """
    can_use_tool = 2 <= round_id <= 4
    allocation = {
        "k_behavior": k_behavior,
        "k_conversation": k_conversation,
        "total_search_k": k_behavior + k_conversation,
    }

    if can_use_tool and planned_tool_name:
        try:
            if planned_tool_name == "general_search":
                tool_output = general_search(
                    search_content=search_question,
                    allocation=allocation,
                    behavior_dir=behavior_dir,
                    conversation_dir=conversation_dir,
                    speaker_strict=speaker_strict,
                )
            elif planned_tool_name == "evidence_linker":
                tool_output = evidence_linker(
                    search_content=search_question,
                    allocation=allocation,
                    behavior_dir=behavior_dir,
                    conversation_dir=conversation_dir,
                    speaker_strict=speaker_strict,
                    target=planned_target,
                )
            elif planned_tool_name == "search_first":
                tool_output = search_first(
                    search_content=search_question,
                    allocation=allocation,
                    behavior_dir=behavior_dir,
                    conversation_dir=conversation_dir,
                )
            elif planned_tool_name == "search_last":
                tool_output = search_last(
                    search_content=search_question,
                    allocation=allocation,
                    behavior_dir=behavior_dir,
                    conversation_dir=conversation_dir,
                )
            elif planned_tool_name == "search_before" and planned_target:
                tool_output = search_before(
                    search_content=search_question,
                    target=planned_target,
                    allocation=allocation,
                    behavior_dir=behavior_dir,
                    conversation_dir=conversation_dir,
                )
            elif planned_tool_name == "search_after" and planned_target:
                tool_output = search_after(
                    search_content=search_question,
                    target=planned_target,
                    allocation=allocation,
                    behavior_dir=behavior_dir,
                    conversation_dir=conversation_dir,
                )
            else:
                raise ValueError(
                    f"Invalid tool configuration: tool={planned_tool_name}, target={planned_target}"
                )
            behavior_hits, conversation_hits = split_tool_search_output(tool_output)
            return behavior_hits, conversation_hits, planned_tool_name
        except Exception as tool_error:
            print(
                f"[Round {round_id}] tool search failed ({planned_tool_name}): {tool_error}. "
                "Falling back to normal search."
            )

    behavior_hits = search_behavior(search_question, k_behavior, behavior_dir)
    conversation_hits = search_conversation(
        search_question,
        speaker_strict,
        conversation_dir,
        k=k_conversation,
    )
    return behavior_hits, conversation_hits, "default"
