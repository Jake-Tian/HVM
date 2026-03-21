from __future__ import annotations


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
