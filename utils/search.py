from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from utils.llm import get_embedding


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    a = np.array(vec1, dtype=float)
    b = np.array(vec2, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return -1.0
    return float(np.dot(a, b) / denom)


def _collect_json_files(file_directory: str | Path) -> list[Path]:
    target = Path(file_directory)
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {target}")
    if target.is_file():
        return [target]
    return sorted(target.glob("*.json"))


def _parse_behavior_row(row) -> tuple[str, str] | None:
    if not isinstance(row, list) or len(row) < 2:
        return None
    timestamp, content = row[0], row[1]
    if not isinstance(timestamp, str) or not isinstance(content, str):
        return None
    return timestamp, content


def _parse_conversation_row(row) -> tuple[str, str, str] | None:
    if not isinstance(row, dict):
        return None
    timestamp = row.get("start_time")
    speaker = row.get("speaker")
    content = row.get("content")
    if not isinstance(timestamp, str) or not isinstance(speaker, str) or not isinstance(content, str):
        return None
    return timestamp, speaker, content


def search_behavior(
    question: str,
    k: int,
    file_directory: str | Path,
    context_window: int = 1,
) -> list[list[str]]:
    """
    Search behavior files and return top-k hits expanded with context.
    For each selected hit, include `context_window` rows before and after.

    Behavior file format:
      [
        [start_time_hhmmss, content, embedding],
        ...
      ]
    """
    if k <= 0:
        return []

    query_embedding = get_embedding(question)
    scored: list[tuple[float, int, int]] = []  # score, file_idx, row_idx
    file_data: list[list] = []

    for file_idx, json_path in enumerate(_collect_json_files(file_directory)):
        with json_path.open("r", encoding="utf-8") as infile:
            data = json.load(infile)
        file_data.append(data if isinstance(data, list) else [])

        for row_idx, item in enumerate(file_data[file_idx]):
            if not isinstance(item, list) or len(item) < 3:
                continue
            parsed = _parse_behavior_row(item)
            if parsed is None:
                continue
            _, _ = parsed
            embedding = item[2]
            if not isinstance(embedding, list):
                continue
            score = _cosine_similarity(query_embedding, embedding)
            scored.append((score, file_idx, row_idx))

    scored.sort(key=lambda x: x[0], reverse=True)
    anchors = scored[:k]

    selected: list[tuple[str, str]] = []
    seen = set()
    for _, file_idx, row_idx in anchors:
        rows = file_data[file_idx]
        start = max(0, row_idx - context_window)
        end = min(len(rows) - 1, row_idx + context_window)
        for idx in range(start, end + 1):
            parsed = _parse_behavior_row(rows[idx])
            if parsed is None:
                continue
            key = (file_idx, idx)
            if key in seen:
                continue
            seen.add(key)
            selected.append(parsed)

    selected.sort(key=lambda x: x[0])
    return [[timestamp, content] for timestamp, content in selected]


def search_conversation(
    question: str,
    speaker_strict: list[str] | None,
    file_directory: str | Path,
    k: int = 50,
    context_window: int = 1,
) -> list[list[str]]:
    """
    Search conversation files and return top-k hits expanded with context.
    For each selected hit, include `context_window` rows before and after.

    Conversation file format:
      [
        {
          "start_time": "hhmmss",
          "speaker": "...",
          "content": "...",
          "embedding": [...]
        },
        ...
      ]
    """
    if k <= 0:
        return []

    query_embedding = get_embedding(question)
    speaker_filter = {s for s in speaker_strict} if speaker_strict else None
    scored: list[tuple[float, int, int]] = []  # score, file_idx, row_idx
    file_data: list[list] = []

    for file_idx, json_path in enumerate(_collect_json_files(file_directory)):
        with json_path.open("r", encoding="utf-8") as infile:
            data = json.load(infile)
        file_data.append(data if isinstance(data, list) else [])

        for row_idx, item in enumerate(file_data[file_idx]):
            parsed = _parse_conversation_row(item)
            if parsed is None:
                continue
            _, speaker, _ = parsed
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                continue
            if speaker_filter is not None and speaker not in speaker_filter:
                continue
            score = _cosine_similarity(query_embedding, embedding)
            scored.append((score, file_idx, row_idx))

    scored.sort(key=lambda x: x[0], reverse=True)
    anchors = scored[:k]

    selected: list[tuple[str, str, str]] = []
    seen = set()
    for _, file_idx, row_idx in anchors:
        rows = file_data[file_idx]
        start = max(0, row_idx - context_window)
        end = min(len(rows) - 1, row_idx + context_window)
        for idx in range(start, end + 1):
            parsed = _parse_conversation_row(rows[idx])
            if parsed is None:
                continue
            key = (file_idx, idx)
            if key in seen:
                continue
            seen.add(key)
            selected.append(parsed)

    selected.sort(key=lambda x: x[0])
    return [[timestamp, speaker, content] for timestamp, speaker, content in selected]


def _to_hhmm(timestamp: str) -> str:
    ts = (timestamp or "").strip()
    if len(ts) >= 4 and ts[:4].isdigit():
        return f"{ts[:2]}:{ts[2:4]}"
    return "??:??"


def _timestamp_key(timestamp: str) -> int:
    ts = (timestamp or "").strip()
    return int(ts) if ts.isdigit() else 10**12


def _normalize_allocation(allocation: dict | int | None) -> tuple[int, int]:
    if isinstance(allocation, int):
        kb = max(0, allocation // 2)
        kc = max(0, allocation - kb)
    elif isinstance(allocation, dict):
        kb = max(0, int(allocation.get("k_behavior", 0) or 0))
        kc = max(0, int(allocation.get("k_conversation", 0) or 0))
        total_k = allocation.get("total_search_k")
        if isinstance(total_k, int):
            target = max(1, min(50, total_k))
            cur = kb + kc
            if cur == 0:
                kb = target // 2
                kc = target - kb
            elif cur != target:
                kb = int(round(kb * target / cur))
                kb = max(0, min(target, kb))
                kc = target - kb
    else:
        kb, kc = 25, 25

    total = kb + kc
    if total == 0:
        return 25, 25
    if total <= 50:
        return kb, kc
    kb = int(round(kb * 50 / total))
    return kb, 50 - kb


def _build_behavior_rows(file_directory: str | Path) -> list[dict]:
    rows: list[dict] = []
    for file_idx, json_path in enumerate(_collect_json_files(file_directory)):
        with json_path.open("r", encoding="utf-8") as infile:
            data = json.load(infile)
        if not isinstance(data, list):
            continue
        for row_idx, item in enumerate(data):
            if not isinstance(item, list) or len(item) < 3:
                continue
            parsed = _parse_behavior_row(item)
            if parsed is None:
                continue
            ts, content = parsed
            emb = item[2]
            if not isinstance(emb, list):
                continue
            line = f"[{_to_hhmm(ts)}] [behavior] {content}"
            rows.append(
                {
                    "source": "behavior",
                    "timestamp": ts,
                    "content": content,
                    "speaker": None,
                    "embedding": emb,
                    "file_idx": file_idx,
                    "row_idx": row_idx,
                    "line": line,
                }
            )
    rows.sort(key=lambda r: (_timestamp_key(r["timestamp"]), r["file_idx"], r["row_idx"]))
    return rows


def _build_conversation_rows(file_directory: str | Path) -> list[dict]:
    rows: list[dict] = []
    for file_idx, json_path in enumerate(_collect_json_files(file_directory)):
        with json_path.open("r", encoding="utf-8") as infile:
            data = json.load(infile)
        if not isinstance(data, list):
            continue
        for row_idx, item in enumerate(data):
            parsed = _parse_conversation_row(item)
            if parsed is None:
                continue
            ts, speaker, content = parsed
            emb = item.get("embedding")
            if not isinstance(emb, list):
                continue
            line = f"[{_to_hhmm(ts)}] [conversation] {speaker}: {content}"
            rows.append(
                {
                    "source": "conversation",
                    "timestamp": ts,
                    "content": content,
                    "speaker": speaker,
                    "embedding": emb,
                    "file_idx": file_idx,
                    "row_idx": row_idx,
                    "line": line,
                }
            )
    rows.sort(key=lambda r: (_timestamp_key(r["timestamp"]), r["file_idx"], r["row_idx"]))
    return rows


def _row_to_output(row: dict):
    if row["source"] == "behavior":
        return [row["timestamp"], row["content"]]
    return [row["timestamp"], row["speaker"], row["content"]]


def _search_topk_rows(rows: list[dict], search_content: str, k: int) -> list[dict]:
    if k <= 0 or not rows:
        return []
    qemb = get_embedding(search_content)
    scored = []
    for i, r in enumerate(rows):
        sim = _cosine_similarity(qemb, r["embedding"])
        scored.append((sim, i))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [rows[i] for _, i in scored[:k]]


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _find_target_index(rows: list[dict], target: str) -> int | None:
    if not rows:
        return None
    t = _normalize_text(target)
    # 1) strict line match
    for i, r in enumerate(rows):
        if _normalize_text(r["line"]) == t:
            return i
    # 2) rich variants
    for i, r in enumerate(rows):
        variants = [r["line"], r["content"]]
        if r["source"] == "conversation":
            variants.append(f'{r["speaker"]}: {r["content"]}')
            variants.append(f'[{_to_hhmm(r["timestamp"])}] {r["speaker"]}: {r["content"]}')
        else:
            variants.append(f'[{_to_hhmm(r["timestamp"])}] {r["content"]}')
        if any(_normalize_text(v) == t for v in variants):
            return i
    # 3) containment fallback
    for i, r in enumerate(rows):
        if t in _normalize_text(r["line"]) or _normalize_text(r["content"]) in t:
            return i
    return None


def general_search(
    search_content: str,
    allocation: dict | int,
    behavior_dir: str | Path,
    conversation_dir: str | Path,
    speaker_strict: list[str] | None = None,
) -> dict:
    """
    Normal similarity search.
    Input: search_content, allocation
    Return top-k most similar results from behavior/conversation.
    """
    kb, kc = _normalize_allocation(allocation)
    behavior_rows = _build_behavior_rows(behavior_dir)
    conversation_rows = _build_conversation_rows(conversation_dir)
    if speaker_strict:
        spk = set(speaker_strict)
        conversation_rows = [r for r in conversation_rows if r["speaker"] in spk]

    b_hits = _search_topk_rows(behavior_rows, search_content, kb)
    c_hits = _search_topk_rows(conversation_rows, search_content, kc)
    return {
        "behavior": [_row_to_output(r) for r in b_hits],
        "conversation": [_row_to_output(r) for r in c_hits],
    }


def _build_linker_rows_from_hits(
    behavior_hits: list[list[str]],
    conversation_hits: list[list[str]],
) -> list[dict]:
    rows: list[dict] = []
    for row in behavior_hits:
        if len(row) < 2:
            continue
        ts, content = row[0], row[1]
        rows.append(
            {
                "source": "behavior",
                "timestamp": ts,
                "content": content,
                "speaker": None,
                "line": f"[{_to_hhmm(ts)}] [behavior] {content}",
            }
        )
    for row in conversation_hits:
        if len(row) < 3:
            continue
        ts, speaker, content = row[0], row[1], row[2]
        rows.append(
            {
                "source": "conversation",
                "timestamp": ts,
                "content": content,
                "speaker": speaker,
                "line": f"[{_to_hhmm(ts)}] [conversation] {speaker}: {content}",
            }
        )
    rows.sort(key=lambda r: _timestamp_key(r["timestamp"]))
    return rows


def _query_alias_expansion(search_content: str) -> set[str]:
    text = _normalize_text(search_content)
    terms = set(re.findall(r"[a-z0-9]+", text))
    alias_groups = [
        {"marker", "pen", "pencil", "chalk", "whiteboard"},
        {"upstairs", "up", "second", "floor"},
        {"table", "desk"},
        {"before", "earlier", "previous", "prior"},
        {"after", "later", "next", "following"},
        {"where", "location", "placed", "put"},
        {"hold", "holding", "hand"},
    ]
    for group in alias_groups:
        if terms & group:
            terms |= group
    return terms


def _row_link_score(row: dict, query_terms: set[str], anchor_idx: int | None, idx: int) -> float:
    row_text = _normalize_text(f"{row.get('content', '')} {row.get('speaker', '')}")
    row_terms = set(re.findall(r"[a-z0-9]+", row_text))
    overlap = len(query_terms & row_terms)
    score = float(overlap)
    if "table" in row_terms or "floor" in row_terms or "stairs" in row_terms:
        score += 0.5
    if "marker" in row_terms or "pen" in row_terms or "pencil" in row_terms:
        score += 0.75
    if anchor_idx is not None:
        score += 1.0 / (1.0 + abs(idx - anchor_idx))
    return score


def evidence_linker(
    search_content: str,
    allocation: dict | int,
    behavior_dir: str | Path,
    conversation_dir: str | Path,
    speaker_strict: list[str] | None = None,
    target: str | None = None,
    max_chain_steps: int = 4,
) -> dict:
    """
    Build a compact cross-source evidence chain from retrieved candidates.
    This tool is useful for deduction questions where facts are spread across
    behavior and conversation lines.
    """
    candidates = general_search(
        search_content=search_content,
        allocation=allocation,
        behavior_dir=behavior_dir,
        conversation_dir=conversation_dir,
        speaker_strict=speaker_strict,
    )
    behavior_hits = candidates.get("behavior") or []
    conversation_hits = candidates.get("conversation") or []
    rows = _build_linker_rows_from_hits(behavior_hits, conversation_hits)
    if not rows:
        return {"behavior": [], "conversation": [], "chain": [], "summary": "(none)"}

    query_terms = _query_alias_expansion(search_content)
    anchor_idx: int | None = None
    if target:
        t = _normalize_text(target)
        for i, r in enumerate(rows):
            if t in _normalize_text(r["line"]) or t in _normalize_text(r["content"]):
                anchor_idx = i
                break

    scored = []
    for i, r in enumerate(rows):
        s = _row_link_score(r, query_terms, anchor_idx, i)
        scored.append((s, i))
    scored.sort(key=lambda x: x[0], reverse=True)

    picked_idx = sorted(i for _, i in scored[: max(1, max_chain_steps)])
    picked_rows = [rows[i] for i in picked_idx]

    chain = []
    b_out: list[list[str]] = []
    c_out: list[list[str]] = []
    for r in picked_rows:
        chain.append(
            {
                "timestamp": r["timestamp"],
                "source": r["source"],
                "text": r["line"],
                "why_selected": "query overlap and temporal bridging",
            }
        )
        if r["source"] == "behavior":
            b_out.append([r["timestamp"], r["content"]])
        else:
            c_out.append([r["timestamp"], r["speaker"], r["content"]])

    summary = " -> ".join(item["text"] for item in chain[:3])
    return {
        "behavior": b_out,
        "conversation": c_out,
        "chain": chain,
        "summary": summary if summary else "(none)",
    }


def search_before(
    search_content: str,
    target: str,
    allocation: dict | int,
    behavior_dir: str | Path,
    conversation_dir: str | Path,
) -> dict:
    """
    Search only BEFORE a target line.
    Target should be a line from previously searched behavior/conversation results.
    """
    kb, kc = _normalize_allocation(allocation)
    behavior_rows = _build_behavior_rows(behavior_dir)
    conversation_rows = _build_conversation_rows(conversation_dir)

    b_idx = _find_target_index(behavior_rows, target)
    c_idx = _find_target_index(conversation_rows, target)
    if b_idx is None and c_idx is None:
        raise ValueError("Target not found in behavior or conversation data.")

    # enforce single-source temporal search
    if b_idx is not None and c_idx is None:
        kc = 0
        kb = kb if kb > 0 else 50
        scope_rows = behavior_rows[:b_idx]
        hits = _search_topk_rows(scope_rows, search_content, kb)
        return {"source": "behavior", "results": [_row_to_output(r) for r in hits]}
    if c_idx is not None and b_idx is None:
        kb = 0
        kc = kc if kc > 0 else 50
        scope_rows = conversation_rows[:c_idx]
        hits = _search_topk_rows(scope_rows, search_content, kc)
        return {"source": "conversation", "results": [_row_to_output(r) for r in hits]}

    # target appears in both; use larger allocated budget as tiebreak
    if kb >= kc:
        scope_rows = behavior_rows[:b_idx]
        hits = _search_topk_rows(scope_rows, search_content, max(kb, 1))
        return {"source": "behavior", "results": [_row_to_output(r) for r in hits]}
    scope_rows = conversation_rows[:c_idx]
    hits = _search_topk_rows(scope_rows, search_content, max(kc, 1))
    return {"source": "conversation", "results": [_row_to_output(r) for r in hits]}


def search_after(
    search_content: str,
    target: str,
    allocation: dict | int,
    behavior_dir: str | Path,
    conversation_dir: str | Path,
) -> dict:
    """
    Search only AFTER a target line.
    """
    kb, kc = _normalize_allocation(allocation)
    behavior_rows = _build_behavior_rows(behavior_dir)
    conversation_rows = _build_conversation_rows(conversation_dir)

    b_idx = _find_target_index(behavior_rows, target)
    c_idx = _find_target_index(conversation_rows, target)
    if b_idx is None and c_idx is None:
        raise ValueError("Target not found in behavior or conversation data.")

    if b_idx is not None and c_idx is None:
        kc = 0
        kb = kb if kb > 0 else 50
        scope_rows = behavior_rows[b_idx + 1 :]
        hits = _search_topk_rows(scope_rows, search_content, kb)
        return {"source": "behavior", "results": [_row_to_output(r) for r in hits]}
    if c_idx is not None and b_idx is None:
        kb = 0
        kc = kc if kc > 0 else 50
        scope_rows = conversation_rows[c_idx + 1 :]
        hits = _search_topk_rows(scope_rows, search_content, kc)
        return {"source": "conversation", "results": [_row_to_output(r) for r in hits]}

    if kb >= kc:
        scope_rows = behavior_rows[b_idx + 1 :]
        hits = _search_topk_rows(scope_rows, search_content, max(kb, 1))
        return {"source": "behavior", "results": [_row_to_output(r) for r in hits]}
    scope_rows = conversation_rows[c_idx + 1 :]
    hits = _search_topk_rows(scope_rows, search_content, max(kc, 1))
    return {"source": "conversation", "results": [_row_to_output(r) for r in hits]}


def search_first(
    search_content: str,
    allocation: dict | int,
    behavior_dir: str | Path,
    conversation_dir: str | Path,
    similarity_threshold: float = 0.5,
) -> dict:
    """
    Search from beginning and return lines with similarity >= threshold.
    """
    kb, kc = _normalize_allocation(allocation)
    qemb = get_embedding(search_content)
    behavior_rows = _build_behavior_rows(behavior_dir)
    conversation_rows = _build_conversation_rows(conversation_dir)

    b_out = []
    for r in behavior_rows:
        if len(b_out) >= kb:
            break
        if _cosine_similarity(qemb, r["embedding"]) >= similarity_threshold:
            b_out.append(_row_to_output(r))

    c_out = []
    for r in conversation_rows:
        if len(c_out) >= kc:
            break
        if _cosine_similarity(qemb, r["embedding"]) >= similarity_threshold:
            c_out.append(_row_to_output(r))

    return {"behavior": b_out, "conversation": c_out}


def search_last(
    search_content: str,
    allocation: dict | int,
    behavior_dir: str | Path,
    conversation_dir: str | Path,
    similarity_threshold: float = 0.5,
) -> dict:
    """
    Search from end (latest to earliest) and return lines with similarity >= threshold.
    """
    kb, kc = _normalize_allocation(allocation)
    qemb = get_embedding(search_content)
    behavior_rows = list(reversed(_build_behavior_rows(behavior_dir)))
    conversation_rows = list(reversed(_build_conversation_rows(conversation_dir)))

    b_out = []
    for r in behavior_rows:
        if len(b_out) >= kb:
            break
        if _cosine_similarity(qemb, r["embedding"]) >= similarity_threshold:
            b_out.append(_row_to_output(r))

    c_out = []
    for r in conversation_rows:
        if len(c_out) >= kc:
            break
        if _cosine_similarity(qemb, r["embedding"]) >= similarity_threshold:
            c_out.append(_row_to_output(r))

    return {"behavior": b_out, "conversation": c_out}