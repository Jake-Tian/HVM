import glob
import json
import pickle
import re
import sys
import traceback
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel

from classes.output_structure import GraphOutputFormat, ParseQueryOutput, VideoOutputFormat
from reason_ablation import reason_k30, reason_no_allocation
from utils.edge_to_string import low_level_edge_to_string
from utils.general import Tee, find_pkl_files
from utils.llm import generate_text_response, get_embedding
from utils.mllm_gpt import generate_messages, get_response
from utils.prompts import (
    prompt_agent_verify_answer_referencing,
    prompt_graph_video,
    prompt_parse_query,
    prompt_video_answer,
    prompt_video_answer_final,
)


class GraphRoundDecisionOutput(BaseModel):
    answer: bool
    content: str
    summary: str | None


def _normalize_character_name(name: str | None) -> str | None:
    if not isinstance(name, str):
        return None
    n = name.strip()
    if not n:
        return None
    if n.startswith("<") and n.endswith(">"):
        return n
    return f"<{n}>"


def _is_behavior_edge(edge) -> bool:
    # Temporarily exclude high-level/appearance edges from retrieval.
    if edge is None:
        return False
    if getattr(edge, "clip_id", 0) <= 0:
        return False
    scene = str(getattr(edge, "scene", "") or "").strip().lower()
    if scene in {"high-level", "appearance"}:
        return False
    return True


def _all_behavior_edges(graph) -> list:
    return [edge for edge in graph.edges.values() if _is_behavior_edge(edge)]


def _ensure_query_triple(triple: list | tuple | None, fallback_content: str) -> list:
    src, content, tgt = "?", fallback_content, "?"
    sw, cw, tw = 0.2, 1.0, 0.2
    if isinstance(triple, (list, tuple)):
        if len(triple) > 0 and triple[0] is not None:
            src = str(triple[0])
        if len(triple) > 1 and triple[1] is not None and str(triple[1]).strip():
            content = str(triple[1])
        if len(triple) > 2 and triple[2] is not None:
            tgt = str(triple[2])
        if len(triple) > 3 and triple[3] is not None:
            sw = float(triple[3])
        if len(triple) > 4 and triple[4] is not None:
            cw = float(triple[4])
        if len(triple) > 5 and triple[5] is not None:
            tw = float(triple[5])
    return [src, content, tgt, sw, cw, tw]


def _build_text_query_triples(
    search_content: str,
    target: str | None = None,
    target_weight: float = 0.8,
    content_weight: float = 1.0,
) -> list[list]:
    normalized_target = _normalize_character_name(target) if target else None
    if normalized_target:
        return [[normalized_target, search_content, "?", target_weight, content_weight, 0.2]]
    return [["?", search_content, "?", 0.2, content_weight, 0.2]]


def _score_behavior_edges(graph, edges: list, query_triples: list[list]) -> list[tuple[float, object]]:
    if not edges:
        return []
    normalized_triples = [_ensure_query_triple(t, fallback_content="?") for t in query_triples]
    triple_embeddings = []
    for triple in normalized_triples:
        src, content, tgt = triple[0], triple[1], triple[2]
        src_emb = None
        if isinstance(src, str) and src and src != "?":
            src_emb = get_embedding(src.strip("<>") if src.startswith("<") and src.endswith(">") else src)
        content_emb = None
        if isinstance(content, str) and content and content != "?":
            content_emb = get_embedding(content)
        tgt_emb = None
        if isinstance(tgt, str) and tgt and tgt != "?":
            tgt_emb = get_embedding(tgt.strip("<>") if tgt.startswith("<") and tgt.endswith(">") else tgt)
        triple_embeddings.append([src_emb, content_emb, tgt_emb])

    scored = []
    for edge in edges:
        best = 0.0
        for idx, triple in enumerate(normalized_triples):
            score = graph._compute_edge_similarity(edge, triple, triple_embeddings[idx])
            best = max(best, score)
        scored.append((best, edge))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _iter_conversation_messages(graph, speaker_strict: list[str] | None):
    normalized = None
    if speaker_strict:
        normalized = set()
        for sp in speaker_strict:
            p = _normalize_character_name(sp)
            if p:
                normalized.add(p)
        if not normalized:
            normalized = None

    for conv_id, conv in graph.conversations.items():
        if conv is None:
            continue
        if normalized is not None and not normalized.issubset(getattr(conv, "speakers", set())):
            continue
        for msg_idx, msg in enumerate(getattr(conv, "messages", [])):
            if not isinstance(msg, list) or len(msg) < 2:
                continue
            speaker = str(msg[0])
            content = str(msg[1])
            clip_id = int(msg[2]) if len(msg) >= 3 and msg[2] is not None else None
            emb = msg[3] if len(msg) >= 4 else None
            yield conv_id, msg_idx, clip_id, speaker, content, emb


def _search_conversation_messages(
    graph,
    search_content: str,
    k: int,
    speaker_strict: list[str] | None = None,
    target: str | None = None,
    clip_filter: tuple[str, int] | None = None,
    order_hint: str | None = None,
) -> list[dict]:
    if k <= 0:
        return []
    qemb = get_embedding(search_content)
    target_norm = (target or "").strip().lower()
    scored = []
    for conv_id, msg_idx, clip_id, speaker, content, emb in _iter_conversation_messages(graph, speaker_strict):
        if clip_filter and clip_id is not None:
            direction, anchor = clip_filter
            if direction == "before" and clip_id >= anchor:
                continue
            if direction == "after" and clip_id <= anchor:
                continue
        try:
            if emb is None:
                emb = get_embedding(f"{speaker.strip('<>')}: {content}")
            sim = float(graph._cosine_similarity(qemb, emb))
        except Exception:
            sim = 0.0
        text = f"{speaker}: {content}".lower()
        if target_norm and target_norm in text:
            sim += 0.25
        scored.append(
            {
                "conversation_id": conv_id,
                "message_index": msg_idx,
                "score": sim,
                "clip_id": clip_id if clip_id is not None else -1,
            }
        )
    if order_hint == "first":
        scored.sort(key=lambda x: (x["clip_id"], -x["score"]))
    elif order_hint == "last":
        scored.sort(key=lambda x: (-x["clip_id"], -x["score"]))
    else:
        scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]


def _conversation_results_to_text(graph, conversation_results: list[dict]) -> str:
    if not conversation_results:
        return "(none)"
    vanilla = []
    for item in conversation_results:
        vanilla.append(
            {
                "conversation_id": item["conversation_id"],
                "message_index": item["message_index"],
                "score": item.get("score", 0.0),
            }
        )
    text = graph.get_conversation_messages_with_context(vanilla, context_window=2)
    return text if text else "(none)"


def _extract_clip_candidates(text: str) -> list[int]:
    if not isinstance(text, str):
        return []
    return [int(x) for x in re.findall(r"\[(\d+)\]", text)]


def _resolve_anchor_clip(target: str | None, previous_rounds: list[dict]) -> int | None:
    if not isinstance(target, str) or not target.strip():
        return None
    direct = re.findall(r"\d+", target)
    if direct:
        try:
            return int(direct[0])
        except Exception:
            pass
    target_norm = target.strip().lower()
    for payload in reversed(previous_rounds):
        for line in str(payload.get("organized_results", "")).splitlines():
            if target_norm in line.lower():
                clips = _extract_clip_candidates(line)
                if clips:
                    return clips[0]
    return None


def _organize_graph_results(
    behavior_edges: list,
    conversation_results: list[dict],
    object_rows: list[tuple[str, int]] | None = None,
) -> str:
    behavior_text = low_level_edge_to_string(behavior_edges) if behavior_edges else "(none)"
    conv_text = _conversation_results_to_text(_GRAPH_CONTEXT["graph"], conversation_results)
    obj_text = "(none)"
    if object_rows:
        obj_lines = [f"- {name}: degree={deg}" for name, deg in object_rows]
        obj_text = "\n".join(obj_lines)
    return (
        "behavior:\n"
        f"{behavior_text}\n\n"
        "conversation:\n"
        f"{conv_text}\n\n"
        "object_list:\n"
        f"{obj_text}"
    )


def _build_tool_schemas() -> list[dict]:
    common_alloc = {
        "total_search_k": {"type": "integer", "minimum": 1, "maximum": 50},
        "k_behavior": {"type": "integer", "minimum": 0, "maximum": 50},
        "k_conversation": {"type": "integer", "minimum": 0, "maximum": 50},
    }
    common_props = {
        "search_content": {"type": "string"},
        "source_scope": {"type": "string", "enum": ["behavior", "conversation", "both"]},
        "speaker_strict": {"type": "array", "items": {"type": "string"}},
        **common_alloc,
    }
    return [
        {
            "type": "function",
            "function": {
                "name": "general_search",
                "description": "General semantic search over graph behavior edges and conversations.",
                "parameters": {
                    "type": "object",
                    "properties": common_props,
                    "required": ["search_content", "total_search_k", "k_behavior", "k_conversation"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_before",
                "description": "Search evidence before a target anchor (typically a clip line).",
                "parameters": {
                    "type": "object",
                    "properties": {**common_props, "target": {"type": "string"}},
                    "required": [
                        "search_content",
                        "target",
                        "total_search_k",
                        "k_behavior",
                        "k_conversation",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_after",
                "description": "Search evidence after a target anchor (typically a clip line).",
                "parameters": {
                    "type": "object",
                    "properties": {**common_props, "target": {"type": "string"}},
                    "required": [
                        "search_content",
                        "target",
                        "total_search_k",
                        "k_behavior",
                        "k_conversation",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_first",
                "description": "Search earliest matching graph evidence.",
                "parameters": {
                    "type": "object",
                    "properties": common_props,
                    "required": ["search_content", "total_search_k", "k_behavior", "k_conversation"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_last",
                "description": "Search latest matching graph evidence.",
                "parameters": {
                    "type": "object",
                    "properties": common_props,
                    "required": ["search_content", "total_search_k", "k_behavior", "k_conversation"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "character_search",
                "description": "Given content and a character target, return the highest-similarity connected edge.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        **common_props,
                        "target": {"type": "string"},
                        "target_weight": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                        "content_weight": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                    },
                    "required": [
                        "search_content",
                        "target",
                        "target_weight",
                        "content_weight",
                        "total_search_k",
                        "k_behavior",
                        "k_conversation",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "object_list",
                "description": "Return objects and their node degrees in the graph.",
                "parameters": {
                    "type": "object",
                    "properties": {"top_n": {"type": "integer", "minimum": 1, "maximum": 200}},
                    "required": ["top_n"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "object_search",
                "description": "Given content and an object target, return the highest-similarity connected edge.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        **common_props,
                        "target": {"type": "string"},
                        "target_weight": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                        "content_weight": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                    },
                    "required": [
                        "search_content",
                        "target",
                        "target_weight",
                        "content_weight",
                        "total_search_k",
                        "k_behavior",
                        "k_conversation",
                    ],
                    "additionalProperties": False,
                },
            },
        },
    ]


def _tool_selection_prompt(
    question: str,
    current_search_question: str,
    previous_summaries: str,
    latest_evidence: str,
) -> str:
    return (
        "Select exactly one retrieval tool call for graph QA.\n"
        "Use function calling only.\n\n"
        "Rules:\n"
        "- Round 1 already did general search; avoid repeating broad search unless justified.\n"
        "- Use search_before/search_after when target timeline anchor exists.\n"
        "- Use search_first/search_last for earliest/latest evidence.\n"
        "- Use character_search/object_search when a specific entity is central.\n"
        "- Use object_list when question likely needs inventory/degree hints.\n"
        "- source_scope controls modality: behavior, conversation, or both.\n"
        "- Keep k_behavior + k_conversation == total_search_k and total_search_k in [1,50].\n\n"
        f"Original question: {question}\n"
        f"Current search question: {current_search_question}\n"
        f"Previous summaries:\n{previous_summaries}\n"
        f"Latest evidence:\n{latest_evidence}\n"
    )


def _choose_tool_call(
    client: OpenAI,
    question: str,
    current_search_question: str,
    previous_summaries: str,
    latest_evidence: str,
) -> tuple[str, dict, int]:
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "Choose one graph retrieval tool call."},
            {
                "role": "user",
                "content": _tool_selection_prompt(
                    question=question,
                    current_search_question=current_search_question,
                    previous_summaries=previous_summaries,
                    latest_evidence=latest_evidence,
                ),
            },
        ],
        tools=_build_tool_schemas(),
        tool_choice="required",
    )
    msg = response.choices[0].message
    tool_calls = msg.tool_calls or []
    if not tool_calls:
        raise ValueError("No tool call returned.")
    call = tool_calls[0]
    args = json.loads(call.function.arguments or "{}")
    tokens = int((response.usage.total_tokens if response.usage else 0) or 0)
    return call.function.name, args, tokens


def _normalize_alloc(args: dict, default_kb: int, default_kc: int) -> tuple[int, int]:
    kb = int(args.get("k_behavior", default_kb) or 0)
    kc = int(args.get("k_conversation", default_kc) or 0)
    total = args.get("total_search_k")
    if isinstance(total, int):
        target = max(1, min(50, total))
        cur = kb + kc
        if cur <= 0:
            kb = target // 2
            kc = target - kb
        elif cur != target:
            kb = int(round(kb * target / cur))
            kb = max(0, min(target, kb))
            kc = target - kb
    kb = max(0, min(50, kb))
    kc = max(0, min(50, kc))
    if kb + kc <= 0:
        kb = max(1, default_kb)
    return kb, kc


def _run_graph_tool(
    graph,
    tool_name: str,
    tool_args: dict,
    fallback_search_question: str,
    default_kb: int,
    default_kc: int,
    default_speaker_strict: list[str] | None,
    parse_query_output,
    previous_rounds: list[dict],
):
    scope = tool_args.get("source_scope")
    if scope not in {"behavior", "conversation", "both"}:
        scope = "both"
    search_content = tool_args.get("search_content")
    if not isinstance(search_content, str) or not search_content.strip():
        search_content = fallback_search_question
    kb, kc = _normalize_alloc(tool_args, default_kb, default_kc)
    speaker_strict = tool_args.get("speaker_strict")
    if not isinstance(speaker_strict, list):
        speaker_strict = default_speaker_strict
    target = tool_args.get("target") if isinstance(tool_args.get("target"), str) else None

    behavior_edges = []
    conversation_results = []
    object_rows = None
    retrieval_target = target

    if tool_name == "object_list":
        degrees = graph.get_node_degrees()
        rows = []
        for obj_name in graph.objects.keys():
            rows.append((obj_name, int(degrees.get(obj_name, 0))))
        rows.sort(key=lambda x: x[1], reverse=True)
        top_n = int(tool_args.get("top_n", 30) or 30)
        object_rows = rows[: max(1, min(top_n, 200))]
        return {
            "behavior_edges": [],
            "conversation_results": [],
            "object_rows": object_rows,
            "search_question": search_content,
            "k_behavior": kb,
            "k_conversation": kc,
            "speaker_strict": speaker_strict,
            "target": retrieval_target,
            "source_scope": scope,
            "method": "object_list",
        }

    if tool_name == "general_search":
        query_triples = parse_query_output.query_triples if parse_query_output else None
        if not query_triples:
            query_triples = _build_text_query_triples(search_content)
        if scope in {"behavior", "both"} and kb > 0:
            scored = _score_behavior_edges(graph, _all_behavior_edges(graph), query_triples)
            behavior_edges = [edge for _, edge in scored[:kb]]
        if scope in {"conversation", "both"} and kc > 0:
            conversation_results = _search_conversation_messages(
                graph=graph,
                search_content=search_content,
                k=kc,
                speaker_strict=speaker_strict,
            )

    elif tool_name in {"search_before", "search_after"}:
        anchor_clip = _resolve_anchor_clip(target, previous_rounds)
        if anchor_clip is None:
            # safe fallback to general search if anchor can't be resolved
            tool_name = "general_search"
            query_triples = _build_text_query_triples(search_content)
            if scope in {"behavior", "both"} and kb > 0:
                scored = _score_behavior_edges(graph, _all_behavior_edges(graph), query_triples)
                behavior_edges = [edge for _, edge in scored[:kb]]
            if scope in {"conversation", "both"} and kc > 0:
                conversation_results = _search_conversation_messages(
                    graph=graph,
                    search_content=search_content,
                    k=kc,
                    speaker_strict=speaker_strict,
                )
        else:
            retrieval_target = f"clip_id={anchor_clip}"
            direction = "before" if tool_name == "search_before" else "after"
            query_triples = _build_text_query_triples(search_content)
            if scope in {"behavior", "both"} and kb > 0:
                scoped = []
                for edge in _all_behavior_edges(graph):
                    clip_id = int(getattr(edge, "clip_id", -1) or -1)
                    if direction == "before" and clip_id < anchor_clip:
                        scoped.append(edge)
                    if direction == "after" and clip_id > anchor_clip:
                        scoped.append(edge)
                scored = _score_behavior_edges(graph, scoped, query_triples)
                behavior_edges = [edge for _, edge in scored[:kb]]
            if scope in {"conversation", "both"} and kc > 0:
                conversation_results = _search_conversation_messages(
                    graph=graph,
                    search_content=search_content,
                    k=kc,
                    speaker_strict=speaker_strict,
                    clip_filter=(direction, anchor_clip),
                )

    elif tool_name in {"search_first", "search_last"}:
        query_triples = _build_text_query_triples(search_content)
        if scope in {"behavior", "both"} and kb > 0:
            scored = _score_behavior_edges(graph, _all_behavior_edges(graph), query_triples)
            if tool_name == "search_first":
                scored.sort(key=lambda x: (int(getattr(x[1], "clip_id", 10**9)), -x[0]))
            else:
                scored.sort(key=lambda x: (-int(getattr(x[1], "clip_id", -1)), -x[0]))
            behavior_edges = [edge for _, edge in scored[:kb]]
        if scope in {"conversation", "both"} and kc > 0:
            conversation_results = _search_conversation_messages(
                graph=graph,
                search_content=search_content,
                k=kc,
                speaker_strict=speaker_strict,
                order_hint="first" if tool_name == "search_first" else "last",
            )

    elif tool_name in {"character_search", "object_search"}:
        target_entity = str(tool_args.get("target") or "").strip()
        if not target_entity:
            raise ValueError(f"{tool_name} requires target.")
        t_weight = float(tool_args.get("target_weight", 1.0) or 1.0)
        c_weight = float(tool_args.get("content_weight", 1.0) or 1.0)
        if tool_name == "character_search":
            target_entity_norm = _normalize_character_name(target_entity)
            query_triples = _build_text_query_triples(
                search_content=search_content,
                target=target_entity_norm,
                target_weight=t_weight,
                content_weight=c_weight,
            )
        else:
            query_triples = [[target_entity, search_content, "?", t_weight, c_weight, 0.2]]
        if scope in {"behavior", "both"} and kb > 0:
            candidates = []
            for edge in _all_behavior_edges(graph):
                s = str(getattr(edge, "source", "") or "")
                t = str(getattr(edge, "target", "") or "")
                if tool_name == "character_search":
                    match = (s == target_entity_norm) or (t == target_entity_norm)
                else:
                    match = (s == target_entity) or (t == target_entity)
                if match:
                    candidates.append(edge)
            scored = _score_behavior_edges(graph, candidates, query_triples)
            behavior_edges = [edge for _, edge in scored[: max(1, kb)]]
        if scope in {"conversation", "both"} and kc > 0:
            conversation_results = _search_conversation_messages(
                graph=graph,
                search_content=search_content,
                k=kc,
                speaker_strict=speaker_strict,
                target=target_entity,
            )

    else:
        raise ValueError(f"Unknown tool: {tool_name}")

    return {
        "behavior_edges": behavior_edges,
        "conversation_results": conversation_results,
        "object_rows": object_rows,
        "search_question": search_content,
        "k_behavior": kb,
        "k_conversation": kc,
        "speaker_strict": speaker_strict,
        "target": retrieval_target,
        "source_scope": scope,
        "method": tool_name,
    }


def _graph_round_decision_prompt(question: str, extracted_knowledge: str) -> str:
    return (
        "You are deciding whether current graph evidence is sufficient to answer a question.\n"
        "Return JSON with fields:\n"
        "- answer: bool\n"
        "- content: if answer=true, final answer sentence; else improved next graph search query\n"
        "- summary: null if answer=true, else concise 2-4 sentence summary of useful evidence\n\n"
        "Rules:\n"
        "- Prefer answer=true when evidence reasonably supports a one-sentence answer.\n"
        "- If answer=false, content must be a focused next search query.\n"
        "- Keep summary strictly grounded in provided evidence.\n\n"
        f"Question: {question}\n"
        f"Extracted graph evidence:\n{extracted_knowledge}\n"
    )


# Shared context for formatting helper.
_GRAPH_CONTEXT: dict = {"graph": None}


def reason(graph, video_name, question, max_graph_rounds: int = 5):

    result = {
        'question': question,
        'parse_query_output': None,
        'graph_search_results': None,
        'graph_rounds': [],
        'decision_response': None,
        'video_answer_outputs': [],
        'token_summaries': {"parse_query": 0, "graph_answer": 0, "video_answer": 0}, 
        'final_answer': None,
    }
    
    print("================================================")
    print("Question: ", question)
    
    #--------------------------------
    # Part 1: Search the graph
    #--------------------------------
    print("\n[Step 1] Searching the graph...")
    try:
        # Parse query using LLM
        parse_query_response, tokens = generate_text_response(prompt_parse_query + "\n" + question, ParseQueryOutput)
        result['parse_query_output'] = str(parse_query_response)
        result['token_summaries']['parse_query'] = tokens
        print("Parse Query Output:")
        print(parse_query_response)
        
        _GRAPH_CONTEXT["graph"] = graph
        parse_allocation = parse_query_response.allocation if hasattr(parse_query_response, "allocation") else None
        k_low_level = int(getattr(parse_allocation, "k_low_level", 30) or 30)
        k_conversations = int(getattr(parse_allocation, "k_conversations", 20) or 20)
        speaker_strict = parse_query_response.speaker_strict if hasattr(parse_query_response, "speaker_strict") else None
        current_search_question = question
        accumulated_summaries: list[str] = []
        client = OpenAI()

        for round_id in range(1, max_graph_rounds + 1):
            retrieval_method = "general_search" if round_id == 1 else None
            retrieval_target = None
            selected_tool_args = None
            selector_tokens = 0
            previous_summaries_text = (
                "\n".join(f"Round {idx + 1}: {s}" for idx, s in enumerate(accumulated_summaries))
                if accumulated_summaries
                else "(none)"
            )
            latest_evidence = result["graph_rounds"][-1]["organized_results"] if result["graph_rounds"] else "(none)"

            if round_id == 1:
                tool_exec = _run_graph_tool(
                    graph=graph,
                    tool_name="general_search",
                    tool_args={
                        "search_content": question,
                        "source_scope": "both",
                        "k_behavior": k_low_level,
                        "k_conversation": k_conversations,
                        "total_search_k": k_low_level + k_conversations,
                    },
                    fallback_search_question=current_search_question,
                    default_kb=k_low_level,
                    default_kc=k_conversations,
                    default_speaker_strict=speaker_strict,
                    parse_query_output=parse_query_response,
                    previous_rounds=result["graph_rounds"],
                )
            else:
                tool_name, tool_args, selector_tokens = _choose_tool_call(
                    client=client,
                    question=question,
                    current_search_question=current_search_question,
                    previous_summaries=previous_summaries_text,
                    latest_evidence=latest_evidence,
                )
                result["token_summaries"]["graph_answer"] += int(selector_tokens or 0)
                selected_tool_args = tool_args
                tool_exec = _run_graph_tool(
                    graph=graph,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    fallback_search_question=current_search_question,
                    default_kb=k_low_level,
                    default_kc=k_conversations,
                    default_speaker_strict=speaker_strict,
                    parse_query_output=parse_query_response,
                    previous_rounds=result["graph_rounds"],
                )
                retrieval_method = tool_exec["method"]
                retrieval_target = tool_exec["target"]

            k_low_level = tool_exec["k_behavior"]
            k_conversations = tool_exec["k_conversation"]
            speaker_strict = tool_exec["speaker_strict"]
            current_search_question = tool_exec["search_question"]

            organized = _organize_graph_results(
                behavior_edges=tool_exec["behavior_edges"],
                conversation_results=tool_exec["conversation_results"],
                object_rows=tool_exec.get("object_rows"),
            )
            if result["graph_search_results"] is None:
                result["graph_search_results"] = organized

            round_payload = {
                "round_id": round_id,
                "search_question": current_search_question,
                "retrieval_method": retrieval_method,
                "retrieval_target": retrieval_target,
                "selected_tool_args": selected_tool_args,
                "k_low_level": k_low_level,
                "k_conversations": k_conversations,
                "speaker_strict": speaker_strict,
                "behavior_hits_count": len(tool_exec["behavior_edges"]),
                "conversation_hits_count": len(tool_exec["conversation_results"]),
                "organized_results": organized,
                "decision_response": None,
                "selector_tokens": int(selector_tokens or 0),
            }

            decision_prompt = _graph_round_decision_prompt(
                question=question,
                extracted_knowledge=organized,
            )
            decision_response, decision_tokens = generate_text_response(
                decision_prompt,
                GraphRoundDecisionOutput,
            )
            result["token_summaries"]["graph_answer"] += int(decision_tokens or 0)
            round_payload["decision_response"] = str(decision_response)
            result["graph_rounds"].append(round_payload)

            if decision_response.answer:
                result["final_answer"] = decision_response.content
                result["decision_response"] = str(decision_response)
                return result
            if decision_response.summary:
                accumulated_summaries.append(decision_response.summary)
            current_search_question = decision_response.content or current_search_question
        
    except Exception as e:
        raise Exception(f"Error searching graph: {e}")
    
    #--------------------------------
    # Part 2: Evaluate searched graph answer
    #--------------------------------
    print("\n[Step 2] Evaluating searched answer...")
    all_graph_knowledge = "\n\n".join(
        f"[Round {r['round_id']}] {r['organized_results']}" for r in result.get("graph_rounds", [])
    )
    if not all_graph_knowledge:
        all_graph_knowledge = str(result.get("graph_search_results") or "(none)")
    prompt = (
        prompt_graph_video
        + "\nExtracted knowledge from graph:\n"
        + all_graph_knowledge
        + "\nQuestion: "
        + question
    )
    try:
        decision_response, tokens = generate_text_response(prompt, GraphOutputFormat)
        result['decision_response'] = str(decision_response)
        result['token_summaries']['graph_answer'] = tokens
        print("Decision response: \n", decision_response)
    except Exception as e:
        raise Exception(f"Error evaluating searched answer: {e}")
    
    # If action is Answer, return immediately
    answer_or_search = decision_response.answer
    content = decision_response.content
    summary = decision_response.summary

    if answer_or_search: 
        result['final_answer'] = content
        return result
    
    #--------------------------------
    # Part 3: Watch the video clips
    #--------------------------------
    # Extract and validate clip IDs from decision content.
    if not isinstance(content, list):
        content = [content]
    clip_ids = []
    for item in content:
        try:
            clip_ids.append(int(item))
        except Exception:
            print(f"Warning: Ignoring invalid clip id from decision output: {item}")
    if not clip_ids:
        raise Exception(f"No valid clip ids returned for video search. Raw content: {content}")
    if len(clip_ids) > 5:
        clip_ids = clip_ids[:5]

    print(f"\n[Step 3] Watching video clips: {clip_ids}")
    summary_dict = dict()

    for clip_id in clip_ids[:-1]:
        print(f"Processing clip {clip_id}...")

        prompt = prompt_video_answer + "\nQuestion: " + question + "\nCurrent clip ID: " + str(clip_id) + "\nPrevious summaries:\n"
        if summary:
            prompt += str(summary)
        for key, value in summary_dict.items():
            prompt += "\n" + f"Clip {key}: {value}"

        frames_dir = Path(f"data/frames/{video_name}") / str(clip_id)
        images = sorted(glob.glob(str(frames_dir / "*.jpg")), key=lambda x: int(Path(x).stem))

        try:
            messages = generate_messages(images, prompt)
            response, tokens = get_response(messages, VideoOutputFormat)
            answer_or_search = response.answer
            clip_summary = response.content
            result['token_summaries']['video_answer'] += int(tokens or 0)
        except Exception as e:
            raise Exception(f"Error processing clip {clip_id}: {e}")

        result['video_answer_outputs'].append(str(response))

        if answer_or_search: 
            result['final_answer'] = clip_summary
            print("Final answer: \n", result['final_answer'])
            return result
        else:
            summary_dict[clip_id] = clip_summary

    # Watch last clip
    clip_id = clip_ids[-1]
    print(f"Processing last clip {clip_id}...")
    prompt = prompt_video_answer_final + "\nQuestion: " + question + "\nCurrent clip ID: " + str(clip_id) + "\nPrevious summaries:\n"
    if summary:
        prompt += str(summary)
    for key, value in summary_dict.items():
        prompt += "\n" + f"Clip {key}: {value}"
    frames_dir = Path(f"data/frames/{video_name}") / str(clip_id)
    images = sorted(glob.glob(str(frames_dir / "*.jpg")), key=lambda x: int(Path(x).stem))

    try:
        messages = generate_messages(images, prompt)
        response, tokens = get_response(messages)
        if response is None or (isinstance(response, str) and not response.strip()):
            # Fallback: use last summary or placeholder so final_answer is never null
            response = (
                summary_dict[clip_ids[-2]] if len(clip_ids) >= 2 and clip_ids[-2] in summary_dict
                else "No answer could be generated from the video."
            )
        result['final_answer'] = response
    except Exception as e:
        raise Exception(f"Error processing last clip {clip_id}: {e}")
    
    print("Final Answer:")
    print(result['final_answer'])

    return result


def evaluate_answer(question, ground_truth_answer, predicted_answer):
    prompt = prompt_agent_verify_answer_referencing.format(
        question=question,
        ground_truth_answer=ground_truth_answer,
        agent_answer=predicted_answer
    )
    
    try: 
        response, _ = generate_text_response(prompt)
        response = response.strip().upper()
        if response.startswith("YES"):
            return True
        elif response.startswith("NO"):
            return False
        else:
            # If response is ambiguous, default to False
            print(f"Warning: Unexpected evaluator response: {response}. Defaulting to False.")
            return False
    except Exception as e:
        print(f"Error evaluating answer: {e}. Defaulting to False.")
        return False


def main():

    original_stdout = sys.stdout
    log_file = open("log.txt", "w", encoding="utf-8")
    sys.stdout = Tee(log_file)

    # If no video names are provided, process all available graph files.
    if len(sys.argv) < 2:
        available_videos = sorted(find_pkl_files())
    else:
        available_videos = sys.argv[1:]

    with open(f"data/robot.json", "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    for video_name in available_videos:
        print("================================================")
        print(f"Processing video {video_name}...")

        output_json_path = Path(f"data/reasoning/{video_name}.json")
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path = Path(f"data/graphs/{video_name}.pkl")
        if not graph_path.exists():
            print(f"Skipping {video_name}: graph not found at {graph_path}")
            continue
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)

        video_questions = questions_data.get(video_name, {}).get("qa_list", [])
        reasoning_results = {}

        for video_question in video_questions:
            question_id = video_question.get("question_id")
            question = video_question.get("question", "")
            answer = video_question.get("answer", "")

            try:
                main_result = reason(graph, video_name, question)
                evaluate_correct = evaluate_answer(question, answer, main_result["final_answer"])
                main_result["evaluate_correct"] = evaluate_correct
            except Exception as e:
                print(f"Error processing question {question_id}: {e}")
                traceback.print_exc()
                main_result = str(e)

        with open(output_json_path, "w") as f:
            json.dump(reasoning_results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved reasoning results for {video_name} to {output_json_path}")

    sys.stdout = original_stdout
    log_file.close()

if __name__ == "__main__":
    main()
