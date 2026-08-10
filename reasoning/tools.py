"""LangGraph agent tools for HVM reasoning.

Tools exposed to the planner/executor agent:
  - general_search          : semantic retrieval over the heterogeneous graph
                               (reuses HVM's existing parse_query + search logic)
  - search_temporal_context : actions/conversations in a window around a clip_id
  - get_frequency_stats     : candidate evidence for repeated completed actions
  - watch_video_clip        : MLLM frame analysis of a specific clip (expensive)
  - complete_task           : signal that enough evidence has been gathered

Each search tool returns (observation_str, token_cost) so the agent can track
token usage. `watch_video_clip` dedup against already-watched clips is handled
in the executor node (langgraph_helper.py), not here.
"""

import glob
import json
from pathlib import Path
from typing import Literal
from langchain.tools import tool
from pydantic import BaseModel, Field

from utils.llm_gpt import generate_text_response
from utils.mllm_gpt import generate_messages, get_response
from reasoning.prompts import (
    prompt_parse_query,
    prompt_parse_query_no_allocation,
)
from utils.search import search_with_parse
from utils.edge_to_string import high_level_edges_to_string, low_level_edge_to_string
from classes.output_structure import ParseQueryOutput, ParseQueryOutputNoAllocation


# ---------------------------------------------------------------------------
# general_search
# ---------------------------------------------------------------------------

def _normalize_triples(query_triples):
    normalized = []
    for one_triple in query_triples or []:
        if isinstance(one_triple, (list, tuple)) and len(one_triple) >= 6:
            row = list(one_triple)
            for idx in (3, 4, 5):
                if not isinstance(row[idx], (int, float)):
                    try:
                        row[idx] = float(row[idx])
                    except (TypeError, ValueError):
                        row[idx] = 1.0
            normalized.append(row)
        else:
            normalized.append(one_triple if isinstance(one_triple, list) else list(one_triple))
    return normalized


def _parse_query_triples(query):
    """Run the existing prompt_parse_query to extract triples/spatial/speaker.

    Returns (query_triples, spatial_constraint, speaker_strict, tokens).
    """
    parse_response, tokens = generate_text_response(
        prompt_parse_query + "\n" + query, ParseQueryOutput
    )
    query_triples, spatial_constraint, speaker_strict = [], None, None
    if isinstance(parse_response, ParseQueryOutput):
        query_triples = parse_response.query_triples
        spatial_constraint = parse_response.spatial_constraint
        speaker_strict = parse_response.speaker_strict
    elif isinstance(parse_response, dict):
        query_triples = parse_response.get("query_triples", [])
        spatial_constraint = parse_response.get("spatial_constraint")
        speaker_strict = parse_response.get("speaker_strict")
    elif isinstance(parse_response, tuple) and parse_response:
        resp = parse_response[0]
        query_triples = getattr(resp, "query_triples", [])
        spatial_constraint = getattr(resp, "spatial_constraint", None)
        speaker_strict = getattr(resp, "speaker_strict", None)

    if not query_triples:
        query_triples = [[query, "?", "?", 0.9, 0.5, 0.5]]
    query_triples = _normalize_triples(query_triples)
    return query_triples, spatial_constraint, speaker_strict, (tokens or 0)


def execute_general_search(graph, query, k_high_level, k_low_level,
                           k_conversations, k_appearance):
    """Semantic search across the graph using LLM-parsed triples.

    The LLM planner controls the per-modality budget (k_*); triples/spatial/
    speaker are extracted via the existing prompt_parse_query for quality.
    """
    total_tokens = 0
    try:
        query_triples, spatial_constraint, speaker_strict, t = _parse_query_triples(query)
        total_tokens += t

        result_sections = []

        if k_high_level > 0:
            high_level_edges = graph.search_high_level_edges(query_triples, k_high_level)
            if high_level_edges:
                hl_str = high_level_edges_to_string(high_level_edges)
                if hl_str:
                    result_sections.append("**High-Level Information (Character Attributes and Relationships):**\n" + hl_str)

        if k_appearance > 0:
            appearance_edges = graph.search_appearance_edges(query_triples, k_appearance)
            if appearance_edges:
                app_str = high_level_edges_to_string(appearance_edges)
                if app_str:
                    result_sections.append("**Appearance Information:**\n" + app_str)

        if k_low_level > 0:
            low_level_edges = graph.search_low_level_edges(
                query_triples, k_low_level, spatial_constraint
            )
            if low_level_edges:
                ll_str = low_level_edge_to_string(low_level_edges)
                if ll_str:
                    result_sections.append("**Low-Level Information (Actions and Events):**\n" + ll_str)

        if k_conversations > 0:
            conv_results = graph.search_conversations(query, k_conversations, speaker_strict)
            if conv_results:
                conv_str = graph.get_conversation_messages_with_context(conv_results)
                if conv_str:
                    result_sections.append("**Conversations:**\n" + conv_str)

        result_str = "\n\n".join(result_sections)
        if not result_str.strip():
            return "No relevant information found for this query.", total_tokens
        return result_str, total_tokens
    except Exception as e:
        return f"Error executing general_search: {e}", total_tokens


# ---------------------------------------------------------------------------
# search_temporal_context
# ---------------------------------------------------------------------------

def execute_search_temporal_context(graph, clip_id, window=1):
    """Return actions and conversations within `window` clips of `clip_id`."""
    result_sections = []

    temporal_edges = [
        edge for edge in graph.edges.values()
        if edge.clip_id > 0 and abs(edge.clip_id - clip_id) <= window
    ]
    if temporal_edges:
        ll_str = low_level_edge_to_string(temporal_edges)
        if ll_str:
            result_sections.append(f"**Actions around clip {clip_id} (window={window}):**\n" + ll_str)

    conv_lines = []
    for conv in graph.conversations.values():
        # A conversation may span multiple clips; check any of them.
        clips = getattr(conv, "clips", None) or [getattr(conv, "clip_id", None)]
        if any(c is not None and abs(c - clip_id) <= window for c in clips):
            conv_lines.append(f"Clip(s) {sorted(clips)}:\n" + conv.format_messages())
    if conv_lines:
        result_sections.append(f"**Conversations around clip {clip_id}:**\n" + "\n".join(conv_lines))

    result_str = "\n\n".join(result_sections)
    if not result_str.strip():
        return f"No temporal information found around clip {clip_id}.", 0
    return result_str, 0


# ---------------------------------------------------------------------------
# get_frequency_stats
# ---------------------------------------------------------------------------

class FrequencyEvent(BaseModel):
    clip_ids: list[int] = Field(default_factory=list)
    evidence: str
    occurrence_count: int = Field(default=1, ge=0)
    status: Literal["confirmed", "probable", "rejected", "merged"]


class FrequencyReport(BaseModel):
    counting_unit: str
    events: list[FrequencyEvent] = Field(default_factory=list)


_FREQUENCY_ANALYSIS_PROMPT = """Build an event ledger for an action-frequency question.

Question:
{query}

Candidate graph and raw episodic evidence:
{evidence}

Rules:
- Define one completed occurrence of the requested action as the counting_unit.
- Count completed actions, not objects, people, categories, locations, instructions,
  plans, preparation, or repeated descriptions of the same action.
- Merge one continuous action described in the same or adjacent clips.
- Keep independent episodes separate after a stop/restart, a later occurrence, or
  a new actor performing the requested action.
- One episode may contain multiple occurrences. Preserve explicit multiplicity such
  as "twice", "six posters", or several clearly completed repetitions in one clip.
- Prefer recall over an unsupported zero. Use probable when the target occurrence is
  strongly implied but completion is not fully explicit. Use rejected for irrelevant
  evidence and merged for duplicate descriptions.
- Do not invent an occurrence when neither graph nor raw evidence supports it.
- Keep the evidence field short and quote or closely paraphrase the supporting fact.

Return the complete event ledger."""


def _load_episodic_memory(video_name):
    path = Path("data/memorization") / f"{video_name}.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload.get("episodic_memory") or {}


def _group_adjacent_clips(clip_ids):
    episodes = []
    for clip_id in sorted(set(clip_ids)):
        if not episodes or clip_id > episodes[-1][-1] + 1:
            episodes.append([clip_id])
        else:
            episodes[-1].append(clip_id)
    return episodes


def _format_frequency_candidates(scored_edges, episodic_memory, max_clips=18):
    selected_clips = []
    for _, edge in scored_edges:
        if edge.clip_id not in selected_clips:
            selected_clips.append(edge.clip_id)
        if len(selected_clips) >= max_clips:
            break

    sections = []
    for episode in _group_adjacent_clips(selected_clips):
        lines = [f"Candidate episode clips: {episode}"]
        for clip_id in episode:
            clip_edges = [
                edge for _, edge in scored_edges
                if edge.clip_id == clip_id
            ][:8]
            for edge in clip_edges:
                target = f" {edge.target}" if edge.target else ""
                lines.append(
                    f"- Graph [{clip_id}]: {edge.source} {edge.content}{target}"
                )

            raw_clip = episodic_memory.get(
                str(clip_id), episodic_memory.get(clip_id, {})
            )
            for behavior in raw_clip.get("characters_behavior") or []:
                lines.append(f"- Raw [{clip_id}]: {behavior}")
            for message in raw_clip.get("conversation") or []:
                if isinstance(message, (list, tuple)) and len(message) >= 2:
                    lines.append(
                        f"- Dialogue [{clip_id}] {message[0]}: {message[1]}"
                    )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _format_frequency_report(report):
    if hasattr(report, "model_dump"):
        data = report.model_dump()
    elif isinstance(report, dict):
        data = report
    else:
        raise TypeError(f"Unsupported frequency report: {type(report)!r}")

    events = data.get("events") or []
    confirmed_count = sum(
        int(event.get("occurrence_count", 0) or 0)
        for event in events
        if event.get("status") == "confirmed"
    )
    probable_count = sum(
        int(event.get("occurrence_count", 0) or 0)
        for event in events
        if event.get("status") == "probable"
    )
    lines = [f"Counting unit: {data.get('counting_unit', '')}"]
    for index, event in enumerate(events, start=1):
        lines.append(
            f"Event {index} | clips {sorted(set(event.get('clip_ids') or []))} | "
            f"{event.get('status')} x{int(event.get('occurrence_count', 0) or 0)} | "
            f"{event.get('evidence', '')}"
        )
    lines.extend([
        f"Confirmed count: {confirmed_count}",
        f"Probable additional count: {probable_count}",
        f"Best count: {confirmed_count + probable_count}",
        "Use the best count unless other explicit evidence in the conversation contradicts it.",
    ])
    return "\n".join(lines)


def execute_get_frequency_stats(graph, video_name, query, top_n=100):
    """Build a recall-oriented event ledger for a repeated completed action."""
    total_tokens = 0
    try:
        parse_response, t = generate_text_response(
            prompt_parse_query_no_allocation + "\n" + query, ParseQueryOutputNoAllocation
        )
        total_tokens += (t or 0)

        query_triples = []
        if isinstance(parse_response, ParseQueryOutputNoAllocation):
            query_triples = parse_response.query_triples
        elif isinstance(parse_response, dict):
            query_triples = parse_response.get("query_triples", [])
        elif isinstance(parse_response, tuple) and parse_response:
            query_triples = getattr(parse_response[0], "query_triples", [])

        if not query_triples:
            query_triples = [[query, "?", "?", 0.9, 0.5, 0.5]]
        query_triples = _normalize_triples(query_triples)

        candidate_edges = [
            edge for edge in graph.edges.values()
            if edge.clip_id > 0 and edge.scene is not None
        ]
        if not candidate_edges:
            return "No low-level actions found in the graph to aggregate.", total_tokens

        # Reuse graph similarity when available, otherwise keep a zero-score fallback.
        scored = []
        if hasattr(graph, "_compute_edge_similarity") and hasattr(graph, "_get_node_embedding"):
            try:
                query_embeddings = []
                from utils.embedding import get_embedding
                for q_triple in query_triples:
                    q_source = q_triple[0] if len(q_triple) > 0 else None
                    q_content = q_triple[1] if len(q_triple) > 1 else None
                    q_target = q_triple[2] if len(q_triple) > 2 else None
                    src_emb = None
                    if q_source and q_source != "?" and isinstance(q_source, str):
                        src_name = q_source.strip("<>") if q_source.startswith("<") and q_source.endswith(">") else q_source
                        try:
                            src_emb = get_embedding(src_name)
                        except Exception:
                            src_emb = None
                    cnt_emb = None
                    if q_content and q_content != "?" and isinstance(q_content, str):
                        try:
                            cnt_emb = get_embedding(q_content)
                        except Exception:
                            cnt_emb = None
                    tgt_emb = None
                    if q_target and q_target != "?" and isinstance(q_target, str):
                        tgt_name = q_target.strip("<>") if q_target.startswith("<") and q_target.endswith(">") else q_target
                        try:
                            tgt_emb = get_embedding(tgt_name)
                        except Exception:
                            tgt_emb = None
                    query_embeddings.append([src_emb, cnt_emb, tgt_emb])
                for edge in candidate_edges:
                    best = 0.0
                    for i, q_triple in enumerate(query_triples):
                        sim = graph._compute_edge_similarity(edge, q_triple, query_embeddings[i])
                        if isinstance(sim, (int, float)) and sim > best:
                            best = sim
                    scored.append((best, edge))
            except Exception:
                scored = [(0.0, edge) for edge in candidate_edges]
        else:
            scored = [(0.0, edge) for edge in candidate_edges]

        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[:top_n]
        episodic_memory = _load_episodic_memory(video_name)
        evidence = _format_frequency_candidates(scored, episodic_memory)
        if not evidence.strip():
            return "No matching actions found to aggregate.", total_tokens

        report, t = generate_text_response(
            _FREQUENCY_ANALYSIS_PROMPT.format(query=query, evidence=evidence),
            FrequencyReport,
        )
        total_tokens += (t or 0)
        return _format_frequency_report(report), total_tokens
    except Exception as e:
        return f"Error executing get_frequency_stats: {e}", total_tokens


# ---------------------------------------------------------------------------
# watch_video_clip
# ---------------------------------------------------------------------------

def execute_watch_video_clip(video_name, clip_id, focus):
    """Run the MLLM over the frames of a single clip. Returns (text, tokens)."""
    frames_dir = Path(f"data/frames/{video_name}") / str(clip_id)
    if not frames_dir.exists():
        return f"Video clip {clip_id} not found at {frames_dir}.", 0

    images = sorted(glob.glob(str(frames_dir / "*.jpg")), key=lambda x: int(Path(x).stem))
    if not images:
        return f"No frames found for clip {clip_id}.", 0

    prompt = (
        f"Focus: {focus}\n\n"
        "Watch this 30-second video clip (sequential frames in chronological order) "
        "and describe what happens related to the focus. Be concise and factual. "
        "Only report what you can actually see in the frames."
    )
    try:
        messages = generate_messages(images, prompt)
        response, tokens = get_response(messages)
        if isinstance(response, str):
            return response, (tokens or 0)
        return str(response), (tokens or 0)
    except Exception as e:
        return f"Error watching video clip {clip_id}: {e}", 0


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

def get_tools(graph, video_name):
    """Build the @tool-decorated tool set bound to a specific graph + video."""

    @tool
    def general_search(query: str,
                       k_low_level: int,
                       k_conversations: int,
                       k_high_level: int,
                       k_appearance: int):
        """
        General semantic search across the video's memory graph.
        Use this FIRST for all questions to get a broad view and find relevant clip_ids.
        Allocate your budget (total k <= 50) across modalities based on the question:
          - k_low_level (0-40): Primary for behaviors, actions, temporal sequence, 'where is' queries.
          - k_conversations (0-40): Primary for 'why', dialogue, sentiment, or causal reasoning.
          - k_high_level (0-15): Secondary for character traits or relationships.
          - k_appearance (0-10): Use ONLY for physical looks, hair, or clothing. Set to 0 if irrelevant.
        The query triples / spatial / speaker filters are extracted automatically from `query`.
        Returns formatted text with [clip_id] timestamps for each piece of evidence.
        """
        return execute_general_search(graph, query, k_high_level, k_low_level,
                                       k_conversations, k_appearance)

    @tool
    def search_temporal_context(clip_id: int):
        """
        Search what happened in and around a specific video clip (window=1 clip on each side).
        Use this only when the answer depends on the local before/after context of an
        event already identified by general_search. This is not the default follow-up
        to general_search. If retrieval is insufficient or misses the target evidence,
        revise the general_search query and modality allocation instead.
        """
        return execute_search_temporal_context(graph, clip_id)

    @tool
    def get_frequency_stats(query: str):
        """
        Use ONLY when the original question asks for the number or frequency of
        repeated completed ACTION occurrences across time. The counting unit must
        be a completed verb occurrence. Do not use this for quantities of objects
        or people, pages, kinds, categories, distinct positions, procedural steps,
        dialogue mentions, or yes/no questions about habits. Returns a structured
        event ledger with confirmed, probable, and recall-oriented best counts.
        """
        return execute_get_frequency_stats(graph, video_name, query)

    @tool
    def watch_video_clip(clip_id: int, focus: str):
        """
        Watch the raw video frames of a specific clip with the MLLM. EXPENSIVE.
        Use this only after graph evidence identifies a candidate clip, and only when
        the unresolved answer is directly visible in that clip, such as an object's
        precise spatial position, clothing/color, object identity, or another concrete
        visual attribute. Do not use it for causes, intentions, relationships, dialogue
        or instructions, temporal order, or action frequency across the video. Do not
        call it merely because graph evidence is incomplete. Provide a narrow `focus`
        describing the single visual detail to verify.
        """
        return execute_watch_video_clip(video_name, clip_id, focus)

    @tool
    def complete_task(ready: bool):
        """
        Call this tool when you have enough information to answer the question,
        or when you exhaust your budget.
        """
        return "Task marked complete. Proceeding to final verification."

    return [general_search, search_temporal_context, get_frequency_stats,
            watch_video_clip, complete_task]
