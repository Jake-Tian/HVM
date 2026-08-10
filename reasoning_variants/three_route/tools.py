"""LangGraph agent tools for HVM reasoning.

Tools exposed to the planner/executor agent:
  - general_search          : semantic retrieval over the heterogeneous graph
                               (reuses HVM's existing parse_query + search logic)
  - search_temporal_context : graph actions/conversations around a clip_id
  - search_action_evidence  : frequency-only raw episodic behavior around a clip
  - search_object_events    : deterministic object location timeline
  - watch_video_clip        : location-only MLLM frame analysis (expensive)
  - complete_task           : signal that enough evidence has been gathered

Each search tool returns (observation_str, token_cost) so the agent can track
token usage. `watch_video_clip` dedup against already-watched clips is handled
in the executor node (langgraph_helper.py), not here.
"""

import glob
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal
from langchain.tools import tool

from utils.llm_gpt import generate_text_response
from utils.mllm_gpt import generate_messages, get_response
from reasoning_variants.three_route.object_event_search import (
    search_object_events as execute_search_object_events,
)
from utils.prompts import prompt_parse_query, prompt_parse_query_no_allocation
from utils.search import search_with_parse
from utils.edge_to_string import high_level_edges_to_string, low_level_edge_to_string
from classes.output_structure import ParseQueryOutput, ParseQueryOutputNoAllocation
from utils.token_usage import empty_usage, merge_usage


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


def _log_noise_retrieval(graph, tool, sub, edges):
    """If `graph` carries a noise-edge manifest (noise-injection experiments),
    count how many of the retrieved `edges` are injected noise edges and append a
    JSON line to `graph.noise_log_path`. No-op for normal (non-noise) graphs."""
    noise_ids = getattr(graph, "noise_edge_ids", None)
    if not noise_ids:
        return
    hits = [int(e.id) for e in edges if getattr(e, "id", None) in noise_ids]
    if not hits:
        return
    log_path = getattr(graph, "noise_log_path", None)
    if not log_path:
        return
    import json as _json
    rec = {"tool": tool, "sub": sub, "n_retrieved": len(edges),
           "n_noise": len(hits), "noise_edge_ids": hits}
    try:
        from pathlib import Path as _P
        _P(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as _f:
            _f.write(_json.dumps(rec) + "\n")
    except Exception:
        pass


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
    total_tokens = empty_usage()
    try:
        query_triples, spatial_constraint, speaker_strict, t = _parse_query_triples(query)
        total_tokens = merge_usage(total_tokens, t)

        result_sections = []

        if k_high_level > 0:
            high_level_edges = graph.search_high_level_edges(query_triples, k_high_level)
            if high_level_edges:
                _log_noise_retrieval(graph, "general_search", "high_level", high_level_edges)
                hl_str = high_level_edges_to_string(high_level_edges)
                if hl_str:
                    result_sections.append("**High-Level Information (Character Attributes and Relationships):**\n" + hl_str)

        if k_appearance > 0:
            appearance_edges = graph.search_appearance_edges(query_triples, k_appearance)
            if appearance_edges:
                _log_noise_retrieval(graph, "general_search", "appearance", appearance_edges)
                app_str = high_level_edges_to_string(appearance_edges)
                if app_str:
                    result_sections.append("**Appearance Information:**\n" + app_str)

        if k_low_level > 0:
            low_level_edges = graph.search_low_level_edges(
                query_triples, k_low_level, spatial_constraint
            )
            if low_level_edges:
                _log_noise_retrieval(graph, "general_search", "low_level", low_level_edges)
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

def execute_search_temporal_context(
    graph,
    clip_id,
    window=1,
):
    """Return graph actions and conversations around a clip."""
    result_sections = []

    temporal_edges = [
        edge for edge in graph.edges.values()
        if edge.clip_id > 0 and abs(edge.clip_id - clip_id) <= window
    ]
    if temporal_edges:
        _log_noise_retrieval(graph, "search_temporal_context", "temporal", temporal_edges)
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
# search_action_evidence
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16)
def _load_video_episodic_memory(video_name):
    path = Path("data/memorization") / f"{video_name}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return payload.get("episodic_memory", {})


def execute_search_action_evidence(
    video_name,
    clip_id,
    window=1,
    episodic_memory=None,
):
    """Return raw pre-triple behaviors and dialogue around one candidate clip."""
    memory = (
        episodic_memory
        if episodic_memory is not None
        else _load_video_episodic_memory(video_name)
    )
    if not memory:
        return f"No episodic memory found for video {video_name}.", 0

    sections = []
    for current_clip in range(max(1, clip_id - window), clip_id + window + 1):
        clip = memory.get(str(current_clip), memory.get(current_clip))
        if not clip:
            continue
        lines = [f"**Raw action evidence for clip {current_clip}:**"]
        behaviors = clip.get("characters_behavior") or []
        if behaviors:
            lines.extend(f"- {behavior}" for behavior in behaviors)
        else:
            lines.append("- No raw behavior descriptions.")
        conversations = clip.get("conversation") or []
        if conversations:
            lines.append("Dialogue:")
            for message in conversations:
                if isinstance(message, (list, tuple)) and len(message) >= 2:
                    lines.append(f"- {message[0]}: {message[1]}")
        sections.append("\n".join(lines))

    if not sections:
        return f"No raw action evidence found around clip {clip_id}.", 0
    return "\n\n".join(sections), 0


# ---------------------------------------------------------------------------
# get_frequency_stats
# ---------------------------------------------------------------------------

def _norm_action_key(edge):
    """Normalize an edge into a (subject, action, object) key for grouping."""
    def _strip(name):
        if name is None:
            return ""
        return str(name).strip().strip("<>").lower()
    return (_strip(edge.source), str(edge.content).strip().lower(), _strip(edge.target))


def execute_get_frequency_stats(graph, query, top_n=100):
    """Aggregate matching low-level edges into global counts.

    For "how many / usually / most" questions, a local top-k sample is biased.
    This tool scores all low-level edges against the parsed query triples,
    takes a wide top-N, then groups by normalized action and deduplicates clips
    within a 5-minute (10-clip) bucket so the same physical action captioned
    multiple times in quick succession is not overcounted.
    """
    total_tokens = empty_usage()
    try:
        parse_response, t = generate_text_response(
            prompt_parse_query_no_allocation + "\n" + query, ParseQueryOutputNoAllocation
        )
        total_tokens = merge_usage(total_tokens, t)

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

        # Score edges by the same similarity mechanism the graph uses for low-level.
        # Reuse graph._compute_edge_similarity when available; otherwise fall back
        # to a coarse lexical match so the tool is always usable.
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
        top_edges = [e for _, e in scored[:top_n]]
        _log_noise_retrieval(graph, "get_frequency_stats", "top_n", top_edges)

        # Group by normalized action, dedupe clips within a 10-clip (5-min) bucket.
        groups = {}
        for edge in top_edges:
            key = _norm_action_key(edge)
            if key not in groups:
                groups[key] = {"count": 0, "clips": [], "scenes": set()}
            bucket = (edge.clip_id // 10) * 10
            if not any(abs(edge.clip_id - c) < 10 for c in groups[key]["clips"]):
                groups[key]["clips"].append(edge.clip_id)
            groups[key]["count"] += 1
            if edge.scene:
                groups[key]["scenes"].add(edge.scene)

        if not groups:
            return "No matching actions found to aggregate.", total_tokens

        lines = [f"**Aggregated action statistics for '{query}' (top {top_n} relevant edges):**"]
        # Sort groups by number of distinct clips descending.
        ordered = sorted(groups.items(), key=lambda kv: len(kv[1]["clips"]), reverse=True)
        for (src, content, tgt), info in ordered[:25]:
            subject = src or "?"
            obj = f" {tgt}" if tgt else ""
            scenes = f" (scenes: {', '.join(sorted(info['scenes']))})" if info["scenes"] else ""
            clips = sorted(info["clips"])
            lines.append(
                f"- {subject} {content}{obj}: observed in {len(clips)} distinct clip-window(s) "
                f"{clips}{scenes}"
            )
        return "\n".join(lines), total_tokens
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
        "Only report what you can actually see in the frames. "
        "Report every supported level of spatial detail and distinguish source, "
        "destination, and current state."
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
        Read graph actions and graph conversations in and around a specific video
        clip (window=1 clip on each side).
        Use this ONLY after finding a candidate clip_id via general_search, to see events
        right before or after it and decide whether nearby graph descriptions are one
        continuous event or separate events.
        """
        return execute_search_temporal_context(graph, clip_id)

    @tool
    def search_action_evidence(clip_id: int):
        """
        Read raw pre-triple behavior descriptions and dialogue from the candidate
        clip and its adjacent clips. ACTION-FREQUENCY QUESTIONS ONLY. Use after
        graph search has found a candidate clip, when the graph does not reveal
        how many completed occurrences happened inside one episode. This tool is
        deterministic and does not call an LLM or watch video frames.
        """
        return execute_search_action_evidence(video_name, clip_id)

    @tool
    def search_object_events(
        object_name: str,
        intent: Literal["source", "destination", "current", "history"],
    ):
        """
        Build a deterministic event timeline for one object. LOCATION QUESTIONS ONLY.
        Use after general_search has identified the target object. This tool resolves
        the canonical object node, collects every incident graph edge, orders events
        by clip and edge order, reconstructs source/destination/current locations,
        expands container-to-furniture location chains, and returns a few exact
        object-matched conversation lines. It does not call an LLM.

        Args:
            object_name: Exact target object, preserving important attributes such
                as color, owner, or type.
            intent: `source` for where it was retrieved from, `destination` for
                where it was or should be put, `current` for latest stable location,
                or `history` for the full location timeline.
        """
        return execute_search_object_events(graph, object_name, intent), 0

    @tool
    def watch_video_clip(clip_id: int, focus: str):
        """
        Watch the raw video frames of a specific clip with the MLLM. EXPENSIVE.
        LOCATION QUESTIONS ONLY. Use after search_object_events has identified a
        candidate event clip, and only when graph evidence leaves an exact spatial
        ambiguity. Rewatch one candidate clip, not a broad time range.

        The `focus` must name the exact object, requested state (source,
        destination, or current), known candidate locations, and the one visual
        distinction to resolve. Ask the model to track the object before pickup,
        while held, and after release when relevant. Do not request a general clip
        summary or use rewatch to discover unrelated events.
        """
        return execute_watch_video_clip(video_name, clip_id, focus)

    @tool
    def complete_task(ready: bool):
        """
        Call this tool when you have enough information to answer the question,
        or when you exhaust your budget.
        """
        return "Task marked complete. Proceeding to final verification."

    return [
        general_search,
        search_temporal_context,
        search_action_evidence,
        search_object_events,
        watch_video_clip,
        complete_task,
    ]
