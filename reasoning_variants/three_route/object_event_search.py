"""Deterministic object-centric event timelines for location reasoning."""

import math
import re


VALID_INTENTS = {"source", "destination", "current", "history"}

_PICKUP_MARKERS = (
    "pick up",
    "picks up",
    "picked up",
    "take",
    "takes",
    "took",
    "retrieve",
    "get",
    "remove",
)
_PLACEMENT_MARKERS = (
    "place",
    "places",
    "placed",
    "put",
    "puts",
    "return",
    "store",
    "hang",
    "insert",
    "set down",
)
_TRANSFER_MARKERS = ("give", "gives", "gave", "hands", "handed", "pass")
_HOLD_MARKERS = ("hold", "holds", "held", "carry", "carries")
_SPATIAL_MARKERS = (
    " is on",
    " is in",
    " is under",
    " is inside",
    " is near",
    " is beside",
    " is from",
    " was on",
    " was in",
    " hangs on",
    " in front of",
    " behind",
)


def _normalize(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def _tokens(text):
    return {
        token
        for token in _normalize(text).split()
        if token not in {"a", "an", "the", "s"}
    }


def _is_character(value):
    text = str(value or "")
    return text.startswith("<") and text.endswith(">")


def _lexical_score(query, candidate):
    query_norm = _normalize(query)
    candidate_norm = _normalize(candidate)
    if query_norm == candidate_norm:
        return 1.0
    query_tokens = _tokens(query)
    candidate_tokens = _tokens(candidate)
    if not query_tokens or not candidate_tokens:
        return 0.0
    overlap = len(query_tokens & candidate_tokens)
    token_score = (
        0.7 * overlap / len(candidate_tokens)
        + 0.3 * overlap / len(query_tokens)
    )
    containment = 0.85 if candidate_norm in query_norm or query_norm in candidate_norm else 0
    return max(token_score, containment)


def _cosine(left, right):
    if left is None or right is None or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _resolve_object_candidates(graph, query, embedding_fn=None):
    objects = getattr(graph, "objects", {}) or {}
    lexical = sorted(
        (
            (_lexical_score(query, name), name)
            for name in objects
        ),
        reverse=True,
    )
    exact = [item for item in lexical if item[0] == 1.0]
    if exact:
        ranked = exact + [item for item in lexical if item[0] < 1.0]
        return ranked[:3], "exact"

    if lexical and lexical[0][0] >= 0.8:
        return lexical[:3], "lexical"

    if embedding_fn is None:
        from utils.embedding import get_embedding

        embedding_fn = get_embedding
    try:
        query_embedding = embedding_fn(query)
        semantic = []
        for name, node in objects.items():
            score = _cosine(query_embedding, getattr(node, "embedding", None))
            semantic.append((score, name))
        semantic.sort(reverse=True)
        if semantic and semantic[0][0] > 0:
            return semantic[:3], "embedding"
    except Exception:
        pass
    return lexical[:3], "lexical"


def _incident_edges(graph, object_name):
    out_ids = set(getattr(graph, "adjacency_list_out", {}).get(object_name, []))
    in_ids = set(getattr(graph, "adjacency_list_in", {}).get(object_name, []))
    edge_ids = out_ids | in_ids
    if not edge_ids:
        edge_ids = {
            edge_id
            for edge_id, edge in getattr(graph, "edges", {}).items()
            if edge.source == object_name or edge.target == object_name
        }
    edges = [
        graph.edges[edge_id]
        for edge_id in edge_ids
        if edge_id in graph.edges and getattr(graph.edges[edge_id], "clip_id", 0) > 0
    ]
    return sorted(edges, key=lambda edge: (edge.clip_id, edge.id))


def _event_type(edge, object_name):
    relation = f" {str(edge.content or '').lower()} "
    if edge.source == object_name and any(marker in relation for marker in _SPATIAL_MARKERS):
        return "observed_location"
    if edge.target == object_name and _is_character(edge.source):
        if any(marker in relation for marker in _PICKUP_MARKERS):
            return "pickup"
        if any(marker in relation for marker in _PLACEMENT_MARKERS):
            return "placement"
        if any(marker in relation for marker in _TRANSFER_MARKERS):
            return "transfer"
        if any(marker in relation for marker in _HOLD_MARKERS):
            return "held"
    return "interaction"


def _preposition(relation):
    text = str(relation or "").lower()
    if "front of" in text:
        return "in front of"
    for word in ("under", "inside", "near", "beside", "behind", "from", "on", "in"):
        if re.search(rf"\b{word}\b", text):
            return word
    return ""


def _expand_location(graph, relation, target, clip_id, depth=2):
    if not target or _is_character(target):
        return str(target or "")
    prefix = _preposition(relation)
    phrase = f"{prefix} {target}".strip()
    current = target
    visited = {current}
    for _ in range(depth):
        if current not in getattr(graph, "objects", {}):
            break
        candidates = [
            edge
            for edge in _incident_edges(graph, current)
            if edge.source == current
            and edge.target
            and not _is_character(edge.target)
            and _event_type(edge, current) == "observed_location"
            and edge.clip_id <= clip_id
            and edge.target not in visited
        ]
        if not candidates:
            break
        edge = max(candidates, key=lambda item: (item.clip_id, item.id))
        next_prefix = _preposition(edge.content)
        phrase += f" {next_prefix} {edge.target}".rstrip()
        current = edge.target
        visited.add(current)
    return phrase


def _evidence(edge):
    target = f" {edge.target}" if edge.target is not None else ""
    return f"{edge.source} {edge.content}{target}"


def _build_events(graph, object_name):
    edges = _incident_edges(graph, object_name)
    spatial = []
    for edge in edges:
        if _event_type(edge, object_name) == "observed_location":
            spatial.append({
                "edge": edge,
                "location": _expand_location(
                    graph, edge.content, edge.target, edge.clip_id
                ),
            })

    events = []
    for index, edge in enumerate(edges, start=1):
        kind = _event_type(edge, object_name)
        source_location = None
        destination_location = None
        if kind == "observed_location":
            destination_location = _expand_location(
                graph, edge.content, edge.target, edge.clip_id
            )
        elif kind == "pickup":
            explicit_source = [
                item
                for item in spatial
                if item["edge"].clip_id == edge.clip_id
                and edge.id < item["edge"].id <= edge.id + 3
                and (
                    "from" in str(item["edge"].content).lower()
                    or str(item["edge"].content).lower().startswith("was ")
                )
            ]
            previous_locations = [
                item
                for item in spatial
                if (item["edge"].clip_id, item["edge"].id)
                < (edge.clip_id, edge.id)
            ]
            if explicit_source:
                source_location = explicit_source[0]["location"]
            elif previous_locations:
                source_location = previous_locations[-1]["location"]
        elif kind == "placement":
            following_locations = [
                item
                for item in spatial
                if item["edge"].clip_id == edge.clip_id
                and edge.id < item["edge"].id <= edge.id + 3
            ]
            if following_locations:
                destination_location = following_locations[0]["location"]

        events.append({
            "event_id": f"E{index}",
            "clip_id": edge.clip_id,
            "edge_id": edge.id,
            "event_type": kind,
            "actor": edge.source if _is_character(edge.source) else None,
            "source_location": source_location,
            "destination_location": destination_location,
            "evidence": _evidence(edge),
        })
    return events


def _matched_conversations(graph, object_name, limit=5):
    pattern = re.compile(rf"\b{re.escape(object_name)}\b", re.IGNORECASE)
    matches = []
    for conversation in getattr(graph, "conversations", {}).values():
        for message in getattr(conversation, "messages", []) or []:
            if not isinstance(message, (list, tuple)) or len(message) < 3:
                continue
            speaker, content, clip_id = message[0], message[1], message[2]
            if pattern.search(str(content or "")):
                matches.append({
                    "clip_id": clip_id,
                    "speaker": speaker,
                    "content": content,
                })
    return sorted(matches, key=lambda item: item["clip_id"])[:limit]


def build_object_event_report(graph, object_name, intent, embedding_fn=None):
    if intent not in VALID_INTENTS:
        raise ValueError(
            f"intent must be one of: {', '.join(sorted(VALID_INTENTS))}"
        )
    ranked, match_type = _resolve_object_candidates(
        graph, object_name, embedding_fn=embedding_fn
    )
    if not ranked:
        return {
            "query_object": object_name,
            "canonical_object": None,
            "match_type": "none",
            "alternatives": [],
            "intent": intent,
            "events": [],
            "conversations": [],
            "answer_candidates": [],
            "rewatch_clip_ids": [],
        }

    canonical = ranked[0][1]
    events = _build_events(graph, canonical)
    stable = [
        event for event in events
        if event["event_type"] == "observed_location"
        and event["destination_location"]
    ]
    if intent == "current":
        answer_candidates = stable[-1:] if stable else []
    elif intent == "source":
        answer_candidates = [
            event for event in events
            if event["event_type"] == "pickup" and event["source_location"]
        ]
    elif intent == "destination":
        answer_candidates = [
            event for event in events
            if event["event_type"] == "placement"
            and event["destination_location"]
        ]
    else:
        answer_candidates = stable

    rewatch = []
    if not answer_candidates and events:
        rewatch = [events[-1]["clip_id"]]
    return {
        "query_object": object_name,
        "canonical_object": canonical,
        "match_type": match_type,
        "alternatives": [
            {"name": name, "score": round(score, 3)}
            for score, name in ranked[1:]
        ],
        "intent": intent,
        "events": events,
        "conversations": _matched_conversations(graph, canonical),
        "answer_candidates": answer_candidates,
        "rewatch_clip_ids": sorted(set(rewatch)),
    }


def format_object_event_report(report):
    if not report.get("canonical_object"):
        return (
            f"Object event search: no graph object matched "
            f"'{report.get('query_object', '')}'."
        )
    alternatives = ", ".join(
        item["name"] for item in report.get("alternatives", [])
    ) or "None"
    lines = [
        "**Object Event Report**",
        f"Query object: {report['query_object']}",
        f"Canonical object: {report['canonical_object']} ({report['match_type']})",
        f"Alternative object nodes: {alternatives}",
        f"Location intent: {report['intent']}",
        "Timeline:",
    ]
    events = report.get("events", [])
    if len(events) > 40:
        lines.append(f"- {len(events) - 40} earlier events omitted")
        events = events[-40:]
    for event in events:
        transition = ""
        if event.get("source_location"):
            transition += f" | from={event['source_location']}"
        if event.get("destination_location"):
            transition += f" | to={event['destination_location']}"
        actor = f" | actor={event['actor']}" if event.get("actor") else ""
        lines.append(
            f"- {event['event_id']} [clip {event['clip_id']}] "
            f"{event['event_type']}{actor}{transition} | {event['evidence']}"
        )

    lines.append("Intent-matched candidates:")
    candidates = report.get("answer_candidates", [])
    if candidates:
        for event in candidates[-5:]:
            location = (
                event.get("source_location")
                or event.get("destination_location")
                or "unknown"
            )
            lines.append(
                f"- [clip {event['clip_id']}] {location} "
                f"({event['event_type']}, actor={event.get('actor') or 'unknown'})"
            )
    else:
        lines.append("- None")

    conversations = report.get("conversations", [])
    if conversations:
        lines.append("Object-matched conversations:")
        for item in conversations:
            lines.append(
                f"- [clip {item['clip_id']}] {item['speaker']}: {item['content']}"
            )
    rewatch = report.get("rewatch_clip_ids", [])
    lines.append(
        "Suggested rewatch clips: "
        + (", ".join(map(str, rewatch)) if rewatch else "None")
    )
    return "\n".join(lines)


def search_object_events(graph, object_name, intent, embedding_fn=None):
    """Return a formatted object timeline without any LLM calls."""
    report = build_object_event_report(
        graph, object_name, intent, embedding_fn=embedding_fn
    )
    return format_object_event_report(report)
