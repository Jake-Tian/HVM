"""
Simple LangChain tools for graph reasoning retrieval.
Style matches `test_langgraph.py`:
- plain `@tool` functions
- detailed argument docstrings
- lightweight validation helpers
"""

from __future__ import annotations
from langchain.tools import tool
from utils.edge_to_string import edge_to_string
from classes.hetero_graph import HeteroGraph


# Graph binding is configured by caller (e.g., in your own builder function).
_BOUND_GRAPH: HeteroGraph | None = None
_BOUND_QUERY: str | None = None
_BOUND_CHOICES: list | None = None


def set_graph(graph: HeteroGraph):
    """Bind graph instance for tool calls in this module."""
    global _BOUND_GRAPH
    _BOUND_GRAPH = graph


def set_question(query: str | None = None, choices: list | None = None):
    """Bind question context (query/choices) for prompt construction."""
    global _BOUND_QUERY, _BOUND_CHOICES
    _BOUND_QUERY = query
    _BOUND_CHOICES = choices


def get_bound_query() -> str | None:
    """Return currently bound query, if any."""
    return _BOUND_QUERY


def get_bound_choices() -> list | None:
    """Return currently bound choices, if any."""
    return _BOUND_CHOICES


def get_bound_graph() -> HeteroGraph | None:
    """Return currently bound graph, if any."""
    return _BOUND_GRAPH


def _require_graph() -> HeteroGraph:
    if _BOUND_GRAPH is None:
        raise ValueError("No graph is bound. Call set_graph(graph) before using search tools.")
    return _BOUND_GRAPH


def _validate_hhmmss(ts: str | None, *, allow_none: bool = False) -> str | None:
    if ts is None:
        if allow_none:
            return None
        raise ValueError("timestamp is required")
    t = str(ts).strip()
    # Accept both "hhmmss" and "hh:mm:ss", normalize to "hhmmss".
    if len(t) == 8 and t[2] == ":" and t[5] == ":":
        t = t.replace(":", "")
    if len(t) != 6 or not t.isdigit():
        raise ValueError(f"Expected hhmmss timestamp, got: {ts!r}")
    return t


def _validate_query_triple(query_triple) -> list:
    if not isinstance(query_triple, (list, tuple)) or len(query_triple) != 6:
        raise ValueError(
            "query_triple must have exactly 6 elements: "
            "[source, content, target, weight_source, weight_content, weight_target]"
        )
    return list(query_triple)


def _validate_query_triples(query_triples) -> list[list]:
    if not isinstance(query_triples, (list, tuple)) or len(query_triples) == 0:
        raise ValueError("query_triples must be a non-empty list of query triples")
    return [_validate_query_triple(t) for t in query_triples]


def _validate_object_query(obj) -> str:
    if obj is None:
        raise ValueError("object_name is required")
    text = str(obj).strip()
    if not text:
        raise ValueError("object_name is required")
    return text


def _reorder_output(output: dict) -> str:
    behavior = output["behavior"]
    conversation = output["conversation"]
    return f"Behavior: {edge_to_string(behavior)}\nConversation: {conversation}"

@tool
def general_search(
    query_triples: list,
    k_behavior: int,
    k_conversation: int,
    speaker_strict: list[str] | None = None,
):
    """
    General semantic retrieval across behavior edges and conversation messages.

    Args:
        query_triples: list of weighted triples. Each triple must be:
            [source, content, target, weight_source, weight_content, weight_target].
            Example:
            [["<Alice>", "puts", "phone", 0.9, 1.0, 0.8]]
        k_behavior: number of behavior edges to return.
        k_conversation: number of conversation messages to return.
        speaker_strict: optional strict speaker filter for conversations.
    """
    graph = _require_graph()
    qts = _validate_query_triples(query_triples)
    return _reorder_output(graph.general_search(
        query_triples=qts,
        k_behavior=int(k_behavior),
        k_conversation=int(k_conversation),
        speaker_strict=speaker_strict,
    ))


@tool
def search_within_time_range(
    begin_time: str | None,
    end_time: str | None,
    query_triple: list,
    k_behavior: int,
    k_conversation: int,
):
    """
    Temporal search inside an inclusive time range.

    Args:
        begin_time: inclusive lower bound in hhmmss format (or None for unbounded start).
        end_time: inclusive upper bound in hhmmss format (or None for unbounded end).
        query_triple: one weighted triple:
            [source, content, target, weight_source, weight_content, weight_target].
        k_behavior: number of behavior edges to return.
        k_conversation: number of conversation messages to return.
    """
    graph = _require_graph()
    bt = _validate_hhmmss(begin_time, allow_none=True)
    et = _validate_hhmmss(end_time, allow_none=True)
    qt = _validate_query_triple(query_triple)
    return _reorder_output(graph.search_within_time_range(
        begin_time=bt,
        end_time=et,
        triples=qt,
        k_behavior=int(k_behavior),
        k_conversation=int(k_conversation),
    ))


@tool
def search_before(
    timestamp: str,
    query_triple: list,
    k_behavior: int,
    k_conversation: int,
):
    """
    Search evidence before (and including) a timestamp.

    Args:
        timestamp: anchor time in hhmmss format (e.g., "123000").
        query_triple: one weighted triple:
            [source, content, target, weight_source, weight_content, weight_target].
        k_behavior: number of behavior edges to return.
        k_conversation: number of conversation messages to return.
    """
    graph = _require_graph()
    ts = _validate_hhmmss(timestamp, allow_none=False)
    qt = _validate_query_triple(query_triple)
    return _reorder_output(graph.search_before(
        timestamp=ts,
        triples=qt,
        k_behavior=int(k_behavior),
        k_conversation=int(k_conversation),
    ))


@tool
def search_after(
    timestamp: str,
    query_triple: list,
    k_behavior: int,
    k_conversation: int,
):
    """
    Search evidence after (and including) a timestamp.

    Args:
        timestamp: anchor time in hhmmss format (e.g., "123000").
        query_triple: one weighted triple:
            [source, content, target, weight_source, weight_content, weight_target].
        k_behavior: number of behavior edges to return.
        k_conversation: number of conversation messages to return.
    """
    graph = _require_graph()
    ts = _validate_hhmmss(timestamp, allow_none=False)
    qt = _validate_query_triple(query_triple)
    return _reorder_output(graph.search_after(
        timestamp=ts,
        triples=qt,
        k_behavior=int(k_behavior),
        k_conversation=int(k_conversation),
    ))


@tool
def search_first(
    query_triple: list,
    k_behavior: int,
    k_conversation: int,
):
    """
    Search from the beginning (earliest evidence first).

    Args:
        query_triple: one weighted triple:
            [source, content, target, weight_source, weight_content, weight_target].
        k_behavior: number of behavior edges to return.
        k_conversation: number of conversation messages to return.
    """
    graph = _require_graph()
    qt = _validate_query_triple(query_triple)
    return _reorder_output(graph.search_first(
        triples=qt,
        k_behavior=int(k_behavior),
        k_conversation=int(k_conversation),
    ))


@tool
def search_last(
    query_triple: list,
    k_behavior: int,
    k_conversation: int,
):
    """
    Search from the end (latest evidence first).

    Args:
        query_triple: one weighted triple:
            [source, content, target, weight_source, weight_content, weight_target].
        k_behavior: number of behavior edges to return.
        k_conversation: number of conversation messages to return.
    """
    graph = _require_graph()
    qt = _validate_query_triple(query_triple)
    return _reorder_output(graph.search_last(
        triples=qt,
        k_behavior=int(k_behavior),
        k_conversation=int(k_conversation),
    ))


@tool
def search_object(object_name: str):
    """
    Search object-centric memory.

    Args:
        object_name: object text query to search.

    Returns:
        A formatted string, including:
        - top-50 similar-ranked object nodes (displayed with degree)
        - full conversations with hard text matches in summary/messages
    """
    graph = _require_graph()
    obj = _validate_object_query(object_name)
    return graph.search_object(obj)

