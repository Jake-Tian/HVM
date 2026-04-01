from classes.edge_class import Edge


def edge_to_string(edges: list[Edge]) -> str:
    """
    Convert a list of edges into timestamped natural-language lines.
    
    Args:
        edges: List of Edge objects
    
    Returns:
        str: One line per edge in temporal order:
             [hh:mm:ss] source content target
    """
    if not edges:
        return ""

    def _edge_time_sort_key(edge: Edge):
        ts = getattr(edge, "timestamp", None)
        ts_str = str(ts).strip() if ts is not None else ""
        if len(ts_str) == 6 and ts_str.isdigit():
            return (0, int(ts_str))
        # Fallback keeps a deterministic order when timestamp is missing/invalid.
        return (1, getattr(edge, "id", 0))

    def _format_hhmmss(ts) -> str:
        ts_str = str(ts).strip() if ts is not None else ""
        if len(ts_str) == 6 and ts_str.isdigit():
            return f"{ts_str[0:2]}:{ts_str[2:4]}:{ts_str[4:6]}"
        return "??:??:??"

    # Sort edges by timestamp (earliest first).
    sorted_edges = sorted(edges, key=_edge_time_sort_key)

    lines = []

    for edge in sorted_edges:
        source_str = "" if edge.source is None else str(edge.source).strip()
        if edge.target is None:
            target_str = ""
        else:
            target_str = str(edge.target).strip()

        if target_str:
            action_line = f"{source_str} {edge.content} {target_str}"
        else:
            action_line = f"{source_str} {edge.content}"

        formatted_line = f"[{_format_hhmmss(getattr(edge, 'timestamp', None))}] {action_line}"
        lines.append(formatted_line)

    return "\n".join(lines)
