import json
from dataclasses import dataclass, asdict


@dataclass
class AbstractionConfig:
    """Configuration for incremental, threshold-based abstraction.

    Used by both the memorization pipeline (process_full_video.py) and the
    ablation script (abstraction_ablation.py) so that the same abstraction
    logic can be replayed at different frequencies without re-running the MLLM.

    Trigger model (unified single threshold per entity type):
        A character is summarized whenever its low-level (clip_id>0) degree
        crosses a multiple of `interval_node` (i.e. at interval_node, 2*interval_node,
        3*interval_node, ...). The first summary therefore happens at interval_node,
        and re-summaries happen every interval_node NEW low-level edges.
        Pairs behave the same with `interval_pair` over their shared low-level
        edge count.

        Because a pair's shared edges are a subset of each member's incident
        edges, a shared count >= interval_pair already implies both members
        have degree >= interval_pair, so no separate per-character degree gate
        is needed for pairs.
    """

    # Master switch for the incremental (replay-driven) abstraction phase.
    # If False, only the final round runs (one-shot abstraction at the end,
    # equivalent to the legacy behavior but gated by the lower bounds below).
    incremental_enabled: bool = False

    # Character-attribute trigger: summarize every interval_node NEW low-level
    # edges (first summary at interval_node).
    interval_node: int = 50

    # Character-relationship trigger: summarize every interval_pair NEW shared
    # low-level edges (first summary at interval_pair).
    interval_pair: int = 20

    # Final round (run after the incremental replay): summarize any character
    # whose low-level degree is at least final_lower_bound_node, and any pair
    # whose shared low-level edge count is at least final_lower_bound_pair.
    # This catches entities that accumulated evidence but never crossed an
    # interval boundary during the replay. Defaults match the legacy one-shot
    # pipeline's gates (degree >= 10 for characters, shared >= 3 for pairs).
    final_lower_bound_node: int = 10
    final_lower_bound_pair: int = 3

    @classmethod
    def from_json(cls, path_or_str: str) -> "AbstractionConfig":
        """Load a config from a JSON file path or a JSON string."""
        try:
            with open(path_or_str, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, OSError):
            data = json.loads(path_or_str)
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_file(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
