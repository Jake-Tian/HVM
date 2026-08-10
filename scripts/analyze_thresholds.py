"""Analyze low-level degree distributions across existing graphs to inform
default abstraction thresholds (tau_node, interval_node, tau_pair, lower bounds).

For each graph pkl, compute:
  - per-character low-level (clip_id>0) degree distribution
  - per-pair shared low-level edge count distribution
  - how the old ">10" gate and candidate new thresholds would partition characters
"""
import glob
import os
import pickle
import sys
from pathlib import Path
from statistics import mean, median

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def low_level_edges_of(graph, node):
    eids = graph.edges_of(node)
    return {eid for eid in eids if graph.edges.get(eid) is not None and graph.edges[eid].clip_id > 0}

def shared_low_level(graph, c1, c2):
    connected = graph.get_connected_edges(c1, c2)
    return sum(1 for e in connected if e.clip_id > 0)

def analyze(path):
    with open(path, "rb") as f:
        g = pickle.load(f)
    chars = [c for c in g.characters if c != "<robot>"]
    degrees = []
    for c in chars:
        degrees.append(len(low_level_edges_of(g, c)))
    # pairs: only among chars with degree>=5 to bound work
    eligible = [c for c, d in zip(chars, degrees) if d >= 5]
    pair_counts = []
    for i in range(len(eligible)):
        for j in range(i+1, len(eligible)):
            pair_counts.append(shared_low_level(g, eligible[i], eligible[j]))
    return degrees, pair_counts

def main():
    paths = sorted(glob.glob(str(PROJECT_ROOT / "data/graphs/*.pkl")))
    # exclude any preabstraction/variant pkls
    paths = [p for p in paths if "_preabstraction" not in os.path.basename(p)]
    all_degrees = []
    all_pairs = []
    per_video_char_counts = []
    for p in paths:
        d, pr = analyze(p)
        all_degrees.extend(d)
        all_pairs.extend(pr)
        per_video_char_counts.append(len(d))
    all_degrees.sort()
    all_pairs.sort()

    def pct(arr, q):
        if not arr:
            return 0
        return arr[min(len(arr)-1, int(q*len(arr)))]

    print(f"Graphs analyzed: {len(paths)}")
    print(f"Total non-robot characters: {len(all_degrees)}")
    print(f"Avg chars/video: {mean(per_video_char_counts):.1f}, median: {median(per_video_char_counts)}")
    print()
    print("=== Per-character low-level degree distribution ===")
    print(f"  min={all_degrees[0]}, p25={pct(all_degrees,0.25)}, median={median(all_degrees):.0f}, "
          f"p75={pct(all_degrees,0.75)}, p90={pct(all_degrees,0.90)}, max={all_degrees[-1]}")
    for thr in [5, 8, 10, 15, 20, 30]:
        n = sum(1 for d in all_degrees if d >= thr)
        print(f"  degree >= {thr}: {n} chars ({100*n/len(all_degrees):.1f}%)")
    print()
    print("=== Per-pair shared low-level edge count (pairs where both chars degree>=5) ===")
    if all_pairs:
        nonzero = [p for p in all_pairs if p > 0]
        print(f"  total pairs: {len(all_pairs)}, with shared edges: {len(nonzero)} ({100*len(nonzero)/len(all_pairs):.1f}%)")
        if nonzero:
            nonzero.sort()
            print(f"  among nonzero: min={nonzero[0]}, p25={pct(nonzero,0.25)}, median={median(nonzero):.0f}, "
                  f"p75={pct(nonzero,0.75)}, p90={pct(nonzero,0.90)}, max={nonzero[-1]}")
        for thr in [3, 5, 8, 10, 15]:
            n = sum(1 for p in all_pairs if p >= thr)
            print(f"  shared >= {thr}: {n} pairs ({100*n/len(all_pairs):.1f}% of all, {100*n/max(1,len(nonzero)):.1f}% of nonzero)")

if __name__ == "__main__":
    main()
