"""
Aggregate the noise-injection experiment results for R1W3.

For each noise rate (p2/p5/p10) and the clean baseline, compute:
  1. Reasoning accuracy (vs baseline 50_30 clean run in data/ablation/reasoning_abs/50_30).
  2. Abstraction pollution rate: how many NEW high-level edges appeared in the
     injected graph compared to the clean 50_30 graph (data/ablation/graphs_abs/50_30).
  3. Noise-edge retrieval count: from the per-video noise-retrieval log written
     during reasoning, sum how many times injected edges were retrieved and in
     which tools.

Reads:
  - data/ablation/reasoning_abs/50_30/<video>.json        (clean 50_30 baseline accuracy)
  - data/ablation/reasoning_noise/<tag>/<video>.json      (injected reasoning accuracy)
  - data/ablation/graphs_abs/50_30/<video>.pkl            (clean 50_30 graph, for pollution diff)
  - data/ablation/graphs_noise/<tag>/<video>.pkl          (injected graph)
  - data/ablation/graphs_noise/<tag>/<video>.noise_manifest.json  (injected edge ids)
  - data/ablation/reasoning_noise/<tag>/<video>.noise.jsonl       (retrieval log)
"""
import glob
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VIDEOS = [
    "bedroom_01", "bedroom_06", "kitchen_09", "kitchen_17",
    "living_room_02", "living_room_15", "living_room_22", "office_01",
    "study_03", "study_05", "study_06", "study_18", "study_23",
]
TAGS = ["p2", "p5", "p10"]
CLEAN_REASON_DIR = "data/ablation/reasoning_abs/50_30"
CLEAN_GRAPH_DIR = "data/ablation/graphs_abs/50_30"


def load_correct(dpath, video):
    f = os.path.join(dpath, f"{video}.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    res = {}
    for k, q in d.items():
        if isinstance(q, dict) and isinstance(q.get("reasoning"), dict):
            res[k] = bool(q["reasoning"].get("evaluate_correct"))
    return res


def high_level_key(edge):
    """A stable key identifying a high-level abstraction edge (source, content, target)."""
    return (edge.source, str(edge.content), str(edge.target))


def hl_edge_set(graph):
    return {high_level_key(e) for e in graph.edges.values()
            if e.clip_id == 0 and e.scene == "high-level"}


def main():
    os.chdir(PROJECT_ROOT)
    rows = {}
    # Clean baseline (50_30, no noise).
    clean_acc = {}
    clean_hl = {}
    for v in VIDEOS:
        c = load_correct(CLEAN_REASON_DIR, v)
        if c is not None:
            clean_acc[v] = c
        gp = os.path.join(CLEAN_GRAPH_DIR, f"{v}.pkl")
        if os.path.exists(gp):
            clean_hl[v] = hl_edge_set(pickle.load(open(gp, "rb")))

    clean_total = sum(len(c) for c in clean_acc.values())
    clean_correct = sum(sum(1 for x in c.values() if x) for c in clean_acc.values())
    print("=" * 78)
    print("NOISE-INJECTION EXPERIMENT (R1W3): robustness to hallucinated triples")
    print("=" * 78)
    print(f"Clean baseline (50_30, no noise): {clean_correct}/{clean_total} = "
          f"{clean_correct/clean_total*100:.1f}%  ({len(clean_acc)} videos)")
    print()

    header = (f"{'tag':5s} {'acc':>10s} {'Δacc':>7s} | {'pollution':>10s} "
              f"{'noise_retrieved':>16s} {'retrieved_videos':>16s}")
    print(header)
    print("-" * len(header))

    per_tool = Counter()
    for tag in TAGS:
        gdir = f"data/ablation/graphs_noise/{tag}"
        rdir = f"data/ablation/reasoning_noise/{tag}"
        if not os.path.isdir(rdir):
            print(f"{tag:5s}  (no results dir)")
            continue

        # Accuracy on common questions vs clean.
        common_correct = 0
        common_total = 0
        # Pollution: new high-level edges in injected graph vs clean 50_30 graph.
        total_new_hl = 0
        total_clean_hl = 0
        # Noise retrieval: sum of noise-edge hits across all retrieval logs.
        total_noise_retrieved = 0
        retrieved_videos = 0

        for v in VIDEOS:
            inj = load_correct(rdir, v)
            if inj is None:
                continue
            # accuracy on common questions
            if v in clean_acc:
                for qid, ok in inj.items():
                    if qid in clean_acc[v]:
                        common_total += 1
                        if ok:
                            common_correct += 1
            # pollution
            gp = os.path.join(gdir, f"{v}.pkl")
            if os.path.exists(gp) and v in clean_hl:
                inj_hl = hl_edge_set(pickle.load(open(gp, "rb")))
                new = inj_hl - clean_hl[v]
                total_new_hl += len(new)
                total_clean_hl += len(clean_hl[v])
            # noise retrieval log
            nlog = os.path.join(rdir, f"{v}.noise.jsonl")
            if os.path.exists(nlog):
                vid_hits = 0
                for line in open(nlog):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    total_noise_retrieved += rec.get("n_noise", 0)
                    vid_hits += rec.get("n_noise", 0)
                    per_tool[rec.get("tool", "?")] += rec.get("n_noise", 0)
                if vid_hits > 0:
                    retrieved_videos += 1

        acc_str = f"{common_correct}/{common_total}"
        acc_pct = common_correct / common_total * 100 if common_total else 0
        delta = acc_pct - (clean_correct / clean_total * 100) if common_total else 0
        poll_pct = (total_new_hl / total_clean_hl * 100) if total_clean_hl else 0
        print(f"{tag:5s} {acc_str:>10s} {delta:>+6.1f}% | "
              f"{total_new_hl:>4d}/{total_clean_hl:<5d}({poll_pct:4.1f}%) "
              f"{total_noise_retrieved:>16d} {retrieved_videos:>16d}")

    print()
    print("Per-tool noise-edge retrieval (sum of noise hits):")
    for tool, n in per_tool.most_common():
        print(f"  {tool:25s}: {n}")
    print()
    print("Notes:")
    print("  - Δacc = injected accuracy - clean baseline accuracy (negative = degradation).")
    print("  - pollution = NEW high-level edges in injected graph vs clean 50_30 graph.")
    print("  - noise_retrieved = total times injected edges appeared in search top-k results.")
    print("  - retrieved_videos = # videos where at least one noise edge was retrieved.")


if __name__ == "__main__":
    main()
