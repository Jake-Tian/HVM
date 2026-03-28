import pickle
from pathlib import Path

from classes.hetero_graph import HeteroGraph

GRAPH_PATH = Path("data/graphs/DAY1.pkl")

with GRAPH_PATH.open("rb") as f:
    graph: HeteroGraph = pickle.load(f)

# Total degree = number of incident directed edges (outgoing + incoming).
degrees = graph.get_node_degrees()

# graph.objects maps object name -> ObjectNode
for name in sorted(graph.objects.keys()):
    d = degrees.get(name, 0)
    print(f"{name}: {d}")
