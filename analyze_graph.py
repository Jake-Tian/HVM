import pickle
from pathlib import Path
from classes.hetero_graph import HeteroGraph

GRAPH_PATH = Path("data/graphs/DAY1.pkl")

with GRAPH_PATH.open("rb") as f:
    graph: HeteroGraph = pickle.load(f)

print(graph.graph_summary())
print(graph.search_object("marker"))