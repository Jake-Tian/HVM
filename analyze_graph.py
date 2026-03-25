import pickle
from classes.hetero_graph import HeteroGraph

with open("data/graphs/DAY1.pkl", "rb") as f:
    graph = pickle.load(f)

print(graph.edges)


