import os
import pickle
import sys
from pathlib import Path

# Add the current directory to sys.path so that pickle can find the classes
sys.path.append(os.getcwd())

def count_edges():
    graphs_dir = Path("data/graphs")
    if not graphs_dir.exists():
        print(f"Directory {graphs_dir} does not exist.")
        return

    pkl_files = sorted(list(graphs_dir.glob("*.pkl")))
    
    if not pkl_files:
        print("No .pkl files found in data/graphs.")
        return

    print(f"{'Graph Name':<30} | {'Edges':<10} | {'Max Clip ID':<12}")
    print("-" * 60)
    
    total_edges = 0
    for pkl_file in pkl_files:
        try:
            with open(pkl_file, "rb") as f:
                graph = pickle.load(f)
            
            # The HeteroGraph class has an 'edges' attribute which is a dict
            edge_count = len(graph.edges)
            
            max_clip_id = "N/A"
            if edge_count > 0:
                # Calculate max clip_id
                max_clip_id = max(getattr(edge, 'clip_id', 0) for edge in graph.edges.values())

            print(f"{pkl_file.stem:<30} | {edge_count:<10} | {max_clip_id:<12}")
            total_edges += edge_count
        except Exception as e:
            print(f"{pkl_file.stem:<30} | Error: {e}")

    print("-" * 60)
    print(f"{'Total':<30} | {total_edges:<10} | {'':<12}")

if __name__ == "__main__":
    count_edges()
