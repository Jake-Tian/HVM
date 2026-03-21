
from classes.hetero_graph import HeteroGraph
from utils.general import load_json


def process_one_hour(day, hour):

    return




def build_memory(day):

    output_graph_path = f"data/graphs/{day}.pkl"
    output_json_path = f"data/memorization/{day}.json"
    if not output_graph_path.exists() or not output_json_path.exists():
        raise FileNotFoundError(f"Graph or JSON file not found: {output_graph_path} or {output_json_path}")
    
    graph = HeteroGraph()


def main():

    day_hour_dict = {
        "DAY1": ["11", "12", "13", "14", "17", "18", "19", "20", "21", "22"], 
        "DAY2": ["10", "11", "12", "13", "15", "16", "17", "18", "20", "21", "22"],
        "DAY3": ["11", "12", "14", "15", "16", "17", "18", "19", "20", "21", "22"],
        "DAY4": ["10", "11", "12", "13", "15", "16", "17", "18", "20", "21", "22"],
        "DAY5": ["11", "12", "13", "15", "16", "17", "18", "19", "20"],
        "DAY6": ["09", "10", "11", "12", "13", "15", "16", "17", "19", "20", "21", "22"],
        "DAY7": ["11", "12", "13", "14", "15", "17", "18", "19", "20"],
    }
    
    
    

