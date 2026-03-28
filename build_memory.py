
import json
import pickle
from pathlib import Path

from classes.hetero_graph import HeteroGraph
from utils.llm import generate_text_response, get_multiple_embeddings
from utils.prompts import prompt_extract_timetriples, prompt_summarize_conversation
from classes.output_structure import TimeTripleList


def build_memory(day):

    output_graph_path = Path(f"data/graphs/{day}.pkl")
    output_token_path = Path(f"data/token/{day}.json")
    
    # find all files in behaviors and conversations for that day
    behavior_files = sorted(
        Path("data/behaviors").glob(f"{day}_*.json"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    conversation_files = sorted(
        Path("data/conversations").glob(f"{day}_*.json"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    print(f"Found {len(behavior_files)} behavior files and {len(conversation_files)} conversation files for {day}")
    
    graph = HeteroGraph()
    token_summaries = {}

    # process each hour
    for i in range(len(behavior_files)):
        behavior_file = behavior_files[i]
        conversation_file = conversation_files[i]
        hour = behavior_file.stem.split("_")[-1]
        if hour not in token_summaries:
            token_summaries[hour] = {"triples": 0, "conversation": 0}
        with open(behavior_file, "r") as f:
            behavior_data = json.load(f)
        with open(conversation_file, "r") as f:
            conversation_data = json.load(f)
        
        intervals = [{"idx": i, "behaviors": [], "conversations": []} for i in range(12)]

        for row in behavior_data:
            minute = row[0][2:4]
            idx = int(minute) // 5
            intervals[idx]["behaviors"].append(row)

        for row in conversation_data:
            minute = row["start_time"][2:4]
            idx = int(minute) // 5
            intervals[idx]["conversations"].append(row)

        # process each 5-minute interval
        for interval in intervals:
            
            behaviors = interval["behaviors"]
            if len(behaviors) != 0: 
                prompt = prompt_extract_timetriples + "\n" + json.dumps(behaviors, ensure_ascii=False)
                try:
                    triple_list, tokens = generate_text_response(prompt, TimeTripleList)
                    token_summaries[hour]["triples"] += int(tokens or 0)
                except Exception as e:
                    print(f"LLM call failed for triple extraction, retrying... Error: {e}")
                    triple_list, tokens = generate_text_response(prompt, TimeTripleList)
                    token_summaries[hour]["triples"] += int(tokens or 0)

                graph.insert_triples(triple_list.triples)

            conversations = interval["conversations"]
            if len(conversations) != 0: 
                conversation_lines = [f"{msg['speaker']}: {msg['content']}" for msg in conversations]
                try:
                    embeddings = get_multiple_embeddings(conversation_lines)
                except Exception as e:
                    print(f"Embedding call failed for conversations, retrying... Error: {e}")
                    embeddings = get_multiple_embeddings(conversation_lines)
                conversation_str = "\n".join(conversation_lines)

                if len(conversations) <= 3:
                    summary = None
                else:
                    prompt = prompt_summarize_conversation + "\n" + json.dumps(conversation_str, ensure_ascii=False)
                    try:
                        summary, tokens = generate_text_response(prompt)
                        token_summaries[hour]["conversation"] += int(tokens or 0)
                    except Exception as e:
                        print(f"LLM call failed for conversation summary, retrying... Error: {e}")
                        summary, tokens = generate_text_response(prompt)
                        token_summaries[hour]["conversation"] += int(tokens or 0)
            
                graph.insert_conversation(conversations, embeddings, summary)
                print(f"Inserted conversation id.{graph.current_conversation_id} into graph")

    try: 
        graph.node_embedding_insertion()
    except Exception as e:
        print(f"Error inserting node embeddings: {e}, retrying...")
        graph.node_embedding_insertion()

    print("Characters: ", graph.characters)

    print(f"Saved token summaries to {output_token_path}")
    output_token_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_token_path, "w") as f:
        json.dump(token_summaries, f, indent=2)

    # Save the graph to a file
    print(f"Saving graph to {output_graph_path}")
    output_graph_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_graph_path, "wb") as f:
        pickle.dump(graph, f)

if __name__ == "__main__":
    build_memory("DAY1")