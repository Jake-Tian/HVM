import glob
import json
import pickle
import sys
import time
import traceback
from pathlib import Path
from classes.hetero_graph import HeteroGraph
from utils.llm import generate_text_response
from utils.mllm_pictures import generate_messages, get_response
from utils.prompts import prompt_generate_episodic_memory, prompt_extract_triples
from utils.general import strip_code_fences, load_video_list, Tee


def process_full_video(frames_dir, output_graph_path=None, output_episodic_memory_path=None):
    """
    Process video frames to build episodic and semantic memory.
    
    Args:
        frames_dir: Path to directory containing frame folders (e.g., "data/frames/gym_01")
        output_graph_path: Path to save the graph pickle file (default: "data/semantic_memory/{video_name}.pkl")
        output_episodic_memory_path: Path to save episodic memory JSON (default: "data/episodic_memory/{video_name}.json")
    
    Returns:
        tuple: (graph, episodic_memory) - The processed graph and episodic memory dictionary
    """
    frames_dir = Path(frames_dir)
    video_name = frames_dir.name
    
    # Set default output paths if not provided
    if output_graph_path is None:
        output_graph_path = f"data/semantic_memory/{video_name}.pkl"
    if output_episodic_memory_path is None:
        output_episodic_memory_path = f"data/episodic_memory/{video_name}.json"
    
    # Get sorted image folders
    image_folders = sorted(
        [str(folder) for folder in frames_dir.iterdir() if folder.is_dir()],
        key=lambda x: int(Path(x).name)
    )

    # image_folders = image_folders[:2] # Comment this out to process all clips
    
    character_appearance = "{}"
    previous_conversation = False
    episodic_memory = dict()
    graph = HeteroGraph()
    # Track final character appearance across all clips
    final_character_appearance = {}
    
    for folder in image_folders:
        try:
            print("--------------------------------")
            print("Processing folder: ", folder)
            clip_id = int(Path(folder).name)
            response_dict = dict()
            # Collect images in the current folder
            current_images = sorted(
                glob.glob(f"{folder}/*.jpg"),
                key=lambda p: int(Path(p).stem) if Path(p).stem.isdigit() else p,
            )

            #--------------------------------
            # Episodic Memory
            #--------------------------------
            # Safety check: ensure character_appearance is always a string for prompt construction
            if isinstance(character_appearance, dict):
                character_appearance_str = json.dumps(character_appearance, indent=2)
            elif isinstance(character_appearance, str):
                character_appearance_str = character_appearance
            else:
                print(f"Warning: character_appearance has unexpected type {type(character_appearance)}, defaulting to empty dict")
                character_appearance_str = "{}"
            
            prompt = "Character appearance from previous videos: \n" + character_appearance_str + "\n" + prompt_generate_episodic_memory
            messages = generate_messages(current_images, prompt)
            try:
                response = get_response(messages)
            except Exception as e:
                print(f"LLM call failed, retrying... Error: {e}")
                response = get_response(messages)
            # Unpack if get_response returns (content, usage) for token tracking
            if isinstance(response, tuple):
                response = response[0]
            response = strip_code_fences(response)
            response_dict = json.loads(response)

            # 1. Process the character's behavior
            behaviors = response_dict.get("characters_behavior", [])
            if not isinstance(behaviors, list):
                behaviors = []
            
            if behaviors and len(behaviors) > 0 and behaviors[0].startswith("Equivalence:"):
                equivalence_parts = behaviors[0].split(":")[1].split(",")
                if len(equivalence_parts) >= 2:
                    behaviors = behaviors[1:]
                    graph.rename_character(equivalence_parts[0].strip(), equivalence_parts[1].strip())
                else:
                    print(f"Warning: Malformed equivalence line '{behaviors[0]}', skipping rename")

            # 2. Process the character appearance
            character_appearance = response_dict.get("character_appearance", {})
            if not isinstance(character_appearance, dict):
                print(f"Warning: character_appearance is not a dictionary, got {type(character_appearance)}")
                character_appearance = {}
            final_character_appearance.update(character_appearance)

            # 3. Process the conversation
            conversation = response_dict.get("conversation", [])
            if not isinstance(conversation, list):
                conversation = []
            
            # Check if previous conversation ended (no conversation in current clip)
            # Extract summary before creating/updating conversation
            if previous_conversation and len(conversation) == 0 and graph.current_conversation_id is not None:
                try:
                    print(f"Extracting summary for completed conversation {graph.current_conversation_id}...")
                    result = graph.extract_conversation_summary(graph.current_conversation_id)
                    print(f"✓ Conversation summary extracted. Attributes: {len(result['character_attributes'])}, Relationships: {len(result['characters_relationships'])}")
                except Exception as e:
                    print(f"✗ Error extracting conversation summary: {e}")
                    traceback.print_exc()

            if len(conversation) > 0:
                graph.update_conversation(clip_id, conversation, previous_conversation=previous_conversation)
                previous_conversation = True  # Set to True for next iteration
            else:
                previous_conversation = False  # No conversation in this clip, reset for next iteration

            scene = response_dict.get("scene")

            #--------------------------------
            # Semantic Memory
            #--------------------------------
            # Ensure behaviors is a list of strings for join operation
            if behaviors:
                behavior_prompt = prompt_extract_triples + "\n" + "\n".join(str(b) for b in behaviors)
                try:
                    triples_response = generate_text_response(behavior_prompt)
                except Exception as e:
                    print(f"LLM call failed, retrying... Error: {e}")
                    triples_response = generate_text_response(behavior_prompt)
                triples_response = strip_code_fences(triples_response)
                triples = json.loads(triples_response)
            else: 
                triples = []
            # Pass character_appearance to insert_triples for matching and merging
            graph.insert_triples(triples, clip_id, scene, character_appearance=character_appearance)
            print(f"Inserted {len(triples)} triples into graph for clip {clip_id}")
            
            character_appearance = json.dumps(character_appearance)

            # Store episodic memory for this clip
            episodic_memory[clip_id] = {
                "folder": folder,
                "characters_behavior": behaviors,
                "conversation": conversation,
                "character_appearance": character_appearance,
                "scene": scene,
                "triples": triples
            }
        except Exception as e:
            print(f"✗ Error processing folder {folder}: {e}")
            traceback.print_exc()
            print("Continuing to next folder...")
            # Safety: ensure character_appearance is converted to string even after exception
            # to prevent TypeError on next iteration
            if isinstance(character_appearance, dict):
                character_appearance = json.dumps(character_appearance, indent=2)
            elif not isinstance(character_appearance, str):
                character_appearance = "{}"
            continue

    # Extract summary for any remaining active conversation at the end
    if previous_conversation and graph.current_conversation_id is not None:
        try:
            print(f"Extracting summary for final conversation {graph.current_conversation_id}...")
            result = graph.extract_conversation_summary(graph.current_conversation_id)
            print(f"✓ Final conversation summary extracted. Attributes: {len(result['character_attributes'])}, Relationships: {len(result['characters_relationships'])}")
        except Exception as e:
            print(f"✗ Error extracting final conversation summary: {e}")
            traceback.print_exc()

    # Insert character appearances as high-level edges
    if final_character_appearance:
        try:
            graph.insert_character_appearances(final_character_appearance)
        except Exception as e:
            print(f"✗ Error inserting character appearances: {e}")
            traceback.print_exc()

    try: 
        graph.node_embedding_insertion()
        graph.edge_embedding_insertion()
    except Exception as e:
        print(f"✗ Error inserting embeddings: {e}")
        traceback.print_exc()
    
    # --------------------------------
    # Abstract Memory
    # --------------------------------
    # Generate character attributes
    print("Generating character attributes...")
    print("Number of edges: ", len(graph.edges))
    degrees = graph.get_node_degrees()
    # Select all characters whose degree is greater than 10
    characters = [character for character in graph.characters if degrees.get(character, 0) > 10]

    for character in characters:
        try: 
            graph.character_attributes(character)
        except Exception as e:
            print(f"✗ Error generating character attributes for {character}: {e}")
            traceback.print_exc()
            print("Continuing to next character...")
            continue
    print("Character attributes generated.")
    print("Number of edges: ", len(graph.edges))

    # Generate character relationships
    # pair up characters and generate relationships
    for i in range(len(characters)-1):
        for j in range(i+1, len(characters)):
            try:
                graph.character_relationships(characters[i], characters[j])
            except Exception as e:
                print(f"✗ Error generating character relationships for {characters[i]} and {characters[j]}: {e}")
                traceback.print_exc()
                print("Continuing to next character pair...")
                continue
    print("Character relationships generated.")
    print("Number of edges: ", len(graph.edges))

    # Save the graph to a file
    output_graph_path = Path(output_graph_path)
    output_graph_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_graph_path, "wb") as f:
        pickle.dump(graph, f)

    # Save the episodic memory to a JSON file
    # output_episodic_memory_path = Path(output_episodic_memory_path)
    # output_episodic_memory_path.parent.mkdir(parents=True, exist_ok=True)
    # with open(output_episodic_memory_path, "w") as f:
    #     json.dump(episodic_memory, f, indent=2)
    # print(f"\n✓ Saved episodic memory for {len(episodic_memory)} clips to {output_episodic_memory_path}")
    
    return graph, episodic_memory


def main():
    """Main function to process video frames."""
    # Redirect stdout to both terminal and log.txt
    original_stdout = sys.stdout
    log_file = open("log.txt", "w", encoding="utf-8")
    sys.stdout = Tee(log_file)
    
    video_names = load_video_list()
    
    # Parse range from command line (e.g., "1-20" or "21-40")
    if len(sys.argv) > 1:
        start, end = map(int, sys.argv[1].split('-'))
        selected = video_names[start-1:end]
    else:
        selected = video_names
    
    # selected = ["kitchen_21"] # Comment this out to process all videos

    for video_name in selected:
        try:
            start_time = time.time()
            frames_dir = Path(f"data/frames/{video_name}")
            if not frames_dir.exists():
                print(f"Skipping {video_name}: frames not found")
                continue
            print(f"\nProcessing {video_name}...")
            graph, episodic_memory = process_full_video(frames_dir)
            print(f"✓ {video_name} complete. Graph has {len(graph.characters)} characters and {len(graph.edges)} edges.")
            end_time = time.time()
            print(f"Time taken: {end_time - start_time} seconds")
        except Exception as e:
            print(f"✗ Error processing video {video_name}: {e}")
            traceback.print_exc()
            print("Continuing to next video...")
            continue
    
    sys.stdout = original_stdout
    log_file.close()

if __name__ == "__main__":
    main()

