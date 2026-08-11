import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.llm_gpt import generate_text_response
from utils.prompts import prompt_summary
from utils.general import strip_code_fences


def extract_clip_summary(episodic_memory, start_clip_id, end_clip_id):
    """
    Extract clip information (clip_id, scene, characters_behavior) from episodic memory JSON.
    
    Args:
        episodic_memory: Episodic memory dictionary
        start_clip_id: Starting clip ID (inclusive)
        end_clip_id: Ending clip ID (inclusive)
    
    Returns:
        str: Formatted string containing clip information
    """
    # Collect information for clips in the range
    summary_parts = []
    
    for clip_id in range(start_clip_id, end_clip_id + 1):
        clip_key = str(clip_id)
        
        if clip_key not in episodic_memory:
            continue
        
        clip_data = episodic_memory[clip_key]
        
        # Extract clip_id, scene, and characters_behavior
        scene = clip_data.get("scene", "Unknown scene")
        behaviors = clip_data.get("characters_behavior", [])
        
        # Format the output for this clip
        summary_parts.append(f"Clip {clip_id} - Scene: {scene}")
        summary_parts.append("Characters' Behavior:")
        
        if behaviors:
            summary_parts.extend(behaviors)
        else:
            summary_parts.append("(No behaviors recorded)")
        
        summary_parts.append("")  # Empty line between clips
    
    return "\n".join(summary_parts)


def summarize_clips(episodic_memory, start_clip_id, end_clip_id):
    """
    Extract clip information and generate a narrative summary using LLM.
    
    Args:
        episodic_memory: Episodic memory dictionary
        start_clip_id: Starting clip ID (inclusive)
        end_clip_id: Ending clip ID (inclusive)
    
    Returns:
        str: A concise narrative paragraph summarizing the clips
    """
    # Extract the clip summary
    clip_summary = extract_clip_summary(episodic_memory, start_clip_id, end_clip_id)
    
    # Create the full prompt
    full_prompt = prompt_summary + "\n" + clip_summary
    
    # Generate summary using LLM
    try:
        response = generate_text_response(full_prompt)
    except Exception as e:
        print(f"LLM call failed, retrying... Error: {e}")
        response = generate_text_response(full_prompt)
    
    # Clean the response (remove code fences if present)
    summary = strip_code_fences(response)
    
    return summary.strip()


if __name__ == "__main__":
    # Example usage
    with open("../data/episodic_memory/bedroom_01_10min.json", "r") as f:
        episodic_memory = json.load(f)
    summary = summarize_clips(episodic_memory, 1, 4)
    print(summary)
