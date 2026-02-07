"""
Video processing utilities for reasoning system.

This module provides functions to process video clips and orchestrate
watching multiple clips in sequence.
"""

from pathlib import Path
from typing import Optional
import glob
from utils.mllm_pictures import generate_messages, get_response
from utils.prompts import prompt_video_answer, prompt_video_answer_final
from utils.token_monitor import TokenMonitor
from .response_parser import parse_video_response


def process_video_clip(clip_id, question, previous_summaries, frames_dir, is_last_clip, token_monitor: Optional[TokenMonitor] = None):
    """
    Process a single video clip.
    
    Args:
        clip_id: The clip ID to process
        question: The question to answer
        previous_summaries: List of previous summaries
        frames_dir: Path to the frames directory
        is_last_clip: Whether this is the last clip
    
    Returns:
        dict with keys: 'clip_id', 'video_answer_output', 'parsed_response' (if not last), 
                       'answer' (if found), 'is_last_clip' (if last)
    """
    # Get the folder for this clip
    clip_folder = frames_dir / str(clip_id)
    if not clip_folder.exists():
        print(f"Warning: Clip folder not found: {clip_folder}, skipping...")
        return {'error': f'Clip folder not found: {clip_folder}'}
    
    # Collect images in the current folder
    current_images = sorted(
        glob.glob(str(clip_folder / "*.jpg")),
        key=lambda p: int(Path(p).stem) if Path(p).stem.isdigit() else p,
    )
    
    if not current_images:
        print(f"Warning: No images found in {clip_folder}, skipping...")
        return {'error': f'No images found in {clip_folder}'}
    
    # Build the prompt for video answer
    if is_last_clip:
        # For the last clip, use prompt_video_answer_final
        prompt_parts = [prompt_video_answer_final]
        prompt_parts.append(f"\n\nQuestion: {question}")
        prompt_parts.append(f"\n\nCurrent clip ID: {clip_id}")
        
        if previous_summaries:
            summaries_text = "\n".join(previous_summaries)
            prompt_parts.append(f"\n\nPrevious summaries:\n{summaries_text}")
        else:
            prompt_parts.append("\n\nPrevious summaries: None (first clip)")
        
        video_prompt = "\n".join(prompt_parts)
    else:
        # For non-last clips, use prompt_video_answer with Action/Content format
        prompt_parts = [prompt_video_answer]
        prompt_parts.append(f"\n\nQuestion: {question}")
        prompt_parts.append(f"\n\nCurrent clip ID: {clip_id}")
        
        if previous_summaries:
            summaries_text = "\n".join(previous_summaries)
            prompt_parts.append(f"\n\nPrevious summaries:\n{summaries_text}")
        else:
            prompt_parts.append("\n\nPrevious summaries: None (first clip)")
        
        video_prompt = "\n".join(prompt_parts)
    
    # Generate messages with images and prompt
    try:
        messages = generate_messages(current_images, video_prompt)
    except Exception as e:
        raise Exception(f"Error generating messages for clip {clip_id}: {e}")
    
    # Get response from MLLM
    try:
        video_response = get_response(messages, token_monitor=token_monitor)
    except Exception as e:
        raise Exception(f"Error getting response for clip {clip_id}: {e}")
    
    result = {
        'clip_id': clip_id,
        'video_answer_output': video_response
    }
    
    if is_last_clip:
        # For the final clip, the response is just the answer (no Action/Content format)
        result['answer'] = video_response.strip()
        result['is_last_clip'] = True
    else:
        # Parse the video response
        try:
            video_parsed = parse_video_response(video_response)
            result['parsed_response'] = video_parsed
            
            # Check if we got an answer
            if video_parsed['action'].upper() == 'ANSWER':
                result['answer'] = video_parsed['content']
        except Exception as e:
            raise Exception(f"Error parsing video response for clip {clip_id}: {e}\nResponse: {video_response}")
    
    return result


def watch_video_clips(question, clip_ids, video_name, initial_summary=None, print_progress=False, token_monitor: Optional[TokenMonitor] = None):
    """
    Watch video clips in sequence and collect responses.
    
    Args:
        question: The question to answer
        clip_ids: List of clip IDs to watch
        video_name: Name of the video
        initial_summary: Initial summary from graph (optional)
        print_progress: Whether to print progress messages (default: False)
    
    Returns:
        dict with keys: 'video_answer_outputs' (list), 'final_answer'
    """
    # Initialize previous summaries
    previous_summaries = []
    if initial_summary:
        previous_summaries.append(f"Graph information: {initial_summary}")
    
    # Process each clip in sequence
    frames_dir = Path(f"data/frames/{video_name}")
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    
    video_answer_outputs = []
    
    for idx, clip_id in enumerate(clip_ids):
        is_last_clip = (idx == len(clip_ids) - 1)
        
        if print_progress:
            print(f"\n   Processing clip {clip_id} ({idx + 1}/{len(clip_ids)})...")
        
        clip_result = process_video_clip(clip_id, question, previous_summaries, frames_dir, is_last_clip, token_monitor=token_monitor)
        
        # Check for errors
        if 'error' in clip_result:
            if print_progress:
                print(f"   Error: {clip_result['error']}")
            continue
        
        video_answer_outputs.append(clip_result)
        
        if print_progress:
            print(f"   Clip {clip_id} response received.")
            if 'answer' in clip_result:
                print(f"   Answer found in clip {clip_id}!")
        
        # If we got an answer, return it
        if 'answer' in clip_result:
            return {
                'video_answer_outputs': video_answer_outputs,
                'final_answer': clip_result['answer']
            }
        
        # Otherwise, accumulate the summary for next clip
        if not is_last_clip:
            parsed = clip_result.get('parsed_response')
            if parsed and parsed['action'].upper() == 'SEARCH':
                previous_summaries.append(f"Clip {clip_id}: {parsed['content']}")
                if print_progress:
                    print(f"   Action: [Search] - continuing to next clip...")
            elif parsed:
                raise ValueError(f"Unknown action in video response: {parsed['action']}")
    
    # If we've watched all clips and still no answer, return the last summary or indicate failure
    if previous_summaries:
        final_answer = f"After watching clips {clip_ids}, the answer could not be definitively determined. Latest information: {previous_summaries[-1]}"
    else:
        final_answer = "Could not determine the answer after watching the requested video clips."
    
    return {
        'video_answer_outputs': video_answer_outputs,
        'final_answer': final_answer
    }
