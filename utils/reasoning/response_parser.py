"""
Response parsing utilities for reasoning system.

This module provides functions to parse LLM responses from various prompts
used in the reasoning pipeline.
"""

import re


def parse_semantic_response(response):
    """
    Parse the response from prompt_semantic_video.
    
    Args:
        response: Raw string response from LLM
    
    Returns:
        dict with keys: 'action', 'content', 'summary' (optional)
    """
    if response is None:
        raise ValueError("Cannot parse None response. The LLM did not return any content.")
    
    if not isinstance(response, str):
        raise TypeError(f"Expected string response, got {type(response)}: {response}")
    
    action_match = re.search(r'Action:\s*\[(Answer|Search)\]', response, re.IGNORECASE)
    content_match = re.search(r'Content:\s*(.+?)(?:\n|Summary:|$)', response, re.DOTALL | re.IGNORECASE)
    summary_match = re.search(r'Summary:\s*(.+?)$', response, re.DOTALL | re.IGNORECASE)

    if action_match and content_match:
        action = action_match.group(1).strip()
        content = content_match.group(1).strip()
        summary = summary_match.group(1).strip() if summary_match else None
        return {'action': action, 'content': content, 'summary': summary}

    # Fallback: LLM sometimes returns simplified "[Answer] <content>" or "[Search] <content>"
    fallback_answer = re.search(r'\[Answer\]\s*(.+)', response, re.DOTALL | re.IGNORECASE)
    fallback_search = re.search(r'\[Search\]\s*(.+)', response, re.DOTALL | re.IGNORECASE)
    if fallback_answer:
        content = fallback_answer.group(1).strip()
        return {'action': 'Answer', 'content': content, 'summary': None}
    if fallback_search:
        content = fallback_search.group(1).strip()
        return {'action': 'Search', 'content': content, 'summary': None}

    raise ValueError(f"Could not parse semantic response. Response: {response}")


def parse_video_response(response):
    """
    Parse the response from prompt_video_answer.
    
    Args:
        response: Raw string response from LLM
    
    Returns:
        dict with keys: 'action', 'content'
    """
    if response is None:
        raise ValueError("Cannot parse None response. The LLM did not return any content.")
    
    if not isinstance(response, str):
        raise TypeError(f"Expected string response, got {type(response)}: {response}")
    
    action_match = re.search(r'Action:\s*\[(Answer|Search)\]', response, re.IGNORECASE)
    content_match = re.search(r'Content:\s*(.+?)$', response, re.DOTALL | re.IGNORECASE)
    
    if not action_match or not content_match:
        raise ValueError(f"Could not parse video response. Response: {response}")
    
    action = action_match.group(1).strip()
    content = content_match.group(1).strip()
    
    return {
        'action': action,
        'content': content
    }


def extract_clip_ids(content):
    """
    Extract clip IDs from content string like "[15, 16]" or "[15,16]".
    
    Args:
        content: String containing clip IDs in bracket notation
    
    Returns:
        list of integers
    """
    # Try to match bracket notation [1, 2, 3]
    match = re.search(r'\[([\d,\s]+)\]', content)
    if match:
        clip_ids_str = match.group(1)
        clip_ids = [int(x.strip()) for x in clip_ids_str.split(',') if x.strip().isdigit()]
        return clip_ids
    
    # Fallback: try to extract all numbers
    numbers = re.findall(r'\d+', content)
    return [int(n) for n in numbers]
