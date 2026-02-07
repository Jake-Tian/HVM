"""
Reasoning with ablation variants and token monitoring.

Three modes:
- original: Full pipeline (graph search + optional video re-watch)
- no_rewatch: Graph only, never watch video
- no_highlevel: Skip high-level edges in search, still watch video when [Search]
"""

import pickle
from typing import Optional

from utils.llm import generate_text_response
from utils.prompts import (
    prompt_parse_query,
    prompt_semantic_video,
    prompt_semantic_answer_only,
    prompt_semantic_video_no_highlevel,
)
from utils.search import search_with_parse
from utils.reasoning import parse_semantic_response, extract_clip_ids, watch_video_clips
from utils.token_monitor import TokenMonitor


def _evaluate_semantic_answer(
    question,
    graph_search_results,
    prompt,
    token_monitor: Optional[TokenMonitor] = None,
    answer_only: bool = False,
):
    """Evaluate whether graph search results are sufficient to answer the question."""
    full_prompt = prompt + "\n\nExtracted knowledge from graph:\n" + graph_search_results + "\n\nQuestion: " + question
    try:
        semantic_response = generate_text_response(full_prompt, token_monitor=token_monitor)
    except Exception as e:
        raise Exception(f"Error generating semantic answer: {e}")
    if answer_only:
        # prompt_semantic_answer_only returns a direct answer string
        parsed = {"action": "Answer", "content": semantic_response.strip(), "summary": None}
    else:
        try:
            parsed = parse_semantic_response(semantic_response)
        except Exception as e:
            raise Exception(f"Error parsing semantic response: {e}\nResponse: {semantic_response}")
    return {"semantic_video_output": semantic_response, "parsed_response": parsed}


def reason_original(question, graph, video_name, token_monitor: Optional[TokenMonitor] = None):
    """Original pipeline: full graph search + optional video re-watch."""
    return _reason(
        question, graph, video_name,
        token_monitor=token_monitor,
        skip_high_level=False,
        allow_video_rewatch=True,
        semantic_prompt=prompt_semantic_video,
    )


def reason_no_rewatch(question, graph, video_name, token_monitor: Optional[TokenMonitor] = None):
    """Ablation: no video re-watch. Answer from graph only using prompt_semantic_answer_only."""
    return _reason(
        question, graph, video_name,
        token_monitor=token_monitor,
        skip_high_level=False,
        allow_video_rewatch=False,
        semantic_prompt=prompt_semantic_answer_only,
        answer_only=True,
    )


def reason_no_highlevel(question, graph, video_name, token_monitor: Optional[TokenMonitor] = None):
    """Ablation: no high-level edges in search, but still watch video when [Search]."""
    return _reason(
        question, graph, video_name,
        token_monitor=token_monitor,
        skip_high_level=True,
        allow_video_rewatch=True,
        semantic_prompt=prompt_semantic_video_no_highlevel,
    )


def _reason(
    question,
    graph,
    video_name,
    *,
    token_monitor: Optional[TokenMonitor] = None,
    skip_high_level: bool = False,
    allow_video_rewatch: bool = True,
    semantic_prompt: str,
    answer_only: bool = False,
):
    """
    Core reasoning logic with configurable ablation options.
    """
    result = {
        "question": question,
        "parse_query_output": None,
        "graph_search_results": None,
        "semantic_video_output": None,
        "video_answer_outputs": None,
        "final_answer": None,
    }

    # Part 1: Search the graph
    try:
        parse_query_response = generate_text_response(prompt_parse_query + "\n" + question, token_monitor=token_monitor)
        result["parse_query_output"] = parse_query_response

        graph_search_results = search_with_parse(
            question, graph, parse_query_response,
            skip_high_level=skip_high_level,
        )
        result["graph_search_results"] = graph_search_results
    except Exception as e:
        raise Exception(f"Error searching graph: {e}")

    # Part 2: Evaluate semantic answer
    try:
        semantic_result = _evaluate_semantic_answer(
            question, result["graph_search_results"],
            prompt=semantic_prompt,
            token_monitor=token_monitor,
            answer_only=answer_only,
        )
        result["semantic_video_output"] = semantic_result["semantic_video_output"]
        parsed = semantic_result["parsed_response"]
    except Exception as e:
        raise Exception(f"Error evaluating semantic answer: {e}")

    # If action is Answer, return immediately
    if parsed["action"].upper() == "ANSWER":
        result["final_answer"] = parsed["content"]
        result["video_answer_outputs"] = []
        return result

    # If action is Search but we don't allow video re-watch, return fallback (content is clip IDs, not answer)
    if not allow_video_rewatch:
        result["final_answer"] = "Cannot determine from graph. Video re-watch is disabled."
        result["video_answer_outputs"] = []
        return result

    # Part 3: Watch video clips
    if parsed["action"].upper() != "SEARCH":
        raise ValueError(f"Unknown action: {parsed['action']}")

    clip_ids = extract_clip_ids(parsed["content"])
    if not clip_ids:
        raise ValueError(f"Could not extract clip IDs from content: {parsed['content']}")

    try:
        video_result = watch_video_clips(
            question,
            clip_ids,
            video_name,
            initial_summary=parsed.get("summary"),
            print_progress=True,
            token_monitor=token_monitor,
        )
        result["video_answer_outputs"] = video_result["video_answer_outputs"]
        result["final_answer"] = video_result["final_answer"]
    except Exception as e:
        raise Exception(f"Error watching video clips: {e}")

    return result
