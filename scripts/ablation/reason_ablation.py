import pickle
import json
import glob
import sys
import traceback
from pathlib import Path
from utils.llm_qwen import generate_text_response
from utils.mllm_qwen import generate_messages, get_response
from utils.prompts import prompt_parse_query, prompt_parse_query_k30, prompt_parse_query_no_allocation, prompt_graph_video, prompt_no_video_rewatch, prompt_video_answer, prompt_video_answer_final
from classes.output_structure import ParseQueryOutput, ParseQueryOutputNoAllocation, GraphOutputFormat, VideoOutputFormat
from utils.search import search_with_parse
from utils.general import find_pkl_files, Tee


def reason_k30(graph, video_name, question):

    result = {
        'question': question,
        'parse_query_output': None,
        'graph_search_results': None,
        'decision_response': None,
        'video_answer_outputs': [],
        'token_summaries': {"parse_query": 0, "graph_answer": 0, "video_answer": 0}, 
        'final_answer': None,
    }
    
    print("================================================")
    print("Ablation study: Reasoning with k=30")
    
    #--------------------------------
    # Part 1: Search the graph
    #--------------------------------
    print("\n[Step 1] Searching the graph...")
    try:
        # Parse query using LLM
        parse_query_response, tokens = generate_text_response(prompt_parse_query_k30 + "\n" + question, ParseQueryOutput)
        result['parse_query_output'] = str(parse_query_response)
        result['token_summaries']['parse_query'] = tokens
        print("Parse Query Output:")
        print(parse_query_response)
        
        # Search the graph with parsed query
        graph_search_results = search_with_parse(question, graph, parse_query_response)
        result['graph_search_results'] = graph_search_results
        
    except Exception as e:
        raise Exception(f"Error searching graph: {e}")
    
    #--------------------------------
    # Part 2: Evaluate searched graph answer
    #--------------------------------
    print("\n[Step 2] Evaluating searched answer...")
    prompt = prompt_graph_video + "\nExtracted knowledge from graph:\n" + result['graph_search_results'] + "\nQuestion: " + question
    try:
        decision_response, tokens = generate_text_response(prompt, GraphOutputFormat)
        result['decision_response'] = str(decision_response)
        result['token_summaries']['graph_answer'] = tokens
        print("Decision response: \n", decision_response)
    except Exception as e:
        raise Exception(f"Error evaluating searched answer: {e}")
    
    # If action is Answer, return immediately
    answer_or_search = decision_response.answer
    content = decision_response.content
    summary = decision_response.summary

    if answer_or_search: 
        result['final_answer'] = content
        return result
    
    #--------------------------------
    # Part 3: Watch the video clips
    #--------------------------------
    # Extract and validate clip IDs from decision content.
    if not isinstance(content, list):
        content = [content]
    clip_ids = []
    for item in content:
        try:
            clip_ids.append(int(item))
        except Exception:
            print(f"Warning: Ignoring invalid clip id from decision output: {item}")
    if not clip_ids:
        raise Exception(f"No valid clip ids returned for video search. Raw content: {content}")
    if len(clip_ids) > 5:
        clip_ids = clip_ids[:5]

    print(f"\n[Step 3] Watching video clips: {clip_ids}")
    summary_dict = dict()

    for clip_id in clip_ids[:-1]:
        print(f"Processing clip {clip_id}...")

        prompt = prompt_video_answer + "\nQuestion: " + question + "\nCurrent clip ID: " + str(clip_id) + "\nPrevious summaries:\n"
        if summary:
            prompt += str(summary)
        for key, value in summary_dict.items():
            prompt += "\n" + f"Clip {key}: {value}"

        frames_dir = Path(f"data/frames/{video_name}") / str(clip_id)
        images = sorted(glob.glob(str(frames_dir / "*.jpg")), key=lambda x: int(Path(x).stem))

        try:
            messages = generate_messages(images, prompt)
            response, tokens = get_response(messages, VideoOutputFormat)
            answer_or_search = response.answer
            clip_summary = response.content
            result['token_summaries']['video_answer'] += int(tokens or 0)
        except Exception as e:
            raise Exception(f"Error processing clip {clip_id}: {e}")

        result['video_answer_outputs'].append(str(response))

        if answer_or_search: 
            result['final_answer'] = clip_summary
            print("Final answer: \n", result['final_answer'])
            return result
        else:
            summary_dict[clip_id] = clip_summary

    # Watch last clip
    clip_id = clip_ids[-1]
    print(f"Processing last clip {clip_id}...")
    prompt = prompt_video_answer_final + "\nQuestion: " + question + "\nCurrent clip ID: " + str(clip_id) + "\nPrevious summaries:\n"
    if summary:
        prompt += str(summary)
    for key, value in summary_dict.items():
        prompt += "\n" + f"Clip {key}: {value}"
    frames_dir = Path(f"data/frames/{video_name}") / str(clip_id)
    images = sorted(glob.glob(str(frames_dir / "*.jpg")), key=lambda x: int(Path(x).stem))

    try:
        messages = generate_messages(images, prompt)
        response, tokens = get_response(messages)
        if response is None or (isinstance(response, str) and not response.strip()):
            response = (
                summary_dict[clip_ids[-2]] if len(clip_ids) >= 2 and clip_ids[-2] in summary_dict
                else "No answer could be generated from the video."
            )
        result['final_answer'] = response
    except Exception as e:
        raise Exception(f"Error processing last clip {clip_id}: {e}")
    
    print("Final Answer:")
    print(result['final_answer'])

    return result


def reason_no_allocation(graph, video_name, question):

    result = {
        'question': question,
        'parse_query_output': None,
        'graph_search_results': None,
        'decision_response': None,
        'video_answer_outputs': [],
        'token_summaries': {"parse_query": 0, "graph_answer": 0, "video_answer": 0},
        'final_answer': None,
    }

    print("================================================")
    print("Ablation study: Reasoning without allocation")

    #--------------------------------
    # Part 1: Search the graph
    #--------------------------------
    print("\n[Step 1] Searching the graph...")
    try:
        # Parse query using LLM (no allocation in output)
        parse_query_response, tokens = generate_text_response(
            prompt_parse_query_no_allocation + "\n" + question,
            ParseQueryOutputNoAllocation
        )
        result['parse_query_output'] = str(parse_query_response)
        result['token_summaries']['parse_query'] = tokens
        print("Parse Query Output:")
        print(parse_query_response)

        # Use fixed default allocation for graph search.
        parse_query_for_search = {
            "query_triples": parse_query_response.query_triples,
            "spatial_constraint": parse_query_response.spatial_constraint,
            "speaker_strict": parse_query_response.speaker_strict,
            "allocation": {
                "k_high_level": 10,
                "k_low_level": 20,
                "k_conversations": 20,
                "k_appearance": 0,
                "total_k": 50,
            },
        }

        # Search the graph with parsed query
        graph_search_results = search_with_parse(question, graph, parse_query_for_search)
        result['graph_search_results'] = graph_search_results

    except Exception as e:
        raise Exception(f"Error searching graph: {e}")

    #--------------------------------
    # Part 2: Evaluate searched graph answer
    #--------------------------------
    print("\n[Step 2] Evaluating searched answer...")
    prompt = prompt_graph_video + "\nExtracted knowledge from graph:\n" + result['graph_search_results'] + "\nQuestion: " + question
    try:
        decision_response, tokens = generate_text_response(prompt, GraphOutputFormat)
        result['decision_response'] = str(decision_response)
        result['token_summaries']['graph_answer'] = tokens
        print("Decision response: \n", decision_response)
    except Exception as e:
        raise Exception(f"Error evaluating searched answer: {e}")

    # If action is Answer, return immediately
    answer_or_search = decision_response.answer
    content = decision_response.content
    summary = decision_response.summary

    if answer_or_search:
        result['final_answer'] = content
        return result

    #--------------------------------
    # Part 3: Watch the video clips
    #--------------------------------
    # Extract and validate clip IDs from decision content.
    if not isinstance(content, list):
        content = [content]
    clip_ids = []
    for item in content:
        try:
            clip_ids.append(int(item))
        except Exception:
            print(f"Warning: Ignoring invalid clip id from decision output: {item}")
    if not clip_ids:
        raise Exception(f"No valid clip ids returned for video search. Raw content: {content}")
    if len(clip_ids) > 5:
        clip_ids = clip_ids[:5]

    print(f"\n[Step 3] Watching video clips: {clip_ids}")
    summary_dict = dict()

    for clip_id in clip_ids[:-1]:
        print(f"Processing clip {clip_id}...")

        prompt = prompt_video_answer + "\nQuestion: " + question + "\nCurrent clip ID: " + str(clip_id) + "\nPrevious summaries:\n"
        if summary:
            prompt += str(summary)
        for key, value in summary_dict.items():
            prompt += "\n" + f"Clip {key}: {value}"

        frames_dir = Path(f"data/frames/{video_name}") / str(clip_id)
        images = sorted(glob.glob(str(frames_dir / "*.jpg")), key=lambda x: int(Path(x).stem))

        try:
            messages = generate_messages(images, prompt)
            response, tokens = get_response(messages, VideoOutputFormat)
            answer_or_search = response.answer
            clip_summary = response.content
            result['token_summaries']['video_answer'] += int(tokens or 0)
        except Exception as e:
            raise Exception(f"Error processing clip {clip_id}: {e}")

        result['video_answer_outputs'].append(str(response))

        if answer_or_search:
            result['final_answer'] = clip_summary
            print("Final answer: \n", result['final_answer'])
            return result
        else:
            summary_dict[clip_id] = clip_summary

    # Watch last clip
    clip_id = clip_ids[-1]
    print(f"Processing last clip {clip_id}...")
    prompt = prompt_video_answer_final + "\nQuestion: " + question + "\nCurrent clip ID: " + str(clip_id) + "\nPrevious summaries:\n"
    if summary:
        prompt += str(summary)
    for key, value in summary_dict.items():
        prompt += "\n" + f"Clip {key}: {value}"
    frames_dir = Path(f"data/frames/{video_name}") / str(clip_id)
    images = sorted(glob.glob(str(frames_dir / "*.jpg")), key=lambda x: int(Path(x).stem))

    try:
        messages = generate_messages(images, prompt)
        response, tokens = get_response(messages)
        if response is None or (isinstance(response, str) and not response.strip()):
            response = (
                summary_dict[clip_ids[-2]] if len(clip_ids) >= 2 and clip_ids[-2] in summary_dict
                else "No answer could be generated from the video."
            )
        result['final_answer'] = response
    except Exception as e:
        raise Exception(f"Error processing last clip {clip_id}: {e}")

    print("Final Answer:")
    print(result['final_answer'])

    return result


def reason_no_video_rewatch(graph, video_name, question):

    result = {
        'question': question,
        'parse_query_output': None,
        'graph_search_results': None,
        'decision_response': None,
        'token_summaries': {"parse_query": 0, "graph_answer": 0},
        'final_answer': None,
    }

    print("================================================")
    print("Ablation study: Reasoning without video rewatch")

    #--------------------------------
    # Part 1: Search the graph
    #--------------------------------
    print("\n[Step 1] Searching the graph...")
    try:
        # Parse query using LLM
        parse_query_response, tokens = generate_text_response(prompt_parse_query + "\n" + question, ParseQueryOutput)
        result['parse_query_output'] = str(parse_query_response)
        result['token_summaries']['parse_query'] = tokens
        print("Parse Query Output:")
        print(parse_query_response)

        # Search the graph with parsed query
        graph_search_results = search_with_parse(question, graph, parse_query_response)
        result['graph_search_results'] = graph_search_results

    except Exception as e:
        raise Exception(f"Error searching graph: {e}")

    #--------------------------------
    # Part 2: Answer from searched graph only
    #--------------------------------
    print("\n[Step 2] Answering from searched graph only...")
    prompt = (
        prompt_no_video_rewatch
        + "\nExtracted knowledge from graph:\n"
        + result['graph_search_results']
        + "\nQuestion: "
        + question
    )
    try:
        response, tokens = generate_text_response(prompt)
        result['decision_response'] = str(response)
        result['token_summaries']['graph_answer'] = tokens
        result['final_answer'] = response
        print("Final answer:")
        print(result['final_answer'])
    except Exception as e:
        raise Exception(f"Error answering from searched graph: {e}")

    return result

