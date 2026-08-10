"""Original fixed three-step reasoning pipeline kept for compatibility."""

import glob
from pathlib import Path

from classes.output_structure import (
    GraphOutputFormat,
    ParseQueryOutput,
    VideoOutputFormat,
)
from utils.llm_gpt import generate_text_response
from utils.mllm_gpt import generate_messages, get_response
from utils.prompts import (
    prompt_graph_video,
    prompt_parse_query,
    prompt_video_answer,
    prompt_video_answer_final,
)
from utils.search import search_with_parse
from utils.token_usage import add_stage_usage, build_token_summary


def reason_pipeline(graph, video_name, question):
    """The original non-agent pipeline: parse -> graph search -> optional
    video clip watching. Kept verbatim for ablation comparisons.
    """
    stage_usage = {}
    result = {
        "question": question,
        "parse_query_output": None,
        "graph_search_results": None,
        "decision_response": None,
        "video_answer_outputs": [],
        "token_summaries": {},
        "final_answer": None,
    }

    print("================================================")
    print("Question: ", question)

    print("\n[Step 1] Searching the graph...")
    try:
        parse_query_response, tokens = generate_text_response(
            prompt_parse_query + "\n" + question,
            ParseQueryOutput,
        )
        result["parse_query_output"] = str(parse_query_response)
        add_stage_usage(stage_usage, "parse_query", tokens)
        print("Parse Query Output:")
        print(parse_query_response)

        graph_search_results = search_with_parse(
            question,
            graph,
            parse_query_response,
        )
        result["graph_search_results"] = graph_search_results
    except Exception as e:
        raise Exception(f"Error searching graph: {e}")

    print("\n[Step 2] Evaluating searched answer...")
    prompt = (
        prompt_graph_video
        + "\nExtracted knowledge from graph:\n"
        + result["graph_search_results"]
        + "\nQuestion: "
        + question
    )
    try:
        decision_response, tokens = generate_text_response(
            prompt,
            GraphOutputFormat,
        )
        result["decision_response"] = str(decision_response)
        add_stage_usage(stage_usage, "graph_answer", tokens)
        print("Decision response: \n", decision_response)
    except Exception as e:
        raise Exception(f"Error evaluating searched answer: {e}")

    answer_or_search = decision_response.answer
    content = decision_response.content
    summary = decision_response.summary

    if answer_or_search:
        result["final_answer"] = content
        result["token_summaries"] = build_token_summary(stage_usage)
        return result

    if not isinstance(content, list):
        content = [content]
    clip_ids = []
    for item in content:
        try:
            clip_ids.append(int(item))
        except Exception:
            print(f"Warning: Ignoring invalid clip id from decision output: {item}")
    if not clip_ids:
        raise Exception(
            f"No valid clip ids returned for video search. Raw content: {content}"
        )
    if len(clip_ids) > 5:
        clip_ids = clip_ids[:5]

    print(f"\n[Step 3] Watching video clips: {clip_ids}")
    summary_dict = dict()

    for clip_id in clip_ids[:-1]:
        print(f"Processing clip {clip_id}...")
        prompt = (
            prompt_video_answer
            + "\nQuestion: "
            + question
            + "\nCurrent clip ID: "
            + str(clip_id)
            + "\nPrevious summaries:\n"
        )
        if summary:
            prompt += str(summary)
        for key, value in summary_dict.items():
            prompt += "\n" + f"Clip {key}: {value}"

        frames_dir = Path(f"data/frames/{video_name}") / str(clip_id)
        images = sorted(
            glob.glob(str(frames_dir / "*.jpg")),
            key=lambda x: int(Path(x).stem),
        )

        try:
            messages = generate_messages(images, prompt)
            response, tokens = get_response(messages, VideoOutputFormat)
            answer_or_search = response.answer
            clip_summary = response.content
            add_stage_usage(stage_usage, "video_answer", tokens)
        except Exception as e:
            raise Exception(f"Error processing clip {clip_id}: {e}")

        result["video_answer_outputs"].append(str(response))
        if answer_or_search:
            result["final_answer"] = clip_summary
            result["token_summaries"] = build_token_summary(stage_usage)
            print("Final answer: \n", result["final_answer"])
            return result
        else:
            summary_dict[clip_id] = clip_summary

    clip_id = clip_ids[-1]
    print(f"Processing last clip {clip_id}...")
    prompt = (
        prompt_video_answer_final
        + "\nQuestion: "
        + question
        + "\nCurrent clip ID: "
        + str(clip_id)
        + "\nPrevious summaries:\n"
    )
    if summary:
        prompt += str(summary)
    for key, value in summary_dict.items():
        prompt += "\n" + f"Clip {key}: {value}"
    frames_dir = Path(f"data/frames/{video_name}") / str(clip_id)
    images = sorted(
        glob.glob(str(frames_dir / "*.jpg")),
        key=lambda x: int(Path(x).stem),
    )

    try:
        messages = generate_messages(images, prompt)
        response, tokens = get_response(messages)
        add_stage_usage(stage_usage, "video_answer", tokens)
        if response is None or (isinstance(response, str) and not response.strip()):
            response = (
                summary_dict[clip_ids[-2]]
                if len(clip_ids) >= 2 and clip_ids[-2] in summary_dict
                else "No answer could be generated from the video."
            )
        result["final_answer"] = response
    except Exception as e:
        raise Exception(f"Error processing last clip {clip_id}: {e}")

    print("Final Answer:")
    print(result["final_answer"])
    result["token_summaries"] = build_token_summary(stage_usage)
    return result
