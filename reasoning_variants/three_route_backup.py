"""CAM reasoning entry point.

The default `reason()` now uses a LangGraph planner/executor/verifier/final_answer
agent (reasoning_variants/three_route/agent.py) with graph search,
graph temporal context,
action-frequency memory, and location-only video rewatch. The original fixed
3-step pipeline is preserved as `reason_pipeline()` for ablation/fallback.

Main loop reads data/robot.json, runs the agent per question, and evaluates the
free-text answer with an LLM judge (YES/NO).
"""

import pickle
import json
import glob
import sys
import time
import traceback
from pathlib import Path
from tqdm import tqdm

from langchain_core.messages import HumanMessage, AIMessage

from utils.llm_gpt import generate_text_response
from utils.mllm_gpt import generate_messages, get_response


def _is_fatal_api_error(exc):
    """Detect API errors that won't recover by retrying (auth / billing / quota).

    Returns True for 401/403 and balance/quota/billing exhaustion messages.
    Rate limits (429) and transient network errors return False — a retry may
    still succeed, so we should NOT abort the whole run on those.
    """
    msg = str(exc).lower()
    # balance / billing / quota exhaustion (proxy returns Chinese or English text)
    if any(k in msg for k in (
        "余额", "balance", "insufficient", "billing",
        "exceeded your current quota", "quota_exceeded", "quota",
    )):
        return True
    # HTTP status embedded in the message, e.g. "Error code: 403 - {...}"
    if "error code: 401" in msg or "error code: 403" in msg:
        return True
    # typed openai exceptions
    try:
        import openai
        if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
            return True
    except Exception:
        pass
    return False

from utils.prompts import (
    prompt_parse_query,
    prompt_graph_video,
    prompt_video_answer,
    prompt_video_answer_final,
    prompt_agent_verify_answer_referencing,
)
from classes.output_structure import (
    ParseQueryOutput,
    GraphOutputFormat,
    VideoOutputFormat,
)
from utils.search import search_with_parse
from reasoning_variants.three_route.agent import build_agent, DEFAULT_BUDGET
from utils.reasoning_trace import build_tool_rounds
from utils.token_usage import add_stage_usage, build_token_summary
from utils.general import find_pkl_files, Tee, QuietStdout, verbose_terminal


# ---------------------------------------------------------------------------
# New agent-based reasoning
# ---------------------------------------------------------------------------

def reason(
    graph,
    video_name,
    question,
    budget=DEFAULT_BUDGET,
):
    """Run the LangGraph agent and return a result dict shaped like the old
    pipeline's output so the main loop and evaluation stay unchanged.
    """
    print("================================================")
    print("Question: ", question)

    app = build_agent(
        graph,
        video_name,
        budget=budget,
    )

    initial_state = {
        "question": question,
        "messages": [HumanMessage(content=question)],
        "findings": [],
        "clip_history": [],
        "tool_call_history": [],
        "location_candidate_clips": [],
        "location_object_name": "",
        "location_intent": "",
        "action_frequency_memory": {},
        "budget": budget,
        "total_tokens": 0,
        "token_details": {},
    }

    result_state = app.invoke(initial_state)

    # Pull the final free-text answer: the last AIMessage without tool calls.
    final_ans = ""
    for msg in reversed(result_state["messages"]):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            final_ans = msg.content
            break

    if not final_ans:
        final_ans = "Could not determine the answer."

    total_tokens = result_state.get("total_tokens", 0)
    token_summaries = build_token_summary(result_state.get("token_details", {}))

    # Save both the tool call and its observation for error analysis.
    rounds = build_tool_rounds(result_state["messages"])

    print("Final Answer:", final_ans)
    print(f"Total Tokens: {total_tokens}")

    return {
        "question": question,
        "rounds": rounds,
        "final_answer": final_ans,
        "action_frequency_memory": result_state.get(
            "action_frequency_memory", {}
        ),
        "token_summaries": token_summaries,
    }


# ---------------------------------------------------------------------------
# Original fixed 3-step pipeline (preserved for ablation / fallback)
# ---------------------------------------------------------------------------

def reason_pipeline(graph, video_name, question):
    """The original non-agent pipeline: parse -> graph search -> optional
    video clip watching. Kept verbatim for ablation comparisons.
    """

    stage_usage = {}
    result = {
        'question': question,
        'parse_query_output': None,
        'graph_search_results': None,
        'decision_response': None,
        'video_answer_outputs': [],
        'token_summaries': {},
        'final_answer': None,
    }

    print("================================================")
    print("Question: ", question)

    # --------------------------------
    # Part 1: Search the graph
    # --------------------------------
    print("\n[Step 1] Searching the graph...")
    try:
        parse_query_response, tokens = generate_text_response(prompt_parse_query + "\n" + question, ParseQueryOutput)
        result['parse_query_output'] = str(parse_query_response)
        add_stage_usage(stage_usage, "parse_query", tokens)
        print("Parse Query Output:")
        print(parse_query_response)

        graph_search_results = search_with_parse(question, graph, parse_query_response)
        result['graph_search_results'] = graph_search_results

    except Exception as e:
        raise Exception(f"Error searching graph: {e}")

    # --------------------------------
    # Part 2: Evaluate searched graph answer
    # --------------------------------
    print("\n[Step 2] Evaluating searched answer...")
    prompt = prompt_graph_video + "\nExtracted knowledge from graph:\n" + result['graph_search_results'] + "\nQuestion: " + question
    try:
        decision_response, tokens = generate_text_response(prompt, GraphOutputFormat)
        result['decision_response'] = str(decision_response)
        add_stage_usage(stage_usage, "graph_answer", tokens)
        print("Decision response: \n", decision_response)
    except Exception as e:
        raise Exception(f"Error evaluating searched answer: {e}")

    answer_or_search = decision_response.answer
    content = decision_response.content
    summary = decision_response.summary

    if answer_or_search:
        result['final_answer'] = content
        result["token_summaries"] = build_token_summary(stage_usage)
        return result

    # --------------------------------
    # Part 3: Watch the video clips
    # --------------------------------
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
            add_stage_usage(stage_usage, "video_answer", tokens)
        except Exception as e:
            raise Exception(f"Error processing clip {clip_id}: {e}")

        result['video_answer_outputs'].append(str(response))

        if answer_or_search:
            result['final_answer'] = clip_summary
            result["token_summaries"] = build_token_summary(stage_usage)
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
        add_stage_usage(stage_usage, "video_answer", tokens)
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
    result["token_summaries"] = build_token_summary(stage_usage)

    return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_answer(question, ground_truth_answer, predicted_answer):
    prompt = prompt_agent_verify_answer_referencing.format(
        question=question,
        ground_truth_answer=ground_truth_answer,
        agent_answer=predicted_answer
    )

    try:
        response, _ = generate_text_response(prompt)
        response = response.strip().upper()
        if response.startswith("YES"):
            return True
        elif response.startswith("NO"):
            return False
        else:
            print(f"Warning: Unexpected evaluator response: {response}. Defaulting to False.")
            return False
    except Exception as e:
        # Fatal API errors (auth/billing/quota) must abort the run, not be
        # silently swallowed as "incorrect" — otherwise a dead account would
        # mark every answer wrong and waste the whole run. Re-raise so the
        # main loop's fatal-error handler can abort cleanly.
        if _is_fatal_api_error(e):
            raise
        print(f"Error evaluating answer: {e}. Defaulting to False.")
        return False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def load_incorrect_question_ids(result_path):
    """Return question IDs explicitly marked incorrect in an earlier result."""
    result_path = Path(result_path)
    with open(result_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    return {
        question_id
        for question_id, record in results.items()
        if isinstance(record, dict)
        and isinstance(record.get("reasoning"), dict)
        and record["reasoning"].get("evaluate_correct") is False
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CAM reasoning over a graph memory.")
    parser.add_argument("videos", nargs="*", help="Video names to process. If omitted, process all *.pkl in --graph-dir.")
    parser.add_argument("--graph-dir", default="data/graphs",
                        help="Directory to read <video>.pkl from (default: data/graphs). "
                             "Use a variant dir for ablation runs, e.g. data/ablation/graphs_abs/30_10.")
    parser.add_argument("--out-dir", default="data/reasoning",
                        help="Directory to write <video>.json reasoning results to "
                             "(default: data/reasoning). Use a variant dir for ablation runs.")
    parser.add_argument(
        "--incorrect-only-from",
        default=None,
        help="Only run questions marked evaluate_correct=false in matching "
             "<video>.json files from this earlier result directory.",
    )
    parser.add_argument(
        "--log-tag",
        default="gpt5mini",
        help="Suffix for per-video reasoning logs (default: gpt5mini).",
    )
    args = parser.parse_args()

    real_stdout = sys.stdout
    Path("data/logs").mkdir(parents=True, exist_ok=True)

    if args.videos:
        available_videos = args.videos
    else:
        available_videos = sorted(find_pkl_files(graph_dir=args.graph_dir))

    with open(f"data/robot.json", "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    for video_name in available_videos:
        output_json_path = Path(args.out_dir) / f"{video_name}.json"
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path = Path(args.graph_dir) / f"{video_name}.pkl"
        if not graph_path.exists():
            print(f"Skipping {video_name}: graph not found at {graph_path}")
            continue
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)

        video_questions = questions_data.get(video_name, {}).get("qa_list", [])
        if args.incorrect_only_from:
            baseline_path = Path(args.incorrect_only_from) / f"{video_name}.json"
            if not baseline_path.exists():
                raise FileNotFoundError(
                    f"Baseline reasoning result not found: {baseline_path}"
                )
            incorrect_ids = load_incorrect_question_ids(baseline_path)
            available_ids = {
                question.get("question_id") for question in video_questions
            }
            missing_ids = incorrect_ids - available_ids
            if missing_ids:
                print(
                    f"Warning: {video_name} has {len(missing_ids)} incorrect "
                    "question IDs not present in data/robot.json: "
                    f"{sorted(missing_ids)}"
                )
            video_questions = [
                question for question in video_questions
                if question.get("question_id") in incorrect_ids
            ]
            print(
                f"Selected {len(video_questions)} incorrect questions for "
                f"{video_name} from {baseline_path}"
            )
        reasoning_results = {}

        # Per-video quiet log: all planner/verifier/per-question prints go to
        # the log file; terminal shows only the tqdm bar + one summary line.
        log_tag = Path(args.log_tag).name
        log_path = Path(f"data/logs/{video_name}_reason_{log_tag}.log")
        log_file = open(log_path, "w", encoding="utf-8")
        sys.stdout = Tee(log_file) if verbose_terminal() else QuietStdout(log_file)

        pbar = tqdm(total=len(video_questions), desc=f"reason {video_name}", file=sys.stderr)
        correct = 0
        total_tokens = 0
        start_time = time.time()
        for video_question in video_questions:
            question_id = video_question.get("question_id")
            question = video_question.get("question", "")
            answer = video_question.get("answer", "")

            try:
                main_result = reason(
                    graph,
                    video_name,
                    question,
                )
                evaluate_correct = evaluate_answer(question, answer, main_result["final_answer"])
                main_result["evaluate_correct"] = evaluate_correct
                if evaluate_correct:
                    correct += 1
                total_tokens += int(main_result.get("token_summaries", {}).get("total", 0) or 0)
                print("Evaluate correct: ", evaluate_correct)
            except Exception as e:
                print(f"Error processing question {question_id}: {e}")
                traceback.print_exc()
                main_result = str(e)

                # Fatal API errors (auth / billing / quota) won't recover by
                # retrying. Save progress, then abort the whole run so we don't
                # burn time hammering a dead account on every remaining question
                # and video. Print to the REAL terminal (stdout was redirected
                # to the quiet log file, so the user would not see this otherwise).
                if _is_fatal_api_error(e):
                    with open(output_json_path, "w", encoding="utf-8") as f:
                        json.dump(reasoning_results, f, indent=2, ensure_ascii=False)
                    pbar.close()
                    sys.stdout = real_stdout
                    log_file.close()
                    msg = (
                        f"\n✗✗✗ FATAL API ERROR — aborting run @ "
                        f"{video_name}/{question_id} ✗✗✗\n"
                        f"  {e}\n\n"
                        f"  This looks like an auth/billing/quota error that won't\n"
                        f"  recover by retrying. Fix the API credentials/balance and\n"
                        f"  rerun. Partial results for this video were saved to\n"
                        f"  {output_json_path}.\n"
                    )
                    print(msg)
                    sys.stderr.write(msg)
                    sys.stderr.flush()
                    sys.exit(1)

            reasoning_results[question_id] = {
                "question": question,
                "ground_truth_answer": answer,
                "reasoning": main_result,
                "timestamp": video_question.get("timestamp"),
                "type": video_question.get("type"),
            }

            # Save iteratively so progress is not lost on failure.
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(reasoning_results, f, indent=2, ensure_ascii=False)

            pbar.set_postfix_str(f"q{question_id}")
            pbar.update(1)
        pbar.close()
        elapsed = time.time() - start_time

        sys.stdout = real_stdout
        log_file.close()
        print(
            f"✓ [{video_name}] reason {elapsed:.0f}s | "
            f"correct={correct}/{len(video_questions)} | tokens={total_tokens}"
        )


if __name__ == "__main__":
    main()
