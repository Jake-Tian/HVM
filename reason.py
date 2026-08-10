"""Run the default five-tool HVM reasoning agent and evaluate its answers.

Alternative implementations live under ``reasoning_variants`` and are not part
of the default CLI path. ``reason_pipeline`` is re-exported below for backward
compatibility with the original fixed three-step implementation.
"""

import json
import pickle
import sys
import time
import traceback
from pathlib import Path

from tqdm import tqdm

from langchain_core.messages import HumanMessage, AIMessage

from utils.llm_gpt import generate_text_response
from utils.prompts import (
    prompt_agent_verify_answer_referencing,
)
from reasoning.agent import build_agent, DEFAULT_BUDGET
from reasoning_variants.fixed_pipeline_backup import reason_pipeline
from utils.api_errors import is_fatal_api_error
from utils.reasoning_trace import build_tool_rounds
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
        "clip_history": [],
        "tool_call_history": [],
        "budget": budget,
        "total_tokens": 0,
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
    token_summaries = {"total": int(total_tokens or 0)}

    # Save both the tool call and its observation for error analysis.
    rounds = build_tool_rounds(result_state["messages"])

    print("Final Answer:", final_ans)
    print(f"Total Tokens: {total_tokens}")

    return {
        "question": question,
        "rounds": rounds,
        "final_answer": final_ans,
        "token_summaries": token_summaries,
    }

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
        # Re-raise unrecoverable API failures instead of scoring them as wrong.
        if is_fatal_api_error(e):
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
    parser = argparse.ArgumentParser(description="HVM reasoning over a graph memory.")
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
        default="five_tools_luna_medium",
        help="Suffix for per-video reasoning logs.",
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

        # Keep detailed output in the per-video log unless verbose mode is enabled.
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

                # Save progress and report to the terminal before aborting.
                if is_fatal_api_error(e):
                    with open(output_json_path, "w", encoding="utf-8") as f:
                        json.dump(reasoning_results, f, indent=2, ensure_ascii=False)
                    pbar.close()
                    sys.stdout = real_stdout
                    log_file.close()
                    msg = (
                        f"\n✗ Fatal API error at {video_name}/{question_id}: {e}\n"
                        f"Partial results saved to {output_json_path}.\n"
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

            # Persist after each question to preserve partial progress.
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
