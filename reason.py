import pickle
import json
import sys
import time
import traceback
from pathlib import Path
from tqdm import tqdm

from langchain_core.messages import HumanMessage, AIMessage

from utils.llm_gpt import generate_text_response
from utils.prompts import prompt_agent_verify_answer_referencing
from reasoning.agent import build_agent
from utils.general import QuietStdout, Tee, find_pkl_files, verbose_terminal
from reasoning.trace import build_tool_rounds

def reason(graph, video_name, question):
    print("================================================")
    print("Question: ", question)

    app = build_agent(graph, video_name)

    initial_state = {
        "question": question,
        "messages": [HumanMessage(content=question)],
        "findings": [],
        "clip_history": [],
        "budget": 5,
        "total_tokens": 0
    }
    
    result_state = app.invoke(initial_state)
    
    final_ans = ""
    for msg in reversed(result_state["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            final_ans = msg.content
            break
            
    total_tokens = result_state.get("total_tokens", 0)
            
    rounds = build_tool_rounds(result_state["messages"])
                
    if not final_ans:
        final_ans = "Could not determine the answer."
                
    print("Final Answer:", final_ans)
    print(f"Total Tokens: {total_tokens}")
    
    result = {
        'question': question,
        'rounds': rounds,
        'final_answer': final_ans,
        'token_summaries': {"total": total_tokens},
        'evaluate_correct': False
    }
    return result

def evaluate_answer(question, ground_truth_answer, predicted_answer):
    prompt = prompt_agent_verify_answer_referencing.format(
        question=question,
        ground_truth_answer=ground_truth_answer,
        agent_answer=predicted_answer
    )
    
    try: 
        response, tokens = generate_text_response(prompt)
        response = response.strip().upper()
        if response.startswith("YES"):
            return True, (tokens or 0)
        elif response.startswith("NO"):
            return False, (tokens or 0)
        else:
            print(f"Warning: Unexpected evaluator response: {response}. Defaulting to False.")
            return False, (tokens or 0)
    except Exception as e:
        print(f"Error evaluating answer: {e}. Defaulting to False.")
        return False, 0


def load_incorrect_question_ids(result_path):
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

    parser = argparse.ArgumentParser(description="HVM-web reasoning over graph memory.")
    parser.add_argument("videos", nargs="*", help="Video names to process.")
    parser.add_argument("--graph-dir", default="data/graphs")
    parser.add_argument("--graph-suffix", default="", help="Suffix before .pkl, e.g. _preabstraction.")
    parser.add_argument("--out-dir", default="data/reasoning")
    parser.add_argument("--incorrect-only-from", default=None)
    parser.add_argument("--log-tag", default="five_tools_luna")
    args = parser.parse_args()

    original_stdout = sys.stdout
    Path("data/logs").mkdir(parents=True, exist_ok=True)

    available_videos = (
        args.videos
        if args.videos
        else sorted(find_pkl_files(
            graph_dir=args.graph_dir,
            graph_suffix=args.graph_suffix,
        ))
    )

    with open(f"data/web_100.json", "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    for video_name in available_videos:
        output_json_path = Path(args.out_dir) / f"{video_name}.json"
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path = Path(args.graph_dir) / f"{video_name}{args.graph_suffix}.pkl"
        if not graph_path.exists():
            print(f"Skipping {video_name}: graph not found at {graph_path}")
            continue
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)

        video_questions = questions_data.get(video_name, {}).get("qa_list", [])
        if args.incorrect_only_from:
            baseline_path = Path(args.incorrect_only_from) / f"{video_name}.json"
            if not baseline_path.exists():
                raise FileNotFoundError(f"Baseline reasoning result not found: {baseline_path}")
            incorrect_ids = load_incorrect_question_ids(baseline_path)
            video_questions = [
                question for question in video_questions
                if question.get("question_id") in incorrect_ids
            ]

        reasoning_results = {}

        log_tag = Path(args.log_tag).name
        log_path = Path("data/logs") / f"{video_name}_reason_{log_tag}.log"
        log_file = open(log_path, "w", encoding="utf-8")
        sys.stdout = Tee(log_file) if verbose_terminal() else QuietStdout(log_file)
        pbar = tqdm(total=len(video_questions), desc=f"reason {video_name}", file=sys.stderr)
        start_time = time.time()

        for video_question in video_questions:
            question_id = video_question.get("question_id")
            question = video_question.get("question", "")
            answer = video_question.get("answer", "")

            try:
                main_result = reason(graph, video_name, question)
                if isinstance(main_result, dict):
                    evaluate_correct, eval_tokens = evaluate_answer(question, answer, main_result.get("final_answer", ""))
                    main_result["evaluate_correct"] = evaluate_correct
                    if "token_summaries" not in main_result:
                        main_result["token_summaries"] = {"total": 0}
                    main_result["token_summaries"]["total"] += eval_tokens
                else:
                    # In case reason() returned a string error
                    main_result = {
                        "error": str(main_result),
                        "evaluate_correct": False,
                        "final_answer": "Error occurred during reasoning.",
                        "token_summaries": {"total": 0}
                    }
            except Exception as e:
                print(f"Error processing question {question_id}: {e}")
                traceback.print_exc()

                main_result = {
                    "error": str(e),
                    "evaluate_correct": False,
                    "final_answer": "Error occurred during reasoning.",
                    "token_summaries": {"total": 0}
                }

            reasoning_results[question_id] = {
                "question": question,
                "ground_truth_answer": answer,
                "reasoning": main_result,
                "timestamp": video_question.get("timestamp"),
                "type": video_question.get("type"),
            }

            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(reasoning_results, f, indent=2, ensure_ascii=False)

            pbar.set_postfix_str(f"q{question_id}")
            pbar.update(1)

        pbar.close()
        elapsed = time.time() - start_time
        sys.stdout = original_stdout
        log_file.close()
        print(
            f"✓ [{video_name}] reasoning {elapsed:.0f}s | "
            f"questions={len(reasoning_results)} | output={output_json_path}"
        )

    sys.stdout = original_stdout

if __name__ == "__main__":
    main()
