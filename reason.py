import pickle
import json
import glob
import sys
import traceback
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage

from utils.llm import generate_text_response
from utils.prompts import prompt_agent_verify_answer_referencing
from utils.langgraph_helper import build_agent
from utils.general import find_pkl_files, Tee

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
            
    rounds = []
    for msg in result_state["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                rounds.append({
                    "tool": tc["name"],
                    "args": tc["args"]
                })
                
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

def main():
    original_stdout = sys.stdout
    log_file = open("log.txt", "w", encoding="utf-8")
    sys.stdout = Tee(log_file)

    if len(sys.argv) < 2:
        available_videos = sorted(find_pkl_files())
    else:
        available_videos = sys.argv[1:]

    with open(f"data/web_100.json", "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    for video_name in available_videos:
        print("================================================")
        print(f"Processing video {video_name}...")

        output_json_path = Path(f"data/reasoning/{video_name}.json")
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path = Path(f"data/graphs/{video_name}.pkl")
        if not graph_path.exists():
            print(f"Skipping {video_name}: graph not found at {graph_path}")
            continue
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)

        video_questions = questions_data.get(video_name, {}).get("qa_list", [])
        reasoning_results = {}

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

        with open(output_json_path, "w") as f:
            json.dump(reasoning_results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved reasoning results for {video_name} to {output_json_path}")

    sys.stdout = original_stdout
    log_file.close()

if __name__ == "__main__":
    main()
