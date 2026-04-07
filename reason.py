import pickle
import json
import sys
import traceback
from pathlib import Path

from utils.langchain_helper import build_agent
from utils.general import extract_choice_from_content, find_pkl_files, Tee
from utils.prompts import prompt_agent_verify_answer_referencing
from utils.llm import generate_text_response

from reason_ablation import reason_k30, reason_no_allocation


def reason(graph, video_name, question, options=None):
    print("================================================")
    print("Question: ", question)
    
    agent = build_agent(graph, question, options, video_name)

    result = {
        "question": question,
        "options": options,
        "final_answer": None,
        "total_tokens": 0,
        "rounds": [],
    }

    messages = []
    messages = agent.invoke({"messages": messages})
    
    for m in messages["messages"]:
        if m.type == "tool":
            result["rounds"].append({
                "type": m.type,
                "content": str(m.content),
            })
        elif m.type == "ai":
            result["rounds"].append({
                "type": m.type,
                "content": str(m.content),
                "tool_calls": m.tool_calls,
            })
            if hasattr(m, 'usage_metadata') and m.usage_metadata:
                result["total_tokens"] += m.usage_metadata.get("total_tokens", 0)

    final_answer_option = extract_choice_from_content(messages["messages"][-1].content)
    result["final_answer"] = final_answer_option
    
    print("Final Answer:")
    print(result['final_answer'])

    return result


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
        print(f"Error evaluating answer: {e}. Defaulting to False.")
        return False


def main():

    original_stdout = sys.stdout
    log_file = open("log.txt", "w", encoding="utf-8")
    sys.stdout = Tee(log_file)

    if len(sys.argv) < 2:
        available_videos = sorted(find_pkl_files())
    else:
        available_videos = sys.argv[1:]

    questions_data = {}
    with open("data/questions.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            vid = item.get("video_id")
            if vid not in questions_data:
                questions_data[vid] = {"qa_list": []}
            questions_data[vid]["qa_list"].append(item)

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

        for i, video_question in enumerate(video_questions, 1):
            question_id = video_question.get("question_id")
            if question_id is None:
                question_id = i
            q_text = video_question.get("question_text", "")
            options = video_question.get("options", {})
            opts_str = "\n".join([f"{k}: {v}" for k, v in options.items()])
            question = f"{q_text}\nOptions:\n{opts_str}\nPlease output ONLY a single letter (A, B, C, or D) corresponding to the correct option."
            answer = video_question.get("correct_answer", "")

            try:
                main_result = reason(graph, video_name, question, options)
                
                final_ans_str = str(main_result.get("final_answer", "")).strip().upper()
                extracted_letter = None
                for char in final_ans_str:
                    if char in ['A', 'B', 'C', 'D']:
                        extracted_letter = char
                        break
                
                if extracted_letter:
                    evaluate_correct = (extracted_letter == answer.strip().upper())
                else:
                    print(f"Warning: Could not extract valid option (A, B, C, D) from answer: {final_ans_str}")
                    evaluate_correct = False
                    
                print("Evaluate Correct: ", evaluate_correct)
                main_result["evaluate_correct"] = evaluate_correct
                main_result["ground_truth_answer"] = answer
                main_result["timestamp"] = video_question.get("timestamp")
                main_result["type"] = video_question.get("category")
                
                reasoning_results[question_id] = main_result
                
            except Exception as e:
                print(f"Error processing question {question_id}: {e}")
                traceback.print_exc()
                reasoning_results[question_id] = {
                    "question": question,
                    "ground_truth_answer": answer,
                    "error": str(e),
                    "timestamp": video_question.get("timestamp"),
                    "type": video_question.get("category")
                }

            # Save iteratively after each question
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(reasoning_results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved reasoning results for {video_name} to {output_json_path}")

    sys.stdout = original_stdout
    log_file.close()

if __name__ == "__main__":
    main()
