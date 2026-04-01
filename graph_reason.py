
import pickle
import json
import sys
import traceback
from pathlib import Path

from utils.search import set_graph, set_question
from utils.langchain_helper import build_agent
from utils.general import extract_choice_from_content, Tee
from utils.prompts import prompt_agent_verify_answer_referencing
from utils.llm import generate_text_response
from classes.hetero_graph import HeteroGraph


def graph_reason(query: str, choices: dict, ground_truth: str) -> dict:
    # Bind graph for utils.search @tool functions.
    set_question(query=query, choices=choices)
    agent = build_agent()

    result = {
        "question": query,
        "options": choices,
        "ground_truth": ground_truth,
        "final_answer_option": None,
        "answer_correct": False,
        "total_tokens": 0,
        "rounds": [],
    }

    messages = []
    messages = agent.invoke({"messages": messages})
    
    for m in messages["messages"]:

        if m.type == "tool":
            result["rounds"].append({
                "type": m.type,
                "content": m.content,
            })
        elif m.type == "ai":
            result["rounds"].append({
                "type": m.type,
                "content": m.content,
                "tool_calls": m.tool_calls,
            })
            result["total_tokens"] += m.usage_metadata.get("total_tokens", 0)

    final_answer_option = extract_choice_from_content(messages["messages"][-1].content)
    result["final_answer_option"] = final_answer_option

    if len(final_answer_option) == 1:
        result["answer_correct"] = final_answer_option == ground_truth
    else:
        result["answer_correct"] = evaluate_answer(query, ground_truth, final_answer_option)

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
            # If response is ambiguous, default to False
            print(f"Warning: Unexpected evaluator response: {response}. Defaulting to False.")
            return False
    except Exception as e:
        print(f"Error evaluating answer: {e}. Defaulting to False.")
        return False


def main():

    original_stdout = sys.stdout
    log_file = open("log.txt", "w", encoding="utf-8")
    sys.stdout = Tee(log_file)

    graph_path = Path("data/graphs/DAY1.pkl")
    qa_path = Path("data/questions/EgoLifeQA_A1_JAKE_DAY1.json")
    output_path = Path("data/results/DAY1.json")

    with open(graph_path, "rb") as f:
        graph = pickle.load(f)
    set_graph(graph)
    with qa_path.open("r", encoding="utf-8") as f:
        qa_data = json.load(f)

    day_results = {}
    
    for item in qa_data:
        qid = str(item.get("ID"))
        question = item.get("question", "")
        options = {
            "A": item.get("choice_a", ""),
            "B": item.get("choice_b", ""),
            "C": item.get("choice_c", ""),
            "D": item.get("choice_d", ""),
        }
        ground_truth = item.get("answer", "")
        print(f"\nProcessing question {qid}: {question}")

        try:
            result = graph_reason(question, options, ground_truth)
            day_results[qid] = result
            print("Answer_correct:", result["answer_correct"])
        except Exception as e:
            print(f"Error processing question {qid}: {e}")
            traceback.print_exc()
            day_results[qid] = {
                "question": question,
                "options": options,
                "ground_truth": ground_truth,
                "error": str(e),
                "answer_correct": False
            }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(day_results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Results saved to {output_path}")

    sys.stdout = original_stdout
    log_file.close()


if __name__ == "__main__":
    main()