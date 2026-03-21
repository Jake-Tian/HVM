import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from classes.output_structure import (
    AllocateSearchOutput,
    AnswerWithSearchResultsFinalOutput,
    AnswerWithSearchResultsOutput,
)
from utils.general import Tee
from utils.llm import generate_text_response
from utils.prompts import (
    prompt_allocate_search,
    prompt_answer_with_search_results,
    prompt_answer_with_search_results_final,
)
from utils.rag_helper import (
    evaluate_multiple_choice_answer,
    format_options,
    harmonize_allocation_with_total,
    organize_results_temporal,
    safe_allocation,
)
from utils.search import search_behavior, search_conversation


def reason(
    behavior_dir: str | Path,
    conversation_dir: str | Path,
    question: str,
    multiple_choice_options: dict[str, str],
    max_rounds: int = 5,
):
    result = {
        "question": question,
        "options": multiple_choice_options,
        "rounds": [],
        "initial_allocation": None,
        "token_summaries": {
            "allocate_search": 0,
            "answer_round": 0,
            "answer_final": 0,
            "total": 0,
        },
        "final_answer_option": None,
        "final_answer_text": None,
        "final_summary": None,
    }

    print("================================================")
    print("Question:", question)

    options_text = format_options(multiple_choice_options)

    # Round-1 allocation
    allocate_prompt = (
        prompt_allocate_search
        + "\nQuestion: "
        + question
        + "\nOptions:\n"
        + options_text
    )
    try:
        allocation, tokens = generate_text_response(allocate_prompt, AllocateSearchOutput)
    except Exception as e:
        raise Exception(f"Error in allocation LLM call: {e}")
    print(allocation)
    result["token_summaries"]["allocate_search"] = int(tokens or 0)
    result["initial_allocation"] = str(allocation)
    k_behavior, k_conversation = safe_allocation(
        allocation.k_behavior, allocation.k_conversation
    )
    speaker_strict = allocation.speaker_strict

    current_question = question
    accumulated_evidence: list[str] = []
    accumulated_summaries: list[str] = []

    for round_id in range(1, max_rounds + 1):
        print(f"\n[Round {round_id}]")
        print(f"Searching with k_behavior={k_behavior}, k_conversation={k_conversation}, speaker_strict={speaker_strict}")

        behavior_hits = search_behavior(current_question, k_behavior, behavior_dir)
        conversation_hits = search_conversation(
            current_question,
            speaker_strict,
            conversation_dir,
            k=k_conversation,
        )
        organized = organize_results_temporal(behavior_hits, conversation_hits)
        print("organized:\n", organized)
        accumulated_evidence.append(f"Round {round_id} evidence:\n{organized}")

        round_payload = {
            "round_id": round_id,
            "search_question": current_question,
            "k_behavior": k_behavior,
            "k_conversation": k_conversation,
            "speaker_strict": speaker_strict,
            "behavior_hits_count": len(behavior_hits),
            "conversation_hits_count": len(conversation_hits),
            "organized_results": organized,
            "decision_response": None,
        }

        # Final round: force final answer generation
        if round_id == max_rounds:
            previous_summaries_text = (
                "\n".join(
                    f"Round {i + 1} summary: {s}"
                    for i, s in enumerate(accumulated_summaries)
                )
                if accumulated_summaries
                else "(none)"
            )
            final_prompt = (
                prompt_answer_with_search_results_final
                + "\nQuestion: "
                + question
                + "\nOptions:\n"
                + options_text
                + "\nPrevious round summaries:\n"
                + previous_summaries_text
                + "\nAccumulated evidence:\n"
                + "\n\n".join(accumulated_evidence)
            )
            try:
                final_response, tokens = generate_text_response(
                    final_prompt, AnswerWithSearchResultsFinalOutput
                )
            except Exception as e:
                raise Exception(f"Error in final-round LLM call (round {round_id}): {e}")
            result["token_summaries"]["answer_final"] += int(tokens or 0)
            round_payload["decision_response"] = str(final_response)
            result["final_answer_option"] = final_response.content
            result["final_answer_text"] = multiple_choice_options.get(final_response.content)
            result["final_summary"] = final_response.summary
            result["rounds"].append(round_payload)
            break

        # Non-final rounds: decide answer or keep searching
        previous_summaries_text = (
            "\n".join(
                f"Round {i + 1} summary: {s}"
                for i, s in enumerate(accumulated_summaries)
            )
            if accumulated_summaries
            else "(none)"
        )
        round_prompt = (
            prompt_answer_with_search_results
            + "\nQuestion: "
            + question
            + "\nOptions:\n"
            + options_text
            + "\nCurrent round search question: "
            + current_question
            + "\nPrevious round summaries:\n"
            + previous_summaries_text
            + "\nRetrieved evidence:\n"
            + organized
        )
        print("round_prompt:\n", round_prompt)
        try:
            decision, tokens = generate_text_response(
                round_prompt, AnswerWithSearchResultsOutput
            )
        except Exception as e:
            raise Exception(f"Error in iterative LLM call (round {round_id}): {e}")
        print("decision:\n", decision)
        result["token_summaries"]["answer_round"] += int(tokens or 0)
        round_payload["decision_response"] = str(decision)
        result["rounds"].append(round_payload)

        if decision.answer:
            result["final_answer_option"] = decision.content
            result["final_answer_text"] = multiple_choice_options.get(decision.content)
            result["final_summary"] = decision.summary
            break

        if decision.summary:
            accumulated_summaries.append(decision.summary)
        current_question = decision.content
        k_behavior, k_conversation = harmonize_allocation_with_total(
            decision.k_behavior,
            decision.k_conversation,
            decision.total_search_k,
        )
        speaker_strict = decision.speaker_strict

    result["token_summaries"]["total"] = (
        int(result["token_summaries"]["allocate_search"])
        + int(result["token_summaries"]["answer_round"])
        + int(result["token_summaries"]["answer_final"])
    )
    return result


def process_one_day(
    day: str,
    qa_root: Path,
    behavior_root: Path,
    conversation_root: Path,
    output_root: Path,
) -> tuple[str, dict]:
    qa_path = qa_root / f"EgoLifeQA_A1_JAKE_{day}.json"
    behavior_path = behavior_root / f"{day}.json"
    conversation_path = conversation_root / f"{day}.json"

    if not qa_path.exists():
        return day, {"error": f"QA file missing at {qa_path}"}
    if not behavior_path.exists():
        return day, {"error": f"behavior file missing at {behavior_path}"}
    if not conversation_path.exists():
        return day, {"error": f"conversation file missing at {conversation_path}"}

    with qa_path.open("r", encoding="utf-8") as f:
        qa_data = json.load(f)

    day_results = {}
    day_token_summaries = {
        "baseline": {"allocate_search": 0, "answer_round": 0, "answer_final": 0, "total": 0},
        "ablation_max_rounds_3": {"allocate_search": 0, "answer_round": 0, "answer_final": 0, "total": 0},
    }
    for item in qa_data:
        qid = str(item.get("ID"))
        question = item.get("question", "")
        options = {
            "A": item.get("choice_a", ""),
            "B": item.get("choice_b", ""),
            "C": item.get("choice_c", ""),
            "D": item.get("choice_d", ""),
        }
        try:
            # Baseline (comparison): full iterative RAG with max_rounds=5
            baseline_result = reason(
                behavior_path,
                conversation_path,
                question,
                options,
                max_rounds=5,
            )
            baseline_result["ground_truth_option"] = item.get("answer")
            baseline_result["evaluate_correct"] = evaluate_multiple_choice_answer(
                predicted_option=baseline_result.get("final_answer_option"),
                ground_truth_option=item.get("answer"),
                options=options,
                predicted_text=baseline_result.get("final_answer_text"),
            )
            for key in day_token_summaries["baseline"]:
                day_token_summaries["baseline"][key] += int(
                    baseline_result.get("token_summaries", {}).get(key, 0) or 0
                )

            # Ablation: fewer retrieval rounds
            ablation_round3 = reason(
                behavior_path,
                conversation_path,
                question,
                options,
                max_rounds=3,
            )
            ablation_round3["ground_truth_option"] = item.get("answer")
            ablation_round3["evaluate_correct"] = evaluate_multiple_choice_answer(
                predicted_option=ablation_round3.get("final_answer_option"),
                ground_truth_option=item.get("answer"),
                options=options,
                predicted_text=ablation_round3.get("final_answer_text"),
            )
            for key in day_token_summaries["ablation_max_rounds_3"]:
                day_token_summaries["ablation_max_rounds_3"][key] += int(
                    ablation_round3.get("token_summaries", {}).get(key, 0) or 0
                )

            day_results[qid] = {
                "question": question,
                "ground_truth_option": item.get("answer"),
                "reasoning": baseline_result,  # baseline comparison
                "ablations": {
                    "max_rounds_3": ablation_round3,
                },
            }
        except Exception as e:
            day_results[qid] = {
                "error": str(e),
                "question": question,
                "options": options,
            }

    output_root.mkdir(parents=True, exist_ok=True)
    day_output_path = output_root / f"rag_EgoLifeQA_A1_JAKE_{day}.json"
    payload = {
        "day": day,
        "day_token_summaries": day_token_summaries,
        "questions": day_results,
    }
    with day_output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return day, {
        "output_path": str(day_output_path),
        "day_token_summaries": day_token_summaries,
        "num_questions": len(day_results),
    }


def main():
    original_stdout = sys.stdout
    log_file = open("log.txt", "w", encoding="utf-8")
    sys.stdout = Tee(log_file)

    qa_root = Path("data/questions")
    behavior_root = Path("data/behaviors")
    conversation_root = Path("data/conversations")
    output_root = Path("data/results")
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "rag_EgoLifeQA_A1_JAKE_by_day_summary.json"

    days = [f"DAY{i}" for i in range(1, 8)]

    # Modify this to set the number of parallel jobs
    max_parallel_jobs = 4
    print(f"Processing {len(days)} days with max {max_parallel_jobs} parallel jobs.")

    results_by_day = {}
    with ThreadPoolExecutor(max_workers=max_parallel_jobs) as executor:
        futures = {
            executor.submit(
                process_one_day,
                day,
                qa_root,
                behavior_root,
                conversation_root,
                output_root,
            ): day
            for day in days
        }
        for fut in as_completed(futures):
            day = futures[fut]
            print("\n================================================")
            print(f"Completed {day}")
            try:
                returned_day, day_results = fut.result()
                results_by_day[returned_day] = day_results
            except Exception as e:
                print(f"Error processing {day}: {e}")
                traceback.print_exc()
                results_by_day[day] = {"error": str(e)}

            # Save incrementally in case long runs are interrupted.
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(results_by_day, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved per-day RAG files under {output_root}")
    print(f"✓ Saved day summary to {summary_path}")
    sys.stdout = original_stdout
    log_file.close()


if __name__ == "__main__":
    main()
