import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel

from classes.output_structure import AllocateSearchOutput, AnswerWithSearchResultsFinalOutput
from utils.general import Tee
from utils.llm import generate_text_response
from utils.prompts import prompt_allocate_search, prompt_answer_with_search_results_final
from utils.rag_helper import (
    evaluate_multiple_choice_answer,
    format_options,
    harmonize_allocation_with_total,
    normalize_option_label,
    organize_results_for_llm,
    safe_allocation,
)
from utils.search import (
    evidence_linker,
    general_search,
    search_after,
    search_before,
    search_behavior,
    search_conversation,
    search_first,
    search_last,
)


class ToolRoundDecisionOutput(BaseModel):
    answer: bool
    content: str
    summary: str | None
    total_search_k: int | None
    k_behavior: int | None
    k_conversation: int | None
    speaker_strict: list[str] | None


def _build_tool_schemas() -> list[dict]:
    common_allocation = {
        "total_search_k": {"type": "integer", "minimum": 1, "maximum": 50},
        "k_behavior": {"type": "integer", "minimum": 0, "maximum": 50},
        "k_conversation": {"type": "integer", "minimum": 0, "maximum": 50},
    }
    common_props = {"search_content": {"type": "string"}, **common_allocation}
    return [
        {
            "type": "function",
            "function": {
                "name": "general_search",
                "description": "Default semantic retrieval over behavior and conversation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        **common_props,
                        "speaker_strict": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["search_content", "total_search_k", "k_behavior", "k_conversation"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "evidence_linker",
                "description": "Link scattered clues across behavior and conversation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        **common_props,
                        "speaker_strict": {"type": "array", "items": {"type": "string"}},
                        "target": {"type": "string"},
                    },
                    "required": ["search_content", "total_search_k", "k_behavior", "k_conversation"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_before",
                "description": "Search only before a concrete target evidence line.",
                "parameters": {
                    "type": "object",
                    "properties": {**common_props, "target": {"type": "string"}},
                    "required": [
                        "search_content",
                        "target",
                        "total_search_k",
                        "k_behavior",
                        "k_conversation",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_after",
                "description": "Search only after a concrete target evidence line.",
                "parameters": {
                    "type": "object",
                    "properties": {**common_props, "target": {"type": "string"}},
                    "required": [
                        "search_content",
                        "target",
                        "total_search_k",
                        "k_behavior",
                        "k_conversation",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_first",
                "description": "Find earliest matching evidence items.",
                "parameters": {
                    "type": "object",
                    "properties": common_props,
                    "required": ["search_content", "total_search_k", "k_behavior", "k_conversation"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_last",
                "description": "Find latest matching evidence items.",
                "parameters": {
                    "type": "object",
                    "properties": common_props,
                    "required": ["search_content", "total_search_k", "k_behavior", "k_conversation"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def _tool_selection_prompt(
    question: str,
    options_text: str,
    current_question: str,
    previous_summaries_text: str,
    last_evidence_text: str,
) -> str:
    return (
        "You are selecting exactly one retrieval tool for iterative QA.\n"
        "Use function calling only. Do not answer directly.\n\n"
        "Rules:\n"
        "- Choose ONE tool call now.\n"
        "- For location-before queries, prefer search_before with a concrete target line when possible.\n"
        "- For multi-hop scattered clues, prefer evidence_linker.\n"
        "- Avoid repeating broad general_search when previous evidence was not decisive.\n"
        "- search_content should include lexical aliases when useful (e.g., marker/pen/pencil/chalk).\n"
        "- total_search_k in [1,50], and k_behavior + k_conversation == total_search_k.\n\n"
        f"Question: {question}\n"
        f"Options:\n{options_text}\n"
        f"Current search question: {current_question}\n"
        f"Previous summaries:\n{previous_summaries_text}\n"
        f"Most recent retrieved evidence:\n{last_evidence_text}\n"
    )


def _decision_prompt(
    question: str,
    options_text: str,
    current_question: str,
    previous_summaries_text: str,
    retrieved_evidence: str,
) -> str:
    return (
        "You are deciding whether current evidence is sufficient for multiple-choice QA.\n\n"
        "Output JSON fields:\n"
        "answer: bool\n"
        "content: if answer=true, one of A/B/C/D; else improved next search question\n"
        "summary: null if answer=true, else 2-4 sentence relevant summary\n"
        "total_search_k: int or null\n"
        "k_behavior: int or null\n"
        "k_conversation: int or null\n"
        "speaker_strict: list[str] or null\n\n"
        "Rules:\n"
        "- If answer=true: content must be A/B/C/D; summary and all allocation fields must be null.\n"
        "- If answer=false: provide focused next query and valid allocation where "
        "k_behavior + k_conversation == total_search_k and 1<=total_search_k<=50.\n"
        "- Summarize only evidence relevant to distinguishing options.\n"
        "- Answer as soon as one option is clearly best supported.\n\n"
        f"Question: {question}\n"
        f"Options:\n{options_text}\n"
        f"Current round search question: {current_question}\n"
        f"Previous round summaries:\n{previous_summaries_text}\n"
        f"Retrieved evidence:\n{retrieved_evidence}\n"
    )


def _choose_tool_call(
    client: OpenAI,
    question: str,
    options_text: str,
    current_question: str,
    previous_summaries_text: str,
    last_evidence_text: str,
) -> tuple[str, dict, int]:
    tools = _build_tool_schemas()
    prompt = _tool_selection_prompt(
        question=question,
        options_text=options_text,
        current_question=current_question,
        previous_summaries_text=previous_summaries_text,
        last_evidence_text=last_evidence_text,
    )
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "Select one retrieval tool call."},
            {"role": "user", "content": prompt},
        ],
        tools=tools,
        tool_choice="required",
    )
    msg = response.choices[0].message
    tool_calls = msg.tool_calls or []
    if not tool_calls:
        raise ValueError("Tool selection returned no tool_calls.")
    call = tool_calls[0]
    args = json.loads(call.function.arguments or "{}")
    tokens = int((response.usage.total_tokens if response.usage else 0) or 0)
    return call.function.name, args, tokens


def _normalize_tool_arguments(
    args: dict,
    fallback_query: str,
) -> tuple[str, int, int, list[str] | None, str | None]:
    search_content = args.get("search_content")
    if not isinstance(search_content, str) or not search_content.strip():
        search_content = fallback_query
    k_behavior, k_conversation = harmonize_allocation_with_total(
        args.get("k_behavior"),
        args.get("k_conversation"),
        args.get("total_search_k"),
    )
    speaker_strict = args.get("speaker_strict")
    if not isinstance(speaker_strict, list):
        speaker_strict = None
    else:
        speaker_strict = [str(x) for x in speaker_strict if str(x).strip()]
        if not speaker_strict:
            speaker_strict = None
    target = args.get("target")
    if not isinstance(target, str) or not target.strip():
        target = None
    return search_content, k_behavior, k_conversation, speaker_strict, target


def _run_selected_tool(
    tool_name: str,
    tool_args: dict,
    fallback_query: str,
    behavior_dir: str | Path,
    conversation_dir: str | Path,
) -> tuple[list[list[str]], list[list[str]], str, str | None, str, int, int, list[str] | None]:
    search_content, kb, kc, speaker_strict, target = _normalize_tool_arguments(
        tool_args, fallback_query
    )
    allocation = {"k_behavior": kb, "k_conversation": kc, "total_search_k": kb + kc}
    try:
        if tool_name == "general_search":
            out = general_search(
                search_content=search_content,
                allocation=allocation,
                behavior_dir=behavior_dir,
                conversation_dir=conversation_dir,
                speaker_strict=speaker_strict,
            )
            return (
                out.get("behavior") or [],
                out.get("conversation") or [],
                "general_search",
                None,
                search_content,
                kb,
                kc,
                speaker_strict,
            )
        if tool_name == "evidence_linker":
            out = evidence_linker(
                search_content=search_content,
                allocation=allocation,
                behavior_dir=behavior_dir,
                conversation_dir=conversation_dir,
                speaker_strict=speaker_strict,
                target=target,
            )
            return (
                out.get("behavior") or [],
                out.get("conversation") or [],
                "evidence_linker",
                target,
                search_content,
                kb,
                kc,
                speaker_strict,
            )
        if tool_name == "search_before":
            if not target:
                raise ValueError("search_before requires target.")
            out = search_before(
                search_content=search_content,
                target=target,
                allocation=allocation,
                behavior_dir=behavior_dir,
                conversation_dir=conversation_dir,
            )
            if out.get("source") == "behavior":
                return (
                    out.get("results") or [],
                    [],
                    "search_before",
                    target,
                    search_content,
                    kb,
                    kc,
                    speaker_strict,
                )
            return (
                [],
                out.get("results") or [],
                "search_before",
                target,
                search_content,
                kb,
                kc,
                speaker_strict,
            )
        if tool_name == "search_after":
            if not target:
                raise ValueError("search_after requires target.")
            out = search_after(
                search_content=search_content,
                target=target,
                allocation=allocation,
                behavior_dir=behavior_dir,
                conversation_dir=conversation_dir,
            )
            if out.get("source") == "behavior":
                return (
                    out.get("results") or [],
                    [],
                    "search_after",
                    target,
                    search_content,
                    kb,
                    kc,
                    speaker_strict,
                )
            return (
                [],
                out.get("results") or [],
                "search_after",
                target,
                search_content,
                kb,
                kc,
                speaker_strict,
            )
        if tool_name == "search_first":
            out = search_first(
                search_content=search_content,
                allocation=allocation,
                behavior_dir=behavior_dir,
                conversation_dir=conversation_dir,
            )
            return (
                out.get("behavior") or [],
                out.get("conversation") or [],
                "search_first",
                None,
                search_content,
                kb,
                kc,
                speaker_strict,
            )
        if tool_name == "search_last":
            out = search_last(
                search_content=search_content,
                allocation=allocation,
                behavior_dir=behavior_dir,
                conversation_dir=conversation_dir,
            )
            return (
                out.get("behavior") or [],
                out.get("conversation") or [],
                "search_last",
                None,
                search_content,
                kb,
                kc,
                speaker_strict,
            )
        raise ValueError(f"Unknown tool: {tool_name}")
    except Exception as e:
        # Safe fallback
        out = general_search(
            search_content=search_content,
            allocation=allocation,
            behavior_dir=behavior_dir,
            conversation_dir=conversation_dir,
            speaker_strict=speaker_strict,
        )
        print(f"[tool fallback] {tool_name} failed: {e}. Falling back to general_search.")
        return (
            out.get("behavior") or [],
            out.get("conversation") or [],
            "general_search",
            None,
            search_content,
            kb,
            kc,
            speaker_strict,
        )


def reason(
    behavior_dir: str | Path,
    conversation_dir: str | Path,
    question: str,
    multiple_choice_options: dict[str, str],
    max_rounds: int = 5,
):
    client = OpenAI()
    result = {
        "question": question,
        "options": multiple_choice_options,
        "rounds": [],
        "initial_allocation": None,
        "token_summaries": {"allocate_search": 0, "answer_round": 0, "answer_final": 0, "total": 0},
        "final_answer_option": None,
        "final_answer_text": None,
        "final_summary": None,
    }

    print("================================================")
    print("Question:", question)
    options_text = format_options(multiple_choice_options)

    allocate_prompt = (
        prompt_allocate_search + "\nQuestion: " + question + "\nOptions:\n" + options_text
    )
    allocation, tokens = generate_text_response(allocate_prompt, AllocateSearchOutput)
    result["token_summaries"]["allocate_search"] = int(tokens or 0)
    result["initial_allocation"] = str(allocation)
    k_behavior, k_conversation = safe_allocation(allocation.k_behavior, allocation.k_conversation)
    speaker_strict = allocation.speaker_strict

    current_question = question
    accumulated_evidence: list[str] = []
    accumulated_summaries: list[str] = []

    for round_id in range(1, max_rounds + 1):
        print(f"\n[Round {round_id}]")

        previous_summaries_text = (
            "\n".join(f"Round {i + 1} summary: {s}" for i, s in enumerate(accumulated_summaries))
            if accumulated_summaries
            else "(none)"
        )
        last_evidence_text = accumulated_evidence[-1] if accumulated_evidence else "(none)"

        selected_tool_name = "default"
        selected_tool_args = {}
        retrieval_target: str | None = None

        if round_id == 1 or round_id == max_rounds:
            behavior_hits = search_behavior(current_question, k_behavior, behavior_dir)
            conversation_hits = search_conversation(
                current_question, speaker_strict, conversation_dir, k=k_conversation
            )
        else:
            tool_name, tool_args, selector_tokens = _choose_tool_call(
                client=client,
                question=question,
                options_text=options_text,
                current_question=current_question,
                previous_summaries_text=previous_summaries_text,
                last_evidence_text=last_evidence_text,
            )
            result["token_summaries"]["answer_round"] += int(selector_tokens or 0)
            selected_tool_name = tool_name
            selected_tool_args = tool_args
            (
                behavior_hits,
                conversation_hits,
                selected_tool_name,
                retrieval_target,
                current_question,
                k_behavior,
                k_conversation,
                speaker_strict,
            ) = _run_selected_tool(
                tool_name=tool_name,
                tool_args=tool_args,
                fallback_query=current_question,
                behavior_dir=behavior_dir,
                conversation_dir=conversation_dir,
            )

        organized = organize_results_for_llm(
            behavior_hits=behavior_hits,
            conversation_hits=conversation_hits,
            retrieval_method=selected_tool_name,
            target=retrieval_target,
        )
        accumulated_evidence.append(f"Round {round_id} evidence:\n{organized}")

        round_payload = {
            "round_id": round_id,
            "search_question": current_question,
            "k_behavior": k_behavior,
            "k_conversation": k_conversation,
            "speaker_strict": speaker_strict,
            "retrieval_method": selected_tool_name,
            "retrieval_target": retrieval_target,
            "selected_tool_args": selected_tool_args if selected_tool_name != "default" else None,
            "behavior_hits_count": len(behavior_hits),
            "conversation_hits_count": len(conversation_hits),
            "organized_results": organized,
            "decision_response": None,
        }

        if round_id == max_rounds:
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
            final_response, final_tokens = generate_text_response(
                final_prompt, AnswerWithSearchResultsFinalOutput
            )
            result["token_summaries"]["answer_final"] += int(final_tokens or 0)
            round_payload["decision_response"] = str(final_response)
            result["final_answer_option"] = final_response.content
            result["final_answer_text"] = multiple_choice_options.get(final_response.content)
            result["final_summary"] = final_response.summary
            result["rounds"].append(round_payload)
            break

        decision_prompt = _decision_prompt(
            question=question,
            options_text=options_text,
            current_question=current_question,
            previous_summaries_text=previous_summaries_text,
            retrieved_evidence=organized,
        )
        decision, decision_tokens = generate_text_response(decision_prompt, ToolRoundDecisionOutput)
        result["token_summaries"]["answer_round"] += int(decision_tokens or 0)
        round_payload["decision_response"] = str(decision)
        result["rounds"].append(round_payload)

        if decision.answer:
            label = normalize_option_label(decision.content)
            result["final_answer_option"] = label
            result["final_answer_text"] = multiple_choice_options.get(label) if label else None
            result["final_summary"] = decision.summary
            break

        if decision.summary:
            accumulated_summaries.append(decision.summary)
        current_question = decision.content
        k_behavior, k_conversation = harmonize_allocation_with_total(
            decision.k_behavior, decision.k_conversation, decision.total_search_k
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
        "reasoning": {"allocate_search": 0, "answer_round": 0, "answer_final": 0, "total": 0},
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
            reasoning_result = reason(
                behavior_path, conversation_path, question, options, max_rounds=5
            )
            reasoning_result["ground_truth_option"] = item.get("answer")
            reasoning_result["evaluate_correct"] = evaluate_multiple_choice_answer(
                predicted_option=reasoning_result.get("final_answer_option"),
                ground_truth_option=item.get("answer"),
                options=options,
                predicted_text=reasoning_result.get("final_answer_text"),
            )
            for key in day_token_summaries["reasoning"]:
                day_token_summaries["reasoning"][key] += int(
                    reasoning_result.get("token_summaries", {}).get(key, 0) or 0
                )
            day_results[qid] = {
                "question": question,
                "ground_truth_option": item.get("answer"),
                "reasoning": reasoning_result,
            }
        except Exception as e:
            day_results[qid] = {"error": str(e), "question": question, "options": options}

    output_root.mkdir(parents=True, exist_ok=True)
    day_output_path = output_root / f"{day}.json"
    payload = {"day": day, "day_token_summaries": day_token_summaries, "questions": day_results}
    with day_output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return day, {
        "output_path": str(day_output_path),
        "day_token_summaries": day_token_summaries,
        "num_questions": len(day_results),
    }


def main():
    original_stdout = sys.stdout
    log_file = open("log_tool.txt", "w", encoding="utf-8")
    sys.stdout = Tee(log_file)

    qa_root = Path("data/questions")
    behavior_root = Path("data/EgoLife/behaviors")
    conversation_root = Path("data/EgoLife/conversations")
    output_root = Path("results_tool")
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "by_day_summary.json"

    days = [f"DAY{i}" for i in range(1, 8)]
    max_parallel_jobs = 4
    print(f"Processing {len(days)} days with max {max_parallel_jobs} parallel jobs.")

    results_by_day = {}
    with ThreadPoolExecutor(max_workers=max_parallel_jobs) as executor:
        futures = {
            executor.submit(
                process_one_day, day, qa_root, behavior_root, conversation_root, output_root
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
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(results_by_day, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved per-day RAG files under {output_root}")
    print(f"✓ Saved day summary to {summary_path}")
    sys.stdout = original_stdout
    log_file.close()


if __name__ == "__main__":
    
    day = "DAY2"
    qa_root = Path("data/questions")
    behavior_root = Path("data/EgoLife/behaviors")
    conversation_root = Path("data/EgoLife/conversations")
    output_root = Path("results_tool")
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "by_day_summary.json"
    process_one_day(day, qa_root, behavior_root, conversation_root, output_root)