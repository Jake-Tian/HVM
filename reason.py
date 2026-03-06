from concurrent.futures.process import _ResultItem
import pickle
import json
import glob
import sys
import traceback
from pathlib import Path
from utils.llm import generate_text_response
from utils.mllm_gpt import generate_messages, get_response
from utils.prompts import prompt_parse_query, prompt_graph_video, prompt_video_answer, prompt_video_answer_final, prompt_agent_verify_answer_referencing
from classes.output_structure import ParseQueryOutput, GraphOutputFormat, VideoOutputFormat
from utils.search import search_with_parse
from utils.general import find_pkl_files, Tee


def reason(graph, video_name, question):

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
    print("Question: ", question)
    
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
        result['final_answer'] = response
    except Exception as e:
        raise Exception(f"Error processing last clip {clip_id}: {e}")
    
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

    # If no video names are provided, process all available graph files.
    if len(sys.argv) < 2:
        available_videos = sorted(find_pkl_files())
    else:
        available_videos = sys.argv[1:]

    with open(f"data/robot.json", "r", encoding="utf-8") as f:
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
                result = reason(graph, video_name, question)
                evaluate_correct = evaluate_answer(question, answer, result["final_answer"])
                print("Evaluate: ", evaluate_correct)

                reasoning_results[question_id] = {
                    "question": question,
                    "ground_truth_answer": answer,
                    "evaluate_correct": evaluate_correct,
                    "reasoning": result,
                    "timestamp": video_question.get("timestamp"),
                    "type": video_question.get("type"),
                }
            except Exception as e:
                print(f"Error processing question {question_id}: {e}")
                traceback.print_exc()
                reasoning_results[question_id] = {
                    "question": question,
                    "ground_truth_answer": answer,
                    "error_message": str(e),
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