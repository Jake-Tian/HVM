import os
import json
from collections import defaultdict

def analyze_results(directory):
    video_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    error_videos = []
    
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            video_name = filename.replace(".json", "")
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    error_count = 0
                    total_count = len(data)
                    for q_id, q_data in data.items():
                        reasoning = q_data.get("reasoning", {})
                        if isinstance(reasoning, str) and "Error" in reasoning:
                            error_count += 1
                    
                    if error_count > 0:
                        error_videos.append((video_name, error_count, total_count))
                    
                    # If some questions are not errors, we still want to count them for stats
                    for q_id, q_data in data.items():
                        reasoning = q_data.get("reasoning", {})
                        if isinstance(reasoning, str):
                            is_correct = False
                        else:
                            is_correct = reasoning.get("evaluate_correct", False)
                        
                        categories = q_data.get("type", [])
                        
                        # Only count non-error questions for accuracy stats
                        if not (isinstance(reasoning, str) and "Error" in reasoning):
                            video_stats[video_name]["total"] += 1
                            if is_correct:
                                video_stats[video_name]["correct"] += 1
                            
                            for cat in categories:
                                category_stats[cat]["total"] += 1
                                if is_correct:
                                    category_stats[cat]["correct"] += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                
    return video_stats, category_stats, error_videos

video_stats, category_stats, error_videos = analyze_results("data/reasoning")

total_correct = sum(v["correct"] for v in video_stats.values())
total_questions = sum(v["total"] for v in video_stats.values())
overall_accuracy = (total_correct / total_questions) * 100 if total_questions > 0 else 0

print(f"--- Overall Accuracy ---")
print(f"Total Correct: {total_correct}")
print(f"Total Questions (excluding errors): {total_questions}")
print(f"Accuracy: {overall_accuracy:.2f}%")

print("\n--- Error Videos (Error Count / Total Questions) ---")
for video, err_count, total in sorted(error_videos):
    print(f"{video}: {err_count}/{total}")
