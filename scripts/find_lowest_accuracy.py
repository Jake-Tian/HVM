import os
import json
from collections import defaultdict

def analyze_results(directory):
    video_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            video_name = filename.replace(".json", "")
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    for q_id, q_data in data.items():
                        reasoning = q_data.get("reasoning", {})
                        if isinstance(reasoning, str):
                            is_correct = False
                            # Skip error cases from accuracy calculation if they are purely processing errors
                            if "Error" in reasoning:
                                continue
                        else:
                            is_correct = reasoning.get("evaluate_correct", False)
                        
                        video_stats[video_name]["total"] += 1
                        if is_correct:
                            video_stats[video_name]["correct"] += 1
            except Exception as e:
                pass
                
    return video_stats

video_stats = analyze_results("data/reasoning")

# Calculate accuracy and filter out videos with 0 total questions (all errors)
video_acc = []
for video, stats in video_stats.items():
    if stats["total"] > 0:
        acc = stats["correct"] / stats["total"]
        video_acc.append((video, acc, stats["correct"], stats["total"]))

# Sort by accuracy (ascending) and then by total questions (descending) to break ties
video_acc.sort(key=lambda x: (x[1], -x[3]))

print("--- 5 Videos with Lowest Accuracy ---")
for i in range(min(5, len(video_acc))):
    video, acc, correct, total = video_acc[i]
    print(f"{i+1}. {video}: {acc*100:.2f}% ({correct}/{total})")
