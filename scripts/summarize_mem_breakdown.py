import json

# Load data
try:
    with open('data/analysis/memorization_tokens.json') as f:
        mem_tokens = json.load(f)
except Exception as e:
    print(f"Error loading mem_tokens: {e}")
    mem_tokens = {}

try:
    with open('data/episodic_memory/episodic_memory.json') as f:
        episodic_data = json.load(f)
except Exception as e:
    print(f"Error loading episodic_memory: {e}")
    episodic_data = {}

# Categories
categories = ['mllm', 'triples', 'attributes', 'relationships', 'conversation']
category_totals = {c: 0 for c in categories}
total_clips = 0

for vid, tokens in mem_tokens.items():
    for c in categories:
        category_totals[c] += tokens.get(c, 0)
        
    if vid in episodic_data:
        total_clips += len(episodic_data[vid])

total_duration_minutes = total_clips * 0.5

print(f"Total Duration (minutes): {total_duration_minutes}")
print(f"\n=== Memorization Breakdown Per Minute ===")

for c in categories:
    avg_per_minute = category_totals[c] / total_duration_minutes if total_duration_minutes > 0 else 0
    print(f"{c.capitalize()}: {avg_per_minute:,.0f} tokens/minute")

