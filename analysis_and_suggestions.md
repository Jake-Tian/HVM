# HVM-Hippo Pipeline Analysis & Suggestions Report

## 1. Accuracy Analysis (Over 1000 Questions)

- **Overall Accuracy:** **70.4%** (699/993 processed)
- **Accuracy without BG Sound Questions:** **71.2%** (643/903)
- **Background Sound Accuracy:** **57.7%** (56/97)
  - *Note: The agent often "guesses" these correctly based on the general scene context (e.g., guessing "birds" in a park), but frequently fails on specific auditory cues.*

## 2. Accuracy by Question Type


| Category         | Accuracy  | Count   |
| ---------------- | --------- | ------- |
| **Summary**      | **92.0%** | 230/250 |
| **Audio**        | **76.0%** | 190/250 |
| **Audio-Visual** | **60.8%** | 152/250 |
| **Visual**       | **50.8%** | 127/250 |


## 3. Common Failure Modes

1. **Visual Granularity (The "Description Gap"):** The `visual` and `audio-visual` categories are the weakest. While `episodic_memory.json` is detailed, it often misses specific small-scale visual features (e.g., "color of the arrow on a sign", "logo on a uniform", "pattern on a shirt").
2. **Temporal Grounding (Clip ID Ambiguity):** The agent often finds the correct *vibe* but cannot pin down the exact *clip_id* for a specific mention (e.g., failing to distinguish between "mentioning MIT" vs. "visiting MIT").
3. **Retrieval Noise in `general_search`:** When the agent searches for a specific object (e.g., "backpack color"), the vector search often returns clips where the *character* appears but the *object* isn't explicitly described in the triple, leading to hallucinations or incorrect guesses.

---

## 4. Suggestions for Improvement

### A. Memorization Phase Enhancements

1. **Auditory Event Extraction:**
  - **Modify `process_full_video.py`**: Add a dedicated step for the MLLM to list "Prominent Background Sounds" for each 30s clip.
  - **Impact**: This converts background sound from a "hidden" modality into searchable text in the graph, making it solvable via `general_search` without always needing the expensive `listen_to_audio` tool.
2. **Visual Detail "Zoom" Prompting:**
  - Update the `prompt_generate_episodic_memory` to explicitly ask for: "List any text, logos, distinct patterns, or colors of background objects visible."
  - **Impact**: Currently, the memory focuses on *actions*. Adding *static details* will help with the low-performing `visual` category.

### B. Reasoning Phase & Tools

1. **Add `visual_verify(clip_id, query)` Tool:**
  - **Logic**: Similar to `listen_to_audio`, this tool would send frames from a specific clip back to the MLLM with a targeted question (e.g., "What color is the arrow on the sign in this clip?").
  - **Impact**: High-precision visual questions (logos, colors) are hard to store in a general graph. It's better to "re-examine" the evidence once the search space is narrowed to 1-2 clips.
2. **Spatial-Temporal Triple Weights (Prompt Update):**
  - Encourage the agent to use higher weights (0.9+) for the *target* object in visual queries. 
  - *Example*: Instead of `["<Angela>", "wears", "backpack"]`, use `["?", "is color of", "backpack"]`.

### C. Component Modifications

1. **Two-Tier Search Strategy:**
  - Modify `reason.py` to automatically trigger a `visual_verify` or `listen_to_audio` if the question category is `visual` or `audio` and the graph results are ambiguous.
2. **Top-K Filtering by Clip Density:**
  - If a search for "wooden shoes" returns clips 18, 19, and 20, the agent should prioritize these as an "event cluster" rather than just looking at the highest-ranked single triple.

