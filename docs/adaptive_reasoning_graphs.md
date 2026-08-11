# Adaptive Reasoning Visualization Options

This document outlines several conceptual designs for your paper's diagram illustrating the "Adaptive Reasoning" process. The goal is to show how the agent's intermediate memory grows iteratively, navigating from uncertainty to the final answer using the budget-constrained LangGraph workflow.

## Option 1: The Cyclic Memory Expansion (Agentic Loop)
**Concept:** Focuses on the iterative loop (`Planner` -> `Executor` -> `Verifier`) and the physical growth of the working memory context.
*   **Visual Layout:** A cyclic flowchart in the center (Planner <-> Executor).
*   **Memory Representation:** A box or cylinder on the side labeled "Agent Intermediate Memory (`State.messages`)".
*   **Progression:** Show 3-4 snapshots of this memory box from t=1 to t=final.
    *   t=1: Broad text search results (e.g., `general_search`).
    *   t=2: Temporal context and refined hints (e.g., `search_temporal_context`).
    *   t=3: Raw video frame descriptions (e.g., `watch_video_clip`).
*   **Key Detail:** As the memory box grows larger, a "Budget" gauge decreases from 5 to 0. The final state breaks out of the loop into the "Final Answer" node.

## Option 2: The Subgraph Extraction & Accumulation (Layered Design)
**Concept:** Highlights how the agent dynamically extracts information from the massive static knowledge graph and pulls it into its active memory.
*   **Visual Layout:** Two layers. The bottom layer is the large, complex "Hierarchical Video Memory" (Nodes, Edges, OCR, Conversations). The top layer is the "Agent Working Memory."
*   **Progression:**
    *   **Step 1:** An arrow from the Agent's `general_search` highlights a large subgraph in the bottom layer, and copies its text summaries into the top layer.
    *   **Step 2:** The Agent uses `search_temporal_context` to highlight a specific node/clip in the bottom layer, pulling related adjacent nodes into the top layer.
    *   **Step 3:** The Agent triggers `watch_video_clip` on a specific clip ID, pulling raw visual data into the top layer.
*   **Takeaway:** The top layer (intermediate memory) starts empty and progressively aggregates a highly relevant, condensed sub-graph that converges directly on the final answer.

## Option 3: The Information Funnel / Spiral
**Concept:** Illustrates the shift from broad, high-uncertainty exploration to narrow, highly specific visual grounding.
*   **Visual Layout:** A funnel or a spiral moving inward toward a bullseye (the Final Answer).
*   **Progression:**
    *   **Outer Ring (Broad):** `general_search` - Consumes budget to scan High-level edges, low-level actions, and conversations. High information volume, low specificity.
    *   **Middle Ring (Focused):** `search_temporal_context` - Anchors to a specific `clip_id`. The agent focuses its memory on a tight window.
    *   **Inner Ring (Grounding):** `watch_video_clip` - Agent memory requests expensive MLLM frame analysis to resolve remaining ambiguities.
    *   **Center:** `complete_task` triggered, yielding the Final Answer.
*   **Takeaway:** As the agent moves inward, the *type* of memory appended changes from broad text embeddings to precise visual descriptions, effectively converging on the answer.

## Option 4: The Timeline Information-Gain Chart
**Concept:** A more quantitative or flowchart-style 2D graph charting the trajectory of the reasoning.
*   **X-axis:** Agent Steps (Budget 5 -> 4 -> 3 -> ... -> 0).
*   **Y-axis:** Information Relevancy / Confidence toward the answer.
*   **Plot:** A step-function line moving upwards.
    *   At Step 1, a block drops onto the timeline: "General Search Results".
    *   At Step 2, another block stacks on top: "Verifier Hint + Temporal Context".
    *   At Step 3, a final block stacks: "Visual Frame Observations".
*   **Takeaway:** Clearly shows the monotonic growth of the intermediate memory (`State.messages`) until the "Confidence Threshold" is crossed and `complete_task` is fired.

## Recommendations for the Paper
*   **If your paper focuses on the Graph Structure:** Go with **Option 2**. It beautifully illustrates the utility of having a Heterogeneous Graph because the agent can surgically extract exactly what it needs.
*   **If your paper focuses on the Agent's intelligence/autonomy:** Go with **Option 1**. Showing the Planner-Executor loop making decisions based on growing memory emphasizes the "adaptive" aspect of the reasoning.
