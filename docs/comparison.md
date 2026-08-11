# Comparison of HVM Agentic Workflows: HVM-web, HVM-Hippo, and HVM-Ego

This document compares the agentic workflow implementations across three projects: `HVM-web`, `HVM-Hippo`, and `HVM-Ego`. While all three use LangGraph and a graph-based approach to video understanding, they differ in their complexity, verification logic, and target use cases.

## Architectural Comparison

| Feature | HVM-web | HVM-Hippo | HVM-Ego |
| :--- | :--- | :--- | :--- |
| **Core Workflow** | **Planner-Executor-Verifier** cycle. | Simple **LLM-Tool Loop** with retries. | **Refined Planner-Executor-Verifier** with temporal logic. |
| **Primary Model** | `gpt-4o` | `gpt-4o-mini` | `gpt-5-mini` (simulated/target) |
| **State Management** | Tracks findings, clip history, and explicit search budget. | Tracks message history, call counts, and retry status. | Tracks tool history, budget, and structured output state. |
| **Search Tools** | `general_search`, `temporal_context`, `watch_video_clip`. | `general_search` (with k-budgeting), `search_object`, etc. | `general_search`, `temporal_search`, `search_object`. |
| **Verification Logic**| Detects empty results and redundant clip watching. | Ensures a valid MCQ option (A/B/C/D) is extracted. | Enforces temporal boundary checks for "first/last/before/after" questions. |

---

## Project Breakdown

### 1. HVM-web
**Focus:** High-fidelity video understanding with visual verification.
- **Key Advantage:** Includes a `watch_video_clip` tool that allows the agent to retrieve raw video frames when the text-based graph is ambiguous.
- **Workflow:** Uses a sophisticated Planner-Executor-Verifier pattern. The **Verifier** node acts as a "critic," providing system hints if the agent gets stuck or performs redundant actions.
- **Budgeting:** Introduces explicit budget management where the agent is encouraged to allocate search "k-values" (e.g., `k_action`, `k_conversation`) across different modalities.

### 2. HVM-Hippo
**Focus:** Efficiency and large-scale graph-QA evaluation.
- **Key Advantage:** Highly efficient and lightweight. It provides a comprehensive "Graph Stats" summary (e.g., character count, edge counts, clip count) in the prompt every round, giving the LLM a clear "map" of the data.
- **Ablation Ready:** Built with testing in mind, including specific modes to test performance without budget allocation or with fixed search depths (`k=30`).
- **Simplicity:** Easier to modify for new datasets where a complex planner-verifier overhead might be overkill.

### 3. HVM-Ego
**Focus:** Complex temporal reasoning and Ego-centric activities.
- **Key Advantage:** **Temporal Intelligence.** The Verifier specifically checks if the question involves temporal boundaries (e.g., "What was the first thing..."). If the agent hasn't used `temporal_search` yet, the Verifier issues a mandatory hint to guide it toward the boundary event.
- **Structured Output:** Uses Pydantic and `with_structured_output` to ensure the final answer is always a valid MCQ choice (A, B, C, or D), reducing parsing errors.
- **Repetition Prevention:** Explicitly checks if the agent is repeating the exact same tool call and forces a strategy change.

---

## Summary of Advantages

### HVM-web
*   **Best for:** Tasks requiring high visual accuracy or multi-modal verification.
*   **Advantage:** The `watch_video_clip` tool bridges the gap between graph abstraction and raw visual data.

### HVM-Hippo
*   **Best for:** Large-scale benchmarks and projects where cost/speed are priorities.
*   **Advantage:** Minimalist architecture and detailed prompt-based graph summaries make it robust for `gpt-4o-mini`.

### HVM-Ego
*   **Best for:** Long-form video QA and complex activity reasoning (e.g., EgoLife).
*   **Advantage:** Advanced verification logic prevents "agent wandering" and ensures strict adherence to temporal reasoning requirements.
