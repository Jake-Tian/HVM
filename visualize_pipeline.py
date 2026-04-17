import os
import sys
import operator
from typing import TypedDict, Annotated

# Ensure the current directory is in the path so we can import local modules
sys.path.append(os.getcwd())

from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    question: str
    messages: Annotated[list, operator.add]
    findings: Annotated[list, operator.add]
    clip_history: Annotated[list, operator.add]
    budget: int

def main():
    print("Building structural representation of the agent...")
    
    # Define nodes as empty functions for visualization purposes
    def planner(state): return state
    def executor(state): return state
    def verifier(state): return state
    def final_answer(state): return state
    
    # Define the conditional routing logic
    def route_after_executor(state): return "verifier"

    # Reconstruct the graph structure manually to avoid LLM/API initialization
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner)
    workflow.add_node("executor", executor)
    workflow.add_node("verifier", verifier)
    workflow.add_node("final_answer", final_answer)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_conditional_edges("executor", route_after_executor, {
        "verifier": "verifier",
        "final_answer": "final_answer"
    })
    workflow.add_edge("verifier", "planner")
    workflow.add_edge("final_answer", END)

    app = workflow.compile()

    # Get the Mermaid string representation
    mermaid_graph = app.get_graph().draw_mermaid()
    
    # Create a standalone HTML file using the Mermaid.js library
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HVM-web Agent Pipeline Visualization</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            background-color: #f4f4f9;
            margin: 0;
            padding: 20px;
        }}
        h1 {{ color: #333; }}
        .container {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-top: 20px;
            width: 90%;
            max-width: 1000px;
        }}
        .mermaid {{
            display: flex;
            justify-content: center;
        }}
        .info {{
            margin-top: 20px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>HVM-web Agentic Workflow</h1>
    <div class="container">
        <div class="mermaid">
            {mermaid_graph}
        </div>
    </div>
    <div class="info">
        <p>This graph shows the iterative loop: <b>Planner</b> &rarr; <b>Executor</b> &rarr; <b>Verifier</b> &rarr; <b>Planner</b>.</p>
        <p>The loop exits to <b>Final Answer</b> when <i>complete_task</i> is called.</p>
    </div>
    <script>
        mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});
    </script>
</body>
</html>
"""

    output_path = "agent_pipeline.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    
    print(f"✓ Visualization saved to: {os.path.abspath(output_path)}")
    print("You can open this file in any web browser to see the interactive graph.")

if __name__ == "__main__":
    main()
