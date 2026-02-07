# python -m utils.search

import json
import pickle
from pathlib import Path
from classes.hetero_graph import HeteroGraph
from utils.general import strip_code_fences
from utils.reasoning.edge_to_string import high_level_edges_to_string, low_level_edge_to_string


def search_with_parse(query, graph, parse_query_response, skip_high_level: bool = False):
    """
    Search the graph and return search results based on a parsed query.
    
    This function:
    1. Parses the parse_query_response to extract search strategy
    2. Searches high-level edges, low-level edges, and conversations
    3. Formats all results into a single natural language string
    4. Returns the formatted search results
    
    Args:
        query: Natural language query string (used for conversation search)
        graph: HeteroGraph instance to search
        parse_query_response: Raw output from prompt_parse_query (JSON string)
    
    Returns:
        str: Formatted string containing all search results in natural language
    """
    # Transfer the strategy into dictionary
    try:
        strategy_dict = json.loads(strip_code_fences(parse_query_response))
    except json.JSONDecodeError as e:
        raise Exception(f"Error parsing strategy JSON: {e}\nRaw strategy response: {parse_query_response}")

    # Extract strategy components with safe access
    triple = strategy_dict.get("query_triple")
    triples = strategy_dict.get("query_triples")
    spatial_constraint = strategy_dict.get("spatial_constraint")
    speaker_strict = strategy_dict.get("speaker_strict")
    allocation = strategy_dict.get("allocation", {})

    if triples and isinstance(triples, list):
        query_triples = triples
    elif triple:
        query_triples = [triple]
    else:
        raise ValueError("query_triple(s) not found in strategy")

    # Get k values from allocation, with defaults as fallback
    k_high_level = allocation.get("k_high_level", 10)
    k_low_level = allocation.get("k_low_level", 10)
    k_conversations = allocation.get("k_conversations", 10)

    # Search the graph
    try:
        # Search high-level edges (skip when ablation: no_highlevel)
        if skip_high_level:
            high_level_edges = []
        else:
            high_level_edges = graph.search_high_level_edges(query_triples, k_high_level)
        
        # Search low-level edges
        low_level_edges = graph.search_low_level_edges(
            query_triples, 
            k_low_level,
            spatial_constraint
        )
        
        # Search conversations (use original query string)
        conversation_results = graph.search_conversations(
            query,
            k_conversations,
            speaker_strict
        )
        
    except Exception as e:
        raise Exception(f"Error searching graph: {e}")

    # Format results into strings
    result_sections = []
    
    # Format high-level edges
    if high_level_edges:
        high_level_str = high_level_edges_to_string(high_level_edges)
        if high_level_str:
            result_sections.append("**High-Level Information (Character Attributes and Relationships): **\n")
            result_sections.append(high_level_str)
            result_sections.append("")
    
    # Format low-level edges
    if low_level_edges:
        low_level_str = low_level_edge_to_string(low_level_edges)
        if low_level_str:
            result_sections.append("**Low-Level Information (Actions and Events): **\n")
            result_sections.append(low_level_str)
            result_sections.append("")
    
    # Format conversations
    if conversation_results:
        conversation_str = graph.get_conversation_messages_with_context(conversation_results)
        if conversation_str:
            result_sections.append("**Conversations: **\n")
            result_sections.append(conversation_str)
    
    # Combine all sections
    graph_search_results = "\n".join(result_sections)
    
    # If no results found, return a message
    if not graph_search_results.strip():
        graph_search_results = "No relevant information found for this query."
    
    return graph_search_results


if __name__ == "__main__":
    # Example usage
    from utils.llm import generate_text_response
    from utils.prompts import prompt_parse_query
    
    with open("data/semantic_memory/gym_01.pkl", "rb") as f:
        graph = pickle.load(f)
    query = "Which takeout should be taken to Anna?"
    
    try:
        parse_query_response = generate_text_response(prompt_parse_query + "\n" + query)
        result = search_with_parse(query, graph, parse_query_response)
        print(result)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        