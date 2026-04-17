# python -m utils.search

import pickle
from classes.hetero_graph import HeteroGraph
from classes.output_structure import ParseQueryOutput
from utils.edge_to_string import high_level_edges_to_string, low_level_edge_to_string


def search_with_parse(query, graph, parse_query_response):
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
        parse_query_response: Parsed output from prompt_parse_query
            (typically ParseQueryOutput; may also be tuple(result, tokens))
    
    Returns:
        str: Formatted string containing all search results in natural language
    """
    # Handle generate_text_response(...) outputs that may be (parsed_obj, tokens)
    if isinstance(parse_query_response, tuple):
        if len(parse_query_response) == 0:
            raise ValueError("parse_query_response tuple is empty")
        parse_query_response = parse_query_response[0]

    # ParseQueryOutput object (new expected format)
    if isinstance(parse_query_response, ParseQueryOutput):
        query_triples = parse_query_response.query_triples
        spatial_constraint = parse_query_response.spatial_constraint
        speaker_strict = parse_query_response.speaker_strict
        k_high_level = parse_query_response.allocation.k_high_level
        k_appearance = parse_query_response.allocation.k_appearance
        k_low_level = parse_query_response.allocation.k_low_level
        k_conversations = parse_query_response.allocation.k_conversations
        k_ocr = getattr(parse_query_response.allocation, "k_ocr", 0)
    # Backward compatibility for dict payloads
    elif isinstance(parse_query_response, dict):
        triple = parse_query_response.get("query_triple")
        triples = parse_query_response.get("query_triples")
        spatial_constraint = parse_query_response.get("spatial_constraint")
        speaker_strict = parse_query_response.get("speaker_strict")
        allocation = parse_query_response.get("allocation", {})

        if triples and isinstance(triples, list):
            query_triples = triples
        elif triple:
            query_triples = [triple]
        else:
            raise ValueError("query_triple(s) not found in strategy")

        # Ensure weight fields (indices 3,4,5) of each triple are float
        normalized = []
        for one_triple in query_triples:
            if isinstance(one_triple, (list, tuple)) and len(one_triple) >= 6:
                row = list(one_triple)
                for idx in (3, 4, 5):
                    if not isinstance(row[idx], (int, float)):
                        try:
                            row[idx] = float(row[idx])
                        except (TypeError, ValueError):
                            row[idx] = 1.0
                normalized.append(row)
            else:
                normalized.append(one_triple if isinstance(one_triple, list) else list(one_triple))
        query_triples = normalized

        k_high_level = allocation.get("k_high_level", 5)
        k_appearance = allocation.get("k_appearance", 0)
        k_low_level = allocation.get("k_low_level", 10)
        k_conversations = allocation.get("k_conversations", 10)
        k_ocr = allocation.get("k_ocr", 0)
    else:
        raise TypeError(
            "parse_query_response must be ParseQueryOutput, dict, or tuple(parsed, tokens). "
            f"Got: {type(parse_query_response).__name__}"
        )

    # Search the graph
    try:
        # Search in two independent parts with separate k budgets.
        high_level_edges = graph.search_high_level_edges(query_triples, max(0, k_high_level))
        print("High-level edges searched: ", len(high_level_edges))
        appearance_edges = graph.search_appearance_edges(query_triples, max(0, k_appearance))
        
        # Search low-level edges
        low_level_edges = graph.search_low_level_edges(
            query_triples, 
            k_low_level,
            spatial_constraint
        )
        print("Low-level edges searched: ", len(low_level_edges))
        
        # Search conversations (use original query string)
        conversation_results = graph.search_conversations(
            query,
            k_conversations,
            speaker_strict
        )
        print("Conversations searched: ", len(conversation_results))

        # Search OCR
        ocr_results = []
        if k_ocr > 0:
            ocr_results = graph.search_ocr_info(query, k_ocr)
            print("OCR searched: ", len(ocr_results))
        
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

    # Format appearance edges
    if appearance_edges:
        appearance_str = high_level_edges_to_string(appearance_edges)
        if appearance_str:
            result_sections.append("**Appearance Information: **\n")
            result_sections.append(appearance_str)
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
            result_sections.append("")

    # Format OCR
    if ocr_results:
        ocr_lines = []
        for ocr in ocr_results:
            ocr_lines.append(f"[{ocr.clip_id}] OCR ({ocr.context}): {ocr.content}")
        if ocr_lines:
            result_sections.append("**OCR Information: **\n")
            result_sections.append("\n".join(ocr_lines))
            result_sections.append("")
    
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
        parse_query_response, _tokens = generate_text_response(
            prompt_parse_query + "\n" + query,
            ParseQueryOutput
        )
        result = search_with_parse(query, graph, parse_query_response)
        print(result)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        