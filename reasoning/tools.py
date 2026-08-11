import glob
from pathlib import Path
from langchain_core.tools import tool

from utils.llm_gpt import generate_text_response
from utils.mllm_gpt import generate_messages, get_response
from utils.prompts import prompt_parse_query_no_allocation
from classes.output_structure import ParseQueryOutputNoAllocation
from utils.edge_to_string import high_level_edges_to_string, low_level_edge_to_string

def execute_general_search(graph, query, k_action, k_conversation, k_ocr, k_high_level=5, k_appearance=5):
    total_tokens = 0
    try:
        parse_response, tokens = generate_text_response(prompt_parse_query_no_allocation + "\n" + query, ParseQueryOutputNoAllocation)
        total_tokens += (tokens or 0)
        
        query_triples = []
        spatial_constraint = None
        speaker_strict = None
        
        if isinstance(parse_response, ParseQueryOutputNoAllocation):
            query_triples = parse_response.query_triples
            spatial_constraint = parse_response.spatial_constraint
            speaker_strict = parse_response.speaker_strict
        elif isinstance(parse_response, dict):
            query_triples = parse_response.get("query_triples", [])
            spatial_constraint = parse_response.get("spatial_constraint")
            speaker_strict = parse_response.get("speaker_strict")
        elif isinstance(parse_response, tuple) and len(parse_response) > 0:
            resp = parse_response[0]
            if hasattr(resp, "query_triples"):
                query_triples = resp.query_triples
                spatial_constraint = getattr(resp, "spatial_constraint", None)
                speaker_strict = getattr(resp, "speaker_strict", None)

        if not query_triples:
            query_triples = [[query, "?", "?", 0.9, 0.5, 0.5]]

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
        
        result_sections = []
        
        # High level and appearance are searched using allocated budget
        if k_high_level > 0:
            high_level_edges = graph.search_high_level_edges(query_triples, k_high_level)
            if high_level_edges:
                hl_str = high_level_edges_to_string(high_level_edges)
                if hl_str:
                    result_sections.append("**High-Level Information:**\n" + hl_str)
        
        if k_appearance > 0:
            appearance_edges = graph.search_appearance_edges(query_triples, k_appearance)
            if appearance_edges:
                app_str = high_level_edges_to_string(appearance_edges)
                if app_str:
                    result_sections.append("**Appearance Information:**\n" + app_str)
        
        if k_action > 0:
            low_level_edges = graph.search_low_level_edges(query_triples, k_action, spatial_constraint)
            if low_level_edges:
                ll_str = low_level_edge_to_string(low_level_edges)
                if ll_str:
                    result_sections.append("**Low-Level Information (Actions):**\n" + ll_str)
                    
        if k_conversation > 0:
            conv_results = graph.search_conversations(query, k_conversation, speaker_strict)
            if conv_results:
                conv_str = graph.get_conversation_messages_with_context(conv_results)
                if conv_str:
                    result_sections.append("**Conversations:**\n" + conv_str)
                    
        if k_ocr > 0:
            ocr_results = graph.search_ocr_info(query, k_ocr)
            if ocr_results:
                ocr_lines = []
                for ocr in ocr_results:
                    ocr_lines.append(f"[{ocr.clip_id}] OCR ({ocr.context}): {ocr.content}")
                if ocr_lines:
                    result_sections.append("**OCR Information:**\n" + "\n".join(ocr_lines))
                    
        result_str = "\n\n".join(result_sections)
        if not result_str.strip():
            return "No relevant information found for this query.", total_tokens
        return result_str, total_tokens
    except Exception as e:
        return f"Error executing search: {e}", total_tokens

def execute_search_temporal_context(graph, clip_id, window=1):
    result_sections = []
    
    temporal_edges = []
    for edge in graph.edges.values():
        if edge.clip_id > 0 and abs(edge.clip_id - clip_id) <= window:
            temporal_edges.append(edge)
            
    if temporal_edges:
        ll_str = low_level_edge_to_string(temporal_edges)
        if ll_str:
            result_sections.append(f"**Actions around clip {clip_id}:**\n" + ll_str)
            
    conv_lines = []
    for conv in graph.conversations.values():
        # A conversation may span multiple clips; check any of them.
        clips = getattr(conv, "clips", None) or [getattr(conv, "clip_id", None)]
        if any(c is not None and abs(c - clip_id) <= window for c in clips):
            conv_lines.append(f"Clip(s) {sorted(clips)}:\n" + conv.format_messages())
    if conv_lines:
        result_sections.append(f"**Conversations around clip {clip_id}:**\n" + "\n".join(conv_lines))
        
    ocr_lines = []
    for ocr in graph.ocr_info:
        if abs(ocr.clip_id - clip_id) <= window:
            ocr_lines.append(f"[{ocr.clip_id}] OCR ({ocr.context}): {ocr.content}")
    if ocr_lines:
        result_sections.append(f"**OCR around clip {clip_id}:**\n" + "\n".join(ocr_lines))
        
    result_str = "\n\n".join(result_sections)
    if not result_str.strip():
        return f"No temporal information found around clip {clip_id}.", 0
    return result_str, 0

def execute_watch_video_clip(video_name, clip_id, focus):
    frames_dir = Path(f"data/frames/{video_name}") / str(clip_id)
    if not frames_dir.exists():
        return f"Video clip {clip_id} not found.", 0
    
    images = sorted(glob.glob(str(frames_dir / "*.jpg")), key=lambda x: int(Path(x).stem))
    if not images:
        return f"No frames found for clip {clip_id}.", 0
        
    prompt = f"Focus: {focus}\nWatch the video clip and describe what happens related to the focus. Be concise."
    try:
        messages = generate_messages(images, prompt)
        response, tokens = get_response(messages)
        if isinstance(response, str):
            return response, (tokens or 0)
        else:
            return str(response), (tokens or 0)
    except Exception as e:
        return f"Error watching video clip {clip_id}: {e}", 0

def get_tools(graph, video_name):
    @tool
    def general_search(query: str, k_action: int, k_conversation: int, k_ocr: int, k_high_level: int, k_appearance: int):
        """
        General semantic search that covers conversation, action, and OCR information.
        Use this FIRST for all questions to get a temporal anchor and find relevant clip_ids.
        Allocate your budget (Total k <= 50) based on the primary modality of the question:
          - k_action (0-30): Primary for behavior, actions, temporal sequence, or 'where is' queries.
          - k_conversation (0-30): Primary for 'why', dialogue, or causal reasoning.
          - k_ocr (0-30): Primary for text on signs, labels, or posters.
          - k_high_level (0-15): Secondary for character traits or relationships.
          - k_appearance (0-10): Use ONLY for physical looks, hair, or clothing. Set to 0 if irrelevant.
        """
        return execute_general_search(graph, query, k_action, k_conversation, k_ocr, k_high_level, k_appearance)

    @tool
    def search_temporal_context(clip_id: int):
        """
        Search what happened in and around a specific video clip (window=1). 
        Use this ONLY after finding a candidate clip_id via general_search to see events right before or after it.
        """
        return execute_search_temporal_context(graph, clip_id)

    @tool
    def watch_video_clip(clip_id: int, focus: str):
        """
        Watch the raw video frames of a specific clip. 
        Use this ONLY after finding a candidate clip_id via general_search. 
        Mandatory for visual questions requiring high detail (e.g., specific placement of objects, visual state, exact counts) when the text graph is insufficient.
        Provide a specific `focus` based on what is missing from the text. It is expensive.
        """
        return execute_watch_video_clip(video_name, clip_id, focus)
    
    @tool
    def complete_task(ready: bool):
        """
        Call this tool when you have enough information to answer the question, or when you exhaust your budget.
        """
        return "Task marked complete. Proceeding to final verification."

    return [general_search, search_temporal_context, watch_video_clip, complete_task]
