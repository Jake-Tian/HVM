import glob
from pathlib import Path
from langchain.tools import tool

from classes.hetero_graph import HeteroGraph
from utils.edge_to_string import high_level_edges_to_string, low_level_edge_to_string
from utils.prompts import prompt_video_answer
from utils.mllm_gpt import generate_messages, get_response
from classes.output_structure import VideoOutputFormat

def get_tools(graph: HeteroGraph, video_name: str, query: str):

    @tool
    def general_search(
        query_triples: list,
        k_low_level: int,
        k_high_level: int,
        k_conversations: int,
        k_appearance: int = 5,
        total_k: int = 0,
        speaker_strict: list[str] = None
    ) -> str:
        """
        General semantic retrieval across behavior edges and conversation messages.
        
        Args:
            query_triples: list of lists, each element list should have the length of 6, source, content, target, and corresponding weights. Example: [["source", "content", "target", 1.0, 1.0, 1.0]]
            k_low_level: number of low-level behavior edges to return.
            k_high_level: number of high-level edges to return.
            k_conversations: number of conversation messages to return.
            k_appearance: number of character appearance descriptions to return (default 5).
            total_k: total number of results (optional).
            speaker_strict: optional strict speaker filter for conversations.
        """
        try:
            # Sanitize query triples to ensure weights are floats
            sanitized_triples = []
            for t in query_triples:
                if isinstance(t, list):
                    new_t = list(t)
                    while len(new_t) < 6:
                        new_t.append(1.0)
                    for i in range(3, 6):
                        try:
                            new_t[i] = float(new_t[i]) if new_t[i] is not None else 1.0
                        except (ValueError, TypeError):
                            new_t[i] = 1.0
                    sanitized_triples.append(new_t)
                else:
                    sanitized_triples.append(t)
            query_triples = sanitized_triples
            
            high_level_edges = graph.search_high_level_edges(query_triples, max(0, k_high_level))
            appearance_edges = graph.search_appearance_edges(query_triples, max(0, k_appearance))
            low_level_edges = graph.search_low_level_edges(query_triples, max(0, k_low_level))
            conversation_results = graph.search_conversations(query, max(0, k_conversations), speaker_strict)
            
            result_sections = []
            if high_level_edges:
                high_level_str = high_level_edges_to_string(high_level_edges)
                if high_level_str:
                    result_sections.append("**High-Level Information: **\n" + high_level_str)
            if appearance_edges:
                appearance_str = high_level_edges_to_string(appearance_edges)
                if appearance_str:
                    result_sections.append("**Appearance Information: **\n" + appearance_str)
            if low_level_edges:
                low_level_str = low_level_edge_to_string(low_level_edges)
                if low_level_str:
                    result_sections.append("**Low-Level Information: **\n" + low_level_str)
            if conversation_results:
                conversation_str = graph.get_conversation_messages_with_context(conversation_results)
                if conversation_str:
                    result_sections.append("**Conversations: **\n" + conversation_str)
            
            res = "\n".join(result_sections)
            return res if res.strip() else "No relevant information found."
        except Exception as e:
            return f"Error in general_search: {str(e)}"

    @tool
    def get_clip_context(clip_id: int) -> str:
        """
        Return the whole conversation and its summary for the given clip id.
        
        Args:
            clip_id: ID of the clip to retrieve context for.
        """
        found = False
        res = []
        for conv_id, conversation in graph.conversations.items():
            if clip_id in conversation.clips:
                found = True
                formatted = conversation.format_messages()
                summary = getattr(conversation, 'summary', '')
                res.append(f"Conversation {conv_id} for clip {clip_id}:\nSummary: {summary}\n{formatted}")
        
        if found:
            return "\n\n".join(res)
        return f"No conversation found for clip {clip_id}."

    @tool
    def video_rewatch(clip_id: int) -> str:
        """
        Only use it when the text result is insufficient, because the cost for video_rewatch is high.
        
        Args:
            clip_id: ID of the clip to rewatch.
        """
        frames_dir = Path(f"data/frames/{video_name}") / str(clip_id)
        if not frames_dir.exists():
            return f"Video frames for clip {clip_id} not found."
            
        images = sorted(glob.glob(str(frames_dir / "*.jpg")), key=lambda x: int(Path(x).stem))
        if not images:
            return f"No images found for clip {clip_id}."
            
        prompt = prompt_video_answer + "\nQuestion: " + query + "\nCurrent clip ID: " + str(clip_id)
        try:
            messages = generate_messages(images, prompt)
            response, _ = get_response(messages, VideoOutputFormat)
            return f"Video Rewatch Answer for clip {clip_id}: answer={response.answer}, content={response.content}"
        except Exception as e:
            return f"Error in video rewatch: {str(e)}"

    return [general_search, get_clip_context, video_rewatch]
