import json
import re
import numpy as np
from .node_class import CharacterNode, ObjectNode
from .edge_class import Edge
from .conversation import Conversation
from .output_structure import ConversationSummary, TimeTriple
from collections import defaultdict
from utils.prompts import prompt_character_summary, prompt_character_relationships, prompt_conversation_summary
from utils.llm import generate_text_response, get_embedding, get_multiple_embeddings
from utils.general import strip_code_fences


class HeteroGraph:
    def __init__(self):

        self.characters = {}   # name → CharacterNode object
        self.objects = {}   # name → ObjectNode object
        self.conversations = {}   # id → Conversation object
        self.edges = {}   # id → Edge object
        self.current_conversation_id = None  # Track the most recent conversation ID

        # adjacency lists for O(1) search
        self.adjacency_list_out = defaultdict(list)  # node → list of edge IDs (outgoing edges)
        self.adjacency_list_in = defaultdict(list)   # node → list of edge IDs (incoming edges)


    # --------------------------------------------------------
    # Node API
    # --------------------------------------------------------
    def add_character(self, name):
        # Ensure name has angle brackets
        if not name.startswith("<") or not name.endswith(">"):
            name = f"<{name}>"
        
        # For other characters, check if already exists
        if name in self.characters:
            return name
        
        character = CharacterNode(name)
        self.characters[character.name] = character
        return character.name
    
    def get_character(self, name):
        """Get a character by name. Returns None if not found."""
        return self.characters.get(name)
    
    def get_node_degrees(self):
        """
        Calculate the degree (number of connected edges) for each node in the graph.
        """
        degrees = {}
        
        # Get all nodes that appear in either adjacency list
        all_nodes = set(self.adjacency_list_out.keys()) | set(self.adjacency_list_in.keys())
        
        # Calculate degree for each node (outgoing + incoming)
        for node in all_nodes:
            out_degree = len(self.adjacency_list_out[node])
            in_degree = len(self.adjacency_list_in[node])
            degrees[node] = out_degree + in_degree
        
        return degrees

    def _parse_node_string(self, node_str):
        """
        Parse a node string to determine if it's a character or object.
        Returns: (is_character, name)
        """
        if node_str is None:
            return (False, None)
        
        node_str = str(node_str).strip()
        
        # Check if it's a character node (surrounded by angle brackets)
        if node_str.startswith("<") and node_str.endswith(">"):
            # Keep the angle brackets for consistency with storage
            character_name = node_str  # Keep angle brackets
            return (True, character_name)
        
        # It's an object node - just return the name as-is
        return (False, node_str)
    
    def get_object_node(self, node_str):
        """
        Get an object node by its string representation.
        For character nodes, returns None (use get_character instead).
        """
        is_char, name = self._parse_node_string(node_str)
        if is_char:
            return None  # It's a character node
        return self.objects.get(name)
    
    def _get_or_create_object_node(self, name):
        """
        Get an existing object node or create a new one.
        Uniqueness is determined by name only.
        Returns: (node_name, node_name) for consistency with existing code
        """
        # Check if object already exists
        if name in self.objects:
            return (name, name)
        
        # Create new object node
        obj_node = ObjectNode(name)
        self.objects[name] = obj_node
        
        return (name, name)
    

    # --------------------------------------------------------
    # Conversation API
    # --------------------------------------------------------
    def insert_conversation(self, conversation_messages, embeddings, summary=""):

        if not conversation_messages:
            return None

        conversation = Conversation(conversation_messages, embeddings, summary)
        self.conversations[conversation.id] = conversation
        self.current_conversation_id = conversation.id
        return conversation.id


    # --------------------------------------------------------
    # Edge API
    # --------------------------------------------------------
    def add_edge(self, edge):
        # Check if source and target nodes exist
        # Edges store node names as strings, so we need to check:
        # - For characters: direct lookup in self.characters (with angle brackets)
        # - For objects: direct lookup in self.objects by name
        
        source_exists = False
        # If source has angle brackets, it's a character; otherwise it's an object
        if edge.source.startswith("<") and edge.source.endswith(">"):
            # It's a character - check directly
            if edge.source in self.characters:
                source_exists = True
        else:
            # It's an object - check by name
            if edge.source in self.objects:
                source_exists = True
        
        target_exists = False
        # Special case: None is allowed as target without creating a node
        if edge.target is None:
            target_exists = True
        # If target has angle brackets, it's a character; otherwise it's an object
        elif edge.target.startswith("<") and edge.target.endswith(">"):
            # It's a character - check directly
            if edge.target in self.characters:
                target_exists = True
        else:
            # It's an object - check by name
            if edge.target in self.objects:
                target_exists = True
        
        if not source_exists:
            raise ValueError(f"Source node '{edge.source}' not found in graph")
        if not target_exists:
            raise ValueError(f"Target node '{edge.target}' not found in graph")

        self.edges[edge.id] = edge
        # Add to both adjacency lists (edges are directed by default)
        self.adjacency_list_out[edge.source].append(edge.id)
        # Handle None target for adjacency list
        if edge.target is not None:
            self.adjacency_list_in[edge.target].append(edge.id)
        else:
            self.adjacency_list_in[None].append(edge.id)

        return edge.id


    def insert_triples(self, triples: list[TimeTriple]):
        """
        Insert normalized time triples into the graph.
        """
        if not isinstance(triples, list) or not triples:
            return

        # Deduplicate by semantic identity only: (source, content, target).
        # Timestamp is intentionally ignored.
        seen_edges = {
            (e.source, e.content, e.target)
            for e in self.edges.values()
            if e is not None
        }
        parsed_edges = []

        for item in triples:
            if not isinstance(item, TimeTriple):
                continue
            timestamp = item.time
            raw_triple = item.triple
            if not isinstance(timestamp, str) or len(timestamp) != 6 or not timestamp.isdigit():
                continue
            if not isinstance(raw_triple, list) or len(raw_triple) < 3:
                continue

            source_str = raw_triple[0]
            edge_content = raw_triple[1]
            target_str = raw_triple[2]

            if source_str is None or edge_content is None:
                continue
            edge_content = str(edge_content).strip()
            if not edge_content:
                continue

            # Parse source node.
            is_char_src, src_name = self._parse_node_string(source_str)
            if is_char_src:
                if src_name not in self.characters:
                    self.add_character(src_name)
                source_node_name = src_name
            else:
                _, source_node_name = self._get_or_create_object_node(src_name)

            # Parse target node.
            if target_str is None or (isinstance(target_str, str) and target_str.lower() == "null"):
                target_node_name = None
            else:
                is_char_tgt, tgt_name = self._parse_node_string(target_str)
                if is_char_tgt:
                    if tgt_name not in self.characters:
                        self.add_character(tgt_name)
                    target_node_name = tgt_name
                else:
                    _, target_node_name = self._get_or_create_object_node(tgt_name)

            edge_key = (source_node_name, edge_content, target_node_name)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            parsed_edges.append((timestamp, source_node_name, target_node_name, edge_content, raw_triple))

        if not parsed_edges:
            return

        edge_contents = [item[3] for item in parsed_edges]
        edge_embeddings = [None] * len(parsed_edges)
        try:
            batch_embeddings = get_multiple_embeddings(edge_contents)
            if len(batch_embeddings) == len(edge_embeddings):
                edge_embeddings = batch_embeddings
            else:
                print(
                    "Warning: embedding batch size mismatch in insert_triples: "
                    f"expected {len(edge_embeddings)}, got {len(batch_embeddings)}"
                )
        except Exception as e:
            print(f"Warning: batch edge embedding insertion failed in insert_triples: {e}")

        for (timestamp, source_node_name, target_node_name, edge_content, raw_triple), edge_embedding in zip(
            parsed_edges, edge_embeddings
        ):
            if edge_embedding is None:
                try:
                    edge_embedding = get_embedding(edge_content)
                except Exception as e:
                    print(f"Warning: failed to embed edge content '{edge_content}' at {timestamp}: {e}")

            edge = Edge(
                timestamp=timestamp,
                source=source_node_name,
                target=target_node_name,
                content=edge_content,
                embedding=edge_embedding,
            )

            try:
                self.add_edge(edge)
            except ValueError as e:
                print(f"Warning: {e}, skipping triple at {timestamp}: {raw_triple}")
                continue

        print(f"Inserted {len(parsed_edges)} triples into graph")

    def edges_of(self, node_id):
        return set(self.adjacency_list_out[node_id]) | set(self.adjacency_list_in[node_id])


    def edge_embedding_insertion(self):
        edge_contents = [edge.content for edge in self.edges.values()]
        embeddings = get_multiple_embeddings(edge_contents)
        for edge, embedding in zip(self.edges.values(), embeddings):
            edge.embedding = embedding
        print(len(embeddings), "edge embeddings inserted")

    def node_embedding_insertion(self, object_batch_size=100):
        """
        Generate embeddings in two phases:
        1) All characters in one batch (angle brackets removed)
        2) Objects in batches of `object_batch_size` (default 100)
        """
        # Phase 1: characters in one batch
        character_items = []
        character_texts = []
        for char_name, char_node in self.characters.items():
            if char_node.embedding is None:
                text = (
                    char_name.strip("<>")
                    if char_name.startswith("<") and char_name.endswith(">")
                    else char_name
                )
                if not (text or "").strip():
                    continue
                character_items.append((char_name, char_node))
                character_texts.append(text.strip())

        inserted_characters = 0
        if character_items:
            try:
                char_embeddings = get_multiple_embeddings(character_texts)
                for (_, char_node), emb in zip(character_items, char_embeddings):
                    char_node.embedding = emb
                    inserted_characters += 1
            except Exception as e:
                print(f"Warning: Failed character batch embedding insertion: {e}")
                for (char_name, char_node), text in zip(character_items, character_texts):
                    try:
                        char_node.embedding = get_embedding(text)
                        inserted_characters += 1
                    except Exception as e2:
                        print(f"Warning: Failed character embedding for {char_name}: {e2}")

        # Phase 2: objects in batches
        object_items = []
        object_texts = []
        for obj_name, obj_node in self.objects.items():
            if obj_node.embedding is None:
                if obj_name is None or not str(obj_name).strip():
                    continue
                object_items.append((obj_name, obj_node))
                object_texts.append(str(obj_name).strip())

        inserted_objects = 0
        if object_items:
            batch_size = max(1, int(object_batch_size))
            for start in range(0, len(object_items), batch_size):
                chunk_items = object_items[start : start + batch_size]
                chunk_texts = object_texts[start : start + batch_size]
                try:
                    obj_embeddings = get_multiple_embeddings(chunk_texts)
                    for (_, obj_node), emb in zip(chunk_items, obj_embeddings):
                        obj_node.embedding = emb
                        inserted_objects += 1
                except Exception as e:
                    print(
                        "Warning: Failed object batch embedding insertion "
                        f"[{start}:{start + len(chunk_items)}]: {e}"
                    )
                    for (obj_name, obj_node), text in zip(chunk_items, chunk_texts):
                        try:
                            obj_node.embedding = get_embedding(text)
                            inserted_objects += 1
                        except Exception as e2:
                            print(f"Warning: Failed object embedding for {obj_name}: {e2}")

        if inserted_characters == 0 and inserted_objects == 0:
            print("No nodes need embedding generation")
            return
        print(
            f"{inserted_characters + inserted_objects} node embeddings inserted "
            f"({inserted_characters} characters, {inserted_objects} objects)"
        )
    

    # --------------------------------------------------------
    # Graph Summary
    # --------------------------------------------------------
    def graph_summary(self) -> str:
        """
        Return a compact textual summary of graph statistics.
        Includes:
        1) Number of behavior triples
        2) Number of conversations
        3) Characters with degree > 50 (name and degree)
        4) Number of object nodes
        """
        num_edges = len(self.edges)
        num_conversations = len(self.conversations)
        num_objects = len(self.objects)

        node_degrees = self.get_node_degrees()
        high_degree_characters = []
        for char_name in self.characters.keys():
            degree = node_degrees.get(char_name, 0)
            if degree > 50:
                high_degree_characters.append((char_name, degree))

        high_degree_characters.sort(key=lambda x: x[1], reverse=True)

        rtn_str = "Graph Summary:\nThe graph is the combination of behavior (stored as triples) and conversation memory.\nThere are two types of nodes in the graph: characters (with angle brackets) and objects. The edges in the graph are the relationships between the nodes."
        rtn_str += "\nGraph time range: 11:00-15:00, 17:00-23:00."

        rtn_str += f"\nNumber of behavior triples: {num_edges}"
        rtn_str += f"\nNumber of conversation messages: {num_conversations}"
        rtn_str += "\nHigh-degree characters:"
        if high_degree_characters:
            for char_name, degree in high_degree_characters:
                rtn_str += f"\n - {char_name}: {degree}"
        else:
            rtn_str += "\n - None"
        rtn_str += f"\nNumber of object nodes: {num_objects}"
        return rtn_str

    # --------------------------------------------------------
    # Search API
    # --------------------------------------------------------
    def _cosine_similarity(self, vec1, vec2):
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector (list or numpy array)
            vec2: Second vector (list or numpy array)
        
        Returns:
            float: Cosine similarity score between -1 and 1
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _get_node_embedding(self, node_str):
        """
        Get stored embedding for a node string if available.
        Both CharacterNode and ObjectNode embeddings are stored during initialization.
        
        Args:
            node_str: Node string representation (e.g., "<Alice>", "coffee")
        
        Returns:
            Embedding vector if stored, None otherwise
        """
        if node_str is None:
            return None
        
        node_str = str(node_str).strip()
        
        # Check if it's a character node
        if node_str.startswith("<") and node_str.endswith(">"):
            char_node = self.get_character(node_str)
            if char_node is not None and hasattr(char_node, 'embedding') and char_node.embedding is not None:
                return char_node.embedding
            return None
        
        # Object nodes have stored embeddings
        obj_node = self.get_object_node(node_str)
        if obj_node is not None and hasattr(obj_node, 'embedding') and obj_node.embedding is not None:
            return obj_node.embedding
        
        return None
    
    def _calculate_node_similarity(self, node1_str, node2_str, node1_embedding, node2_embedding):
        """
        Calculate similarity between two nodes.
        Handles edge cases: None, "?", and missing embeddings.
        
        Args:
            node1_str: First node string (can be None or "?")
            node2_str: Second node string (can be None or "?")
            node1_embedding: Embedding for node1 (can be None)
            node2_embedding: Embedding for node2 (can be None)
        
        Returns:
            float: Similarity score between 0 and 1
        """
        # Handle None or "?" cases
        if node1_str is None or node1_str == "?" or node2_str is None or node2_str == "?":
            return 0.0
        
        # Convert to string and strip
        node1_str = str(node1_str).strip()
        node2_str = str(node2_str).strip()
        
        # Handle empty strings
        if not node1_str or not node2_str:
            return 0.0
        
        # Character-Character matching: exact name match
        if node1_str.startswith("<") and node1_str.endswith(">") and node2_str.startswith("<") and node2_str.endswith(">"):
            return 1.0 if node1_str == node2_str else 0.0
        
        # For Object-Object and Object-Character: use cosine similarity
        # Need both embeddings to be available
        if node1_embedding is None or node2_embedding is None:
            return 0.0
        
        try:
            return self._cosine_similarity(node1_embedding, node2_embedding)
        except Exception:
            return 0.0

    
    def _compute_edge_similarity(self, edge, query_triple, query_embeddings):
        """
        Compute the similarity between an edge and a query triple.
        Handles edge cases: None edge, None/empty query_triple, "?" values, missing embeddings.

        Args:
            edge: Edge object (can be None)
            query_triple: [source, content, target, source_weight, content_weight, target_weight] (can contain None or "?")
            query_embeddings: [source_embedding, content_embedding, target_embedding] (can contain None)
        
        Returns:
            float: Similarity score between the edge and the query triple (0.0 if edge cases)
        """
        # Handle None edge
        if edge is None:
            return 0.0
        
        # Handle None or invalid query_triple
        if query_triple is None or not isinstance(query_triple, (list, tuple)) or len(query_triple) < 6:
            return 0.0
        
        # Handle None or invalid query_embeddings
        if query_embeddings is None or not isinstance(query_embeddings, (list, tuple)) or len(query_embeddings) < 3:
            return 0.0
        
        # Extract query components
        q_source = query_triple[0]
        q_content = query_triple[1]
        q_target = query_triple[2]
        def _to_float_weight(v, default=1.0):
            if v is None:
                return float(default)
            try:
                return float(v)
            except Exception:
                return float(default)

        q_source_weight = _to_float_weight(query_triple[3], 1.0)
        q_content_weight = _to_float_weight(query_triple[4], 1.0)
        q_target_weight = _to_float_weight(query_triple[5], 1.0)
        
        # Extract embeddings
        source_emb = query_embeddings[0]
        content_emb = query_embeddings[1]
        target_emb = query_embeddings[2]
        
        # Content similarity (handle None/empty content and missing embeddings)
        content_sim = 0.0
        if q_content and q_content != "?" and edge.content and content_emb is not None:
            if edge.embedding is not None:
                try:
                    content_sim = self._cosine_similarity(content_emb, edge.embedding) * q_content_weight
                except Exception:
                    # Fallback to exact match
                    if edge.content == q_content:
                        content_sim = q_content_weight
        
        # Normal direction: (query source, edge source) and (query target, edge target)
        normal_q_source_sim = 0.0
        normal_q_target_sim = 0.0
        if q_source and q_source != "?" and edge.source is not None:
            edge_source_emb = self._get_node_embedding(edge.source)
            normal_q_source_sim = self._calculate_node_similarity(q_source, edge.source, source_emb, edge_source_emb) * q_source_weight
        
        if q_target and q_target != "?" and edge.target is not None:
            edge_target_emb = self._get_node_embedding(str(edge.target))
            normal_q_target_sim = self._calculate_node_similarity(q_target, str(edge.target), target_emb, edge_target_emb) * q_target_weight
        
        # Reversed direction: (query source, edge target) and (query target, edge source)
        reversed_q_source_sim = 0.0
        reversed_q_target_sim = 0.0
        if q_source and q_source != "?" and edge.target is not None:
            edge_target_emb = self._get_node_embedding(str(edge.target))
            reversed_q_source_sim = self._calculate_node_similarity(q_source, str(edge.target), source_emb, edge_target_emb) * q_source_weight
        
        if q_target and q_target != "?" and edge.source is not None:
            edge_source_emb = self._get_node_embedding(edge.source)
            reversed_q_target_sim = self._calculate_node_similarity(q_target, edge.source, target_emb, edge_source_emb) * q_target_weight
        
        # Return the maximum of normal and reversed directions plus content similarity
        return content_sim + max(normal_q_source_sim + normal_q_target_sim, reversed_q_source_sim + reversed_q_target_sim)


    def search_edges(self, query_triples, k):

        if not query_triples:
            return []
        
        # Normalize query_triples to list of lists
        # Filter out None values
        query_triples = [q for q in query_triples if q is not None]
        if not query_triples:
            return []
        if isinstance(query_triples[0], str):
            query_triples = [query_triples]
        
        # Pre-compute query embeddings for each triple component
        # Store as list per triple: [source_emb, content_emb, target_emb]
        query_triple_embeddings = []
        for q_triple in query_triples:
            if q_triple is None:
                query_triple_embeddings.append([None, None, None])
                continue
            
            q_source = q_triple[0] if isinstance(q_triple, (list, tuple)) and len(q_triple) > 0 else None
            q_content = q_triple[1] if isinstance(q_triple, (list, tuple)) and len(q_triple) > 1 else None
            q_target = q_triple[2] if isinstance(q_triple, (list, tuple)) and len(q_triple) > 2 else None
            
            # Compute embeddings using the shared helper.
            query_triple_embeddings.append(
                [
                    self._compute_embed(q_source),
                    self._compute_embed(q_content),
                    self._compute_embed(q_target),
                ]
            )
        
        candidate_edges = []
        for edge_id, edge in self.edges.items():
            candidate_edges.append(edge)
        
        if not candidate_edges:
            return []
        
        # Score edges based on embedding similarity with bidirectional matching
        # Formula: Similarity = (weight_source*source + weight_content*content + weight_target*target)
        scored_edges = []
        for edge in candidate_edges:
            base_similarity = 0.0
            
            # Match against each query triple using embeddings
            for i, q_triple in enumerate(query_triples):
                if q_triple is None:
                    continue
                
                # Extract weights (default to 1.0 if not provided)
                q_source_weight = q_triple[3] if isinstance(q_triple, (list, tuple)) and len(q_triple) > 3 and q_triple[3] is not None else 1.0
                q_content_weight = q_triple[4] if isinstance(q_triple, (list, tuple)) and len(q_triple) > 4 and q_triple[4] is not None else 1.0
                q_target_weight = q_triple[5] if isinstance(q_triple, (list, tuple)) and len(q_triple) > 5 and q_triple[5] is not None else 1.0
                
                # Prepare query triple with provided weights (no normalization)
                query_triple_with_weights = [
                    q_triple[0] if isinstance(q_triple, (list, tuple)) and len(q_triple) > 0 else None,
                    q_triple[1] if isinstance(q_triple, (list, tuple)) and len(q_triple) > 1 else None,
                    q_triple[2] if isinstance(q_triple, (list, tuple)) and len(q_triple) > 2 else None,
                    q_source_weight,
                    q_content_weight,
                    q_target_weight
                ]
                
                # Compute similarity (edge cases handled in _compute_edge_similarity)
                query_embeddings = query_triple_embeddings[i]
                triple_similarity = self._compute_edge_similarity(edge, query_triple_with_weights, query_embeddings)
                base_similarity = max(base_similarity, triple_similarity)  # Keep max across all query triples
            
            # Final score: base_similarity
            score = base_similarity
            
            scored_edges.append((score, edge))
        
        # Sort by score (descending) and return top-k
        scored_edges.sort(key=lambda x: x[0], reverse=True)
        return [edge for _, edge in scored_edges[:k]]
    
    
    def search_conversations(self, query, k, speaker_strict=None):
        """
        Search for top-k conversation messages using embedding-based similarity.
        
        Args:
            query: either a natural-language string OR a query triple/list-of-triples.
                   If a triple is provided, we combine (source, content, target) into one sentence
                   (same style as `_search_temporal_range`) before embedding.
            k: Number of top messages to return
            speaker_strict: Optional list of speakers to filter by (e.g., ["<Alice>", "<Bob>"])
                          Only return conversations where ALL specified speakers are present
        
        Returns:
            list: List of dictionaries with format:
                {
                    "conversation_id": int,
                    "message_index": int,
                    "score": float
                }
        """
        if not query:
            return []

        # Allow query to be a triple: [source, content, target, w_source, w_content, w_target]
        # or list-of-triples: [[...6], [...6], ...]. We use the first triple to form a sentence.
        query_text = None
        if isinstance(query, str):
            query_text = query
        elif isinstance(query, (list, tuple)):
            triple = None
            if len(query) == 6 and not isinstance(query[0], (list, tuple)):
                triple = query
            elif len(query) > 0 and isinstance(query[0], (list, tuple)) and len(query[0]) >= 3:
                triple = query[0]

            if triple is not None:
                q_source = triple[0] if len(triple) > 0 else None
                q_content = triple[1] if len(triple) > 1 else None
                q_target = triple[2] if len(triple) > 2 else None

                s_source = self._format_node_for_sentence(q_source) or ""
                s_target = self._format_node_for_sentence(q_target) or ""
                s_content = (q_content or "").strip() if isinstance(q_content, str) else str(q_content or "").strip()

                query_text = " ".join(
                    part for part in [s_source, s_content, s_target] if part and part != "?"
                ).strip()
                if not query_text:
                    query_text = s_content if s_content else None

        if not isinstance(query_text, str) or not query_text.strip():
            return []
        
        # Get embedding for query
        try:
            query_embedding = get_embedding(query_text)
        except Exception as e:
            print(f"Warning: Failed to get query embedding: {e}")
            return []
        
        # Search through all conversations
        scored_messages = []
        
        for conv_id, conversation in self.conversations.items():
            # Filter by speaker_strict if provided
            if speaker_strict:
                # Normalize speaker names (add angle brackets if needed)
                normalized_speakers = set()
                for speaker in speaker_strict:
                    if not speaker.startswith("<") or not speaker.endswith(">"):
                        normalized_speakers.add(f"<{speaker}>")
                    else:
                        normalized_speakers.add(speaker)
                
                # Check if ALL specified speakers are in this conversation
                if not normalized_speakers.issubset(conversation.speakers):
                    continue
            
            # Search through messages in this conversation
            for msg_idx, message in enumerate(conversation.messages):
                if not isinstance(message, list) or len(message) < 3:
                    continue
                
                speaker = message[1]
                content = message[2]
                
                if not content or not isinstance(content, str):
                    continue
                
                # Use stored embedding (index 3) - embeddings are pre-computed when messages are added
                try:
                    if len(message) >= 4 and message[3] is not None:
                        message_embedding = message[3]  # Use stored embedding from message (pre-computed)
                    else:
                        # Fallback: compute embedding if not stored (shouldn't happen normally)
                        # Remove angle brackets from speaker name for embedding consistency
                        speaker_name = speaker
                        if speaker_name.startswith("<") and speaker_name.endswith(">"):
                            speaker_name = speaker_name[1:-1]
                        formatted_message = f"{speaker_name}: {content}"
                        message_embedding = get_embedding(formatted_message)
                    
                    text_similarity = self._cosine_similarity(query_embedding, message_embedding)
                except Exception:
                    # Fallback to keyword matching
                    formatted_message = f"{speaker}: {content}"
                    formatted_lower = formatted_message.lower()
                    query_lower = query_text.lower()
                    if query_lower in formatted_lower or any(word in formatted_lower for word in query_lower.split()):
                        text_similarity = 0.5
                    else:
                        text_similarity = 0.0
                
                # Calculate final score
                score = text_similarity
                
                # Only include messages with positive score
                if score > 0:
                    scored_messages.append({
                        "conversation_id": conv_id,
                        "message_index": msg_idx,
                        "score": score
                    })
        
        # Sort by score (descending) and return top-k
        scored_messages.sort(key=lambda x: x["score"], reverse=True)
        return scored_messages[:k]
    

    def get_conversation_messages_with_context(self, search_results, context_window=2):
        """
        Given the output of search_conversations(), return messages with context window in temporal order.
        Merges overlapping message ranges to avoid duplicates.
        
        Args:
            search_results: List of dictionaries from search_conversations() with format:
                {
                    "conversation_id": int,
                    "message_index": int,
                    "score": float
                }
            context_window: Number of messages before and after to include for context (default: 2)
        
        Returns:
            str: Formatted string with conversation summaries and messages.
                Format: "Conversation 1: Summary of the conversation. \nAnna: ... \nSusan: ...\n\nConversation 2: ..."
        """
        if not search_results:
            return ""
        
        # Group results by conversation_id
        conversation_indices = {}
        for result in search_results:
            conv_id = result.get("conversation_id")
            msg_idx = result.get("message_index")
            if conv_id is None or msg_idx is None:
                continue
            
            if conv_id not in conversation_indices:
                conversation_indices[conv_id] = []
            conversation_indices[conv_id].append(msg_idx)
        
        # Process each conversation and build formatted string
        formatted_conversations = []
        
        for conv_id, message_indices in conversation_indices.items():
            # Get the conversation
            conversation = self.conversations.get(conv_id)
            if conversation is None:
                continue
            
            if not conversation.messages:
                continue
            
            # Merge overlapping ranges
            # Create ranges with context window for each matched message
            ranges = []
            for msg_idx in message_indices:
                start_idx = max(0, msg_idx - context_window)
                end_idx = min(len(conversation.messages), msg_idx + context_window + 1)
                ranges.append((start_idx, end_idx))
            
            # Sort ranges by start index
            ranges.sort(key=lambda x: x[0])
            
            # Merge overlapping ranges
            merged_ranges = []
            if ranges:
                merged_start, merged_end = ranges[0]
                for start, end in ranges[1:]:
                    if start <= merged_end:
                        # Overlapping or adjacent - merge
                        merged_end = max(merged_end, end)
                    else:
                        # Non-overlapping - save current and start new
                        merged_ranges.append((merged_start, merged_end))
                        merged_start, merged_end = start, end
                # Add the last range
                merged_ranges.append((merged_start, merged_end))
            
            # Extract messages from merged ranges
            all_message_indices = set()
            for start, end in merged_ranges:
                all_message_indices.update(range(start, end))
            
            # Sort indices to maintain temporal order
            sorted_indices = sorted(all_message_indices)
            
            message_lines = []
            for idx in sorted_indices:
                if idx < len(conversation.messages):
                    msg = conversation.messages[idx]
                    if isinstance(msg, list) and len(msg) >= 3:
                        start_time = msg[0]
                        speaker = msg[1]
                        content = msg[2]
                        
                        # Remove angle brackets from speaker name
                        speaker_name = speaker
                        if speaker_name.startswith("<") and speaker_name.endswith(">"):
                            speaker_name = speaker_name[1:-1]
                        
                        # Format with start_time: [hh:mm:ss] Speaker: content
                        if start_time is not None:
                            ts = str(start_time).strip()
                            if len(ts) == 6 and ts.isdigit():
                                ts = f"{ts[0:2]}:{ts[2:4]}:{ts[4:6]}"
                            message_lines.append(f"[{ts}] {speaker_name}: {content}")
                        else:
                            message_lines.append(f"{speaker_name}: {content}")
            
            if message_lines:
                # Get conversation summary (if available)
                summary = conversation.summary if hasattr(conversation, 'summary') and conversation.summary else ""
                
                # Format: "Conversation {id}: {summary}\n{message1}\n{message2}..."
                if summary:
                    conversation_text = f"Conversation {conv_id}: {summary}\n" + "\n".join(message_lines)
                else:
                    conversation_text = f"Conversation {conv_id}:\n" + "\n".join(message_lines)
                
                formatted_conversations.append(conversation_text)
        
        # Join all conversations with double newline separator
        return "\n\n".join(formatted_conversations)
    

    def general_search(
        self,
        query_triples: list[list[str | float | None]],
        k_behavior: int,
        k_conversation: int,
        speaker_strict: list[str] | None = None,
    ):
        """
        General graph search over:
        - behavior edges (triple similarity)
        - conversation messages (query sentence built from source/content/target)
        
        Args:
            query_triples: List of query triples
            k_behavior: Number of top behavior edges to return
            k_conversation: Number of top conversation messages to return
            speaker_strict: optional list of speakers for conversation filtering
        """
        behavior_edges = self.search_edges(query_triples, k_behavior) if k_behavior > 0 else []
        conversation_results = (
            self.search_conversations(query_triples, k_conversation, speaker_strict=speaker_strict)
            if k_conversation > 0
            else []
        )
        return {"behavior": behavior_edges, "conversation": self.get_conversation_messages_with_context(conversation_results)}

    def search_object(self, obj: str) -> str:
        """
        Search graph memory with an object string.

        Steps:
        1) Embedding similarity over object nodes, return top-50 most similar object names.
        2) Hard match (word-to-word, case-insensitive) in conversation summaries/messages.
           If matched, return the full conversation(s).

        Returns:
            str: formatted search report
        """
        query = "" if obj is None else str(obj).strip()
        if not query:
            return "Object search query is empty."

        # ---- Part 1: Top-50 objects ranked by similarity, displayed with degree ----
        object_lines = []
        try:
            query_emb = get_embedding(query)
            scored_objects = []
            for obj_name, obj_node in self.objects.items():
                emb = getattr(obj_node, "embedding", None)
                if emb is None:
                    continue
                try:
                    sim = float(self._cosine_similarity(query_emb, emb))
                except Exception:
                    continue
                scored_objects.append((sim, obj_name))
            scored_objects.sort(key=lambda x: x[0], reverse=True)
            top_objects = scored_objects[:50]
            node_degrees = self.get_node_degrees()
            if top_objects:
                object_lines.append("Top-50 object nodes (ranked by similarity, shown with degree):")
                for _, obj_name in top_objects:
                    degree = node_degrees.get(obj_name, 0)
                    object_lines.append(f"- {obj_name}: degree={degree}")
            else:
                object_lines.append("Top-50 object nodes (ranked by similarity, shown with degree):\n- None")
        except Exception as e:
            object_lines.append(
                f"Top-50 object nodes (ranked by similarity, shown with degree):\n- Failed to compute embeddings: {e}"
            )

        # ---- Part 2: Hard match in conversation summaries/messages ----
        # Word-boundary hard match, case-insensitive.
        pattern = re.compile(rf"\b{re.escape(query)}\b", flags=re.IGNORECASE)
        matched_conv_ids = []
        for conv_id, conversation in self.conversations.items():
            summary = conversation.summary if hasattr(conversation, "summary") and conversation.summary else ""
            matched = bool(pattern.search(summary))
            if not matched:
                for msg in getattr(conversation, "messages", []) or []:
                    if not isinstance(msg, list) or len(msg) < 3:
                        continue
                    content = msg[2]
                    if isinstance(content, str) and pattern.search(content):
                        matched = True
                        break
            if matched:
                matched_conv_ids.append(conv_id)

        matched_conv_ids.sort()
        conversation_lines = ["Matched conversations (hard match in summary/messages):"]
        if not matched_conv_ids:
            conversation_lines.append("- None")
        else:
            for conv_id in matched_conv_ids:
                conversation = self.conversations.get(conv_id)
                if conversation is None:
                    continue
                summary = conversation.summary if hasattr(conversation, "summary") and conversation.summary else ""
                if summary:
                    conversation_lines.append(f"Conversation {conv_id}: {summary}")
                else:
                    conversation_lines.append(f"Conversation {conv_id}:")

                for msg in getattr(conversation, "messages", []) or []:
                    if not isinstance(msg, list) or len(msg) < 3:
                        continue
                    start_time = msg[0]
                    speaker = msg[1]
                    content = msg[2]
                    ts = str(start_time).strip() if start_time is not None else ""
                    if len(ts) == 6 and ts.isdigit():
                        ts = f"{ts[0:2]}:{ts[2:4]}:{ts[4:6]}"
                        conversation_lines.append(f"[{ts}] {speaker}: {content}")
                    else:
                        conversation_lines.append(f"{speaker}: {content}")
                conversation_lines.append("")

        header = f"Object search query: {query}"
        return "\n".join([header, "", *object_lines, "", *conversation_lines]).strip()


    def _format_node_for_sentence(self, node_str: str | None) -> str | None:
        """
        Normalize a node label for use inside a natural-language sentence.
        - Character nodes are stored as "<Alice>" etc; strip angle brackets.
        - Objects are stored as plain strings; keep as-is.
        """
        if node_str is None:
            return None
        s = str(node_str).strip()
        if not s:
            return None
        if s.startswith("<") and s.endswith(">"):
            return s[1:-1].strip()
        return s

    def _parse_hhmmss_int(self, hhmmss: str) -> int:
        ts = str(hhmmss).strip()
        if len(ts) != 6 or not ts.isdigit():
            raise ValueError(f"Expected hhmmss timestamp as 6 digits, got: {hhmmss!r}")
        return int(ts)

    def _compute_embed(self, node):
        """
        Compute an embedding for a node-like string used in triple matching.
        Returns None for None / "?" / empty strings.

        - Character nodes are stored as "<Alice>" -> embed "Alice" (strip angle brackets).
        - Object nodes are plain strings -> embed as-is (trim whitespace).
        """
        if node is None:
            return None
        if not isinstance(node, str):
            node = str(node)
        node = node.strip()
        if not node or node == "?":
            return None
        if node.startswith("<") and node.endswith(">"):
            node = node[1:-1].strip()
        if not node:
            return None
        return get_embedding(node)

    def search_within_time_range(
        self,
        begin_time: str | None,
        end_time: str | None,
        triples: list,
        k_behavior: int,
        k_conversation: int,
    ):
        """
        Search for evidence between begin/end bounds (inclusive).
        Either boundary may be None (one-sided search).

        Args:
            begin_time: hhmmss string (6 digits) or None
            end_time: hhmmss string (6 digits) or None
            triples: [source, content, target, weight_source, weight_content, weight_target]
            k_behavior: number of top behavior edges to return
            k_conversation: number of top conversation messages to return

        Returns:
            dict with:
              - "behavior": list[Edge]
              - "conversation": formatted conversation context string
        """
        return self._search_temporal_range(
            begin_time=begin_time,
            end_time=end_time,
            triples=triples,
            k_behavior=k_behavior,
            k_conversation=k_conversation,
        )

    # Backwards-compatible wrappers (implemented via search_within_time_range).
    def search_before(self, timestamp: str, triples: list, k_behavior: int, k_conversation: int):
        return self.search_within_time_range(
            begin_time=None,
            end_time=timestamp,
            triples=triples,
            k_behavior=k_behavior,
            k_conversation=k_conversation,
        )

    def search_after(self, timestamp: str, triples: list, k_behavior: int, k_conversation: int):
        return self.search_within_time_range(
            begin_time=timestamp,
            end_time=None,
            triples=triples,
            k_behavior=k_behavior,
            k_conversation=k_conversation,
        )

    def _search_temporal_range(
        self,
        *,
        begin_time: str | None,
        end_time: str | None,
        triples: list,
        k_behavior: int,
        k_conversation: int,
    ):
        anchor_begin = self._parse_hhmmss_int(begin_time) if begin_time is not None else None
        anchor_end = self._parse_hhmmss_int(end_time) if end_time is not None else None

        # ---- Parse query triple + weights ----
        if not isinstance(triples, (list, tuple)) or len(triples) != 6:
            raise ValueError(
                "triples must be a list of 6 elements: "
                "[source, content, target, weight_source, weight_content, weight_target]"
            )
        q_source, q_content, q_target, w_source, w_content, w_target = triples
        w_source = float(w_source) if w_source is not None else 1.0
        w_content = float(w_content) if w_content is not None else 1.0
        w_target = float(w_target) if w_target is not None else 1.0

        # ---- Behavior (edge) search ----
        behavior_out: list[Edge] = []
        if k_behavior and k_behavior > 0:
            # Pre-compute query embeddings; skip wildcards ("?") to save API calls.
            q_source_emb = self._compute_embed(q_source)
            q_content_emb = self._compute_embed(q_content)
            q_target_emb = self._compute_embed(q_target)

            query_triple_with_weights = [q_source, q_content, q_target, w_source, w_content, w_target]
            query_embeddings = [q_source_emb, q_content_emb, q_target_emb]

            scored_edges: list[tuple[float, Edge]] = []
            for edge in self.edges.values():
                if edge is None or not isinstance(edge.timestamp, str):
                    continue
                try:
                    t = int(edge.timestamp)
                except Exception:
                    continue
                if anchor_begin is not None and not (t >= anchor_begin):
                    continue
                if anchor_end is not None and not (t <= anchor_end):
                    continue

                score = self._compute_edge_similarity(
                    edge,
                    query_triple=query_triple_with_weights,
                    query_embeddings=query_embeddings,
                )
                if score > 0:
                    scored_edges.append((score, edge))

            scored_edges.sort(key=lambda x: x[0], reverse=True)
            behavior_out = [e for _, e in scored_edges[:k_behavior]]

        # ---- Conversation message search ----
        conversation_out: list[dict] = []
        if k_conversation and k_conversation > 0:
            s_source = self._format_node_for_sentence(q_source) or ""
            s_target = self._format_node_for_sentence(q_target) or ""
            s_content = (q_content or "").strip() if isinstance(q_content, str) else str(q_content or "").strip()

            # Build a single query sentence; keep it simple and consistent.
            # Example: "Alice hits Tasha"
            query_sentence = " ".join(part for part in [s_source, s_content, s_target] if part and part != "?")
            if not query_sentence:
                query_sentence = s_content if s_content else "?"

            query_embedding = get_embedding(query_sentence)

            scored_msgs: list[dict] = []
            for conv_id, conversation in self.conversations.items():
                for msg_idx, message in enumerate(getattr(conversation, "messages", []) or []):
                    if not isinstance(message, list) or len(message) < 3:
                        continue
                    msg_time = message[0]
                    if msg_time is None:
                        continue
                    try:
                        msg_t = int(str(msg_time).strip())
                    except Exception:
                        continue
                    if anchor_begin is not None and not (msg_t >= anchor_begin):
                        continue
                    if anchor_end is not None and not (msg_t <= anchor_end):
                        continue

                    speaker = message[1]
                    content = message[2]
                    emb = message[3] if len(message) >= 4 else None
                    if emb is None:
                        # Fallback embedding; align with how build_memory computed it: "Speaker: content"
                        speaker_name = self._format_node_for_sentence(speaker) or ""
                        emb = get_embedding(f"{speaker_name}: {content}")

                    sim = float(self._cosine_similarity(query_embedding, emb))
                    # Lightweight weight usage: emphasize content weight if provided.
                    sim *= (w_content if w_content is not None else 1.0)
                    if sim > 0:
                        scored_msgs.append(
                            {
                                "conversation_id": conv_id,
                                "message_index": msg_idx,
                                "score": sim,
                            }
                        )

            scored_msgs.sort(key=lambda x: x["score"], reverse=True)
            conversation_out = scored_msgs[:k_conversation]

        return {
            "behavior": behavior_out,
            "conversation": self.get_conversation_messages_with_context(conversation_out),
        }

    def _normalize_query_triples(self, triples: list):
        """
        Accept either:
        - a single triple: [source, content, target, w_source, w_content, w_target]
        - a list of triples: [[...6], [...6], ...]
        """
        if not isinstance(triples, (list, tuple)):
            raise ValueError("triples must be a list")
        if len(triples) == 6 and (not isinstance(triples[0], (list, tuple))):
            return [triples]
        # list-of-triples
        if len(triples) > 0 and isinstance(triples[0], (list, tuple)) and len(triples[0]) == 6:
            return list(triples)
        raise ValueError(
            "triples must be either a single 6-element triple or a list of 6-element triples"
        )

    def search_first(self, triples: list, k_behavior: int, k_conversation: int):
        """
        Scan from the beginning (earliest timestamps) and return items whose
        matching score is >= confidence_level, until k is reached.

        Returns: {"behavior": list[Edge], "conversation": formatted context string}
        """
        confidence_level = 0.5
        query_triples = self._normalize_query_triples(triples)

        # Pre-compute behavior query embeddings and weighted triples.
        behavior_query = []
        for q in query_triples:
            q_source, q_content, q_target, w_source, w_content, w_target = q
            behavior_query.append(
                (
                    [q_source, q_content, q_target, float(w_source or 1.0), float(w_content or 1.0), float(w_target or 1.0)],
                    [self._compute_embed(q_source), self._compute_embed(q_content), self._compute_embed(q_target)],
                )
            )

        behavior_out: list[Edge] = []
        if k_behavior and k_behavior > 0:
            edges_sorted = []
            for edge in self.edges.values():
                if edge is None or not isinstance(edge.timestamp, str):
                    continue
                try:
                    t = int(edge.timestamp)
                except Exception:
                    continue
                edges_sorted.append((t, edge))
            edges_sorted.sort(key=lambda x: x[0])  # earliest first

            for _, edge in edges_sorted:
                best_score = 0.0
                for q_with_weights, q_embeddings in behavior_query:
                    score = self._compute_edge_similarity(
                        edge,
                        query_triple=q_with_weights,
                        query_embeddings=q_embeddings,
                    )
                    if score > best_score:
                        best_score = score
                if best_score >= confidence_level:
                    behavior_out.append(edge)
                    if len(behavior_out) >= k_behavior:
                        break

        # Pre-compute conversation query embeddings.
        conversation_query = []
        for q in query_triples:
            q_source, q_content, q_target, w_source, w_content, w_target = q
            s_source = self._format_node_for_sentence(q_source) or ""
            s_target = self._format_node_for_sentence(q_target) or ""
            s_content = (q_content or "").strip() if isinstance(q_content, str) else str(q_content or "").strip()
            query_sentence = " ".join(part for part in [s_source, s_content, s_target] if part and part != "?")
            if not query_sentence:
                query_sentence = s_content if s_content else "?"
            conversation_query.append((get_embedding(query_sentence), float(w_content or 1.0)))

        conversation_out: list[dict] = []
        if k_conversation and k_conversation > 0:
            messages_sorted = []
            for conv_id, conversation in self.conversations.items():
                for msg_idx, message in enumerate(getattr(conversation, "messages", []) or []):
                    if not isinstance(message, list) or len(message) < 3:
                        continue
                    msg_time = message[0]
                    if msg_time is None:
                        continue
                    try:
                        msg_t = int(str(msg_time).strip())
                    except Exception:
                        continue
                    messages_sorted.append((msg_t, conv_id, msg_idx, message))
            messages_sorted.sort(key=lambda x: x[0])  # earliest first

            for _, conv_id, msg_idx, message in messages_sorted:
                emb = message[3] if len(message) >= 4 else None
                if emb is None:
                    speaker_name = self._format_node_for_sentence(message[1]) or ""
                    emb = get_embedding(f"{speaker_name}: {message[2]}")

                best_sim = 0.0
                for q_emb, w_content in conversation_query:
                    sim = float(self._cosine_similarity(q_emb, emb)) * (w_content if w_content is not None else 1.0)
                    if sim > best_sim:
                        best_sim = sim
                if best_sim >= confidence_level:
                    conversation_out.append(
                        {
                            "conversation_id": conv_id,
                            "message_index": msg_idx,
                            "score": best_sim,
                        }
                    )
                    if len(conversation_out) >= k_conversation:
                        break

        return {
            "behavior": behavior_out,
            "conversation": self.get_conversation_messages_with_context(conversation_out),
        }

    def search_last(self, triples: list, k_behavior: int, k_conversation: int):
        """
        Scan from the end (latest timestamps) and return items whose
        matching score is >= confidence_level, until k is reached.

        Returns: {"behavior": list[Edge], "conversation": formatted context string}
        """
        confidence_level = 0.5
        query_triples = self._normalize_query_triples(triples)

        # Pre-compute behavior query embeddings and weighted triples.
        behavior_query = []
        for q in query_triples:
            q_source, q_content, q_target, w_source, w_content, w_target = q
            behavior_query.append(
                (
                    [q_source, q_content, q_target, float(w_source or 1.0), float(w_content or 1.0), float(w_target or 1.0)],
                    [self._compute_embed(q_source), self._compute_embed(q_content), self._compute_embed(q_target)],
                )
            )

        behavior_out: list[Edge] = []
        if k_behavior and k_behavior > 0:
            edges_sorted = []
            for edge in self.edges.values():
                if edge is None or not isinstance(edge.timestamp, str):
                    continue
                try:
                    t = int(edge.timestamp)
                except Exception:
                    continue
                edges_sorted.append((t, edge))
            edges_sorted.sort(key=lambda x: x[0], reverse=True)  # latest first

            for _, edge in edges_sorted:
                best_score = 0.0
                for q_with_weights, q_embeddings in behavior_query:
                    score = self._compute_edge_similarity(
                        edge,
                        query_triple=q_with_weights,
                        query_embeddings=q_embeddings,
                    )
                    if score > best_score:
                        best_score = score
                if best_score >= confidence_level:
                    behavior_out.append(edge)
                    if len(behavior_out) >= k_behavior:
                        break

        # Pre-compute conversation query embeddings.
        conversation_query = []
        for q in query_triples:
            q_source, q_content, q_target, w_source, w_content, w_target = q
            s_source = self._format_node_for_sentence(q_source) or ""
            s_target = self._format_node_for_sentence(q_target) or ""
            s_content = (q_content or "").strip() if isinstance(q_content, str) else str(q_content or "").strip()
            query_sentence = " ".join(part for part in [s_source, s_content, s_target] if part and part != "?")
            if not query_sentence:
                query_sentence = s_content if s_content else "?"
            conversation_query.append((get_embedding(query_sentence), float(w_content or 1.0)))

        conversation_out: list[dict] = []
        if k_conversation and k_conversation > 0:
            messages_sorted = []
            for conv_id, conversation in self.conversations.items():
                for msg_idx, message in enumerate(getattr(conversation, "messages", []) or []):
                    if not isinstance(message, list) or len(message) < 3:
                        continue
                    msg_time = message[0]
                    if msg_time is None:
                        continue
                    try:
                        msg_t = int(str(msg_time).strip())
                    except Exception:
                        continue
                    messages_sorted.append((msg_t, conv_id, msg_idx, message))
            messages_sorted.sort(key=lambda x: x[0], reverse=True)  # latest first

            for _, conv_id, msg_idx, message in messages_sorted:
                emb = message[3] if len(message) >= 4 else None
                if emb is None:
                    speaker_name = self._format_node_for_sentence(message[1]) or ""
                    emb = get_embedding(f"{speaker_name}: {message[2]}")

                best_sim = 0.0
                for q_emb, w_content in conversation_query:
                    sim = float(self._cosine_similarity(q_emb, emb)) * (w_content if w_content is not None else 1.0)
                    if sim > best_sim:
                        best_sim = sim
                if best_sim >= confidence_level:
                    conversation_out.append(
                        {
                            "conversation_id": conv_id,
                            "message_index": msg_idx,
                            "score": best_sim,
                        }
                    )
                    if len(conversation_out) >= k_conversation:
                        break

        return {
            "behavior": behavior_out,
            "conversation": self.get_conversation_messages_with_context(conversation_out),
        }

    
    
