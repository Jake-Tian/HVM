import json
import re
import numpy as np
from .node_class import CharacterNode, ObjectNode
from .edge_class import Edge
from .conversation import Conversation
from .output_structure import ConversationSummary
from collections import defaultdict
from utils.prompts import prompt_character_summary, prompt_character_relationships, prompt_conversation_summary
from utils.llm_gpt import generate_text_response
from utils.embedding import get_embedding, get_multiple_embeddings
from utils.general import strip_code_fences
from utils.token_usage import empty_usage, merge_usage, usage_total


class HeteroGraph:
    def __init__(self):

        self.characters = {}   # name → CharacterNode object
        self.objects = {}   # name → ObjectNode object
        self.conversations = {}   # id → Conversation object
        self.edges = {}   # id → Edge object
        self.current_conversation_id = None  # Track the most recent conversation ID

        # Adjacency lists support constant-time lookup.
        self.adjacency_list_out = defaultdict(list)  # node → list of edge IDs (outgoing edges)
        self.adjacency_list_in = defaultdict(list)   # node → list of edge IDs (incoming edges)

        # Track evidence already consumed by incremental abstraction.
        self._char_consumed_edges = defaultdict(set)   # char name → set[edge_id]
        self._char_last_degree = defaultdict(int)     # char name → low-level degree at last summary
        self._pair_consumed_edges = defaultdict(set)  # (c1, c2) → set[edge_id]
        self._pair_last_degree = defaultdict(int)     # (c1, c2) → shared low-level count at last summary

        robot = CharacterNode("<robot>")
        self.characters[robot.name] = robot


    # --------------------------------------------------------
    # Node API
    # --------------------------------------------------------
    def add_character(self, name):
        if not name.startswith("<") or not name.endswith(">"):
            name = f"<{name}>"
        
        if name in self.characters:
            return name
        
        character = CharacterNode(name)
        self.characters[character.name] = character
        return character.name
    
    def get_character(self, name):
        """Get a character by name. Returns None if not found."""
        return self.characters.get(name)
    
    def rename_character(self, old_name, new_name):
        """
        Rename/merge a placeholder character and update all references throughout the graph.

        Behavior:
        1. old_name must be an existing placeholder in <character_X> format.
        2. new_name can be provided either as "<name>" or "name" (stored as "<name>").
        3. If new_name exists, merge old_name into new_name by redirecting all connected edges.
        4. If new_name does not exist, create a new CharacterNode and redirect edges.
        5. Remove old_name from character table and adjacency lists.

        Returns:
            bool: True if successful, False for invalid/missing old_name.
        """
        if not old_name.startswith("<") or not old_name.endswith(">"):
            old_name = f"<{old_name}>"

        new_name_plain = new_name.strip("<>")
        new_name_stored = f"<{new_name_plain}>"

        if old_name not in self.characters:
            return False

        if not re.match(r'^<character_\d+>$', old_name):
            return False  # Only <character_X> format can be renamed

        if old_name == new_name_stored:
            return True

        old_character = self.characters[old_name]

        if new_name_stored not in self.characters:
            self.characters[new_name_stored] = CharacterNode(
                new_name_stored,
                embedding=getattr(old_character, "embedding", None),
            )

        all_edge_ids = set(self.adjacency_list_out.get(old_name, [])) | set(self.adjacency_list_in.get(old_name, []))

        for edge_id in all_edge_ids:
            edge = self.edges.get(edge_id)
            if edge is None:
                continue
            if edge.source == old_name:
                edge.source = new_name_stored
            if edge.target == old_name:
                edge.target = new_name_stored

        old_out = self.adjacency_list_out.pop(old_name, [])
        old_in = self.adjacency_list_in.pop(old_name, [])
        self.adjacency_list_out[new_name_stored] = list(
            set(self.adjacency_list_out.get(new_name_stored, [])) | set(old_out)
        )
        self.adjacency_list_in[new_name_stored] = list(
            set(self.adjacency_list_in.get(new_name_stored, [])) | set(old_in)
        )

        for conversation in self.conversations.values():
            if conversation is None:
                continue
            for msg in conversation.messages:
                if isinstance(msg, list) and len(msg) >= 1 and msg[0] == old_name:
                    msg[0] = new_name_stored
            if hasattr(conversation, "speakers") and isinstance(conversation.speakers, set):
                if old_name in conversation.speakers:
                    conversation.speakers.discard(old_name)
                    conversation.speakers.add(new_name_stored)

        del self.characters[old_name]

        print(f"Renamed character: {old_name} -> {new_name_stored}")

        return True
    
    def get_node_degrees(self):
        """
        Calculate the degree (number of connected edges) for each node in the graph.
        """
        degrees = {}
        
        all_nodes = set(self.adjacency_list_out.keys()) | set(self.adjacency_list_in.keys())
        
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
        
        if node_str.startswith("<") and node_str.endswith(">"):
            character_name = node_str  # Keep angle brackets
            return (True, character_name)
        
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
        if name in self.objects:
            return (name, name)
        
        obj_node = ObjectNode(name)
        self.objects[name] = obj_node
        
        return (name, name)
    

    # --------------------------------------------------------
    # Conversation API
    # --------------------------------------------------------
    def update_conversation(self, clip_id, messages, previous_conversation=False):
        """
        Update or create a conversation in the graph.
        
        Args:
            clip_id: ID of the current clip
            messages: List of [speaker, content] pairs for the conversation (2 elements)
            previous_conversation: If True, update the current conversation; if False, create a new one
        
        Returns:
            int: Conversation ID
        """
        if not messages:
            return None
        
        if previous_conversation and self.current_conversation_id is not None:
            conversation = self.conversations.get(self.current_conversation_id)
            if conversation:
                conversation.add_messages(messages, clip_id)
                conversation.add_clip(clip_id)
                return conversation.id
            else:
                previous_conversation = False
        
        if not previous_conversation:
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, list) and len(msg) >= 2:
                    speaker = msg[0]
                    content = msg[1]
                    # Embed messages without speaker angle brackets.
                    speaker_name = speaker
                    if speaker_name.startswith("<") and speaker_name.endswith(">"):
                        speaker_name = speaker_name[1:-1]
                    formatted_msg = f"{speaker_name}: {content}"
                    try:
                        embedding = get_embedding(formatted_msg)
                    except Exception as e:
                        print(f"Warning: Failed to get embedding for message, using None: {e}")
                        embedding = None
                    formatted_messages.append([speaker, content, clip_id, embedding])
            
            conversation = Conversation(clip_id=clip_id, messages=formatted_messages)
            self.conversations[conversation.id] = conversation
            self.current_conversation_id = conversation.id
            return conversation.id


    # --------------------------------------------------------
    # Edge API
    # --------------------------------------------------------
    def _find_existing_high_level_edge(self, source, content, target, clip_id=0, scene=None):
        """
        Find an existing high-level edge that matches the given parameters exactly.
        Note: "Anna friends with Susan" and "Susan friends with Anna" are treated as different edges.
        
        Args:
            source: Source node name
            content: Edge content (attribute or relationship)
            target: Target node name (None for attributes)
            clip_id: Clip ID (should be 0 for high-level/appearance edges)
            scene: Edge scene namespace ("high-level" or "appearance")
        
        Returns:
            Edge object if found, None otherwise
        """
        for edge_id, edge in self.edges.items():
            if edge.clip_id != clip_id:
                continue
            if edge.scene != scene:
                continue
            
            if edge.content != content:
                continue
            
            if target is None:
                if edge.source == source and edge.target is None:
                    return edge
            else:
                if edge.source == source and edge.target == target:
                    return edge
        
        return None
    
    def add_edge(self, edge):
        source_exists = False
        if edge.source.startswith("<") and edge.source.endswith(">"):
            if edge.source in self.characters:
                source_exists = True
        else:
            if edge.source in self.objects:
                source_exists = True
        
        target_exists = False
        if edge.target is None:
            target_exists = True
        elif edge.target.startswith("<") and edge.target.endswith(">"):
            if edge.target in self.characters:
                target_exists = True
        else:
            if edge.target in self.objects:
                target_exists = True
        
        if not source_exists:
            raise ValueError(f"Source node '{edge.source}' not found in graph")
        if not target_exists:
            raise ValueError(f"Target node '{edge.target}' not found in graph")

        self.edges[edge.id] = edge
        self.adjacency_list_out[edge.source].append(edge.id)
        if edge.target is not None:
            self.adjacency_list_in[edge.target].append(edge.id)
        else:
            self.adjacency_list_in[None].append(edge.id)

        return edge.id

    def add_high_level_edge(self, edge):
        """
        Add a high-level edge (clip_id=0) with duplicate checking.
        If a duplicate exists, updates confidence score if new one is higher.
        
        Args:
            edge: Edge object to add (must have clip_id=0)
        
        Returns:
            edge.id if added/updated, None if skipped
        """
        if edge.clip_id != 0:
            return self.add_edge(edge)

        # Normalize namespace for high-level edges.
        if edge.scene is None:
            edge.scene = "high-level"
        
        existing_edge = self._find_existing_high_level_edge(
            source=edge.source,
            content=edge.content,
            target=edge.target,
            clip_id=edge.clip_id,
            scene=edge.scene,
        )
        
        if existing_edge:
            new_confidence = getattr(edge, 'confidence', None)
            old_confidence = getattr(existing_edge, 'confidence', None)
            
            if new_confidence is not None and (old_confidence is None or new_confidence > old_confidence):
                existing_edge.confidence = new_confidence
                return existing_edge.id
            else:
                return None
        else:
            return self.add_edge(edge)

    def _match_and_merge_character(self, char_name, character_appearance, similarity_threshold=0.85):
        """
        Match a new character with existing characters based on appearance similarity.
        If a match is found with a <character_X> (not named character), merge them.
        
        Args:
            char_name: Character name to match (e.g., "<character_3>")
            character_appearance: Dictionary mapping character names to appearance descriptions
            similarity_threshold: Minimum similarity to consider a match (default: 0.85)
        
        Returns:
            str: The character name to use (either original or merged name), or None if no match
        """
        if not character_appearance or char_name not in character_appearance:
            return None
        
        new_appearance = character_appearance[char_name]
        
        try:
            new_appearance_emb = get_embedding(new_appearance)
        except Exception as e:
            print(f"Warning: Failed to get embedding for character appearance: {e}")
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for existing_char_name in self.characters:
            if not existing_char_name.startswith("<character_") or existing_char_name == "<robot>":
                continue
            
            if existing_char_name in character_appearance:
                existing_appearance = character_appearance[existing_char_name]
                try:
                    existing_appearance_emb = get_embedding(existing_appearance)
                    sim = self._cosine_similarity(new_appearance_emb, existing_appearance_emb)
                    if sim > best_similarity and sim >= similarity_threshold:
                        best_similarity = sim
                        best_match = existing_char_name
                except Exception as e:
                    continue
        
        if best_match:
            old_name_plain = best_match.strip("<>")
            new_name_plain = char_name.strip("<>")
            
            if self.rename_character(old_name_plain, new_name_plain):
                if best_match in character_appearance:
                    del character_appearance[best_match]
                print(f"Matched and merged character: {best_match} -> {char_name} (similarity: {best_similarity:.3f})")
                return char_name
        
        return None
    
    def insert_triples(self, triples, clip_id, scene):
        """
        Insert triples into the graph.
        
        Args:
            triples: List of triples, each triple is [source, edge_content, target]
            clip_id: ID of the clip these triples belong to
            scene: Scene name for these triples
            character_appearance: Kept for backward compatibility; not used for matching.
        
        Rules:
        1. Each triple: [source, edge_content, target]
        2. Elements with angle brackets <> are character nodes, otherwise object nodes
        3. Character nodes are created when first encountered in triples
        4. Uniqueness of object nodes is determined by name only
        5. Don't insert duplicate edges in the same list
        """
        if not triples:
            return
        
        # Deduplicate triples before batched embedding.
        seen_edges = set()
        parsed_edges = []

        # Compute scene embedding once per clip.
        scene_embedding = None
        if scene is not None and str(scene).strip():
            try:
                scene_embedding = get_embedding(str(scene))
            except Exception as e:
                print(f"Warning: failed to generate scene embedding for clip {clip_id}: {e}")

        for triple in triples:
            if not isinstance(triple, list) or len(triple) < 3:
                continue
            
            source_str = triple[0]
            edge_content = triple[1]
            target_str = triple[2]
            
            if source_str is None:
                continue
            
            if edge_content is None:
                continue
            edge_content = str(edge_content).strip()
            if not edge_content:
                continue
            
            is_char_src, src_name = self._parse_node_string(source_str)
            
            if is_char_src:
                if src_name not in self.characters:
                    self.add_character(src_name)
                    source_node_name = src_name
                else:
                    source_node_name = src_name
            else:
                _, source_node_name = self._get_or_create_object_node(src_name)
            
            if target_str is None or (isinstance(target_str, str) and target_str.lower() == "null"):
                target_node_name = None
            else:
                is_char_tgt, tgt_name = self._parse_node_string(target_str)
                
                if is_char_tgt:
                    if tgt_name not in self.characters:
                        self.add_character(tgt_name)
                        target_node_name = tgt_name
                    else:
                        target_node_name = tgt_name
                else:
                    _, target_node_name = self._get_or_create_object_node(tgt_name)
            
            edge_key = (source_node_name, target_node_name, edge_content)
            if edge_key in seen_edges:
                continue  # Skip duplicate
            
            seen_edges.add(edge_key)

            parsed_edges.append((source_node_name, target_node_name, edge_content, triple))

        if not parsed_edges:
            return

        edge_contents = [item[2] for item in parsed_edges]
        edge_embeddings = [None] * len(parsed_edges)
        try:
            batch_embeddings = get_multiple_embeddings(edge_contents)
            if len(batch_embeddings) == len(edge_embeddings):
                edge_embeddings = batch_embeddings
            else:
                print(
                    f"Warning: embedding batch size mismatch in clip {clip_id}: "
                    f"expected {len(edge_embeddings)}, got {len(batch_embeddings)}"
                )
        except Exception as e:
            print(f"Warning: batch edge embedding insertion failed for clip {clip_id}: {e}")

        for (source_node_name, target_node_name, edge_content, triple), edge_embedding in zip(parsed_edges, edge_embeddings):
            if edge_embedding is None:
                try:
                    edge_embedding = get_embedding(edge_content)
                except Exception as e:
                    print(f"Warning: failed to embed edge content '{edge_content}' in clip {clip_id}: {e}")

            edge = Edge(
                clip_id=clip_id,
                source=source_node_name,
                target=target_node_name,
                content=edge_content,
                scene=scene,
                embedding=edge_embedding,
                scene_embedding=scene_embedding,
            )
            try:
                self.add_edge(edge)
            except ValueError as e:
                print(f"Warning: {e}, skipping triple: {triple}")
                continue

    def edges_of(self, node_id):
        return set(self.adjacency_list_out[node_id]) | set(self.adjacency_list_in[node_id])

    def get_connected_edges(self, character1, character2):
        return self._get_connected_edges(character1, character2)

    def _get_connected_edges(self, character1, character2, through_clip=None):
        """
        Get all edges directly or indirectly connected between two characters.
        
        Direct connection: An edge where one character is source and the other is target (or vice versa).
        Indirect connection: character1 connects to an object, and that object connects to character2,
        where the clip_id difference between the two edges is less than 4.
        
        Args:
            character1: First character name (with or without angle brackets, e.g., "<Alice>" or "Alice")
            character2: Second character name (with or without angle brackets, e.g., "<Bob>" or "Bob")
            through_clip: If provided, ignore evidence from later clips.
        
        Returns:
            list: List of Edge objects that are directly or indirectly connected between the two characters
        """
        if not character1.startswith("<") or not character1.endswith(">"):
            character1 = f"<{character1}>"
        if not character2.startswith("<") or not character2.endswith(">"):
            character2 = f"<{character2}>"
        
        if character1 not in self.characters:
            raise ValueError(f"Character '{character1}' not found in graph")
        if character2 not in self.characters:
            raise ValueError(f"Character '{character2}' not found in graph")
        
        result_edges = []
        result_edge_ids = set()
        
        # Collect direct and short-window object-mediated connections.
        char1_edges = self.edges_of(character1)
        char2_edges = self.edges_of(character2)
        
        direct_edges = char1_edges & char2_edges
        for edge_id in direct_edges:
            if edge_id not in result_edge_ids:
                edge = self.edges.get(edge_id)
                if edge is not None and (through_clip is None or edge.clip_id <= through_clip):
                    result_edges.append(edge)
                    result_edge_ids.add(edge_id)
        
        for edge_id in char1_edges:
            edge1 = self.edges.get(edge_id)
            if edge1 is None:
                continue
            if through_clip is not None and edge1.clip_id > through_clip:
                continue
            
            other_node = None
            if edge1.source == character1:
                other_node = edge1.target
            elif edge1.target == character1:
                other_node = edge1.source
            
            if other_node is None:
                continue
            
            is_char, _ = self._parse_node_string(other_node)
            if is_char:
                continue  # Skip if it's a character
            
            object_edges = self.edges_of(other_node)
            for edge_id2 in object_edges:
                edge2 = self.edges.get(edge_id2)
                if edge2 is None:
                    continue
                if through_clip is not None and edge2.clip_id > through_clip:
                    continue
                
                connects_to_char2 = False
                if (edge2.source == other_node and edge2.target == character2) or \
                   (edge2.target == other_node and edge2.source == character2):
                    connects_to_char2 = True
                
                if connects_to_char2:
                    clip_diff = abs(edge1.clip_id - edge2.clip_id)
                    if clip_diff < 4:
                        if edge_id not in result_edge_ids:
                            result_edges.append(edge1)
                            result_edge_ids.add(edge_id)
                        if edge_id2 not in result_edge_ids:
                            result_edges.append(edge2)
                            result_edge_ids.add(edge_id2)
        
        return result_edges

    def edge_embedding_insertion(self):
        edge_contents = [edge.content for edge in self.edges.values()]
        embeddings = get_multiple_embeddings(edge_contents)
        for edge, embedding in zip(self.edges.values(), embeddings):
            edge.embedding = embedding
        print(len(embeddings), "edge embeddings inserted")

    def insert_high_level_and_appearance_embeddings(self, batch_size=512):
        """
        Insert/update embeddings for high-level and appearance edges together.
        Processes both namespaces in one combined embedding pass with batching.
        """
        target_edges = [
            edge for edge in self.edges.values()
            if edge.clip_id == 0 and edge.scene in {"high-level", "appearance"}
        ]
        if not target_edges:
            print("No high-level/appearance edges found for embedding insertion")
            return {"high-level": 0, "appearance": 0, "skipped": 0}

        inserted_by_scene = {"high-level": 0, "appearance": 0}
        skipped = 0

        # Bound embedding request size.
        for start in range(0, len(target_edges), batch_size):
            chunk_edges = target_edges[start:start + batch_size]
            chunk_texts = []
            chunk_indices = []

            for idx, edge in enumerate(chunk_edges):
                if edge.content is None:
                    skipped += 1
                    continue
                text = str(edge.content).strip()
                if not text:
                    skipped += 1
                    continue
                chunk_texts.append(text)
                chunk_indices.append(idx)

            if not chunk_texts:
                continue

            try:
                chunk_embeddings = get_multiple_embeddings(chunk_texts)
                if len(chunk_embeddings) != len(chunk_texts):
                    print(
                        "Warning: high-level/appearance embedding batch size mismatch "
                        f"(expected {len(chunk_texts)}, got {len(chunk_embeddings)})"
                    )
                for local_i, embedding in enumerate(chunk_embeddings):
                    if local_i >= len(chunk_indices):
                        break
                    edge = chunk_edges[chunk_indices[local_i]]
                    edge.embedding = embedding
                    if edge.scene in inserted_by_scene:
                        inserted_by_scene[edge.scene] += 1
            except Exception as e:
                print(
                    "Warning: failed batch embedding insertion for high-level/appearance "
                    f"edges [{start}:{start + len(chunk_edges)}]: {e}"
                )
                # Fall back to per-edge embedding.
                for idx in chunk_indices:
                    edge = chunk_edges[idx]
                    try:
                        edge.embedding = get_embedding(str(edge.content).strip())
                        if edge.scene in inserted_by_scene:
                            inserted_by_scene[edge.scene] += 1
                    except Exception as e2:
                        skipped += 1
                        print(f"Warning: failed embedding for {edge.scene} edge '{edge.content}': {e2}")

        total_inserted = inserted_by_scene["high-level"] + inserted_by_scene["appearance"]
        print(
            f"{total_inserted} high-level/appearance edge embeddings inserted "
            f"(high-level={inserted_by_scene['high-level']}, "
            f"appearance={inserted_by_scene['appearance']}, skipped={skipped})"
        )
        return {
            "high-level": inserted_by_scene["high-level"],
            "appearance": inserted_by_scene["appearance"],
            "skipped": skipped,
        }
    
    def node_embedding_insertion(self):
        """
        Generate embeddings for all nodes (characters and objects) in batch.
        This is more efficient than generating embeddings one by one during node creation.
        """
        node_names_for_embedding = []
        node_objects = []
        
        for char_name, char_node in self.characters.items():
            if char_node.embedding is None:
                name_for_embedding = char_name.strip("<>") if char_name.startswith("<") and char_name.endswith(">") else char_name
                node_names_for_embedding.append(name_for_embedding)
                node_objects.append(('character', char_node))
        
        for obj_name, obj_node in self.objects.items():
            if obj_node.embedding is None:
                node_names_for_embedding.append(obj_name)
                node_objects.append(('object', obj_node))
        
        if not node_names_for_embedding:
            print("No nodes need embedding generation")
            return
        
        try:
            embeddings = get_multiple_embeddings(node_names_for_embedding)
            for (node_type, node), embedding in zip(node_objects, embeddings):
                node.embedding = embedding
            print(f"{len(embeddings)} node embeddings inserted ({len([n for n, _ in node_objects if n == 'character'])} characters, {len([n for n, _ in node_objects if n == 'object'])} objects)")
        except Exception as e:
            print(f"Warning: Failed to generate node embeddings in batch: {e}")
            for (node_type, node), name in zip(node_objects, node_names_for_embedding):
                try:
                    node.embedding = get_embedding(name)
                except Exception as e2:
                    print(f"Warning: Failed to generate embedding for {name}: {e2}")


    # --------------------------------------------------------
    # Abstract Information API
    # --------------------------------------------------------
    def _reset_abstraction_state(self):
        """Clear incremental-abstraction bookkeeping.

        Called at the start of run_abstraction so that replaying abstraction
        (e.g. from a checkpoint in the ablation script) starts from a clean
        slate regardless of any prior incremental state.
        """
        self._char_consumed_edges = defaultdict(set)
        self._char_last_degree = defaultdict(int)
        self._pair_consumed_edges = defaultdict(set)
        self._pair_last_degree = defaultdict(int)

    def _low_level_edges_of(self, node, through_clip=None):
        """Return incident low-level edge ids visible through the given clip."""
        edge_ids = self.edges_of(node)
        return {
            eid for eid in edge_ids
            if self.edges.get(eid) is not None
            and self.edges[eid].clip_id > 0
            and (through_clip is None or self.edges[eid].clip_id <= through_clip)
        }

    def _low_level_degree(self, node, through_clip=None):
        return len(self._low_level_edges_of(node, through_clip))

    def _existing_attributes_of(self, character_name):
        """Return list of [attribute, confidence] for already-extracted high-level
        attribute edges of this character (clip_id=0, scene='high-level', target=None)."""
        if not character_name.startswith("<") or not character_name.endswith(">"):
            character_name = f"<{character_name}>"
        result = []
        for eid in self.edges_of(character_name):
            edge = self.edges.get(eid)
            if edge is None:
                continue
            if edge.clip_id == 0 and edge.scene == "high-level" and edge.target is None and edge.source == character_name:
                result.append([edge.content, getattr(edge, "confidence", None)])
        return result

    def _existing_relationships_between(self, character1, character2):
        """Return list of [source, relationship, target, confidence] for already-
        extracted high-level relationship edges between the two characters
        (either direction)."""
        if not character1.startswith("<") or not character1.endswith(">"):
            character1 = f"<{character1}>"
        if not character2.startswith("<") or not character2.endswith(">"):
            character2 = f"<{character2}>"
        result = []
        seen = set()
        for eid in (self.edges_of(character1) | self.edges_of(character2)):
            edge = self.edges.get(eid)
            if edge is None:
                continue
            if edge.clip_id != 0 or edge.scene != "high-level" or edge.target is None:
                continue
            pair = frozenset({edge.source, edge.target})
            if pair != frozenset({character1, character2}):
                continue
            key = (edge.source, edge.content, edge.target)
            if key in seen:
                continue
            seen.add(key)
            result.append([edge.source, edge.content, edge.target, getattr(edge, "confidence", None)])
        return result

    def _shared_low_level_edge_ids(self, character1, character2, through_clip=None):
        """Return the set of low-level edge ids that connect the two characters
        (directly, or indirectly through an object within a clip window of 4),
        i.e. the low-level subset of get_connected_edges."""
        connected = self._get_connected_edges(character1, character2, through_clip)
        return {e.id for e in connected if e.clip_id > 0}

    def character_attributes(self, character_name, incremental=False, through_clip=None):
        """
        Extract character attributes by analyzing edges connected to the character.

        This function:
        1. Collects edges (incoming and outgoing) connected to the character
        2. Formats them as a readable string (one edge per line)
        3. Combines with prompt_character_summary
        4. Uses LLM to generate character attributes
        5. Parses the LLM output and creates attribute edges in the graph

        Args:
            character_name: Character name (with or without angle brackets, e.g., "<Alice>" or "Alice")
            incremental: If True, only feed NEW low-level edges (since the last summary of
                this character) and inject already-extracted attributes into the prompt so
                the LLM does not regenerate them. Also updates the consumed-edge bookkeeping.
            through_clip: If provided, only use evidence available through that clip.

        Returns:
            int: Token usage for this LLM call
        """
        if not character_name.startswith("<") or not character_name.endswith(">"):
            character_name = f"<{character_name}>"

        if character_name not in self.characters:
            raise ValueError(f"Character '{character_name}' not found in graph")

        if incremental:
            current_low = self._low_level_edges_of(character_name, through_clip)
            new_ids = current_low - self._char_consumed_edges.get(character_name, set())
            if not new_ids:
                print(f"Info: Skip incremental character_attributes for {character_name}: no new low-level edges.")
                return 0
            edge_ids = new_ids
        else:
            edge_ids = self.edges_of(character_name)

        if not edge_ids:
            print(f"Info: Skip character_attributes for {character_name}: no connected edges.")
            return 0

        edge_lines = []
        for edge_id in sorted(edge_ids):  # Sort for consistent ordering
            edge = self.edges.get(edge_id)
            if edge is None:
                continue

            target_str = edge.target if edge.target is not None else "null"
            edge_str = f"{edge.source}, {edge.content}, {target_str}"
            if edge.scene:
                edge_str += f", scene: {edge.scene}"

            edge_lines.append(edge_str)

        edges_text = "\n".join(edge_lines)

        existing_section = ""
        if incremental:
            existing_attrs = self._existing_attributes_of(character_name)
            if existing_attrs:
                existing_section = (
                    "\n\nExisting attributes already extracted (do NOT regenerate these "
                    "unless the new evidence below contradicts or meaningfully extends them):\n"
                    + json.dumps(existing_attrs, ensure_ascii=False)
                )

        full_prompt = (
            f"Character: {character_name}\n\n"
            f"Character behaviors (from graph edges):\n{edges_text}"
            f"{existing_section}\n{prompt_character_summary}"
        )
        try:
            attributes_response, tokens = generate_text_response(full_prompt)
        except Exception as e:
            print(f"LLM call failed, retrying... Error: {e}")
            attributes_response, tokens = generate_text_response(full_prompt)

        attributes_response = strip_code_fences(attributes_response)
        try:
            attributes_dict = json.loads(attributes_response)
        except json.JSONDecodeError as e:
            import re
            matches = re.findall(r'"([^"]*)"\s*:\s*(\d+)', attributes_response)
            if matches:
                attributes_dict = {k: int(v) for k, v in matches}
            else:
                print(f"Failed to parse LLM response as JSON: {e}")
                print(f"Response was: {attributes_response}")
                return tokens

        for attribute_name, confidence in attributes_dict.items():
            if not isinstance(confidence, (int, float)) or confidence < 50:
                continue

            edge = Edge(
                clip_id=0,
                source=character_name,
                target=None,
                content=attribute_name,
                scene="high-level",
                confidence=confidence
            )
            try:
                self.add_high_level_edge(edge)
            except Exception as e:
                print(
                    f"Warning: Failed to add high-level attribute edge for {character_name} "
                    f"(attribute='{attribute_name}', confidence={confidence}): {e}"
                )

        # Advance incremental bookkeeping after a successful summary.
        if incremental:
            self._char_consumed_edges[character_name].update(edge_ids)
            self._char_last_degree[character_name] = len(current_low)

        return tokens

    def character_relationships(self, character1, character2, incremental=False, through_clip=None):
        """
        Extract character relationships by analyzing all edges between two characters.

        This function:
        1. Gets all edges directly or indirectly connected between the two characters
        2. Formats them as a readable string (one edge per line)
        3. Combines with prompt_character_relationships
        4. Uses LLM to generate relationship descriptions
        5. Parses the LLM output and creates relationship edges in the graph

        Args:
            character1: First character name (with or without angle brackets, e.g., "<Alice>" or "Alice")
            character2: Second character name (with or without angle brackets, e.g., "<Bob>" or "Bob")
            incremental: If True, only feed NEW shared low-level edges (since the last summary of
                this pair) and inject already-extracted relationships into the prompt so the LLM
                does not regenerate them. Also updates the consumed-edge bookkeeping.
            through_clip: If provided, only use evidence available through that clip.

        Returns:
            int: Token usage for this LLM call
        """
        if not character1.startswith("<") or not character1.endswith(">"):
            character1 = f"<{character1}>"
        if not character2.startswith("<") or not character2.endswith(">"):
            character2 = f"<{character2}>"

        if character1 not in self.characters:
            raise ValueError(f"Character '{character1}' not found in graph")
        if character2 not in self.characters:
            raise ValueError(f"Character '{character2}' not found in graph")

        pair_key = tuple(sorted((character1, character2)))

        if incremental:
            current_shared = self._shared_low_level_edge_ids(character1, character2, through_clip)
            new_ids = current_shared - self._pair_consumed_edges.get(pair_key, set())
            if not new_ids:
                print(f"Info: Skip incremental character_relationships for {character1}, {character2}: no new shared low-level edges.")
                return 0
            new_edges = [self.edges[eid] for eid in sorted(new_ids)]
            connected_edges = new_edges
        else:
            connected_edges = self.get_connected_edges(character1, character2)

        if not connected_edges or len(connected_edges) < 3:
            print(
                f"Info: Skip character_relationships for {character1}, {character2}: "
                f"insufficient connected edges ({len(connected_edges) if connected_edges else 0} < 3)."
            )
            return 0

        edge_lines = []
        for edge in sorted(connected_edges, key=lambda e: (e.clip_id, e.id)):  # Sort by clip_id for chronological order
            target_str = edge.target if edge.target is not None else "null"
            edge_str = f"{edge.source}, {edge.content}, {target_str}"
            if edge.scene:
                edge_str += f", scene: {edge.scene}"

            edge_lines.append(edge_str)

        edges_text = "\n".join(edge_lines)

        existing_section = ""
        if incremental:
            existing_rels = self._existing_relationships_between(character1, character2)
            if existing_rels:
                existing_section = (
                    "\n\nExisting relationships already extracted (do NOT regenerate these "
                    "unless the new evidence below contradicts or meaningfully extends them):\n"
                    + json.dumps(existing_rels, ensure_ascii=False)
                )

        full_prompt = (
            f"Character 1: {character1}\nCharacter 2: {character2}\n\n"
            f"Character interactions (from graph edges):\n{edges_text}"
            f"{existing_section}\n{prompt_character_relationships}"
        )
        try:
            relationships_response, tokens = generate_text_response(full_prompt)
        except Exception as e:
            print(f"LLM call failed, retrying... Error: {e}")
            relationships_response, tokens = generate_text_response(full_prompt)

        relationships_response = strip_code_fences(relationships_response)
        try:
            relationships_list = json.loads(relationships_response)
        except json.JSONDecodeError as e:
            import re
            matches = re.findall(r'\[\s*"[^"]*"\s*,\s*"[^"]*"\s*,\s*"[^"]*"\s*,\s*\d+\s*\]', relationships_response)
            if matches:
                relationships_list = []
                for m in matches:
                    try:
                        relationships_list.append(json.loads(m))
                    except:
                        continue
            else:
                print(f"Failed to parse LLM response as JSON: {e}")
                print(f"Response was: {relationships_response}")
                return tokens

        relationships_created = []
        for rel in relationships_list:
            if not isinstance(rel, list) or len(rel) < 4:
                continue

            rel_char1, relationship, rel_char2, confidence = rel[0], rel[1], rel[2], rel[3]

            if not isinstance(confidence, (int, float)) or confidence < 50:
                continue

            if not rel_char1.startswith("<") or not rel_char1.endswith(">"):
                rel_char1 = f"<{rel_char1}>"
            if not rel_char2.startswith("<") or not rel_char2.endswith(">"):
                rel_char2 = f"<{rel_char2}>"

            if (rel_char1 == character1 and rel_char2 == character2) or \
               (rel_char1 == character2 and rel_char2 == character1):
                edge = Edge(
                    clip_id=0,
                    source=rel_char1,
                    target=rel_char2,
                    content=relationship,
                    scene="high-level",
                    confidence=confidence
                )
                try:
                    self.add_high_level_edge(edge)
                    relationships_created.append(rel)
                except Exception as e:
                    print(f"Failed to add relationship edge: {e}")
                    pass

        # Advance incremental bookkeeping after a successful summary.
        if incremental:
            self._pair_consumed_edges[pair_key].update(new_ids)
            self._pair_last_degree[pair_key] = len(current_shared)

        return tokens

    # --------------------------------------------------------
    # Incremental / threshold-driven abstraction driver
    # --------------------------------------------------------
    def _characters_touched_in_clip(self, clip_id):
        """Return the set of character names that have at least one low-level
        edge in the given clip."""
        touched = set()
        for edge in self.edges.values():
            if edge.clip_id != clip_id:
                continue
            for endpoint in (edge.source, edge.target):
                if endpoint is None:
                    continue
                if endpoint.startswith("<") and endpoint.endswith(">") and endpoint in self.characters:
                    touched.add(endpoint)
        return touched

    def _incremental_step(self, clip_id, config):
        """Check thresholds for characters/pairs touched by `clip_id` and fire
        incremental summaries when crossed. Mirrors the behavior of interleaving
        abstraction after this clip's edges were inserted.

        Unified trigger: a character is summarized whenever its low-level degree
        crosses a multiple of interval_node (first summary at interval_node, then
        every interval_node NEW edges). Pairs behave the same with interval_pair
        over their shared low-level edge count. No separate per-character degree
        gate is needed for pairs because shared edges are a subset of each
        member's incident edges.
        """
        touched = self._characters_touched_in_clip(clip_id)
        if not touched:
            return 0, 0

        attr_tokens = empty_usage()
        rel_tokens = empty_usage()

        # Summarize characters after each interval_node of new evidence.
        eligible_chars = []
        for char in touched:
            degree = self._low_level_degree(char, clip_id)
            last = self._char_last_degree.get(char, 0)
            if degree >= config.interval_node and (degree - last) >= config.interval_node:
                eligible_chars.append(char)
        for char in eligible_chars:
            try:
                attr_tokens = merge_usage(
                    attr_tokens,
                    self.character_attributes(
                        char,
                        incremental=True,
                        through_clip=clip_id,
                    ),
                )
            except Exception as e:
                print(f"✗ Error in incremental character_attributes for {char}: {e}")

        # Evaluate pairs touched by this clip at each interval_pair crossing.
        evaluated_pairs = set()
        for c1 in touched:
            for c2 in self.characters:
                if c2 == c1:
                    continue
                pair_key = tuple(sorted((c1, c2)))
                if pair_key in evaluated_pairs:
                    continue
                evaluated_pairs.add(pair_key)
                count = len(self._shared_low_level_edge_ids(c1, c2, clip_id))
                last = self._pair_last_degree.get(pair_key, 0)
                if count >= config.interval_pair and (count - last) >= config.interval_pair:
                    try:
                        rel_tokens = merge_usage(
                            rel_tokens,
                            self.character_relationships(
                                c1,
                                c2,
                                incremental=True,
                                through_clip=clip_id,
                            ),
                        )
                    except Exception as e:
                        print(f"✗ Error in incremental character_relationships for {c1}, {c2}: {e}")

        return attr_tokens, rel_tokens

    def _final_round(self, config):
        """After the incremental replay, summarize any character/pair whose
        low-level edge count is at least the lower bound. Uses incremental mode
        so already-consumed edges are skipped (cheap when fully summarized)."""
        attr_tokens = empty_usage()
        rel_tokens = empty_usage()

        # Characters above the node lower bound.
        for char in list(self.characters.keys()):
            if self._low_level_degree(char) < config.final_lower_bound_node:
                continue
            try:
                attr_tokens = merge_usage(
                    attr_tokens,
                    self.character_attributes(char, incremental=True),
                )
            except Exception as e:
                print(f"✗ Error in final character_attributes for {char}: {e}")

        # Pairs above the pair lower bound (both members above node lower bound).
        eligible = [c for c in self.characters
                    if self._low_level_degree(c) >= config.final_lower_bound_node]
        for i in range(len(eligible) - 1):
            for j in range(i + 1, len(eligible)):
                c1, c2 = eligible[i], eligible[j]
                if len(self._shared_low_level_edge_ids(c1, c2)) < config.final_lower_bound_pair:
                    continue
                try:
                    rel_tokens = merge_usage(
                        rel_tokens,
                        self.character_relationships(c1, c2, incremental=True),
                    )
                except Exception as e:
                    print(f"✗ Error in final character_relationships for {c1}, {c2}: {e}")

        return attr_tokens, rel_tokens

    def run_abstraction(self, config):
        """Run the full threshold-based abstraction pipeline.

        1. (optional) Incremental replay: iterate clips in order, firing
           character_attributes / character_relationships whenever a node/pair
           crosses its threshold, feeding only NEW low-level edges + existing
           attributes to the LLM.
        2. Final round: summarize any node/pair whose low-level edge count is
           above the lower bound (catches nodes that never crossed the
           incremental threshold).

        Resets incremental bookkeeping at the start so this is safe to call on
        a checkpoint (e.g. from abstraction_ablation.py) regardless of prior
        state. Returns a token-usage summary dict.
        """
        from utils.abstraction_config import AbstractionConfig  # local import avoids cycles
        if not isinstance(config, AbstractionConfig):
            raise TypeError("config must be an AbstractionConfig instance")

        self._reset_abstraction_state()

        # Pickle does not preserve Edge._id_counter, so resume above existing ids.
        from classes.edge_class import Edge
        Edge._id_counter = max(self.edges.keys()) if self.edges else 0

        attr_tokens = empty_usage()
        rel_tokens = empty_usage()

        if config.incremental_enabled:
            clips = sorted({e.clip_id for e in self.edges.values() if e.clip_id > 0})
            print(f"run_abstraction: incremental replay over {len(clips)} clips "
                  f"(interval_node={config.interval_node}, interval_pair={config.interval_pair})")
            for clip_id in clips:
                a, r = self._incremental_step(clip_id, config)
                attr_tokens = merge_usage(attr_tokens, a)
                rel_tokens = merge_usage(rel_tokens, r)
        else:
            print("run_abstraction: incremental phase disabled, only final round will run")

        print("run_abstraction: final round")
        a, r = self._final_round(config)
        attr_tokens = merge_usage(attr_tokens, a)
        rel_tokens = merge_usage(rel_tokens, r)

        print(
            "run_abstraction done. "
            f"attributes_tokens={usage_total(attr_tokens)}, "
            f"relationships_tokens={usage_total(rel_tokens)}"
        )
        return {
            "attributes_tokens": usage_total(attr_tokens),
            "relationships_tokens": usage_total(rel_tokens),
            "attributes_usage": attr_tokens,
            "relationships_usage": rel_tokens,
        }

    def extract_conversation_summary(self, conversation_id):

        conversation = self.conversations.get(conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation with id {conversation_id} not found in graph")
        
        if not conversation.messages:
            return {
                "summary": "",
                "character_attributes": [],
                "characters_relationships": []
            }, 0
        
        formatted_messages = conversation.format_messages()
        full_prompt = prompt_conversation_summary + "\n" + formatted_messages

        try:
            response, tokens = generate_text_response(full_prompt, text_format=ConversationSummary)
        except Exception as e:
            print(f"LLM call failed, retrying... Error: {e}")
            response, tokens = generate_text_response(full_prompt, text_format=ConversationSummary)

        if isinstance(response, str):
            try:
                response = ConversationSummary.model_validate_json(strip_code_fences(response))
            except Exception as e:
                print(f"Failed to parse conversation summary response: {e}")
                return {
                    "summary": "",
                    "character_attributes": [],
                    "characters_relationships": []
                }, tokens

        summary = response.summary
        character_attributes = response.character_attributes
        characters_relationships = response.characters_relationships
        
        conversation.summary = summary

        for attr_item in character_attributes:
            char_name = attr_item[0]
            attribute = attr_item[1]
            confidence = attr_item[2]
            
            if not isinstance(confidence, (int, float)) or confidence < 50:
                continue
            
            if not char_name.startswith("<") or not char_name.endswith(">"):
                char_name = f"<{char_name}>"
            
            if char_name not in self.characters:
                self.add_character(char_name)
                print(f"Info: Added character '{char_name}' to graph from conversation summary")
            
            edge = Edge(
                clip_id=0,
                source=char_name,
                target=None,
                content=attribute,
                scene="high-level",
                confidence=confidence
            )
            try:
                self.add_high_level_edge(edge)
            except Exception as e:
                print(f"Warning: Failed to add attribute edge for {char_name}: {e}")
        
        for rel_item in characters_relationships:
            char1 = rel_item[0]
            relationship = rel_item[1]
            char2 = rel_item[2]
            confidence = rel_item[3]
            
            if not isinstance(confidence, (int, float)) or confidence < 50:
                continue
            
            if not char1.startswith("<") or not char1.endswith(">"):
                char1 = f"<{char1}>"
            if not char2.startswith("<") or not char2.endswith(">"):
                char2 = f"<{char2}>"
            
            if char1 not in self.characters:
                self.add_character(char1)
                print(f"Info: Added character '{char1}' to graph from conversation summary")
            if char2 not in self.characters:
                self.add_character(char2)
                print(f"Info: Added character '{char2}' to graph from conversation summary")
            
            edge = Edge(
                clip_id=0,
                source=char1,
                target=char2,
                content=relationship,
                scene="high-level",
                confidence=confidence
            )
            try:
                self.add_high_level_edge(edge)
            except Exception as e:
                print(f"Warning: Failed to add relationship edge between {char1} and {char2}: {e}")
        
        return {
            "summary": summary,
            "character_attributes": character_attributes,
            "characters_relationships": characters_relationships
        }, tokens
    
    def insert_character_appearances(self, character_appearance):
        """
        Insert character appearances as high-level edges after all clips are processed.
        Each comma-separated feature in the appearance description becomes a separate edge.
        
        Args:
            character_appearance: Dictionary mapping character names to appearance descriptions
                                  Can be a dict or JSON string
        """
        if isinstance(character_appearance, str):
            try:
                character_appearance = json.loads(character_appearance)
            except json.JSONDecodeError:
                print("Warning: Failed to parse character_appearance JSON string")
                character_appearance = {}
        
        if not isinstance(character_appearance, dict):
            print("Warning: character_appearance is not a dictionary")
            return
        
        print(f"Inserting character appearances for {len(character_appearance)} characters...")
        
        total_edges = 0
        for char_name, appearance_desc in character_appearance.items():
            if not char_name.startswith("<") or not char_name.endswith(">"):
                char_name = f"<{char_name}>"
            
            if char_name not in self.characters:
                print(f"Warning: Character '{char_name}' not found in graph, skipping appearance")
                continue
            
            if isinstance(appearance_desc, list):
                appearance_str = ", ".join(str(item) for item in appearance_desc)
            elif isinstance(appearance_desc, dict):
                appearance_str = ", ".join(f"{k}: {v}" for k, v in appearance_desc.items())
            else:
                appearance_str = str(appearance_desc)
            
            appearance_features = [feature.strip() for feature in appearance_str.split(",") if feature.strip()]
            
            for feature in appearance_features:
                edge = Edge(
                    clip_id=0,
                    source=char_name,
                    target=None,
                    content=f"{feature}",
                    scene="appearance",
                    confidence=100  # Appearance is factual, so high confidence
                )
                try:
                    self.add_high_level_edge(edge)
                    total_edges += 1
                except Exception as e:
                    print(f"Warning: Failed to add appearance edge for {char_name} (feature: {feature}): {e}")
        
        print(f"✓ Character appearances inserted: {total_edges} appearance edges created")


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
        q_source_weight = query_triple[3] if query_triple[3] is not None else 1.0
        q_content_weight = query_triple[4] if query_triple[4] is not None else 1.0
        q_target_weight = query_triple[5] if query_triple[5] is not None else 1.0
        
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

    
    def search_high_level_edges(self, query_triples, k):
        """
        Search for top-k high-level edges (clip_id=0, scene="high-level")
        using embedding-based similarity.
        High-level edges represent character attributes and relationships only.
        
        Args:
            query_triples: List of query triples in format [source, content, target, source_weight, content_weight, target_weight] or single triple
            k: Number of top results to return
        
        Returns:
            list: List of Edge objects, sorted by relevance (embedding similarity + confidence)
        """
        if not query_triples or k <= 0:
            return []
        
        # Normalize query_triples to list of lists
        # Filter out None values
        query_triples = [q for q in query_triples if q is not None]
        if not query_triples:
            return []
        if isinstance(query_triples[0], str):
            query_triples = [query_triples]
        
        # Filter high-level namespace only (exclude appearance).
        candidate_edges = []
        for edge_id, edge in self.edges.items():
            if edge.clip_id == 0 and edge.scene == "high-level":
                candidate_edges.append(edge)
        
        if not candidate_edges:
            return []
        
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
            
            # Compute embeddings (skip "?" to avoid unnecessary API calls)
            source_emb = None
            if q_source and q_source != "?" and isinstance(q_source, str):
                source_for_emb = q_source.strip("<>") if q_source.startswith("<") and q_source.endswith(">") else q_source
                source_emb = get_embedding(source_for_emb)
            
            content_emb = None
            if q_content and q_content != "?" and isinstance(q_content, str):
                content_emb = get_embedding(q_content)
            
            target_emb = None
            if q_target and q_target != "?" and isinstance(q_target, str):
                target_for_emb = q_target.strip("<>") if q_target.startswith("<") and q_target.endswith(">") else q_target
                target_emb = get_embedding(target_for_emb)
            
            query_triple_embeddings.append([source_emb, content_emb, target_emb])
        
        # Score edges based on embedding similarity with bidirectional matching
        scored_edges = []
        for edge in candidate_edges:
            score = 0.0
            
            # Match against each query triple using embeddings (use max across triples)
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
                
                query_embeddings = query_triple_embeddings[i]
                triple_score = self._compute_edge_similarity(edge, query_triple_with_weights, query_embeddings)
                score = max(score, triple_score)
            
            # Add confidence score if available
            if hasattr(edge, 'confidence') and edge.confidence:
                score += edge.confidence / 100.0 * 0.3  # Weight confidence
            
            scored_edges.append((score, edge))
        
        # Sort by score (descending) and return top-k
        scored_edges.sort(key=lambda x: x[0], reverse=True)
        return [edge for _, edge in scored_edges[:k]]

    def _compute_appearance_edge_similarity(self, edge, query_triple, query_embeddings):
        """
        Compute similarity between an appearance edge and query triple.
        Appearance edges always have source=character and target=None, so only
        source/content terms are used (no target matching and no reverse direction).
        """
        if edge is None:
            return 0.0
        if query_triple is None or not isinstance(query_triple, (list, tuple)) or len(query_triple) < 6:
            return 0.0
        if query_embeddings is None or not isinstance(query_embeddings, (list, tuple)) or len(query_embeddings) < 3:
            return 0.0

        q_source = query_triple[0]
        q_content = query_triple[1]
        q_source_weight = query_triple[3] if query_triple[3] is not None else 1.0
        q_content_weight = query_triple[4] if query_triple[4] is not None else 1.0

        source_emb = query_embeddings[0]
        content_emb = query_embeddings[1]

        source_sim = 0.0
        if q_source and q_source != "?" and edge.source is not None:
            edge_source_emb = self._get_node_embedding(edge.source)
            source_sim = self._calculate_node_similarity(q_source, edge.source, source_emb, edge_source_emb) * q_source_weight

        content_sim = 0.0
        if q_content and q_content != "?" and edge.content and content_emb is not None:
            if edge.embedding is not None:
                try:
                    content_sim = self._cosine_similarity(content_emb, edge.embedding) * q_content_weight
                except Exception:
                    if edge.content == q_content:
                        content_sim = q_content_weight

        return source_sim + content_sim

    def search_appearance_edges(self, query_triples, k):
        """
        Search for top-k appearance edges (clip_id=0, scene="appearance").
        Appearance edges are source=character, target=None and content=appearance descriptor.
        """
        if not query_triples or k <= 0:
            return []

        query_triples = [q for q in query_triples if q is not None]
        if not query_triples:
            return []
        if isinstance(query_triples[0], str):
            query_triples = [query_triples]

        candidate_edges = []
        for edge_id, edge in self.edges.items():
            if edge.clip_id == 0 and edge.scene == "appearance":
                candidate_edges.append(edge)

        if not candidate_edges:
            return []

        # Reuse the same query embeddings shape [source_emb, content_emb, target_emb].
        query_triple_embeddings = []
        for q_triple in query_triples:
            if q_triple is None:
                query_triple_embeddings.append([None, None, None])
                continue

            q_source = q_triple[0] if isinstance(q_triple, (list, tuple)) and len(q_triple) > 0 else None
            q_content = q_triple[1] if isinstance(q_triple, (list, tuple)) and len(q_triple) > 1 else None

            source_emb = None
            if q_source and q_source != "?" and isinstance(q_source, str):
                source_for_emb = q_source.strip("<>") if q_source.startswith("<") and q_source.endswith(">") else q_source
                source_emb = get_embedding(source_for_emb)

            content_emb = None
            if q_content and q_content != "?" and isinstance(q_content, str):
                content_emb = get_embedding(q_content)

            query_triple_embeddings.append([source_emb, content_emb, None])

        scored_edges = []
        for edge in candidate_edges:
            score = 0.0

            for i, q_triple in enumerate(query_triples):
                if q_triple is None:
                    continue

                q_source_weight = q_triple[3] if isinstance(q_triple, (list, tuple)) and len(q_triple) > 3 and q_triple[3] is not None else 1.0
                q_content_weight = q_triple[4] if isinstance(q_triple, (list, tuple)) and len(q_triple) > 4 and q_triple[4] is not None else 1.0

                query_triple_with_weights = [
                    q_triple[0] if isinstance(q_triple, (list, tuple)) and len(q_triple) > 0 else None,
                    q_triple[1] if isinstance(q_triple, (list, tuple)) and len(q_triple) > 1 else None,
                    None,
                    q_source_weight,
                    q_content_weight,
                    0.0,
                ]

                query_embeddings = query_triple_embeddings[i]
                triple_score = self._compute_appearance_edge_similarity(edge, query_triple_with_weights, query_embeddings)
                score = max(score, triple_score)

            if hasattr(edge, "confidence") and edge.confidence:
                score += edge.confidence / 100.0 * 0.3

            scored_edges.append((score, edge))

        scored_edges.sort(key=lambda x: x[0], reverse=True)
        return [edge for _, edge in scored_edges[:k]]
    

    def search_low_level_edges(self, query_triples, k, spatial_constraints=None):
        """
        Search for top-k low-level edges (clip_id>0, scene is not None) using embedding-based similarity.
        Low-level edges represent specific actions and states.
        
        Args:
            query_triples: List of query triples in format [source, content, target] or single triple
            k: Number of top results to return
            spatial_constraints: Optional spatial constraint (location/scene string)
        
        Returns:
            list: List of Edge objects, sorted by relevance (embedding similarity + scene similarity)
        """
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
            
            # Compute embeddings (skip "?" to avoid unnecessary API calls)
            source_emb = None
            if q_source and q_source != "?" and isinstance(q_source, str):
                source_for_emb = q_source.strip("<>") if q_source.startswith("<") and q_source.endswith(">") else q_source
                source_emb = get_embedding(source_for_emb)
            
            content_emb = None
            if q_content and q_content != "?" and isinstance(q_content, str):
                content_emb = get_embedding(q_content)
            
            target_emb = None
            if q_target and q_target != "?" and isinstance(q_target, str):
                target_for_emb = q_target.strip("<>") if q_target.startswith("<") and q_target.endswith(">") else q_target
                target_emb = get_embedding(target_for_emb)
            
            query_triple_embeddings.append([source_emb, content_emb, target_emb])
        
        # Pre-compute spatial constraint embedding if provided
        spatial_embedding = None
        if spatial_constraints:
            if isinstance(spatial_constraints, str):
                spatial_embedding = get_embedding(spatial_constraints)
            elif isinstance(spatial_constraints, dict):
                location = spatial_constraints.get("location")
                scene = spatial_constraints.get("scene")
                if location:
                    spatial_embedding = get_embedding(location)
                elif scene:
                    spatial_embedding = get_embedding(scene)
        
        # Filter low-level edges (clip_id>0, scene is not None)
        candidate_edges = []
        for edge_id, edge in self.edges.items():
            if edge.clip_id > 0 and edge.scene is not None:
                candidate_edges.append(edge)
        
        if not candidate_edges:
            return []
        
        # Score edges based on embedding similarity with bidirectional matching
        # Formula: Similarity = (weight_source*source + weight_content*content + weight_target*target) * scene_similarity
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
            
            # Calculate scene similarity
            scene_sim = 1.0  # Default to 1.0 if no spatial constraint (no penalty)
            if spatial_embedding and edge.scene:
                try:
                    if getattr(edge, "scene_embedding", None) is not None:
                        edge_scene_emb = edge.scene_embedding
                    else:
                        edge_scene_emb = get_embedding(edge.scene)
                    scene_sim = self._cosine_similarity(spatial_embedding, edge_scene_emb)
                except Exception:
                    # Fallback to substring match
                    if isinstance(spatial_constraints, str):
                        if spatial_constraints.lower() in edge.scene.lower():
                            scene_sim = 1.0
                        else:
                            scene_sim = 0.0
                    else:
                        scene_sim = 0.0
            
            # Final score: base_similarity * scene_similarity
            score = base_similarity * scene_sim
            
            scored_edges.append((score, edge))
        
        # Sort by score (descending) and return top-k
        scored_edges.sort(key=lambda x: x[0], reverse=True)
        return [edge for _, edge in scored_edges[:k]]
    
    
    def search_conversations(self, query, k, speaker_strict=None):
        """
        Search for top-k conversation messages using embedding-based similarity.
        
        Args:
            query: Query string (natural language question)
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
        if not query or not isinstance(query, str):
            return []
        
        # Get embedding for query
        try:
            query_embedding = get_embedding(query)
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
                if not isinstance(message, list) or len(message) < 2:
                    continue
                
                speaker = message[0]
                content = message[1]  # content is at index 1
                
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
                    query_lower = query.lower()
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
            
            # Extract messages and format as "[clip_id] Speaker: content"
            message_lines = []
            for idx in sorted_indices:
                if idx < len(conversation.messages):
                    msg = conversation.messages[idx]
                    if isinstance(msg, list) and len(msg) >= 2:
                        speaker = msg[0]
                        content = msg[1]
                        clip_id = msg[2] if len(msg) >= 3 and msg[2] is not None else None
                        
                        # Remove angle brackets from speaker name
                        speaker_name = speaker
                        if speaker_name.startswith("<") and speaker_name.endswith(">"):
                            speaker_name = speaker_name[1:-1]
                        
                        # Format with clip_id: [clip_id] Speaker: content
                        if clip_id is not None:
                            message_lines.append(f"[{clip_id}] {speaker_name}: {content}")
                        else:
                            # Fallback if clip_id is missing
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
