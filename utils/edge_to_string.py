def high_level_edges_to_string(edges):
    """
    Convert a list of high-level edges (character attributes and relationships) to a natural language string.
    
    High-level edges come from search_high_level_edges() and represent:
    - Character attributes: source -> None (target is None)
    - Character relationships: source -> target (target is a character)
    
    Args:
        edges: List of Edge objects from search_high_level_edges()
    
    Returns:
        str: Natural language description of the edges, formatted for LLM understanding.
            Attributes and relationships are separated into distinct sections.
    """
    if not edges:
        return ""
    
    # Separate edges into attributes and relationships
    attributes = []
    relationships = []
    
    for edge in edges:
        # Remove angle brackets from source for readability
        source_name = edge.source
        if source_name.startswith("<") and source_name.endswith(">"):
            source_name = source_name[1:-1]
        
        # Get confidence score (default to None if not available)
        confidence = edge.confidence if hasattr(edge, 'confidence') and edge.confidence is not None else None
        
        if edge.target is None:
            # Character attribute: source -> None
            attributes.append({
                'character': source_name,
                'attribute': edge.content,
                'confidence': confidence
            })
        else:
            # Character relationship: source -> target
            target_name = edge.target
            if target_name.startswith("<") and target_name.endswith(">"):
                target_name = target_name[1:-1]
            
            relationships.append({
                'character1': source_name,
                'character2': target_name,
                'relationship': edge.content,
                'confidence': confidence
            })
    
    # Build natural language string
    lines = []
    
    # Character Attributes section
    if attributes:
        lines.append("Character Attributes:")
        # Group attributes by character
        char_attributes = {}
        for attr in attributes:
            char = attr['character']
            if char not in char_attributes:
                char_attributes[char] = []
            char_attributes[char].append(attr)
        
        for char, attrs in sorted(char_attributes.items()):
            attr_strs = []
            for attr in attrs:
                if attr['confidence'] is not None:
                    attr_strs.append(f"{attr['attribute']} ({attr['confidence']})")
                else:
                    attr_strs.append(attr['attribute'])
            lines.append(f"- {char} is: {', '.join(attr_strs)}")
    
    # Character Relationships section
    if relationships:
        if attributes:
            lines.append("")  # Blank line separator
        lines.append("Character Relationships:")
        
        for rel in relationships:
            if rel['confidence'] is not None:
                line = f"- {rel['character1']} {rel['relationship']} {rel['character2']} ({rel['confidence']})"
            else:
                line = f"- {rel['character1']} {rel['relationship']} {rel['character2']}"
            lines.append(line)
    
    return "\n".join(lines)


def low_level_edge_to_string(edges):
    """
    Convert a list of low-level edges (actions/states) to a natural language string.
    Low-level edges come from search_low_level_edges() and represent specific actions and states.
    
    Args:
        edges: List of Edge objects from search_low_level_edges()
    
    Returns:
        str: Natural language description of actions in temporal order (sorted by edge.id),
             with clip_id and scene information for each line.
    """
    if not edges:
        return ""
    
    # Sort edges by edge.id for temporal order
    sorted_edges = sorted(edges, key=lambda e: e.id)
    
    lines = []
    
    for edge in sorted_edges:
        # Parse and format source node
        source_str = format_node_for_natural_language(edge.source)
        
        # Parse and format target node
        if edge.target is None:
            target_str = ""
        else:
            target_str = format_node_for_natural_language(edge.target)
        
        # Build the action description
        if target_str:
            action_line = f"{source_str} {edge.content} {target_str}"
        else:
            action_line = f"{source_str} {edge.content}"
        
        # Format: [clip_id] action. (scene)
        scene_str = edge.scene if edge.scene else ""
        if scene_str:
            formatted_line = f"[{edge.clip_id}] {action_line}. ({scene_str})"
        else:
            formatted_line = f"[{edge.clip_id}] {action_line}."
        
        lines.append(formatted_line)
    
    return "\n".join(lines)


def format_node_for_natural_language(node_str):
    """
    Format a node string to natural language, handling character nodes and object nodes
    with ownership and attributes.
    
    Object node formats:
    - "object" -> "object"
    - "object@owner" -> "owner's object"
    - "object#attribute" -> "attribute object"
    - "object@owner#attribute" -> "owner's attribute object"
    - "object#attribute@owner" -> "owner's attribute object"
    
    Character node format:
    - "<Character>" -> "Character"
    
    Args:
        node_str: Node string representation
    
    Returns:
        str: Natural language formatted node
    """
    if node_str is None:
        return ""
    
    node_str = str(node_str).strip()
    
    # Check if it's a character node (has angle brackets)
    if node_str.startswith("<") and node_str.endswith(">"):
        # Remove angle brackets
        return node_str[1:-1]
    
    # It's an object node - parse ownership and attributes
    owner = None
    attribute = None
    name = node_str
    
    # Check for @ and # symbols
    at_pos = node_str.find("@")
    hash_pos = node_str.find("#")
    
    if at_pos != -1 and hash_pos != -1:
        # Both @ and # exist - determine order
        if at_pos < hash_pos:
            # Format: "object@owner#attribute"
            parts = node_str.split("@", 1)
            name = parts[0]
            rest = parts[1]
            if "#" in rest:
                owner_part, attribute = rest.split("#", 1)
                owner = owner_part
            else:
                owner = rest
        else:
            # Format: "object#attribute@owner"
            parts = node_str.split("#", 1)
            name = parts[0]
            rest = parts[1]
            if "@" in rest:
                attribute_part, owner = rest.split("@", 1)
                attribute = attribute_part
            else:
                attribute = rest
    elif at_pos != -1:
        # Only @ exists: "object@owner"
        parts = node_str.split("@", 1)
        name = parts[0]
        owner = parts[1]
    elif hash_pos != -1:
        # Only # exists: "object#attribute"
        parts = node_str.split("#", 1)
        name = parts[0]
        attribute = parts[1]
    
    # Remove angle brackets from owner if it's a character reference
    if owner and owner.startswith("<") and owner.endswith(">"):
        owner = owner[1:-1]
    
    # Build natural language string
    parts_list = []
    
    if owner:
        parts_list.append(f"{owner}'s")
    
    if attribute:
        parts_list.append(attribute)
    
    parts_list.append(name)
    
    return " ".join(parts_list)
