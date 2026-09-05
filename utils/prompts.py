
prompt_generate_episodic_memory = """
You are given a 30-second video represented as sequential frames (pictures in chronological order). 

Your tasks: 

1. **Characters' Behavior**
   - Describe each character's behavior in chronological order.
   - Include:
     (a) Interaction with objects in the scene.
     (b) Interaction with other characters.
     (c) Actions and movements.
   - Do NOT repeat the information from the conversation. 
   - When the character interacts with objects in the scene, include precise location information for placement, retrieval, or movement:
     - Furniture/container names: "dressing table", "bedside table", "wardrobe", "shoe cabinet", "refrigerator", "microwave oven"
     - Spatial modifiers: "below", "above", "left side", "right side", "beside", "in front of", "next to", "under", "on the counter"
     - Complete location descriptions: Always combine furniture names with spatial modifiers to form hierarchical locations (e.g., "cabinet below the dressing table", "second layer of the refrigerator", "cabinet on the left side of the wardrobe"). For retrieval actions, include source locations (e.g., "takes towel from Susan's bag", "gets mask from cabinet below dressing table")
   - Each entry must describe exactly one event/detail. Split sentences if needed.
   - Example output:
     ["<Alice> enters the room.", 
      "<Alice> takes cap from the cabinet on the left side of the wardrobe.", 
      "<Alice> sits with <Bob> side by side on the couch.", 
      "<Bob> watches TV.", 
      "<robot> puts the coffee on the table."]

2. **Conversation**
   - Record the dialogue based on subtitles.
   - Always use characters' real name existed in the subtitle.
   - Output format: List of two-element lists [character, content].
   - Example output: [["<Alice>", "Hello, my name is Alice."], ["<Bob>", "Hi, I'm Bob. Nice to meet you."]]

3. **Characters' Appearance**
   - Describe or update each character's appearance: facial features, clothing, body shape, hairstyle, or other distinctive characteristics.
   - Each characteristic should be concise, separated by commas.
   - Existing characters: Update if changes observed (hair, clothing), enhance if new details visible, otherwise keep unchanged. Keep appearance info even if character leaves scene.
   - If a character is not visible in scene, then do not include in the characters' appearance list.
   - Example output:
     [Appearance(name="<Alice>", appearance="female, fat, ponytail, wear glasses, short-sleeved shirt, blue jeans, white sneakers"),
      Appearance(name="<Bob>", appearance="male, thin, short hair, no glasses, black jacket, black pants, black shoes")]

4. **Scene**: Use one word or phrase to describe the scene in the current video (eg. "bedroom", "gym", "office", etc.).

# All the above information should be provided in JSON format.

## Character Naming Rules:
- Use angle brackets to represent characters (eg. <Alice>, <Bob>, <robot>, <character_1> etc.) in behaviors, conversation, and character appearance.
- Include the robot (<robot>) if present:
  - It wears black gloves and has no visible face (it holds the camera).
  - Describe its behavior and conversation.
  - Do NOT include robot in character appearance information, but include it in behaviors and conversation.
- There are two types of characters: named characters and unknown characters.
  - Named characters are characters with a known name (eg. <Alice>, <Bob>, <robot>).
  - Unknown characters are characters with an unknown name (eg. <character_1>, <character_2>, etc.).

## Character Matching Rules:
For characters appearing in the video (MUST follow before creating new characters):
- First, check if the subtitle name is provided. If so, use it as the character name (eg. <Anna>). 
  - Also compare the character name with the unknown characters (<character_X>) in the appearance list. 
    - If high similarity is found, add "Equivalence: <character_X>, <Anna>" at **start of behaviors**. This indicates that the character is the same as the unknown character.
    - If the equivalence is found, refer to this character by its character name instead of <character_X> in the behaviors, conversation, and character appearance.
- If a character's name is not provided in the subtitle, match this character's appearance to the characters in the appearance list. 
  - If high similarity is found, refer to this character by its name in the appearance list. 
  - If no match is found, create a new unknown character with lowest available number starting from <character_1>. 
  - Our goal is to minimize the number of unknown characters. When uncertain, match to existing rather than creating new.

Additional Rules:
- Maintain strict chronological order.
- Avoid repetition in both behavior and conversation.
- If no behavior or conversation is observed, return an empty list for behaviors and conversation.
"""


prompt_extract_triples = """
Convert a JSON array of action sentences into structured triples:
`{"triples": [{"source": ..., "content": ..., "target": ...}]}`.

Graph contract:
- `source` and `target` are stable entity nodes. `content` is only the action or relation between them.
- The affected object of an action must be the target, not part of `content`.
  Correct: `{"source": "<Alice>", "content": "takes", "target": "towel"}`.
  Wrong: `{"source": "<Alice>", "content": "takes towel", "target": "bag"}`.
- Keep target names concise and reusable. Put relative spatial detail in `content`.
  Example: "the pen is in the holder on the right side of the desk" becomes
  `{"source": "pen", "content": "is in the holder on the right side of", "target": "desk"}`.

Extraction rules:
- Preserve sentence order and action order. Split distinct actors, actions, or objects into separate triples.
- Characters keep angle brackets, such as `<Alice>` and `<robot>`. Objects never use angle brackets.
- Use simple-present, verb-centered relations. Preserve modifiers, ownership, direction, and hierarchical location details.
- Resolve pronouns to explicit entity names. Do not invent entities or facts. Keep only distinct facts.
- For placement, emit both the action and resulting state. "<robot> puts coffee on the table" becomes
  `{"source": "<robot>", "content": "puts", "target": "coffee"}` and
  `{"source": "coffee", "content": "is on", "target": "table"}`.
- For retrieval, emit both the action and source relation, but never claim that the object remains there.
  "<Alice> takes towel from Susan's bag" becomes
  `{"source": "<Alice>", "content": "takes", "target": "towel"}` and
  `{"source": "towel", "content": "is taken from", "target": "Susan's bag"}`.
- Use a null target only when an action truly has no affected or related entity.
"""


prompt_summary = """
You are given a sequence of video clips (each clip is 30 seconds long) with scene descriptions and character behaviors.
Your task is to summarize this information into a concise, narrative paragraph.

### INPUT FORMAT
The input consists of multiple clips, each with:
- Clip ID and Scene name
- A list of character behaviors (actions and events)

### OUTPUT REQUIREMENTS
- Write a single, coherent paragraph (3-5 sentences)
- Describe the sequence of events in chronological order
- Include key actions, character interactions, and scene transitions
- Use natural, flowing language (not a bulleted list)
- Focus on the main narrative flow and significant events
- Keep character names as provided (e.g., <character_1>, <robot>)
- Do not include clip numbers or scene labels in the summary
- There might be conflict or misleading information provided, you should be able to handle it and provide a coherent summary.

Now summarize the following clips:
"""


prompt_character_summary = """
You are given a character's name and a list of their behaviors in chronological order.

Your task is to summarize the character's attributes: 
- Personality (eg. confident, nervous)
- Role/profession (eg. host, newcomer) 
- Interests or background (when inferable) 
- Distinctive behaviors or traits (eg. speaks formally, fidgets). 
Avoid restating visual facts—focus on identity construction.

For each attribute, you should also provide a confidence score between 0 and 100. 
If the confidence score is less than 50, you should not include the attribute in the output.

Output a JSON dictionary (key: attribute, value: confidence score). 
Example: {"student": 90, "enthusiastic": 80, "likes to read": 70, "professional": 50, "likes to play games": 60}
"""


prompt_character_relationships = """
You are given a list of character interactions in chronological order.
Your task is to extract the relationships between the characters:
- Roles (eg. friends, colleagues, host-guest, teacher-student, parent-child, etc.)
- Attitudes/Emotions (eg. respect, dislike, friendly, etc.)
- Power dynamics (eg. who leads, equal, etc.)
- Evidence of cooperation
- Exclusion, conflict, competition, etc. 

Additional rules:
- Only store the abstract relationships between the characters.
- Do NOT include any actual actions or summary of actions in the output (eg. <Alice> speaks with <Bob>, <Alice> plays games with <Bob>, etc.). 
- Do not generate repetitive or symmetric information. 

For each relationship, you should also provide a confidence score between 0 and 100.
If the confidence score is less than 50, you should not include the relationship in the output.
It is acceptable to only generate a few relationships if you don't have enough information.

Output a JSON array (list of lists). 
**CRITICAL**: Ensure the output starts with `[` and ends with `]`, and contains all relationship lists within this single outer array.
Each list contains four elements: [character1, relationship, character2, confidence score]. 
Example: [["<Alice>", "is friend with", "<Bob>", 90], ["<Alice>", "is teacher of", "<Charlie>", 80], ["<Charlie>", "respects", "<Alice>", 70]]
"""


prompt_conversation_summary = """
You are given a conversation between several characters.

Your tasks: 

### OUTPUT FORMAT
Return a JSON object with the following keys:
1. "summary": A string summarizing the key topics, decisions, or outcomes (2-4 concise sentences).
2. "character_attributes": A list of [character, attribute, confidence_score] triplets.
3. "characters_relationships": A list of [character1, relationship, character2, confidence_score] quadruplets.

Example:
{
  "summary": "Alice and Bob discussed their upcoming project. They agreed on a timeline and assigned tasks.",
  "character_attributes": [["<Alice>", "organized", 85], ["<Bob>", "cautious", 70]],
  "characters_relationships": [["<Alice>", "is friend with", "<Bob>", 90]]
}

### DETAILED INSTRUCTIONS

1. **Summary**
- Summarize the key topics, decisions, or outcomes discussed in the conversation.
- Write 2-4 concise sentences covering the main themes and important points.
- Focus on what was discussed and decided, not on individual statements.
- Example: "Alice and Bob discussed their upcoming project. They agreed on a timeline and assigned tasks. Bob expressed concerns about the deadline, which Alice addressed by suggesting additional resources."

2. **Character Attributes**
- Extract each character's attributes revealed through their dialogue and interaction style.
- Focus on: personality traits, role/profession, interests, background information (when mentioned).
- **DO NOT** include:
  - Physical appearance or visual characteristics (use appearance data instead)
  - Concrete actions or behaviors (e.g., "asked a question", "walked away")
  - Temporary emotional states (use persistent personality traits instead)
  - Information not directly supported by the conversation
- Confidence scores range from 0-100. Only include attributes with confidence >= 50.
- Avoid redundant or overly similar attributes (e.g., don't include both "friendly" and "kind" unless distinctly different).
- Use angle brackets for character names (e.g., "<Alice>", "<Bob>").
- Output format: List of three-element lists: [character, attribute, confidence_score].
- Example: [["<Alice>", "organized", 85], ["<Alice>", "problem-solver", 75], ["<Bob>", "detail-oriented", 80], ["<Bob>", "cautious", 70]]
- Bad Examples (do not follow): 
  [["<Alice>", "asked a question", 90],  // WRONG: This is an action, not an attribute
  ["<Bob>", "has brown hair", 80]]       // WRONG: This is appearance, not attribute

3. **Character Relationships**
- Extract abstract relationships between characters based on their dialogue interactions.
- Include: roles (friends, colleagues, teacher-student, etc.), attitudes (respect, dislike, etc.), 
  power dynamics, evidence of cooperation/conflict/exclusion/competition.
- **DO NOT** include:
  - Specific actions or events (e.g., "<Alice> speaks with <Bob>", "<Alice> asked <Bob> about X")
  - Temporary interactions (focus on underlying relationship patterns)
  - Dialogue content or topics discussed (focus on the relationship itself, not what they discussed)
- Confidence scores range from 0-100. Only include relationships with confidence >= 50.
- Do not generate symmetric duplicates (if "<Alice> respects <Bob>" is included, don't automatically include reverse unless explicitly different).
- It is acceptable to generate only a few relationships if there is insufficient information.
- Output format: List of four-element lists: [character1, relationship, character2, confidence_score].
- Example: [["<Alice>", "is friend with", "<Bob>", 90], ["<Alice>", "is teacher of", "<Charlie>", 80], ["<Charlie>", "respects", "<Alice>", 70]]
- Bad Examples (do not follow): 
  [["<Alice>", "spoke with", "<Bob>", 90],     // WRONG: This is an action, not a relationship
  ["<Alice>", "discussed the project", "<Bob>", 85]]  // WRONG: This is dialogue content, not relationship

### EDGE CASES
- If conversation has only one character speaking: focus on their attributes, skip relationships.
- If conversation is empty or unclear: return empty arrays for attributes and relationships, provide a brief summary noting the issue.
- If character names are ambiguous: use the names as provided in the conversation.

Now summarize the following conversation in JSON format:
"""


#--------------------------------
# Reasoning Prompts
#--------------------------------

prompt_parse_query = """
You are a query parser for a knowledge graph system that stores video information in a hierarchical structure.

## GRAPH STRUCTURE

**HIGH-LEVEL EDGES**: Character attributes/relationships
- Format: `["<Alice>", "confident", null]` or `["<Alice>", "is friend with", "<Bob>"]`
- **Limited quantity** (<10 per query) - allocate 5-10 max when needed, fewer otherwise
- Use for: character traits, relationships, "who is" queries

**APPEARANCE EDGES**: Character appearance features
- Format: `["<Alice>", "wears red hoodie", null]`
- Use ONLY when question asks about appearance (looks, clothing, hairstyle, body shape, facial features, accessories).
- Do NOT use for behavior/action/location questions.

**LOW-LEVEL EDGES**: Specific actions/states with scene info
- Format: `["<Alice>", "picks up", "coffee"]` or `["coffee", "is on", "table"]`
- Most abundant source - allocate 30-45 for action-focused queries
- Use for: specific actions, temporal/spatial queries ("what did X do", "where is X")

**CONVERSATIONS**: Dialogue transcripts `[speaker, text]` pairs
- Allocate 10-45 based on query needs
- Use for: "why" questions, dialogue content, causal reasoning

## YOUR TASK

Given a query and budget `k=50`, output in JSON format:

1. **Query triple(s)**: Output **`query_triples`** as a list of 1 to 3 triples.
   - Triple format: `[source, content, target, source_weight, content_weight, target_weight]`
   - Use `null` for missing components, normalize to graph format (angle brackets for characters)
   - If the question is complex (needs an extra constraint), split into:
     - **Main triple** (first in the list): the core ask with higher weights
     - **Helper triple** (second in the list): supporting constraint with lower weights
     - **Additional triple** (third in the list): additional information with lowest weights
   - **Assign weights** (0.0-1.0):

**Weight Rules**:
- **High (0.7-1.0)**: Specific character/object names (e.g., "Anna", "coffee", "the red cup") - use 0.9-1.0 for critical entities
- **Medium (0.4-0.7)**: General objects/locations (e.g., "cup", "room") - use 0.5-0.7 for context
- **Low (0.1-0.4)**: What we're searching for - question marks ("?"), relationship terms ("relationship", "friendship"), unknown actions - use 0.2-0.4 for search targets, 0.1-0.2 for vague terms

**Special Rules for Location Queries**:
- **Object-first weighting is mandatory**: If the question centers on a concrete object, give that object's source or target weight `0.9-1.0`. Unknown actors, unknown locations, and generic relation words must not outweigh the object.
- **Cover both states and transitions**: When useful, emit one state triple such as `[X, "is at", "?", 1.0, 0.3, 0.0]` and one transition triple such as `["?", "puts or moves", X, 0.0, 0.3, 1.0]`. This lets retrieval find both explicit locations and the action that created the state.
- **Preserve hierarchical locations**: When parsing location queries, keep complete hierarchical location phrases as single entities in target fields (e.g., "cabinet on the left side of the wardrobe", "cabinet below the dressing table", "table on the right of the water dispenser"). Do NOT split them into separate components.
- **Temporal-spatial queries**: 
  - "where is X now?" → Use triple `[X, "is at", "?", ...]` with high weight on X. The search should prioritize the most recent state edges (highest clip_id).
  - "last time" / "last place" → Use triple `[X, "is at", "?", ...]` and prioritize edges with highest clip_id values.
  - "where should X be placed?" → Use triple `[X, "should be placed at", "?", ...]` or `[X, "is placed at", "?", ...]` to find placement instructions.
- **Source location queries**: "where can robot get X?" / "where did X get Y from?" → Use triple `[X, "gets", "Y", ...]` or `[Y, "is in", "?", ...]` to find source locations. Include a helper triple if needed: `[Y, "is from", "?", ...]`.
- **Allocation for location queries**: Prioritize low-level edges (35-45) since they contain spatial information. Use conversations (5-10) only if placement instructions might be mentioned in dialogue.

2. **Allocation** `{k_high_level, k_low_level, k_conversations, k_appearance}`:
   - Total must be ≤ 50
   - High-level: 5-10 max (limited availability)
   - Low-level: 30-45 for action queries
   - Conversations: 10-45 based on needs
   - Appearance:
     - If the question is about character appearance, clothing, hairstyle, facial/body features, accessories:
       allocate `k_appearance` > 0 (typically 5-15).
     - Otherwise, `k_appearance` MUST be 0.
   - `total_k` must equal: `k_high_level + k_low_level + k_conversations + k_appearance`

3. **speaker_strict**: 
   - Set to `["<Anna>", "<Susan>"]` when query asks about dialogue between specific speakers
   - Set to `null` otherwise

4. **spatial_constraint**: Location string only for general spaces (e.g., gym, office, kitchen, bedroom, living room, meeting room). Do NOT use objects or furniture (e.g., table, dressing table, sofa) as spatial constraints. Otherwise `null`.

## EXAMPLES

**Example 1**: "What is Anna's relationship with Susan?"
ParseQueryOutput(
  query_triples=[["<Anna>", "relationship", "<Susan>", 0.95, 0.2, 0.95]],
  spatial_constraint=None,
  speaker_strict=None,
  allocation=ParseQueryAllocation(
    k_high_level=10, k_low_level=10, k_conversations=30, k_appearance=0, total_k=50,
    reasoning="Relationship query - use high-level for relationships, conversations for evidence"
  )
)

**Example 2**: "What did Emma do with the coffee in the kitchen?"
ParseQueryOutput(
  query_triples=[["<Emma>", "?", "coffee", 0.95, 0.15, 0.9]],
  spatial_constraint="kitchen",
  speaker_strict=None,
  allocation=ParseQueryAllocation(
    k_high_level=5, k_low_level=38, k_conversations=7, k_appearance=0, total_k=50,
    reasoning="Action query - prioritize low-level edges"
  )
)

**Example 3**: "What did Emily and David discuss?"
ParseQueryOutput(
  query_triples=[["<Emily>", "discusses", "<David>", 0.9, 0.3, 0.9]],
  spatial_constraint=None,
  speaker_strict=["<Emily>", "<David>"],
  allocation=ParseQueryAllocation(
    k_high_level=2, k_low_level=3, k_conversations=45, k_appearance=0, total_k=50,
    reasoning="Dialogue query - prioritize conversations with specific speakers"
  )
)

**Example 4**: "How many things on the dressing table are not often used by Lily?"
ParseQueryOutput(
  query_triples=[
    ["<Lily>", "use", "?", 0.9, 0.7, 0.4],
    ["?", "is on", "dressing table", 0.2, 0.4, 0.4]
  ],
  spatial_constraint=None,
  speaker_strict=None,
  allocation=ParseQueryAllocation(
    k_high_level=2, k_low_level=40, k_conversations=8, k_appearance=0, total_k=50,
    reasoning="Main triple targets usage by Lily; helper triple constrains items to dressing table"
  )
)

**Example 5**: "where is the tape now?"
ParseQueryOutput(
  query_triples=[
    ["tape", "is at", "?", 1.0, 0.3, 0.0],
    ["?", "puts or moves", "tape", 0.0, 0.3, 1.0]
  ],
  spatial_constraint=None,
  speaker_strict=None,
  allocation=ParseQueryAllocation(
    k_high_level=2, k_low_level=42, k_conversations=6, k_appearance=0, total_k=50,
    reasoning="Temporal-spatial query - 'now' means most recent location. Prioritize low-level edges with highest clip_id to find current state"
  )
)

Now parse the following query and allocate k=50 in JSON format:
"""


prompt_parse_query_k30 = """
You are a query parser for a knowledge graph system that stores video information in a hierarchical structure.

## GRAPH STRUCTURE

**HIGH-LEVEL EDGES**: Character attributes/relationships
- Format: `["<Alice>", "confident", null]` or `["<Alice>", "is friend with", "<Bob>"]`
- **Limited quantity** (<10 per query) - allocate 5-10 max when needed, fewer otherwise
- Use for: character traits, relationships, "who is" queries

**APPEARANCE EDGES**: Character appearance features
- Format: `["<Alice>", "wears red hoodie", null]`
- Use ONLY when question asks about appearance (looks, clothing, hairstyle, body shape, facial features, accessories).
- Do NOT use for behavior/action/location questions.

**LOW-LEVEL EDGES**: Specific actions/states with scene info
- Format: `["<Alice>", "picks up", "coffee"]` or `["coffee", "is on", "table"]`
- Most abundant source - allocate 15-24 for action-focused queries
- Use for: specific actions, temporal/spatial queries ("what did X do", "where is X")

**CONVERSATIONS**: Dialogue transcripts `[speaker, text]` pairs
- Allocate 5-24 based on query needs
- Use for: "why" questions, dialogue content, causal reasoning

## YOUR TASK

Given a query and budget `k=30`, output the following in JSON format:

1. **Query triple(s)**: Output **`query_triples`** as a list of 1 to 3 triples.
   - Triple format: `[source, content, target, source_weight, content_weight, target_weight]`
   - Use `null` for missing components, normalize to graph format (angle brackets for characters)
   - If the question is complex (needs an extra constraint), split into:
     - **Main triple** (first in the list): the core ask with higher weights
     - **Helper triple** (second in the list): supporting constraint with lower weights
     - **Additional triple** (third in the list): additional information with lowest weights
   - **Assign weights** (0.0-1.0):

**Weight Rules**:
- **High (0.7-1.0)**: Specific character/object names (e.g., "Anna", "coffee", "the red cup") - use 0.9-1.0 for critical entities
- **Medium (0.4-0.7)**: General objects/locations (e.g., "cup", "room") - use 0.5-0.7 for context
- **Low (0.1-0.4)**: What we're searching for - question marks ("?"), relationship terms ("relationship", "friendship"), unknown actions - use 0.2-0.4 for search targets, 0.1-0.2 for vague terms

**Special Rules for Location Queries**:
- **Preserve hierarchical locations**: When parsing location queries, keep complete hierarchical location phrases as single entities in target fields (e.g., "cabinet on the left side of the wardrobe", "cabinet below the dressing table", "table on the right of the water dispenser"). Do NOT split them into separate components.
- **Temporal-spatial queries**:
  - "where is X now?" -> Use triple `[X, "is at", "?", ...]` with high weight on X. The search should prioritize the most recent state edges (highest clip_id).
  - "last time" / "last place" -> Use triple `[X, "is at", "?", ...]` and prioritize edges with highest clip_id values.
  - "where should X be placed?" -> Use triple `[X, "should be placed at", "?", ...]` or `[X, "is placed at", "?", ...]` to find placement instructions.
- **Source location queries**: "where can robot get X?" / "where did X get Y from?" -> Use triple `[X, "gets", "Y", ...]` or `[Y, "is in", "?", ...]` to find source locations. Include a helper triple if needed: `[Y, "is from", "?", ...]`.
- **Allocation for location queries**: Prioritize low-level edges (18-24) since they contain spatial information. Use conversations (3-6) only if placement instructions might be mentioned in dialogue.

2. **Allocation** `{k_high_level, k_low_level, k_conversations, k_appearance}`:
   - Total must be <= 30
   - High-level: 5-10 max (limited availability)
   - Low-level: 15-24 for action queries
   - Conversations: 5-24 based on needs
   - Appearance:
     - If the question is about character appearance, clothing, hairstyle, facial/body features, accessories:
       allocate `k_appearance` > 0 (typically 3-10).
     - Otherwise, `k_appearance` MUST be 0.
   - `total_k` must equal: `k_high_level + k_low_level + k_conversations + k_appearance`

3. **speaker_strict**:
   - Set to `["<Anna>", "<Susan>"]` when query asks about dialogue between specific speakers
   - Set to `null` otherwise

4. **spatial_constraint**: Location string only for general spaces (e.g., gym, office, kitchen, bedroom, living room, meeting room). Do NOT use objects or furniture (e.g., table, dressing table, sofa) as spatial constraints. Otherwise `null`.

## EXAMPLES

**Example 1**: "What is Anna's relationship with Susan?"
ParseQueryOutput(
  query_triples=[["<Anna>", "relationship", "<Susan>", 0.95, 0.2, 0.95]],
  spatial_constraint=None,
  speaker_strict=None,
  allocation=ParseQueryAllocation(
    k_high_level=8, k_low_level=5, k_conversations=17, k_appearance=0, total_k=30,
    reasoning="Relationship query - use high-level for relationships, conversations for evidence"
  )
)

**Example 2**: "What did Emma do with the coffee in the kitchen?"
ParseQueryOutput(
  query_triples=[["<Emma>", "?", "coffee", 0.95, 0.15, 0.9]],
  spatial_constraint="kitchen",
  speaker_strict=None,
  allocation=ParseQueryAllocation(
    k_high_level=5, k_low_level=21, k_conversations=4, k_appearance=0, total_k=30,
    reasoning="Action query - prioritize low-level edges"
  )
)

**Example 3**: "What did Emily and David discuss?"
ParseQueryOutput(
  query_triples=[["<Emily>", "discusses", "<David>", 0.9, 0.3, 0.9]],
  spatial_constraint=None,
  speaker_strict=["<Emily>", "<David>"],
  allocation=ParseQueryAllocation(
    k_high_level=5, k_low_level=2, k_conversations=23, k_appearance=0, total_k=30,
    reasoning="Dialogue query - prioritize conversations with specific speakers"
  )
)

**Example 4**: "How many things on the dressing table are not often used by Lily?"
ParseQueryOutput(
  query_triples=[
    ["<Lily>", "use", "?", 0.9, 0.7, 0.4],
    ["?", "is on", "dressing table", 0.2, 0.4, 0.4]
  ],
  spatial_constraint=None,
  speaker_strict=None,
  allocation=ParseQueryAllocation(
    k_high_level=5, k_low_level=20, k_conversations=5, k_appearance=0, total_k=30,
    reasoning="Main triple targets usage by Lily; helper triple constrains items to dressing table"
  )
)

**Example 5**: "where is the tape now?"
ParseQueryOutput(
  query_triples=[["tape", "is at", "?", 0.8, 0.5, 0.15]],
  spatial_constraint=None,
  speaker_strict=None,
  allocation=ParseQueryAllocation(
    k_high_level=5, k_low_level=21, k_conversations=4, k_appearance=0, total_k=30,
    reasoning="Temporal-spatial query - 'now' means most recent location. Prioritize low-level edges with highest clip_id to find current state"
  )
)

Now parse the following query and allocate k=30 in JSON format:
"""


prompt_parse_query_no_allocation = """
You are a query parser for a knowledge graph system that stores video information in a hierarchical structure.

## YOUR TASK

Given a query, output the following in JSON format:

1. **Query triple(s)**: Output **`query_triples`** as a list of 1 to 3 triples.
   - Triple format: `[source, content, target, source_weight, content_weight, target_weight]`
   - Use `null` for missing components, normalize to graph format (angle brackets for characters)
   - If the question is complex (needs an extra constraint), split into:
     - **Main triple** (first in the list): the core ask with higher weights
     - **Helper triple** (second in the list): supporting constraint with lower weights
     - **Additional triple** (third in the list): additional information with lowest weights
   - **Assign weights** (0.0-1.0):

**Weight Rules**:
- **High (0.7-1.0)**: Specific character/object names (e.g., "Anna", "coffee", "the red cup") - use 0.9-1.0 for critical entities
- **Medium (0.4-0.7)**: General objects/locations (e.g., "cup", "room") - use 0.5-0.7 for context
- **Low (0.1-0.4)**: What we're searching for - question marks ("?"), relationship terms ("relationship", "friendship"), unknown actions - use 0.2-0.4 for search targets, 0.1-0.2 for vague terms

**Special Rules for Location Queries**:
- **Preserve hierarchical locations**: When parsing location queries, keep complete hierarchical location phrases as single entities in target fields (e.g., "cabinet on the left side of the wardrobe", "cabinet below the dressing table", "table on the right of the water dispenser"). Do NOT split them into separate components.
- **Temporal-spatial queries**:
  - "where is X now?" -> Use triple `[X, "is at", "?", ...]` with high weight on X.
  - "last time" / "last place" -> Use triple `[X, "is at", "?", ...]`.
  - "where should X be placed?" -> Use triple `[X, "should be placed at", "?", ...]` or `[X, "is placed at", "?", ...]`.
- **Source location queries**: "where can robot get X?" / "where did X get Y from?" -> Use triple `[X, "gets", "Y", ...]` or `[Y, "is in", "?", ...]`. Include a helper triple if needed: `[Y, "is from", "?", ...]`.

2. **speaker_strict**:
   - Set to `["<Anna>", "<Susan>"]` when query asks about dialogue between specific speakers
   - Set to `null` otherwise

3. **spatial_constraint**: Location string only for general spaces (e.g., gym, office, kitchen, bedroom, living room, meeting room). Do NOT use objects or furniture (e.g., table, dressing table, sofa) as spatial constraints. Otherwise `null`.

## EXAMPLES

**Example 1**: "What is Anna's relationship with Susan?"
query_triples=[["<Anna>", "relationship", "<Susan>", 0.95, 0.2, 0.95]] spatial_constraint=None speaker_strict=None

**Example 2**: "What did Emma do with the coffee in the kitchen?"
query_triples=[["<Emma>", "?", "coffee", 0.95, 0.15, 0.9]] spatial_constraint="kitchen" speaker_strict=None

**Example 3**: "How many things on the dressing table are not often used by Lily?"
query_triples=[
  ["<Lily>", "use", "?", 0.9, 0.7, 0.4],
  ["?", "is on", "dressing table", 0.2, 0.4, 0.4]
] spatial_constraint=None speaker_strict=None

Now parse the following query in JSON format:
"""


prompt_graph_episodic = """
You are a reasoning system that evaluates whether information extracted from a knowledge graph is sufficient to answer a question.

The system processes video information in three layers:
1. **Video**: Videos are split into 30-second segments, each assigned a unique clip_id (1, 2, 3, ...)
2. **Text**: Each segment's text descriptions (behaviors, conversations, scenes) are stored by clip_id
3. **Graph**: Text is converted into graph edges with two types:
   - **High-level** : Abstract attributes/relationships
   - **Low-level** : Specific actions/states with temporal and spatial information
   Each edge's clip_id links back to its original video segment. 
All the current information provided is from the graph.

Input format: 
1. **Parentheses (X)**: Confidence scores (0-100) in high-level information, indicating reliability.
  Example: Anna is: health-conscious (80) means 80% confidence.
2. **Square brackets [X]**: Clip IDs indicating timestamps. Each clip = 30 seconds: clip 1 = 0-30s, clip 2 = 30-60s, clip 3 = 60-90s, etc.
  Applies to both low-level actions and conversation messages.
  Example: [1] Anna walk. (ping-pong room) means this occurred during clip 1 (0-30 seconds).

Decision criteria: 
1. Answer directly ([Answer]) when the current information provides a clear, complete answer.
2. Search text memory ([Search]) when the current information is incomplete or ambiguous.

Output the answer in the format: 
Action: [Answer] or [Search]
Content: <your answer here> or <updated query>

If the action is [Search], provide an updated query that would help retrieve the missing information. The query should be more specific or focus on the aspects that are unclear or missing from the current search results. Use natural language and be precise about what information you need.

Examples:

Question: Who is the best friend of Alice?
Output:
Action: [Answer]
Content: Bob is Alice's best friend.

Question: Why did Alice leave the room?
Output:
Action: [Search]
Content: What happened before Alice left the room that caused her to leave?
"""


prompt_graph_video = """
You are a reasoning system that evaluates whether information extracted from a knowledge graph is sufficient to answer a question.

You will be provided with extracted knowledge from the video graph, including three components: high-level information (character attributes/relationships), low-level information (actions/states), and conversations.

Input format: 
- **Parentheses (X)**: Confidence scores (0-100) in high-level information, indicating reliability.
  Example: Anna is: health-conscious (80) means 80% confidence.
- **Square brackets [X]**: Clip IDs indicating timestamps. Each clip = 30 seconds: clip 1 = 0-30s, clip 2 = 30-60s, clip 3 = 60-90s, etc.
  Applies to both low-level actions and conversation messages.
  Example: [1] Anna walk. (ping-pong room) means this occurred during clip 1 (0-30 seconds).

Output should include the following in JSON format: 
1. answer: True or False
2. content: <your answer here> or [clip_id1, clip_id2, ...]
3. summary: <only present when answer is False - summary of extracted information from the graph>

**Decision criteria**: 
1. Answer directly when the current graph information provides a clear answer to the question. You should make reasonable deductions and inferences from the available information when appropriate. If the information is sufficient to answer the question (even if not explicitly stated verbatim), choose Answer.
  When the graph allows a reasonable one-sentence answer (even if not perfect), prefer Answer (answer=True). Only choose Search video when the graph is clearly insufficient (e.g. missing key entity, no location detail, or count cannot be inferred).
  Output should be: 
  - answer: True
  - content: Provide a concise, direct answer in ONE SENTENCE. Be brief and to the point. Do NOT include additional explanations or context beyond what is necessary to answer the question.
  - summary: None

2. Search video memory when the extracted information is insufficient or ambiguous and cannot support a reasonable answer through deduction.
  Output should be:
  - answer: False
  - content: Provide a list of video clip IDs (as integers) ranked by relevance: [clip_id1, clip_id2, ...]
  - summary: Provide a concise summary of extracted graph information relevant to the question, including key events, character information, conversations, and temporal/spatial context.

Special types of questions:
1. Spatial/Location questions (questions asking "where" or "which place"):
  - answer is True only if the location information includes specific furniture, containers, or precise spatial relationships. Generic room names alone are INSUFFICIENT.
  - If the location information is generic (e.g., "kitchen", "office", "living room"), answer is False:
    - content: Provide a list of video clip IDs (as integers) ranked by relevance: [clip_id1, clip_id2, ...]. 
    - Include the clips where the actions occured, and the clips before and after the actions if neccessary. 
    - summary: Focus the summary on object locations, character actions involving the object, spatial relationships and temporal sequences. Exclude irrelevant information such as character attributes, relationships, and conversations.

2. Temporal Sequence questions (questions asking about event order, sequence, "first", "before", "after", "should X be done first", "what happened before/after X", etc.):
  - answer is True only if the graph clearly shows the temporal sequence with sufficient detail (e.g., clip IDs show clear sequence).
  - If the temporal sequence is unclear, ambiguous, or missing key events, answer is False.
  - **Distinguish**: "Should X be done first?" asks about INSTRUCTIONS/intended order, while "What happened before/after X?" asks about ACTUAL sequence.
  - If answer is False:
    - content: Focus on clips BEFORE or AFTER the key action/event based on the temporal context of the question. If information is insufficient, prioritize clips that occur BEFORE the key action.
    - summary: MUST clearly indicate temporal information with explicit clip IDs. Format: "In clip [X], [character/event] [action]." Use this format for each relevant event in chronological order.
    - Focus the summary on events in chronological order with explicit clip IDs, character actions and their sequence, and what happened before/after the key action. Exclude irrelevant information.

3. Counting questions (questions asking "how many", "how many times", "how many pieces", "how many kinds", etc.):
  - answer is True only if the graph provides explicit counts or all occurrences can be clearly enumerated from the graph information.
  - If counts are vague, incomplete, or if multiple occurrences might be missed, then answer is False: 
    - content: Include ALL clips where the mentioned event takes place. Also include clip IDs between two mentioned events if necessary to ensure no counting is missed during the information storage step.
    - summary: MUST clearly indicate counts per clip with explicit clip IDs. Format: "In clip [X], [character/event] [action] [count]." Example: "In clip 18, A did something once; in clip 20, A did something twice."
    - Focus the summary on enumerating each occurrence with its clip ID and count, ensuring all events are accounted for. Exclude irrelevant information.

Examples:

Question: What is the relationship between Anna and Susan?
Extracted information: High-level: Anna competes with Susan (85). Anna is competitive (90). Susan is competitive (88). Low-level: [12] Anna challenges Susan to a game. [15] Anna and Susan prepare for competition. Conversations: Conversation 1: Anna and Susan discuss their upcoming game, with Anna expressing confidence in winning.
Output:
GraphVideoOutput(
  answer=True,
  content="Anna and Susan are competitors.",
  summary=None
)

Question: What did Anna decide to drink before the game?
Extracted information: High-level: Anna is health-conscious (80). Anna prefers water (85). Low-level: [15] Anna picks up Anna's water bottle. [16] Susan picks up Susan's sports drink. Conversations: Conversation 2: Anna says "I just want a bottle of water. That's fine. No sports soda for me." Susan responds "you never drink sports soda, and just mineral water."
Output:
GraphVideoOutput(
  answer=True,
  content="Anna decided to drink water before the game.",
  summary=None
)

Question: What happened after Alice received the gift?
Extracted information: High-level: (no sequence information) Low-level: [15] Bob gives Alice wrapped gift box. [16] Alice unwraps gift box. [17] Alice reads book. Conversations: (no relevant conversations)
Output:
GraphVideoOutput(
  answer=True,
  content="After receiving the gift, Alice unwrapped it and then read the book.",
  summary=None
)

Question: Where is the book Lucky read just now?
Extracted information: High-level: (no location information) Low-level: [8] Lucky reads book. (bedroom) [9] Lucky places book. (bedroom) Conversations: (no relevant conversations)
Output:
GraphVideoOutput(
  answer=False,
  content=[8, 9, 7],
  summary="The graph shows Lucky reading a book at clip 8 and placing it at clip 9, both in the bedroom. However, the specific location within the bedroom (e.g., which furniture or surface) is not captured in the graph. In clip 8, Lucky reads the book. In clip 9, Lucky places the book. The exact placement location (e.g., bedside table, desk, shelf) requires visual inspection of the video frames."
)

Question: Should the balloons be put up first?
Extracted information: High-level: (no sequence instructions) Low-level: [10] Betty instructs to put up balloons. (living room) [12] Betty and Linda write message on balloons. (living room) [14] Betty and Linda put up balloons. (living room) Conversations: (no relevant conversations)
Output:
GraphVideoOutput(
  answer=False,
  content=[10, 12, 14, 11, 13],
  summary="The graph shows multiple events but the temporal sequence is unclear. In clip 10, Betty instructs to put up balloons. In clip 12, Betty and Linda write message on balloons. In clip 14, Betty and Linda put up balloons. However, the graph does not clearly indicate whether the instruction in clip 10 specifies the order, or what happens before putting up the balloons. The sequence requires visual verification to determine the intended order."
)

Question: How many times was the air-conditioning remote used?
Extracted information: High-level: (no count information) Low-level: [8] Robot uses air-conditioning remote. (meeting room) [11] Robot uses air-conditioning remote. (meeting room) Conversations: (no relevant conversations)
Output:
GraphVideoOutput(
  answer=False,
  content=[8, 11, 9, 10],
  summary="The graph shows the air-conditioning remote being used at clip 8 and clip 11. However, to ensure accurate counting and verify no uses were missed between these clips, all clips from 8 to 11 should be checked. In clip 8, the remote was used once; in clip 11, the remote was used once. Clips 9-10 are included to ensure no counting is missed during the information storage step."
)
"""


prompt_no_video_rewatch = """
You are a reasoning system that answers the question based on the searched information from a video.

You will be provided with the extracted knowledge from the video graph, including three components: high-level information (character attributes/relationships), low-level information (actions/states), and conversations.

Input format: 
- **Parentheses (X)**: Confidence scores (0-100) in high-level information, indicating reliability.
  Example: Anna is: health-conscious (80) means 80% confidence.
- **Square brackets [X]**: Clip IDs indicating timestamps. Each clip = 30 seconds: clip 1 = 0-30s, clip 2 = 30-60s, clip 3 = 60-90s, etc.
  Applies to both low-level actions and conversation messages.
  Example: [1] Anna walk. (ping-pong room) means this occurred during clip 1 (0-30 seconds).

Output: Provide a concise, direct answer in ONE SENTENCE. Be brief and to the point. Do NOT include additional explanations or context beyond what is necessary to answer the question.
Answers like "I don't know" or "The information is not sufficient to answer the question" are NOT allowed. You can guess the answer based on the information provided.
"""


# Ablation: no high-level - graph search excludes character attributes/relationships
prompt_graph_video_no_highlevel = """
You are a reasoning system that evaluates whether information extracted from a knowledge graph is sufficient to answer a question.

You will be provided with extracted knowledge from the video graph, including characters' behaviors and conversations.

Input format:
- **Square brackets [X]**: Clip IDs indicating timestamps. Each clip = 30 seconds: clip 1 = 0-30s, clip 2 = 30-60s, clip 3 = 60-90s, etc.
  Applies to both low-level actions and conversation messages.
  Example: [1] Anna walk. (ping-pong room) means this occurred during clip 1 (0-30 seconds).

Output should include the following in JSON format: 
1. answer: True or False
2. content: <your answer here> or [clip_id1, clip_id2, ...]
3. summary: <only present when answer is False - summary of extracted information from the graph>

**Decision criteria**: 
1. Answer directly when the current graph information provides a clear answer to the question. You should make reasonable deductions and inferences from the available information when appropriate. If the information is sufficient to answer the question (even if not explicitly stated verbatim), choose Answer.
  Output should be: 
  - answer: True
  - content: Provide a concise, direct answer in ONE SENTENCE. Be brief and to the point. Do NOT include additional explanations or context beyond what is necessary to answer the question.
  - summary: None

2. Search video memory when the extracted information is insufficient or ambiguous and cannot support a reasonable answer through deduction.
  Output should be:
  - answer: False
  - content: Provide a list of video clip IDs (as integers) ranked by relevance: [clip_id1, clip_id2, ...]
  - summary: Provide a concise summary of extracted graph information relevant to the question, including key events, character information, conversations, and temporal/spatial context.

Special types of questions:
1. Spatial/Location questions (questions asking "where" or "which place"):
  - answer is True only if the location information includes specific furniture, containers, or precise spatial relationships. Generic room names alone are INSUFFICIENT.
  - If the location information is generic (e.g., "kitchen", "office", "living room"), answer is False.
  - If answer is False:
    - Provide a list of video clip IDs (as integers) ranked by relevance: [clip_id1, clip_id2, ...].
    - Include the clips where the actions occured, and the clips before and after the actions if neccessary.
    - Focus the summary on object locations, character actions involving the object, spatial relationships and temporal sequences.

2. Temporal Sequence questions (questions asking about event order, sequence, "first", "before", "after", "should X be done first", "what happened before/after X", etc.):
  - answer is True only if the graph clearly shows the temporal sequence with sufficient detail (e.g., clip IDs show clear sequence).
  - If the temporal sequence is unclear, ambiguous, or missing key events, answer is False.
  - **Distinguish**: "Should X be done first?" asks about INSTRUCTIONS/intended order, while "What happened before/after X?" asks about ACTUAL sequence.
  - If answer is False:
    - **Clip selection priority**: Focus on clips BEFORE or AFTER the key action/event based on the temporal context of the question. If information is insufficient, prioritize clips that occur BEFORE the key action.
    - **Summary format**: MUST clearly indicate temporal information with explicit clip IDs. Format: "In clip [X], [character/event] [action]." Use this format for each relevant event in chronological order.
    - Focus the summary on events in chronological order with explicit clip IDs, character actions and their sequence, and what happened before/after the key action.

3. Counting questions (questions asking "how many", "how many times", "how many pieces", "how many kinds", etc.):
  - answer is True only if the graph provides explicit counts or if all occurrences can be clearly enumerated from the graph information.
  - If counts are vague, incomplete, or if multiple occurrences might be missed, then answer is False.
  - If answer is False:
    - **Clip selection**: Include ALL clips where the mentioned event takes place. Also include clip IDs between two mentioned events if necessary to ensure no counting is missed during the information storage step.
    - **Summary format**: MUST clearly indicate counts per clip with explicit clip IDs. Format: "In clip [X], [character/event] [action] [count]." Example: "In clip 18, A did something once; in clip 20, A did something twice."
    - Focus the summary on enumerating each occurrence with its clip ID and count, ensuring all events are accounted for.

Examples:

Question: What did Anna decide to drink before the game?
Extracted information: High-level: (not available) Low-level: [15] Anna picks up Anna's water bottle. [16] Susan picks up Susan's sports drink. Conversations: Anna says "I just want a bottle of water. That's fine. No sports soda for me." Susan responds "you never drink sports soda, and just mineral water."
Output:
GraphVideoOutput(
  answer=True,
  content="Anna decided to drink water before the game.",
  summary=None
)

Question: What happened after Alice received the gift?
Extracted information: High-level: (not available) Low-level: [15] Bob gives Alice wrapped gift box. [16] Alice unwraps gift box. [17] Alice reads book. Conversations: (no relevant conversations)
Output:
GraphVideoOutput(
  answer=True,
  content="After receiving the gift, Alice unwrapped it and then read the book.",
  summary=None
)

Question: What is the relationship between Anna and Susan?
Extracted information: High-level: (not available) Low-level: [12] Anna challenges Susan to a game. [15] Anna and Susan prepare for competition. Conversations: Anna and Susan discuss their upcoming game, with Anna expressing confidence in winning.
Output:
GraphVideoOutput(
  answer=False,
  content=[12, 15, 14, 16],
  summary="The graph shows low-level actions but does not contain character relationships (high-level excluded). The relationship between Anna and Susan cannot be inferred from actions and conversations alone. Clips 12 and 15 show competitive preparation; additional clips may reveal their relationship dynamics."
)

Question: Where is the book Lucky read just now?
Extracted information: High-level: (not available) Low-level: [8] Lucky reads book. (bedroom) [9] Lucky places book. (bedroom) Conversations: (no relevant conversations)
Output:
GraphVideoOutput(
  answer=False,
  content=[8, 9, 7],
  summary="The graph shows Lucky reading a book at clip 8 and placing it at clip 9, both in the bedroom. However, the specific location within the bedroom (e.g., which furniture or surface) is not captured in the graph. In clip 8, Lucky reads the book. In clip 9, Lucky places the book. The exact placement location (e.g., bedside table, desk, shelf) requires visual inspection of the video frames."
)

Question: Should the balloons be put up first?
Extracted information: High-level: (not available) Low-level: [10] Betty instructs to put up balloons. (living room) [12] Betty and Linda write message on balloons. (living room) [14] Betty and Linda put up balloons. (living room) Conversations: (no relevant conversations)
Output:
GraphVideoOutput(
  answer=False,
  content=[10, 12, 14, 11, 13],
  summary="The graph shows multiple events but the temporal sequence is unclear. In clip 10, Betty instructs to put up balloons. In clip 12, Betty and Linda write message on balloons. In clip 14, Betty and Linda put up balloons. However, the graph does not clearly indicate whether the instruction in clip 10 specifies the order, or what happens before putting up the balloons. The sequence requires visual verification to determine the intended order."
)

Question: How many times was the air-conditioning remote used?
Extracted information: High-level: (not available) Low-level: [8] Robot uses air-conditioning remote. (meeting room) [11] Robot uses air-conditioning remote. (meeting room) Conversations: (no relevant conversations)
Output:
GraphVideoOutput(
  answer=False,
  content=[8, 11, 9, 10],
  summary="The graph shows the air-conditioning remote being used at clip 8 and clip 11. However, to ensure accurate counting and verify no uses were missed between these clips, all clips from 8 to 11 should be checked. In clip 8, the remote was used once; in clip 11, the remote was used once. Clips 9-10 are included to ensure no counting is missed during the information storage step."
)

Now evaluate the following:
"""


prompt_video_answer = """
You are given a 30-second video clip represented as sequential frames (pictures in chronological order) and a question.

**Important**: 
- You may also receive summaries from previous video clips that have already been watched. These summaries contain information from earlier clips and are provided to help answer questions that require information spanning multiple video clips.
- You will receive the current clip ID. Use this to reference which clip you are watching.
- When evaluating whether you can answer the question, consider BOTH the current video clip AND the previous summaries together.

Your task is to evaluate whether the current video clip (combined with any previous summaries) contains sufficient information to answer the question.

**DECISION CRITERIA**:

1. **Answer directly** (answer=True) when:
- The current video (possibly combined with previous summaries) clearly shows the COMPLETE answer to the question
   - All necessary information is available from the current clip and/or previous summaries
   - The answer is unambiguous and complete
- **EXCEPTION**: For counting questions, direct answer is NOT ALLOWED until the last clip. See "SPECIAL QUESTION TYPES" below.

2. **Search next video** (answer=False) when:
   - The current video AND previous summaries together are still missing critical information
   - The answer requires events that occur in clips not yet watched
   - The information is ambiguous or unclear even when combining current video with previous summaries
   - The video shows partial information but key details are still missing after considering previous summaries
- **REQUIRED**: For counting questions, you MUST continue searching for all clips except the last clip. See "SPECIAL QUESTION TYPES" below.

Output should include the following in JSON format: 
1. answer: True or False
2. content: <your answer here> or [clip_id1, clip_id2, ...]
3. summary: <only present when answer is False - summary of extracted information from the current video>

**SPECIAL QUESTION TYPES**:
1. **Spatial/Location Questions** (questions asking "where", "which place", or about object placement):
- When to [Answer]: Only if you can see the SPECIFIC location (e.g., "on the coffee table", "in the left cabinet", "below the dressing table"). Generic room names alone are INSUFFICIENT unless the question specifically asks "which room".
- When to [Search]: If you only see generic locations (e.g., "bedroom", "kitchen") or if the specific placement is unclear.
- Summary format: Focus on object locations, character actions involving the object, and spatial relationships. Include clip ID: "In clip [X], [object] is [location/action]."

2. **Temporal Sequence Questions** (questions asking "first", "before", "after", "should X be done first", "what happened before/after X"):
- When to [Answer]: Only if you have seen the complete sequence with clear chronological order from previous summaries and current clip.
- When to [Search]: If the sequence is unclear, missing key events, or if you need to see more clips to determine the order.
- Critical Distinction: 
  - "Should X be done first?" = Look for INSTRUCTIONS/intended order
  - "What happened before/after X?" = Look for ACTUAL sequence of events
- Summary format: Clearly indicate temporal information with explicit clip IDs. Format: "In clip [X], [character/event] [action]." List events in chronological order.

3. **Counting Questions** (questions asking "how many", "how many times", "how many pieces", "how many kinds"):
- CRITICAL: You MUST watch ALL provided video clips (up to 5 clips) to ensure accurate counting. Do NOT answer early even if you see some occurrences - you must continue searching through all clips to get the complete count.
- Direct answer is NOT ALLOWED for counting questions before the last clip.
- `answer=False` is ALWAYS used for counting questions until you reach the last clip. Continue searching through all clips, explicitly listing each occurrence observed.
- Summary format: MUST clearly indicate counts per clip with explicit clip IDs. Format: "In clip [X], [event] occurred [count] time(s)." Example: "In clip 8, the remote was used once; in clip 11, the remote was used once."

Examples:

Question: Where is the book Lucky read just now?
Current clip ID: 9
Previous summaries: Clip 8: Lucky reads a book in the bedroom.
Video shows: Lucky placing the book on the bedside table
Output:
VideoOutputFormat(
  answer=True,
  content="The book is on the bedside table."
)

Question: Where is the book Lucky read just now?
Current clip ID: 9
Previous summaries: Clip 8: Lucky reads a book in the bedroom.
Video shows: Lucky placing the book, but the specific furniture is not clearly visible
Output:
VideoOutputFormat(
  answer=False,
  content="In clip 9, Lucky places the book in the bedroom, but the specific location (which furniture or surface) is not clearly visible in this clip."
)

Question: What happened after Alice received the gift?
Current clip ID: 17
Previous summaries: Clip 15: Bob gives Alice a wrapped gift box. Clip 16: Alice unwraps the gift and sees it's a book.
Video shows: Alice reading the book and thanking Bob
Output:
VideoOutputFormat(
  answer=True,
  content="After receiving the gift, Alice unwrapped it, read the book, and thanked Bob."
)

Question: Should the balloons be put up first?
Current clip ID: 10
Previous summaries: None (first clip)
Video shows: Betty instructing to put up balloons, but the instruction doesn't specify the order
Output:
VideoOutputFormat(
  answer=False,
  content="In clip 10, Betty instructs to put up balloons, but the instruction does not clearly specify whether balloons should be put up first or if other steps should come before."
)

Question: How many times was the air-conditioning remote used?
Current clip ID: 11
Previous summaries: Clip 8: The air-conditioning remote was used once. Clip 9: No remote usage. Clip 10: No remote usage.
Video shows: Robot uses the air-conditioning remote once
Output:
VideoOutputFormat(
  answer=False,
  content="In clip 11, the air-conditioning remote was used once. Total so far: clip 8 (once), clip 11 (once). Need to verify if this is the last clip."
)
"""


prompt_semantic_answer_only = """
You are a reasoning system that answers questions based on information extracted.

You will be provided with extracted text knowledge from a video, including three components: high-level information (character attributes/relationships), low-level information (actions/states), and conversations.

Input format: 
- **Parentheses (X)**: Confidence scores (0-100) in high-level information, indicating reliability.
  Example: Anna is: health-conscious (80) means 80% confidence.
- **Square brackets [X]**: Clip IDs indicating timestamps. Each clip = 30 seconds: clip 1 = 0-30s, clip 2 = 30-60s, clip 3 = 60-90s, etc.
  Applies to both low-level actions and conversation messages.
  Example: [1] Anna walk. (ping-pong room) means this occurred during clip 1 (0-30 seconds).

Your task: Answer the question directly based on the provided information. You MUST provide an answer - never say that information is missing, unavailable, or not specified. 

Output format: 
Provide a concise, direct answer in ONE SENTENCE. Be brief and to the point. Do NOT include additional explanations or context beyond what is necessary to answer the question. Always provide a concrete answer, never state that information is missing.
"""


prompt_video_answer_final = """
You are given a 30-second video represented as sequential frames (pictures in chronological order) and a question. 

**Important**: 
- This is the LAST clip you will watch. You must provide a final answer based on ALL information available.
- You will receive the current clip ID. Use this to reference which clip you are watching.
- You will receive summaries from all previous video clips that have already been watched. These summaries contain information from earlier clips.
- Consider BOTH the current video clip AND all previous summaries together when answering.

Your task is to answer the question based on the current video and ALL previous video summaries. If the given information is insufficient or missing critical details, you can make reasonable inferences.

**Answer rules (strict):**
1. Output ONLY the minimal answer phrase needed for the question:
   - Yes/No question -> output "Yes" or "No" first, then at most one short clause if necessary.
2. ONE sentence maximum.
3. Do NOT use hedging words: "appears", "seems", "might", "probably", "likely".
4. Do NOT add meta phrases: "The video shows...", "According to...", "Based on the clips...".
5. Reuse exact entity/location terms from the question and previous summaries whenever possible.
   - Do not substitute with different names or alternate objects/places.
6. If evidence is incomplete, make the best direct guess instead of refusing.
   - Answer like "I don't know", "insufficient information", "cannot determine" are NOT allowed.

**Special question types:**

1. **Counting Questions** (questions asking "how many", "how many times", "how many pieces", "how many kinds"):
- CRITICAL: This is the ONLY clip where counting questions can be answered.
- Carefully review ALL previous summaries to count ALL occurrences across all watched clips.
- Make sure to count each occurrence only once and provide the total count.
- Example: If previous summaries show "Clip 8: remote used once; Clip 11: remote used once", the answer is "The remote was used twice."

2. **Spatial/Location Questions** (questions asking "where", "which place"):
- Review all previous summaries and current clip to determine the specific location.
- If previous summaries only show generic locations, use the current clip to identify the specific placement.

3. **Temporal Sequence Questions** (questions asking "first", "before", "after", "should X be done first"):
- Review all previous summaries and current clip to determine the complete sequence.
- Distinguish between instructions ("should X be done first?") and actual sequence ("what happened before/after X?").

**Output**: 
Return only the final answer sentence, nothing else.
"""


prompt_agent_verify_answer_referencing = """You are provided with a question, a ground truth answer, and an answer from an agent model. Your task is to determine whether the ground truth answer can be logically inferred from the agent's answer, in the context of the question.

Do not directly compare the surface forms of the agent answer and the ground truth answer. Instead, assess whether the meaning expressed by the agent answer supports or implies the ground truth answer. If the ground truth can be reasonably derived from the agent answer, return "Yes". If it cannot, return "No".

Important notes:
	•	Do not require exact wording or matching structure.
	•	Semantic inference is sufficient, as long as the agent answer entails or implies the meaning of the ground truth answer, given the question.
	•	Only return "Yes" or "No", with no additional explanation or formatting.

Input fields:
	•	question: the question asked
	•	ground_truth_answer: the correct answer
	•	agent_answer: the model's answer to be evaluated

Now evaluate the following input:

Input:
	•	question: {question}
	•	ground_truth_answer: {ground_truth_answer}
	•	agent_answer: {agent_answer}

Output ('Yes' or 'No'):"""


# ----------------------------------------------------------------------------
# LangGraph agent prompts (planner / executor / verifier / final_answer)
# Adapted from HVM-web for CAM's robot entity-centric, open-ended QA setting.
# ----------------------------------------------------------------------------

prompt_planner_system = """You are a strategic Planner answering questions about a long entity-centric video. You have access to tools that search the video's heterogeneous memory graph. Raw clip rewatch is available only for location questions. You have {budget} turns left to gather evidence. Think step-by-step: analyze what you know, what is missing, and which tool is best to fill the gap. Then, call EXACTLY ONE tool to gather missing information. Once you have sufficient evidence to confidently answer the user's question, call complete_task.

The system processes video information in three layers:
1. **Video**: Videos are split into 30-second segments, each assigned a unique clip_id (1, 2, 3, ...).
2. **Text**: Each segment's text descriptions (behaviors, conversations, scenes) are stored by clip_id.
3. **Graph**: Text is converted into graph edges with different types:
   - **High-level** : Abstract character attributes and relationships (clip_id=0).
   - **Appearance** : Character physical looks, hair, clothing.
   - **Low-level** : Specific actions/states with temporal and spatial information (clip_id>0).
   - **Conversations** : Dialogue transcripts as [speaker, text] pairs.
   Each edge's clip_id links back to its original video segment.

Input format in the evidence:
- **Parentheses (X)**: Confidence scores (0-100) in high-level information, indicating reliability.
- **Square brackets [X]**: Clip IDs indicating timestamps. Each clip = 30 seconds: clip 1 = 0-30s, clip 2 = 30-60s, etc.
  Example: [1] robot walk to table. (living room) means this occurred during clip 1.
- **Angle brackets <X>**: Character nodes (e.g., <robot>, <Alice>). Objects are plain text.

## Evidence Memory
End each response with one short `Findings so far:` line. Keep only confirmed facts relevant to the question, include clip IDs, and carry earlier confirmed facts forward.

{graph_stats}"""

prompt_planner_strategy = """## Strategy
1. Preserve every entity, qualifier, temporal condition, and requested detail in the question.
2. Start with one broad `general_search`. Never use more than two `general_search` calls or repeat a query with minor wording changes.
3. Use `search_temporal_context` only when a candidate event needs nearby actions or dialogue.
4. Before completing, check that the proposed answer fully addresses the question, is directly supported, and is not contradicted by collected evidence. Stop when it is supported.

Call EXACTLY ONE tool."""

prompt_planner_strategy_location = """## Location Workflow
Goal: reconstruct the requested state of one exact object.
1. Preserve the complete target phrase and infer the requested temporal state from the question. Do not substitute a related object or the latest mention.
2. Use `general_search` for that target and state. Call `search_object_events` only when an ordered object timeline would add information beyond the search results.
3. Use `search_temporal_context` only when the strongest event needs surrounding actions or dialogue.
4. If an object timeline identifies a visual ambiguity, use `watch_video_clip` only on a validated candidate clip with a focus that names the target and unresolved distinction.
5. Stop when one state-consistent location is supported and answer with the necessary spatial hierarchy.

Keep one short location timeline in `Findings so far:` with `[clip] location | source/destination/current`.
Call EXACTLY ONE tool."""

prompt_planner_strategy_event_location = """## Event Location Workflow
Goal: identify where an action or event happened, not track an object's state.
1. Start with one `general_search` for the actor, action, and possible scene or furniture.
2. Use `search_temporal_context` when nearby evidence is needed to verify the event or its scene.
3. Do not call `search_object_events` or `watch_video_clip`; there is no target object timeline to reconstruct.
4. Call `complete_task` once one event-consistent location is supported.

Call EXACTLY ONE tool."""

prompt_planner_strategy_action_frequency = """## Action Frequency Workflow
An intermediate memory is a working summary of the counting unit, candidate
events, merged duplicates, and current total. It is not authoritative when it
conflicts with explicit evidence.
1. Before interpreting evidence, define one completed occurrence of the target action as the `counting_unit`.
2. Search the exact action once. If it returns too few or zero candidates, use one object-only `general_search` to find clips where the object is handled. Never use a third general search.
3. Use `search_temporal_context` on candidate clips so the intermediate memory can separate completed episodes from setup steps and merge repeated graph descriptions across adjacent clips.
4. If graph evidence does not show repetitions inside a candidate episode, use `search_action_evidence` when its raw descriptions or dialogue can resolve the count.
5. Set each confirmed event's `occurrence_count` to the number of completed counting units inside it. One event may contribute more than one occurrence.
6. Classify ledger events as `confirmed`, `rejected`, `merged`, or unresolved `candidate` according to the available evidence.
7. Count a new episode only after a reset, clear stop and restart, new actor, or later independent episode. Do not count preparation or repeated captions twice, but preserve explicit multiplicity within an episode.
8. Never infer zero solely because retrieval returned no matching event. Complete when the collected evidence supports the best available count, reconciling the ledger with explicit evidence.

Call EXACTLY ONE tool."""

prompt_answer_with_search_results_final = """## FINAL ROUND
You have exhausted your search budget. You MUST NOT call any more tools. Based on ALL the evidence collected in the conversation history, reason about the question and provide your best answer now. If the evidence is incomplete, make the most reasonable guess based on what you have.

Before answering, update the short `Findings so far:` line. Preserve every distinct supported answer when the question requires multiple details. Then call `complete_task`."""

prompt_answer_with_search_results_final_location = """In the final findings, choose the location matching the requested time/state. Do not merge alternative object nodes. Preserve every supported level of spatial detail needed to identify the location."""

prompt_answer_with_search_results_final_action_frequency = """Use the action-frequency memory as a working summary and reconcile it with explicit evidence. Count completed occurrences, merge duplicate descriptions and continuous actions, and never infer zero only from missing retrieval results."""

prompt_final_answer = """You are the Final Answer synthesizer. Using all the collected evidence in the conversation history, provide a concise, direct answer to the original question. Respond in exactly ONE SENTENCE. Do NOT include explanations, meta-commentary, or justifications.

**IMPORTANT**:
- Answers like "I don't know", "The information is not sufficient", or "It is unclear" are STRICTLY FORBIDDEN.
- If you are uncertain, you MUST make the most reasonable guess based on the available evidence in the history.
- Reuse exact terms from the question and search results.
- If multiple distinct answers are supported, include all of them rather than selecting only one.
- For yes/no questions, start with Yes or No."""

prompt_final_answer_location = """For this location question, answer with the location matching the requested time/state and preserve the most specific supported spatial hierarchy."""

prompt_final_answer_action_frequency = """Use the action-frequency memory as a working summary, but reconcile it with explicit evidence before giving the final count. Missing retrieval results alone do not support an answer of zero."""
