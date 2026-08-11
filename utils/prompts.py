
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
     - Complete location descriptions: Always combine furniture names with spatial modifiers to form hierarchical locations (e.g., "cabinet below the dressing table", "second layer of the refrigerator", "cabinet on the left side of the wardrobe"). For retrieval actions, include source locations (e.g., "takes towel from bag", "gets mask from cabinet below dressing table")
   - Each entry must describe exactly one event/detail. Split sentences if needed.
   - Example output:
     ["<male_1> enters the room.", 
      "<male_1> takes cap from the cabinet on the left side of the wardrobe.", 
      "<male_1> sits with <female_1> side by side on the couch.", 
      "<female_1> watches TV.", 
      "<male_2> puts the coffee on the table."]

2. **Conversation**
   - Record the dialogue based on subtitles.
   - If a name is explicitly provided in the subtitle, use it. Otherwise, refer to characters using consistent IDs (e.g., <male_1>, <female_1>, <job_1>).
   - Output format: List of two-element lists [character, content].
   - Example output: [["<male_1>", "Hello, how are you?"], ["<female_1>", "I am fine, thank you."]]

3. **Characters' Appearance**
   - Describe or update each character's appearance: facial features, clothing, body shape, hairstyle, or other distinctive characteristics.
   - Each characteristic should be concise, separated by commas.
   - Existing characters: Update if changes observed (hair, clothing), enhance if new details visible, otherwise keep unchanged. Keep appearance info even if character leaves scene.
   - If a character is not visible in scene, then do not include in the characters' appearance list.
   - Example output:
     [Appearance(name="<male_1>", appearance="male, short hair, wear glasses, blue t-shirt, black pants"),
      Appearance(name="<female_1>", appearance="female, long hair, ponytail, red dress, white sneakers")]

4. **Scene**: Use one word or phrase to describe the scene in the current video (eg. "bedroom", "gym", "office", etc.).

5. **OCR**: Extract text from video frames (signs, labels, remarks). Use JSON format with context and content.
   - Example output:
     [OCR(context="sign in the store", content="Open 24 Hours"),
      OCR(context="remark in the video", content="Special Offer: 50% Off")]

# All the above information should be provided in JSON format.

## Character Naming Rules:
- Use angle brackets to represent characters (eg. <male_1>, <female_1>, <job_1>, <Alice> etc.) in behaviors, conversation, and character appearance.
- Priorities for naming:
  1. Use consistent generic IDs based on gender and number (e.g., <male_1>, <male_2>, <female_1>). This is the PREFERRED method as specific names are often unavailable.
  2. Use job-based IDs if the identity is clear (e.g., <police_1>, <waiter_1>).
  3. Use a name ONLY if it is explicitly and unambiguously stated in subtitles or conversation. 
- Do NOT guess names. Maintain consistency across clips using the same ID for the same person.

## Character Matching Rules:
For characters appearing in the video (MUST follow before creating new characters):
- Match the character in the current clip to the characters already identified in the appearance list. 
- Use facial features, clothing, and other distinctive traits to maintain ID consistency.
- If high similarity is found, refer to this character by its existing ID in the list. 
- If no match is found, create a new ID following the naming rules above.
- Our goal is to minimize the number of unique character IDs. When uncertain, match to existing rather than creating new.

Additional Rules:
- Maintain strict chronological order.
- Avoid repetition in both behavior and conversation.
- If no behavior or conversation is observed, return an empty list for behaviors and conversation.
"""

prompt_extract_triples = """
Convert a JSON array of action sentences into structured triples:
`{"triples": [{"source": ..., "content": ..., "target": ...}]}`.

Graph contract:
- `source` and `target` are stable, reusable entity nodes; `content` is the action or relation between them.
- The affected object must be the target, never hidden inside `content`.
  Correct: `{"source": "<male_1>", "content": "takes", "target": "towel"}`.
  Wrong: `{"source": "<male_1>", "content": "takes towel", "target": null}`.
- Each triple must express exactly one semantic fact. Do not merge an action with a
  separate spatial or state relation.

Extraction rules:
- Preserve sentence order and action order. Split distinct actors, actions, and objects.
- Characters keep their provided angle-bracketed IDs, such as `<male_1>` and
  `<female_1>`; objects never use angle brackets.
- Use simple-present, verb-centered relations. Preserve meaningful modifiers,
  ownership, direction, and hierarchical location details.
- Keep reusable entity names concise, but do not discard identity-relevant modifiers
  such as `raw`, `red`, `wooden`, `upper`, or `left-side`.
- Resolve pronouns and possessives to explicit entities when the input supports it.
  Do not invent entities or facts. Keep only distinct, supported facts.
- Use a null target only when the action truly has no affected or related entity.

Fact splitting examples:
- `<Chris> holds a smartphone beside the steamer` becomes:
  `{"source": "<Chris>", "content": "holds", "target": "smartphone"}` and
  `{"source": "smartphone", "content": "is beside", "target": "steamer"}`.
- `<Alice> picks up the book and the pen` becomes one `picks up` triple per object.
- `<Alice> and <Bob> exit` becomes one `exit` triple per character.

State and source-location rules:
- For placement, emit both the action and the resulting state:
  `<robot> puts coffee on the table` becomes
  `{"source": "<robot>", "content": "puts", "target": "coffee"}` and
  `{"source": "coffee", "content": "is on", "target": "table"}`.
- For retrieval, preserve the source relation without implying that the object
  remains there: `<Alice> takes a towel from Susan's bag` becomes
  `{"source": "<Alice>", "content": "takes", "target": "towel"}` and
  `{"source": "towel", "content": "is taken from", "target": "Susan's bag"}`.
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
- Keep character names/IDs as provided (e.g., <male_1>, <female_1>)
- Do not include clip numbers or scene labels in the summary
- There might be conflict or misleading information provided, you should be able to handle it and provide a coherent summary.

Now summarize the following clips:
"""


prompt_character_summary = """
You are given a character's name/ID and a list of their behaviors in chronological order.

Your task is to summarize the character's attributes: 
- Personality (eg. confident, nervous)
- Role/profession (eg. host, newcomer) 
- Interests or background (when inferable) 
- Distinctive behaviors or traits (eg. speaks formally, fidgets). 
Avoid restating visual facts—focus on identity construction.

For each attribute, you should also provide a confidence score between 0 and 100. 
If the confidence score is less than 50, you should not include the attribute in the output.

Output a JSON dictionary (key: attribute, value: confidence score). 
Example: {"energetic": 90, "professional": 80, "focused": 70}
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
- Do NOT include any actual actions or summary of actions in the output (eg. <male_1> speaks with <female_1>). 
- Do not generate repetitive or symmetric information. 

For each relationship, you should also provide a confidence score between 0 and 100.
If the confidence score is less than 50, you should not include the relationship in the output.
It is acceptable to only generate a few relationships if you don't have enough information.

Output a JSON array (list of lists). 
**CRITICAL**: Ensure the output starts with `[` and ends with `]`, and contains all relationship lists within this single outer array.
Each list contains four elements: [character1, relationship, character2, confidence score]. 
Example: [["<male_1>", "is friend with", "<female_1>", 90], ["<male_2>", "collaborates with", "<male_1>", 80]]
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
  "summary": "The characters discussed the project timeline. They agreed on the next steps and assigned tasks.",
  "character_attributes": [["<male_1>", "organized", 85], ["<female_1>", "cautious", 70]],
  "characters_relationships": [["<male_1>", "is friend with", "<female_1>", 90]]
}

### DETAILED INSTRUCTIONS

1. **Summary**
- Summarize the key topics, decisions, or outcomes discussed in the conversation.
- Write 2-4 concise sentences covering the main themes and important points.
- Focus on what was discussed and decided, not on individual statements.

2. **Character Attributes**
- Extract each character's attributes revealed through their dialogue and interaction style.
- Focus on: personality traits, role/profession, interests, background information (when mentioned).
- **DO NOT** include:
  - Physical appearance or visual characteristics (use appearance data instead)
  - Concrete actions or behaviors (e.g., "asked a question", "walked away")
  - Temporary emotional states (use persistent personality traits instead)
  - Information not directly supported by the conversation
- Confidence scores range from 0-100. Only include attributes with confidence >= 50.
- Avoid redundant or overly similar attributes.
- Use angle brackets for character IDs (e.g., "<male_1>", "<female_1>").
- Output format: List of three-element lists: [character, attribute, confidence_score].

3. **Character Relationships**
- Extract abstract relationships between characters based on their dialogue interactions.
- Include: roles, attitudes, power dynamics, evidence of cooperation/conflict.
- **DO NOT** include:
  - Specific actions or events (e.g., "<male_1> speaks with <female_1>")
  - Temporary interactions (focus on underlying relationship patterns)
  - Dialogue content or topics discussed (focus on the relationship itself)
- Confidence scores range from 0-100. Only include relationships with confidence >= 50.
- Do not generate symmetric duplicates.
- It is acceptable to generate only a few relationships if there is insufficient information.
- Output format: List of four-element lists: [character1, relationship, character2, confidence_score].

### EDGE CASES
- If conversation has only one character speaking: focus on their attributes, skip relationships.
- If conversation is empty or unclear: return empty arrays for attributes and relationships, provide a brief summary noting the issue.
- If character names are ambiguous: use the names/IDs as provided in the conversation.

Now summarize the following conversation in JSON format:
"""


#--------------------------------
# Reasoning Prompts
#--------------------------------

prompt_parse_query = """
You are a query parser for a knowledge graph system that stores video information in a hierarchical structure.

## GRAPH STRUCTURE

**HIGH-LEVEL EDGES**: Character attributes/relationships
- Format: `["<male_1>", "confident", null]` or `["<male_1>", "is friend with", "<female_1>"]`
- **Limited quantity** (<10 per query) - allocate 5-10 max when needed, fewer otherwise
- Use for: character traits, relationships, "who is" queries

**APPEARANCE EDGES**: Character appearance features
- Format: `["<male_1>", "wears red hoodie", null]`
- Use ONLY when question asks about appearance (looks, clothing, hairstyle, body shape, facial features, accessories).
- Do NOT use for behavior/action/location questions.

**LOW-LEVEL EDGES**: Specific actions/states with scene info
- Format: `["<male_1>", "picks up", "coffee"]` or `["coffee", "is on", "table"]`
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
- **High (0.7-1.0)**: Specific character/object names (e.g., "coffee", "the red cup") - use 0.9-1.0 for critical entities
- **Medium (0.4-0.7)**: General objects/locations (e.g., "cup", "room") - use 0.5-0.7 for context
- **Low (0.1-0.4)**: What we're searching for - question marks ("?"), relationship terms ("relationship", "friendship"), unknown actions - use 0.2-0.4 for search targets, 0.1-0.2 for vague terms

**Special Rules for Location Queries**:
- **Preserve hierarchical locations**: When parsing location queries, keep complete hierarchical location phrases as single entities in target fields (e.g., "cabinet on the left side of the wardrobe", "cabinet below the dressing table"). Do NOT split them into separate components.
- **Temporal-spatial queries**: 
  - "where is X now?" → Use triple `[X, "is at", "?", ...]` with high weight on X. The search should prioritize the most recent state edges (highest clip_id).
  - "last time" / "last place" → Use triple `[X, "is at", "?", ...]` and prioritize edges with highest clip_id values.
  - "where should X be placed?" → Use triple `[X, "should be placed at", "?", ...]` or `[X, "is placed at", "?", ...]` to find placement instructions.
- **Source location queries**: "where did X get Y from?" → Use triple `[X, "gets", "Y", ...]` or `[Y, "is in", "?", ...]` to find source locations. Include a helper triple if needed: `[Y, "is from", "?", ...]`.
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
   - Set to `["<male_1>", "<female_1>"]` when query asks about dialogue between specific speakers
   - Set to `null` otherwise

4. **spatial_constraint**: Location string only for general spaces (e.g., gym, office, kitchen, bedroom, living room, meeting room). Do NOT use objects or furniture (e.g., table, dressing table, sofa) as spatial constraints. Otherwise `null`.

## EXAMPLES

**Example 1**: "What is the relationship between the two main characters?"
ParseQueryOutput(
  query_triples=[["<male_1>", "relationship", "<female_1>", 0.9, 0.2, 0.9]],
  spatial_constraint=None,
  speaker_strict=None,
  allocation=ParseQueryAllocation(
    k_high_level=10, k_low_level=10, k_conversations=30, k_appearance=0, total_k=50,
    reasoning="Relationship query - use high-level for relationships, conversations for evidence"
  )
)

**Example 2**: "What did the man do with the coffee in the kitchen?"
ParseQueryOutput(
  query_triples=[["<male_1>", "?", "coffee", 0.9, 0.15, 0.9]],
  spatial_constraint="kitchen",
  speaker_strict=None,
  allocation=ParseQueryAllocation(
    k_high_level=5, k_low_level=38, k_conversations=7, k_appearance=0, total_k=50,
    reasoning="Action query - prioritize low-level edges"
  )
)

Now parse the following query and allocate k=50 in JSON format:
"""


prompt_parse_query_k30 = """
You are a query parser for a knowledge graph system that stores video information in a hierarchical structure.

## TASK

Given a query and budget `k=30`, output the following in JSON format:

1. **Query triple(s)**: Output **`query_triples`** as a list of 1 to 3 triples.
   - Triple format: `[source, content, target, source_weight, content_weight, target_weight]`
   - Use generic IDs (e.g., <male_1>) instead of names unless name is explicitly provided.
   - Assign weights (0.0-1.0) based on importance.

2. **Allocation** `{k_high_level, k_low_level, k_conversations, k_appearance}`:
   - Total must be <= 30.
   - High-level: 5-10 max.
   - Low-level: 15-24 for action queries.
   - Conversations: 5-24 based on needs.
   - Appearance: 3-10 for physical queries, otherwise 0.

3. **speaker_strict**: Set to specific speaker IDs if query asks about dialogue between them, otherwise `null`.

4. **spatial_constraint**: Location string only for general spaces. Otherwise `null`.

Now parse the following query and allocate k=30 in JSON format:
"""


prompt_parse_query_no_allocation = """
You are a query parser for a knowledge graph system that stores video information in a hierarchical structure.

## YOUR TASK

Given a query, output the following in JSON format:

1. **Query triple(s)**: Output **`query_triples`** as a list of 1 to 3 triples.
   - Triple format: `[source, content, target, source_weight, content_weight, target_weight]`
   - Use `null` for missing components, normalize to generic IDs (e.g., <male_1>) unless name is explicitly provided.
   - **Assign weights** (0.0-1.0).

**Special Rules for Location Queries**:
- **Preserve hierarchical locations**: keep complete hierarchical location phrases (e.g., "cabinet below the dressing table"). Do NOT split them.
- **Temporal-spatial queries**: Use triples to find most recent state or source locations.

2. **speaker_strict**: Set to IDs when query asks about specific dialogue, otherwise `null`.

3. **spatial_constraint**: Location string only for general spaces. Otherwise `null`.

Now parse the following query in JSON format:
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
