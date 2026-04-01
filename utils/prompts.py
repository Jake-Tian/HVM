
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
You are given a list of **action sentences** describing character behavior.  
Convert each sentence into **triples** of the form:

[source, content, target]

Return **ONLY** a valid JSON array (list of lists).  
No explanation. No markdown. No extra text.

## OUTPUT FORMAT
- Strict JSON only
- Use double quotes
- No trailing commas
- Each triple must be:
  [source, content, target]
- Preserve the **original sentence order**
- Preserve the **original action order** within each sentence

## DEFINITIONS
- **Source**: the entity performing the action or whose state is described
- **Content**: the action, relation, or state (verb-centered)
- **Target**: the entity the action is applied to or related to  
  Use `null` if none exists

## EXTRACTION PRIORITY (FOLLOW IN ORDER)

1. Identify actors (sources)
2. Identify actions / relations (content)
3. Identify affected entities (targets)
4. Resolve pronouns and possessives
5. Split compound structures
6. Normalize verbs
7. Add state relations
8. Deduplicate implied redundancy

## RULES: 

1. SOURCE & TARGET (ENTITIES)
- May be:
  - Characters (use verbatim names with angle brackets)
  - Objects (nouns, physical or abstract)
- **Character/Object format rule (STRICT)**:
  - Characters: must keep angle brackets, e.g. `<Alice>`, `<robot>`, `<character_1>`.
  - Objects: must **NOT** use angle brackets.
  - If an object appears with angle brackets in input, remove them in output.
- Copy entity names **verbatim** (except removing angle brackets from objects)
- Use `null` if no target exists
- Do **not** invent entities

2. CONTENT (VERBS / RELATIONS)
- Use **simple present tense** only  
  Examples: walks, puts, looks at
- Avoid progressive or continuous forms  
  is walking → walks
- Include relevant prepositions or direction
  - turns left
  - looks at
  - moves forward
- Include adverbs when present
  - runs quickly
  - smiles happily

3. BODY PART MERGING
- Merge body parts into the verb
- Do NOT create body-part objects
Examples:
- "<Alice> hits <Bob>'s head" → ["<Alice>", "hits head", "<Bob>"]
- "<Emma> touches <David>'s shoulder" → ["<Emma>", "touches shoulder", "<David>"]

4. COMMUNICATION ACTIONS
- Encode communication directly
- Do NOT create abstract objects (e.g., "question", "message")
Examples:
- "<Tom> asks <Mary>" → ["<Tom>", "asks", "<Mary>"]
- "<Lisa> greets <John>" → ["<Lisa>", "greets", "<John>"]

5. OBJECT HANDLING
- Objects are nouns
- NEVER wrap object nodes with angle brackets (`< >`)
- Singularize plurals  
  books → book
- Keep adjectives attached to the object  
  eg. "red cup"
- Keep named objects verbatim  
  eg. "bottle of Nescafe"
- Split compound objects into separate triples
  eg. "<Alice> picks up the book and the pen" → ["<Alice>", "picks up", "book"], ["<Alice>", "picks up", "pen"]

6. PRONOUN & POSSESSIVE RESOLUTION
- NEVER use pronouns (his, her, their)
- Replace possessives with explicit ownership:
  - his wallet → <John>'s wallet
- Default ownership to the **nearest subject** if ambiguous

7. MULTIPLE RELATIONS
- Multiple subjects: Each subject gets its own triple
  eg. "<Alice> and <Bob> exit" → ["<Alice>", "exit", null], ["<Bob>", "exit", null]
- Multiple verbs: Each verb becomes a separate triple
  eg. "<Lisa> dances and sings" → ["<Lisa>", "dances", null], ["<Lisa>", "sings", null]
- Multiple objects: Each object becomes a separate triple

8. STATE REPRESENTATION
- If an action implies a **resulting state**, add a state triple.
- For location/spatial relations, put location information in the **content** field and keep
  source/target as clean entities (character or noun). Do NOT put long location phrases in target.
Examples:
- "<robot> puts coffee on table" → ["<robot>", "puts", "coffee"], ["coffee", "is on", "table"]
- "<Alice> takes towel from Susan's bag" → ["<Alice>", "takes", "towel"], ["towel", "is in", "Susan's bag"]
- "<Betty> sits on the right side of the sofa" → ["<Betty>", "sits on the right side of", "sofa"]

9. INFORMATION CONSISTENCY
- Ensure there is no information loss when converting the sentence into triples. Do NOT remove any adjectives, adverbs, or other modifiers.

10. DEDUPLICATION
- Keep only **distinct, meaningful** actions
- Do NOT duplicate states already implied by a stronger action
- Redistribution of information across triples is allowed

11. FALLBACK RULE
If unsure, output a **minimal transformation**:
[source, verb, target]

## EXAMPLE: 
Input:
[
  "<Michael> pats <Susan>'s shoulder and smiles.",
  "<robot> places the red cup on the counter.",
  "<Lisa> dances and sings happily.",
  "<John> takes his wallet and keys from the drawer.", 
  "<Betty> carries a red plastic bag in her left hand and a white plastic bag in her right hand."
]

Output:
[
  ["<Michael>", "pats shoulder", "<Susan>"],
  ["<Michael>", "smiles", null],
  ["<robot>", "places", "red cup"],
  ["red cup", "is on", "counter"],
  ["<Lisa>", "dances happily", null],
  ["<Lisa>", "sings happily", null],
  ["<John>", "takes", "John's wallet"],
  ["<John>", "takes", "John's key"], 
  ["<Betty>", "carries in left hand", "red plastic bag"],
  ["<Betty>", "carries in right hand", "white plastic bag"]
]

Now convert the following list of action sentences into triples:
"""


prompt_extract_timetriples = """
You are given translated dense-caption text segments from an egocentric video.
The input is a JSON array of segments. Each segment is exactly a two-element list:
  `[start_timestamp, sentence]`
- `start_timestamp`: 6-digit string `hhmmss` (same values you must use in output `time` fields).
- `sentence`: the short caption text for that segment.

Your task is to convert this text into a list of TimeTriple items.
Characters in this video are only: <I>, <Alice>, <Tasha>, <Lucia>, <Katrina>, and <Shure>.

## Target schema
- Output must match:
  TimeTripleList(
    triples=[
      TimeTriple(time="hhmmss", triple=[source, content, target]),
      ...
    ]
  )
- `time` must use one of the provided segment start timestamps exactly.
- Timestamp format is strictly 6 digits: `hhmmss` (example: `110943`).
- `triple` must always contain exactly 3 strings: [source, content, target]
- If target is missing, use the literal string "null" (not Python null).

## RULES: 

1. SOURCE & TARGET (ENTITIES)
- There are two types of entities:
  - Characters (use verbatim names with angle brackets)
  - Objects (nouns, physical or abstract)
- **Character/Object format rule (STRICT)**:
  - Characters: must keep angle brackets, e.g. `<I>`, `<Alice>`, `<Tasha>`, `<Lucia>`, `<Katrina>`, `<Shure>`.
  - Objects: must **NOT** use angle brackets.
- Copy entity names **verbatim** (except removing angle brackets from objects)
- Use `null` if no target exists
- Do **not** invent entities

2. CONTENT (VERBS / RELATIONS)
- Use **simple present tense** only  
  Examples: walks, puts, looks at
- Avoid progressive or continuous forms  
  is walking → walks
- Include relevant prepositions or direction
  - turns left
  - looks at
  - moves forward
- Include adverbs when present
  - runs quickly
  - smiles happily

3. BODY PART MERGING
- Merge body parts into the verb
- Do NOT create body-part objects
Examples:
- "<Alice> hits <Tasha>'s head" → ["<Alice>", "hits head", "<Tasha>"]
- "<Lucia> touches <Shure>'s shoulder" → ["<Lucia>", "touches shoulder", "<Shure>"]

4. OBJECT HANDLING
- Objects are nouns
- NEVER wrap object nodes with angle brackets (`< >`)
- Keep adjectives attached to the object  
  eg. "red cup"
- Keep named objects verbatim  
  eg. "bottle of Nescafe"
- Split compound objects into separate triples
  eg. "<Alice> picks up the book and the pen" → ["<Alice>", "picks up", "book"], ["<Alice>", "picks up", "pen"]

5. PRONOUN & POSSESSIVE RESOLUTION
- NEVER use pronouns (his, her, their)
- Replace possessives with explicit ownership:
  - his wallet → <Shure>'s wallet
- Default ownership to the **nearest subject** if ambiguous

6. MULTIPLE RELATIONS
- Multiple subjects: Each subject gets its own triple
  eg. "<Alice> and <Bob> exit" → ["<Alice>", "exit", null], ["<Bob>", "exit", null]
- Multiple verbs: Each verb becomes a separate triple
  eg. "<Lisa> dances and sings" → ["<Lisa>", "dances", null], ["<Lisa>", "sings", null]
- Multiple objects: Each object becomes a separate triple

7. INFORMATION CONSISTENCY
- Ensure there is no information loss when converting the sentence into triples. Do NOT remove any adjectives, adverbs, or other modifiers.

## Good examples
Input segment:
["110959", "I put my phone in the middle of the dining table."]
Output triples:
 - TimeTriple(time="110959", triple=["<I>", "puts", "phone"])
 - TimeTriple(time="110959", triple=["phone", "is in the middle of", "dining table"])

Input segment:
["111005", "Katrina asked me a question."]
Output triples:
 - TimeTriple(time="111005", triple=["<Katrina>", "asks", "<I>"])

Input segment:
["111012", "Tasha waves at Lucia."]
Output triples:
 - TimeTriple(time="111012", triple=["<Tasha>", "waves at", "<Lucia>"])

Now convert the following input segments:
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
Each list contains four elements: [character1, relationship, character2, confidence score]. 
Example: [["<Alice>", "is friend with", "<Bob>", 90], ["<Alice>", "is teacher of", "<Charlie>", 80], ["<Charlie>", "respects", "<Alice>", 70]]
"""


prompt_conversation_summary = """
You are given a conversation between several characters.

Your tasks: 

1. **Summary**
- Summarize the key topics, decisions, or outcomes discussed in the conversation.
- Write 2-4 concise sentences covering the main themes and important points.
- Focus on what was discussed and decided, not on individual statements.
- Output format: string
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

Now summarize the following conversation:
"""


prompt_summarize_conversation = """
You are given a conversation between several characters.

Your task is to summarize the conversation into a concise, narrative paragraph.

Rules:
- Write a single, coherent paragraph (3-5 sentences)
- Use natural, flowing language (not a bulleted list)
- Keep character names as provided (e.g., <Shure>, <Tasha>, <Katrina>)

Now summarize the following conversation:
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

Given a query and budget `k=50`, output:

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
  query_triples=[["tape", "is at", "?", 0.8, 0.5, 0.15]],
  spatial_constraint=None,
  speaker_strict=None,
  allocation=ParseQueryAllocation(
    k_high_level=2, k_low_level=42, k_conversations=6, k_appearance=0, total_k=50,
    reasoning="Temporal-spatial query - 'now' means most recent location. Prioritize low-level edges with highest clip_id to find current state"
  )
)

Now parse the following query and allocate k=50:
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

Given a query and budget `k=30`, output:

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

Now parse the following query and allocate k=30:
"""


prompt_parse_query_no_allocation = """
You are a query parser for a knowledge graph system that stores video information in a hierarchical structure.

## YOUR TASK

Given a query, output:

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

Now parse the following query:
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

Output should include: 
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

Output should include: 
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

Output should include: 
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


#--------------------------------
# Prompts for EgoLife
#--------------------------------

prompt_allocate_search = """
You are a search-budget allocator for an external database.

The database has two sources:
1) **behaviors**: action/state/location events from visual observations
2) **conversations**: spoken dialogue content

Given a question, allocate a fixed budget **k=50** between the two sources.

## Output format
1. k_behavior: int
2. k_conversation: int
3. speaker_strict: the list of speakers
4. reasoning: one short sentence

Constraints:
- `k_behavior + k_conversation == 50`
- `k_behavior >= 0`, `k_conversation >= 0`
- `speaker_strict`:
  - use `["Jake", "Shure"]` style when question explicitly asks about specific speakers' dialogue
  - if the question uses first-person references ("I", "me", "my"), map that speaker to `"Jake"` when speaker filtering is necessary for dialogue retrieval
  - use `null` otherwise
- Prefer integers (no decimals)
- No markdown, no extra text

## Allocation principles
1. **Behavior-priority questions** (allocate more to behaviors: usually 32-45)
   - action/object manipulation: "what did X do", "who used/picked/moved..."
   - spatial/location: "where is/was...", placement, source location
   - temporal sequence from events: "before/after/first/last"
   - counting event occurrences: "how many times..."

2. **Conversation-priority questions** (allocate more to conversations: usually 32-45)
   - dialogue content: "what did they discuss/say/ask/answer/mention"
   - intention/plan/preference from speech: "plan to", "decide to", "want to"
   - causal questions likely verbalized: "why..."
   - if specific speakers are named for dialogue, set `speaker_strict` to those names
   - if first-person ("I"/"me"/"my") appears and speaker filtering helps, include `"Jake"` in `speaker_strict`

3. **Balanced questions** (allocate near-even: 20-30 each)
   - require both action evidence and dialogue evidence
   - speaker + behavior mixed constraints

## Examples
Question: "Where was the black marker before?"
Output:
{"k_behavior": 42, "k_conversation": 8, "speaker_strict": null, "reasoning": "Location query depends mainly on observed object placement events."}

Question: "What did Jake and Shure discuss this morning?"
Output:
{"k_behavior": 12, "k_conversation": 38, "speaker_strict": ["Jake", "Shure"], "reasoning": "Discussion content is primarily in dialogue between named speakers."}

Question: "Who helped me while I was cleaning in the kitchen?"
Output:
{"k_behavior": 36, "k_conversation": 14, "speaker_strict": null, "reasoning": "Main evidence is co-occurring actions, with possible supporting dialogue."}

Now allocate k=50 for the following question:
"""


prompt_answer_with_search_results = """
You are a reasoning module for iterative retrieval.

You will receive:
1) a user question
2) multiple-choice options (A/B/C/D)
2) retrieved search results from behavior and/or conversation memory

Your task is to decide whether the retrieved results are sufficient to answer the question.

## Decision
1. If sufficient:
   - return the selected option as the final answer.
2. If insufficient:
   - return:
     - an updated question for the next retrieval round (more specific and focused on missing information)
     - a concise summary of what is already known from current results
     - one search tool call for the next round (choose exactly one from the provided search methods)

Output format:
1. Answer: True or False,
2. Content: the selected option (e.g., "A") if Answer=True, otherwise updated question for next round,
3. Summary: None if Answer=True, otherwise concise summary of current search results
4. tool_name: str or None
5. target: str or None
6. total_search_k: int or None
7. k_behavior: int or None
8. k_conversation: int or None
9. speaker_strict: list[str] or None

Constraints:
- If `answer=true`:
  - `content` must be exactly one option label: `"A"`, `"B"`, `"C"`, or `"D"`.
  - `summary` must be None.
  - `tool_name`, `target`, `total_search_k`, `k_behavior`, `k_conversation`, `speaker_strict` must be null.
- If `answer=false`:
  - `content` must be a better retrieval question, not an answer.
  - `content` should target missing evidence that can distinguish among options.
  - `summary` must capture only key facts relevant to the given question (2-4 sentences).
  - Do NOT include irrelevant events, entities, or background details.
  - LLM should decide next-round total search budget:
    - `1 <= total_search_k <= 50`
    - Use smaller budgets (e.g., 15-40) when the missing evidence is narrow/specific to reduce distraction.
  - `k_behavior + k_conversation == total_search_k`
  - choose allocation based on missing evidence:
    - prefer behavior when missing actions/locations/order/counting
    - prefer conversation when missing discussion/intent/why
  - `speaker_strict`:
    - use explicit names when dialogue between specific speakers is needed
    - map first-person references ("I", "me", "my") to "Jake" only if speaker filtering is helpful
    - otherwise use null
  - choose exactly ONE tool from:
    ["general_search", "evidence_linker", "search_before", "search_after", "search_first", "search_last"]
  - tool usage guidance:
    - `general_search`: default semantic retrieval. Use when you just need the most relevant evidence without extra temporal constraints.
    - `evidence_linker`: use when clues are scattered and require multi-hop deduction across behavior + conversation.
    - `search_before`: temporal-backward retrieval. Use when the question asks what happened before a known event/line (`target`).
    - `search_after`: temporal-forward retrieval. Use when the question asks what happened after a known event/line (`target`).
    - `search_first`: earliest-occurrence retrieval. Use when the question asks who/what happened first or earliest.
    - `search_last`: latest-occurrence retrieval. Use when the question asks who/what happened last or most recently.
  - tool-selection policy (important):
    - avoid repeating the same tool with nearly the same query if the previous round did not add decisive evidence.
    - if two consecutive rounds are still "insufficient" with similar summaries, you MUST switch tool (or switch to a clearly different retrieval strategy).
    - for location-before questions ("where ... before"), prefer `search_before` with a concrete `target` line when possible.
    - when evidence is split across behavior and conversation and requires deduction, prefer `evidence_linker`.
    - use `general_search` mainly for broad first-pass recall, not for repeated fallback loops.
    - during rounds 2-4, try at least one non-`general_search` tool unless current evidence is already sufficient to answer.
  - lexical bridging:
    - when an object may have aliases (e.g., marker/pen/pencil/chalk), include those aliases in `content` to improve retrieval coverage.
    - preserve key entities and temporal cues from current evidence in the rewritten `content`.
  - set `tool_name` to that method name.
  - `target` is REQUIRED when `tool_name` is `search_before` or `search_after`.
  - `target` is OPTIONAL when `tool_name` is `evidence_linker` (set when an anchor line is helpful).
  - for `general_search`, `search_first`, `search_last`, `target` must be null.
  - Use `content` as `search_content` for the selected tool call.
  - The selected tool should use (`total_search_k`, `k_behavior`, `k_conversation`) as allocation input.
  - (`k_behavior` + `k_conversation`) must equal `total_search_k`.
- Do NOT output markdown or extra text.
- Reuse concrete entities/timestamps from retrieved results when helpful.

Decision guidance:
- Set `answer=true` when current evidence can reasonably eliminate other options and support one best option.
- Continue searching (`answer=false`) when key evidence is truly missing or multiple options remain similarly plausible.
- If one option is clearly more supported than others, answer now.
- Do not keep issuing near-identical `general_search` requests across rounds without changing strategy.

Now process the following input:
"""


prompt_answer_with_search_results_final = """
This is the final round of the QA task.

You will receive:
1) the question
2) options A/B/C/D
3) accumulated retrieved evidence from all previous search rounds

You must choose one option based on the accumulated retrieved evidence.
If you are not sure, choose the option that is most supported by the evidence. The answer like "I don't know", "insufficient information", "cannot determine" are NOT allowed.
The output must be exactly one letter.
"""


prompt_agent = """
You will receive:
1) a user question
2) multiple-choice options (A/B/C/D)
3) The searched behavior and conversation results from previous rounds

Your task is to decide whether the retrieved results are sufficient to answer the question.

## Decision: 
1. If sufficient:
   - return a single letter (A/B/C/D) that indicates the best option.
2. If insufficient:
   - choose the most suitable search tool for the next round.
   - explicitly identify the missing knowledge and update the next query triple to target that gap.

Tool usage guidance: 
- `general_search`: default semantic retrieval. Use when you just need the most relevant evidence without extra temporal constraints.
- `search_before`: temporal-backward retrieval. Use when the question asks what happened before a known event/line.
- `search_after`: temporal-forward retrieval. Use when the question asks what happened after a known event/line.
- `search_first`: earliest-occurrence retrieval. Use when the question asks who/what happened first or earliest.
- `search_last`: latest-occurrence retrieval. Use when the question asks who/what happened last or most recently.
- `search_object`: object-centric retrieval. Use when the question asks about a specific object. It could be used in early rounds for a broad search. Do not use it in the final round. 

Round constraints:
- **Round 1 is mandatory**: you must choose `general_search`.
- In Round 1, k_behavior + k_conversation must be 50. 
- From Round 2 onward, if previous results are insufficient, you must update the query triple(s) based on the missing knowledge.
- Do not repeat the exact same triple(s) when previous results were insufficient unless you also change strategy/tool and clearly justify it by a different missing gap.
- `search_object` can be used as intermediate search. If the object in the question does not exist in the graph, you need to use synonyms that exist in the graph for the follow-up search.
- If the question asks what happened before/after a known event, you first need to find the event in the graph, then use `search_before` or `search_after` using the timestamp of the event.

The search methods take following input: 
1. **Query triple**: 
  - Triple format: [source, content, target, source_weight, content_weight, target_weight]
  - Use "?" for missing or unknown components, normalize to graph format. 
  - Use angle brackets for characters (eg. <Tasha>, <Lucia>, <I>)
  - Weight assigning rules: 
     - **High (0.7-1.0)**: Specific character/object names (e.g., "<Alice>", "coffee", "the red cup") - use 0.9-1.0 for critical entities
     - **Medium (0.4-0.7)**: General objects/locations (e.g., "cup", "room") - use 0.5-0.7 for context
     - **Low (0.1-0.4)**: What we're searching for - question marks ("?"), unknown actions, vague terms
  - If searching based on the triple from the question is not helpful, you can also generate triples based on choices.
  - When insufficient, revise at least one of (source/content/target) to directly probe the missing fact (entity, relation, time anchor, or location cue).

2. **Search budget**:
  - k_behavior and k_conversation are the search budgets for behavior and conversation respectively.
  - You should decide how many bahavior triples and conversation messages to be searched based on the question and the previous results.
  - 1 <= k_behavior + k_conversation <= 50. 
  - You can choose only search for behavior and conversation. In this case, the other one should be 0.
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