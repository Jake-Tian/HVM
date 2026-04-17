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
      "<Bob> watches TV."]

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

4. **Scene**: Use one word or phrase to describe the scene in the current video (eg. "bedroom", "gym", "office", etc.). This field is optional and should only be provided if the scene is clearly identifiable.

5. **OCR Information**
   - Extract text information visible in the video frames, such as:
     - Remarks or notes in the video.
     - Signs in a store, labels on products, or text on posters.
     - Any other visible text that is NOT a subtitle.
   - For each extracted text, provide:
     (a) Context: A simple description of where the information was extracted (e.g., "sign in the store", "remark in the video", "label on the box").
     (b) Content: The actual text content.
   - Do NOT include subtitle information here; subtitles should only be recorded in the **Conversation** section.
   - Example output:
     [OCR(context="sign in the store", content="Open 24 Hours"),
      OCR(context="remark in the video", content="Special Offer: 50% Off")]

## Character Naming Rules:
- Use angle brackets to represent characters (eg. <Alice>, <Bob>, <male_1> etc.) in behaviors, conversation, and character appearance.
- There are three cases to name a character:
  1. The name can be deduced from the conversation or subtitles. Refer to the character by his/her name (e.g., <Alice>, <Bob>).
  2. The name is not provided, but you can deduce the character's job or identity (e.g., police, mailsman). Refer to the character by <job_number> (e.g., <police_1>, <mailsman_1>). Only refer to a character by their job when it is clear. Do not make any unsure guesses on a character's job.
  3. Neither the name nor the job can be inferred. Name the character by gender and a number (e.g., <male_1>, <male_2>, <female_1>).
- For the second and third cases, if the character's name is provided in the following video clips, use the equivalence line to update the character's name.

## Character Matching Rules:
For characters appearing in the video (MUST follow before creating new characters):
- First, check if the subtitle name is provided. If so, use it as the character name (eg. <Anna>). 
  - Also compare the character name with the unknown characters (e.g., <male_1>, <police_1>) in the appearance list. 
    - If high similarity is found, add "Equivalence: <unknown_id>, <Anna>" at **start of behaviors**. This indicates that the character is the same as the previously identified unknown character.
    - If the equivalence is found, refer to this character by its character name instead of the unknown ID in the behaviors, conversation, and character appearance.
- If a character's name is not provided in the subtitle, match this character's appearance to the characters in the appearance list. 
  - If high similarity is found, refer to this character by its name in the appearance list. 
  - If no match is found, create a new character following the naming rules above.
  - Our goal is to minimize the number of unique character IDs. When uncertain, match to existing rather than creating new.

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
  - Characters: must keep angle brackets, e.g. `<Alice>`, `<male_1>`, `<police_1>`.
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
  - his wallet → John's wallet
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
- "<Alice> puts coffee on table" → ["<Alice>", "puts", "coffee"], ["coffee", "is on", "table"]
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
  "<male_1> places the red cup on the counter.",
  "<Lisa> dances and sings happily.",
  "<John> takes his wallet and keys from the drawer.", 
  "<Betty> carries a red plastic bag in her left hand and a white plastic bag in her right hand."
]

Output:
[
  ["<Michael>", "pats shoulder", "<Susan>"],
  ["<Michael>", "smiles", null],
  ["<male_1>", "places", "red cup"],
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
- Keep character names as provided (e.g., <male_1>, <police_1>)
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

**OCR**: Text extracted from video frames (signs, labels, remarks)
- Format: `{"context": "sign in store", "content": "Open 24 Hours"}`
- Allocate 5-15 when query asks about visible text, signs, labels, or written information.
- Otherwise, `k_ocr` should be 0.

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
- **Source location queries**: "where can character get X?" / "where did X get Y from?" → Use triple `[X, "gets", "Y", ...]` or `[Y, "is in", "?", ...]` to find source locations. Include a helper triple if needed: `[Y, "is from", "?", ...]`.
- **Allocation for location queries**: Prioritize low-level edges (35-45) since they contain spatial information. Use conversations (5-10) only if placement instructions might be mentioned in dialogue.

2. **Allocation** `{k_high_level, k_low_level, k_conversations, k_appearance, k_ocr}`:
   - Total must be ≤ 50
   - High-level: 5-10 max (limited availability)
   - Low-level: 30-45 for action queries
   - Conversations: 10-45 based on needs
   - Appearance:
     - If the question is about character appearance, clothing, hairstyle, facial/body features, accessories:
       allocate `k_appearance` > 0 (typically 5-15).
     - Otherwise, `k_appearance` MUST be 0.
   - OCR:
     - If the question is about visible text, signs, labels, or written information:
       allocate `k_ocr` > 0 (typically 5-15).
     - Otherwise, `k_ocr` MUST be 0.
   - `total_k` must equal: `k_high_level + k_low_level + k_conversations + k_appearance + k_ocr`

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
    k_high_level=10, k_low_level=10, k_conversations=30, k_appearance=0, k_ocr=0, total_k=50,
    reasoning="Relationship query - use high-level for relationships, conversations for evidence"
  )
)

**Example 2**: "What did Emma do with the coffee in the kitchen?"
ParseQueryOutput(
  query_triples=[["<Emma>", "?", "coffee", 0.95, 0.15, 0.9]],
  spatial_constraint="kitchen",
  speaker_strict=None,
  allocation=ParseQueryAllocation(
    k_high_level=5, k_low_level=38, k_conversations=7, k_appearance=0, k_ocr=0, total_k=50,
    reasoning="Action query - prioritize low-level edges"
  )
)

**Example 3**: "What did Emily and David discuss?"
ParseQueryOutput(
  query_triples=[["<Emily>", "discusses", "<David>", 0.9, 0.3, 0.9]],
  spatial_constraint=None,
  speaker_strict=["<Emily>", "<David>"],
  allocation=ParseQueryAllocation(
    k_high_level=2, k_low_level=3, k_conversations=45, k_appearance=0, k_ocr=0, total_k=50,
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
    k_high_level=2, k_low_level=40, k_conversations=8, k_appearance=0, k_ocr=0, total_k=50,
    reasoning="Main triple targets usage by Lily; helper triple constrains items to dressing table"
  )
)

**Example 5**: "where is the tape now?"
ParseQueryOutput(
  query_triples=[["tape", "is at", "?", 0.8, 0.5, 0.15]],
  spatial_constraint=None,
  speaker_strict=None,
  allocation=ParseQueryAllocation(
    k_high_level=2, k_low_level=42, k_conversations=6, k_appearance=0, k_ocr=0, total_k=50,
    reasoning="Temporal-spatial query - 'now' means most recent location. Prioritize low-level edges with highest clip_id to find current state"
  )
)

**Example 6**: "What does the sign in the store say?"
ParseQueryOutput(
  query_triples=[["sign", "says", "?", 0.9, 0.7, 0.3]],
  spatial_constraint=None,
  speaker_strict=None,
  allocation=ParseQueryAllocation(
    k_high_level=2, k_low_level=10, k_conversations=3, k_appearance=0, k_ocr=35, total_k=50,
    reasoning="OCR query - prioritize OCR information for visible text"
  )
)

Now parse the following query and allocate k=50:
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
- **Source location queries**: "where can character get X?" / "where did X get Y from?" -> Use triple `[X, "gets", "Y", ...]` or `[Y, "is in", "?", ...]`. Include a helper triple if needed: `[Y, "is from", "?", ...]`.

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

prompt_planner_system = """You are a strategic Planner answering questions about a video. You have access to tools that search the video's knowledge graph and watch raw clips. You have {budget} turns left to gather evidence. Think step-by-step: analyze what you know, what is missing, and which tool is best to fill the gap. Then, call EXACTLY ONE tool to gather missing information. Once you have sufficient evidence to confidently answer the user's question, call complete_task.

The system processes video information in three layers:
1. **Video**: Videos are split into 30-second segments, each assigned a unique clip_id (1, 2, 3, ...)
2. **Text**: Each segment's text descriptions (behaviors, conversations, scenes, OCR) are stored by clip_id
3. **Graph**: Text is converted into graph edges with different types:
   - **High-level** : Abstract attributes/relationships
   - **Appearance** : Character physical looks, hair, clothing
   - **Low-level** : Specific actions/states with temporal and spatial information
   - **Conversations** : Dialogue transcripts [speaker, text] pairs
   - **OCR** : Text extracted from video frames (signs, labels, remarks)
   Each edge's clip_id links back to its original video segment.
All the current text information provided is from the graph.

Input format:
- **Parentheses (X)**: Confidence scores (0-100) in high-level information, indicating reliability.
  Example: Anna is: health-conscious (80) means 80% confidence.
- **Square brackets [X]**: Clip IDs indicating timestamps. Each clip = 30 seconds: clip 1 = 0-30s, clip 2 = 30-60s, clip 3 = 60-90s, etc.
  Applies to both low-level actions and conversation messages.
  Example: [1] Anna walk. (ping-pong room) means this occurred during clip 1 (0-30 seconds)."""

prompt_planner_strategy = """## Tool usage guidance:
- `general_search`: Use this FIRST for all questions to get a temporal anchor and find relevant `clip_id`s. Allocate your budget (Total k <= 50) based on the primary modality of the question:
  - **k_action (0-30)**: Primary for behavior, actions, temporal sequence, or 'where is' queries.
  - **k_conversation (0-30)**: Primary for 'why', dialogue, or causal reasoning.
  - **k_ocr (0-30)**: Primary for text on signs, labels, or posters.
  - **k_high_level (0-10)**: Secondary for character traits or relationships.
  - **k_appearance (0-15)**: Use ONLY for physical looks, hair, or clothing. Set to 0 if the question is irrelevant to appearance.
  **IMPORTANT (First Round)**: If this is your first tool call, you MUST use `general_search` and allocate the FULL budget (Total k=50) across these modalities to ensure a broad understanding of the video.

- `search_temporal_context`: Use this ONLY after finding a candidate `clip_id` via `general_search` to see events right before or after it.

- `watch_video_clip`: Use this ONLY after finding a candidate `clip_id` via `general_search`. Mandatory for visual questions requiring high detail (e.g., specific placement of objects, visual state, exact counts) when text graph is insufficient. Provide a specific `focus` based on what is missing from the text.

## Strategy for Clip ID Selection:
1. Look for the `[clip_id]` next to the most relevant action or conversation in `general_search` results.
2. Pay attention to temporal context: if looking for what happened *before* an event, search the clip(s) preceding the event.
3. Use that specific `clip_id` as input for `watch_video_clip` or `search_temporal_context`.

Analyze the conversation history above. What is the most effective next step to solve the question?"""

prompt_final_answer = """You are the Final Answer synthesizer. Using all the collected evidence in the conversation history, provide a concise, direct answer to the original question. Respond in exactly ONE SENTENCE. Do NOT include explanations, meta-commentary, or justifications. 

**IMPORTANT**: 
- Answers like "I don't know", "The information is not sufficient", or "It is unclear" are STRICTLY FORBIDDEN. 
- If you are uncertain, you MUST make the most reasonable guess based on the available evidence in the history. 
- Reuse exact names and terms from the question and search results."""
