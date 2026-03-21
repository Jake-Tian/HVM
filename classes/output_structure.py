from pydantic import BaseModel

# Memorization Structures
class TimeTriple(BaseModel):
    time: str
    triple: list[str]

class TimeTripleList(BaseModel):
    triples: list[TimeTriple]

class Appearance(BaseModel):
    name: str
    appearance: str

class EpisodicFormat(BaseModel):
    behaviors: list[str]
    conversation: list[list[str]]
    characters_appearance: list[Appearance]
    scene: str

class ConversationSummary(BaseModel):
    summary: str
    character_attributes: list[list[str | int | float]]
    characters_relationships: list[list[str | int | float]]

# Reasoning Structures
class ParseQueryAllocation(BaseModel):
    k_high_level: int
    k_low_level: int
    k_conversations: int
    k_appearance: int
    total_k: int
    reasoning: str

class ParseQueryOutput(BaseModel):
    # [source, content, target, source_weight, content_weight, target_weight]
    query_triples: list[list[str | float | None]]
    spatial_constraint: str | None
    speaker_strict: list[str] | None
    allocation: ParseQueryAllocation

class ParseQueryOutputNoAllocation(BaseModel):
    query_triples: list[list[str | float | None]]
    spatial_constraint: str | None
    speaker_strict: list[str] | None

class AllocateSearchOutput(BaseModel):
    k_behavior: int
    k_conversation: int
    speaker_strict: list[str] | None
    reasoning: str

class AnswerWithSearchResultsOutput(BaseModel):
    answer: bool
    content: str
    summary: str | None
    total_search_k: int | None
    k_behavior: int | None
    k_conversation: int | None
    speaker_strict: list[str] | None

class AnswerWithSearchResultsFinalOutput(BaseModel):
    content: str
    summary: str

class GraphOutputFormat(BaseModel):
    answer: bool
    content: str | list[int]
    summary: str | None

class VideoOutputFormat(BaseModel):
    answer: bool
    content: str