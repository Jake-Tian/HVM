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
class ParseQueryOutput(BaseModel):
    # [source, content, target, source_weight, content_weight, target_weight]
    query_triples: list[list[str | float | None]]
    speaker_strict: list[str] | None
    k_behavior: int
    k_conversation: int


class AnswerWithSearchResultsOutput(BaseModel):
    answer: bool
    content: str
    summary: str | None
    tool_name: str | None
    target: str | None
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