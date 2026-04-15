from pydantic import BaseModel

# Memorization Structures
class Appearance(BaseModel):
    name: str
    appearance: str

class OCRInfo(BaseModel):
    context: str
    content: str

class EpisodicFormat(BaseModel):
    behaviors: list[str]
    conversation: list[list[str]]
    characters_appearance: list[Appearance]
    scene: str | None
    ocr: list[OCRInfo]

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
    k_ocr: int
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

class GraphOutputFormat(BaseModel):
    answer: bool
    content: str | list[int]
    summary: str | None

class VideoOutputFormat(BaseModel):
    answer: bool
    content: str