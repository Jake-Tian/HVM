from pydantic import BaseModel, Field

# Memorization Structures
class Appearance(BaseModel):
    name: str
    appearance: str

class OCRInfo(BaseModel):
    context: str
    content: str

class EpisodicFormat(BaseModel):
    behaviors: list[str] = Field(default_factory=list)
    conversation: list[list[str]] = Field(default_factory=list)
    characters_appearance: list[Appearance] = Field(default_factory=list)
    scene: str | None = None
    ocr: list[OCRInfo] = Field(default_factory=list)

class ExtractedTriple(BaseModel):
    source: str
    content: str
    target: str | None = None

class TripleExtraction(BaseModel):
    triples: list[ExtractedTriple] = Field(default_factory=list)

class ConversationSummary(BaseModel):
    summary: str
    character_attributes: list[list[str | int | float]] = Field(default_factory=list)
    characters_relationships: list[list[str | int | float]] = Field(default_factory=list)

# Reasoning Structures
class ParseQueryAllocation(BaseModel):
    k_high_level: int
    k_low_level: int
    k_conversations: int
    k_appearance: int
    k_ocr: int
    total_k: int
    reasoning: str = ""

class ParseQueryOutput(BaseModel):
    # [source, content, target, source_weight, content_weight, target_weight]
    query_triples: list[list[str | float | None]] = Field(default_factory=list)
    spatial_constraint: str | None = None
    speaker_strict: list[str] | None = None
    allocation: ParseQueryAllocation

class ParseQueryOutputNoAllocation(BaseModel):
    query_triples: list[list[str | float | None]] = Field(default_factory=list)
    spatial_constraint: str | None = None
    speaker_strict: list[str] | None = None

class GraphOutputFormat(BaseModel):
    answer: bool
    content: str | list[int]
    summary: str | None = None

class VideoOutputFormat(BaseModel):
    answer: bool
    content: str
