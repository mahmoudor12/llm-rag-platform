"""
Pydantic-Schemas für den API-Vertrag.
"""
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    include_sources: bool = True


class Source(BaseModel):
    source: str
    page: int | None = None
    chunk_id: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    latency_ms: int
    retrieval_count: int
    model: str
    provider: str


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str
    dependencies: dict[str, bool]


class ProviderInfo(BaseModel):
    provider: str
    model: str
    embedding_model: str