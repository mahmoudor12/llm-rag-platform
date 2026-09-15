"""
Pydantic-Schemas für die Ingestion-Pipeline.

Diese Schemas sind der **Data Contract** zwischen den Ingestion-Stufen.
Ein `LoadedDocument` geht durch Cleaning, wird zu `CleanedDocument`,
dann gechunkt zu `Chunk`.
"""
from datetime import datetime
from pydantic import BaseModel, Field


class LoadedDocument(BaseModel):
    """Ein rohes, geladenes Dokument."""

    document_id: str = Field(description="Eindeutige ID, z.B. 'fastapi_tutorial_01'")
    source: str = Field(description="Dateiname oder URL")
    title: str = Field(description="Lesbarer Titel")
    content: str = Field(description="Roher Text")
    content_hash: str = Field(description="SHA256 des Inhalts (für Dedup)")
    metadata: dict = Field(default_factory=dict, description="Zusätzliche Metadaten")


class CleanedDocument(BaseModel):
    """Ein bereinigtes Dokument."""

    document_id: str
    source: str
    title: str
    content: str
    content_hash: str
    metadata: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    """Ein Chunk aus einem Dokument — die Einheit, die embeddet wird."""

    chunk_id: str = Field(description="z.B. 'fastapi_tutorial_01_chunk_007'")
    document_id: str
    source: str
    title: str
    text: str
    chunk_index: int = Field(ge=0)
    content_hash: str = Field(description="SHA256 des Chunk-Texts")
    embedding_model: str = Field(description="Name des Embedding-Modells (später gesetzt)")
    ingestion_version: str = Field(default="v1.0")
    metadata: dict = Field(default_factory=dict)