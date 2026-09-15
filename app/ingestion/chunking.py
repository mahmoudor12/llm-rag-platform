"""
Chunking-Strategien.

Wir implementieren bewusst ZWEI Strategien, um später im Experiment E1
die Qualität vergleichen zu können:

    1. FixedTokenChunker   — Baseline, 800 Tokens / 100 Overlap
    2. StructureAwareChunker — respektiert Markdown-Überschriften

Der Chunk ist die Einheit, die später embeddet und in Qdrant gespeichert wird.
Die Chunk-Qualität entscheidet über die Retrieval-Qualität.
"""
import hashlib
import logging
import re
from abc import ABC, abstractmethod

import tiktoken

from app.ingestion.schemas import Chunk, CleanedDocument

logger = logging.getLogger(__name__)

# Standard-Encoding — Approximation für Token-Zählung.
# Qwen hat einen eigenen Tokenizer, aber für Chunk-Größen-Bemessung
# ist tiktoken eine praktikable Approximation.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_ENCODING.encode(text))


def _hash_chunk(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class Chunker(ABC):
    """Interface für alle Chunking-Strategien."""

    @abstractmethod
    def chunk(self, doc: CleanedDocument) -> list[Chunk]:
        """Zerlegt ein Dokument in Chunks."""


class FixedTokenChunker(Chunker):
    """
    Baseline: Token-basiertes Chunking mit Overlap.

    Einfach zu verstehen, schnell, oft überraschend gut. Der Nachteil:
    Es kann mitten in Sätzen oder Code-Blöcken schneiden.
    """

    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        if overlap >= chunk_size:
            raise ValueError("overlap muss kleiner als chunk_size sein")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc: CleanedDocument) -> list[Chunk]:
        tokens = _ENCODING.encode(doc.content)
        if not tokens:
            return []

        chunks: list[Chunk] = []
        start = 0
        idx = 0

        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = _ENCODING.decode(chunk_tokens).strip()

            if chunk_text:
                chunks.append(self._make_chunk(doc, chunk_text, idx))
                idx += 1

            if end >= len(tokens):
                break
            start = end - self.overlap

        return chunks

    @staticmethod
    def _make_chunk(doc: CleanedDocument, text: str, idx: int) -> Chunk:
        return Chunk(
            chunk_id=f"{doc.document_id}_chunk_{idx:04d}",
            document_id=doc.document_id,
            source=doc.source,
            title=doc.title,
            text=text,
            chunk_index=idx,
            content_hash=_hash_chunk(text),
            embedding_model="",  # wird später gesetzt
            metadata={"strategy": "fixed_token"},
        )


class StructureAwareChunker(Chunker):
    """
    Respektiert Markdown-Überschriften.

    Zerlegt das Dokument zuerst an Markdown-Headings (#, ##, ###), dann
    werden zu große Abschnitte weiter in Token-Batches unterteilt.

    Vorteil: Jeder Chunk ist semantisch zusammenhängend.
    Nachteil: Chunk-Größen variieren stark.
    """

    HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def __init__(self, max_chunk_tokens: int = 800, overlap: int = 100):
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap = overlap
        self._fallback = FixedTokenChunker(chunk_size=max_chunk_tokens, overlap=overlap)

    def chunk(self, doc: CleanedDocument) -> list[Chunk]:
        # Dokument an Überschriften zerlegen
        sections = self._split_by_headings(doc.content)
        chunks: list[Chunk] = []
        idx = 0

        for section_title, section_text in sections:
            section_text = section_text.strip()
            if not section_text:
                continue

            # Wenn der Abschnitt zu groß ist: intern mit FixedTokenChunker splitten
            if _token_len(section_text) > self.max_chunk_tokens:
                # Konstruiere temporäres Dokument für Fallback-Chunker
                tmp_doc = doc.model_copy(update={"content": section_text})
                sub_chunks = self._fallback.chunk(tmp_doc)
                for sc in sub_chunks:
                    sc.chunk_id = f"{doc.document_id}_chunk_{idx:04d}"
                    sc.chunk_index = idx
                    sc.metadata = {
                        "strategy": "structure_aware",
                        "section": section_title,
                    }
                    chunks.append(sc)
                    idx += 1
            else:
                chunks.append(Chunk(
                    chunk_id=f"{doc.document_id}_chunk_{idx:04d}",
                    document_id=doc.document_id,
                    source=doc.source,
                    title=doc.title,
                    text=section_text,
                    chunk_index=idx,
                    content_hash=_hash_chunk(section_text),
                    embedding_model="",
                    metadata={
                        "strategy": "structure_aware",
                        "section": section_title,
                    },
                ))
                idx += 1

        return chunks

    def _split_by_headings(self, content: str) -> list[tuple[str, str]]:
        """
        Teilt den Content an Markdown-Headings.
        Gibt Liste von (heading_title, section_text) zurück.
        """
        sections: list[tuple[str, str]] = []
        matches = list(self.HEADING_RE.finditer(content))

        if not matches:
            # Keine Headings → ein einziger Abschnitt
            return [("", content)]

        # Text vor dem ersten Heading
        if matches[0].start() > 0:
            preamble = content[:matches[0].start()].strip()
            if preamble:
                sections.append(("preamble", preamble))

        # Jeder Abschnitt: von einem Heading bis zum nächsten
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_title = match.group(2).strip()
            section_text = content[start:end].strip()
            sections.append((section_title, section_text))

        return sections


# --- Convenience-Funktionen ---

def chunk_document(doc: CleanedDocument, chunker: Chunker) -> list[Chunk]:
    """Chunkt ein einzelnes Dokument mit der übergebenen Strategie."""
    try:
        return chunker.chunk(doc)
    except Exception as e:
        logger.error("Chunking fehlgeschlagen für %s: %s", doc.document_id, e)
        return []


def chunk_documents(
    docs: list[CleanedDocument],
    chunker: Chunker | None = None,
) -> list[Chunk]:
    """Chunkt eine Liste von Dokumenten. Default: FixedTokenChunker(800/100)."""
    if chunker is None:
        chunker = FixedTokenChunker(chunk_size=800, overlap=100)

    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunker))

    logger.info(
        "Chunking abgeschlossen: %d Dokumente → %d Chunks",
        len(docs),
        len(all_chunks),
    )
    return all_chunks