"""
Document Loader.

Unterstützt:
    - Markdown (.md)
    - Text (.txt)
    - PDF (.pdf) — über pypdf (kommt später)

Für die Portfolio-Version starten wir mit Markdown + Text.
PDF-Support kann später ergänzt werden.
"""
import hashlib
import logging
from pathlib import Path

from app.ingestion.schemas import LoadedDocument

logger = logging.getLogger(__name__)


def _hash_content(content: str) -> str:
    """SHA256-Hash des Inhalts — für Dedup und Change-Detection."""
    return "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _extract_title_from_markdown(content: str, fallback: str) -> str:
    """Extrahiert die erste H1-Überschrift aus Markdown."""
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def load_markdown(path: Path) -> LoadedDocument:
    """Lädt eine Markdown-Datei."""
    content = path.read_text(encoding="utf-8")
    title = _extract_title_from_markdown(content, fallback=path.stem)

    # document_id aus relativem Pfad (ohne Endung)
    doc_id = str(path.relative_to(path.parents[1] if len(path.parents) > 1 else path.parent))
    doc_id = doc_id.replace("\\", "_").replace("/", "_").rsplit(".", 1)[0]

    return LoadedDocument(
        document_id=doc_id,
        source=path.name,
        title=title,
        content=content,
        content_hash=_hash_content(content),
        metadata={"format": "markdown", "path": str(path)},
    )


def load_text(path: Path) -> LoadedDocument:
    """Lädt eine Textdatei."""
    content = path.read_text(encoding="utf-8")
    doc_id = path.stem

    return LoadedDocument(
        document_id=doc_id,
        source=path.name,
        title=path.stem,
        content=content,
        content_hash=_hash_content(content),
        metadata={"format": "text", "path": str(path)},
    )


# Registry: Dateiendung → Loader-Funktion
LOADERS = {
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".txt": load_text,
}


def load_documents(raw_dir: Path) -> list[LoadedDocument]:
    """
    Lädt alle unterstützten Dateien aus einem Verzeichnis (rekursiv).

    Fehlerhafte Dateien werden geloggt, blockieren aber nicht die Pipeline.
    """
    documents: list[LoadedDocument] = []
    skipped = 0

    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue

        loader = LOADERS.get(path.suffix.lower())
        if loader is None:
            logger.debug("Überspringe Datei mit unbekanntem Format: %s", path.name)
            skipped += 1
            continue

        try:
            doc = loader(path)
            documents.append(doc)
        except Exception as e:
            logger.error("Fehler beim Laden von %s: %s", path, e)
            skipped += 1

    logger.info(
        "Geladen: %d Dokumente, übersprungen: %d",
        len(documents),
        skipped,
    )
    return documents