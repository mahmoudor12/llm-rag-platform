"""
Text-Normalisierung.

Entfernt Artefakte, die das Chunking und Retrieval stören:
    - übermäßige Leerzeilen
    - HTML-Reste
    - Markdown-Bilder (die als Text keinen Wert haben)
    - FastAPI-Include-Directives ({* ... *})
    - Trailing Whitespace
"""
import logging
import re

from bs4 import BeautifulSoup

from app.ingestion.schemas import CleanedDocument, LoadedDocument

logger = logging.getLogger(__name__)

# Regex-Muster
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+\n")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]+\)")            # ![alt](url)
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")            # [text](url) → text
_MD_BADGE = re.compile(r"\[!\[[^\]]*\]\([^)]+\)\]\([^)]+\)")  # verschachtelt
# FastAPI-Include-Directives: {* ../path/file.py hl[10] *}
# Können mehrzeilig sein → DOTALL
_MD_INCLUDE = re.compile(r"\{\*.*?\*\}", re.DOTALL)


def clean_markdown(content: str) -> str:
    """Bereinigt Markdown-Text."""
    # 1. Badges und Bilder raus — sie haben keinen Textwert
    content = _MD_BADGE.sub("", content)
    content = _MD_IMAGE.sub("", content)

    # 2. Links: [text](url) → text
    content = _MD_LINK.sub(r"\1", content)

    # 3. FastAPI-Include-Directives entfernen (Rauschen im Retrieval-Kontext)
    content = _MD_INCLUDE.sub("", content)

    # 4. HTML-Reste entfernen
    if "<" in content and ">" in content:
        try:
            content = BeautifulSoup(content, "lxml").get_text("\n")
        except Exception as e:
            logger.warning("HTML-Bereinigung fehlgeschlagen: %s", e)

    return content


def clean_text(content: str) -> str:
    """Generische Text-Bereinigung (für alle Formate)."""
    # 1. Windows-Zeilenenden → Unix
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Trailing Whitespace entfernen
    content = _TRAILING_WS.sub("\n", content)

    # 3. Mehrfache Leerzeilen normalisieren
    content = _MULTI_NEWLINE.sub("\n\n", content)

    # 4. Am Anfang/Ende trimmen
    content = content.strip()

    return content


def clean_document(doc: LoadedDocument) -> CleanedDocument:
    """Bereinigt ein Dokument basierend auf seinem Format."""
    content = doc.content

    if doc.metadata.get("format") == "markdown":
        content = clean_markdown(content)

    content = clean_text(content)

    return CleanedDocument(
        document_id=doc.document_id,
        source=doc.source,
        title=doc.title,
        content=content,
        content_hash=doc.content_hash,
        metadata=doc.metadata,
    )


def clean_documents(docs: list[LoadedDocument]) -> list[CleanedDocument]:
    """Bereinigt eine Liste von Dokumenten."""
    cleaned: list[CleanedDocument] = []
    for doc in docs:
        try:
            cleaned.append(clean_document(doc))
        except Exception as e:
            logger.error("Fehler beim Bereinigen von %s: %s", doc.document_id, e)
    logger.info("Bereinigt: %d Dokumente", len(cleaned))
    return cleaned