
"""
FastAPI Service fuer die RAG-Plattform.

Endpoints:
    GET  /health               Service-Status
    GET  /ready                Readiness-Check
    GET  /v1/provider          Provider-Info
    GET  /v1/documents         Indexierte Dokumente (Uebersicht)
    POST /v1/query             RAG-Query: Retrieval + Generation + Quellen
"""
import logging
import time
from collections import Counter
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.embeddings import get_embedding_provider
from app.generation import SYSTEM_PROMPT, build_prompt
from app.llm import create_provider
from app.llm.provider import LLMProvider
from app.retrieval import QdrantStore
from app.schemas import (
    DocumentInfo,
    HealthResponse,
    ProviderInfo,
    QueryRequest,
    QueryResponse,
    Source,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_provider: LLMProvider | None = None
_store: QdrantStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _provider, _store
    logger.info(
        "Starte RAG API - Provider=%s, Model=%s, Collection=%s",
        settings.llm_provider,
        settings.llm_model,
        settings.qdrant_collection,
    )
    _provider = create_provider()
    _store = QdrantStore()
    yield
    logger.info("Fahre RAG API herunter")
    _provider = None
    _store = None


app = FastAPI(
    title="LLM / RAG Platform",
    description="Retrieval-Augmented Generation als Service.",
    version="0.2.0",
    lifespan=lifespan,
)


def _check_dependencies() -> dict[str, bool]:
    checks: dict[str, bool] = {}
    try:
        checks["llm"] = _provider.health() if _provider else False
    except Exception as e:
        logger.warning("LLM-Check fehlgeschlagen: %s", e)
        checks["llm"] = False
    try:
        _store.client.get_collections()
        checks["qdrant"] = True
    except Exception as e:
        logger.warning("Qdrant-Check fehlgeschlagen: %s", e)
        checks["qdrant"] = False
    return checks


@app.get("/health", response_model=HealthResponse)
def health():
    deps = _check_dependencies()
    return HealthResponse(
        status="ok" if all(deps.values()) else "degraded",
        provider=settings.llm_provider,
        model=settings.llm_model,
        dependencies=deps,
    )


@app.get("/ready")
def ready():
    deps = _check_dependencies()
    if not all(deps.values()):
        raise HTTPException(status_code=503, detail={"dependencies": deps})
    return {"status": "ready"}


@app.get("/v1/provider", response_model=ProviderInfo)
def provider_info():
    return ProviderInfo(
        provider=settings.llm_provider,
        model=settings.llm_model,
        embedding_model=settings.embedding_model,
    )


@app.get("/v1/documents", response_model=list[DocumentInfo])
def list_documents():
    try:
        points, _ = _store.client.scroll(
            collection_name=settings.qdrant_collection,
            limit=5000,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant-Fehler: {e}")

    counter: Counter[str] = Counter()
    titles: dict[str, str] = {}
    for p in points:
        src = p.payload.get("source", "unknown")
        counter[src] += 1
        if src not in titles:
            titles[src] = p.payload.get("title", src)

    return [
        DocumentInfo(source=src, title=titles[src], chunk_count=cnt)
        for src, cnt in sorted(counter.items())
    ]


@app.post("/v1/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if _provider is None or _store is None:
        raise HTTPException(status_code=503, detail="Service nicht initialisiert")

    start = time.perf_counter()

    try:
        provider_emb = get_embedding_provider()
        qvec = provider_emb.encode_query(req.question)
    except Exception as e:
        logger.exception("Embedding fehlgeschlagen")
        raise HTTPException(status_code=500, detail=f"Embedding-Fehler: {e}")

    try:
        results = _store.search(qvec, top_k=req.top_k)
    except Exception as e:
        logger.exception("Qdrant-Suche fehlgeschlagen")
        raise HTTPException(status_code=503, detail=f"Retrieval-Fehler: {e}")

    if not results:
        return QueryResponse(
            answer=(
                "Die bereitgestellten Dokumente enthalten dafuer "
                "keine ausreichende Information."
            ),
            sources=[],
            latency_ms=int((time.perf_counter() - start) * 1000),
            retrieval_count=0,
            model=settings.llm_model,
            provider=settings.llm_provider,
        )

    prompt = build_prompt(req.question, results)

    try:
        llm_resp = _provider.generate(prompt=prompt, system=SYSTEM_PROMPT)
    except Exception as e:
        logger.exception("LLM-Call fehlgeschlagen")
        raise HTTPException(status_code=502, detail=f"LLM-Fehler: {e}")

    sources = [
        Source(
            source=r.source,
            title=r.title,
            chunk_id=r.chunk_id,
            score=round(r.score, 4),
        )
        for r in results
    ]

    latency_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "Query beantwortet: retrieval=%d, llm_ms=%d, total_ms=%d",
        len(results),
        llm_resp.latency_ms,
        latency_ms,
    )

    return QueryResponse(
        answer=llm_resp.text,
        sources=sources if req.include_sources else [],
        latency_ms=latency_ms,
        retrieval_count=len(results),
        model=llm_resp.model,
        provider=llm_resp.provider,
    )
