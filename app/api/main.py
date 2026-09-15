"""
FastAPI Service für die RAG-Plattform.

Aktuelle Version: Provider-Health + Direkt-LLM-Query (ohne Retrieval).
Retrieval folgt in Tag 5-7, wenn Embeddings + Qdrant-Index stehen.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from qdrant_client import QdrantClient

from app.config import settings
from app.llm import create_provider
from app.llm.provider import LLMProvider
from app.schemas import (
    HealthResponse,
    ProviderInfo,
    QueryRequest,
    QueryResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# --- Global state (im lifespan initialisiert) ---
_provider: LLMProvider | None = None
_qdrant: QdrantClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _provider, _qdrant
    logger.info(
        "Starte RAG API — Provider=%s, Model=%s",
        settings.llm_provider,
        settings.llm_model,
    )
    _provider = create_provider()
    _qdrant = QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        timeout=5.0,
    )
    yield
    logger.info("Fahre RAG API herunter")
    _provider = None
    _qdrant = None


app = FastAPI(
    title="LLM / RAG Platform",
    description=(
        "Retrieval-Augmented Generation als Service. "
        "Provider-agnostisch (Ollama lokal oder Anthropic API)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


def _check_dependencies() -> dict[str, bool]:
    checks: dict[str, bool] = {}

    try:
        checks["ollama"] = _provider.health() if _provider else False
    except Exception as e:
        logger.warning("Ollama-Check fehlgeschlagen: %s", e)
        checks["ollama"] = False

    try:
        _qdrant.get_collections()
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


@app.post("/v1/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Baseline-Endpoint: direkter LLM-Call ohne Retrieval.
    Retrieval folgt in Tag 5-7 (Embeddings + Qdrant-Index).
    """
    if _provider is None:
        raise HTTPException(status_code=503, detail="Provider nicht initialisiert")

    system = (
        "Du bist ein präziser, sachlicher Assistent. "
        "Antworte kurz auf Deutsch, maximal 4 Sätze."
    )

    try:
        resp = _provider.generate(prompt=req.question, system=system)
    except Exception as e:
        logger.exception("LLM-Call fehlgeschlagen")
        raise HTTPException(status_code=502, detail=f"LLM-Fehler: {e}")

    return QueryResponse(
        answer=resp.text,
        sources=[],                 # noch kein Retrieval
        latency_ms=resp.latency_ms,
        retrieval_count=0,
        model=resp.model,
        provider=resp.provider,
    )