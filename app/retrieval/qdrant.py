
"""
Qdrant-Wrapper fuer Vektor-Suche.
"""
import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.ingestion.schemas import Chunk
from app.config import settings

logger = logging.getLogger(__name__)


class SearchResult:
    def __init__(self, chunk_id, score, payload):
        self.chunk_id = chunk_id
        self.score = score
        self.payload = payload

    @property
    def text(self):
        return self.payload.get("text", "")

    @property
    def source(self):
        return self.payload.get("source", "unknown")

    @property
    def title(self):
        return self.payload.get("title", "")

    def __repr__(self):
        return f"<SearchResult score={self.score:.4f} source={self.source}>"


class QdrantStore:
    def __init__(self, host=None, port=None, collection_name=None):
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.collection_name = collection_name or settings.qdrant_collection
        self.client = QdrantClient(host=self.host, port=self.port, timeout=30.0)

    def collection_exists(self):
        try:
            collections = self.client.get_collections().collections
            return any(c.name == self.collection_name for c in collections)
        except Exception as e:
            logger.error("Fehler beim Collection-Check: %s", e)
            return False

    def recreate_collection(self, vector_size):
        if self.collection_exists():
            logger.warning("Loesche existierende Collection: %s", self.collection_name)
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info("Collection angelegt: %s (dim=%d)", self.collection_name, vector_size)

    def ensure_collection(self, vector_size):
        if not self.collection_exists():
            self.recreate_collection(vector_size)
        else:
            info = self.client.get_collection(self.collection_name)
            existing_dim = info.config.params.vectors.size
            if existing_dim != vector_size:
                raise ValueError(
                    f"Collection existiert mit dim={existing_dim}, "
                    f"Modell liefert dim={vector_size}"
                )

    @staticmethod
    def _chunk_to_point(chunk, embedding):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
        payload = {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "source": chunk.source,
            "title": chunk.title,
            "text": chunk.text,
            "chunk_index": chunk.chunk_index,
            "content_hash": chunk.content_hash,
            "embedding_model": chunk.embedding_model,
            "ingestion_version": chunk.ingestion_version,
            "metadata": chunk.metadata,
        }
        return PointStruct(id=point_id, vector=embedding, payload=payload)

    def upsert_chunks(self, chunks, embeddings, batch_size=64):
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks und Embeddings muessen gleich lang sein")
        total = len(chunks)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            points = [
                self._chunk_to_point(chunks[i], embeddings[i].tolist())
                for i in range(start, end)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info("Upsert: %d/%d", end, total)
        return total

    def count(self):
        info = self.client.get_collection(self.collection_name)
        return info.points_count or 0

    def search(self, query_vector, top_k=5, source_filter=None):
        query_filter = None
        if source_filter:
            query_filter = Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source_filter))]
            )
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            SearchResult(
                chunk_id=hit.payload.get("chunk_id", str(hit.id)),
                score=hit.score,
                payload=hit.payload,
            )
            for hit in hits
        ]
