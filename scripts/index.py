
"""
Index-CLI: Chunks -> Embeddings -> Qdrant.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.embeddings import get_embedding_provider
from app.ingestion.schemas import Chunk
from app.retrieval import QdrantStore

DEFAULT_INPUT = Path(__file__).parent.parent / "data" / "processed" / "chunks_fixed_800_100.jsonl"


def load_chunks(path):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(Chunk(**json.loads(line)))
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Index chunks into Qdrant")
    parser.add_argument("--file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--collection", type=str, default=None)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not args.file.exists():
        print(f"FEHLER: Datei nicht gefunden: {args.file}")
        sys.exit(1)

    print("=" * 60)
    print(f"Indexing: {args.file.name}")
    print("=" * 60)

    print(f"\n[1/4] Lade Chunks...")
    chunks = load_chunks(args.file)
    print(f"      {len(chunks)} Chunks geladen")

    print(f"\n[2/4] Lade Embedding-Modell: {settings.embedding_model}")
    provider = get_embedding_provider()
    print(f"      Dimension: {provider.dimension}")

    collection_name = args.collection or settings.qdrant_collection
    store = QdrantStore(collection_name=collection_name)

    print(f"\n[3/4] Qdrant-Collection: {collection_name}")
    if args.recreate:
        print("      -> recreate erzwungen")
        store.recreate_collection(vector_size=provider.dimension)
    else:
        store.ensure_collection(vector_size=provider.dimension)
        existing = store.count()
        if existing > 0:
            print(f"      WARNUNG: Collection enthaelt bereits {existing} Punkte")

    print(f"\n[4/4] Embedde und speichere...")
    start = time.time()
    texts = [c.text for c in chunks]
    embeddings = provider.encode_documents(texts, batch_size=args.batch_size, show_progress=True)
    embed_time = time.time() - start
    print(f"      Embedding fertig in {embed_time:.1f}s ({len(chunks)/embed_time:.1f} chunks/s)")

    for c in chunks:
        c.embedding_model = settings.embedding_model

    stored = store.upsert_chunks(chunks, embeddings)
    total_time = time.time() - start
    print(f"\nOK: Fertig in {total_time:.1f}s")
    print(f"   {stored} Chunks in '{collection_name}' gespeichert")
    print(f"   Gesamt in Collection: {store.count()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
