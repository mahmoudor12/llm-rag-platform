"""
Ingestion-CLI.

Führt die komplette Offline-Pipeline aus:
    1. Laden aus data/raw/
    2. Bereinigen
    3. Chunking (wählbare Strategie)
    4. Speichern als JSONL

Aufruf:
    python -m scripts.ingest                          # default: fixed 800/100
    python -m scripts.ingest --strategy structure     # structure-aware
    python -m scripts.ingest --chunk-size 400 --overlap 50
"""
import argparse
import json
import sys
import time
from pathlib import Path

# Projektroot in sys.path (damit `app.*` importierbar ist)
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.chunking import (  # noqa: E402
    FixedTokenChunker,
    StructureAwareChunker,
    chunk_documents,
)
from app.ingestion.cleaning import clean_documents  # noqa: E402
from app.ingestion.loaders import load_documents  # noqa: E402

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).parent.parent / "data" / "processed"


def main():
    parser = argparse.ArgumentParser(description="Ingestion-Pipeline")
    parser.add_argument(
        "--strategy",
        choices=["fixed", "structure"],
        default="fixed",
        help="Chunking-Strategie",
    )
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--output", type=str, default=None,
                        help="Output-JSONL (default: abgeleitet aus Strategie)")
    args = parser.parse_args()

    start = time.time()

    # --- Strategie wählen ---
    if args.strategy == "fixed":
        chunker = FixedTokenChunker(chunk_size=args.chunk_size, overlap=args.overlap)
        default_name = f"chunks_fixed_{args.chunk_size}_{args.overlap}.jsonl"
    else:
        chunker = StructureAwareChunker(
            max_chunk_tokens=args.chunk_size, overlap=args.overlap
        )
        default_name = f"chunks_structure_{args.chunk_size}_{args.overlap}.jsonl"

    output_path = OUT_DIR / (args.output or default_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Pipeline ---
    print("=" * 60)
    print(f"Ingestion-Pipeline — Strategie: {args.strategy}")
    print("=" * 60)

    print(f"\n[1/4] Lade Dokumente aus: {RAW_DIR}")
    documents = load_documents(RAW_DIR)
    if not documents:
        print("❌ Keine Dokumente gefunden. Erst `python -m scripts.download_corpus` ausführen.")
        sys.exit(1)

    print(f"\n[2/4] Bereinige Dokumente...")
    cleaned = clean_documents(documents)

    print(f"\n[3/4] Chunke Dokumente...")
    chunks = chunk_documents(cleaned, chunker=chunker)

    # --- Statistik ---
    if chunks:
        token_counts = [len(c.text.split()) for c in chunks]  # grob
        avg_words = sum(token_counts) / len(token_counts)
        print(f"      Chunks gesamt: {len(chunks)}")
        print(f"      Ø Wörter/Chunk: {avg_words:.0f}")
        print(f"      Min/Max Wörter: {min(token_counts)}/{max(token_counts)}")

    print(f"\n[4/4] Speichere nach: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")

    duration = time.time() - start
    print(f"\n✅ Fertig in {duration:.1f}s")
    print(f"   {len(documents)} Dokumente → {len(chunks)} Chunks")
    print(f"   Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()