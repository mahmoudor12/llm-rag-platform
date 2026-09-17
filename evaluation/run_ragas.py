"""
RAGAS-Evaluation fuer die RAG-Plattform.

Fuehrt das Golden Dataset durch die RAG-Pipeline und berechnet:
    - Faithfulness
    - Answer Relevancy
    - Context Precision
    - Context Recall

Aufruf:
    python -m evaluation.run_ragas
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# RAGAS initialisiert intern OpenAI. Wir setzen einen Fake-Key,
# weil der LLM-Judge ueber Ollama laeuft.
os.environ.setdefault("OPENAI_API_KEY", "sk-fake-key-for-local-eval")

from datasets import Dataset
from ragas import evaluate

# Neue Import-Pfade (RAGAS 0.2.x+) mit Fallback
try:
    from ragas.metrics.collections import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
except ImportError:
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

from app.embeddings import get_embedding_provider
from app.generation import SYSTEM_PROMPT, build_prompt
from app.llm import create_provider
from app.retrieval import QdrantStore
from langchain_community.chat_models import ChatOllama

GOLDEN_PATH = Path(__file__).parent / "golden_dataset.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"


def load_golden() -> list[dict]:
    """Laedt das Golden Dataset aus JSONL."""
    if not GOLDEN_PATH.exists():
        raise FileNotFoundError(
            f"Golden Dataset fehlt: {GOLDEN_PATH}\n"
            f"Erstelle die Datei mit 10+ Frage/Antwort-Paaren."
        )
    items = []
    with open(GOLDEN_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_pipeline(question: str, top_k: int = 2) -> dict:
    """Fuehrt eine Frage durch die komplette RAG-Pipeline."""
    emb = get_embedding_provider()
    store = QdrantStore()
    llm = create_provider()

    qvec = emb.encode_query(question)
    results = store.search(qvec, top_k=top_k)

    contexts = [r.text for r in results]
    prompt = build_prompt(question, results)
    response = llm.generate(prompt=prompt, system=SYSTEM_PROMPT)

    return {
        "answer": response.text,
        "contexts": contexts,
        "retrieval_count": len(results),
        "latency_ms": response.latency_ms,
    }


def main():
    print("=" * 60)
    print("RAGAS-Evaluation")
    print("=" * 60)

    golden = load_golden()
    print(f"\n[1/3] Golden Dataset: {len(golden)} Fragen")

    print(f"\n[2/3] Fuehre RAG-Pipeline aus...")
    rows = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    start = time.time()
    for i, item in enumerate(golden, 1):
        print(f"  [{i}/{len(golden)}] {item['question'][:60]}...")
        try:
            result = run_pipeline(item["question"])
            rows["question"].append(item["question"])
            rows["answer"].append(result["answer"])
            rows["contexts"].append(result["contexts"])
            rows["ground_truth"].append(item["ground_truth"])
        except Exception as e:
            print(f"    FEHLER: {e}")

    pipeline_time = time.time() - start
    print(f"      Pipeline fertig in {pipeline_time:.0f}s")

    if not rows["question"]:
        print("FEHLER: Keine erfolgreichen Queries.")
        sys.exit(1)

    print(f"\n[3/3] Berechne RAGAS-Metriken...")
    dataset = Dataset.from_dict(rows)

    judge_llm = ChatOllama(
        model="qwen2.5:1.5b",
        base_url="http://localhost:11434",
        temperature=0.1,
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
    )

    print("\n" + "=" * 60)
    print("ERGEBNIS")
    print("=" * 60)

    df = result.to_pandas()
    for metric in [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]:
        if metric in df.columns:
            avg = df[metric].mean()
            bar = "#" * int(avg * 30)
            print(f"  {metric:<22} {avg:.4f}  {bar}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"ragas_{timestamp}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nDetails gespeichert: {out_path}")


if __name__ == "__main__":
    main()
