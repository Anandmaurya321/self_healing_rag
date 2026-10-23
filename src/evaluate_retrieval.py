
import json
from pathlib import Path

import chromadb

from ingestion import load_document, split_document
from embedding import load_embedding_model
from dense_retrieval import semantic_search
from bm25_retrieval import build_bm25_index, bm25_search
from hybrid_retrieval import hybrid_search


# Paths are relative to the src directory.
PROJECT_ROOT = Path("..")
DOCUMENT_PATH = PROJECT_ROOT / "data/documents/company_policy.txt"
EVAL_PATH = PROJECT_ROOT / "test/retrieval_eval.jsonl"

# Keep the same ChromaDB path and collection as retrieval.py.
CHROMA_PATH = "/chromadb"
COLLECTION_NAME = "company_policy"

K = 2


def load_eval_dataset(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def calculate_metrics(results, expected_ids, k=2):
    """Calculate ranking metrics for an answerable query."""
    retrieved_ids = [item["id"] for item in results[:k]]
    relevant = set(expected_ids)

    hits = sum(1 for item_id in retrieved_ids if item_id in relevant)

    recall = hits / len(relevant) if relevant else None
    precision = hits / k

    reciprocal_rank = 0.0
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant:
            reciprocal_rank = 1 / rank
            break

    return {
        "recall": recall,
        "precision": precision,
        "mrr": reciprocal_rank,
    }


def main():
    # Load the original chunks.
    text = load_document(str(DOCUMENT_PATH))
    chunks = split_document(text)

    dataset = load_eval_dataset(EVAL_PATH)

    # Verify that ChromaDB contains the expected indexed chunks.
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    stored = collection.get(include=["documents"])
    stored_by_id = dict(zip(stored["ids"], stored["documents"]))

    expected_documents = {
        f"policy_{i}": chunk for i, chunk in enumerate(chunks)
    }

    if stored_by_id != expected_documents:
        raise ValueError(
            "ChromaDB does not match the current policy chunks. "
            "Check the database path and re-index the collection "
            "before evaluating."
        )

    # Build each retriever once.
    model = load_embedding_model()
    bm25 = build_bm25_index(chunks)

    dense_metrics = []
    bm25_metrics = []
    hybrid_metrics = []
    unanswerable_results = []
    

    print(f"Evaluation questions: {len(dataset)}")
    print(f"Policy chunks: {len(chunks)}")
    print(f"Metrics use K={K}; unanswerable questions are excluded.")

    for item in dataset:
        question = item["question"]
        expected_ids = item["expected_section_ids"]

        # Dense retrieval.
        dense_results = semantic_search(
            query=question,
            model=model,
            collection=collection,
            n_results=K,
        )

        # BM25 retrieval.
        sparse_results = bm25_search(
            query=question,
            chunks=chunks,
            bm25=bm25,
            n_results=K,
        )

        hybrid_results = hybrid_search(
            query=question,
            chunks=chunks,
            model=model,
            collection=collection,
            bm25=bm25,
            n_results=K,
        )

        if expected_ids:
            dense_metrics.append(
                calculate_metrics(dense_results, expected_ids, K)
            )
            bm25_metrics.append(
                calculate_metrics(sparse_results, expected_ids, K)
            )
            hybrid_metrics.append(
                calculate_metrics(hybrid_results, expected_ids, K)
            )
        else:
            # Do not count an unanswerable query as a normal recall test.
            unanswerable_results.append({
                "id": item["id"],
                "dense_top_id": (
                    dense_results[0]["id"] if dense_results else None
                ),
                "bm25_top_id": (
                    sparse_results[0]["id"] if sparse_results else None
                ),
                "bm25_top_score": (
                    sparse_results[0]["score"] if sparse_results else 0.0
                ),
            })

        # Compact output: no document text.
        dense_top = dense_results[0]["id"] if dense_results else "None"
        bm25_top = sparse_results[0]["id"] if sparse_results else "None"
        hybrid_top = (hybrid_results[0]["id"] if hybrid_results else "None")

        print(
            f'{item["id"]}: expected={expected_ids or "NONE"} | '
            f'Dense={dense_top} | BM25={bm25_top} | Hybrid={hybrid_top}'
        )

    def report(name, rows):
        if not rows:
            print(f"\n{name}: no answerable queries to evaluate.")
            return

        print(f"\n{name}")
        for metric in ("recall", "precision", "mrr"):
            avg = sum(row[metric] for row in rows) / len(rows)
            print(f"  {metric.upper()}@{K}: {avg:.4f}")

    report("DENSE RETRIEVAL", dense_metrics)
    report("BM25 RETRIEVAL", bm25_metrics)
    report("HYBRID RETRIEVAL", hybrid_metrics)
    
    print("\nUNANSWERABLE QUESTIONS")
    for result in unanswerable_results:
        print(
            f'  {result["id"]}: '
            f'Dense top={result["dense_top_id"]}, '
            f'BM25 top={result["bm25_top_id"]}, '
            f'BM25 score={result["bm25_top_score"]:.4f}'
        )


if __name__ == "__main__":
    main()



