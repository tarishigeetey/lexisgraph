"""Ingest CUAD contracts into Qdrant.

Pipeline: load contracts -> chunk each -> embed + index in Qdrant.

Run:  uv run python scripts/ingest.py --limit 20
(Start with a small --limit to prove it works, then raise it.)
"""

from __future__ import annotations

import argparse
import time

from lexisgraph.ingest.chunker import chunk_text
from lexisgraph.ingest.loader import load_contracts
from lexisgraph.retrieval.dense import DenseRetriever
from lexisgraph.retrieval.embedder import Embedder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=20, help="Number of contracts to ingest (start small)."
    )
    parser.add_argument("--recreate", action="store_true", help="Drop and rebuild the collection.")
    args = parser.parse_args()

    print("Loading embedding model...")
    embedder = Embedder()
    retriever = DenseRetriever(embedder=embedder)
    retriever.ensure_collection(recreate=args.recreate)

    all_chunks = []
    n_docs = 0
    for source, text in load_contracts(limit=args.limit):
        chunks = chunk_text(text, source=source)
        all_chunks.extend(chunks)
        n_docs += 1
    print(f"Loaded {n_docs} contracts -> {len(all_chunks)} chunks.")

    t0 = time.time()
    indexed = retriever.index(all_chunks)
    dt = time.time() - t0
    print(f"Indexed {indexed} chunks in {dt:.1f}s.")

    # Smoke test: run a query so you see retrieval working immediately.
    q = "who is responsible for indemnifying the other party?"
    print(f"\nSmoke-test query: {q}")
    for i, hit in enumerate(retriever.search(q, limit=3), 1):
        print(f"  {i}. [{hit.score:.3f}] {hit.source}")
        print(f"     {hit.text[:120].strip()}...")


if __name__ == "__main__":
    main()
