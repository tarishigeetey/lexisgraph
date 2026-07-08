# LexisGraph — Enterprise Legal RAG

Production-grade retrieval-augmented generation over legal contracts, built on
the CUAD dataset. Combines hybrid retrieval (dense + BM25), reranking, a Neo4j
knowledge graph, PII masking, guardrails, and a RAGAS-gated evaluation harness.

## Status

Under active development. Built incrementally:

- [x] Project skeleton, typed config, containerized infra
- [x] CUAD loader + overlapping chunker
- [x] Dense retrieval (Qdrant, cosine)
- [ ] BM25 sparse retrieval + Reciprocal Rank Fusion
- [ ] Cross-encoder reranking
- [ ] Neo4j knowledge graph + multi-hop retrieval
- [ ] Adaptive query router
- [ ] PII masking (Presidio) + guardrails
- [ ] RAGAS evaluation with faithfulness gate
- [ ] FastAPI serving layer
- [ ] ColPali multimodal indexing (GPU stage)

## Architecture

```
CUAD contracts
   -> loader -> chunker -> embedder -> Qdrant (dense)
                                    -> Elasticsearch (BM25)   [planned]
                                    -> Neo4j (graph)          [planned]
   query -> router -> retrievers -> RRF fusion -> reranker -> LLM (Ollama)
                                                           -> RAGAS gate
```

## Quickstart

```bash
# 1. Infrastructure
docker compose up -d

# 2. Config
cp .env.example .env          # then set CUAD_DIR to your CUAD path

# 3. Dependencies
uv sync

# 4. Ingest a small batch and smoke-test retrieval
uv run python scripts/ingest.py --limit 20 --recreate

# 5. Tests
uv run pytest
```

## Tech stack

Python 3.11 · Qdrant · sentence-transformers (bge-small) · FastAPI · Ollama ·
Neo4j · Elasticsearch · RAGAS · Docker Compose · uv
