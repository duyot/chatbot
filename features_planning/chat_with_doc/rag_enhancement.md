# Agentic RAG — enhanced architecture (post-2026-05-16, OpenRouter cutover 2026-06-28)

The chat-with-doc pipeline is implemented as a LangGraph state machine in
`backend/app/services/rag/`. See the design spec at
`docs/superpowers/specs/2026-05-15-agentic-rag-enhancement-design.md` and the
implementation plan at `docs/superpowers/plans/2026-05-15-agentic-rag-enhancement.md`.
On 2026-06-28 the stack moved off self-hosted Ollama/TEI: chat LLM is now
`anthropic/claude-haiku-4.5` via OpenRouter, embeddings are
`qwen/qwen3-embedding-8b` (truncated to 1536 dims via OpenAI-compatible
`dimensions=`) also via OpenRouter, and reranking calls OpenRouter's
dedicated `/v1/rerank` endpoint (cross-encoder; default
`anthropic/claude-haiku-4.5` or set to `nvidia/llama-nemotron-rerank-vl-1b-v2`
for a real reranker). Every model call exits through a single
OpenRouter API key.

## Flow

```
rewrite_query  ->  retrieve_and_rerank  ->  grade_chunks
                                              |
                       +----------------------+----------------------+
                       v useful               v retry (<=2)         v give_up
                  generate_answer       rewrite_and_retry       generate_answer
                       |                       |                   (NOT_FOUND)
                       |                       ^                       |
                       |                       +- back to retrieve     |
                       v                                                v
                  faithfulness_check  --------------------------------- END
```

## Module layout

| File | Responsibility |
|---|---|
| `rag/graph.py` | LangGraph wiring + `agentic_rag_stream()` entry point |
| `rag/nodes.py` | Six nodes (rewrite_query, retrieve_and_rerank, grade_chunks, rewrite_and_retry, generate_answer, faithfulness_check) |
| `rag/state.py` | `AgentState` TypedDict |
| `rag/prompts.py` | All prompts |
| `rag/retrieval.py` | hybrid_search -> rrf_fuse -> rerank -> fetch_parents |
| `rag/reranker.py` | OpenRouter `/v1/rerank` cross-encoder; httpx POST with `{model, query, documents, top_n}`, returns sorted `(chunk, relevance_score)` |

## Data model

`document_parent_chunks` stores large (~1500 token) parents fed to the LLM;
`document_chunks` are 300-token children that get embedded + FTS-indexed and
carry `parent_id`. See migration `0004_parent_child_chunks.py`.

## Evaluation

Golden Q/A set at `backend/evals/golden_set.yaml`. Run with:

`cd backend && python -m evals.run_eval --name <run_name>`
`python -m evals.run_eval --compare <run_a> <run_b>`

Baseline lives at `backend/evals/results/baseline_*.json`. Run an eval before
shipping any retrieval/agent change.

## Tunables

All in `backend/app/config.py`: `vector_top_k`, `fts_top_k`, `rrf_k`,
`rerank_top_n`, `rerank_score_floor`, `max_retrieval_retries`, `strict_grader`,
`reranker_model` (override to use a stronger model for ranking only, e.g.
`anthropic/claude-sonnet-4.6`).

## Debug payload

When `LOG_LEVEL=DEBUG`, the streamed `done` event carries `debug.attempted_queries`,
`debug.retry_count`, `debug.intent`, and `debug.notes` — useful for tracing why a
question landed on a particular answer path.
