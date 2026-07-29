# Task: Enhance Ingestion - Retrieval of document
## Context
1. D:\development\chatbot\features_planning\7.enhancing_rag_pipeline\claude_contextual_embedding.md is the guideline from claude to enhance the RAG process
## Requirements
1. Read deeply the guideline in claude_contextual_embedding.md to see how to enhance the rag pipeline
2. Refer to D:\development\chatbot\wiki on how Ingestion - Retrieval currently work.
3. Provide the plan for the enhancement of flow.
4. The key idea I can see is:
a. apply Contextual Embeddings for the chunk, which mean the chunk may contain  a concise explanation of what the chunk contains and where it fits in the overall file. which may richer vector representations.
b. apply contextual BM25: Hybrid Search, which including bm25 search and semantic search (vector search), then Score fusion with RRF
c. Reranking on final set of result with the best match.
## References
1. D:\development\chatbot\features_planning\7.enhancing_rag_pipeline\claude_contextual_embedding.md
2. D:\development\chatbot\wiki