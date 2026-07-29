
## 1. Document Summarization for Enhanced Retrieval
### Main idea
- Instead of merly storing embbeded data, storing metada (heading, raw content) and ai summary for each chunk
- Use raw meta data + summary for embedding process
### Process
- Load document chunks
- Each chunk -> use AI to generate short summary
- Store both meta data + summary into database

### How to store summary in database
- Create embedding for combined text (heading - summary - raw content)
- Store embedding + metadata (including summary) into database\

### Enhanced Retrieval Using Summary-Indexed
- Parallel with retrieve the result from similarity query, we get the meta data (heading, summary, raw content)
- this enriched context is used to generate user's query

```python
def retrieve_level_two(query, db):
    results = db.search(query, k=3)
    context = ""
    for result in results:
        chunk = result["metadata"]
        context += f"\n <document> \n {chunk['chunk_heading']}\n\nText\n {chunk['text']} \n\nSummary: \n {chunk['summary']} \n </document> \n"  # show model all 3 items
    return results, context
    
def answer_query_level_two(query, db):
    documents, context = retrieve_base(query, db)
    prompt = f"""
    You have been tasked with helping us to answer the following query:
    <query>
    {query}
    </query>
    You have access to the following documents which are meant to provide context as you answer the query:
    <documents>
    {context}
    </documents>
    Please remain faithful to the underlying context, and only deviate from it if you are 100% sure that you know the answer already.
    Answer the question now, and avoid providing preamble such as 'Here is the answer', etc
    """
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.content[0].text

```


### 2. Re-ranking

### idea
- claude is provided with {query} + {documents}, and be asked for selecting and ranking most relevant documents
### plan
 - retrieve relevants chunks through vector database
 - re-rank the set by asking claude to re-rank the set and get the top 3
 - parse the response and pass to LLM to generate final anwser

```python
def rerank_results(query: str, results: list[dict], k: int = 5) -> list[dict]:
    # Prepare the summaries with their indices
    summaries = []
    print(len(results))

    for i, result in enumerate(results):
        summary = f"[{i}] Document Summary: {result['metadata']['summary']}"
        summaries.append(summary)
    joined_summaries = "\n\n".join(summaries)

    prompt = f"""
    Query: {query}
    You are about to be given a group of documents, each preceded by its index number in square brackets. Your task is to select the only {k} most relevant documents from the list to help us answer the query.

    <documents>
    {joined_summaries}
    </documents>

    Output only the indices of {k} most relevant documents in order of relevance, separated by commas, enclosed in XML tags here:
    <relevant_indices>put the numbers of your indices here, seeparted by commas</relevant_indices>
    """
    
```



# Advance with contextual embedding:
[claude-cookbooks/capabilities/contextual-embeddings/guide.ipynb at main · anthropics/claude-cookbooks](https://github.com/anthropics/claude-cookbooks/blob/main/capabilities/contextual-embeddings/guide.ipynb)

## Contextual Embeddings
With basic RAG, individual chunks often lack sufficient context when embedded in isolation. Contextual Embeddings solve this by using Claude to generate a brief description that "situates" each chunk within its source document. We then embed the chunk together with this context, creating richer vector representations.

For each chunk in our codebase dataset, we pass both the chunk and its full source file to Claude. Claude generates a concise explanation of what the chunk contains and where it fits in the overall file. This context gets prepended to the chunk before embedding.

### Cost and Latency Considerations

**When does this cost occur?** The contextualization happens once at ingestion time, not during every query. Unlike techniques like HyDE (hypothetical document embeddings) that add latency to each search, contextual embeddings are a one-time cost when building your vector database. Prompt caching makes this practical. Since we process all chunks from the same document sequentially, we can leverage prompt caching for significant savings.

1. First chunk: We write the full document to cache (pay a small premium)
2. Subsequent chunks: Read the document from cache (90% discount on those tokens)
3. Cache lasts 5 minutes, plenty of time to process all chunks in a document

**Cost example**: For 800-token chunks in 8k-token documents with 100 tokens of generated context, the total cost is $1.02 per million document tokens. You'll see the cache savings in the logs when you run the code below.

**Note:** Some embedding models have fixed input token limits. If you see worse performance with contextual embeddings, your contextualized chunks may be getting truncated—consider using an embedding model with a larger context window.

```python
DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>
"""

CHUNK_CONTEXT_PROMPT = """
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""

def situate_context(doc: str, chunk: str) -> str:
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1024,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc),
                        "cache_control": {
                            "type": "ephemeral"
                        },  # we will make use of prompt caching for the full documents
                    },
                    {
                        "type": "text",
                        "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk),
                    },
                ],
            }
        ],
    )
    return response
```

## Contextual BM25: Hybrid Search

Contextual embeddings alone improved our Pass@10 from 87% to 92%. We can push performance even higher by combining semantic search with keyword-based search using **Contextual BM25**—a hybrid approach that reduces retrieval failure rates further.

### Why Hybrid Search?

Semantic search excels at understanding meaning and context, but can miss exact keyword matches. BM25 (a probabilistic keyword ranking algorithm) excels at finding specific terms, but lacks semantic understanding. By combining both, we get the best of both worlds:

- **Semantic search**: Captures conceptual similarity and paraphrases
- **BM25**: Catches exact terminology, function names, and specific phrases
- **Reciprocal Rank Fusion**: Intelligently merges results from both sources

### What is BM25?

BM25 is a probabilistic ranking function that improves upon TF-IDF by accounting for document length and term saturation. It's widely used in production search engines (including Elasticsearch) for its effectiveness at ranking keyword relevance. For technical details, see [this blog post](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables).

Instead of only searching the raw chunk content, we search both the chunk _and_ the contextual description we generated earlier. This means BM25 can match keywords in either the original text or the explanatory context.


## How the Hybrid Search Works

The retrieve_advanced function below implements a three-step process:

1. Retrieve candidates: Get top 150 results from both semantic search and BM25
2. Score fusion: Combine rankings using weighted Reciprocal Rank Fusion
    - Default: 80% weight to semantic search, 20% to BM25
    - These weights are tunable—experiment to optimize for your use case
3. Return top-k: Select the highest-scoring results after fusion

The weighting system lets you balance between semantic understanding and keyword precision based on your data characteristics.


## Reranking

We've achieved strong results with hybrid search (93.21% Pass@10), but there's one more technique that can squeeze out additional performance: **reranking**.

### What is Reranking?

Reranking is a two-stage retrieval approach:

1. **Stage 1 - Broad Retrieval**: Cast a wide net by retrieving more candidates than you need (e.g., retrieve 100 chunks)
2. **Stage 2 - Precise Selection**: Use a specialized reranking model to score these candidates and select only the top-k most relevant ones

**Why does this work?** Initial retrieval methods (embeddings, BM25) are optimized for speed across millions of documents. Reranking models are slower but more accurate—they can afford to do deeper analysis on a smaller candidate set. This creates a speed/accuracy trade-off that works well in practice.

### Our Reranking Approach

For this example, we'll use a simpler reranking pipeline that builds on contextual embeddings alone (not the full hybrid search). Here's the process:

1. **Over-retrieve**: Get 10x more results than needed (e.g., retrieve 100 chunks when we need 10)``````
2. **Rerank with Cohere**: Use Cohere's `rerank-english-v3.0` model to score all candidates
3. **Select top-k**: Return only the highest-scoring results

The reranking model has access to both the original chunk content and the contextual descriptions we generated, giving it rich information to make precise relevance judgments.

### Expected Performance

Adding reranking delivers a modest but meaningful improvement:

- **Without reranking**: 92.34% Pass@10 (contextual embeddings alone)
- **With reranking**: ~95% Pass@10 (additional 2-3% gain)

This might seem small, but in production systems, reducing failures from 7.66% to ~5% can significantly improve user experience. The trade-off is query latency—reranking adds ~100-200ms per query depending on candidate set size.


**Key Takeaways:**

1. **Contextual embeddings provided the largest single improvement** (+5-7 percentage points), validating that adding document-level context to chunks significantly improves retrieval quality. This technique alone gets you 90% of the way to optimal performance.
    
2. **Reranking achieves the highest absolute performance**, reaching 95.26% Pass@10—meaning the correct chunk appears in the top 10 results for 95% of queries. This represents a **47% reduction in retrieval failures** compared to baseline RAG (from 12.85% failure rate down to 4.74%).
    
3. **Trade-offs matter**: Each technique adds complexity and cost:
    
    - Contextual embeddings: One-time ingestion cost (~$3 for this dataset with prompt caching)
    - Hybrid search: Requires Elasticsearch infrastructure and maintenance
    - Reranking: Adds 100-200ms query latency and per-query API costs (~$0.002 per query)
4. **Choose your approach** based on your requirements:
    
    - **High-volume, cost-sensitive**: Contextual embeddings alone (92% Pass@10, no per-query costs)
    - **Maximum accuracy, latency-tolerant**: Full reranking pipeline (95% Pass@10, best precision)
    - **Balanced production system**: Hybrid search for strong performance without per-query costs (93% Pass@10)

For most production RAG systems, **contextual embeddings provide the best performance-to-cost ratio**, delivering 92% Pass@10 with only one-time ingestion costs. Hybrid search and reranking are available when you need that extra 2-3 percentage points of precision and can afford the additional infrastructure or query costs.