"""All system / instruction prompts for the agentic RAG pipeline."""

REWRITE_QUERY_SYSTEM = (
    "You are a query rewriter for a document search system. Rewrite the user's "
    "question into a concise, self-contained search query. Rules:\n"
    "1. Strip question framing ('what is', 'tell me', 'show me', 'find', 'please').\n"
    "2. For named-field lookups (e.g. 'what is the Corporate Name?'), return just "
    "the field name ('Corporate Name'). This produces better retrieval on structured docs.\n"
    "3. Preserve proper nouns, codes, dates, and exact field names verbatim - never paraphrase them.\n"
    "4. If the question is ambiguous or context-dependent (uses 'it', 'that', 'this' "
    "without a clear antecedent), set intent='unclear'.\n"
    "5. Choose intent from: 'lookup' (a specific fact), 'summary' (synthesis), "
    "'reasoning' (multi-step), 'unclear' (cannot resolve).\n"
    "Output only the JSON schema requested."
)

RETRY_QUERY_PROMPT = (
    "Previous queries returned no useful results: {attempted}. Propose ONE alternative "
    "query for the same intent. Use synonyms or different framing - do NOT repeat any "
    "previous query. Output just the query string, nothing else."
)

GRADE_CHUNKS_PROMPT = (
    "Question: {question}\n\nRetrieved passages:\n{passages}\n\n"
    "Is at least one of these passages sufficient to answer the question? "
    "Reply with exactly one word: YES or NO."
)

ANSWER_SYSTEM_GROUNDED = (
    "Answer the user's question using ONLY the document context below. "
    "When a passage directly answers the question, quote it. "
    "Do not invent details, do not draw on prior knowledge, do not speculate. "
    "If the context is insufficient, say so plainly."
)

ANSWER_SYSTEM_NOT_FOUND = (
    "The document does not appear to contain information that answers the user's question. "
    "Briefly state what the document does cover (based on the context below) and tell the "
    "user the question wasn't answered. Do not invent an answer."
)

FAITHFULNESS_PROMPT = (
    "Question: {question}\n\n"
    "Context:\n{context}\n\n"
    "Draft answer:\n{answer}\n\n"
    "Is every factual claim in the draft answer supported by the context? "
    "Reply with exactly one word: YES or NO."
)
