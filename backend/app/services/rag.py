from typing import List
from sqlalchemy.orm import Session
from langchain_core.tools import tool

from ..models import DocumentChunk
from .ingestion import embed_text


def make_search_tool(document_id: str, db: Session, retrieved_chunks: list):
    @tool
    def search_document(query: str) -> str:
        """Search the document for chunks relevant to the query.
        Call with different phrasings if the first results are insufficient."""
        vector = embed_text(query)
        chunks = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.embedding.cosine_distance(vector))
            .limit(5)
            .all()
        )
        retrieved_chunks.extend(chunks)
        return "\n\n---\n\n".join(c.content for c in chunks) or "No relevant content found."

    return search_document
