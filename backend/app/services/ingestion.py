import logging
import os
from typing import List, Tuple
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from sqlalchemy.orm import Session

from ..config import settings
from ..models import DocumentChunk, DocumentParentChunk

logger = logging.getLogger(__name__)


def parse_file(file_path: str, file_name: str) -> str:
    ext = os.path.splitext(file_name)[1].lower()
    if ext == ".pdf":
        doc = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
    elif ext == ".docx":
        doc = DocxDocument(file_path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    else:
        text = f"[image: {file_name}]"
    logger.info("parse_file: file=%s type=%s text_len=%d", file_name, ext or "image", len(text))
    return text


def _parent_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=1500,
        chunk_overlap=0,
    )


def _child_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=300,
        chunk_overlap=50,
    )


def chunk_text(text: str) -> Tuple[List[str], List[List[str]]]:
    """Returns (parents, children_per_parent). children_per_parent[i] are the child
    chunks derived from parents[i]."""
    parents = _parent_splitter().split_text(text)
    child_splitter = _child_splitter()
    children_per_parent = [child_splitter.split_text(p) for p in parents]
    n_children = sum(len(c) for c in children_per_parent)
    logger.info(
        "chunk_text: input_len=%d parents=%d children=%d",
        len(text), len(parents), n_children,
    )
    return parents, children_per_parent


def _openai_client() -> OpenAI:
    # Embeddings are routed through OpenRouter's OpenAI-compatible /v1/embeddings.
    return OpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )


def embed_text(text: str) -> List[float]:
    client = _openai_client()
    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=[text],
        dimensions=settings.embedding_dim,
    )
    return response.data[0].embedding


def embed_chunks(chunks: List[str]) -> List[List[float]]:
    client = _openai_client()
    embeddings: List[List[float]] = []
    batch_size = 100
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    logger.info("embed_chunks: total=%d batches=%d", len(chunks), total_batches)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = i // batch_size + 1
        logger.debug("embed_chunks: batch=%d/%d size=%d", batch_num, total_batches, len(batch))
        response = client.embeddings.create(
            model=settings.openai_embedding_model,
            input=batch,
            dimensions=settings.embedding_dim,
        )
        embeddings.extend(item.embedding for item in response.data)
    logger.info("embed_chunks: done embeddings=%d", len(embeddings))
    return embeddings


def store_chunks(
    db: Session,
    document_id: str,
    parents: List[str],
    children_per_parent: List[List[str]],
    child_embeddings: List[List[float]],
) -> None:
    """Insert parents then children. Children carry parent_id."""
    parent_rows = [
        DocumentParentChunk(
            document_id=document_id,
            parent_index=i,
            content=parent_text,
        )
        for i, parent_text in enumerate(parents)
    ]
    db.add_all(parent_rows)
    db.flush()  # populate parent_rows[*].id

    child_rows: List[DocumentChunk] = []
    embed_iter = iter(child_embeddings)
    global_idx = 0
    for parent_row, children in zip(parent_rows, children_per_parent):
        for child_text in children:
            child_rows.append(DocumentChunk(
                document_id=document_id,
                parent_id=parent_row.id,
                chunk_index=global_idx,
                content=child_text,
                embedding=next(embed_iter),
            ))
            global_idx += 1
    db.bulk_save_objects(child_rows)
    db.commit()
    logger.info(
        "store_chunks: parents=%d children=%d document_id=%s",
        len(parent_rows), len(child_rows), document_id,
    )
