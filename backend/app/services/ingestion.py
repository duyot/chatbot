import logging
import os
from typing import List
import fitz  # PyMuPDF
import httpx
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session

from ..config import settings
from ..models import DocumentChunk

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


def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    logger.info("chunk_text: input_len=%d chunks=%d", len(text), len(chunks))
    return chunks


def embed_text(text: str) -> List[float]:
    with httpx.Client() as client:
        response = client.post(
            f"{settings.ollama_base_url}/api/embed",
            json={"model": settings.ollama_embedding_model, "input": [text]},
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]


def embed_chunks(chunks: List[str]) -> List[List[float]]:
    embeddings: List[List[float]] = []
    batch_size = 100
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    logger.info("embed_chunks: total=%d batches=%d", len(chunks), total_batches)
    with httpx.Client() as client:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            logger.debug("embed_chunks: batch=%d/%d size=%d", batch_num, total_batches, len(batch))
            response = client.post(
                f"{settings.ollama_base_url}/api/embed",
                json={"model": settings.ollama_embedding_model, "input": batch},
            )
            response.raise_for_status()
            embeddings.extend(response.json()["embeddings"])
    logger.info("embed_chunks: done embeddings=%d", len(embeddings))
    return embeddings


def store_chunks(db: Session, document_id: str, chunks: List[str], embeddings: List[List[float]]) -> None:
    rows = [
        DocumentChunk(
            document_id=document_id,
            chunk_index=i,
            content=chunk,
            embedding=embedding,
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    db.bulk_save_objects(rows)
    db.commit()
    logger.info("store_chunks: inserted=%d document_id=%s", len(rows), document_id)
