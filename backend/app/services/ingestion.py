import os
from typing import List
import fitz  # PyMuPDF
import httpx
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session

from ..config import settings
from ..models import DocumentChunk


def parse_file(file_path: str, file_name: str) -> str:
    ext = os.path.splitext(file_name)[1].lower()
    if ext == ".pdf":
        doc = fitz.open(file_path)
        return "\n".join(page.get_text() for page in doc)
    elif ext == ".docx":
        doc = DocxDocument(file_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    else:
        return f"[image: {file_name}]"


def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)


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
    with httpx.Client() as client:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            response = client.post(
                f"{settings.ollama_base_url}/api/embed",
                json={"model": settings.ollama_embedding_model, "input": batch},
            )
            response.raise_for_status()
            embeddings.extend(response.json()["embeddings"])
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
