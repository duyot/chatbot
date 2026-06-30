import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from sqlalchemy.orm import Session

from ..config import settings
from ..models import DocumentChunk, DocumentParentChunk
from .ocr_client import ocr_image, OCRError

logger = logging.getLogger(__name__)


# --- Parsed representation --------------------------------------------------

@dataclass
class PageContent:
    """One logical page of a source document."""
    page: int                          # 1-based
    text: str
    source: str                        # "native" | "ocr"
    ocr_confidence: Optional[float] = None


@dataclass
class ParsedDocument:
    pages: List[PageContent]
    metadata: dict

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.pages if p.text)


@dataclass
class ParentChunk:
    content: str
    page_start: int
    page_end: int
    source: str


@dataclass
class ChildChunk:
    content: str
    page: int
    source: str
    ocr_confidence: Optional[float] = None


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
_MIME_BY_EXT = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


# --- Parsing ----------------------------------------------------------------

def _ocr_page_image(image_bytes: bytes, filename: str) -> Tuple[str, Optional[float]]:
    """OCR a rendered page/image, degrading gracefully. Returns ("", None) when
    OCR is disabled or the service errors, so one bad page doesn't fail the
    whole document (the doc-level guard in the worker handles a fully-empty doc)."""
    if not settings.ocr_enabled:
        return "", None
    try:
        return ocr_image(image_bytes, filename=filename)
    except OCRError:
        logger.warning("_ocr_page_image: OCR failed, treating page as empty file=%s", filename)
        return "", None


def _parse_pdf(file_path: str, file_name: str) -> ParsedDocument:
    """Hybrid: use the native text layer when present; OCR scanned pages."""
    doc = fitz.open(file_path)
    pages: List[PageContent] = []
    ocr_pages = 0
    native_pages = 0
    for i, page in enumerate(doc):
        native_text = page.get_text() or ""
        if len(native_text.strip()) >= settings.ocr_min_text_chars:
            pages.append(PageContent(page=i + 1, text=native_text, source="native"))
            native_pages += 1
        else:
            pix = page.get_pixmap(dpi=settings.ocr_dpi)
            img_bytes = pix.tobytes("png")
            text, conf = _ocr_page_image(img_bytes, f"{file_name}#p{i + 1}.png")
            pages.append(PageContent(page=i + 1, text=text, source="ocr", ocr_confidence=conf))
            ocr_pages += 1
    metadata = {
        "page_count": len(pages),
        "ocr_pages": ocr_pages,
        "native_pages": native_pages,
        "ocr_engine": "paddleocr" if ocr_pages else None,
    }
    return ParsedDocument(pages=pages, metadata=metadata)


def _parse_docx(file_path: str) -> ParsedDocument:
    doc = DocxDocument(file_path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return ParsedDocument(
        pages=[PageContent(page=1, text=text, source="native")],
        metadata={"page_count": 1, "ocr_pages": 0, "native_pages": 1, "ocr_engine": None},
    )


def _parse_image(file_path: str, file_name: str) -> ParsedDocument:
    with open(file_path, "rb") as f:
        img_bytes = f.read()
    text, conf = _ocr_page_image(img_bytes, file_name)
    return ParsedDocument(
        pages=[PageContent(page=1, text=text, source="ocr", ocr_confidence=conf)],
        metadata={"page_count": 1, "ocr_pages": 1, "native_pages": 0, "ocr_engine": "paddleocr"},
    )


def parse_document(file_path: str, file_name: str) -> ParsedDocument:
    """Parse a source file into per-page text + metadata.

    PDFs are hybrid (native text layer, OCR fallback per page); images are
    always OCR'd; DOCX is treated as a single native page.
    """
    ext = os.path.splitext(file_name)[1].lower()
    if ext == ".pdf":
        parsed = _parse_pdf(file_path, file_name)
    elif ext == ".docx":
        parsed = _parse_docx(file_path)
    elif ext in _IMAGE_EXTS:
        parsed = _parse_image(file_path, file_name)
    else:
        parsed = ParsedDocument(pages=[], metadata={"page_count": 0})
    parsed.metadata["mime_type"] = _MIME_BY_EXT.get(ext)
    total_len = sum(len(p.text) for p in parsed.pages)
    logger.info(
        "parse_document: file=%s type=%s pages=%d ocr_pages=%s text_len=%d",
        file_name, ext or "?", len(parsed.pages),
        parsed.metadata.get("ocr_pages"), total_len,
    )
    return parsed


# --- Chunking ---------------------------------------------------------------

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
    """Low-level splitter for a single text block. Returns (parents,
    children_per_parent). children_per_parent[i] are the child chunks derived
    from parents[i]."""
    parents = _parent_splitter().split_text(text)
    child_splitter = _child_splitter()
    children_per_parent = [child_splitter.split_text(p) for p in parents]
    n_children = sum(len(c) for c in children_per_parent)
    logger.info(
        "chunk_text: input_len=%d parents=%d children=%d",
        len(text), len(parents), n_children,
    )
    return parents, children_per_parent


def chunk_document(parsed: ParsedDocument) -> Tuple[List[ParentChunk], List[List[ChildChunk]]]:
    """Chunk page-by-page so every parent maps to exactly one page, giving exact
    page attribution. Children inherit their parent's page/source/confidence."""
    parents: List[ParentChunk] = []
    children_per_parent: List[List[ChildChunk]] = []
    for page in parsed.pages:
        if not page.text.strip():
            continue
        p_texts, c_per_p = chunk_text(page.text)
        for p_text, c_texts in zip(p_texts, c_per_p):
            parents.append(ParentChunk(
                content=p_text,
                page_start=page.page,
                page_end=page.page,
                source=page.source,
            ))
            children_per_parent.append([
                ChildChunk(
                    content=c,
                    page=page.page,
                    source=page.source,
                    ocr_confidence=page.ocr_confidence,
                )
                for c in c_texts
            ])
    n_children = sum(len(c) for c in children_per_parent)
    logger.info(
        "chunk_document: pages=%d parents=%d children=%d",
        len(parsed.pages), len(parents), n_children,
    )
    return parents, children_per_parent


# --- Embeddings -------------------------------------------------------------

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


# --- Persistence ------------------------------------------------------------

def store_chunks(
    db: Session,
    document_id: str,
    parents: List[ParentChunk],
    children_per_parent: List[List[ChildChunk]],
    child_embeddings: List[List[float]],
) -> None:
    """Insert parents then children. Children carry parent_id and page metadata.
    child_embeddings must be in the same flattened order as children_per_parent."""
    parent_rows = [
        DocumentParentChunk(
            document_id=document_id,
            parent_index=i,
            content=p.content,
            page_start=p.page_start,
            page_end=p.page_end,
            source=p.source,
        )
        for i, p in enumerate(parents)
    ]
    db.add_all(parent_rows)
    db.flush()  # populate parent_rows[*].id

    child_rows: List[DocumentChunk] = []
    embed_iter = iter(child_embeddings)
    global_idx = 0
    for parent_row, children in zip(parent_rows, children_per_parent):
        for child in children:
            child_rows.append(DocumentChunk(
                document_id=document_id,
                parent_id=parent_row.id,
                chunk_index=global_idx,
                content=child.content,
                embedding=next(embed_iter),
                page=child.page,
                source=child.source,
                ocr_confidence=child.ocr_confidence,
            ))
            global_idx += 1
    db.bulk_save_objects(child_rows)
    db.commit()
    logger.info(
        "store_chunks: parents=%d children=%d document_id=%s",
        len(parent_rows), len(child_rows), document_id,
    )
