"""
Module: parser.py
Description: Robust PDF ingestion engine using PyMuPDF (fitz) for enterprise
             financial document parsing, table extraction, and page-level metadata tracking.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymupdf as fitz
from pydantic import BaseModel, Field

# Configure module-level logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class DocumentChunk(BaseModel):
    """
    Represents a granular chunk of text or tabular data extracted from a document,
    carrying rich metadata for precise source citation.
    """
    chunk_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the chunk"
    )
    content: str = Field(
        ...,
        description="The textual or tabular content of the chunk"
    )
    page_number: int = Field(
        ...,
        ge=1,
        description="1-indexed page number from which the chunk originated"
    )
    source_file: str = Field(
        ...,
        description="File name or relative path of the source document"
    )
    chunk_index: int = Field(
        default=0,
        ge=0,
        description="Sequential index of the chunk within the document"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional custom metadata (e.g. table indicator, bounding boxes, section headers)"
    )

    def to_citation_header(self) -> str:
        """Returns a standardized citation label for prompt context attribution."""
        return f"[Source: {self.source_file}, Page: {self.page_number}]"


class PDFParser:
    """
    Enterprise-grade PDF ingestion engine that parses text and structured tables,
    preserving page numbering and metadata for rigorous downstream citation.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        extract_tables: bool = True
    ) -> None:
        """
        Initialize the PDF parser.

        Args:
            chunk_size: Maximum character length for a single text chunk.
            chunk_overlap: Number of characters to overlap between sequential chunks.
            extract_tables: Whether to detect and extract tabular structures into markdown.
        """
        if chunk_size <= chunk_overlap:
            raise ValueError("chunk_size must be strictly greater than chunk_overlap.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extract_tables = extract_tables

    def parse_pdf(self, file_path: str | Path) -> List[DocumentChunk]:
        """
        Parse a PDF file, extracting text and tables page by page with full metadata tracking.

        Args:
            file_path: Absolute or relative path to the target PDF file.

        Returns:
            List of DocumentChunk instances with content and page metadata.

        Raises:
            FileNotFoundError: If the specified PDF file does not exist.
            ValueError: If the file is corrupted or not a valid PDF.
        """
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"PDF file not found at path: {file_path}")

        source_name = path_obj.name
        chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        logger.info(f"Opening PDF document: {source_name}")

        try:
            doc = fitz.open(str(path_obj))
        except Exception as e:
            logger.error(f"Failed to open PDF '{file_path}': {e}", exc_info=True)
            raise ValueError(f"Unable to parse document '{file_path}'. Error: {e}") from e

        try:
            if doc.is_encrypted:
                raise ValueError(f"PDF '{source_name}' is password protected and cannot be ingested.")

            total_pages = doc.page_count
            logger.info(f"Processing '{source_name}' ({total_pages} total pages)...")

            for page_idx in range(total_pages):
                page_num = page_idx + 1  # 1-indexed
                page = doc.load_page(page_idx)

                # 1. Extract structured tables if enabled
                table_texts: List[str] = []
                if self.extract_tables:
                    try:
                        table_texts = self._extract_tables_from_page(page)
                    except Exception as ex:
                        logger.warning(f"Table extraction warning on page {page_num}: {ex}")

                # 2. Extract standard text blocks
                raw_text = page.get_text("text").strip()

                # Combine text and table content
                combined_page_content_parts = []
                if raw_text:
                    combined_page_content_parts.append(raw_text)
                if table_texts:
                    combined_page_content_parts.append("\n\n--- Extracted Tables ---\n" + "\n\n".join(table_texts))

                page_content = "\n\n".join(combined_page_content_parts).strip()

                if not page_content:
                    logger.warning(f"Page {page_num} of '{source_name}' contains no readable text or tables.")
                    continue

                # 3. Chunk page content while preserving page attribution
                page_chunks = self._chunk_text(
                    text=page_content,
                    page_number=page_num,
                    source_file=source_name,
                    start_chunk_idx=global_chunk_idx
                )

                for chunk in page_chunks:
                    chunks.append(chunk)
                    global_chunk_idx += 1

            logger.info(f"Successfully extracted {len(chunks)} chunks from '{source_name}'.")
            return chunks

        finally:
            doc.close()

    def _extract_tables_from_page(self, page: fitz.Page) -> List[str]:
        """
        Detects tables on a given PyMuPDF page and renders them into Markdown tables.
        """
        markdown_tables: List[str] = []
        try:
            # PyMuPDF built-in table finder (fitz.Page.find_tables())
            tabs = page.find_tables()
            for tab in tabs:
                df_data = tab.extract()
                if not df_data or len(df_data) < 2:
                    continue

                # Construct markdown table representation
                header = [str(col).strip() if col is not None else "" for col in df_data[0]]
                separator = ["---"] * len(header)
                rows = []
                for row in df_data[1:]:
                    formatted_row = [str(cell).strip().replace("\n", " ") if cell is not None else "" for cell in row]
                    rows.append("| " + " | ".join(formatted_row) + " |")

                table_md = "| " + " | ".join(header) + " |\n| " + " | ".join(separator) + " |\n" + "\n".join(rows)
                markdown_tables.append(table_md)
        except Exception as e:
            logger.debug(f"find_tables() not available or encountered non-critical error: {e}")

        return markdown_tables

    def _chunk_text(
        self,
        text: str,
        page_number: int,
        source_file: str,
        start_chunk_idx: int
    ) -> List[DocumentChunk]:
        """
        Splits text into overlapping chunks without crossing page boundaries,
        attaching exact source and page metadata.
        """
        chunks: List[DocumentChunk] = []
        text_len = len(text)

        if text_len <= self.chunk_size:
            return [
                DocumentChunk(
                    content=text,
                    page_number=page_number,
                    source_file=source_file,
                    chunk_index=start_chunk_idx,
                    metadata={"character_count": text_len}
                )
            ]

        start = 0
        current_idx = start_chunk_idx

        while start < text_len:
            end = start + self.chunk_size

            if end < text_len:
                # Find a natural paragraph or sentence breakpoint to avoid cutting words
                split_point = text.rfind("\n\n", start, end)
                if split_point == -1:
                    split_point = text.rfind(". ", start, end)
                if split_point == -1:
                    split_point = text.rfind("\n", start, end)
                if split_point == -1:
                    split_point = text.rfind(" ", start, end)

                if split_point > start + (self.chunk_size // 2):
                    end = split_point + 1

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        content=chunk_text,
                        page_number=page_number,
                        source_file=source_file,
                        chunk_index=current_idx,
                        metadata={
                            "character_count": len(chunk_text),
                            "start_char": start,
                            "end_char": end
                        }
                    )
                )
                current_idx += 1

            # Advance with overlap
            start = end - self.chunk_overlap
            if start >= text_len or end >= text_len:
                break

        return chunks
