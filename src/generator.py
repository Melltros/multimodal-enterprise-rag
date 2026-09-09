"""
Module: generator.py
Description: Synthesis engine with strict citation attribution, grounded context injection,
             and LLM generation using GPT-4o.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field

from src.parser import DocumentChunk

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class GenerationResponse(BaseModel):
    """
    Structured output returned by the synthesis engine containing the final answer,
    extracted citations, and metadata.
    """
    query: str = Field(..., description="The original user query")
    answer: str = Field(..., description="The grounded synthesized answer")
    citations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of cited sources with document name and page number"
    )
    context_chunks_used: List[DocumentChunk] = Field(
        default_factory=list,
        description="The retrieved chunks passed into the prompt context"
    )
    model_name: str = Field(..., description="The LLM model used for synthesis")


class RAGGenerator:
    """
    Synthesis engine that formats retrieved context chunks and enforces strict
    grounding and source citation constraints.
    """

    SYSTEM_PROMPT = (
        "You are an expert enterprise financial research assistant.\n"
        "Your task is to answer the user's question accurately, concisely, and factually based ONLY on the provided context.\n\n"
        "STRICT CONSTRAINTS:\n"
        "1. Answer the user question using ONLY the provided context. Do NOT use any external knowledge or make assumptions.\n"
        "2. If the answer cannot be found in the context, say: 'I cannot find the answer in the provided documents.'\n"
        "3. You MUST append the exact page number citations to your facts in the format: [Source: <filename>, Page: <page_number>].\n"
        "4. Be precise with financial figures, dates, percentages, and table data."
    )

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        mock_mode: bool = False
    ) -> None:
        """
        Initialize the RAG Generator.

        Args:
            openai_api_key: OpenAI API key. If None, reads from OPENAI_API_KEY environment variable.
            model_name: Model identifier (e.g., 'gpt-4o', 'gpt-4o-mini').
            temperature: Sampling temperature (0.0 for deterministic factual responses).
            mock_mode: If True, returns a mock grounded response for testing without API keys.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.mock_mode = mock_mode

        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key and not mock_mode:
            logger.warning("No OPENAI_API_KEY found. Operating in mock generation mode.")
            self.mock_mode = True

        if not self.mock_mode and api_key:
            self.openai_client: Optional[OpenAI] = OpenAI(api_key=api_key)
        else:
            self.openai_client = None

    def format_context(self, context_chunks: List[DocumentChunk]) -> str:
        """
        Formats retrieved DocumentChunk instances into a clean, structured context block
        with clear citation boundary markers.
        """
        if not context_chunks:
            return "No relevant context found."

        formatted_blocks = []
        for i, chunk in enumerate(context_chunks, 1):
            header = f"--- CONTEXT BLOCK {i} [Source: {chunk.source_file}, Page: {chunk.page_number}] ---"
            block = f"{header}\n{chunk.content.strip()}"
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)

    def generate_answer(
        self,
        query: str,
        context_chunks: List[DocumentChunk]
    ) -> GenerationResponse:
        """
        Synthesize an answer using the provided retrieved context chunks.

        Args:
            query: The user's query.
            context_chunks: List of retrieved context chunks from the vector store.

        Returns:
            GenerationResponse with answer, citations, and metadata.
        """
        formatted_context = self.format_context(context_chunks)

        user_prompt = (
            f"CONTEXT DOCUMENTS:\n"
            f"==================\n"
            f"{formatted_context}\n"
            f"==================\n\n"
            f"USER QUESTION: {query}\n\n"
            f"Provide a factual answer grounded strictly in the context above, citing the source and page number for each claim."
        )

        if self.mock_mode or not self.openai_client:
            logger.info("Executing mock response synthesis.")
            return self._generate_mock_response(query, context_chunks)

        logger.info(f"Generating synthesis with model '{self.model_name}'...")
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature
            )

            raw_answer = response.choices[0].message.content or ""
            citations = self._extract_citations(raw_answer, context_chunks)

            return GenerationResponse(
                query=query,
                answer=raw_answer,
                citations=citations,
                context_chunks_used=context_chunks,
                model_name=self.model_name
            )

        except Exception as e:
            logger.error(f"Error during LLM answer synthesis: {e}", exc_info=True)
            raise RuntimeError(f"LLM Generation failed: {e}") from e

    def _extract_citations(self, answer_text: str, chunks: List[DocumentChunk]) -> List[Dict[str, Any]]:
        """
        Parses citation patterns like [Source: filename, Page: X] or correlates with chunk metadata.
        """
        pattern = r"\[Source:\s*([^,\]]+),\s*Page:\s*(\d+)\]"
        matches = re.findall(pattern, answer_text)

        citations = []
        seen = set()

        for source_file, page_str in matches:
            key = (source_file.strip(), int(page_str))
            if key not in seen:
                seen.add(key)
                citations.append({
                    "source_file": source_file.strip(),
                    "page_number": int(page_str)
                })

        # If explicit regex citation was omitted by model, fallback to chunks provided
        if not citations and chunks:
            for chunk in chunks:
                key = (chunk.source_file, chunk.page_number)
                if key not in seen:
                    seen.add(key)
                    citations.append({
                        "source_file": chunk.source_file,
                        "page_number": chunk.page_number
                    })

        return citations

    def _generate_mock_response(
        self,
        query: str,
        context_chunks: List[DocumentChunk]
    ) -> GenerationResponse:
        """
        Produces an intelligent mock response for automated tests / dry-run mode.
        """
        if not context_chunks:
            return GenerationResponse(
                query=query,
                answer="I cannot find the answer in the provided documents.",
                citations=[],
                context_chunks_used=[],
                model_name="mock-generator"
            )

        # Build answer referencing available chunks
        primary_chunk = context_chunks[0]
        answer = (
            f"Based on the financial report, according to Page {primary_chunk.page_number}, "
            f"the company's Q3 net revenue was $4.82 billion (representing a 14.2% YoY growth), "
            f"with operating income of $1.15 billion. "
            f"[Source: {primary_chunk.source_file}, Page: {primary_chunk.page_number}]"
        )

        return GenerationResponse(
            query=query,
            answer=answer,
            citations=[{"source_file": primary_chunk.source_file, "page_number": primary_chunk.page_number}],
            context_chunks_used=context_chunks,
            model_name="mock-generator"
        )
