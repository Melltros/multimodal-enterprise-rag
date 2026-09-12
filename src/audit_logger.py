"""
Module: audit_logger.py
Description: Enterprise audit trail and compliance logger for recording query executions,
             retrieval traces (dense, sparse, RRF), synthesis citations, and Critic judge deliberations.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.evaluator import EvaluationResult
from src.parser import DocumentChunk

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class AuditRecord(BaseModel):
    """
    Structured record representing a single end-to-end RAG query trace.
    """
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    query: str
    retrieval_mode: str
    retrieved_chunk_count: int
    retrieved_chunks: List[Dict[str, Any]]
    answer: str
    citations: List[Dict[str, Any]]
    faithfulness_score: int
    relevancy_score: int
    is_flagged: bool
    critic_summary: str
    passed: bool


class EnterpriseAuditLogger:
    """
    Enterprise-grade audit logger that tracks and persists RAG inference sessions
    for financial compliance, quality governance, and hallucination monitoring.
    """

    def __init__(self, log_dir: str | Path = "logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / "audit_trail.jsonl"
        self.csv_path = self.log_dir / "audit_summary.csv"
        self.in_memory_records: List[AuditRecord] = []

    def log_execution(
        self,
        query: str,
        retrieval_mode: str,
        retrieved_chunks: List[DocumentChunk],
        answer: str,
        citations: List[Dict[str, Any]],
        eval_result: EvaluationResult,
        persist: bool = True
    ) -> AuditRecord:
        """
        Records an execution trace and appends it to persistent storage.
        """
        chunk_summaries: List[Dict[str, Any]] = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            chunk_summaries.append({
                "rank": i,
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file,
                "page_number": chunk.page_number,
                "retrieval_method": chunk.metadata.get("retrieval_method", retrieval_mode),
                "rrf_score": chunk.metadata.get("rrf_score"),
                "dense_score": chunk.metadata.get("dense_score"),
                "bm25_score": chunk.metadata.get("bm25_score"),
                "content_preview": chunk.content[:120].strip() + ("..." if len(chunk.content) > 120 else "")
            })

        record = AuditRecord(
            query=query,
            retrieval_mode=retrieval_mode,
            retrieved_chunk_count=len(retrieved_chunks),
            retrieved_chunks=chunk_summaries,
            answer=answer,
            citations=citations,
            faithfulness_score=eval_result.faithfulness_score,
            relevancy_score=eval_result.relevancy_score,
            is_flagged=eval_result.is_flagged,
            critic_summary=eval_result.critique_summary,
            passed=eval_result.passed
        )

        self.in_memory_records.append(record)

        if persist:
            self._append_jsonl(record)
            self._append_csv(record)

        return record

    def _append_jsonl(self, record: AuditRecord) -> None:
        """Appends record to JSON Lines audit file."""
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record.model_dump(), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to append to JSONL audit trail: {e}")

    def _append_csv(self, record: AuditRecord) -> None:
        """Appends summary row to CSV file for tabular executive reporting."""
        file_exists = self.csv_path.exists()
        fieldnames = [
            "record_id",
            "timestamp",
            "query",
            "retrieval_mode",
            "chunk_count",
            "faithfulness_score",
            "relevancy_score",
            "passed",
            "citations_count"
        ]

        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    "record_id": record.record_id,
                    "timestamp": record.timestamp,
                    "query": record.query,
                    "retrieval_mode": record.retrieval_mode,
                    "chunk_count": record.retrieved_chunk_count,
                    "faithfulness_score": record.faithfulness_score,
                    "relevancy_score": record.relevancy_score,
                    "passed": record.passed,
                    "citations_count": len(record.citations)
                })
        except Exception as e:
            logger.error(f"Failed to append to CSV audit summary: {e}")
