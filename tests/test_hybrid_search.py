"""
Unit and Integration Tests for Hybrid Search (BM25 + RRF) and Enterprise Audit Logger.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from src.audit_logger import EnterpriseAuditLogger
from src.evaluator import EvaluationResult
from src.hybrid_search import BM25Okapi, HybridRetriever, ReciprocalRankFusion
from src.parser import DocumentChunk
from src.vector_store import VectorStoreManager


class TestHybridSearchAndAudit(unittest.TestCase):
    def setUp(self):
        self.sample_chunks = [
            DocumentChunk(
                content="Apex Global reported Q3 Net Revenue of $4.82 billion, up 14.2% YoY.",
                page_number=4,
                source_file="sample_report.pdf",
                chunk_index=0
            ),
            DocumentChunk(
                content="Operating expenses rose to $2.195 billion due to AI compute investments.",
                page_number=3,
                source_file="sample_report.pdf",
                chunk_index=1
            ),
            DocumentChunk(
                content="Consolidated Balance Sheet shows Cash and Cash Equivalents of $1.85 billion.",
                page_number=2,
                source_file="sample_report.pdf",
                chunk_index=2
            )
        ]

    def test_bm25_tokenization_and_search(self):
        """Verify BM25 tokenization preserves financial symbols and retrieves exact matches."""
        bm25 = BM25Okapi()
        bm25.index_documents(self.sample_chunks)

        self.assertEqual(bm25.corpus_size, 3)
        self.assertGreater(bm25.avg_doc_len, 0)

        # Exact financial number search
        results = bm25.search("4.82 billion", top_k=2)
        self.assertGreater(len(results), 0)
        top_chunk, score = results[0]
        self.assertEqual(top_chunk.page_number, 4)
        self.assertGreater(score, 0.0)

    def test_reciprocal_rank_fusion(self):
        """Verify RRF mathematical combination and ranking."""
        rrf = ReciprocalRankFusion(k=60)
        c1, c2, c3 = self.sample_chunks

        ranked_lists = {
            "dense": [(c1, 0.95), (c2, 0.85)],
            "bm25": [(c2, 4.2), (c1, 3.8), (c3, 1.1)],
        }

        fused = rrf.fuse(ranked_lists, top_k=3)
        self.assertEqual(len(fused), 3)

        # Both c1 and c2 should have 'hybrid' retrieval_method since they appear in both
        fused_map = {c.chunk_id: c for c in fused}
        self.assertEqual(fused_map[c1.chunk_id].metadata["retrieval_method"], "hybrid")
        self.assertEqual(fused_map[c2.chunk_id].metadata["retrieval_method"], "hybrid")
        self.assertEqual(fused_map[c3.chunk_id].metadata["retrieval_method"], "bm25")

        # Check RRF math: c1 rank in dense=1 (1/61), rank in bm25=2 (1/62) -> sum = 0.016393 + 0.016129 = 0.032522
        expected_c1 = (1.0 / 61) + (1.0 / 62)
        self.assertAlmostEqual(fused_map[c1.chunk_id].metadata["rrf_score"], expected_c1, places=4)

    def test_hybrid_retriever_modes(self):
        """Verify HybridRetriever switches modes between hybrid, dense, and bm25."""
        vector_mgr = VectorStoreManager(collection_name="test_hybrid", mock_mode=True)
        vector_mgr.index_chunks(self.sample_chunks)

        retriever = HybridRetriever(vector_manager=vector_mgr, documents=self.sample_chunks)

        # Test hybrid mode
        hybrid_results = retriever.search("Q3 net revenue", top_k=2, mode="hybrid")
        self.assertEqual(len(hybrid_results), 2)
        self.assertIn(hybrid_results[0].metadata["retrieval_method"], ["hybrid", "dense", "bm25"])

        # Test dense mode
        dense_results = retriever.search("Q3 net revenue", top_k=2, mode="dense")
        self.assertEqual(len(dense_results), 2)
        self.assertEqual(dense_results[0].metadata["retrieval_method"], "dense")

        # Test bm25 mode
        bm25_results = retriever.search("operating expenses", top_k=1, mode="bm25")
        self.assertEqual(len(bm25_results), 1)
        self.assertEqual(bm25_results[0].metadata["retrieval_method"], "bm25")
        self.assertEqual(bm25_results[0].page_number, 3)

    def test_enterprise_audit_logger(self):
        """Verify enterprise audit trail persists execution events to JSONL and CSV."""
        temp_dir = tempfile.mkdtemp()
        try:
            logger = EnterpriseAuditLogger(log_dir=temp_dir)
            mock_eval = EvaluationResult(
                faithfulness_score=5,
                faithfulness_reasoning="All numbers match page 4.",
                relevancy_score=5,
                relevancy_reasoning="Direct answer.",
                is_flagged=False,
                critique_summary="Passed audit check."
            )

            record = logger.log_execution(
                query="What was the Q3 revenue?",
                retrieval_mode="hybrid",
                retrieved_chunks=self.sample_chunks[:1],
                answer="Q3 Revenue was $4.82B [Source: sample_report.pdf, Page: 4].",
                citations=[{"source_file": "sample_report.pdf", "page_number": 4}],
                eval_result=mock_eval,
                persist=True
            )

            self.assertTrue(record.passed)
            self.assertEqual(len(logger.in_memory_records), 1)

            # Check JSONL file exists and is not empty
            jsonl_file = Path(temp_dir) / "audit_trail.jsonl"
            self.assertTrue(jsonl_file.exists())
            self.assertGreater(jsonl_file.stat().st_size, 0)

            # Check CSV file exists and has headers + row
            csv_file = Path(temp_dir) / "audit_summary.csv"
            self.assertTrue(csv_file.exists())
            lines = csv_file.read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(lines), 2)
            self.assertIn("faithfulness_score", lines[0])
            self.assertIn("hybrid", lines[1])

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
