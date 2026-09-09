"""
Unit and Integration Tests for Multimodal Enterprise RAG Pipeline.
"""

from pathlib import Path
import unittest

from src.parser import PDFParser, DocumentChunk
from src.vector_store import VectorStoreManager
from src.generator import RAGGenerator
from src.evaluator import CriticEvaluator, EvaluationResult
from data.create_sample_pdf import generate_sample_financial_report


class TestEnterpriseRAGPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_pdf_path = Path("data/sample_report.pdf")
        if not cls.test_pdf_path.exists():
            cls.test_pdf_path = generate_sample_financial_report(str(cls.test_pdf_path))

    def test_01_pdf_parser(self):
        """Verify PDF parser extracts text, tables, and retains 1-indexed page numbers."""
        parser = PDFParser(chunk_size=800, chunk_overlap=100)
        chunks = parser.parse_pdf(self.test_pdf_path)
        
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertIsInstance(chunk, DocumentChunk)
            self.assertGreaterEqual(chunk.page_number, 1)
            self.assertLessEqual(chunk.page_number, 4)
            self.assertEqual(chunk.source_file, "sample_report.pdf")
            self.assertTrue(len(chunk.content.strip()) > 0)
            self.assertIn("Page:", chunk.to_citation_header())

    def test_02_vector_store_in_memory(self):
        """Verify Qdrant in-memory indexing and semantic retrieval."""
        vector_mgr = VectorStoreManager(
            collection_name="test_reports",
            mock_mode=True
        )
        sample_chunks = [
            DocumentChunk(
                content="Apex Global reported Q3 revenue of $4.82 billion.",
                page_number=4,
                source_file="sample_report.pdf",
                chunk_index=0
            ),
            DocumentChunk(
                content="Operating expenses were $2.195 billion for Q3.",
                page_number=3,
                source_file="sample_report.pdf",
                chunk_index=1
            )
        ]
        indexed = vector_mgr.index_chunks(sample_chunks)
        self.assertEqual(indexed, 2)

        results = vector_mgr.search("What was the revenue?", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].source_file, "sample_report.pdf")
        self.assertIn(results[0].page_number, [3, 4])

    def test_03_generator_citations(self):
        """Verify answer synthesis and citation extraction logic."""
        generator = RAGGenerator(mock_mode=True)
        context = [
            DocumentChunk(
                content="According to Page 4, Q3 Net Revenue was $4.82B.",
                page_number=4,
                source_file="sample_report.pdf",
                chunk_index=0
            )
        ]
        response = generator.generate_answer("What was the Q3 revenue?", context)
        self.assertIn("4.82", response.answer)
        self.assertTrue(len(response.citations) > 0)
        self.assertEqual(response.citations[0]["page_number"], 4)

    def test_04_evaluator_critic(self):
        """Verify Critic scoring and flag rules."""
        critic = CriticEvaluator(mock_mode=True)
        context = [
            DocumentChunk(
                content="Net revenue was $4.82 billion on page 4.",
                page_number=4,
                source_file="sample_report.pdf",
                chunk_index=0
            )
        ]
        result = critic.evaluate(
            query="What was the net revenue?",
            context_chunks=context,
            generated_answer="Net revenue was $4.82 billion [Source: sample_report.pdf, Page: 4]."
        )
        self.assertIsInstance(result, EvaluationResult)
        self.assertGreaterEqual(result.faithfulness_score, 4)
        self.assertGreaterEqual(result.relevancy_score, 4)
        self.assertFalse(result.is_flagged)
        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()
