"""
Unit and Integration Tests for Neural Cross-Reranker and Maximal Marginal Relevance (MMR) Filter.
"""

import unittest
from src.parser import DocumentChunk
from src.reranker import (
    DeterministicCrossScorer,
    MaximalMarginalRelevanceFilter,
    NeuralCrossReranker,
)


class TestNeuralCrossRerankerAndMMR(unittest.TestCase):
    def setUp(self):
        self.chunk_revenue = DocumentChunk(
            content="Apex Global reported Q3 net revenue of $4.82 billion, up 14.2% YoY.",
            page_number=4,
            source_file="sample_report.pdf",
            chunk_index=0
        )
        self.chunk_revenue_duplicate = DocumentChunk(
            content="Apex Global reported Q3 net revenue of $4.82 billion, representing 14.2% YoY growth.",
            page_number=1,
            source_file="sample_report.pdf",
            chunk_index=1
        )
        self.chunk_expenses = DocumentChunk(
            content="Operating expenses were $2.195 billion for Q3, primarily driven by AI R&D and data centers.",
            page_number=3,
            source_file="sample_report.pdf",
            chunk_index=2
        )
        self.chunk_balance_sheet = DocumentChunk(
            content="Cash and cash equivalents totaled $1.85 billion on the consolidated balance sheet.",
            page_number=2,
            source_file="sample_report.pdf",
            chunk_index=3
        )
        self.chunk_irrelevant = DocumentChunk(
            content="The legal governance committee met on Tuesday to review corporate bylaws and compliance filings.",
            page_number=2,
            source_file="sample_report.pdf",
            chunk_index=4
        )

    def test_deterministic_cross_scorer_relevance(self):
        """Verify cross scorer scores relevant passages significantly higher than irrelevant ones."""
        scorer = DeterministicCrossScorer()
        query = "What was the Q3 net revenue?"

        score_rev = scorer.score(query, self.chunk_revenue.content)
        score_exp = scorer.score(query, self.chunk_expenses.content)
        score_irr = scorer.score(query, self.chunk_irrelevant.content)

        # Revenue chunk has exact match for 'q3', 'net', 'revenue'
        self.assertGreater(score_rev, score_irr)
        self.assertGreater(score_rev, score_exp)
        self.assertGreaterEqual(score_rev, 0.0)
        self.assertLessEqual(score_rev, 1.0)

    def test_numeric_and_currency_bonus(self):
        """Verify presence of numbers and financial symbols enhances scoring precision."""
        scorer = DeterministicCrossScorer()
        query = "revenue $4.82 billion"

        score_with_num = scorer.score(query, self.chunk_revenue.content)
        score_without_num = scorer.score(query, "Apex Global reported general revenue increases.")

        self.assertGreater(score_with_num, score_without_num)

    def test_mmr_filter_suppresses_redundancy(self):
        """Verify MMR penalizes near-duplicate passages and promotes diverse information."""
        # Using lambda=0.5 to strongly reward diversity
        mmr = MaximalMarginalRelevanceFilter(diversity_lambda=0.50)

        # candidate 1 and candidate 2 have almost identical wording about revenue
        # candidate 3 is about operating expenses
        candidates = [
            (self.chunk_revenue, 0.95),
            (self.chunk_revenue_duplicate, 0.93),
            (self.chunk_expenses, 0.80),
        ]

        results = mmr.filter(candidates, top_k=2)
        self.assertEqual(len(results), 2)

        selected_chunks = [r[0] for r in results]

        # The first must be chunk_revenue (highest initial score)
        self.assertEqual(selected_chunks[0].chunk_id, self.chunk_revenue.chunk_id)

        # The second chunk selected should be chunk_expenses (novel information), NOT the duplicate!
        self.assertEqual(
            selected_chunks[1].chunk_id,
            self.chunk_expenses.chunk_id,
            "MMR failed to suppress near-duplicate chunk in favor of diverse content."
        )

        # Check penalty recorded on duplicate vs selected
        penalty_selected = results[1][2]
        self.assertGreaterEqual(penalty_selected, 0.0)

    def test_neural_cross_reranker_end_to_end(self):
        """Verify end-to-end NeuralCrossReranker executes and enriches DocumentChunk metadata."""
        reranker = NeuralCrossReranker(diversity_lambda=0.70, mock_mode=True)
        query = "Q3 net revenue and financial performance"

        candidates = [
            self.chunk_irrelevant,
            self.chunk_expenses,
            self.chunk_revenue,
            self.chunk_balance_sheet
        ]

        top_k = 2
        reranked = reranker.rerank(query=query, documents=candidates, top_k=top_k)

        self.assertEqual(len(reranked), top_k)

        # Top result should be the revenue chunk
        self.assertEqual(reranked[0].chunk_id, self.chunk_revenue.chunk_id)

        # Verify all metadata attributes exist and are populated
        for rank, chunk in enumerate(reranked, 1):
            self.assertIn("rerank_score", chunk.metadata)
            self.assertIn("mmr_score", chunk.metadata)
            self.assertIn("redundancy_penalty", chunk.metadata)
            self.assertIn("rerank_latency_ms", chunk.metadata)
            self.assertEqual(chunk.metadata["rerank_rank"], rank)
            self.assertEqual(chunk.metadata["rerank_method"], "Neural-CrossEncoder+MMR")

    def test_edge_cases(self):
        """Verify handling of empty lists and requested top_k exceeding pool size."""
        reranker = NeuralCrossReranker(mock_mode=True)

        # Empty candidate list
        self.assertEqual(reranker.rerank("test query", [], top_k=5), [])

        # top_k larger than candidates
        candidates = [self.chunk_revenue]
        res = reranker.rerank("test query", candidates, top_k=5)
        self.assertEqual(len(res), 1)


if __name__ == "__main__":
    unittest.main()
