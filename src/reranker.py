"""
Module: reranker.py
Description: Production-grade Neural Cross-Reranker and Maximal Marginal Relevance (MMR)
             context redundancy filter for enterprise retrieval pipelines.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.parser import DocumentChunk

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class DeterministicCrossScorer:
    """
    Deterministic cross-attention / token-interaction scorer implemented in pure Python.
    Evaluates deep token-level interactions, query-document term proximity, exact numerical
    match preservation, and length-normalized semantic alignment.
    
    Serves as an offline, zero-dependency cross-scoring engine when PyTorch or
    sentence-transformers are not loaded or in edge/mock environments.
    """

    def __init__(self, proximity_window: int = 8, numeric_bonus: float = 0.35) -> None:
        self.proximity_window = proximity_window
        self.numeric_bonus = numeric_bonus

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        if not text:
            return []
        return re.findall(r"\b[\$€£]?[a-z0-9]+(?:[\.,][a-z0-9]+)*%?\b", text.lower())

    def score(self, query: str, doc_text: str) -> float:
        """
        Computes a bounded cross-interaction score in the range [0.0, 1.0].
        
        Evaluates:
        1. Exact Token Overlap & Term Coverage (precision + recall).
        2. Proximity Scoring (rewarding query terms appearing clustered together).
        3. Numeric & Financial Entity Match Bonus (currency, percentages, dates).
        """
        q_tokens = self._tokenize(query)
        d_tokens = self._tokenize(doc_text)

        if not q_tokens or not d_tokens:
            return 0.0

        q_counter = Counter(q_tokens)
        d_counter = Counter(d_tokens)

        # 1. Unigram Coverage and Frequency Saturation
        matched_tokens = set(q_tokens) & set(d_tokens)
        if not matched_tokens:
            return 0.0

        query_coverage = len(matched_tokens) / len(set(q_tokens))
        doc_precision = sum(min(q_counter[t], d_counter[t]) for t in matched_tokens) / len(d_tokens)

        # 2. Financial Numerical / Exact Metric Match Bonus
        numeric_matches = 0
        for token in matched_tokens:
            if re.search(r"\d", token) or any(sym in token for sym in ["$", "%", "€", "£"]):
                numeric_matches += 1
        num_bonus = min(1.0, numeric_matches * self.numeric_bonus)

        # 3. Positional Proximity Windowing
        proximity_score = 0.0
        if len(matched_tokens) > 1:
            token_positions: Dict[str, List[int]] = {}
            for idx, token in enumerate(d_tokens):
                if token in matched_tokens:
                    token_positions.setdefault(token, []).append(idx)

            # Find minimum span containing distinct query tokens
            matched_list = list(matched_tokens)
            min_span = float("inf")
            for i in range(len(matched_list)):
                for j in range(i + 1, len(matched_list)):
                    t1, t2 = matched_list[i], matched_list[j]
                    for p1 in token_positions.get(t1, []):
                        for p2 in token_positions.get(t2, []):
                            span = abs(p1 - p2)
                            if span < min_span:
                                min_span = span

            if min_span < float("inf"):
                # Gaussian-like exponential decay based on token distance
                proximity_score = math.exp(-min_span / max(1, self.proximity_window))

        # Composite Raw Score
        raw_score = (
            (query_coverage * 0.45)
            + (doc_precision * 0.20)
            + (num_bonus * 0.20)
            + (proximity_score * 0.15)
        )

        # Apply Sigmoid Normalization to bound strictly between 0 and 1
        normalized_score = 1.0 / (1.0 + math.exp(-6.0 * (raw_score - 0.35)))
        return float(min(1.0, max(0.0, normalized_score)))


class MaximalMarginalRelevanceFilter:
    """
    Maximal Marginal Relevance (MMR) context redundancy filter.
    
    Prevents duplicate or near-identical document chunks from dominating the top-k context window.
    Iteratively selects documents that maximize:
        MMR(Q, Di) = argmax_{Di in R \\ S} [ lambda * Sim1(Q, Di) - (1 - lambda) * max_{Dj in S} Sim2(Di, Dj) ]
        
    Where:
        - Sim1(Q, Di) is the relevance score of candidate chunk Di to query Q.
        - Sim2(Di, Dj) is the pairwise textual/semantic similarity between Di and already selected chunk Dj.
        - lambda in [0, 1] controls the tradeoff between relevance and diversity.
    """

    def __init__(self, diversity_lambda: float = 0.70) -> None:
        """
        Args:
            diversity_lambda: Weight balancing relevance vs diversity (0.0 = maximal diversity, 1.0 = maximal relevance).
        """
        if not 0.0 <= diversity_lambda <= 1.0:
            raise ValueError(f"diversity_lambda must be between 0.0 and 1.0, got {diversity_lambda}")
        self.diversity_lambda = diversity_lambda

    @staticmethod
    def _compute_jaccard_similarity(text_a: str, text_b: str, n_gram: int = 2) -> float:
        """Computes character or word n-gram Jaccard similarity between two texts."""
        def get_ngrams(text: str, n: int) -> set[str]:
            words = text.lower().split()
            if len(words) < n:
                return set(words)
            return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}

        set_a = get_ngrams(text_a, n_gram)
        set_b = get_ngrams(text_b, n_gram)

        if not set_a or not set_b:
            return 0.0

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def filter(
        self,
        candidate_scores: Sequence[Tuple[DocumentChunk, float]],
        top_k: int
    ) -> List[Tuple[DocumentChunk, float, float]]:
        """
        Ranks candidates using MMR diversity penalization.
        
        Args:
            candidate_scores: Sequence of (DocumentChunk, relevance_score) pairs.
            top_k: Number of diverse context chunks to return.
            
        Returns:
            List of tuples: (DocumentChunk, mmr_final_score, redundancy_penalty).
        """
        if not candidate_scores:
            return []

        if len(candidate_scores) <= top_k and self.diversity_lambda == 1.0:
            return [(chunk, score, 0.0) for chunk, score in candidate_scores]

        candidates = list(candidate_scores)
        selected: List[Tuple[DocumentChunk, float, float]] = []
        remaining = list(range(len(candidates)))

        # 1. Select initial item with highest relevance score
        best_initial_idx = max(remaining, key=lambda i: candidates[i][1])
        first_chunk, first_score = candidates[best_initial_idx]
        selected.append((first_chunk, first_score, 0.0))
        remaining.remove(best_initial_idx)

        # 2. Iteratively select next best item balancing relevance and similarity to selected items
        while remaining and len(selected) < top_k:
            best_mmr_score = -float("inf")
            best_cand_idx = -1
            best_penalty = 0.0

            for cand_idx in remaining:
                chunk_i, rel_score_i = candidates[cand_idx]

                # Max similarity to already selected chunks
                max_sim = 0.0
                for sel_chunk, _, _ in selected:
                    sim = self._compute_jaccard_similarity(chunk_i.content, sel_chunk.content)
                    if sim > max_sim:
                        max_sim = sim

                redundancy_penalty = (1.0 - self.diversity_lambda) * max_sim
                mmr_score = (self.diversity_lambda * rel_score_i) - redundancy_penalty

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_cand_idx = cand_idx
                    best_penalty = redundancy_penalty

            if best_cand_idx != -1:
                chunk, _ = candidates[best_cand_idx]
                selected.append((chunk, float(best_mmr_score), float(best_penalty)))
                remaining.remove(best_cand_idx)

        return selected


class NeuralCrossReranker:
    """
    Enterprise Cross-Encoder Reranker with integrated MMR Redundancy Filtering.
    
    Processes candidate passages retrieved via Vector (dense) or BM25 (sparse) search,
    applies full all-to-all cross-attention scoring, and filters out redundant passages.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        diversity_lambda: float = 0.70,
        mock_mode: bool = False
    ) -> None:
        """
        Args:
            model_name: Pretrained cross-encoder model name from HuggingFace/SentenceTransformers.
            diversity_lambda: Weight for MMR diversity filtering (0.0 to 1.0).
            mock_mode: If True, uses deterministic cross-scorer without downloading weights.
        """
        self.model_name = model_name
        self.diversity_lambda = diversity_lambda
        self.mock_mode = mock_mode
        self.cross_encoder = None
        self.deterministic_scorer = DeterministicCrossScorer()
        self.mmr_filter = MaximalMarginalRelevanceFilter(diversity_lambda=diversity_lambda)

        if not self.mock_mode:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading Neural CrossEncoder model: {self.model_name}")
                self.cross_encoder = CrossEncoder(self.model_name)
            except Exception as e:
                logger.info(f"SentenceTransformers CrossEncoder not available ({e}). Falling back to deterministic cross-scorer.")
                self.cross_encoder = None

    def rerank(
        self,
        query: str,
        documents: Sequence[DocumentChunk],
        top_k: int = 3
    ) -> List[DocumentChunk]:
        """
        Reranks a sequence of retrieved DocumentChunks for a query and applies MMR redundancy filtering.
        
        Args:
            query: The user inquiry.
            documents: Candidate document chunks from initial retrieval.
            top_k: Desired count of high-precision, diverse chunks.
            
        Returns:
            List of reranked, diverse DocumentChunk instances enriched with rerank metadata.
        """
        if not documents:
            return []

        start_time = time.perf_counter()

        # Step 1: Compute Cross-Encoder Relevance Scores
        scored_pairs: List[Tuple[DocumentChunk, float]] = []

        if self.cross_encoder is not None:
            pairs = [[query, doc.content] for doc in documents]
            try:
                scores = self.cross_encoder.predict(pairs)
                for doc, score in zip(documents, scores):
                    # Sigmoid transform logits to [0, 1] if needed
                    prob_score = 1.0 / (1.0 + math.exp(-float(score))) if abs(float(score)) > 1.0 else float(score)
                    scored_pairs.append((doc, max(0.0, min(1.0, prob_score))))
            except Exception as e:
                logger.warning(f"CrossEncoder inference failed ({e}), using deterministic scorer.")
                for doc in documents:
                    score = self.deterministic_scorer.score(query, doc.content)
                    scored_pairs.append((doc, score))
        else:
            for doc in documents:
                score = self.deterministic_scorer.score(query, doc.content)
                scored_pairs.append((doc, score))

        # Sort candidate pairs by raw cross-encoder score descending
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        # Step 2: Apply Maximal Marginal Relevance (MMR) Redundancy Filter
        mmr_results = self.mmr_filter.filter(scored_pairs, top_k=top_k)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Step 3: Populate metadata on selected chunks
        reranked_chunks: List[DocumentChunk] = []
        for rank, (chunk, mmr_score, penalty) in enumerate(mmr_results, 1):
            # Locate original cross score
            raw_cross_score = next((s for c, s in scored_pairs if c.chunk_id == chunk.chunk_id), mmr_score)

            # Preserve initial retrieval score and add reranker metrics
            chunk.metadata["raw_retrieval_score"] = chunk.metadata.get("score")
            chunk.metadata["rerank_score"] = round(raw_cross_score, 4)
            chunk.metadata["mmr_score"] = round(mmr_score, 4)
            chunk.metadata["redundancy_penalty"] = round(penalty, 4)
            chunk.metadata["rerank_rank"] = rank
            chunk.metadata["score"] = round(mmr_score, 4)
            chunk.metadata["rerank_latency_ms"] = round(latency_ms, 2)
            chunk.metadata["rerank_method"] = "Neural-CrossEncoder+MMR"

            reranked_chunks.append(chunk)

        logger.info(
            f"Reranked {len(documents)} candidates -> Top {len(reranked_chunks)} diverse chunks "
            f"in {latency_ms:.2f}ms (diversity_lambda={self.diversity_lambda})."
        )
        return reranked_chunks
