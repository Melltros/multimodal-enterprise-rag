"""
Module: hybrid_search.py
Description: Production-grade Hybrid Retrieval engine combining Dense Semantic Vector Search
             (Qdrant) and Sparse Lexical Keyword Search (BM25) via Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.parser import DocumentChunk

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class BM25Okapi:
    """
    Okapi BM25 sparse lexical ranking model implemented in pure Python with
    zero external dependencies. Provides deterministic, high-speed keyword retrieval
    specifically tailored for exact numbers, codes, and terminology in enterprise reports.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25) -> None:
        """
        Args:
            k1: Term frequency saturation parameter (typically 1.2 to 2.0).
            b: Document length normalization parameter (0.0 = no normalization, 1.0 = full).
            epsilon: Floor value for IDF to avoid non-positive weights on common terms.
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lengths: List[int] = []
        self.doc_term_freqs: List[Counter[str]] = []
        self.doc_freqs: Counter[str] = Counter()
        self.idf: Dict[str, float] = {}
        self.chunks: List[DocumentChunk] = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Converts text into normalized alphanumeric tokens.
        Preserves numbers, dollar signs, percentages, and acronyms crucial for financial data.
        """
        if not text:
            return []
        cleaned = text.lower()
        tokens = re.findall(r"\b[\$€£]?[a-z0-9]+(?:[\.,][a-z0-9]+)*%?\b", cleaned)
        return tokens

    def index_documents(self, documents: Sequence[DocumentChunk]) -> None:
        """
        Indexes a list of DocumentChunk instances for BM25 retrieval.
        Computes term frequencies, document lengths, and inverse document frequencies.
        """
        self.chunks = list(documents)
        self.corpus_size = len(self.chunks)
        self.doc_lengths = []
        self.doc_term_freqs = []
        self.doc_freqs = Counter()
        self.idf = {}

        if self.corpus_size == 0:
            self.avg_doc_len = 0.0
            return

        total_length = 0
        for doc in self.chunks:
            tokens = self.tokenize(doc.content)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len

            term_freq = Counter(tokens)
            self.doc_term_freqs.append(term_freq)
            for term in term_freq:
                self.doc_freqs[term] += 1

        self.avg_doc_len = total_length / self.corpus_size if self.corpus_size > 0 else 0.0

        # Compute IDF with standard Okapi formula
        idf_sum = 0.0
        negative_idfs = []
        for term, freq in self.doc_freqs.items():
            # Standard BM25 IDF: ln((N - n + 0.5) / (n + 0.5) + 1)
            idf_val = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[term] = idf_val
            idf_sum += idf_val
            if idf_val < 0:
                negative_idfs.append(term)

        # Apply epsilon floor to any negative IDF terms based on average positive IDF
        average_idf = idf_sum / len(self.idf) if self.idf else 0.0
        eps = self.epsilon * average_idf
        for term in negative_idfs:
            self.idf[term] = eps

        logger.info(
            f"BM25 index built with {self.corpus_size} documents, "
            f"vocab size {len(self.idf)}, avg doc length {self.avg_doc_len:.1f} tokens."
        )

    def score_document(self, query_tokens: Sequence[str], doc_idx: int) -> float:
        """
        Computes the BM25 score of a single document against query tokens.
        """
        if doc_idx < 0 or doc_idx >= self.corpus_size or self.avg_doc_len == 0.0:
            return 0.0

        doc_len = self.doc_lengths[doc_idx]
        term_freqs = self.doc_term_freqs[doc_idx]
        score = 0.0

        for token in query_tokens:
            if token not in term_freqs:
                continue
            tf = term_freqs[token]
            idf = self.idf.get(token, 0.0)
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Executes lexical BM25 search and returns top-k ranked chunks with scores.
        """
        if self.corpus_size == 0 or not query.strip():
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores: List[Tuple[int, float]] = []
        for idx in range(self.corpus_size):
            s = self.score_document(query_tokens, idx)
            if s > 0.0:
                scores.append((idx, s))

        # Sort descending by score
        scores.sort(key=lambda x: x[1], reverse=True)
        results: List[Tuple[DocumentChunk, float]] = []
        for idx, s in scores[:top_k]:
            results.append((self.chunks[idx], float(s)))

        return results


class ReciprocalRankFusion:
    """
    Implements Reciprocal Rank Fusion (RRF) for unifying disparate retrieval rankings.
    Formula: RRF(d) = sum(1 / (k + rank_m(d))) for all retrieval systems m.
    """

    def __init__(self, k: int = 60) -> None:
        """
        Args:
            k: Ranking smoothing constant (standard empirical default is 60).
        """
        self.k = k

    def fuse(
        self,
        ranked_lists: Dict[str, Sequence[Tuple[DocumentChunk, float]]],
        top_k: int = 5
    ) -> List[DocumentChunk]:
        """
        Fuses multiple ranked lists (e.g. 'dense', 'bm25') into a single deduplicated ranking.

        Args:
            ranked_lists: Dict mapping retriever name ('dense', 'bm25') to ranked list of (chunk, score).
            top_k: Number of unified results to return.

        Returns:
            List of DocumentChunk instances sorted by descending RRF score,
            enriched with ranking attribution metadata.
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}
        rank_breakdown: Dict[str, Dict[str, int]] = {}
        score_breakdown: Dict[str, Dict[str, float]] = {}

        for system_name, candidates in ranked_lists.items():
            for rank, (chunk, raw_score) in enumerate(candidates, start=1):
                cid = chunk.chunk_id
                if cid not in chunk_map:
                    # Make a shallow copy of chunk to safely attach fused metadata
                    chunk_map[cid] = chunk.model_copy(deep=True)
                    rrf_scores[cid] = 0.0
                    rank_breakdown[cid] = {}
                    score_breakdown[cid] = {}

                # Calculate RRF contribution
                contribution = 1.0 / (self.k + rank)
                rrf_scores[cid] += contribution
                rank_breakdown[cid][system_name] = rank
                score_breakdown[cid][system_name] = raw_score

        # Sort chunk IDs by descending RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        results: List[DocumentChunk] = []
        for cid in sorted_ids[:top_k]:
            chunk = chunk_map[cid]
            ranks = rank_breakdown[cid]
            scores = score_breakdown[cid]

            # Determine retrieval attribution
            if "dense" in ranks and "bm25" in ranks:
                method = "hybrid"
            elif "dense" in ranks:
                method = "dense"
            else:
                method = "bm25"

            chunk.metadata["retrieval_method"] = method
            chunk.metadata["rrf_score"] = round(rrf_scores[cid], 6)
            chunk.metadata["dense_rank"] = ranks.get("dense")
            chunk.metadata["bm25_rank"] = ranks.get("bm25")
            chunk.metadata["dense_score"] = scores.get("dense")
            chunk.metadata["bm25_score"] = scores.get("bm25")
            chunk.metadata["score"] = chunk.metadata["rrf_score"]

            results.append(chunk)

        return results


class HybridRetriever:
    """
    Unified retrieval orchestrator providing Dense, BM25 Lexical, or Hybrid RRF search.
    """

    def __init__(
        self,
        vector_manager: Any,
        documents: Optional[Sequence[DocumentChunk]] = None,
        rrf_k: int = 60
    ) -> None:
        """
        Args:
            vector_manager: Initialized VectorStoreManager instance.
            documents: Optional list of documents to index immediately in BM25.
            rrf_k: Reciprocal Rank Fusion constant (default 60).
        """
        self.vector_manager = vector_manager
        self.bm25 = BM25Okapi()
        self.rrf = ReciprocalRankFusion(k=rrf_k)
        self.indexed_chunks: List[DocumentChunk] = []

        if documents:
            self.index_documents(documents)

    def index_documents(self, documents: Sequence[DocumentChunk]) -> None:
        """Indexes documents into both the BM25 engine and retains chunk references."""
        self.indexed_chunks = list(documents)
        self.bm25.index_documents(documents)

    def search(
        self,
        query: str,
        top_k: int = 3,
        mode: str = "hybrid"
    ) -> List[DocumentChunk]:
        """
        Performs retrieval according to the specified mode ('hybrid', 'dense', 'bm25').

        Args:
            query: The user query string.
            top_k: Maximum number of chunks to return.
            mode: 'hybrid' (fused RRF), 'dense' (vector cosine), or 'bm25' (keyword).
        """
        mode = mode.lower().strip()
        fetch_limit = max(top_k * 2, 5)

        if mode == "dense":
            chunks = self.vector_manager.search(query=query, top_k=top_k)
            for c in chunks:
                c.metadata.setdefault("retrieval_method", "dense")
            return chunks

        if mode == "bm25":
            bm25_results = self.bm25.search(query=query, top_k=top_k)
            results: List[DocumentChunk] = []
            for chunk, score in bm25_results:
                c = chunk.model_copy(deep=True)
                c.metadata["retrieval_method"] = "bm25"
                c.metadata["score"] = score
                results.append(c)
            return results

        if mode == "hybrid":
            # 1. Retrieve dense candidates with scores
            dense_candidates = self.vector_manager.search_with_scores(query=query, top_k=fetch_limit)
            # 2. Retrieve sparse candidates with scores
            bm25_candidates = self.bm25.search(query=query, top_k=fetch_limit)

            # If either list is empty, fallback gracefully
            if not dense_candidates and not bm25_candidates:
                return []
            if not dense_candidates:
                return [c for c, _ in bm25_candidates[:top_k]]
            if not bm25_candidates:
                return [c for c, _ in dense_candidates[:top_k]]

            # Fuse with RRF
            fused = self.rrf.fuse(
                ranked_lists={
                    "dense": dense_candidates,
                    "bm25": bm25_candidates,
                },
                top_k=top_k
            )
            return fused

        raise ValueError(f"Unknown retrieval mode: '{mode}'. Supported modes are: 'hybrid', 'dense', 'bm25'.")
