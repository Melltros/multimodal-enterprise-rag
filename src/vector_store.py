"""
Module: vector_store.py
Description: Vector embedding generation using OpenAI models and vector storage/retrieval
             with Qdrant (in-memory or remote server).
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models

from src.parser import DocumentChunk

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class VectorStoreManager:
    """
    Manages embedding generation via OpenAI and vector indexing/semantic search
    in a Qdrant collection.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        qdrant_url: Optional[str] = ":memory:",
        qdrant_api_key: Optional[str] = None,
        collection_name: str = "financial_reports",
        vector_dim: int = 1536,
        mock_mode: bool = False
    ) -> None:
        """
        Initialize the VectorStoreManager.

        Args:
            openai_api_key: OpenAI API key. If None, reads from OPENAI_API_KEY environment variable.
            embedding_model: OpenAI embedding model name.
            qdrant_url: Qdrant endpoint or ':memory:' for local in-memory storage.
            qdrant_api_key: Optional API key for hosted Qdrant Cloud instances.
            collection_name: Name of the vector collection.
            vector_dim: Dimensionality of the embedding vectors (1536 for text-embedding-3-small).
            mock_mode: If True, uses deterministic mock embeddings without requiring an API key.
        """
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.mock_mode = mock_mode

        # Initialize OpenAI Client
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key and not mock_mode:
            logger.warning("No OPENAI_API_KEY provided. Operating in mock embedding mode.")
            self.mock_mode = True

        if not self.mock_mode and api_key:
            self.openai_client: Optional[OpenAI] = OpenAI(api_key=api_key)
        else:
            self.openai_client = None

        # Initialize Qdrant Client
        target_url = qdrant_url or os.getenv("QDRANT_URL", ":memory:")
        target_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")

        try:
            if target_url == ":memory:":
                logger.info("Initializing in-memory Qdrant instance.")
                self.qdrant_client = QdrantClient(location=":memory:")
            else:
                logger.info(f"Connecting to Qdrant at {target_url}")
                self.qdrant_client = QdrantClient(url=target_url, api_key=target_key)
        except Exception as e:
            logger.error(f"Failed to initialize QdrantClient: {e}", exc_info=True)
            raise ConnectionError(f"Could not initialize Qdrant: {e}") from e

        self._ensure_collection_exists()

    def _ensure_collection_exists(self) -> None:
        """Creates the Qdrant collection if it does not already exist."""
        try:
            collections = self.qdrant_client.get_collections().collections
            existing_names = [col.name for col in collections]
            if self.collection_name not in existing_names:
                logger.info(f"Creating Qdrant collection '{self.collection_name}' with dim {self.vector_dim}")
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=rest_models.VectorParams(
                        size=self.vector_dim,
                        distance=rest_models.Distance.COSINE
                    )
                )
        except Exception as e:
            logger.error(f"Error ensuring collection '{self.collection_name}': {e}", exc_info=True)
            raise

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for a single text snippet.

        Args:
            text: Input string.

        Returns:
            List of floats representing the embedding vector.
        """
        if self.mock_mode or not self.openai_client:
            return self._generate_mock_embedding(text)

        try:
            # Clean newlines for optimal embedding quality
            clean_text = text.replace("\n", " ").strip()
            if not clean_text:
                clean_text = " "

            response = self.openai_client.embeddings.create(
                input=[clean_text],
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding generation failed: {e}", exc_info=True)
            raise

    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text snippets in batch.
        """
        if not texts:
            return []

        if self.mock_mode or not self.openai_client:
            return [self._generate_mock_embedding(t) for t in texts]

        try:
            clean_texts = [t.replace("\n", " ").strip() or " " for t in texts]
            response = self.openai_client.embeddings.create(
                input=clean_texts,
                model=self.embedding_model
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}", exc_info=True)
            raise

    def _generate_mock_embedding(self, text: str) -> List[float]:
        """
        Generates deterministic pseudo-embeddings based on string hashing
        for testing and offline demonstration without an API key.
        """
        import random
        # Seed by hash to ensure determinism for same string
        h = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
        rng = random.Random(h)
        vec = [rng.gauss(0, 1) for _ in range(self.vector_dim)]
        # Normalize to unit length
        norm = (sum(x * x for x in vec)) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def index_chunks(self, chunks: List[DocumentChunk], batch_size: int = 64) -> int:
        """
        Upserts document chunks with their embeddings and metadata into Qdrant.

        Args:
            chunks: List of DocumentChunk objects.
            batch_size: Number of chunks per batch.

        Returns:
            Total number of indexed chunks.
        """
        if not chunks:
            logger.warning("No chunks provided to index.")
            return 0

        logger.info(f"Indexing {len(chunks)} chunks into collection '{self.collection_name}'...")
        total_indexed = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.content for c in batch]
            embeddings = self.get_embeddings_batch(texts)

            points = []
            for chunk, vector in zip(batch, embeddings):
                # Deterministic integer or UUID id for Qdrant point
                point_id = str(chunk.chunk_id)
                payload = {
                    "content": chunk.content,
                    "page_number": chunk.page_number,
                    "source_file": chunk.source_file,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata
                }
                points.append(
                    rest_models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                )

            try:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True
                )
                total_indexed += len(batch)
            except Exception as e:
                logger.error(f"Failed to upsert points batch into Qdrant: {e}", exc_info=True)
                raise

        logger.info(f"Successfully indexed {total_indexed} points.")
        return total_indexed

    def search(self, query: str, top_k: int = 3) -> List[DocumentChunk]:
        """
        Performs semantic vector search for a given user query.

        Args:
            query: The search query string.
            top_k: Number of most relevant chunks to return (default 3).

        Returns:
            List of top_k DocumentChunk instances sorted by relevance score.
        """
        if not query.strip():
            logger.warning("Empty search query received.")
            return []

        logger.info(f"Searching top {top_k} chunks for query: '{query}'")
        query_vector = self.get_embedding(query)

        try:
            hits = []
            if hasattr(self.qdrant_client, "query_points"):
                response = self.qdrant_client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                    with_payload=True
                )
                hits = response.points
            elif hasattr(self.qdrant_client, "search"):
                hits = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    with_payload=True
                )

            retrieved_chunks: List[DocumentChunk] = []
            for hit in hits:
                payload = hit.payload or {}
                chunk = DocumentChunk(
                    chunk_id=str(hit.id),
                    content=payload.get("content", ""),
                    page_number=payload.get("page_number", 1),
                    source_file=payload.get("source_file", "unknown"),
                    chunk_index=payload.get("chunk_index", 0),
                    metadata={
                        **payload.get("metadata", {}),
                        "score": getattr(hit, "score", 0.0)
                    }
                )
                retrieved_chunks.append(chunk)

            logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks from vector store.")
            return retrieved_chunks

        except Exception as e:
            logger.error(f"Vector search failed in collection '{self.collection_name}': {e}", exc_info=True)
            raise
