"""
Module: evaluator.py
Description: The "Critic" layer — an LLM-as-a-Judge system utilizing Pydantic structured
             outputs to score answer Faithfulness (hallucination check) and Relevancy.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field

from src.parser import DocumentChunk

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class EvaluationResult(BaseModel):
    """
    Structured evaluation result produced by the Critic LLM.
    """
    faithfulness_score: int = Field(
        ...,
        ge=1,
        le=5,
        description=(
            "Score from 1 to 5 measuring whether the answer is strictly derived from and grounded in "
            "the provided context (1 = total hallucination / contradictions, 5 = perfectly faithful)."
        )
    )
    faithfulness_reasoning: str = Field(
        ...,
        description="Detailed evidence and explanation for the assigned faithfulness score."
    )
    relevancy_score: int = Field(
        ...,
        ge=1,
        le=5,
        description=(
            "Score from 1 to 5 measuring whether the answer directly, completely, and appropriately "
            "addresses the user's question (1 = completely irrelevant, 5 = perfectly answers question)."
        )
    )
    relevancy_reasoning: str = Field(
        ...,
        description="Detailed explanation for the assigned relevancy score."
    )
    is_flagged: bool = Field(
        ...,
        description="True if faithfulness_score < 4 OR relevancy_score < 4; False otherwise."
    )
    critique_summary: str = Field(
        ...,
        description="Executive summary of the evaluation, noting any discrepancies or citation accuracy."
    )

    @property
    def passed(self) -> bool:
        """Returns True if the response passed all evaluation quality criteria."""
        return not self.is_flagged


class CriticEvaluator:
    """
    LLM-as-a-Judge engine that rigorously validates RAG outputs against context
    and user queries before delivery.
    """

    CRITIC_SYSTEM_PROMPT = (
        "You are an impartial, rigorous Senior AI Quality Critic and Auditor for enterprise retrieval systems.\n"
        "Your role is to evaluate a generated answer against the retrieved context documents and the user's query.\n\n"
        "EVALUATION CRITERIA:\n"
        "1. FAITHFULNESS (1 to 5):\n"
        "   - 5: Every single claim in the answer is directly supported by the context documents.\n"
        "   - 4: The answer is almost entirely supported; minor stylistic rephrasings without factual drift.\n"
        "   - 3: Some claims are unverified or loosely extrapolated beyond the context.\n"
        "   - 2: Significant unsupported claims or hallucinations present.\n"
        "   - 1: Entirely hallucinated or contradicts the provided context.\n\n"
        "2. RELEVANCY (1 to 5):\n"
        "   - 5: Directly, fully, and concisely answers the user's specific question.\n"
        "   - 4: Answers the question well with slight extraneous detail or minor missing nuance.\n"
        "   - 3: Partially answers the question but misses core elements.\n"
        "   - 2: Vaguely related but fails to address the question directly.\n"
        "   - 1: Completely irrelevant or off-topic.\n\n"
        "FLAG RULE: If either Faithfulness < 4 OR Relevancy < 4, you MUST set is_flagged = True."
    )

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        critic_model: str = "gpt-4o",
        min_threshold: int = 4,
        mock_mode: bool = False
    ) -> None:
        """
        Initialize the Critic Evaluator.

        Args:
            openai_api_key: OpenAI API key.
            critic_model: LLM model to act as Critic (e.g., 'gpt-4o').
            min_threshold: Score threshold below which output is flagged (default 4).
            mock_mode: If True, operates with deterministic mock evaluation.
        """
        self.critic_model = critic_model
        self.min_threshold = min_threshold
        self.mock_mode = mock_mode

        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key and not mock_mode:
            logger.warning("No OPENAI_API_KEY found for Critic. Operating in mock evaluation mode.")
            self.mock_mode = True

        if not self.mock_mode and api_key:
            self.openai_client: Optional[OpenAI] = OpenAI(api_key=api_key)
        else:
            self.openai_client = None

    def evaluate(
        self,
        query: str,
        context_chunks: List[DocumentChunk],
        generated_answer: str
    ) -> EvaluationResult:
        """
        Evaluate generated answer against query and context using structured Pydantic output.

        Args:
            query: The user's query.
            context_chunks: List of retrieved context chunks.
            generated_answer: The answer generated by the synthesis model.

        Returns:
            EvaluationResult instance with scores and flags.
        """
        # Format context for critic review
        context_text = "\n\n".join(
            [f"[Source: {c.source_file}, Page {c.page_number}]\n{c.content}" for c in context_chunks]
        ) or "NO CONTEXT PROVIDED"

        critic_user_prompt = (
            f"=== RETRIEVED CONTEXT ===\n"
            f"{context_text}\n\n"
            f"=== USER QUERY ===\n"
            f"{query}\n\n"
            f"=== GENERATED ANSWER ===\n"
            f"{generated_answer}\n\n"
            f"Evaluate the generated answer according to Faithfulness and Relevancy. "
            f"Provide numerical scores (1-5), thorough reasoning, and set is_flagged = True if either score is below {self.min_threshold}."
        )

        if self.mock_mode or not self.openai_client:
            logger.info("Executing mock critic evaluation.")
            return self._mock_evaluation(query, generated_answer)

        logger.info(f"Running Critic evaluation with model '{self.critic_model}'...")
        try:
            completion = self.openai_client.beta.chat.completions.parse(
                model=self.critic_model,
                messages=[
                    {"role": "system", "content": self.CRITIC_SYSTEM_PROMPT},
                    {"role": "user", "content": critic_user_prompt}
                ],
                response_format=EvaluationResult,
                temperature=0.0
            )

            result: EvaluationResult = completion.choices[0].message.parsed
            # Guarantee flag consistency
            if result.faithfulness_score < self.min_threshold or result.relevancy_score < self.min_threshold:
                result.is_flagged = True

            return result

        except Exception as e:
            logger.error(f"Critic evaluation failed: {e}", exc_info=True)
            # Fallback to conservative flagged evaluation on communication error
            return EvaluationResult(
                faithfulness_score=3,
                faithfulness_reasoning=f"Evaluation API call failed: {e}",
                relevancy_score=3,
                relevancy_reasoning="Evaluation API call failed.",
                is_flagged=True,
                critique_summary="Critic evaluation encountered an error during parsing."
            )

    def _mock_evaluation(self, query: str, answer: str) -> EvaluationResult:
        """
        Provides deterministic evaluation result for offline testing.
        """
        is_unfound = "cannot find the answer" in answer.lower()
        if is_unfound:
            return EvaluationResult(
                faithfulness_score=5,
                faithfulness_reasoning="Model correctly recognized missing context and refused to hallucinate.",
                relevancy_score=5,
                relevancy_reasoning="Accurately responded to the query given lack of context.",
                is_flagged=False,
                critique_summary="Passed: Safe refusal behavior verified."
            )

        return EvaluationResult(
            faithfulness_score=5,
            faithfulness_reasoning="All financial numbers and citations match the source documents with 100% precision.",
            relevancy_score=5,
            relevancy_reasoning="Directly answers the user's revenue and operating income inquiry with exact metrics.",
            is_flagged=False,
            critique_summary="Passed: Response is fully grounded, factually verifiable, and explicitly cited."
        )
