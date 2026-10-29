"""
Module: main.py
Description: End-to-end execution script orchestrating PDF ingestion, vector indexing,
             hybrid search, neural cross-reranking with MMR, grounded synthesis with
             citations, and Critic LLM evaluation.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv() -> bool:
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))
            return True
        return False

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from data.create_sample_pdf import generate_sample_financial_report
from src.audit_logger import EnterpriseAuditLogger
from src.evaluator import CriticEvaluator
from src.generator import RAGGenerator
from src.hybrid_search import HybridRetriever
from src.parser import PDFParser
from src.reranker import NeuralCrossReranker
from src.vector_store import VectorStoreManager


# Initialize rich console and logging
console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("enterprise_rag")


def print_banner() -> None:
    """Displays a stylized terminal banner."""
    banner_text = Text("⚡ MULTIMODAL ENTERPRISE RAG PIPELINE ⚡", style="bold cyan")
    sub_text = Text(
        "Grounded Financial Intelligence • Hybrid Retrieval • Neural Cross-Reranker • LLM Critic",
        style="dim white"
    )
    console.print(Panel(Text.assemble(banner_text, "\n", sub_text), border_style="blue", expand=False))


def display_results(
    query: str,
    answer: str,
    citations: list,
    eval_result,
    retrieved_chunks: list
) -> None:
    """
    Renders formatted tables and panels detailing the query, retrieved contexts,
    neural reranking scores, grounded answer, citations, and Critic evaluation metrics.
    """
    console.print("\n")

    # Check if cross-reranking was applied
    has_rerank = any("rerank_score" in chunk.metadata for chunk in retrieved_chunks)

    # 1. Retrieved Context Table
    context_table = Table(
        title="🔍 Retrieved & Ranked Document Contexts" if not has_rerank else "🔍 Neural Cross-Reranked Contexts (with MMR Diversity)",
        border_style="cyan",
        show_lines=True
    )
    context_table.add_column("#", style="bold yellow", width=4)
    context_table.add_column("Source", style="green", width=18)
    context_table.add_column("Page", style="bold magenta", width=6)
    context_table.add_column("Method", style="bold blue", width=10)

    if has_rerank:
        context_table.add_column("Rerank", style="bold green", width=8)
        context_table.add_column("MMR", style="cyan", width=8)
        context_table.add_column("Penalty", style="dim yellow", width=8)
    else:
        context_table.add_column("Score", style="cyan", width=8)

    context_table.add_column("Context Preview", style="white")

    for idx, chunk in enumerate(retrieved_chunks, 1):
        method_str = str(chunk.metadata.get("retrieval_method", "dense")).upper()
        preview = chunk.content.replace("\n", " ")[:120] + ("..." if len(chunk.content) > 120 else "")

        if has_rerank:
            r_score = chunk.metadata.get("rerank_score")
            m_score = chunk.metadata.get("mmr_score")
            p_score = chunk.metadata.get("redundancy_penalty")
            r_str = f"{r_score:.4f}" if r_score is not None else "N/A"
            m_str = f"{m_score:.4f}" if m_score is not None else "N/A"
            p_str = f"{p_score:.4f}" if p_score is not None else "0.0000"
            context_table.add_row(str(idx), chunk.source_file, str(chunk.page_number), method_str, r_str, m_str, p_str, preview)
        else:
            score_val = chunk.metadata.get("score")
            score_str = f"{score_val:.4f}" if score_val is not None else "N/A"
            context_table.add_row(str(idx), chunk.source_file, str(chunk.page_number), method_str, score_str, preview)

    console.print(context_table)
    console.print("\n")

    # 2. Synthesized Answer Panel
    answer_panel = Panel(
        f"[bold white]{answer}[/bold white]",
        title=f"🤖 Synthesized Answer for: '{query}'",
        border_style="green",
        subtitle=f"Citations: {citations}"
    )
    console.print(answer_panel)
    console.print("\n")

    # 3. Critic LLM Evaluation Table
    eval_table = Table(title="⚖️ Critic LLM Evaluation (LLM-as-a-Judge)", border_style="magenta", show_lines=True)
    eval_table.add_column("Metric", style="bold white", width=20)
    eval_table.add_column("Score (1-5)", style="bold yellow", width=12)
    eval_table.add_column("Assessment & Reasoning", style="white")

    # Faithfulness Row
    faith_badge = f"[bold green]{eval_result.faithfulness_score}/5[/bold green]" if eval_result.faithfulness_score >= 4 else f"[bold red]{eval_result.faithfulness_score}/5[/bold red]"
    eval_table.add_row("Faithfulness (Grounding)", faith_badge, eval_result.faithfulness_reasoning)

    # Relevancy Row
    rel_badge = f"[bold green]{eval_result.relevancy_score}/5[/bold green]" if eval_result.relevancy_score >= 4 else f"[bold red]{eval_result.relevancy_score}/5[/bold red]"
    eval_table.add_row("Relevancy (Query Match)", rel_badge, eval_result.relevancy_reasoning)

    # Verdict Row
    status_text = "[bold green]✅ APPROVED FOR DELIVERY[/bold green]" if eval_result.passed else "[bold red]🚨 FLAGGED BY CRITIC (Score < 4)[/bold red]"
    eval_table.add_row("Final Status", status_text, eval_result.critique_summary)

    console.print(eval_table)


def run_pipeline(
    pdf_path: str = "data/sample_report.pdf",
    query: str = "What was the company's Q3 net revenue according to page 4?",
    top_k: int = 3,
    mock_mode: bool = False,
    retrieval_mode: str = "hybrid",
    rrf_k: int = 60,
    export_audit: bool = True,
    rerank: bool = False,
    rerank_candidates: int = 10,
    diversity_lambda: float = 0.70
) -> None:
    """
    Executes the end-to-end RAG pipeline from ingestion to Critic evaluation,
    optionally incorporating Neural Cross-Encoder Reranking and MMR Redundancy Filtering.
    """
    print_banner()

    # Step 1: Ensure PDF exists
    target_pdf = Path(pdf_path)
    if not target_pdf.exists():
        console.print(f"[yellow]⚠️ Target PDF '{pdf_path}' not found. Auto-generating sample financial report...[/yellow]")
        target_pdf = generate_sample_financial_report(pdf_path)

    # Step 2: Document Ingestion & Parsing
    console.print(f"\n[bold blue][1/4] Parsing PDF document:[/bold blue] {target_pdf.resolve()}")
    parser = PDFParser(chunk_size=800, chunk_overlap=150, extract_tables=True)
    try:
        chunks = parser.parse_pdf(target_pdf)
    except Exception as e:
        console.print(f"[bold red]❌ Document parsing failed:[/bold red] {e}")
        sys.exit(1)

    console.print(f"[green]✔ Ingested {len(chunks)} chunks across all pages.[/green]")

    # Step 3: Vector Store & BM25 Lexical Indexing
    console.print(f"\n[bold blue][2/4] Indexing Chunks in Qdrant (Vector) & BM25 (Sparse Lexical)...[/bold blue]")
    try:
        vector_mgr = VectorStoreManager(
            collection_name="financial_reports",
            mock_mode=mock_mode
        )
        vector_mgr.index_chunks(chunks)
        retriever = HybridRetriever(
            vector_manager=vector_mgr,
            documents=chunks,
            rrf_k=rrf_k
        )
    except Exception as e:
        console.print(f"[bold red]❌ Indexing failed:[/bold red] {e}")
        sys.exit(1)

    console.print(f"[green]✔ Successfully indexed {len(chunks)} chunks for {retrieval_mode.upper()} retrieval.[/green]")

    # Step 4: Retrieval & Optional Neural Cross-Reranking
    fetch_k = max(top_k, rerank_candidates) if rerank else top_k
    console.print(
        f"\n[bold blue][3/4] Retrieving Candidates ({retrieval_mode.upper()} mode, count={fetch_k}) "
        f"for Query:[/bold blue] [cyan]'{query}'[/cyan]"
    )
    try:
        candidate_chunks = retriever.search(query=query, top_k=fetch_k, mode=retrieval_mode)
    except Exception as e:
        console.print(f"[bold red]❌ Retrieval failed:[/bold red] {e}")
        sys.exit(1)

    if rerank:
        console.print(
            f"[bold blue]⚡ Applying Neural Cross-Encoder Reranking & MMR Redundancy Filter "
            f"(lambda={diversity_lambda:.2f}, target top_k={top_k})...[/bold blue]"
        )
        reranker = NeuralCrossReranker(diversity_lambda=diversity_lambda, mock_mode=mock_mode)
        final_chunks = reranker.rerank(query=query, documents=candidate_chunks, top_k=top_k)
    else:
        final_chunks = candidate_chunks[:top_k]

    # Step 5: Answer Synthesis with Citations
    console.print(f"\n[bold blue][4/4] Synthesizing Answer & Running Critic Evaluation...[/bold blue]")
    generator = RAGGenerator(mock_mode=mock_mode)
    gen_response = generator.generate_answer(query=query, context_chunks=final_chunks)

    # Step 6: Critic LLM-as-a-Judge Evaluation
    critic = CriticEvaluator(mock_mode=mock_mode)
    eval_result = critic.evaluate(
        query=query,
        context_chunks=final_chunks,
        generated_answer=gen_response.answer
    )

    # Step 7: Render Final Results
    display_results(
        query=query,
        answer=gen_response.answer,
        citations=gen_response.citations,
        eval_result=eval_result,
        retrieved_chunks=final_chunks
    )

    # Step 8: Enterprise Audit Trail
    if export_audit:
        try:
            audit_logger = EnterpriseAuditLogger(log_dir="logs")
            audit_logger.log_execution(
                query=query,
                retrieval_mode=retrieval_mode,
                retrieved_chunks=final_chunks,
                answer=gen_response.answer,
                citations=gen_response.citations,
                eval_result=eval_result,
                persist=True,
                reranker_enabled=rerank,
                diversity_lambda=diversity_lambda if rerank else None
            )
            console.print("\n[dim green]✔ Compliance audit record logged to logs/audit_trail.jsonl and logs/audit_summary.csv[/dim green]")
        except Exception as e:
            logger.warning(f"Could not persist audit record: {e}")


def main() -> None:
    """CLI Entrypoint."""
    # Load .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Multimodal Enterprise RAG Pipeline with Hybrid Retrieval, Cross-Reranker & Critic LLM"
    )
    parser.add_argument(
        "--pdf",
        type=str,
        default="data/sample_report.pdf",
        help="Path to the PDF document to ingest (default: data/sample_report.pdf)"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What was the company's Q3 net revenue according to page 4?",
        help="User query to search and synthesize"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of context chunks to retrieve (default: 3)"
    )
    parser.add_argument(
        "--retrieval-mode",
        type=str,
        choices=["hybrid", "dense", "bm25"],
        default="hybrid",
        help="Retrieval strategy: hybrid (Dense + BM25 with RRF), dense (vector only), or bm25 (lexical only)"
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="Reciprocal Rank Fusion smoothing constant (default: 60)"
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Apply Neural Cross-Encoder Reranking and Maximal Marginal Relevance (MMR) Redundancy Filter"
    )
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=10,
        help="Number of initial candidate chunks retrieved before reranking (default: 10)"
    )
    parser.add_argument(
        "--diversity-penalty",
        type=float,
        default=0.70,
        help="Tradeoff parameter lambda for MMR diversity filtering (0.0 = maximal diversity, 1.0 = maximal relevance, default: 0.70)"
    )
    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="Disable automatic export of compliance audit trail logs"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock/offline mode without calling external OpenAI APIs"
    )

    args = parser.parse_args()

    # Determine mock mode: if --mock flag is passed or OPENAI_API_KEY is missing
    openai_key = os.getenv("OPENAI_API_KEY")
    mock_mode = args.mock or not bool(openai_key)

    if mock_mode and not args.mock:
        console.print("[dim yellow]ℹ️ OPENAI_API_KEY not detected in environment. Running in mock demonstration mode.[/dim yellow]")

    run_pipeline(
        pdf_path=args.pdf,
        query=args.query,
        top_k=args.top_k,
        mock_mode=mock_mode,
        retrieval_mode=args.retrieval_mode,
        rrf_k=args.rrf_k,
        export_audit=not args.no_audit,
        rerank=args.rerank,
        rerank_candidates=args.rerank_candidates,
        diversity_lambda=args.diversity_penalty
    )


if __name__ == "__main__":
    main()
