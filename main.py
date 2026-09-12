"""
Module: main.py
Description: End-to-end execution script orchestrating PDF ingestion, vector indexing,
             grounded synthesis with citations, and Critic LLM evaluation.
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
from src.vector_store import VectorStoreManager


# Initialize rich console and logging
console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("enterprise_rag")


def print_banner() -> None:
    """Displays a stylized terminal banner."""
    banner_text = Text("⚡ MULTIMODAL ENTERPRISE RAG PIPELINE ⚡", style="bold cyan")
    sub_text = Text("Grounded Financial PDF Intelligence • Source Citations • LLM-as-a-Judge Critic", style="dim white")
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
    grounded answer, citations, and Critic evaluation metrics.
    """
    console.print("\n")

    # 1. Retrieved Context Table
    context_table = Table(title="🔍 Retrieved Document Contexts", border_style="cyan", show_lines=True)
    context_table.add_column("#", style="bold yellow", width=4)
    context_table.add_column("Source", style="green", width=20)
    context_table.add_column("Page", style="bold magenta", width=6)
    context_table.add_column("Method", style="bold blue", width=10)
    context_table.add_column("Score", style="cyan", width=8)
    context_table.add_column("Context Preview", style="white")

    for idx, chunk in enumerate(retrieved_chunks, 1):
        score_val = chunk.metadata.get("score")
        score_str = f"{score_val:.4f}" if score_val is not None else "N/A"
        method_str = str(chunk.metadata.get("retrieval_method", "dense")).upper()
        preview = chunk.content.replace("\n", " ")[:130] + ("..." if len(chunk.content) > 130 else "")
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
    export_audit: bool = True
) -> None:
    """
    Executes the end-to-end RAG pipeline from ingestion to Critic evaluation.
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

    # Step 4: Hybrid / Dense / BM25 Search & Retrieval
    console.print(f"\n[bold blue][3/4] Retrieving Top {top_k} Contexts ({retrieval_mode.upper()} mode) for Query:[/bold blue] [cyan]'{query}'[/cyan]")
    try:
        retrieved_chunks = retriever.search(query=query, top_k=top_k, mode=retrieval_mode)
    except Exception as e:
        console.print(f"[bold red]❌ Retrieval failed:[/bold red] {e}")
        sys.exit(1)

    # Step 5: Answer Synthesis with Citations
    console.print(f"\n[bold blue][4/4] Synthesizing Answer & Running Critic Evaluation...[/bold blue]")
    generator = RAGGenerator(mock_mode=mock_mode)
    gen_response = generator.generate_answer(query=query, context_chunks=retrieved_chunks)

    # Step 6: Critic LLM-as-a-Judge Evaluation
    critic = CriticEvaluator(mock_mode=mock_mode)
    eval_result = critic.evaluate(
        query=query,
        context_chunks=retrieved_chunks,
        generated_answer=gen_response.answer
    )

    # Step 7: Render Final Results
    display_results(
        query=query,
        answer=gen_response.answer,
        citations=gen_response.citations,
        eval_result=eval_result,
        retrieved_chunks=retrieved_chunks
    )

    # Step 8: Enterprise Audit Trail
    if export_audit:
        try:
            audit_logger = EnterpriseAuditLogger(log_dir="logs")
            audit_logger.log_execution(
                query=query,
                retrieval_mode=retrieval_mode,
                retrieved_chunks=retrieved_chunks,
                answer=gen_response.answer,
                citations=gen_response.citations,
                eval_result=eval_result,
                persist=True
            )
            console.print("\n[dim green]✔ Compliance audit record logged to logs/audit_trail.jsonl and logs/audit_summary.csv[/dim green]")
        except Exception as e:
            logger.warning(f"Could not persist audit record: {e}")


def main() -> None:
    """CLI Entrypoint."""
    # Load .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Multimodal Enterprise RAG Pipeline with Hybrid Retrieval & Critic LLM Evaluation"
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
        export_audit=not args.no_audit
    )


if __name__ == "__main__":
    main()
