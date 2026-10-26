"""Main CLI entrypoint for Agentic RAG Coder."""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Ensure Windows console supports UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Safe .env loader
def load_env_file():
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        env_path = ROOT_DIR / ".env"
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())

load_env_file()

from src.ingestion.loader import LocalDirectoryLoader, save_hash_manifest
from src.ingestion.chunker import DocumentChunker
from src.ingestion.vector_store import ChromaVectorStore
from src.memory.memory_manager import SessionMemoryManager

# Try importing Rich for advanced console formatting
try:
    from rich import print as rprint
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None


def print_banner(text: str):
    if HAS_RICH:
        console.print(Panel.fit(f"[bold cyan]{text}[/bold cyan]"))
    else:
        print(f"\n=== {text} ===")


def print_success(text: str):
    if HAS_RICH:
        console.print(f"[bold green]✓ {text}[/bold green]")
    else:
        print(f"✓ {text}")


def print_error(text: str):
    if HAS_RICH:
        console.print(f"[bold red]Error: {text}[/bold red]")
    else:
        print(f"ERROR: {text}")


def run_index(path: str = "./data/raw", chunk_size: int = 800, overlap: int = 100, reset: bool = False):
    """Recursively scan local files, chunk them, embed, and incrementally upsert into ChromaDB."""
    print_banner("Agentic RAG Coder — Knowledge Ingestion Pipeline")
    start_time = time.time()

    target_path = Path(path).resolve()
    if not target_path.exists():
        print_error(f"Target directory does not exist: {target_path}")
        sys.exit(1)

    manifest_path = "./data/chroma_db/file_hashes.json"
    if reset:
        if HAS_RICH:
            console.print("[yellow]Resetting existing collection and hash manifest...[/yellow]")
        else:
            print("Resetting existing collection and hash manifest...")
        if Path(manifest_path).exists():
            Path(manifest_path).unlink()

    loader = LocalDirectoryLoader(
        base_path=str(target_path),
        manifest_path=manifest_path,
    )
    new_or_modified_docs, deleted_files, updated_manifest = loader.load_incremental()

    all_files_count = len(loader.scan_files())
    unchanged_count = all_files_count - len(new_or_modified_docs)

    if HAS_RICH:
        summary_table = Table(title="Directory Ingestion Summary", show_header=True, header_style="bold magenta")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Count", style="green")
        summary_table.add_row("Total Files Detected", str(all_files_count))
        summary_table.add_row("Unchanged Files (Skipped)", str(unchanged_count))
        summary_table.add_row("New or Modified Files", str(len(new_or_modified_docs)))
        summary_table.add_row("Deleted Files (Purged)", str(len(deleted_files)))
        console.print(summary_table)
    else:
        print(f"Total Files: {all_files_count} | Unchanged: {unchanged_count} | New/Modified: {len(new_or_modified_docs)} | Deleted: {len(deleted_files)}")

    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "chroma_default")
    vector_store = ChromaVectorStore(
        persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db"),
        embedding_provider=embedding_provider,
    )

    if reset:
        vector_store.reset_collection()

    for doc in new_or_modified_docs:
        vector_store.delete_by_source_file(doc.metadata["file_path"])
    for del_file in deleted_files:
        vector_store.delete_by_source_file(del_file)

    if not new_or_modified_docs:
        print_success("Everything is up-to-date. No files required indexing.")
        return

    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = chunker.chunk_documents(new_or_modified_docs)
    print(f"Generated {len(chunks)} semantic chunks.")

    upserted_count = vector_store.upsert_chunks(chunks)
    save_hash_manifest(manifest_path, updated_manifest)

    duration = round(time.time() - start_time, 2)
    stats = vector_store.get_stats()

    if HAS_RICH:
        console.print(Panel(
            f"[bold green]✓ Indexing complete in {duration}s![/bold green]\n"
            f"• Chunks Indexed This Run: [bold]{upserted_count}[/bold]\n"
            f"• Total Knowledge Base Chunks: [bold]{stats['total_chunks']}[/bold]\n"
            f"• Backend: [cyan]{stats.get('backend', 'VectorStore')}[/cyan]\n"
            f"• Persist Directory: [dim]{stats['persist_directory']}[/dim]",
            title="[bold green]Success[/bold green]",
            border_style="green",
        ))
    else:
        print(f"\n✓ Indexing complete in {duration}s!")
        print(f"  - Chunks Indexed: {upserted_count}")
        print(f"  - Total in DB: {stats['total_chunks']}")
        print(f"  - Backend: {stats.get('backend', 'VectorStore')}")


def run_ask(question: str, provider: Optional[str] = None, use_memory: bool = True, verbose: bool = True):
    """Ask a coding question grounded in the local RAG knowledge base."""
    print_banner(f"Query: {question}")

    memory_manager = SessionMemoryManager()
    session_context = memory_manager.get_recent_context(max_turns=2) if use_memory else ""

    active_provider = (provider or os.getenv("DEFAULT_LLM_PROVIDER", "groq")).lower().strip()
    key_map = {
        "groq": "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    needed_env_var = key_map.get(active_provider, "GROQ_API_KEY")

    # First check if CrewAI is available
    try:
        from src.agents.crew_definitions import CrewFactory
        if not os.getenv(needed_env_var):
            print_error(f"{needed_env_var} is not set in your .env file.")
            sys.exit(1)

        print(f"Initializing CrewAI (Provider: {active_provider}, Hierarchical Process)...")
        factory = CrewFactory()
        crew = factory.create_coding_crew(user_query=question, session_context=session_context)
        result = crew.kickoff()
        result_text = str(result)

    except Exception as e:
        # Graceful fallback: local RAG + Direct LLM call or Grounded Synthesis
        print(f"[Note] CrewAI execution note ({e}). Falling back to Direct Grounded Synthesis...")
        from src.tools.vector_query_tool import VectorStoreQueryTool
        from src.tools.web_search_tool import WebSearchTool

        retriever = VectorStoreQueryTool()
        retrieved_context = retriever._run(question)

        researcher = WebSearchTool()
        web_context = researcher._run(question)

        # Assemble grounded response
        response_parts = [
            f"### Analysis for Query: `{question}`\n",
            "#### 1. Grounded Context Retrieved",
            retrieved_context,
            "\n#### 2. External Research Status",
            web_context,
            "\n#### 3. Recommended Solution & Engineering Guidance",
            f"Based on your local codebase files:\n",
        ]

        if "LOCAL_CONTEXT_FOUND" in retrieved_context:
            response_parts.append(
                "Your local knowledge base provides direct references for this query. "
                "See the exact file paths and snippets listed above in section 1."
            )
        else:
            response_parts.append(
                "> [!WARNING]\n"
                "> No matching local context was found in your indexed files. "
                "Ensure you have indexed the relevant project files via `python src/main.py index`."
            )

        response_parts.append("\n#### 4. Sources & Citations")
        if "LOCAL_CONTEXT_FOUND" in retrieved_context:
            response_parts.append("- **Local Codebase**: See matched chunks in Section 1")
        else:
            response_parts.append("- **Local Codebase**: No matching documents found")

        if "WEB_SEARCH_SUCCESS" in web_context:
            response_parts.append("- **External Web**: Authoritative URLs in Section 2")
        else:
            response_parts.append("- **External Web**: Search unavailable (no active Tavily/SerpAPI key)")

        result_text = "\n".join(response_parts)

    # Print response
    if HAS_RICH:
        console.print("\n" + "=" * 60)
        console.print(Panel(
            Markdown(result_text),
            title="[bold green]AI Senior Engineer Response[/bold green]",
            border_style="green",
            expand=True,
        ))
    else:
        print("\n" + "=" * 60)
        print("AI Senior Engineer Response:")
        print("=" * 60)
        print(result_text)

    memory_manager.add_turn(query=question, response=result_text)


def run_stats():
    """Display vector store and configuration statistics."""
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "chroma_default")
    vector_store = ChromaVectorStore(
        persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db"),
        embedding_provider=embedding_provider,
    )
    s = vector_store.get_stats()

    if HAS_RICH:
        stats_table = Table(title="System & Vector Store Status", show_header=True, header_style="bold cyan")
        stats_table.add_column("Property", style="bold")
        stats_table.add_column("Value", style="green")
        stats_table.add_row("Collection Name", s["collection_name"])
        stats_table.add_row("Total Chunks in DB", str(s["total_chunks"]))
        stats_table.add_row("Persist Directory", s["persist_directory"])
        stats_table.add_row("Backend", s.get("backend", "ChromaDB"))
        stats_table.add_row("Embedding Provider", s["embedding_provider"])
        stats_table.add_row("Default LLM Provider", os.getenv("DEFAULT_LLM_PROVIDER", "groq"))
        stats_table.add_row("Web Search Configured", "Yes" if os.getenv("TAVILY_API_KEY") or os.getenv("SERPAPI_API_KEY") else "No (Local RAG only)")
        console.print(stats_table)
    else:
        print("\n=== System & Vector Store Status ===")
        for k, v in s.items():
            print(f"  {k}: {v}")
        print(f"  Default LLM Provider: {os.getenv('DEFAULT_LLM_PROVIDER', 'groq')}")


def run_history():
    memory_manager = SessionMemoryManager()
    if not memory_manager.history:
        print("No conversation history recorded in this session.")
        return

    print(f"\nSession History ({len(memory_manager.history)} turns):")
    for i, turn in enumerate(memory_manager.history, 1):
        print(f"\n--- Turn {i} ({turn.timestamp}) ---")
        print(f"Q: {turn.query}")
        print(f"A: {turn.response[:200]}...")


def run_clear_memory():
    memory_manager = SessionMemoryManager()
    memory_manager.clear()
    print_success("Session memory cleared successfully.")


def main():
    parser = argparse.ArgumentParser(description="[AI] Personalized AI Coding Agent with Local RAG & CrewAI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # index command
    idx_p = subparsers.add_parser("index", help="Index local directory into vector store")
    idx_p.add_argument("--path", "-p", default="./data/raw", help="Path to local folder")
    idx_p.add_argument("--chunk-size", "-c", type=int, default=800, help="Chunk size")
    idx_p.add_argument("--overlap", "-o", type=int, default=100, help="Chunk overlap")
    idx_p.add_argument("--reset", action="store_true", help="Reset collection before indexing")

    # ask command
    ask_p = subparsers.add_parser("ask", help="Ask a question grounded in knowledge base")
    ask_p.add_argument("question", help="Coding question")
    ask_p.add_argument("--provider", help="LLM provider: groq, openai, anthropic")
    ask_p.add_argument("--no-memory", action="store_true", help="Disable session memory")

    # stats command
    subparsers.add_parser("stats", help="Display vector store stats")

    # history command
    subparsers.add_parser("history", help="Show session conversation history")

    # clear-memory command
    subparsers.add_parser("clear-memory", help="Clear session memory")

    args = parser.parse_args()

    if args.command == "index":
        run_index(path=args.path, chunk_size=args.chunk_size, overlap=args.overlap, reset=args.reset)
    elif args.command == "ask":
        run_ask(question=args.question, provider=args.provider, use_memory=not args.no_memory)
    elif args.command == "stats":
        run_stats()
    elif args.command == "history":
        run_history()
    elif args.command == "clear-memory":
        run_clear_memory()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
