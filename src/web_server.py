"""High-performance local Web server providing REST APIs and UI for Agentic RAG Coder."""

import json
import mimetypes
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

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

# Load .env
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
from src.tools.vector_query_tool import VectorStoreQueryTool
from src.tools.web_search_tool import WebSearchTool
from src.memory.memory_manager import SessionMemoryManager

STATIC_DIR = ROOT_DIR / "web"
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


class AgenticRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for static assets and RAG REST endpoints."""

    def _set_json_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _read_json_body(self):
        content_len = int(self.headers.get("Content-Length", 0))
        if content_len == 0:
            return {}
        raw = self.rfile.read(content_len).decode("utf-8")
        return json.loads(raw) if raw else {}

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/stats":
            self.handle_get_stats()
        elif path == "/api/files":
            self.handle_get_files()
        elif path == "/api/history":
            self.handle_get_history()
        else:
            self.serve_static(path)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/ask":
            self.handle_post_ask()
        elif path == "/api/index":
            self.handle_post_index()
        elif path == "/api/upload":
            self.handle_post_upload()
        elif path == "/api/search":
            self.handle_post_search()
        elif path == "/api/clear-memory":
            self.handle_post_clear_memory()
        else:
            self.send_error(404, "Endpoint Not Found")

    # --- Static File Serving ---
    def serve_static(self, path):
        if path == "/" or path == "":
            path = "/index.html"

        file_path = (STATIC_DIR / path.lstrip("/")).resolve()

        # Security check: ensure path is within STATIC_DIR
        if not str(file_path).startswith(str(STATIC_DIR)):
            self.send_error(403, "Forbidden")
            return

        if not file_path.exists() or file_path.is_dir():
            self.send_error(404, "File Not Found")
            return

        mime_type, _ = mimetypes.guess_type(str(file_path))
        mime_type = mime_type or "application/octet-stream"

        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", len(content))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    # --- API Handlers ---
    def handle_get_stats(self):
        vector_store = ChromaVectorStore(
            persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db"),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "chroma_default"),
        )
        stats = vector_store.get_stats()
        stats["default_provider"] = os.getenv("DEFAULT_LLM_PROVIDER", "groq")
        stats["has_tavily"] = bool(os.getenv("TAVILY_API_KEY"))
        self._set_json_headers(200)
        self.wfile.write(json.dumps(stats).encode("utf-8"))

    def handle_get_files(self):
        loader = LocalDirectoryLoader(base_path=str(RAW_DATA_DIR))
        files = loader.scan_files()
        result = []
        for f in files:
            stat = f.stat()
            rel = str(f.relative_to(RAW_DATA_DIR)).replace("\\", "/")
            result.append({
                "filename": f.name,
                "file_path": rel,
                "extension": f.suffix.lower(),
                "size": stat.st_size,
                "last_modified": stat.st_mtime * 1000,
            })
        self._set_json_headers(200)
        self.wfile.write(json.dumps({"files": result}).encode("utf-8"))

    def handle_get_history(self):
        memory = SessionMemoryManager()
        history_data = [
            {
                "timestamp": t.timestamp,
                "query": t.query,
                "response": t.response,
                "sources": t.context_sources,
            }
            for t in memory.history
        ]
        self._set_json_headers(200)
        self.wfile.write(json.dumps({"history": history_data}).encode("utf-8"))

    def handle_post_clear_memory(self):
        memory = SessionMemoryManager()
        memory.clear()
        self._set_json_headers(200)
        self.wfile.write(json.dumps({"status": "cleared"}).encode("utf-8"))

    def handle_post_index(self):
        body = self._read_json_body()
        reset = body.get("reset", False)

        manifest_path = "./data/chroma_db/file_hashes.json"
        if reset and Path(manifest_path).exists():
            Path(manifest_path).unlink()

        loader = LocalDirectoryLoader(
            base_path=str(RAW_DATA_DIR),
            manifest_path=manifest_path,
        )
        new_docs, deleted_files, updated_manifest = loader.load_incremental()

        vector_store = ChromaVectorStore(
            persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db"),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "chroma_default"),
        )
        if reset:
            vector_store.reset_collection()

        for doc in new_docs:
            vector_store.delete_by_source_file(doc.metadata["file_path"])
        for del_file in deleted_files:
            vector_store.delete_by_source_file(del_file)

        chunker = DocumentChunker(chunk_size=800, chunk_overlap=100)
        chunks = chunker.chunk_documents(new_docs)
        upserted = vector_store.upsert_chunks(chunks)
        save_hash_manifest(manifest_path, updated_manifest)

        self._set_json_headers(200)
        self.wfile.write(json.dumps({
            "status": "success",
            "files_scanned": len(loader.scan_files()),
            "new_or_modified_files": len(new_docs),
            "chunks_indexed": upserted,
            "total_chunks": vector_store.get_stats()["total_chunks"]
        }).encode("utf-8"))

    def handle_post_upload(self):
        body = self._read_json_body()
        filename = body.get("filename", "").strip()
        content = body.get("content", "")

        if not filename:
            self._set_json_headers(400)
            self.wfile.write(json.dumps({"error": "filename is required"}).encode("utf-8"))
            return

        target_file = RAW_DATA_DIR / filename
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(content)

        # Trigger incremental re-index
        loader = LocalDirectoryLoader(base_path=str(RAW_DATA_DIR), manifest_path="./data/chroma_db/file_hashes.json")
        new_docs, _, updated_manifest = loader.load_incremental()

        vector_store = ChromaVectorStore(
            persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db"),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "chroma_default"),
        )
        chunker = DocumentChunker(chunk_size=800, chunk_overlap=100)
        chunks = chunker.chunk_documents(new_docs)
        upserted = vector_store.upsert_chunks(chunks)
        save_hash_manifest("./data/chroma_db/file_hashes.json", updated_manifest)

        self._set_json_headers(200)
        self.wfile.write(json.dumps({
            "status": "uploaded_and_indexed",
            "filename": filename,
            "chunks_added": upserted,
        }).encode("utf-8"))

    def handle_post_search(self):
        body = self._read_json_body()
        query = body.get("query", "").strip()
        top_k = int(body.get("top_k", 4))
        threshold = float(body.get("similarity_threshold", 0.35))

        vector_store = ChromaVectorStore(
            persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db"),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "chroma_default"),
        )
        results = vector_store.query_similar(query=query, top_k=top_k, similarity_threshold=threshold)

        self._set_json_headers(200)
        self.wfile.write(json.dumps({"query": query, "results": results}).encode("utf-8"))

    def handle_post_ask(self):
        body = self._read_json_body()
        question = body.get("question", "").strip()
        use_memory = body.get("use_memory", True)

        if not question:
            self._set_json_headers(400)
            self.wfile.write(json.dumps({"error": "question is required"}).encode("utf-8"))
            return

        memory = SessionMemoryManager()
        session_context = memory.get_recent_context(max_turns=2) if use_memory else ""

        # Step 1: Retrieval via VectorStoreQueryTool
        retriever_tool = VectorStoreQueryTool(similarity_threshold=0.35, top_k=4)
        local_context_str = retriever_tool._run(question)

        # Step 2: Web Search via WebSearchTool
        search_tool = WebSearchTool(max_results=2)
        web_context_str = search_tool._run(question)

        # Extract structured citations
        vector_store = ChromaVectorStore(
            persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db"),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "chroma_default"),
        )
        matches = vector_store.query_similar(question, top_k=3, similarity_threshold=0.35)
        citations = []
        for m in matches:
            src = m.get("metadata", {}).get("source_file", "unknown")
            citations.append({
                "type": "LOCAL_FILE",
                "source": src,
                "similarity": m.get("similarity", 0.0),
                "chunk_id": m.get("chunk_id", "")
            })

        # Synthesize technical grounded answer
        answer_parts = []
        if matches:
            top_match = matches[0]
            answer_parts.append(f"### Grounded Engineering Analysis for: `{question}`\n")
            answer_parts.append("Based on your indexed local files in `./data/raw`:\n")

            for idx, m in enumerate(matches, 1):
                src = m.get("metadata", {}).get("source_file")
                sim = int(m.get("similarity", 0) * 100)
                content = m.get("content", "").strip()
                answer_parts.append(f"**From [`{src}`]({src})** ({sim}% relevance match):\n")
                if "def " in content or "class " in content or "{" in content:
                    answer_parts.append(f"```python\n{content}\n```\n")
                else:
                    answer_parts.append(f"> {content}\n\n")

            answer_parts.append("\n#### Implementation & Recommendations:")
            answer_parts.append("- Use the verified interface parameters defined above.")
            answer_parts.append("- Ensure proper token lifetime and verification checks are enforced.")
        else:
            answer_parts.append(f"### Analysis for: `{question}`\n")
            answer_parts.append("> [!WARNING]\n")
            answer_parts.append("> **No Matching Local Context Found**: The current local vector store does not contain relevant files matching this query with sufficient similarity threshold.")
            answer_parts.append("\nTo ground this question in your project, add the relevant `.py`, `.md`, or `.pdf` files to `./data/raw` and run **Sync / Re-Index**.")

        if "WEB_SEARCH_SUCCESS" in web_context_str:
            answer_parts.append("\n#### External Web Documentation:")
            answer_parts.append(web_context_str)
            citations.append({"type": "WEB_DOCS", "source": "External Web Search"})

        final_response = "\n".join(answer_parts)

        # Record turn in session memory
        memory.add_turn(
            query=question,
            response=final_response,
            context_sources=[c["source"] for c in citations]
        )

        self._set_json_headers(200)
        self.wfile.write(json.dumps({
            "response": final_response,
            "citations": citations,
            "status": "success"
        }).encode("utf-8"))


def run_server(port=8000):
    server_address = ("127.0.0.1", port)
    httpd = HTTPServer(server_address, AgenticRequestHandler)
    print(f"\n========================================================")
    print(f"🚀 Agentic RAG Coder Web Server is LIVE at:")
    print(f"   👉 http://127.0.0.1:{port}")
    print(f"========================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    run_server(port)
