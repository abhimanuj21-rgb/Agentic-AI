"""VectorStoreQueryTool for retrieving grounded context from ChromaDB."""

import os
from typing import Any, Dict, List, Optional, Type
try:
    from pydantic import BaseModel, Field
except ImportError:
    class BaseModel:
        pass
    def Field(*args, **kwargs):
        return None

# Try importing CrewAI BaseTool, fallback to a standalone BaseTool if not installed yet
try:
    from crewai.tools import BaseTool
except ImportError:
    class BaseTool:  # Fallback base class
        name: str = ""
        description: str = ""
        args_schema: Optional[Type[BaseModel]] = None

        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

        def _run(self, *args, **kwargs):
            raise NotImplementedError


class VectorQueryInput(BaseModel):
    """Input schema for VectorStoreQueryTool."""
    query: str = Field(..., description="The coding question or semantic search query to look up in the local codebase.")


class VectorStoreQueryTool(BaseTool):
    """CrewAI tool for querying the local ChromaDB vector store."""
    name: str = "vector_store_query_tool"
    description: str = (
        "Search the local knowledge base (codebase, documentation, and notes) "
        "stored in ChromaDB. Returns top matching snippets with file paths, "
        "similarity scores, and metadata."
    )
    args_schema: Type[BaseModel] = VectorQueryInput

    persist_directory: str = "./data/chroma_db"
    collection_name: str = "knowledge_base"
    top_k: int = 4
    similarity_threshold: float = 0.35

    def _run(self, query: str) -> str:
        """Execute semantic search against local ChromaDB."""
        try:
            from src.ingestion.vector_store import ChromaVectorStore

            embedding_provider = os.getenv("EMBEDDING_PROVIDER", "chroma_default")
            embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
            env_thresh = os.getenv("SIMILARITY_THRESHOLD")
            thresh = float(env_thresh) if env_thresh else self.similarity_threshold

            store = ChromaVectorStore(
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
            )

            results = store.query_similar(
                query=query,
                top_k=self.top_k,
                similarity_threshold=thresh,
            )

            if not results:
                return (
                    "STATUS: NO_LOCAL_CONTEXT_FOUND\n"
                    f"No local documents or code chunks matched the query '{query}' "
                    f"with similarity threshold >= {self.similarity_threshold}."
                )

            formatted_output = [
                f"STATUS: LOCAL_CONTEXT_FOUND (Retrieved {len(results)} relevant chunks)\n"
            ]

            for i, item in enumerate(results, 1):
                meta = item.get("metadata", {})
                source = meta.get("source_file", "unknown")
                chunk_id = item.get("chunk_id", f"chunk_{i}")
                sim = item.get("similarity", 0.0)
                content = item.get("content", "").strip()

                formatted_output.append(
                    f"--- [Result {i}] Source: {source} (Chunk: {chunk_id}, Relevance: {sim:.2f}) ---\n"
                    f"{content}\n"
                )

            return "\n".join(formatted_output)

        except Exception as e:
            return f"STATUS: ERROR\nError querying local vector store: {str(e)}"
