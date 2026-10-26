"""Standalone isolation test for VectorStoreQueryTool and Retriever Agent."""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools.vector_query_tool import VectorStoreQueryTool
from src.ingestion.vector_store import ChromaVectorStore
from src.ingestion.chunker import DocumentChunk


def test_vector_query_tool_isolation():
    print("=" * 60)
    print("TESTING VECTOR STORE QUERY TOOL IN ISOLATION")
    print("=" * 60)

    # 1. Setup sample data in vector store
    store = ChromaVectorStore(
        persist_directory="./data/chroma_db",
        collection_name="test_retriever_collection",
        embedding_provider="chroma_default",
    )

    test_chunk = DocumentChunk(
        chunk_id="test_service.py#chunk_0001",
        content="def hash_password(password: str) -> str:\n    return sha256(password.encode()).hexdigest()",
        metadata={
            "source_file": "services/auth.py",
            "filename": "auth.py",
            "extension": ".py",
        },
    )
    store.upsert_chunks([test_chunk])

    # 2. Test query tool with matching query
    tool = VectorStoreQueryTool(
        persist_directory="./data/chroma_db",
        collection_name="test_retriever_collection",
        top_k=2,
        similarity_threshold=0.5,
    )

    output = tool._run("How do I hash a password?")
    print("\n--- Match Test Output ---")
    print(output)
    assert "LOCAL_CONTEXT_FOUND" in output, "Tool should find local context for password hashing."
    assert "services/auth.py" in output, "Output should cite services/auth.py."

    # 3. Test query tool with non-matching query (Guardrail check)
    output_no_match = tool._run("How do I bake a chocolate cake?")
    print("\n--- Non-Match Guardrail Test Output ---")
    print(output_no_match)
    assert "NO_LOCAL_CONTEXT_FOUND" in output_no_match, "Tool should report no context for unrelated queries."

    print("\n[SUCCESS] VectorStoreQueryTool isolation test passed!\n")


if __name__ == "__main__":
    test_vector_query_tool_isolation()
