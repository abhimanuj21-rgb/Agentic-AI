"""Standalone test script for the ingestion pipeline (loader -> chunker -> ChromaDB)."""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.loader import LocalDirectoryLoader, save_hash_manifest
from src.ingestion.chunker import DocumentChunker
from src.ingestion.vector_store import ChromaVectorStore


def run_ingestion_test():
    print("=" * 60)
    print("RUNNING INGESTION PIPELINE STANDALONE TEST")
    print("=" * 60)

    test_raw_dir = Path("./data/raw")
    test_raw_dir.mkdir(parents=True, exist_ok=True)

    test_manifest = Path("./data/chroma_db/test_file_hashes.json")
    if test_manifest.exists():
        test_manifest.unlink()

    # 1. Create sample test documents if empty
    sample_py = test_raw_dir / "sample_service.py"
    sample_md = test_raw_dir / "api_guide.md"

    if not sample_py.exists():
        sample_py.write_text(
            '"""User Authentication and Token Service."""\n\n'
            'class AuthService:\n'
            '    def __init__(self, secret_key: str):\n'
            '        self.secret_key = secret_key\n\n'
            '    def generate_jwt(self, user_id: str) -> str:\n'
            '        """Generate signed JWT token for user."""\n'
            '        return f"token_for_{user_id}_{self.secret_key[:4]}"\n\n'
            '    def verify_token(self, token: str) -> bool:\n'
            '        """Verify token authenticity."""\n'
            '        return token.startswith("token_for_")\n',
            encoding="utf-8",
        )
        print(f"Created sample test file: {sample_py}")

    if not sample_md.exists():
        sample_md.write_text(
            '# API Authentication Guide\n\n'
            'Our system uses the AuthService for bearer token generation.\n\n'
            '## Security Requirements\n'
            '- Secret keys must be at least 32 characters long.\n'
            '- Tokens expire after 60 minutes.\n'
            '- Always verify tokens using the `verify_token` method.\n',
            encoding="utf-8",
        )
        print(f"Created sample test file: {sample_md}")

    # 2. Test Loader
    print("\n--- STEP 1: Testing Directory Loader ---")
    loader = LocalDirectoryLoader(
        base_path=str(test_raw_dir),
        manifest_path="./data/chroma_db/test_file_hashes.json",
    )
    docs, deleted, manifest = loader.load_incremental()
    print(f"Discovered new/modified docs: {len(docs)}")
    print(f"Detected deleted files: {len(deleted)}")
    for d in docs:
        print(f"  - Loaded: {d.metadata['file_path']} ({d.metadata['file_size']} bytes)")

    assert len(docs) >= 2, "Expected at least 2 sample documents to be loaded."

    # 3. Test Chunker
    print("\n--- STEP 2: Testing Document Chunker ---")
    chunker = DocumentChunker(chunk_size=400, chunk_overlap=50)
    chunks = chunker.chunk_documents(docs)
    print(f"Generated semantic chunks: {len(chunks)}")
    for c in chunks[:3]:
        print(f"  - Chunk ID: {c.chunk_id} | Length: {len(c.content)} chars")

    assert len(chunks) >= 2, "Expected chunks to be generated."

    # 4. Test Chroma Vector Store
    print("\n--- STEP 3: Testing Chroma Vector Store Upsert ---")
    vector_store = ChromaVectorStore(
        persist_directory="./data/chroma_db",
        collection_name="test_knowledge_base",
        embedding_provider="chroma_default",
    )
    upserted_count = vector_store.upsert_chunks(chunks)
    save_hash_manifest("./data/chroma_db/test_file_hashes.json", manifest)
    print(f"Upserted chunks into ChromaDB: {upserted_count}")

    stats = vector_store.get_stats()
    print(f"Vector Store Stats: {stats}")
    assert stats["total_chunks"] >= upserted_count

    # 5. Test Incremental Indexing (Re-running with unchanged files)
    print("\n--- STEP 4: Testing Incremental Hash Detection ---")
    loader_second_run = LocalDirectoryLoader(
        base_path=str(test_raw_dir),
        manifest_path="./data/chroma_db/test_file_hashes.json",
    )
    docs_2nd, _, _ = loader_second_run.load_incremental()
    print(f"Second run new/modified docs count: {len(docs_2nd)} (Expected: 0)")
    assert len(docs_2nd) == 0, "Incremental indexing failed: unchanged files were re-read!"

    # 6. Test Semantic Querying
    print("\n--- STEP 5: Testing Semantic Similarity Query ---")
    query = "How do I generate a JWT token and verify it?"
    print(f"Query: '{query}'")
    matches = vector_store.query_similar(query, top_k=2, similarity_threshold=0.5)
    print(f"Found {len(matches)} matching chunks:")
    for m in matches:
        print(f"  - [{m['similarity']:.2f}] {m['metadata'].get('source_file')} -> {m['content'][:80]}...")

    assert len(matches) > 0, "Expected semantic matches for query."
    print("\n[SUCCESS] Ingestion pipeline test passed completely!\n")


if __name__ == "__main__":
    run_ingestion_test()
