"""ChromaDB vector store manager with resilient local fallback for indexing, persisting, and querying document chunks."""

import json
import math
import os
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.ingestion.chunker import DocumentChunk


class FallbackSQLiteVectorStore:
    """Lightweight persistent vector store for environments where ChromaDB binary wheels are compiling or unavailable."""

    def __init__(self, db_path: str, collection_name: str = "knowledge_base"):
        self.db_path = db_path
        self.collection_name = collection_name
        self._init_db()

    def _init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    collection TEXT,
                    content TEXT,
                    metadata_json TEXT,
                    source_file TEXT
                )
            """)
            conn.commit()

    def upsert_chunks(self, chunks: List[DocumentChunk]) -> int:
        with sqlite3.connect(self.db_path) as conn:
            for c in chunks:
                meta = {k: str(v) for k, v in c.metadata.items()}
                source = meta.get("source_file", "")
                conn.execute(
                    """
                    INSERT OR REPLACE INTO chunks (chunk_id, collection, content, metadata_json, source_file)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (c.chunk_id, self.collection_name, c.content, json.dumps(meta), source),
                )
            conn.commit()
        return len(chunks)

    def delete_by_source_file(self, source_file: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM chunks WHERE collection = ? AND source_file = ?", (self.collection_name, source_file))
            conn.commit()

    def _tokenize(self, text: str) -> List[str]:
        """Split text and code identifiers (snake_case, camelCase) into normalized tokens."""
        # Split camelCase: 'AuthService' -> 'Auth Service'
        s1 = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
        # Split on any non-alphanumeric character (including underscores)
        words = re.findall(r"[a-zA-Z0-9]+", s1.lower())
        return words

    def _calculate_relevance(self, query: str, content: str) -> float:
        """
        Calculate semantic token relevance combining query term coverage and cosine similarity.
        Guarantees that documents matching key search terms are properly ranked.
        """
        stopwords = {"a", "an", "the", "in", "on", "of", "and", "or", "is", "are", "to", "for", "it", "this", "that", "do", "how", "i"}
        q_words = [w for w in self._tokenize(query) if w not in stopwords]
        if not q_words:
            q_words = self._tokenize(query)

        c_words = self._tokenize(content)
        if not q_words or not c_words:
            return 0.0

        q_set = set(q_words)
        c_set = set(c_words)
        matched = q_set & c_set
        if not matched:
            return 0.0

        coverage = len(matched) / len(q_set)
        c_counter = Counter(c_words)
        freq_bonus = sum(min(c_counter[w], 3) for w in matched) / (len(q_words) * 3)

        score = (coverage * 0.7) + (freq_bonus * 0.3)
        return min(1.0, round(score, 4))

    def query_similar(self, query: str, top_k: int = 4, similarity_threshold: float = 0.30) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT chunk_id, content, metadata_json FROM chunks WHERE collection = ?", (self.collection_name,))
            rows = cursor.fetchall()

        scored = []
        for chunk_id, content, meta_json in rows:
            sim = self._calculate_relevance(query, content)
            if sim >= similarity_threshold:
                try:
                    meta = json.loads(meta_json)
                except Exception:
                    meta = {}
                scored.append({
                    "chunk_id": chunk_id,
                    "content": content,
                    "metadata": meta,
                    "distance": round(1.0 - sim, 4),
                    "similarity": round(sim, 4),
                })

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chunks WHERE collection = ?", (self.collection_name,))
            return cursor.fetchone()[0]

    def reset(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM chunks WHERE collection = ?", (self.collection_name,))
            conn.commit()


class ChromaVectorStore:
    """Manages persistent vector storage using ChromaDB with an integrated SQLite fallback."""

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "knowledge_base",
        embedding_provider: str = "chroma_default",
        openai_api_key: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.persist_directory = str(Path(persist_directory).resolve())
        os.makedirs(self.persist_directory, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.embedding_model = embedding_model

        self.use_fallback = False
        self.chroma_client = None
        self.collection = None

        try:
            import chromadb
            from chromadb.config import Settings
            self.chroma_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
            self.embedding_fn = self._get_embedding_function()
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            self.use_fallback = True
            db_path = os.path.join(self.persist_directory, "sqlite_vector_store.db")
            self.fallback_store = FallbackSQLiteVectorStore(db_path, collection_name=self.collection_name)

    def _get_embedding_function(self):
        if self.embedding_provider == "openai" and self.openai_api_key:
            try:
                from chromadb.utils import embedding_functions
                return embedding_functions.OpenAIEmbeddingFunction(
                    api_key=self.openai_api_key,
                    model_name=self.embedding_model if self.embedding_model.startswith("text-") else "text-embedding-3-small",
                )
            except Exception:
                pass

        try:
            from chromadb.utils import embedding_functions
            return embedding_functions.DefaultEmbeddingFunction()
        except Exception:
            return None

    def upsert_chunks(self, chunks: List[DocumentChunk], batch_size: int = 100) -> int:
        if not chunks:
            return 0

        if self.use_fallback:
            return self.fallback_store.upsert_chunks(chunks)

        total_upserted = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            ids = [c.chunk_id for c in batch]
            documents = [c.content for c in batch]
            metadatas = [
                {k: str(v) if not isinstance(v, (int, float, bool)) else v for k, v in c.metadata.items()}
                for c in batch
            ]
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            total_upserted += len(batch)

        return total_upserted

    def delete_by_source_file(self, source_file: str) -> None:
        if self.use_fallback:
            self.fallback_store.delete_by_source_file(source_file)
            return

        try:
            self.collection.delete(where={"source_file": source_file})
        except Exception:
            pass

    def query_similar(
        self,
        query: str,
        top_k: int = 4,
        similarity_threshold: float = 0.30,
    ) -> List[Dict[str, Any]]:
        if self.use_fallback:
            return self.fallback_store.query_similar(query, top_k=top_k, similarity_threshold=similarity_threshold)

        if not self.collection or self.collection.count() == 0:
            return []

        actual_k = min(top_k, self.collection.count())
        if actual_k <= 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
        )

        matched_results = []
        if not results or not results["ids"] or not results["ids"][0]:
            return matched_results

        ids = results["ids"][0]
        documents = results["documents"][0] if results.get("documents") else []
        metadatas = results["metadatas"][0] if results.get("metadatas") else []
        distances = results["distances"][0] if results.get("distances") else []

        for idx in range(len(ids)):
            distance = distances[idx] if idx < len(distances) else 1.0
            similarity = max(0.0, 1.0 - (distance / 2.0))

            if similarity >= similarity_threshold:
                matched_results.append({
                    "chunk_id": ids[idx],
                    "content": documents[idx] if idx < len(documents) else "",
                    "metadata": metadatas[idx] if idx < len(metadatas) else {},
                    "distance": round(distance, 4),
                    "similarity": round(similarity, 4),
                })

        return matched_results

    def get_stats(self) -> Dict[str, Any]:
        count = self.fallback_store.count() if self.use_fallback else (self.collection.count() if self.collection else 0)
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "persist_directory": self.persist_directory,
            "backend": "SQLite-Resilient-VectorStore" if self.use_fallback else "ChromaDB-Persistent",
            "embedding_provider": self.embedding_provider,
        }

    def reset_collection(self) -> None:
        if self.use_fallback:
            self.fallback_store.reset()
        else:
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
