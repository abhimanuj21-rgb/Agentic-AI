"""Semantic text and code chunking for RAG ingestion."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from src.ingestion.loader import LoadedDocument


@dataclass
class DocumentChunk:
    """Represents a discrete semantic chunk of a document ready for embedding."""
    chunk_id: str
    content: str
    metadata: Dict[str, str] = field(default_factory=dict)


class DocumentChunker:
    """Chunks documents using semantic character splitting optimized for code and documentation."""

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text(self, text: str) -> List[str]:
        """Split text using LangChain's splitter or fallback to custom recursive splitter."""
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            separators = [
                "\n\n```",
                "\n```",
                "\nclass ",
                "\ndef ",
                "\nfunction ",
                "\nexport ",
                "\nconst ",
                "\nlet ",
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ]
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=separators,
                length_function=len,
                is_separator_regex=False,
            )
            return splitter.split_text(text)
        except ImportError:
            # Fallback pure-Python recursive chunking if langchain splitter is not installed yet
            return self._fallback_split(text)

    def _fallback_split(self, text: str) -> List[str]:
        """Custom sliding-window splitter with overlap."""
        if not text:
            return []
        chunks = []
        start = 0
        text_len = len(text)
        step = max(1, self.chunk_size - self.chunk_overlap)

        while start < text_len:
            end = min(text_len, start + self.chunk_size)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += step
        return chunks

    def chunk_document(self, doc: LoadedDocument) -> List[DocumentChunk]:
        """Split a LoadedDocument into a list of DocumentChunks with consistent IDs and metadata."""
        if not doc.content or not doc.content.strip():
            return []

        raw_chunks = self._split_text(doc.content)
        total_chunks = len(raw_chunks)
        chunks: List[DocumentChunk] = []

        file_path = doc.metadata.get("file_path", "unknown")
        filename = doc.metadata.get("filename", "unknown")

        for idx, text in enumerate(raw_chunks):
            chunk_id = f"{file_path}#chunk_{idx:04d}"
            chunk_metadata = dict(doc.metadata)
            chunk_metadata.update({
                "chunk_id": chunk_id,
                "chunk_index": str(idx),
                "total_chunks": str(total_chunks),
                "source_file": file_path,
                "filename": filename,
            })
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                content=text,
                metadata=chunk_metadata,
            ))

        return chunks

    def chunk_documents(self, docs: List[LoadedDocument]) -> List[DocumentChunk]:
        """Chunk a list of LoadedDocuments."""
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
