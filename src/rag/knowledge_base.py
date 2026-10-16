from typing import List, Dict, Optional
from datetime import datetime

from src.rag.document_loader import DocumentLoader
from src.rag.retriever import KnowledgeRetriever
from src.storage.knowledge_manager import KnowledgeManager


class APMOAKnowledgeBase:
    """
    Persistent knowledge base used by APMOA's RAG layer.

    Responsibilities:
    1. Discover documents registered by KnowledgeManager.
    2. Load their contents using DocumentLoader.
    3. Build the semantic FAISS index.
    4. Mark documents as indexed in documents.json.
    5. Provide semantic retrieval for the ReasoningAgent.
    """

    def __init__(
        self,
        knowledge_dir: str = "data/knowledge",
    ):
        self.knowledge_dir = knowledge_dir

        self.loader = DocumentLoader(knowledge_dir)
        self.retriever = KnowledgeRetriever()
        self.knowledge_manager = KnowledgeManager()

        self.documents: List[Dict[str, str]] = []

        self.enabled = True
        self.top_k = 3
        self.similarity_threshold = 0.5

        self.last_built = None

    # =========================================================
    # BUILD / UPDATE
    # =========================================================

    def build(self):
        """
        Build the RAG index using documents registered
        with KnowledgeManager.
        """

        print("\n" + "=" * 60)
        print("BUILDING APMOA KNOWLEDGE BASE")
        print("=" * 60)

        registered_documents = self.knowledge_manager.list_documents()

        print(
            f"Registered knowledge documents: "
            f"{len(registered_documents)}"
        )

        if not registered_documents:
            self.documents = []
            print("No registered knowledge documents found.")
            print("=" * 60)
            return

        # -----------------------------------------------------
        # Load managed documents
        # -----------------------------------------------------

        managed_paths = []

        for document in registered_documents:
            path = document.get("path")

            if not path:
                continue

            managed_paths.append(path)

        if not managed_paths:
            self.documents = []
            print("No valid document paths found.")
            print("=" * 60)
            return

        # -----------------------------------------------------
        # Load document chunks
        # -----------------------------------------------------

        all_documents = []

        for path in managed_paths:

            try:
                chunks = self._load_single_document(path)

                all_documents.extend(chunks)

                print(
                    f"Loaded: {path} "
                    f"-> {len(chunks)} chunks"
                )

            except Exception as e:

                print(
                    f"Warning: Could not load "
                    f"{path}: {e}"
                )

        self.documents = all_documents

        print(
            f"\nTotal chunks loaded: "
            f"{len(self.documents)}"
        )

        # -----------------------------------------------------
        # Build vector index
        # -----------------------------------------------------

        if not self.documents:
            print("No chunks available. Index not built.")
            print("=" * 60)
            return

        self.retriever.build_index(self.documents)

        self.last_built = datetime.now().isoformat()

        # -----------------------------------------------------
        # Mark documents as indexed
        # -----------------------------------------------------

        for document in registered_documents:

            document_id = document.get("id")

            if not document_id:
                continue

            document_chunks = [
                chunk
                for chunk in self.documents
                if chunk.get("document_id") == document_id
            ]

            try:
                self.knowledge_manager.mark_indexed(
                    document_id=document_id,
                    chunks=len(document_chunks),
                )

            except Exception as e:
                print(
                    f"Warning: Could not update "
                    f"metadata for {document_id}: {e}"
                )

        print("\nSemantic index built successfully.")

        print("=" * 60)

    def _load_single_document(
        self,
        path: str,
    ) -> List[Dict]:
        """
        Load a single managed document using the existing
        DocumentLoader implementation.
        """

        # DocumentLoader currently works on the knowledge
        # directory. Therefore we load the individual file
        # directly according to its supported format.

        from pathlib import Path

        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Knowledge document not found: {file_path}"
            )

        extension = file_path.suffix.lower()

        if extension in [".txt", ".md"]:
            text = file_path.read_text(
                encoding="utf-8"
            )

        elif extension == ".pdf":
            text = self._extract_pdf(file_path)

        elif extension == ".docx":
            text = self._extract_docx(file_path)

        else:
            raise ValueError(
                f"Unsupported knowledge format: {extension}"
            )

        if not text.strip():
            return []

        # -----------------------------------------------------
        # Chunk text
        # -----------------------------------------------------

        chunks = self._chunk_text(text)

        document_id = file_path.parent.name

        results = []

        for i, chunk in enumerate(chunks):

            results.append(
                {
                    "text": chunk,
                    "source": file_path.name,
                    "document_id": document_id,
                    "document_name": file_path.stem,
                    "document_type": extension.upper().replace(".", ""),
                    "chunk_id": f"{document_id}_{i}",
                    "created_at": datetime.now().isoformat(),
                }
            )

        return results

    # =========================================================
    # TEXT EXTRACTION
    # =========================================================

    @staticmethod
    def _extract_pdf(file_path):
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))

        pages = []

        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)

        return "\n".join(pages)

    @staticmethod
    def _extract_docx(file_path):
        from docx import Document

        document = Document(str(file_path))

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        return "\n".join(paragraphs)

    # =========================================================
    # CHUNKING
    # =========================================================

    @staticmethod
    def _chunk_text(
        text: str,
        chunk_size: int = 1000,
        overlap: int = 150,
    ) -> List[str]:
        """
        Simple character-based chunking.

        This keeps the implementation deterministic and
        independent of the embedding model.
        """

        text = text.strip()

        if not text:
            return []

        chunks = []

        start = 0
        text_length = len(text)

        while start < text_length:

            end = min(
                start + chunk_size,
                text_length,
            )

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            start = end - overlap

        return chunks

    # =========================================================
    # REBUILD
    # =========================================================

    def rebuild_index(self):
        self.build()

    def update_index(self):
        """
        Currently performs a full rebuild.

        This is intentional for the prototype because the
        knowledge base remains small and deterministic.
        """

        self.build()

    # =========================================================
    # DOCUMENT MANAGEMENT
    # =========================================================

    def add_document(
        self,
        filename: str,
        content: bytes,
    ):
        """
        Add a document through KnowledgeManager.
        """

        self.knowledge_manager.save_uploaded_file(
            uploaded_file=content,
            name=filename,
        )

        self.update_index()

    def get_documents(self) -> List[Dict]:
        """
        Return document-level information from the
        persistent KnowledgeManager registry.
        """

        return self.knowledge_manager.list_documents()

    def get_index_stats(self) -> Dict:
        return {
            "documents": len(
                self.knowledge_manager.list_documents()
            ),
            "chunks": len(self.documents),
            "model": (
                self.retriever.model_name
                if hasattr(
                    self.retriever,
                    "model_name",
                )
                else "unknown"
            ),
            "last_built": self.last_built,
            "status": (
                "Ready"
                if self.documents
                else "Empty"
            ),
        }

    # =========================================================
    # RETRIEVE
    # =========================================================

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[Dict]:

        if not self.enabled:
            return []

        k = top_k or self.top_k

        return self.retriever.search(
            query=query,
            top_k=k,
            threshold=self.similarity_threshold,
        )

    def search(
        self,
        query: str,
    ) -> List[Dict]:

        return self.retrieve(query)

    # =========================================================
    # CONTEXT
    # =========================================================

    def get_context(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> str:

        results = self.retrieve(
            query=query,
            top_k=top_k,
        )

        if not results:
            return ""

        context_parts = []

        for result in results:

            context_parts.append(
                f"Source: "
                f"{result.get('source', 'unknown')}\n"
                f"Relevance: "
                f"{result.get('score', 0):.4f}\n"
                f"{result.get('text', '')}"
            )

        return "\n\n".join(context_parts)