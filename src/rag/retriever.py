from typing import List, Dict, Optional

import faiss
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class KnowledgeRetriever:
    """
    Semantic retriever for the APMOA knowledge base.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.model_name = model_name
        self.model = None
        if SentenceTransformer:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                print(f"Warning: Could not load SentenceTransformer model '{model_name}': {e}. Continuing without embeddings.")

        self.documents: List[Dict[str, str]] = []
        self.index = None

    # =========================================================
    # BUILD INDEX
    # =========================================================

    def build_index(
        self,
        documents: List[Dict[str, str]],
    ):
        if not documents:
            raise ValueError(
                "Cannot build RAG index from empty documents."
            )

        self.documents = documents

        if not self.model:
            print("sentence_transformers not available. Skipping embedding.")
            return

        texts = [
            document["text"]
            for document in documents
        ]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    # =========================================================
    # SEARCH
    # =========================================================

    def search(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = 0.0
    ) -> List[Dict]:

        if not query or not query.strip():
            raise ValueError(
                "Retrieval query cannot be empty."
            )

        if self.index is None or not self.model:
            # Fallback if no model or index
            return []

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        scores, indices = self.index.search(
            query_embedding,
            min(top_k * 2, len(self.documents)), # Retrieve more to filter by threshold
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0 or float(score) < threshold:
                continue

            document = self.documents[index]

            results.append(
                {
                    "source": document.get("source", "unknown"),
                    "text": document.get("text", ""),
                    "score": float(score),
                    "chunk_id": document.get("chunk_id", ""),
                    "document_type": document.get("document_type", "TXT"),
                    "document_name": document.get("document_name", "unknown")
                }
            )

            if len(results) >= top_k:
                break

        return results
