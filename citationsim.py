import re
import numpy as np
from sentence_transformers import SentenceTransformer
from vertexai.language_models import TextEmbeddingModel
from typing import Optional

# Default similarity threshold
SIMILARITY_THRESHOLD = 0.75

# Task-based model mapping
MODEL_MAP = {
    "text": "all-mpnet-base-v2",
    "scientific": "allenai/scibert_scivocab_uncased",
    "code": "microsoft/codebert-base"
}

# Cache for models to avoid reloading them multiple times
_models: dict[str, SentenceTransformer] = {}

class CitationValidator:
    def __init__(
        self,
        task_type: str = "auto",
        threshold: float = SIMILARITY_THRESHOLD,
        use_vertex_model: bool = False,
        vertex_model_name: str = "text-embedding-005",
        vertex_credentials_path: Optional[str] = None
    ):
        self.task_type = task_type
        self.threshold = threshold
        self.use_vertex_model = use_vertex_model
        self.vertex_model_name = vertex_model_name
        self.vertex_credentials_path = vertex_credentials_path
        self.model = None

    def initialize_model(self, summary: str) -> None:
        print("Initializing model...")

        if self.use_vertex_model:
            if not _VERTEX_AVAILABLE:
                raise ImportError("vertexai module not found. Please install and authenticate.")

            if self.vertex_credentials_path:
                import os
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.vertex_credentials_path  # Set credentials path

            self.model = TextEmbeddingModel.from_pretrained(self.vertex_model_name)
        else:
            task = self.task_type if self.task_type != "auto" else detect_task(summary)
            model_name = MODEL_MAP.get(task, MODEL_MAP["text"])
            self.model = SentenceTransformer(model_name)

    def _compute_similarity_scores(self, query_vec: np.ndarray, citation_vecs: np.ndarray) -> np.ndarray:
        """
        Computes cosine similarity between the query and candidate citation vectors.
        """
        return citation_vecs.dot(query_vec)

    def validate(self, summary: str, citations: list[str]) -> list[dict]:
        """
        Validate if each citation semantically supports the summary.
        
        Args:
            summary: LLM-generated summary/claim.
            citations: List of candidate citation texts.
        
        Returns:
            A list of dictionaries containing the citation, similarity score, and support status.
        """
        if not self.model:
            raise ValueError("Model has not been initialized. Call 'initialize_model()' first.")

        summary_vec = self.model.encode([summary], normalize_embeddings=True)[0]
        citation_vecs = self.model.encode(citations, normalize_embeddings=True)
        scores = self._compute_similarity_scores(summary_vec, citation_vecs)

        return [
            {
                "citation": cite,
                "score": float(score),
                "supported": bool(score >= self.threshold)
            }
            for cite, score in zip(citations, scores)
        ]

# Example usage:
if __name__ == "__main__":
    # First, initialize the model with a specified task type
    validator = CitationValidator(task_type="text")  # Example: text, scientific, or code
    validator.initialize_model()

    # Then, you can perform multiple validation calls
    citations = [
        "Transformer models rely on self-attention to handle long-range dependencies.",
        "RNNs process sequences step by step without explicit attention mechanisms.",
        "CNNs are mainly used for image data."
    ]
    summary = "Transformers introduced self-attention for context-aware NLP models."

    results = validator.validate(summary, citations)
    for r in results:
        status = "✅ Supported" if r["supported"] else "❌ Flagged"
        print(f"{status} | {r['score']:.3f} | {r['citation']}")
