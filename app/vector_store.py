"""
Vector embedding module using sentence-transformers with ONNX runtime.
Generates embeddings for document chunks and user queries locally.
"""

import numpy as np
from typing import List, Union


class ONNXEmbedder:
    """
    Local embedding model using sentence-transformers.
    Generates 384-dimensional vectors using all-MiniLM-L6-v2.
    No API calls needed — runs entirely on CPU.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                        Default: all-MiniLM-L6-v2 (384-dim, fast, good quality)
        """
        self.model_name = model_name
        self.model = None
        self.dimension = 384  # all-MiniLM-L6-v2 produces 384-dim vectors

    def _load_model(self):
        """Lazy-load the model on first use."""
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for one or more text strings.
        
        Args:
            texts: A single string or a list of strings to embed
            
        Returns:
            numpy array of shape (n_texts, 384) with normalized embeddings
        """
        self._load_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query string.
        
        Args:
            query: The search query string
            
        Returns:
            numpy array of shape (384,) with normalized embedding
        """
        return self.embed(query)[0]


# Singleton instance for reuse across the application
_embedder_instance = None


def get_embedder() -> ONNXEmbedder:
    """Get or create the singleton ONNX embedder instance."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = ONNXEmbedder()
    return _embedder_instance