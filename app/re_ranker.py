"""
Re-ranking module using cross-encoder models.
Re-orders documents by jointly encoding query + document pairs for more accurate relevance scoring.
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional


class CrossEncoderReRanker:
    """
    Re-ranks documents using a cross-encoder model.
    
    Cross-encoders process the query and document TOGETHER (as a pair),
    which is slower but more accurate than bi-encoders (like sentence-transformers).
    
    Uses the ms-marco-MiniLM-L-6-v2 model, which is lightweight and optimized
    for document retrieval relevance scoring.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the cross-encoder re-ranker.
        
        Args:
            model_name: Name of the cross-encoder model.
                        Default: cross-encoder/ms-marco-MiniLM-L-6-v2
                        (~80MB, fast inference, good for re-ranking)
        """
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        """Lazy-load the model on first use."""
        if self.model is None:
            try:
                from sentence_transformers import CrossEncoder
                print(f"📥 Loading cross-encoder re-ranker ({self.model_name})...")
                self.model = CrossEncoder(self.model_name)
                print("✅ Re-ranker loaded")
            except ImportError:
                print("⚠️  sentence-transformers not installed. Re-ranking disabled.")
                print("   Install with: pip install sentence-transformers")
                return False
            except Exception as e:
                print(f"⚠️  Failed to load re-ranker: {e}")
                return False
        return True

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents by relevance to the query using cross-encoder scores.
        
        Args:
            query: The user's question
            documents: List of document dictionaries (must have 'content' field)
            top_k: Number of top documents to return
            
        Returns:
            Re-ordered list of documents with '_rerank_score' field added
        """
        self._load_model()

        if not documents:
            return documents

        # Create query-document pairs (truncate content to avoid token limits)
        pairs = [[query, doc.get("content", "")[:512]] for doc in documents]

        # Get relevance scores from cross-encoder
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            print(f"⚠️  Re-ranking failed: {e}. Returning original order.")
            return documents

        # Add scores and re-sort by relevance
        for i, doc in enumerate(documents):
            doc["_rerank_score"] = float(scores[i])

        documents.sort(key=lambda x: x["_rerank_score"], reverse=True)
        
        # Return top-k re-ranked documents
        return documents[:top_k]


# Singleton instance for reuse across the application
_reranker_instance = None


def get_reranker() -> CrossEncoderReRanker:
    """Get or create the singleton re-ranker instance."""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = CrossEncoderReRanker()
    return _reranker_instance