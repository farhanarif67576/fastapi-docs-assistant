"""
Minimal search index for text fields with keyword filtering and field boosting.
Vendored from the minsearch package used in LLM Zoomcamp.
"""

import math
import pandas as pd
from typing import List, Dict, Optional, Any


class Index:
    """
    A simple in-memory search index that supports:
    - Text fields (TF-IDF based search)
    - Keyword fields (exact match filtering)
    - Field boosting
    - Filtering
    """

    def __init__(self, text_fields: List[str], keyword_fields: List[str]):
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields
        self.documents = []
        self.df = None
        self.idf = {}
        self.doc_vectors = []

    def fit(self, documents: List[Dict[str, Any]]):
        """Build the index from a list of document dictionaries."""
        self.documents = documents
        self.df = pd.DataFrame(documents)

        # Build IDF
        n_docs = len(documents)
        for field in self.text_fields:
            doc_containing = 0
            for doc in documents:
                val = str(doc.get(field, ""))
                if val.strip():
                    doc_containing += 1
            self.idf[field] = math.log((n_docs + 1) / (doc_containing + 1)) + 1

        # Build document vectors (term frequency per field)
        self.doc_vectors = []
        for doc in documents:
            vec = {}
            for field in self.text_fields:
                val = str(doc.get(field, ""))
                terms = val.lower().split()
                for term in terms:
                    if term not in vec:
                        vec[term] = {}
                    if field not in vec[term]:
                        vec[term][field] = 0
                    vec[term][field] += 1
            self.doc_vectors.append(vec)

    def search(
        self,
        query: str,
        filter_dict: Dict[str, Any] = None,
        boost_dict: Dict[str, float] = None,
        num_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search the index for documents matching the query.
        
        Args:
            query: The search query string
            filter_dict: Dictionary of keyword field filters (e.g., {"category": "tutorial"})
            boost_dict: Dictionary of field boost values (e.g., {"title": 3.0, "content": 1.0})
            num_results: Number of results to return
            
        Returns:
            List of document dictionaries with scores
        """
        if filter_dict is None:
            filter_dict = {}
        if boost_dict is None:
            boost_dict = {}

        # Tokenize query
        query_terms = query.lower().split()

        # Score each document
        results = []
        for idx, doc in enumerate(self.documents):
            # Apply keyword filters
            matches_filter = True
            for field, value in filter_dict.items():
                if str(doc.get(field, "")).lower() != str(value).lower():
                    matches_filter = False
                    break
            if not matches_filter:
                continue

            # Calculate TF-IDF score
            score = 0.0
            for term in query_terms:
                if term in self.doc_vectors[idx]:
                    for field in self.text_fields:
                        tf = self.doc_vectors[idx][term].get(field, 0)
                        if tf > 0:
                            idf = self.idf.get(field, 1.0)
                            boost = boost_dict.get(field, 1.0)
                            score += (1 + math.log(tf)) * idf * boost

            results.append((doc, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top-k results with scores
        output = []
        for doc, score in results[:num_results]:
            doc_copy = doc.copy()
            doc_copy["_score"] = score
            output.append(doc_copy)

        return output