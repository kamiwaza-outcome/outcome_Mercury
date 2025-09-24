"""
Hybrid Search Implementation combining Milvus vector search with BM25 keyword search
using Reciprocal Rank Fusion (RRF) for optimal results.
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import json
import re
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import hashlib

# BM25 implementations
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    import bm25s
except ImportError:
    bm25s = None

# Text processing
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
except ImportError:
    nltk = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a search result with content, metadata, and scores."""
    content: str
    metadata: Dict[str, Any]
    vector_score: float = 0.0
    bm25_score: float = 0.0
    combined_score: float = 0.0
    rank_vector: int = 0
    rank_bm25: int = 0
    doc_id: str = ""

class TextPreprocessor:
    """Handles text preprocessing for BM25 search."""

    def __init__(self):
        self.stemmer = PorterStemmer() if nltk else None
        self.stop_words = set()
        self._initialize_nltk()

    def _initialize_nltk(self):
        """Initialize NLTK resources."""
        if nltk:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                self.stop_words = set(stopwords.words('english'))
            except Exception as e:
                logger.warning(f"Could not initialize NLTK: {e}")

    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25 indexing."""
        if not text:
            return []

        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())

        # Tokenize
        if nltk:
            try:
                tokens = word_tokenize(text)
            except:
                tokens = text.split()
        else:
            tokens = text.split()

        # Remove stopwords and stem
        processed_tokens = []
        for token in tokens:
            if len(token) > 2 and token not in self.stop_words:
                if self.stemmer:
                    try:
                        token = self.stemmer.stem(token)
                    except:
                        pass
                processed_tokens.append(token)

        return processed_tokens

class BM25Search:
    """BM25 keyword search implementation with multiple backend options."""

    def __init__(self, use_bm25s: bool = True):
        self.use_bm25s = use_bm25s and bm25s is not None
        self.preprocessor = TextPreprocessor()
        self.bm25_index = None
        self.documents = []
        self.tokenized_docs = []
        self.doc_metadata = []

    async def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents for BM25 search."""
        try:
            self.documents = documents
            self.doc_metadata = [doc.get('metadata', {}) for doc in documents]

            # Preprocess documents
            self.tokenized_docs = []
            for doc in documents:
                content = doc.get('content', '')
                tokens = self.preprocessor.preprocess_text(content)
                self.tokenized_docs.append(tokens)

            # Build BM25 index
            if self.use_bm25s:
                self._build_bm25s_index()
            else:
                self._build_rank_bm25_index()

            logger.info(f"Indexed {len(documents)} documents for BM25 search")

        except Exception as e:
            logger.error(f"Error indexing documents for BM25: {e}")
            raise

    def _build_bm25s_index(self):
        """Build index using bm25s library."""
        try:
            # Create corpus for bm25s
            corpus = [' '.join(tokens) for tokens in self.tokenized_docs]

            # Create and fit BM25 index
            self.bm25_index = bm25s.BM25()
            self.bm25_index.fit(corpus)

        except Exception as e:
            logger.error(f"Error building bm25s index: {e}")
            # Fallback to rank-bm25
            self.use_bm25s = False
            self._build_rank_bm25_index()

    def _build_rank_bm25_index(self):
        """Build index using rank-bm25 library."""
        if BM25Okapi is None:
            raise ImportError("rank-bm25 library not installed")

        self.bm25_index = BM25Okapi(self.tokenized_docs)

    async def search(self, query: str, limit: int = 10) -> List[Tuple[int, float]]:
        """Search using BM25 and return document indices with scores."""
        try:
            if self.bm25_index is None:
                return []

            # Preprocess query
            query_tokens = self.preprocessor.preprocess_text(query)
            if not query_tokens:
                return []

            if self.use_bm25s:
                return self._search_bm25s(query_tokens, limit)
            else:
                return self._search_rank_bm25(query_tokens, limit)

        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []

    def _search_bm25s(self, query_tokens: List[str], limit: int) -> List[Tuple[int, float]]:
        """Search using bm25s library."""
        query_str = ' '.join(query_tokens)
        scores = self.bm25_index.query(query_str, k=limit)

        # Convert to list of (index, score) tuples
        results = []
        for i, score in enumerate(scores):
            if i < len(self.documents):
                results.append((i, float(score)))

        return sorted(results, key=lambda x: x[1], reverse=True)[:limit]

    def _search_rank_bm25(self, query_tokens: List[str], limit: int) -> List[Tuple[int, float]]:
        """Search using rank-bm25 library."""
        scores = self.bm25_index.get_scores(query_tokens)

        # Get top results
        doc_scores = [(i, score) for i, score in enumerate(scores)]
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return doc_scores[:limit]

class ReciprocalRankFusion:
    """Implements Reciprocal Rank Fusion (RRF) algorithm for combining rankings."""

    @staticmethod
    def fuse_rankings(
        rankings: List[List[Tuple[int, float]]],
        k: int = 60,
        weights: Optional[List[float]] = None
    ) -> List[Tuple[int, float]]:
        """
        Fuse multiple rankings using RRF algorithm.

        Args:
            rankings: List of rankings, each containing (doc_id, score) tuples
            k: RRF constant (default: 60)
            weights: Optional weights for each ranking system

        Returns:
            List of (doc_id, combined_score) tuples sorted by combined score
        """
        if not rankings:
            return []

        if weights is None:
            weights = [1.0] * len(rankings)

        # Initialize score dictionary
        rrf_scores = defaultdict(float)

        # Apply RRF formula for each ranking
        for ranking_idx, ranking in enumerate(rankings):
            weight = weights[ranking_idx]
            for rank, (doc_id, _) in enumerate(ranking, 1):
                rrf_scores[doc_id] += weight * (1.0 / (k + rank))

        # Sort by combined RRF score
        fused_results = [(doc_id, score) for doc_id, score in rrf_scores.items()]
        fused_results.sort(key=lambda x: x[1], reverse=True)

        return fused_results

class HybridSearch:
    """
    Hybrid search implementation combining Milvus vector search with BM25 keyword search.
    Uses Reciprocal Rank Fusion (RRF) to combine results optimally.
    """

    def __init__(self, milvus_rag, vector_weight: float = 0.6, bm25_weight: float = 0.4):
        """
        Initialize hybrid search.

        Args:
            milvus_rag: MilvusRAG instance for vector search
            vector_weight: Weight for vector search (default: 0.6)
            bm25_weight: Weight for BM25 search (default: 0.4)
        """
        self.milvus_rag = milvus_rag
        self.bm25_search = BM25Search()
        self.rrf = ReciprocalRankFusion()

        # Normalize weights
        total_weight = vector_weight + bm25_weight
        self.vector_weight = vector_weight / total_weight
        self.bm25_weight = bm25_weight / total_weight

        # Document storage for BM25
        self.indexed_documents = []
        self.doc_id_to_index = {}

        # Score normalization
        self.vector_scaler = MinMaxScaler()
        self.bm25_scaler = MinMaxScaler()

        logger.info(f"Initialized hybrid search with weights - Vector: {self.vector_weight:.2f}, BM25: {self.bm25_weight:.2f}")

    async def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents for both vector and BM25 search."""
        try:
            # Store documents for reference
            self.indexed_documents = documents

            # Create document ID mapping
            for i, doc in enumerate(documents):
                doc_hash = hashlib.md5(doc.get('content', '').encode()).hexdigest()
                self.doc_id_to_index[doc_hash] = i

            # Index in BM25
            await self.bm25_search.index_documents(documents)

            # Index in Milvus (assuming documents are already indexed)
            logger.info(f"Indexed {len(documents)} documents for hybrid search")

        except Exception as e:
            logger.error(f"Error indexing documents for hybrid search: {e}")
            raise

    async def search(
        self,
        query: str,
        limit: int = 10,
        vector_limit: int = 20,
        bm25_limit: int = 20,
        rrf_k: int = 60,
        use_rrf: bool = True
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining vector and BM25 results.

        Args:
            query: Search query
            limit: Final number of results to return
            vector_limit: Number of results from vector search
            bm25_limit: Number of results from BM25 search
            rrf_k: RRF constant for rank fusion
            use_rrf: Whether to use RRF or weighted scoring

        Returns:
            List of SearchResult objects sorted by combined score
        """
        try:
            # Perform both searches concurrently
            vector_task = self._vector_search(query, vector_limit)
            bm25_task = self._bm25_search(query, bm25_limit)

            vector_results, bm25_results = await asyncio.gather(vector_task, bm25_task)

            if use_rrf:
                return await self._combine_with_rrf(
                    vector_results, bm25_results, limit, rrf_k
                )
            else:
                return await self._combine_with_weights(
                    vector_results, bm25_results, limit
                )

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return []

    async def _vector_search(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Perform vector search using Milvus."""
        try:
            results = await self.milvus_rag.search_similar(query, limit)
            return results
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []

    async def _bm25_search(self, query: str, limit: int) -> List[Tuple[int, float]]:
        """Perform BM25 search."""
        try:
            results = await self.bm25_search.search(query, limit)
            return results
        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []

    async def _combine_with_rrf(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Tuple[int, float]],
        limit: int,
        rrf_k: int
    ) -> List[SearchResult]:
        """Combine results using Reciprocal Rank Fusion."""
        try:
            # Prepare rankings for RRF
            vector_ranking = []
            bm25_ranking = []

            # Convert vector results to ranking format
            for i, result in enumerate(vector_results):
                content = result.get('content', '')
                doc_hash = hashlib.md5(content.encode()).hexdigest()
                if doc_hash in self.doc_id_to_index:
                    doc_idx = self.doc_id_to_index[doc_hash]
                    vector_ranking.append((doc_idx, result.get('distance', 0.0)))

            # BM25 results are already in the right format
            bm25_ranking = bm25_results

            # Apply RRF
            fused_results = self.rrf.fuse_rankings(
                [vector_ranking, bm25_ranking],
                k=rrf_k,
                weights=[self.vector_weight, self.bm25_weight]
            )

            # Convert to SearchResult objects
            search_results = []
            for doc_idx, rrf_score in fused_results[:limit]:
                if doc_idx < len(self.indexed_documents):
                    doc = self.indexed_documents[doc_idx]

                    # Find original scores
                    vector_score = 0.0
                    bm25_score = 0.0

                    for v_idx, v_score in vector_ranking:
                        if v_idx == doc_idx:
                            vector_score = 1.0 - v_score  # Convert distance to similarity
                            break

                    for b_idx, b_score in bm25_ranking:
                        if b_idx == doc_idx:
                            bm25_score = b_score
                            break

                    result = SearchResult(
                        content=doc.get('content', ''),
                        metadata=doc.get('metadata', {}),
                        vector_score=vector_score,
                        bm25_score=bm25_score,
                        combined_score=rrf_score,
                        doc_id=str(doc_idx)
                    )
                    search_results.append(result)

            return search_results

        except Exception as e:
            logger.error(f"Error combining results with RRF: {e}")
            return []

    async def _combine_with_weights(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Tuple[int, float]],
        limit: int
    ) -> List[SearchResult]:
        """Combine results using weighted scoring."""
        try:
            # Normalize scores
            if vector_results:
                vector_scores = [1.0 - result.get('distance', 0.0) for result in vector_results]
                vector_scores_norm = self._normalize_scores(vector_scores)
            else:
                vector_scores_norm = []

            if bm25_results:
                bm25_scores = [score for _, score in bm25_results]
                bm25_scores_norm = self._normalize_scores(bm25_scores)
            else:
                bm25_scores_norm = []

            # Create combined results
            all_results = {}

            # Add vector results
            for i, result in enumerate(vector_results):
                content = result.get('content', '')
                doc_hash = hashlib.md5(content.encode()).hexdigest()

                score = vector_scores_norm[i] if i < len(vector_scores_norm) else 0.0

                all_results[doc_hash] = SearchResult(
                    content=content,
                    metadata=result.get('metadata', {}),
                    vector_score=score,
                    bm25_score=0.0,
                    combined_score=self.vector_weight * score,
                    doc_id=doc_hash
                )

            # Add BM25 results
            for i, (doc_idx, _) in enumerate(bm25_results):
                if doc_idx < len(self.indexed_documents):
                    doc = self.indexed_documents[doc_idx]
                    content = doc.get('content', '')
                    doc_hash = hashlib.md5(content.encode()).hexdigest()

                    bm25_score = bm25_scores_norm[i] if i < len(bm25_scores_norm) else 0.0

                    if doc_hash in all_results:
                        all_results[doc_hash].bm25_score = bm25_score
                        all_results[doc_hash].combined_score += self.bm25_weight * bm25_score
                    else:
                        all_results[doc_hash] = SearchResult(
                            content=content,
                            metadata=doc.get('metadata', {}),
                            vector_score=0.0,
                            bm25_score=bm25_score,
                            combined_score=self.bm25_weight * bm25_score,
                            doc_id=doc_hash
                        )

            # Sort by combined score and return top results
            sorted_results = sorted(
                all_results.values(),
                key=lambda x: x.combined_score,
                reverse=True
            )

            return sorted_results[:limit]

        except Exception as e:
            logger.error(f"Error combining results with weights: {e}")
            return []

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range."""
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return [1.0] * len(scores)

        return [(score - min_score) / (max_score - min_score) for score in scores]

    async def adaptive_search(
        self,
        query: str,
        limit: int = 10,
        query_type: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Perform adaptive hybrid search with dynamic weight adjustment.

        Args:
            query: Search query
            limit: Number of results to return
            query_type: Optional query type hint ('navigational', 'informational', 'transactional')

        Returns:
            List of SearchResult objects
        """
        # Analyze query to determine optimal weights
        vector_weight, bm25_weight = self._analyze_query_for_weights(query, query_type)

        # Temporarily adjust weights
        original_vector_weight = self.vector_weight
        original_bm25_weight = self.bm25_weight

        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

        try:
            results = await self.search(query, limit)
        finally:
            # Restore original weights
            self.vector_weight = original_vector_weight
            self.bm25_weight = original_bm25_weight

        return results

    def _analyze_query_for_weights(
        self,
        query: str,
        query_type: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Analyze query characteristics to determine optimal weights.

        Returns:
            Tuple of (vector_weight, bm25_weight)
        """
        # Default weights
        vector_w = 0.6
        bm25_w = 0.4

        # Adjust based on query type if provided
        if query_type == 'navigational':
            # Favor keyword search for specific item lookup
            vector_w = 0.3
            bm25_w = 0.7
        elif query_type == 'informational':
            # Favor semantic search for conceptual queries
            vector_w = 0.8
            bm25_w = 0.2
        else:
            # Analyze query characteristics
            query_lower = query.lower()

            # Count specific patterns
            has_quotes = '"' in query
            has_exact_terms = any(term in query_lower for term in ['exactly', 'specifically', 'literal'])
            has_conceptual_terms = any(term in query_lower for term in ['similar', 'like', 'about', 'related'])
            is_short_query = len(query.split()) <= 3
            has_technical_terms = any(term in query_lower for term in ['api', 'function', 'class', 'method'])

            # Adjust weights based on patterns
            if has_quotes or has_exact_terms or is_short_query:
                # Favor BM25 for exact matches
                vector_w = 0.4
                bm25_w = 0.6
            elif has_conceptual_terms:
                # Favor vector search for conceptual queries
                vector_w = 0.7
                bm25_w = 0.3
            elif has_technical_terms:
                # Balance for technical queries
                vector_w = 0.5
                bm25_w = 0.5

        # Normalize weights
        total = vector_w + bm25_w
        return vector_w / total, bm25_w / total

    async def get_search_analytics(self, query: str) -> Dict[str, Any]:
        """
        Get analytics for a search query showing how different methods perform.

        Args:
            query: Search query to analyze

        Returns:
            Dictionary with analytics data
        """
        try:
            # Perform searches
            vector_results = await self._vector_search(query, 10)
            bm25_results = await self._bm25_search(query, 10)
            hybrid_results = await self.search(query, 10)

            analytics = {
                'query': query,
                'vector_search': {
                    'count': len(vector_results),
                    'avg_score': np.mean([1.0 - r.get('distance', 0.0) for r in vector_results]) if vector_results else 0.0
                },
                'bm25_search': {
                    'count': len(bm25_results),
                    'avg_score': np.mean([score for _, score in bm25_results]) if bm25_results else 0.0
                },
                'hybrid_search': {
                    'count': len(hybrid_results),
                    'avg_combined_score': np.mean([r.combined_score for r in hybrid_results]) if hybrid_results else 0.0
                },
                'overlap': {
                    'vector_bm25_overlap': self._calculate_overlap(vector_results, bm25_results),
                    'unique_to_vector': len(vector_results) - self._calculate_overlap(vector_results, bm25_results),
                    'unique_to_bm25': len(bm25_results) - self._calculate_overlap(vector_results, bm25_results)
                },
                'recommended_weights': self._analyze_query_for_weights(query)
            }

            return analytics

        except Exception as e:
            logger.error(f"Error generating search analytics: {e}")
            return {}

    def _calculate_overlap(self, vector_results: List[Dict], bm25_results: List[Tuple]) -> int:
        """Calculate overlap between vector and BM25 results."""
        vector_contents = {r.get('content', '') for r in vector_results}
        bm25_contents = set()

        for doc_idx, _ in bm25_results:
            if doc_idx < len(self.indexed_documents):
                content = self.indexed_documents[doc_idx].get('content', '')
                bm25_contents.add(content)

        return len(vector_contents.intersection(bm25_contents))

# Example usage and integration with existing MilvusRAG
async def create_hybrid_search_system(milvus_rag) -> HybridSearch:
    """
    Create and initialize a hybrid search system with the existing MilvusRAG instance.

    Args:
        milvus_rag: Existing MilvusRAG instance

    Returns:
        Initialized HybridSearch instance
    """
    # Create hybrid search with default weights (60% vector, 40% BM25)
    hybrid_search = HybridSearch(milvus_rag, vector_weight=0.6, bm25_weight=0.4)

    # Index any existing documents from Milvus for BM25 search
    # This would typically be done when documents are added to the system

    return hybrid_search