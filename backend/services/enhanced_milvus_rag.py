"""
Enhanced MilvusRAG with Hybrid Search capabilities.
Extends the existing MilvusRAG class to include BM25 and hybrid search functionality.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import asyncio
from pathlib import Path

from .milvus_rag import MilvusRAG
from .hybrid_search import HybridSearch, SearchResult

logger = logging.getLogger(__name__)

class EnhancedMilvusRAG(MilvusRAG):
    """
    Enhanced version of MilvusRAG that includes hybrid search capabilities
    combining vector search with BM25 keyword search.
    """

    def __init__(self, enable_hybrid: bool = True, vector_weight: float = 0.6, bm25_weight: float = 0.4):
        """
        Initialize Enhanced MilvusRAG with hybrid search capabilities.

        Args:
            enable_hybrid: Whether to enable hybrid search functionality
            vector_weight: Weight for vector search in hybrid mode
            bm25_weight: Weight for BM25 search in hybrid mode
        """
        super().__init__()
        self.enable_hybrid = enable_hybrid
        self.hybrid_search = None
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self._document_cache = []  # Cache for BM25 indexing

    async def initialize(self):
        """Initialize both Milvus and hybrid search systems."""
        # Initialize base Milvus system
        await super().initialize()

        # Initialize hybrid search if enabled
        if self.enable_hybrid:
            try:
                self.hybrid_search = HybridSearch(
                    milvus_rag=self,
                    vector_weight=self.vector_weight,
                    bm25_weight=self.bm25_weight
                )
                logger.info("Hybrid search initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize hybrid search: {e}")
                self.enable_hybrid = False

    async def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents in both vector and BM25 systems."""
        # Index in base Milvus system
        await super().index_documents(documents)

        # Cache documents and index in hybrid search if enabled
        if self.enable_hybrid and self.hybrid_search:
            try:
                self._document_cache.extend(documents)
                await self.hybrid_search.index_documents(self._document_cache)
                logger.info(f"Indexed {len(documents)} documents in hybrid search system")
            except Exception as e:
                logger.error(f"Failed to index documents in hybrid search: {e}")

    async def search_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector search only.
        This maintains compatibility with the base class.
        """
        return await super().search_similar(query, limit)

    async def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        search_mode: str = "hybrid",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and BM25 results.

        Args:
            query: Search query
            limit: Number of results to return
            search_mode: Search mode ('hybrid', 'vector', 'bm25', 'adaptive')
            **kwargs: Additional search parameters

        Returns:
            List of search results with enhanced scoring information
        """
        if not self.enable_hybrid or not self.hybrid_search:
            logger.warning("Hybrid search not available, falling back to vector search")
            return await self.search_similar(query, limit)

        try:
            if search_mode == "vector":
                # Vector search only
                results = await self.search_similar(query, limit)
                return self._format_results(results, search_mode="vector")

            elif search_mode == "bm25":
                # BM25 search only
                bm25_results = await self.hybrid_search._bm25_search(query, limit)
                return self._format_bm25_results(bm25_results, search_mode="bm25")

            elif search_mode == "adaptive":
                # Adaptive hybrid search with dynamic weights
                query_type = kwargs.get('query_type')
                results = await self.hybrid_search.adaptive_search(query, limit, query_type)
                return self._format_hybrid_results(results, search_mode="adaptive")

            else:  # search_mode == "hybrid"
                # Standard hybrid search
                use_rrf = kwargs.get('use_rrf', True)
                rrf_k = kwargs.get('rrf_k', 60)
                vector_limit = kwargs.get('vector_limit', limit * 2)
                bm25_limit = kwargs.get('bm25_limit', limit * 2)

                results = await self.hybrid_search.search(
                    query=query,
                    limit=limit,
                    vector_limit=vector_limit,
                    bm25_limit=bm25_limit,
                    rrf_k=rrf_k,
                    use_rrf=use_rrf
                )
                return self._format_hybrid_results(results, search_mode="hybrid")

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to vector search
            return await self.search_similar(query, limit)

    def _format_results(self, results: List[Dict[str, Any]], search_mode: str) -> List[Dict[str, Any]]:
        """Format vector search results."""
        formatted_results = []
        for result in results:
            formatted_result = {
                'content': result.get('content', ''),
                'metadata': result.get('metadata', {}),
                'distance': result.get('distance', 1.0),
                'search_mode': search_mode,
                'scores': {
                    'vector_score': 1.0 - result.get('distance', 0.0),
                    'bm25_score': 0.0,
                    'combined_score': 1.0 - result.get('distance', 0.0)
                }
            }
            formatted_results.append(formatted_result)
        return formatted_results

    def _format_bm25_results(self, results: List[tuple], search_mode: str) -> List[Dict[str, Any]]:
        """Format BM25 search results."""
        formatted_results = []
        for doc_idx, score in results:
            if doc_idx < len(self._document_cache):
                doc = self._document_cache[doc_idx]
                formatted_result = {
                    'content': doc.get('content', ''),
                    'metadata': doc.get('metadata', {}),
                    'distance': 1.0 - min(score, 1.0),  # Convert to distance-like score
                    'search_mode': search_mode,
                    'scores': {
                        'vector_score': 0.0,
                        'bm25_score': score,
                        'combined_score': score
                    }
                }
                formatted_results.append(formatted_result)
        return formatted_results

    def _format_hybrid_results(self, results: List[SearchResult], search_mode: str) -> List[Dict[str, Any]]:
        """Format hybrid search results."""
        formatted_results = []
        for result in results:
            formatted_result = {
                'content': result.content,
                'metadata': result.metadata,
                'distance': 1.0 - result.combined_score,  # Convert to distance-like score
                'search_mode': search_mode,
                'scores': {
                    'vector_score': result.vector_score,
                    'bm25_score': result.bm25_score,
                    'combined_score': result.combined_score
                },
                'rankings': {
                    'vector_rank': result.rank_vector,
                    'bm25_rank': result.rank_bm25
                }
            }
            formatted_results.append(formatted_result)
        return formatted_results

    async def get_company_context(self, query: str = "Kamiwaza company information", use_hybrid: bool = True) -> str:
        """
        Get relevant company context for RFP response using hybrid search if available.

        Args:
            query: Search query for company information
            use_hybrid: Whether to use hybrid search or fall back to vector search

        Returns:
            Formatted company context string
        """
        try:
            if use_hybrid and self.enable_hybrid:
                results = await self.hybrid_search(query, limit=10, search_mode="adaptive")
            else:
                results = await self.search_similar(query, limit=10)

            context = "KAMIWAZA COMPANY INFORMATION:\n\n"
            for result in results:
                content = result.get('content', '')
                search_mode = result.get('search_mode', 'vector')
                scores = result.get('scores', {})

                # Add scoring information for transparency
                context += f"[Source: {search_mode} search"
                if 'combined_score' in scores:
                    context += f", Score: {scores['combined_score']:.3f}"
                context += f"]\n{content}\n\n"

            return context

        except Exception as e:
            logger.error(f"Error getting company context with hybrid search: {e}")
            # Fallback to base implementation
            return await super().get_company_context(query)

    async def search_past_rfps(self, requirements: str, use_hybrid: bool = True) -> List[Dict[str, Any]]:
        """
        Search for similar past RFP responses using hybrid search if available.

        Args:
            requirements: Requirements to search for
            use_hybrid: Whether to use hybrid search

        Returns:
            List of past RFP responses
        """
        try:
            if use_hybrid and self.enable_hybrid:
                results = await self.hybrid_search(
                    requirements,
                    limit=5,
                    search_mode="adaptive",
                    query_type="informational"
                )
            else:
                results = await self.search_similar(requirements, limit=5)

            # Filter for RFP responses
            past_rfps = []
            for result in results:
                metadata = result.get('metadata', {})
                if metadata.get('type') == 'rfp_response':
                    past_rfps.append(result)

            return past_rfps

        except Exception as e:
            logger.error(f"Error searching past RFPs with hybrid search: {e}")
            # Fallback to base implementation
            return await super().search_past_rfps(requirements)

    async def search_analytics(self, query: str) -> Dict[str, Any]:
        """
        Get detailed analytics for how different search methods perform on a query.

        Args:
            query: Query to analyze

        Returns:
            Analytics data showing performance of different search methods
        """
        if not self.enable_hybrid or not self.hybrid_search:
            return {"error": "Hybrid search not available"}

        try:
            analytics = await self.hybrid_search.get_search_analytics(query)
            return analytics
        except Exception as e:
            logger.error(f"Error getting search analytics: {e}")
            return {"error": str(e)}

    async def optimize_search_weights(
        self,
        test_queries: List[str],
        expected_results: Optional[List[List[str]]] = None
    ) -> Dict[str, float]:
        """
        Optimize search weights based on test queries and expected results.

        Args:
            test_queries: List of test queries
            expected_results: Optional list of expected result contents for each query

        Returns:
            Dictionary with optimal weights
        """
        if not self.enable_hybrid or not self.hybrid_search:
            return {"error": "Hybrid search not available"}

        # This is a simplified weight optimization
        # In production, you'd use more sophisticated evaluation metrics
        best_weights = {"vector_weight": 0.6, "bm25_weight": 0.4}
        best_score = 0.0

        weight_combinations = [
            (0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5),
            (0.4, 0.6), (0.3, 0.7), (0.2, 0.8)
        ]

        try:
            for vector_w, bm25_w in weight_combinations:
                # Temporarily set weights
                original_vector_w = self.hybrid_search.vector_weight
                original_bm25_w = self.hybrid_search.bm25_weight

                self.hybrid_search.vector_weight = vector_w
                self.hybrid_search.bm25_weight = bm25_w

                total_score = 0.0
                for query in test_queries:
                    results = await self.hybrid_search.search(query, limit=5)
                    # Simple scoring based on result count and average combined score
                    if results:
                        avg_score = sum(r.combined_score for r in results) / len(results)
                        total_score += avg_score

                avg_score = total_score / len(test_queries) if test_queries else 0.0

                if avg_score > best_score:
                    best_score = avg_score
                    best_weights = {"vector_weight": vector_w, "bm25_weight": bm25_w}

                # Restore original weights
                self.hybrid_search.vector_weight = original_vector_w
                self.hybrid_search.bm25_weight = original_bm25_w

            logger.info(f"Optimal weights found: {best_weights} with score: {best_score}")
            return best_weights

        except Exception as e:
            logger.error(f"Error optimizing search weights: {e}")
            return {"error": str(e)}

    async def batch_hybrid_search(
        self,
        queries: List[str],
        limit_per_query: int = 5,
        search_mode: str = "hybrid"
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform batch hybrid search for multiple queries efficiently.

        Args:
            queries: List of queries to search
            limit_per_query: Number of results per query
            search_mode: Search mode to use

        Returns:
            Dictionary mapping queries to their results
        """
        results = {}

        try:
            # Execute searches concurrently
            search_tasks = [
                self.hybrid_search(query, limit_per_query, search_mode)
                for query in queries
            ]

            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

            for query, result in zip(queries, search_results):
                if isinstance(result, Exception):
                    logger.error(f"Error in batch search for query '{query}': {result}")
                    results[query] = []
                else:
                    results[query] = result

        except Exception as e:
            logger.error(f"Error in batch hybrid search: {e}")
            # Return empty results for all queries
            results = {query: [] for query in queries}

        return results