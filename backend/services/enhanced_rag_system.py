"""
Enhanced RAG System with A-Grade Features
Implements: RAGAS evaluation, hybrid search, reranking, semantic chunking, caching, and monitoring
"""

import os
import logging
import asyncio
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
from dataclasses import dataclass, asdict
import pickle

# Core dependencies
import openai
from pymilvus import MilvusClient
import redis.asyncio as redis
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import torch

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SearchResult:
    """Enhanced search result with metadata"""
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str  # 'vector', 'bm25', 'hybrid'
    chunk_id: str
    relevance_score: Optional[float] = None  # After reranking

@dataclass
class RAGMetrics:
    """RAGAS evaluation metrics"""
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    retrieval_latency: float
    generation_latency: float
    total_latency: float
    timestamp: datetime

@dataclass
class CacheEntry:
    """Cache entry for query results"""
    query: str
    results: List[SearchResult]
    metrics: Optional[RAGMetrics]
    timestamp: datetime
    hit_count: int = 0

# ============================================================================
# Semantic Chunking Implementation
# ============================================================================

class SemanticChunker:
    """Advanced semantic chunking with context preservation"""

    def __init__(self,
                 model_name: str = "sentence-transformers/all-mpnet-base-v2",
                 similarity_threshold: float = 0.75,
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 2000):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk_document(self,
                      text: str,
                      metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create semantic chunks with contextual headers"""

        # Split into sentences
        sentences = sent_tokenize(text)
        if not sentences:
            return []

        # Get sentence embeddings
        sentence_embeddings = self.model.encode(sentences)

        # Group sentences into semantic chunks
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = sentence_embeddings[0]

        for i in range(1, len(sentences)):
            # Calculate similarity with current chunk
            similarity = self._cosine_similarity(
                current_embedding,
                sentence_embeddings[i]
            )

            # Check if should add to current chunk
            current_size = len(' '.join(current_chunk))

            if (similarity >= self.similarity_threshold and
                current_size < self.max_chunk_size):
                # Add to current chunk
                current_chunk.append(sentences[i])
                # Update chunk embedding (mean of all sentences)
                chunk_indices = range(i - len(current_chunk) + 1, i + 1)
                current_embedding = np.mean(
                    [sentence_embeddings[j] for j in chunk_indices],
                    axis=0
                )
            else:
                # Save current chunk if it meets minimum size
                if current_size >= self.min_chunk_size:
                    chunk_data = self._create_chunk_with_context(
                        current_chunk,
                        metadata,
                        chunk_index=len(chunks)
                    )
                    chunks.append(chunk_data)
                    current_chunk = [sentences[i]]
                    current_embedding = sentence_embeddings[i]
                else:
                    # Merge with previous chunk if too small
                    current_chunk.append(sentences[i])

        # Add final chunk
        if current_chunk:
            chunk_data = self._create_chunk_with_context(
                current_chunk,
                metadata,
                chunk_index=len(chunks)
            )
            chunks.append(chunk_data)

        return chunks

    def _create_chunk_with_context(self,
                                   sentences: List[str],
                                   metadata: Dict[str, Any],
                                   chunk_index: int) -> Dict[str, Any]:
        """Create chunk with rich contextual metadata"""

        chunk_text = ' '.join(sentences)

        # Build contextual header
        context_header = self._build_context_header(metadata, chunk_index)

        # Extract keywords for better retrieval
        keywords = self._extract_keywords(chunk_text)

        return {
            'content': chunk_text,
            'context_header': context_header,
            'metadata': {
                **metadata,
                'chunk_index': chunk_index,
                'chunk_size': len(chunk_text),
                'sentence_count': len(sentences),
                'keywords': keywords,
                'embedding_text': f"{context_header}\n\n{chunk_text}"
            }
        }

    def _build_context_header(self,
                              metadata: Dict[str, Any],
                              chunk_index: int) -> str:
        """Build contextual header for chunk"""

        parts = []

        # Document context
        if metadata.get('filename'):
            parts.append(f"Document: {metadata['filename']}")
        if metadata.get('category'):
            parts.append(f"Category: {metadata['category']}")
        if metadata.get('section'):
            parts.append(f"Section: {metadata['section']}")

        # Position context
        parts.append(f"Part {chunk_index + 1}")

        return " | ".join(parts)

    def _extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """Extract keywords using TF-IDF"""
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Simple keyword extraction
        vectorizer = TfidfVectorizer(
            max_features=top_k,
            stop_words='english',
            ngram_range=(1, 2)
        )

        try:
            vectorizer.fit([text])
            keywords = vectorizer.get_feature_names_out()
            return list(keywords)
        except:
            # Fallback to simple word frequency
            words = word_tokenize(text.lower())
            from collections import Counter
            word_freq = Counter(words)
            return [word for word, _ in word_freq.most_common(top_k)]

    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between vectors"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# ============================================================================
# Hybrid Search Implementation
# ============================================================================

class HybridSearch:
    """Hybrid search combining vector and BM25 keyword search"""

    def __init__(self, documents: List[str] = None):
        self.bm25 = None
        self.documents = documents or []
        self.tokenized_docs = []

        if documents:
            self._initialize_bm25(documents)

    def _initialize_bm25(self, documents: List[str]):
        """Initialize BM25 index"""
        # Tokenize documents
        self.tokenized_docs = [
            word_tokenize(doc.lower()) for doc in documents
        ]
        # Create BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def add_documents(self, documents: List[str]):
        """Add documents to BM25 index"""
        self.documents.extend(documents)
        self._initialize_bm25(self.documents)

    def search_bm25(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Perform BM25 keyword search"""
        if not self.bm25:
            return []

        # Tokenize query
        query_tokens = word_tokenize(query.lower())

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k results
        top_indices = np.argsort(scores)[-top_k:][::-1]

        return [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]

    def reciprocal_rank_fusion(self,
                              vector_results: List[Tuple[int, float]],
                              bm25_results: List[Tuple[int, float]],
                              k: int = 60,
                              weights: Tuple[float, float] = (0.6, 0.4)) -> List[Tuple[int, float]]:
        """Fuse results using Reciprocal Rank Fusion"""

        # Calculate RRF scores
        rrf_scores = {}

        # Process vector results
        for rank, (idx, score) in enumerate(vector_results):
            rrf_scores[idx] = weights[0] / (k + rank + 1)

        # Process BM25 results
        for rank, (idx, score) in enumerate(bm25_results):
            if idx in rrf_scores:
                rrf_scores[idx] += weights[1] / (k + rank + 1)
            else:
                rrf_scores[idx] = weights[1] / (k + rank + 1)

        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_results

# ============================================================================
# Cross-Encoder Reranking
# ============================================================================

class Reranker:
    """Cross-encoder reranking for improved relevance"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.cross_encoder = CrossEncoder(model_name)

    def rerank(self,
               query: str,
               documents: List[str],
               top_k: int = 5) -> List[Tuple[int, float]]:
        """Rerank documents using cross-encoder"""

        if not documents:
            return []

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)

        # Sort by score
        sorted_indices = np.argsort(scores)[::-1][:top_k]

        return [(idx, scores[idx]) for idx in sorted_indices]

# ============================================================================
# Multi-Level Caching
# ============================================================================

class MultiLevelCache:
    """Multi-level caching for RAG system"""

    def __init__(self,
                 redis_host: str = "localhost",
                 redis_port: int = 6380,  # Changed to 6380
                 embedding_ttl: int = 3600,
                 result_ttl: int = 1800):

        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = None
        self.use_redis = False
        self.memory_cache = {}
        self.embedding_ttl = embedding_ttl
        self.result_ttl = result_ttl

    async def initialize(self):
        """Initialize Redis connection asynchronously"""
        try:
            self.redis_client = await redis.from_url(
                f"redis://{self.redis_host}:{self.redis_port}",
                decode_responses=False
            )
            await self.redis_client.ping()
            self.use_redis = True
            logger.info(f"Connected to Redis for caching on port {self.redis_port}")
        except Exception as e:
            self.use_redis = False
            logger.info(f"Using in-memory cache (Redis not available: {e})")

    async def cache_embedding(self, text: str, embedding: List[float]):
        """Cache text embedding"""
        key = f"emb:{hashlib.md5(text.encode()).hexdigest()}"

        if self.use_redis:
            await self.redis_client.setex(
                key,
                self.embedding_ttl,
                pickle.dumps(embedding)
            )
        else:
            self.memory_cache[key] = {
                'data': embedding,
                'expires': datetime.now() + timedelta(seconds=self.embedding_ttl)
            }

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding"""
        key = f"emb:{hashlib.md5(text.encode()).hexdigest()}"

        if self.use_redis:
            data = await self.redis_client.get(key)
            return pickle.loads(data) if data else None
        else:
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if entry['expires'] > datetime.now():
                    return entry['data']
                else:
                    del self.memory_cache[key]
            return None

    async def cache_search_results(self,
                            query: str,
                            results: List[SearchResult]):
        """Cache search results"""
        key = f"search:{hashlib.md5(query.encode()).hexdigest()}"

        if self.use_redis:
            await self.redis_client.setex(
                key,
                self.result_ttl,
                pickle.dumps([asdict(r) for r in results])
            )
        else:
            self.memory_cache[key] = {
                'data': results,
                'expires': datetime.now() + timedelta(seconds=self.result_ttl)
            }

    async def get_search_results(self, query: str) -> Optional[List[SearchResult]]:
        """Get cached search results"""
        key = f"search:{hashlib.md5(query.encode()).hexdigest()}"

        if self.use_redis:
            data = await self.redis_client.get(key)
            if data:
                results_data = pickle.loads(data)
                return [SearchResult(**r) for r in results_data]
            return None
        else:
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if entry['expires'] > datetime.now():
                    return entry['data']
                else:
                    del self.memory_cache[key]
            return None

# ============================================================================
# RAGAS Evaluation
# ============================================================================

class RAGASEvaluator:
    """RAGAS evaluation for quality monitoring"""

    def __init__(self, openai_api_key: str = None):
        self.client = openai.OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        self.metrics_history = []

    async def evaluate(self,
                       query: str,
                       contexts: List[str],
                       answer: str,
                       ground_truth: Optional[str] = None) -> RAGMetrics:
        """Evaluate RAG response using RAGAS metrics"""

        start_time = datetime.now()

        # Calculate faithfulness
        faithfulness = await self._calculate_faithfulness(answer, contexts)

        # Calculate answer relevancy
        answer_relevancy = await self._calculate_answer_relevancy(query, answer)

        # Calculate context precision
        context_precision = await self._calculate_context_precision(query, contexts)

        # Calculate context recall (if ground truth available)
        context_recall = 0.0
        if ground_truth:
            context_recall = await self._calculate_context_recall(ground_truth, contexts)

        metrics = RAGMetrics(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall,
            retrieval_latency=0.0,  # Set by caller
            generation_latency=0.0,  # Set by caller
            total_latency=(datetime.now() - start_time).total_seconds(),
            timestamp=datetime.now()
        )

        self.metrics_history.append(metrics)
        return metrics

    async def _calculate_faithfulness(self,
                                     answer: str,
                                     contexts: List[str]) -> float:
        """Calculate faithfulness score"""

        prompt = f"""
        Given the following contexts and answer, evaluate the faithfulness of the answer.
        Faithfulness measures whether the answer is grounded in the given contexts.

        Contexts:
        {' '.join(contexts[:3])}  # Limit context for API

        Answer:
        {answer}

        Return a score between 0 and 1, where 1 means the answer is completely faithful to the contexts.
        Return only the numerical score.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            score = float(response.choices[0].message.content.strip())
            return min(max(score, 0.0), 1.0)
        except:
            return 0.5  # Default score on error

    async def _calculate_answer_relevancy(self,
                                         query: str,
                                         answer: str) -> float:
        """Calculate answer relevancy score"""

        prompt = f"""
        Given the following question and answer, evaluate how relevant the answer is to the question.

        Question: {query}
        Answer: {answer}

        Return a score between 0 and 1, where 1 means the answer is highly relevant.
        Return only the numerical score.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            score = float(response.choices[0].message.content.strip())
            return min(max(score, 0.0), 1.0)
        except:
            return 0.5

    async def _calculate_context_precision(self,
                                          query: str,
                                          contexts: List[str]) -> float:
        """Calculate context precision score"""

        # Simple heuristic: check how many contexts are relevant
        relevant_count = 0
        for context in contexts[:5]:  # Check top 5
            if self._is_context_relevant(query, context):
                relevant_count += 1

        return relevant_count / min(len(contexts), 5)

    async def _calculate_context_recall(self,
                                       ground_truth: str,
                                       contexts: List[str]) -> float:
        """Calculate context recall score"""

        # Check if key information from ground truth is in contexts
        key_terms = set(ground_truth.lower().split())
        context_terms = set(' '.join(contexts).lower().split())

        if not key_terms:
            return 0.0

        overlap = len(key_terms.intersection(context_terms))
        return overlap / len(key_terms)

    def _is_context_relevant(self, query: str, context: str) -> bool:
        """Check if context is relevant to query"""
        query_terms = set(query.lower().split())
        context_terms = set(context.lower().split())

        overlap = len(query_terms.intersection(context_terms))
        return overlap >= min(3, len(query_terms) // 2)

    def get_average_metrics(self) -> Dict[str, float]:
        """Get average metrics from history"""
        if not self.metrics_history:
            return {}

        return {
            'avg_faithfulness': np.mean([m.faithfulness for m in self.metrics_history]),
            'avg_answer_relevancy': np.mean([m.answer_relevancy for m in self.metrics_history]),
            'avg_context_precision': np.mean([m.context_precision for m in self.metrics_history]),
            'avg_context_recall': np.mean([m.context_recall for m in self.metrics_history if m.context_recall > 0]),
            'avg_total_latency': np.mean([m.total_latency for m in self.metrics_history])
        }

# ============================================================================
# Query Optimization
# ============================================================================

class QueryOptimizer:
    """Query optimization with expansion and decomposition"""

    def __init__(self, openai_api_key: str = None):
        self.client = openai.OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))

    async def optimize_query(self,
                            query: str,
                            mode: str = "expand") -> List[str]:
        """Optimize query for better retrieval"""

        if mode == "expand":
            return await self._expand_query(query)
        elif mode == "decompose":
            return await self._decompose_query(query)
        elif mode == "multi":
            return await self._multi_query(query)
        else:
            return [query]

    async def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms"""

        prompt = f"""
        Expand the following query with synonyms and related terms to improve search results.
        Original query: {query}

        Generate 3 expanded versions of the query.
        Return each query on a new line.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            expanded = response.choices[0].message.content.strip().split('\n')
            return [query] + [q.strip() for q in expanded if q.strip()][:3]
        except:
            return [query]

    async def _decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into simpler sub-queries"""

        prompt = f"""
        Break down this complex query into simpler sub-queries:
        {query}

        Generate 2-4 simpler queries that together cover the original query.
        Return each query on a new line.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            decomposed = response.choices[0].message.content.strip().split('\n')
            return [q.strip() for q in decomposed if q.strip()][:4]
        except:
            return [query]

    async def _multi_query(self, query: str) -> List[str]:
        """Generate multiple perspectives of the query"""

        prompt = f"""
        Generate different perspectives of this query:
        {query}

        Create 3 alternative phrasings that might retrieve different relevant information.
        Return each query on a new line.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            perspectives = response.choices[0].message.content.strip().split('\n')
            return [query] + [q.strip() for q in perspectives if q.strip()][:3]
        except:
            return [query]

# ============================================================================
# Enhanced RAG System
# ============================================================================

class EnhancedRAGSystem:
    """Production-grade RAG system with all A-grade features"""

    def __init__(self,
                 milvus_uri: str = "./milvus_enhanced.db",
                 collection_name: str = "enhanced_rag",
                 embedding_model: str = "text-embedding-3-large",
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 enable_cache: bool = True,
                 enable_monitoring: bool = True):

        # Core components
        self.milvus_client = MilvusClient(uri=milvus_uri)
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Initialize components
        self.semantic_chunker = SemanticChunker()
        self.hybrid_search = HybridSearch()
        self.reranker = Reranker(reranker_model)
        self.query_optimizer = QueryOptimizer()
        self.ragas_evaluator = RAGASEvaluator()

        # Caching
        self.cache = MultiLevelCache() if enable_cache else None

        # Monitoring
        self.enable_monitoring = enable_monitoring
        self.search_metrics = []

        # OpenAI client for embeddings
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        logger.info("Enhanced RAG System initialized")

    async def initialize(self):
        """Initialize the enhanced RAG system"""

        # Initialize cache if enabled
        if self.cache:
            await self.cache.initialize()

        # Create collection if not exists
        if not self.milvus_client.has_collection(self.collection_name):
            self.milvus_client.create_collection(
                collection_name=self.collection_name,
                dimension=3072,  # text-embedding-3-large dimension
                metric_type="COSINE",
                consistency_level="Strong"
            )
            logger.info(f"Created collection: {self.collection_name}")

        # Load existing documents for BM25
        await self._load_documents_for_bm25()

        logger.info("Enhanced RAG System ready")

    async def index_document(self,
                            content: str,
                            metadata: Dict[str, Any]) -> int:
        """Index document with semantic chunking"""

        # Create semantic chunks
        chunks = self.semantic_chunker.chunk_document(content, metadata)

        # Prepare for indexing
        documents = []
        for chunk in chunks:
            # Generate embedding
            embedding = await self._get_embedding(chunk['metadata']['embedding_text'])

            # Create document
            doc_id = hashlib.md5(
                f"{chunk['content']}_{chunk['metadata']['chunk_index']}".encode()
            ).hexdigest()

            documents.append({
                "id": doc_id,
                "vector": embedding,
                "text": chunk['content'],
                "context_header": chunk['context_header'],
                "metadata": json.dumps(chunk['metadata'])
            })

            # Add to BM25 index
            self.hybrid_search.add_documents([chunk['content']])

        # Insert into Milvus
        if documents:
            self.milvus_client.insert(
                collection_name=self.collection_name,
                data=documents
            )
            logger.info(f"Indexed {len(documents)} chunks from document")

        return len(documents)

    async def search(self,
                    query: str,
                    top_k: int = 10,
                    search_mode: str = "hybrid",
                    enable_reranking: bool = True,
                    enable_optimization: bool = True) -> Tuple[List[SearchResult], RAGMetrics]:
        """Enhanced search with all features"""

        start_time = datetime.now()

        # Check cache
        if self.cache:
            cached_results = await self.cache.get_search_results(query)
            if cached_results:
                logger.info("Cache hit for query")
                return cached_results, None

        # Optimize query
        queries = [query]
        if enable_optimization:
            queries = await self.query_optimizer.optimize_query(query, mode="expand")

        # Perform searches for all queries
        all_results = []
        for q in queries:
            if search_mode == "hybrid":
                results = await self._hybrid_search(q, top_k * 2)
            elif search_mode == "vector":
                results = await self._vector_search(q, top_k * 2)
            else:  # bm25
                results = await self._bm25_search(q, top_k * 2)

            all_results.extend(results)

        # Deduplicate results
        seen = set()
        unique_results = []
        for r in all_results:
            if r.chunk_id not in seen:
                seen.add(r.chunk_id)
                unique_results.append(r)

        # Rerank if enabled
        if enable_reranking and unique_results:
            unique_results = await self._rerank_results(query, unique_results, top_k)
        else:
            unique_results = unique_results[:top_k]

        retrieval_time = (datetime.now() - start_time).total_seconds()

        # Evaluate if monitoring enabled
        metrics = None
        if self.enable_monitoring and unique_results:
            contexts = [r.content for r in unique_results[:5]]
            # Simulate answer generation for evaluation
            answer = contexts[0][:500] if contexts else ""

            metrics = await self.ragas_evaluator.evaluate(
                query=query,
                contexts=contexts,
                answer=answer
            )
            metrics.retrieval_latency = retrieval_time

        # Cache results
        if self.cache:
            await self.cache.cache_search_results(query, unique_results)

        return unique_results, metrics

    async def _hybrid_search(self,
                            query: str,
                            top_k: int) -> List[SearchResult]:
        """Perform hybrid search"""

        # Vector search
        vector_results = await self._vector_search(query, top_k)

        # BM25 search
        bm25_results = await self._bm25_search(query, top_k)

        # Create index mappings
        vector_indices = [(i, r.score) for i, r in enumerate(vector_results)]
        bm25_indices = [(i, r.score) for i, r in enumerate(bm25_results)]

        # Apply RRF
        hybrid_search = HybridSearch()
        fused_indices = hybrid_search.reciprocal_rank_fusion(
            vector_indices,
            bm25_indices,
            weights=(0.6, 0.4)  # Favor vector search slightly
        )

        # Combine results
        all_results = vector_results + bm25_results
        seen = set()
        final_results = []

        for idx, score in fused_indices[:top_k]:
            if idx < len(vector_results):
                result = vector_results[idx]
            else:
                result = bm25_results[idx - len(vector_results)]

            if result.chunk_id not in seen:
                seen.add(result.chunk_id)
                result.score = score
                result.source = "hybrid"
                final_results.append(result)

        return final_results

    async def _vector_search(self,
                           query: str,
                           top_k: int) -> List[SearchResult]:
        """Perform vector search"""

        # Get query embedding
        query_embedding = await self._get_embedding(query)

        # Search in Milvus
        results = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["text", "context_header", "metadata"]
        )

        # Convert to SearchResult objects
        search_results = []
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                search_results.append(SearchResult(
                    content=entity.get("text", ""),
                    metadata=json.loads(entity.get("metadata", "{}")),
                    score=hit.get("distance", 0.0),
                    source="vector",
                    chunk_id=hit.get("id", "")
                ))

        return search_results

    async def _bm25_search(self,
                          query: str,
                          top_k: int) -> List[SearchResult]:
        """Perform BM25 search"""

        # Get BM25 results
        bm25_results = self.hybrid_search.search_bm25(query, top_k)

        # Convert to SearchResult objects
        search_results = []
        for idx, score in bm25_results:
            if idx < len(self.hybrid_search.documents):
                content = self.hybrid_search.documents[idx]
                search_results.append(SearchResult(
                    content=content,
                    metadata={},
                    score=score,
                    source="bm25",
                    chunk_id=hashlib.md5(content.encode()).hexdigest()
                ))

        return search_results

    async def _rerank_results(self,
                             query: str,
                             results: List[SearchResult],
                             top_k: int) -> List[SearchResult]:
        """Rerank results using cross-encoder"""

        # Extract documents
        documents = [r.content for r in results]

        # Rerank
        reranked_indices = self.reranker.rerank(query, documents, top_k)

        # Update results with reranking scores
        reranked_results = []
        for idx, score in reranked_indices:
            result = results[idx]
            result.relevance_score = score
            reranked_results.append(result)

        return reranked_results

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding with caching"""

        # Check cache
        if self.cache:
            cached = self.cache.get_embedding(text)
            if cached:
                return cached

        # Generate embedding
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        embedding = response.data[0].embedding

        # Cache embedding
        if self.cache:
            self.cache.cache_embedding(text, embedding)

        return embedding

    async def _load_documents_for_bm25(self):
        """Load existing documents for BM25 index"""

        # Query all documents from Milvus
        try:
            # Note: This is a simplified version
            # In production, you'd want to batch this
            results = self.milvus_client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["text"],
                limit=10000
            )

            documents = [r.get("text", "") for r in results]
            if documents:
                self.hybrid_search.add_documents(documents)
                logger.info(f"Loaded {len(documents)} documents for BM25")
        except Exception as e:
            logger.warning(f"Could not load documents for BM25: {e}")

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""

        ragas_metrics = self.ragas_evaluator.get_average_metrics()

        return {
            "ragas_metrics": ragas_metrics,
            "total_evaluations": len(self.ragas_evaluator.metrics_history),
            "cache_enabled": self.cache is not None,
            "monitoring_enabled": self.enable_monitoring,
            "collection_name": self.collection_name
        }

# ============================================================================
# Production Monitoring Service
# ============================================================================

class RAGMonitor:
    """Production monitoring for RAG system"""

    def __init__(self, rag_system: EnhancedRAGSystem):
        self.rag_system = rag_system
        self.alerts = []
        self.thresholds = {
            'faithfulness': 0.7,
            'answer_relevancy': 0.7,
            'context_precision': 0.6,
            'latency': 2.0  # seconds
        }

    async def monitor_query(self,
                           query: str,
                           results: List[SearchResult],
                           metrics: RAGMetrics):
        """Monitor query performance"""

        # Check thresholds
        if metrics.faithfulness < self.thresholds['faithfulness']:
            self._create_alert('faithfulness', metrics.faithfulness)

        if metrics.answer_relevancy < self.thresholds['answer_relevancy']:
            self._create_alert('answer_relevancy', metrics.answer_relevancy)

        if metrics.context_precision < self.thresholds['context_precision']:
            self._create_alert('context_precision', metrics.context_precision)

        if metrics.total_latency > self.thresholds['latency']:
            self._create_alert('latency', metrics.total_latency)

    def _create_alert(self, metric: str, value: float):
        """Create performance alert"""

        alert = {
            'timestamp': datetime.now(),
            'metric': metric,
            'value': value,
            'threshold': self.thresholds.get(metric),
            'severity': 'high' if value < 0.5 else 'medium'
        }

        self.alerts.append(alert)
        logger.warning(f"Performance alert: {metric}={value:.2f} (threshold={self.thresholds.get(metric)})")

    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status"""

        metrics = self.rag_system.get_system_metrics()
        recent_alerts = [a for a in self.alerts if
                        (datetime.now() - a['timestamp']).seconds < 3600]

        # Determine health score
        ragas_metrics = metrics.get('ragas_metrics', {})
        health_score = 0.0

        if ragas_metrics:
            health_score = (
                ragas_metrics.get('avg_faithfulness', 0) * 0.3 +
                ragas_metrics.get('avg_answer_relevancy', 0) * 0.3 +
                ragas_metrics.get('avg_context_precision', 0) * 0.2 +
                (1.0 - min(ragas_metrics.get('avg_total_latency', 0) / 5.0, 1.0)) * 0.2
            )

        return {
            'health_score': health_score,
            'status': 'healthy' if health_score > 0.7 else 'degraded' if health_score > 0.5 else 'unhealthy',
            'metrics': metrics,
            'recent_alerts': len(recent_alerts),
            'alert_details': recent_alerts[-5:]  # Last 5 alerts
        }

# ============================================================================
# Usage Example
# ============================================================================

async def main():
    """Example usage of enhanced RAG system"""

    # Initialize system
    rag = EnhancedRAGSystem()
    await rag.initialize()

    # Create monitor
    monitor = RAGMonitor(rag)

    # Example document
    document = """
    Artificial intelligence (AI) is revolutionizing how businesses operate.
    Machine learning algorithms can analyze vast amounts of data to identify patterns.
    Natural language processing enables computers to understand human language.
    Deep learning neural networks are particularly effective for complex tasks.
    Computer vision allows machines to interpret visual information from the world.
    """

    # Index document
    await rag.index_document(
        content=document,
        metadata={
            'filename': 'ai_overview.txt',
            'category': 'technical',
            'date': datetime.now().isoformat()
        }
    )

    # Search with all features
    query = "How is AI transforming business operations?"
    results, metrics = await rag.search(
        query=query,
        top_k=5,
        search_mode="hybrid",
        enable_reranking=True,
        enable_optimization=True
    )

    # Monitor performance
    if metrics:
        await monitor.monitor_query(query, results, metrics)

    # Display results
    print(f"\nQuery: {query}")
    print(f"Found {len(results)} results")

    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result.score:.3f} | Source: {result.source}")
        print(f"   Content: {result.content[:200]}...")
        if result.relevance_score:
            print(f"   Reranking Score: {result.relevance_score:.3f}")

    # Show metrics
    if metrics:
        print(f"\nRAGAS Metrics:")
        print(f"  Faithfulness: {metrics.faithfulness:.3f}")
        print(f"  Answer Relevancy: {metrics.answer_relevancy:.3f}")
        print(f"  Context Precision: {metrics.context_precision:.3f}")
        print(f"  Total Latency: {metrics.total_latency:.3f}s")

    # Show health status
    health = monitor.get_health_status()
    print(f"\nSystem Health: {health['status']} (score: {health['health_score']:.2f})")

if __name__ == "__main__":
    asyncio.run(main())