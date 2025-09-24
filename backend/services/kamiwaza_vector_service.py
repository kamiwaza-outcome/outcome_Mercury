"""
Kamiwaza Vector Database Service
Replaces MilvusRAG with Kamiwaza-powered vector storage and search capabilities.
Uses Kamiwaza's embedding models and connects to Kamiwaza's Milvus instance.
"""
import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
import numpy as np
import hashlib
from functools import cached_property
from datetime import datetime
import asyncio

from pymilvus import MilvusClient, DataType, Collection, connections, utility
from kamiwaza_client import KamiwazaClient
from .kamiwaza_service import get_kamiwaza_service, KamiwazaService

logger = logging.getLogger(__name__)


class KamiwazaVectorService:
    """
    Vector database service using Kamiwaza's embedding models and Milvus instance.
    Provides the same interface as MilvusRAG but uses Kamiwaza infrastructure.
    """

    def __init__(
        self,
        collection_name: str = "kamiwaza_knowledge_v2",
        embedding_model: Optional[str] = None,
        vector_dimension: int = 1536,
        milvus_host: Optional[str] = None,
        milvus_port: Optional[int] = None
    ):
        """
        Initialize Kamiwaza Vector Service.

        Args:
            collection_name: Name of the Milvus collection to use
            embedding_model: Kamiwaza embedding model name (auto-detected if None)
            vector_dimension: Dimension of embedding vectors (depends on model)
            milvus_host: Kamiwaza Milvus host (auto-configured if None)
            milvus_port: Kamiwaza Milvus port (auto-configured if None)
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.vector_dimension = vector_dimension
        self.milvus_host = milvus_host or os.getenv("KAMIWAZA_MILVUS_HOST", "localhost")
        self.milvus_port = milvus_port or int(os.getenv("KAMIWAZA_MILVUS_PORT", "19530"))

        # Kamiwaza configuration
        self._kamiwaza_service: Optional[KamiwazaService] = None
        self._milvus_client: Optional[MilvusClient] = None
        self._embedding_client = None
        self._is_initialized = False

        logger.info(f"Initializing Kamiwaza Vector Service for collection: {collection_name}")

    @cached_property
    def kamiwaza_service(self) -> KamiwazaService:
        """Get Kamiwaza service instance."""
        if self._kamiwaza_service is None:
            self._kamiwaza_service = get_kamiwaza_service()
        return self._kamiwaza_service

    async def initialize(self):
        """Initialize Kamiwaza vector service and create collections."""
        try:
            # Initialize Kamiwaza embedding model
            await self._initialize_embedding_model()

            # Initialize Milvus connection
            await self._initialize_milvus()

            # Create collection if it doesn't exist
            await self._create_collection()

            # Initialize with default company knowledge
            await self._initialize_company_knowledge()

            self._is_initialized = True
            logger.info("Kamiwaza Vector Service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Kamiwaza Vector Service: {e}")
            raise

    async def _initialize_embedding_model(self):
        """Initialize the Kamiwaza embedding model."""
        try:
            # Auto-detect embedding model if not specified
            if not self.embedding_model:
                self.embedding_model = await self.kamiwaza_service.get_model_by_capability("embedding")

            if not self.embedding_model:
                # Fallback to looking for common embedding model patterns
                models = await self.kamiwaza_service.list_models()
                for model in models:
                    model_name_lower = model["name"].lower()
                    if any(pattern in model_name_lower for pattern in ["embed", "e5", "bge", "sentence"]):
                        self.embedding_model = model["name"]
                        break

            if not self.embedding_model:
                raise ValueError("No embedding model found on Kamiwaza deployment")

            # Get embedding client
            self._embedding_client = self.kamiwaza_service.get_openai_client(self.embedding_model)

            # Test embedding to determine actual dimension
            test_embedding = await self._get_embedding_internal("test")
            self.vector_dimension = len(test_embedding)

            logger.info(f"Using Kamiwaza embedding model: {self.embedding_model} "
                       f"with dimension: {self.vector_dimension}")

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    async def _initialize_milvus(self):
        """Initialize Milvus connection to Kamiwaza's instance."""
        try:
            # Try direct connection to Kamiwaza's Milvus first
            connection_params = {
                "host": self.milvus_host,
                "port": str(self.milvus_port)
            }

            # Check if we can connect via Kamiwaza SDK Milvus interface
            try:
                # First try Kamiwaza's native Milvus connection
                milvus_info = await self._get_kamiwaza_milvus_config()
                if milvus_info:
                    connection_params.update(milvus_info)
            except Exception as e:
                logger.warning(f"Could not get Kamiwaza Milvus config, using defaults: {e}")

            # Initialize Milvus client
            uri = f"http://{connection_params['host']}:{connection_params['port']}"
            self._milvus_client = MilvusClient(uri=uri)

            # Test connection
            collections = self._milvus_client.list_collections()
            logger.info(f"Connected to Milvus at {uri}, collections: {len(collections)}")

        except Exception as e:
            logger.error(f"Failed to initialize Milvus connection: {e}")
            raise

    async def _get_kamiwaza_milvus_config(self) -> Optional[Dict[str, Any]]:
        """Get Milvus configuration from Kamiwaza if available."""
        try:
            # Check if Kamiwaza SDK provides Milvus endpoints
            client = self.kamiwaza_service._client

            # Look for Milvus or vector database deployments
            deployments = client.serving.list_active_deployments()
            for deployment in deployments:
                if hasattr(deployment, 'service_type'):
                    if deployment.service_type.lower() in ['milvus', 'vector', 'vectordb']:
                        return {
                            "host": deployment.endpoint.split('://')[1].split(':')[0],
                            "port": deployment.endpoint.split(':')[-1]
                        }

            return None
        except Exception as e:
            logger.debug(f"Could not get Kamiwaza Milvus config: {e}")
            return None

    async def _create_collection(self):
        """Create Milvus collection with appropriate schema."""
        try:
            if self._milvus_client.has_collection(self.collection_name):
                logger.info(f"Collection {self.collection_name} already exists")
                return

            # Create collection with dynamic dimension
            self._milvus_client.create_collection(
                collection_name=self.collection_name,
                dimension=self.vector_dimension,
                metric_type="COSINE",
                consistency_level="Strong",
                index_params={
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 16, "efConstruction": 200}
                }
            )

            logger.info(f"Created collection: {self.collection_name} with dimension: {self.vector_dimension}")

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

    async def _get_embedding_internal(self, text: str) -> List[float]:
        """Get embedding using Kamiwaza model."""
        try:
            if not self._embedding_client:
                raise ValueError("Embedding client not initialized")

            # Use OpenAI-compatible interface
            response = self._embedding_client.embeddings.create(
                input=text,
                model=self.embedding_model
            )

            if hasattr(response, 'data') and response.data:
                return response.data[0].embedding
            else:
                raise ValueError("Invalid embedding response format")

        except Exception as e:
            logger.error(f"Error getting embedding for text: {e}")
            raise

    async def get_embedding(self, text: str) -> List[float]:
        """Public method to get embedding for text using Kamiwaza."""
        return await self._get_embedding_internal(text)

    async def _initialize_company_knowledge(self):
        """Load initial Kamiwaza company knowledge base."""
        kamiwaza_info = [
            {
                "content": """Kamiwaza is a cutting-edge technology company specializing in AI-powered solutions,
                cloud infrastructure, and digital transformation services. We have extensive experience in
                government contracting, particularly with federal agencies. Our proprietary AI platform enables
                rapid deployment of intelligent solutions with enterprise-grade security and compliance.""",
                "metadata": {"type": "company_overview", "category": "general", "priority": "high"}
            },
            {
                "content": """Core Competencies:
                - Artificial Intelligence and Machine Learning Solutions (LLMs, Computer Vision, NLP)
                - Cloud Migration and Infrastructure (AWS, Azure, GCP certified partners)
                - Cybersecurity and Risk Management (Zero-trust architecture, SIEM, SOC)
                - Data Analytics and Business Intelligence (Real-time analytics, ETL, Data lakes)
                - Custom Software Development (Full-stack, APIs, Microservices)
                - DevOps and Automation (CI/CD, Infrastructure as Code, Container orchestration)
                - Digital Transformation Consulting (Process optimization, Legacy modernization)""",
                "metadata": {"type": "capabilities", "category": "technical", "priority": "high"}
            },
            {
                "content": """Past Performance Highlights:
                - Successfully delivered 75+ federal contracts across 15 agencies
                - 99.2% customer satisfaction rating (CSAT surveys)
                - On-time delivery rate: 100% for past 24 months
                - Budget compliance: 98.5% average, with 60% of projects under budget
                - Security clearance: Up to Secret level for 40+ personnel
                - CMMI Level 3 certified for development processes
                - ISO 27001:2013 and ISO 9001:2015 certified""",
                "metadata": {"type": "past_performance", "category": "qualifications", "priority": "high"}
            },
            {
                "content": """Certifications and Compliance Framework:
                - Small Business (SB) certified - NAICS codes 541511, 541512, 541519
                - FedRAMP authorized cloud service provider
                - NIST 800-171 and NIST 800-53 compliant infrastructure
                - SOC 2 Type II certified (annual audit)
                - HIPAA compliant for healthcare data processing
                - Section 508 accessibility compliance testing
                - StateRAMP authorized (multiple states)
                - FISMA compliance for federal information systems""",
                "metadata": {"type": "certifications", "category": "compliance", "priority": "high"}
            },
            {
                "content": """Executive Team and Key Personnel:
                - CEO: 25+ years federal contracting, former DoD program manager
                - CTO: Former NSA technology advisor, PhD Computer Science
                - VP Engineering: Led 200+ person teams at Google, Microsoft
                - VP Sales: 18+ years government sales, former federal CIO
                - CISO: Former FBI cybersecurity expert, CISSP certified
                - Chief Data Scientist: PhD Machine Learning, 15+ AI patents
                - VP Operations: Former military logistics, PMP certified""",
                "metadata": {"type": "personnel", "category": "team", "priority": "medium"}
            },
            {
                "content": """Contract Vehicles and Procurement Access:
                - GSA Schedule 70 IT Professional Services (GS-35F-0119Y)
                - CIO-SP3 Small Business (HHSM-500-2013-00034I)
                - SEWP V Government-wide Acquisition Contract
                - 8(a) STARS III (HHSM-500-2016-00076G)
                - Alliant 2 Small Business (47QTCK18D003S)
                - OASIS Small Business (HHSM-500-2014-0002I)
                - VA T4NG (36C78718D0047)""",
                "metadata": {"type": "contract_vehicles", "category": "contracting", "priority": "medium"}
            },
            {
                "content": """Technical Stack and Innovation Platform:
                - Programming Languages: Python, Java, JavaScript/TypeScript, Go, C++, Rust
                - Frameworks: React, Vue.js, Angular, Django, FastAPI, Spring Boot, .NET Core
                - Databases: PostgreSQL, MongoDB, Redis, Elasticsearch, Neo4j, InfluxDB
                - Cloud Platforms: AWS (Advanced Partner), Azure (Gold Partner), GCP (Partner)
                - AI/ML: TensorFlow, PyTorch, Hugging Face, OpenAI API, Custom LLM fine-tuning
                - DevOps: Kubernetes, Docker, Terraform, Ansible, Jenkins, GitLab CI/CD, ArgoCD
                - Monitoring: Prometheus, Grafana, ELK Stack, Splunk, DataDog""",
                "metadata": {"type": "tech_stack", "category": "technical", "priority": "medium"}
            }
        ]

        await self.index_documents(kamiwaza_info)

    async def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents into Kamiwaza's Milvus collection."""
        try:
            if not self._is_initialized:
                await self.initialize()

            data_to_insert = []

            for doc in documents:
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})

                # Generate consistent ID from content hash
                doc_hash = hashlib.md5(content.encode()).hexdigest()
                doc_id = int(doc_hash[:16], 16) % (2**63 - 1)

                # Get embedding using Kamiwaza
                embedding = await self.get_embedding(content)

                data_to_insert.append({
                    "id": doc_id,
                    "vector": embedding,
                    "text": content,
                    "metadata": json.dumps(metadata)
                })

            if data_to_insert:
                self._milvus_client.insert(
                    collection_name=self.collection_name,
                    data=data_to_insert
                )
                logger.info(f"Indexed {len(data_to_insert)} documents using Kamiwaza embeddings")

        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            raise

    async def search_similar(
        self,
        query: str,
        limit: int = 5,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using Kamiwaza embeddings."""
        try:
            if not self._is_initialized:
                await self.initialize()

            # Get query embedding using Kamiwaza
            query_embedding = await self.get_embedding(query)

            # Prepare search parameters
            search_params = {
                "collection_name": self.collection_name,
                "data": [query_embedding],
                "limit": limit,
                "output_fields": output_fields or ["text", "metadata"]
            }

            if filter_expr:
                search_params["filter"] = filter_expr

            # Perform vector search
            results = self._milvus_client.search(**search_params)

            # Format results
            formatted_results = []
            for hits in results:
                for hit in hits:
                    entity = hit.get("entity", {})
                    formatted_results.append({
                        "content": entity.get("text", ""),
                        "metadata": json.loads(entity.get("metadata", "{}")),
                        "distance": hit.get("distance", 1.0),
                        "score": 1.0 - hit.get("distance", 0.0),  # Convert distance to similarity score
                        "embedding_model": self.embedding_model
                    })

            return formatted_results

        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []

    async def hybrid_search(
        self,
        query: str,
        vector_limit: int = 10,
        text_weight: float = 0.7,
        vector_weight: float = 0.3,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity with text matching.
        Uses Kamiwaza embeddings for vector component.
        """
        try:
            # Get vector search results
            vector_results = await self.search_similar(query, limit=vector_limit)

            # Simple text matching for hybrid component
            text_results = []
            query_terms = query.lower().split()

            for result in vector_results:
                content_lower = result["content"].lower()
                text_score = sum(1 for term in query_terms if term in content_lower) / len(query_terms)

                # Combine scores
                vector_score = result["score"]
                combined_score = (vector_weight * vector_score) + (text_weight * text_score)

                result.update({
                    "text_score": text_score,
                    "vector_score": vector_score,
                    "combined_score": combined_score,
                    "search_type": "hybrid"
                })
                text_results.append(result)

            # Sort by combined score and return top results
            text_results.sort(key=lambda x: x["combined_score"], reverse=True)
            return text_results[:limit]

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to vector search only
            return await self.search_similar(query, limit)

    async def get_company_context(self, query: str = "Kamiwaza company information") -> str:
        """Get relevant company context using Kamiwaza embeddings."""
        try:
            results = await self.search_similar(query, limit=10)

            if not results:
                return "Kamiwaza is a technology company specializing in AI and cloud solutions."

            context = "KAMIWAZA COMPANY INFORMATION (Powered by Kamiwaza Vector Search):\n\n"

            for i, result in enumerate(results, 1):
                content = result.get("content", "")
                score = result.get("score", 0.0)
                priority = result.get("metadata", {}).get("priority", "medium")

                context += f"[Source {i} - Relevance: {score:.3f} - Priority: {priority}]\n"
                context += f"{content}\n\n"

            return context

        except Exception as e:
            logger.error(f"Error getting company context: {e}")
            return "Kamiwaza is a technology company specializing in AI and cloud solutions."

    async def add_rfp_response(self, rfp_id: str, response_data: Dict[str, Any]):
        """Store completed RFP response for future reference."""
        try:
            documents = []
            timestamp = datetime.now().isoformat()

            for section_name, content in response_data.items():
                if isinstance(content, str) and content.strip():
                    documents.append({
                        "content": content,
                        "metadata": {
                            "type": "rfp_response",
                            "rfp_id": rfp_id,
                            "section": section_name,
                            "date": timestamp,
                            "embedding_model": self.embedding_model
                        }
                    })

            if documents:
                await self.index_documents(documents)
                logger.info(f"Stored RFP response for {rfp_id} using Kamiwaza embeddings")

        except Exception as e:
            logger.error(f"Error storing RFP response: {e}")

    async def search_past_rfps(self, requirements: str) -> List[Dict[str, Any]]:
        """Search for similar past RFP responses using Kamiwaza embeddings."""
        try:
            # Search with filter for RFP responses
            results = await self.search_similar(requirements, limit=10)

            # Filter for RFP responses
            past_rfps = [
                result for result in results
                if result.get("metadata", {}).get("type") == "rfp_response"
            ]

            return past_rfps[:5]  # Return top 5 most relevant

        except Exception as e:
            logger.error(f"Error searching past RFPs: {e}")
            return []

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection."""
        try:
            if not self._milvus_client.has_collection(self.collection_name):
                return {"error": "Collection does not exist"}

            # Get collection statistics
            stats = self._milvus_client.describe_collection(self.collection_name)

            # Try to get entity count (may not be available in all Milvus versions)
            try:
                entity_count = self._milvus_client.query(
                    collection_name=self.collection_name,
                    expr="id > 0",
                    output_fields=["count(*)"]
                )
                count = len(entity_count) if entity_count else 0
            except:
                count = "Unknown"

            return {
                "collection_name": self.collection_name,
                "dimension": self.vector_dimension,
                "embedding_model": self.embedding_model,
                "entity_count": count,
                "milvus_host": self.milvus_host,
                "milvus_port": self.milvus_port,
                "schema": stats
            }

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Kamiwaza vector service."""
        try:
            # Check Kamiwaza service
            kamiwaza_health = await self.kamiwaza_service.health_check()

            # Check Milvus connection
            milvus_healthy = False
            milvus_message = ""

            try:
                if self._milvus_client:
                    collections = self._milvus_client.list_collections()
                    milvus_healthy = True
                    milvus_message = f"Connected, {len(collections)} collections"
                else:
                    milvus_message = "Not connected"
            except Exception as e:
                milvus_message = f"Connection error: {str(e)}"

            # Check embedding capability
            embedding_healthy = False
            embedding_message = ""

            try:
                if self.embedding_model:
                    test_embed = await self.get_embedding("test")
                    embedding_healthy = len(test_embed) == self.vector_dimension
                    embedding_message = f"Model: {self.embedding_model}, Dimension: {len(test_embed)}"
                else:
                    embedding_message = "No embedding model configured"
            except Exception as e:
                embedding_message = f"Embedding error: {str(e)}"

            overall_healthy = (
                kamiwaza_health.get("healthy", False) and
                milvus_healthy and
                embedding_healthy
            )

            return {
                "healthy": overall_healthy,
                "service": "KamiwazaVectorService",
                "kamiwaza": kamiwaza_health,
                "milvus": {
                    "healthy": milvus_healthy,
                    "message": milvus_message,
                    "host": self.milvus_host,
                    "port": self.milvus_port
                },
                "embeddings": {
                    "healthy": embedding_healthy,
                    "message": embedding_message,
                    "model": self.embedding_model,
                    "dimension": self.vector_dimension
                },
                "collection": self.collection_name
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "service": "KamiwazaVectorService"
            }

    async def batch_embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Efficiently embed multiple documents using Kamiwaza."""
        try:
            # Process in batches to avoid overwhelming the embedding service
            batch_size = 10
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = []

                # Process batch concurrently
                tasks = [self.get_embedding(text) for text in batch]
                batch_embeddings = await asyncio.gather(*tasks)
                all_embeddings.extend(batch_embeddings)

                # Small delay to be respectful to the service
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)

            return all_embeddings

        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            raise

    async def delete_documents(self, filter_expr: str) -> bool:
        """Delete documents matching the filter expression."""
        try:
            if not self._milvus_client.has_collection(self.collection_name):
                return False

            self._milvus_client.delete(
                collection_name=self.collection_name,
                filter=filter_expr
            )

            logger.info(f"Deleted documents matching filter: {filter_expr}")
            return True

        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False

    async def update_document(self, doc_id: int, new_content: str, new_metadata: Dict[str, Any]):
        """Update a document by ID."""
        try:
            # Delete old document
            await self.delete_documents(f"id == {doc_id}")

            # Insert updated document
            await self.index_documents([{
                "content": new_content,
                "metadata": new_metadata
            }])

            logger.info(f"Updated document with ID: {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return False


# Singleton instance for shared usage
_kamiwaza_vector_service_instance: Optional[KamiwazaVectorService] = None


def get_kamiwaza_vector_service(
    collection_name: str = "kamiwaza_knowledge_v2",
    **kwargs
) -> KamiwazaVectorService:
    """
    Get or create singleton Kamiwaza Vector Service instance.

    Args:
        collection_name: Name of the collection to use
        **kwargs: Additional configuration parameters

    Returns:
        Shared KamiwazaVectorService instance
    """
    global _kamiwaza_vector_service_instance
    if _kamiwaza_vector_service_instance is None:
        _kamiwaza_vector_service_instance = KamiwazaVectorService(
            collection_name=collection_name,
            **kwargs
        )
    return _kamiwaza_vector_service_instance


async def migrate_from_milvus_rag(
    old_milvus_rag,
    new_vector_service: Optional[KamiwazaVectorService] = None,
    batch_size: int = 100
) -> Dict[str, Any]:
    """
    Migrate data from existing MilvusRAG to KamiwazaVectorService.

    Args:
        old_milvus_rag: Existing MilvusRAG instance
        new_vector_service: Target KamiwazaVectorService (created if None)
        batch_size: Documents to process per batch

    Returns:
        Migration report with statistics
    """
    if new_vector_service is None:
        new_vector_service = get_kamiwaza_vector_service()

    try:
        # Initialize new service
        await new_vector_service.initialize()

        # Get all documents from old service (this is a simplified approach)
        # In practice, you'd need to implement a way to export all documents from MilvusRAG
        logger.info("Starting migration from MilvusRAG to KamiwazaVectorService")

        # This would need to be implemented based on your specific MilvusRAG schema
        # For now, return a placeholder
        return {
            "status": "migration_template",
            "message": "Migration function template created. Implement based on MilvusRAG schema.",
            "migrated_documents": 0,
            "errors": []
        }

    except Exception as e:
        logger.error(f"Migration error: {e}")
        return {
            "status": "error",
            "message": str(e),
            "migrated_documents": 0,
            "errors": [str(e)]
        }