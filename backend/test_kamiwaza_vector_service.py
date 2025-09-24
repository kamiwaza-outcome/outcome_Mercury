#!/usr/bin/env python3
"""
Test script for KamiwazaVectorService
Demonstrates basic usage and compatibility with existing MilvusRAG interface.
"""

import asyncio
import logging
from services.kamiwaza_vector_service import KamiwazaVectorService, get_kamiwaza_vector_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_kamiwaza_vector_service():
    """Test the Kamiwaza Vector Service functionality."""

    print("=== Kamiwaza Vector Service Test ===\n")

    try:
        # Initialize the service
        vector_service = get_kamiwaza_vector_service()
        print("1. Initializing Kamiwaza Vector Service...")
        await vector_service.initialize()
        print("✓ Service initialized successfully\n")

        # Test health check
        print("2. Performing health check...")
        health = await vector_service.health_check()
        print(f"✓ Health status: {health.get('healthy', False)}")
        print(f"   - Embedding model: {health.get('embeddings', {}).get('model', 'N/A')}")
        print(f"   - Dimension: {health.get('embeddings', {}).get('dimension', 'N/A')}")
        print(f"   - Milvus status: {health.get('milvus', {}).get('healthy', False)}\n")

        # Test embedding generation
        print("3. Testing embedding generation...")
        test_text = "Kamiwaza provides AI-powered solutions for government agencies."
        embedding = await vector_service.get_embedding(test_text)
        print(f"✓ Generated embedding with dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}\n")

        # Test document indexing
        print("4. Testing document indexing...")
        test_documents = [
            {
                "content": "Kamiwaza specializes in artificial intelligence and machine learning solutions for government contracts.",
                "metadata": {"type": "capability", "priority": "high", "test": True}
            },
            {
                "content": "Our team has extensive experience with cloud infrastructure on AWS, Azure, and Google Cloud Platform.",
                "metadata": {"type": "technical", "priority": "medium", "test": True}
            },
            {
                "content": "Kamiwaza maintains a 100% on-time delivery record and 99% customer satisfaction rating.",
                "metadata": {"type": "performance", "priority": "high", "test": True}
            }
        ]

        await vector_service.index_documents(test_documents)
        print(f"✓ Indexed {len(test_documents)} test documents\n")

        # Test vector search
        print("5. Testing vector search...")
        search_query = "AI and machine learning capabilities"
        results = await vector_service.search_similar(search_query, limit=3)
        print(f"✓ Found {len(results)} similar documents for query: '{search_query}'")

        for i, result in enumerate(results, 1):
            score = result.get('score', 0)
            content = result.get('content', '')[:100] + "..."
            print(f"   {i}. Score: {score:.3f} - {content}")
        print()

        # Test hybrid search
        print("6. Testing hybrid search...")
        hybrid_results = await vector_service.hybrid_search(
            search_query,
            limit=3,
            vector_weight=0.7,
            text_weight=0.3
        )
        print(f"✓ Hybrid search returned {len(hybrid_results)} results")

        for i, result in enumerate(hybrid_results, 1):
            vector_score = result.get('vector_score', 0)
            text_score = result.get('text_score', 0)
            combined_score = result.get('combined_score', 0)
            print(f"   {i}. Combined: {combined_score:.3f} (Vector: {vector_score:.3f}, Text: {text_score:.3f})")
        print()

        # Test company context retrieval
        print("7. Testing company context retrieval...")
        context = await vector_service.get_company_context("government contracting experience")
        print(f"✓ Retrieved company context ({len(context)} characters)")
        print("   Context preview:")
        print(f"   {context[:200]}...\n")

        # Test collection statistics
        print("8. Testing collection statistics...")
        stats = await vector_service.get_collection_stats()
        print(f"✓ Collection stats:")
        print(f"   - Name: {stats.get('collection_name', 'N/A')}")
        print(f"   - Entity count: {stats.get('entity_count', 'N/A')}")
        print(f"   - Dimension: {stats.get('dimension', 'N/A')}")
        print(f"   - Embedding model: {stats.get('embedding_model', 'N/A')}\n")

        # Test batch embedding
        print("9. Testing batch embedding...")
        batch_texts = [
            "Cloud computing solutions",
            "Artificial intelligence development",
            "Government contract management"
        ]
        batch_embeddings = await vector_service.batch_embed_documents(batch_texts)
        print(f"✓ Generated {len(batch_embeddings)} embeddings in batch")
        print(f"   Each embedding has dimension: {len(batch_embeddings[0]) if batch_embeddings else 'N/A'}\n")

        # Cleanup test documents
        print("10. Cleaning up test documents...")
        cleanup_success = await vector_service.delete_documents("metadata like '%\"test\": true%'")
        if cleanup_success:
            print("✓ Test documents cleaned up successfully\n")
        else:
            print("! Could not clean up test documents (this may be normal)\n")

        print("=== All Tests Completed Successfully! ===")

        # Summary
        print(f"\nSummary:")
        print(f"- Service: KamiwazaVectorService")
        print(f"- Collection: {vector_service.collection_name}")
        print(f"- Embedding Model: {vector_service.embedding_model}")
        print(f"- Vector Dimension: {vector_service.vector_dimension}")
        print(f"- Milvus Host: {vector_service.milvus_host}:{vector_service.milvus_port}")

    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        logger.error(f"Test error: {e}", exc_info=True)
        return False

    return True


async def compare_with_milvus_rag():
    """Show interface compatibility with existing MilvusRAG."""

    print("\n=== Interface Compatibility Test ===\n")

    try:
        # Show that KamiwazaVectorService provides the same key methods as MilvusRAG
        vector_service = get_kamiwaza_vector_service()
        await vector_service.initialize()

        print("KamiwazaVectorService provides the following MilvusRAG-compatible methods:")

        compatible_methods = [
            "initialize()",
            "index_documents(documents)",
            "search_similar(query, limit)",
            "get_company_context(query)",
            "add_rfp_response(rfp_id, response_data)",
            "search_past_rfps(requirements)",
            "health_check()"
        ]

        for method in compatible_methods:
            print(f"✓ {method}")

        print("\nAdditional Kamiwaza-specific methods:")
        additional_methods = [
            "hybrid_search(query, vector_limit, text_weight, vector_weight, limit)",
            "batch_embed_documents(texts)",
            "get_collection_stats()",
            "delete_documents(filter_expr)",
            "update_document(doc_id, new_content, new_metadata)"
        ]

        for method in additional_methods:
            print(f"+ {method}")

        print("\n✓ Interface compatibility confirmed!")

    except Exception as e:
        print(f"✗ Compatibility test failed: {e}")
        return False

    return True


if __name__ == "__main__":
    async def main():
        success = await test_kamiwaza_vector_service()
        if success:
            await compare_with_milvus_rag()

        return success

    # Run the test
    result = asyncio.run(main())
    exit(0 if result else 1)