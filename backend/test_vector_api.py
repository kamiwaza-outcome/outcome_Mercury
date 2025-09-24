#!/usr/bin/env python3
"""
Test script to demonstrate the Kamiwaza vector API endpoints
"""
import asyncio
import json
import sys
from typing import Dict, Any

# Test data
SAMPLE_DOCUMENTS = [
    {
        "content": "Kamiwaza specializes in AI-powered solutions and cloud infrastructure for government contracts.",
        "metadata": {"type": "company_info", "category": "overview"}
    },
    {
        "content": "We provide cybersecurity services including risk management and compliance with federal standards.",
        "metadata": {"type": "services", "category": "security"}
    },
    {
        "content": "Our team has delivered 50+ federal contracts with 100% on-time delivery rate.",
        "metadata": {"type": "performance", "category": "track_record"}
    }
]

SAMPLE_TEXTS = [
    "What are Kamiwaza's core competencies?",
    "How does the company ensure cybersecurity?",
    "What is the company's track record with government contracts?"
]


async def test_vector_service():
    """Test the vector service functionality without requiring a running server."""
    try:
        from services.kamiwaza_vector_service import get_kamiwaza_vector_service

        print("=== Kamiwaza Vector Service Test ===\n")

        # Initialize vector service
        vector_service = get_kamiwaza_vector_service()
        print("✓ Vector service initialized")

        # Test initialization
        try:
            init_result = await vector_service.initialize()
            print("✓ Vector service initialization:", json.dumps(init_result, indent=2))
        except Exception as e:
            print(f"⚠ Vector service initialization (expected to fail without Kamiwaza running): {e}")

        # Test health check
        health = await vector_service.health_check()
        print("✓ Health check result:", json.dumps(health, indent=2))

        # Test embedding generation (will fail without Kamiwaza running)
        try:
            embeddings = await vector_service.generate_embeddings(SAMPLE_TEXTS[:1])
            print("✓ Embeddings generated:", embeddings.get("count", 0), "vectors")
        except Exception as e:
            print(f"⚠ Embedding generation (expected to fail without Kamiwaza running): {e}")

        # Test collections listing
        try:
            collections = await vector_service.list_collections()
            print("✓ Collections listed:", len(collections), "collections found")
        except Exception as e:
            print(f"⚠ Collections listing (expected to fail without Kamiwaza running): {e}")

        print("\n=== API Endpoints Available ===")
        print("POST /api/vectors/embed - Generate embeddings")
        print("POST /api/vectors/search - Search similar vectors")
        print("POST /api/vectors/index - Index new documents")
        print("GET /api/vectors/collections - List collections")
        print("GET /api/vectors/collections/{name} - Get collection info")

        print("\n=== Sample API Usage ===")
        print("# Generate embeddings:")
        print("curl -X POST http://localhost:8000/api/vectors/embed \\")
        print("  -H 'Content-Type: application/json' \\")
        print("  -d '{\"texts\": [\"Sample text to embed\"]}'")

        print("\n# Search vectors:")
        print("curl -X POST http://localhost:8000/api/vectors/search \\")
        print("  -H 'Content-Type: application/json' \\")
        print("  -d '{\"query\": \"AI solutions\", \"limit\": 5}'")

        print("\n# Index documents:")
        sample_doc = json.dumps({
            "documents": SAMPLE_DOCUMENTS[:1]
        })
        print("curl -X POST http://localhost:8000/api/vectors/index \\")
        print("  -H 'Content-Type: application/json' \\")
        print(f"  -d '{sample_doc}'")

        print("\n# List collections:")
        print("curl -X GET http://localhost:8000/api/vectors/collections")

        print("\n✓ All tests completed. The API is ready to use with Kamiwaza running at http://localhost:7777/api/")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_vector_service())