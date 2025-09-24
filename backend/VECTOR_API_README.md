# Kamiwaza Vector API

This document describes the new vector operations API that integrates with Kamiwaza's local embedding models and Milvus vector database.

## Overview

The vector API provides the following capabilities:
- Generate embeddings using local Kamiwaza embedding models
- Store and index documents in Kamiwaza's Milvus database
- Search for similar vectors/documents
- Manage collections

## Configuration

Add these environment variables to your `.env` file:

```env
# Kamiwaza Vector Service Configuration
KAMIWAZA_DEFAULT_COLLECTION=mercury_documents
KAMIWAZA_EMBEDDING_MODEL=text-embedding-3-large
```

## API Endpoints

### 1. Generate Embeddings
**POST** `/api/vectors/embed`

Generate embeddings for text(s) using Kamiwaza's local embedding models.

**Request:**
```json
{
  "texts": ["Sample text to embed", "Another text"],
  "model": "text-embedding-3-large"  // optional
}
```

**Response:**
```json
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
  "model": "text-embedding-3-large",
  "dimension": 3072,
  "count": 2
}
```

### 2. Search Vectors
**POST** `/api/vectors/search`

Search for similar vectors in the database.

**Request:**
```json
{
  "query": "AI solutions for government",
  "collection": "mercury_documents",  // optional
  "limit": 5,
  "threshold": 0.7
}
```

**Response:**
```json
{
  "query": "AI solutions for government",
  "collection": "mercury_documents",
  "results": [
    {
      "id": "doc_1",
      "content": "Document content...",
      "metadata": {"type": "company_info"},
      "score": 0.95
    }
  ],
  "count": 1,
  "model_used": "text-embedding-3-large"
}
```

### 3. Index Documents
**POST** `/api/vectors/index`

Index new documents into the vector database.

**Request:**
```json
{
  "documents": [
    {
      "content": "Kamiwaza provides AI solutions...",
      "metadata": {"type": "company_info", "source": "website"}
    }
  ],
  "collection": "mercury_documents"  // optional
}
```

**Response:**
```json
{
  "collection": "mercury_documents",
  "indexed_count": 1,
  "model_used": "text-embedding-3-large",
  "dimension": 3072
}
```

### 4. List Collections
**GET** `/api/vectors/collections`

List all available vector collections.

**Response:**
```json
[
  {
    "name": "mercury_documents",
    "description": "Default Mercury documents collection",
    "document_count": 150,
    "dimension": 3072,
    "created_at": "2024-01-01T00:00:00Z"
  }
]
```

### 5. Get Collection Info
**GET** `/api/vectors/collections/{collection_name}`

Get detailed information about a specific collection.

**Response:**
```json
{
  "name": "mercury_documents",
  "description": "Default Mercury documents collection",
  "document_count": 150,
  "dimension": 3072,
  "created_at": "2024-01-01T00:00:00Z"
}
```

## Example Usage

### Generate embeddings for RFP analysis
```bash
curl -X POST http://localhost:8000/api/vectors/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Government cloud security requirements", "Federal compliance standards"]
  }'
```

### Index company documents
```bash
curl -X POST http://localhost:8000/api/vectors/index \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "content": "Kamiwaza has extensive experience with federal agencies...",
        "metadata": {"type": "capability", "category": "government"}
      }
    ]
  }'
```

### Search for relevant company information
```bash
curl -X POST http://localhost:8000/api/vectors/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "cybersecurity expertise federal contracts",
    "limit": 3
  }'
```

## Integration with Kamiwaza

This API integrates with:
- **Kamiwaza at http://localhost:7777/api/** - For model access and embedding generation
- **Kamiwaza's Milvus database** - For vector storage and search
- **Local embedding models** - No external API calls required

## Error Handling

The API returns appropriate HTTP status codes:
- `200` - Success
- `400` - Bad request (invalid input)
- `404` - Collection not found
- `503` - Service unavailable (Kamiwaza not running)

## Testing

Run the test script to verify the API:
```bash
python test_vector_api.py
```

This will test all endpoints and provide example usage commands.