"""
Simple backend API with Kamiwaza integration for Mercury Blue ALLY
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
import os
import logging
from datetime import datetime

# Import our Kamiwaza-powered services
from services.openai_service import OpenAIService
from services.kamiwaza_vector_service import get_kamiwaza_vector_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mercury RFP API - Kamiwaza Powered")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3003", "http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ProcessRequest(BaseModel):
    text: str
    model_name: Optional[str] = None
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7

class ProcessResponse(BaseModel):
    result: str
    model_used: str
    timestamp: str

# Vector API Models
class EmbeddingRequest(BaseModel):
    texts: Union[str, List[str]]
    model: Optional[str] = None

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimension: int
    count: int

class VectorSearchRequest(BaseModel):
    query: str
    collection: Optional[str] = None
    limit: int = 5
    threshold: float = 0.7

class VectorSearchResponse(BaseModel):
    query: str
    collection: str
    results: List[Dict[str, Any]]
    count: int
    model_used: str

class IndexDocumentsRequest(BaseModel):
    documents: List[Dict[str, Any]]
    collection: Optional[str] = None

class IndexDocumentsResponse(BaseModel):
    collection: str
    indexed_count: int
    model_used: str
    dimension: int

class CollectionInfo(BaseModel):
    name: str
    description: Optional[str] = None
    document_count: int = 0
    dimension: int = 0
    created_at: str

# Initialize services (now powered by Kamiwaza)
openai_service = None
vector_service = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global openai_service, vector_service
    try:
        openai_service = OpenAIService()
        logger.info("OpenAI service (Kamiwaza-powered) initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI service: {e}")
        openai_service = None

    try:
        vector_service = get_kamiwaza_vector_service()
        await vector_service.initialize()
        logger.info("Vector service (Kamiwaza-powered) initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize vector service: {e}")
        vector_service = None

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Mercury RFP Automation API - Powered by Kamiwaza",
        "status": "running",
        "endpoints": {
            "models": "/api/models",
            "process": "/api/process",
            "health": "/api/health",
            "vectors": {
                "embed": "/api/vectors/embed",
                "search": "/api/vectors/search",
                "index": "/api/vectors/index",
                "collections": "/api/vectors/collections"
            }
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    try:
        health_info = {}
        vector_health = {}

        if openai_service:
            health_info = await openai_service.health_check()

        if vector_service:
            vector_health = await vector_service.health_check()

        overall_healthy = (
            health_info.get("healthy", False) and
            vector_health.get("healthy", False)
        )

        return {
            "status": "healthy" if overall_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "kamiwaza": health_info,
            "vector_service": vector_health
        }
    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@app.get("/api/models")
async def list_models():
    """List available Kamiwaza models."""
    if not openai_service:
        raise HTTPException(status_code=503, detail="Kamiwaza service not initialized")

    try:
        models = await openai_service.list_available_models()
        if not models:
            raise HTTPException(status_code=503, detail="No Kamiwaza models available")

        return {
            "models": models,
            "default_model": openai_service.model,
            "fallback_model": openai_service.fallback_model,
            "total": len(models)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=503, detail=f"Kamiwaza connection failed: {str(e)}")

@app.post("/api/models/select")
async def select_model(request: Dict[str, Any]):
    """Select a model for the session."""
    try:
        if not openai_service:
            raise HTTPException(status_code=503, detail="Service not initialized")

        model_name = request.get("model_name")
        capability = request.get("capability")

        if not model_name and not capability:
            raise ValueError("Either model_name or capability must be provided")

        selected = await openai_service.select_model(model_name=model_name, capability=capability)
        openai_service.model = selected

        return {
            "selected_model": selected,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Failed to select model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process", response_model=ProcessResponse)
async def process_text(request: ProcessRequest):
    """Process text using Kamiwaza models."""
    if not openai_service:
        raise HTTPException(status_code=503, detail="Kamiwaza service not initialized")

    try:
        # Process with Kamiwaza
        result = await openai_service.get_completion(
            prompt=request.text,
            model=request.model_name,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        if not result:
            raise HTTPException(status_code=503, detail="Failed to get response from Kamiwaza")

        return ProcessResponse(
            result=result,
            model_used=request.model_name or openai_service.model,
            timestamp=datetime.utcnow().isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=503, detail=f"Kamiwaza processing failed: {str(e)}")

@app.get("/api/rfps/pending")
async def get_pending_rfps():
    """Get pending RFPs."""
    # Return empty list - no mock data
    return {
        "rfps": [],
        "total": 0
    }

@app.post("/api/rfps/process")
async def process_rfp(request: Dict[str, Any]):
    """Process an RFP using Kamiwaza models."""
    if not openai_service:
        raise HTTPException(status_code=503, detail="Kamiwaza service not initialized")

    notice_id = request.get("notice_id")
    if not notice_id:
        raise HTTPException(status_code=400, detail="notice_id is required")

    try:
        result = await openai_service.get_completion(
            prompt=f"Analyze RFP {notice_id} and provide key requirements",
            max_tokens=500
        )

        if not result:
            raise HTTPException(status_code=503, detail="Failed to analyze RFP with Kamiwaza")

        return {
            "notice_id": notice_id,
            "status": "processed",
            "analysis": result,
            "model_used": openai_service.model
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RFP processing failed: {e}")
        raise HTTPException(status_code=503, detail=f"RFP processing failed: {str(e)}")

# Vector API Endpoints
@app.post("/api/vectors/embed", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """Generate embeddings for text(s) using Kamiwaza local embedding models."""
    if not vector_service:
        raise HTTPException(status_code=503, detail="Vector service not initialized")

    try:
        result = await vector_service.generate_embeddings(
            texts=request.texts,
            model=request.model
        )

        return EmbeddingResponse(
            embeddings=result["embeddings"],
            model=result["model"],
            dimension=result["dimension"],
            count=result["count"]
        )

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(status_code=503, detail=f"Embedding generation failed: {str(e)}")

@app.post("/api/vectors/search", response_model=VectorSearchResponse)
async def search_vectors(request: VectorSearchRequest):
    """Search for similar vectors in Kamiwaza's Milvus database."""
    if not vector_service:
        raise HTTPException(status_code=503, detail="Vector service not initialized")

    try:
        result = await vector_service.search_vectors(
            query=request.query,
            collection=request.collection,
            limit=request.limit,
            threshold=request.threshold
        )

        return VectorSearchResponse(
            query=result["query"],
            collection=result["collection"],
            results=result["results"],
            count=result["count"],
            model_used=result["model_used"]
        )

    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise HTTPException(status_code=503, detail=f"Vector search failed: {str(e)}")

@app.post("/api/vectors/index", response_model=IndexDocumentsResponse)
async def index_documents(request: IndexDocumentsRequest):
    """Index documents into Kamiwaza's vector database."""
    if not vector_service:
        raise HTTPException(status_code=503, detail="Vector service not initialized")

    try:
        result = await vector_service.index_documents(
            documents=request.documents,
            collection=request.collection
        )

        return IndexDocumentsResponse(
            collection=result["collection"],
            indexed_count=result["indexed_count"],
            model_used=result["model_used"],
            dimension=result["dimension"]
        )

    except Exception as e:
        logger.error(f"Document indexing failed: {e}")
        raise HTTPException(status_code=503, detail=f"Document indexing failed: {str(e)}")

@app.get("/api/vectors/collections", response_model=List[CollectionInfo])
async def list_collections():
    """List all available vector collections in Kamiwaza's Milvus."""
    if not vector_service:
        raise HTTPException(status_code=503, detail="Vector service not initialized")

    try:
        collections = await vector_service.list_collections()

        return [
            CollectionInfo(
                name=col["name"],
                description=col.get("description"),
                document_count=col.get("document_count", 0),
                dimension=col.get("dimension", 0),
                created_at=col.get("created_at", "")
            )
            for col in collections
        ]

    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=503, detail=f"Failed to list collections: {str(e)}")

@app.get("/api/vectors/collections/{collection_name}")
async def get_collection_info(collection_name: str):
    """Get detailed information about a specific collection."""
    if not vector_service:
        raise HTTPException(status_code=503, detail="Vector service not initialized")

    try:
        collection_info = await vector_service.get_collection_info(collection_name)

        return CollectionInfo(
            name=collection_info["name"],
            description=collection_info.get("description"),
            document_count=collection_info.get("document_count", 0),
            dimension=collection_info.get("dimension", 0),
            created_at=collection_info.get("created_at", "")
        )

    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        raise HTTPException(status_code=404, detail=f"Collection not found or error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)