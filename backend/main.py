from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, AsyncGenerator
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import json
import logging
from datetime import datetime
import redis.asyncio as redis

from services.google_sheets import GoogleSheetsService
from services.browser_scraper import BrowserScraper
from services.sam_api import SamApiClient
from services.milvus_rag import MilvusRAG
from services.document_generator import DocumentGenerator
from services.orchestration_agent import OrchestrationAgent
from services.google_drive import GoogleDriveService
from services.google_drive_extractor import GoogleDriveRFPExtractor
from services.architecture_manager import ArchitectureManager, ProcessingMode, ArchitectureConfig, ProcessingResult
from services.multi_agent_orchestrator import MultiAgentOrchestrator
from services.multi_pass_orchestrator import MultiPassOrchestrator
from services.anti_template_engine import AntiTemplateEngine
from services.federal_rfp_agents import get_federal_agents
from services.comprehensive_review_agent import ComprehensiveReviewAgent
from services.refinement_analyzer import RefinementAnalyzer
from services.document_refiner import DocumentRefiner
from services.mission_control import MissionControl
from services.openai_service import OpenAIService, CIRCUIT_BREAKER
from agents.human_review_agent import HumanReviewAgent
from services.chatbot_service import ChatbotService
from services.enhanced_chatbot_service import EnhancedChatbotService

load_dotenv()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Mercury RFP Automation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RFPProcessRequest(BaseModel):
    sheet_id: Optional[str] = None
    force_process_all: bool = False
    architecture_mode: Optional[ProcessingMode] = None
    session_id: Optional[str] = None

class RFPStatus(BaseModel):
    notice_id: str
    title: str
    status: str
    progress: float
    message: str
    documents_generated: List[str] = []
    errors: List[str] = []
    architecture_used: Optional[str] = None
    quality_scores: Optional[Dict[str, float]] = None
    uniqueness_score: Optional[float] = None

class ArchitectureRequest(BaseModel):
    mode: ProcessingMode
    config: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None

class ComparisonRequest(BaseModel):
    notice_id: str
    include_metrics: bool = True

rfp_status_store: Dict[str, RFPStatus] = {}

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Initializing Mercury RFP Automation System...")

    try:
        # Initialize core services
        app.state.sheets_service = GoogleSheetsService()
        app.state.browser_scraper = BrowserScraper()
        app.state.sam_api = SamApiClient()
        app.state.milvus_rag = MilvusRAG()
        app.state.doc_generator = DocumentGenerator()
        app.state.drive_service = GoogleDriveService()
        try:
            await app.state.drive_service.initialize()
            logger.info("Google Drive service initialized successfully")
        except Exception as drive_error:
            logger.warning(f"Google Drive initialization failed: {drive_error}. Documents will be saved locally only.")
            app.state.drive_service = None
        app.state.drive_extractor = GoogleDriveRFPExtractor()

        # Initialize classic orchestration
        app.state.orchestration = OrchestrationAgent(
            doc_generator=app.state.doc_generator,
            milvus_rag=app.state.milvus_rag
        )

        # Initialize new multi-agent architecture
        app.state.multi_agent_orchestrator = MultiAgentOrchestrator(
            milvus_rag=app.state.milvus_rag
        )

        # Initialize enhanced multi-pass orchestrator (replaces primitive keyword matching)
        app.state.multi_pass_orchestrator = MultiPassOrchestrator()
        logger.info("Multi-pass orchestrator initialized with 5-pass processing")

        # Add federal specialized agents to multi-agent orchestrator
        federal_agents = get_federal_agents()
        app.state.multi_agent_orchestrator.agents.extend(federal_agents)
        logger.info(f"Added {len(federal_agents)} federal specialized agents")

        # Initialize architecture manager
        app.state.architecture_manager = ArchitectureManager(
            orchestration_agent=app.state.orchestration,
            multi_agent_orchestrator=app.state.multi_agent_orchestrator
        )

        # Initialize anti-template engine
        app.state.anti_template_engine = AntiTemplateEngine()

        # Initialize refinement services
        app.state.review_agent = ComprehensiveReviewAgent(milvus_rag=app.state.milvus_rag)
        app.state.refinement_analyzer = RefinementAnalyzer(milvus_rag=app.state.milvus_rag)
        app.state.document_refiner = DocumentRefiner(milvus_rag=app.state.milvus_rag)
        logger.info("Refinement services initialized")

        # Initialize Mission Control for real-time logging
        app.state.mission_control = MissionControl()
        await app.state.mission_control.initialize()
        logger.info("Mission Control initialized")

        # Initialize Human Review Agent
        app.state.human_review_agent = HumanReviewAgent()
        logger.info("Human Review Agent initialized")

        # Initialize Chatbot Service (keeping old one for compatibility)
        app.state.chatbot_service = ChatbotService(
            document_refiner=app.state.document_refiner,
            milvus_rag=app.state.milvus_rag
        )
        await app.state.chatbot_service.initialize()
        logger.info("Chatbot Service initialized")

        # Initialize Enhanced Chatbot Service with multi-agent system
        app.state.enhanced_chatbot = EnhancedChatbotService()
        await app.state.enhanced_chatbot.initialize()
        logger.info("Enhanced Chatbot Service with Multi-Agent System initialized")

        await app.state.milvus_rag.initialize()
        logger.info("All services initialized successfully")
        logger.info(f"Total agents available: {len(app.state.multi_agent_orchestrator.agents)}")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

@app.get("/")
async def root():
    return {"message": "Mercury RFP Automation System", "status": "operational"}

@app.get("/api/rfps/pending")
async def get_pending_rfps():
    """Get list of RFPs with checked boxes from Google Sheets"""
    try:
        # Check if sheets_service exists and initialize if needed
        if not hasattr(app.state, 'sheets_service'):
            logger.warning("sheets_service not found in app.state, initializing...")
            app.state.sheets_service = GoogleSheetsService()
            logger.info("sheets_service initialized successfully")

        # Use the app.state instance
        pending_rfps = app.state.sheets_service.get_checked_rfps()
        return {"rfps": pending_rfps, "count": len(pending_rfps)}
    except Exception as e:
        logger.error(f"Error fetching pending RFPs: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rfps/process")
async def process_rfps(request: RFPProcessRequest, background_tasks: BackgroundTasks):
    """Process checked RFPs from Google Sheets"""
    try:
        sheet_id = request.sheet_id or os.getenv("TRACKING_SHEET_ID")
        pending_rfps = app.state.sheets_service.get_checked_rfps(sheet_id)
        
        if not pending_rfps:
            return {"message": "No RFPs with checked boxes found", "count": 0}
        
        for rfp in pending_rfps:
            # Log what we're getting from Google Sheets
            logger.info(f"RFP data from sheets: {rfp}")

            # Skip RFPs with "Failed" status unless forcing
            if not request.force_process_all and rfp.get('status', '').lower() == 'failed':
                logger.warning(f"Skipping RFP {rfp.get('notice_id')} with Failed status - requires manual intervention")
                continue

            # Ensure URL is set before adding to background task
            background_tasks.add_task(
                process_single_rfp,
                rfp,
                architecture_mode=request.architecture_mode,
                session_id=request.session_id
            )
            rfp_status_store[rfp['notice_id']] = RFPStatus(
                notice_id=rfp['notice_id'],
                title=rfp['title'],
                status="queued",
                progress=0.0,
                message="RFP queued for processing"
            )
        
        return {
            "message": f"Started processing {len(pending_rfps)} RFPs",
            "rfps": [rfp['notice_id'] for rfp in pending_rfps]
        }
    except Exception as e:
        logger.error(f"Error processing RFPs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_single_rfp(
    rfp_data: Dict[str, Any],
    architecture_mode: Optional[ProcessingMode] = None,
    session_id: Optional[str] = None
):
    """Process a single RFP through the entire pipeline"""
    notice_id = rfp_data['notice_id']

    # Clean up old output files before starting new processing
    output_dir = Path(f"output/{notice_id}")
    if output_dir.exists():
        import shutil
        logger.info(f"Cleaning up old output files for {notice_id}")
        shutil.rmtree(output_dir)

    try:
        rfp_status_store[notice_id].status = "downloading"
        rfp_status_store[notice_id].progress = 0.1
        rfp_status_store[notice_id].message = "Downloading RFP documents..."
        
        rfp_documents = {}
        rfp_metadata = {}

        # Check data source - prefer Google Drive if available
        if rfp_data.get('data_source') == 'google_drive' and rfp_data.get('drive_folder_url'):
            logger.info(f"Using Google Drive extraction for {notice_id}")
            logger.info(f"Drive folder URL: {rfp_data['drive_folder_url']}")
            try:
                rfp_documents, rfp_metadata = await app.state.drive_extractor.extract_from_folder(
                    rfp_data['drive_folder_url'],
                    notice_id
                )
                logger.info(f"Successfully extracted {len(rfp_documents)} documents from Google Drive")
                # Add title from sheets data if not in metadata
                if 'title' not in rfp_metadata and 'title' in rfp_data:
                    rfp_metadata['title'] = rfp_data['title']
            except Exception as e:
                logger.error(f"Google Drive extraction failed for {notice_id}: {e}")
                # When Google Drive is configured but fails, we should error out rather than using wrong data
                error_msg = f"Google Drive extraction failed: {e}. Please ensure the Drive folder contains RFP documents."
                rfp_status_store[notice_id].status = "failed"
                rfp_status_store[notice_id].error = error_msg
                raise Exception(error_msg)

            # Check if Google Drive returned no documents
            if not rfp_documents:
                error_msg = f"Google Drive folder appears to be empty for {notice_id}. Please ensure RFP documents are in the folder: {rfp_data.get('drive_folder_url')}"
                logger.error(error_msg)
                rfp_status_store[notice_id].status = "failed"
                rfp_status_store[notice_id].error = error_msg
                raise Exception(error_msg)

        # Only use SAM API/Browser if Google Drive is NOT configured
        elif rfp_data.get('data_source') != 'google_drive':
            # Check if SAM API key is configured
            sam_api_available = bool(os.getenv("SAM_API_KEY"))

            # Try SAM API first if available (it's faster and more reliable when it works)
            if sam_api_available and not rfp_data.get('url'):
                logger.info(f"Attempting to use SAM API for {notice_id} (no URL provided)")
                try:
                    # Try to get metadata from SAM API
                    sam_metadata = await app.state.sam_api.get_opportunity(notice_id)
                    if sam_metadata and sam_metadata.get('ui_link'):
                        # If we got a URL from SAM API, use it for Browser Use
                        rfp_data['url'] = sam_metadata['ui_link']
                        logger.info(f"Got URL from SAM API: {rfp_data['url']}")

                    # Also try to get attachments from SAM API
                    sam_documents = await app.state.sam_api.download_attachments(notice_id)
                    if sam_documents:
                        rfp_documents.update(sam_documents)
                        logger.info(f"Downloaded {len(sam_documents)} documents from SAM API")

                    rfp_metadata.update(sam_metadata or {})
                except Exception as e:
                    logger.warning(f"SAM API failed for {notice_id}: {e}")
            elif not sam_api_available:
                logger.info("SAM API key not configured, using Browser Use only")
        
            # Use Browser Use for web scraping (either as primary or to supplement SAM API)
            if rfp_data.get('url'):
                logger.info(f"Using Browser Use to scrape RFP from URL: {rfp_data.get('url')}")
                try:
                    browser_documents, browser_metadata = await app.state.browser_scraper.scrape_rfp(
                        rfp_data.get('url'), notice_id
                    )
                    # Merge Browser Use results with any SAM API results
                    rfp_documents.update(browser_documents)
                    rfp_metadata.update(browser_metadata)
                    logger.info(f"Browser Use successfully scraped RFP {notice_id}: {len(browser_documents)} documents")
                except Exception as e:
                    logger.error(f"Browser Use failed for {notice_id}: {e}")
                    # If we have some data from SAM API, continue; otherwise fail
                    if not rfp_documents:
                        raise Exception(f"Both Browser Use and SAM API failed: {e}")
                    else:
                        logger.warning(f"Browser Use failed but continuing with {len(rfp_documents)} SAM API documents")
            elif not rfp_documents:
                # No URL and no documents from SAM API
                error_msg = f"No URL provided for {notice_id} and SAM API didn't return documents"
                logger.error(error_msg)
                raise Exception(error_msg)
        
        rfp_status_store[notice_id].progress = 0.3
        rfp_status_store[notice_id].status = "analyzing"
        rfp_status_store[notice_id].message = "Creating Northstar document..."
        
        northstar_doc = await app.state.orchestration.create_northstar_document(
            rfp_documents=rfp_documents,
            rfp_metadata=rfp_metadata,
            company_context=await app.state.milvus_rag.get_company_context()
        )
        
        rfp_status_store[notice_id].progress = 0.5
        rfp_status_store[notice_id].status = "generating"
        rfp_status_store[notice_id].message = "Generating response documents..."

        # Get architecture configuration
        if session_id:
            config = await app.state.architecture_manager.get_architecture_config(
                session_id,
                user_preference=architecture_mode
            )
        else:
            # Default to classic if no session
            config = ArchitectureConfig(
                mode=architecture_mode or ProcessingMode.CLASSIC,
                session_id=session_id or f"default_{notice_id}"
            )

        # Process with selected architecture
        if config.mode in [ProcessingMode.CLASSIC, ProcessingMode.AB_TEST]:
            # Use classic orchestration for classic and A/B test modes (A/B will randomly assign)
            generated_documents = await app.state.orchestration.generate_rfp_response(
                northstar_document=northstar_doc,
                rfp_documents=rfp_documents,
                notice_id=notice_id
            )
            rfp_status_store[notice_id].architecture_used = "classic"
        elif config.mode == ProcessingMode.DYNAMIC:
            # Use enhanced multi-pass orchestrator for dynamic mode (5-pass processing)
            logger.info(f"DYNAMIC MODE: Using Multi-Pass Orchestrator for {notice_id}")
            logger.info(f"Multi-pass orchestrator exists: {hasattr(app.state, 'multi_pass_orchestrator')}")

            # Get company context
            company_context = await app.state.milvus_rag.get_company_context()

            # Execute 5-pass orchestration
            logger.info(f"Calling multi_pass_orchestrator.orchestrate()")
            orchestrator_result = await app.state.multi_pass_orchestrator.orchestrate(
                northstar_document=northstar_doc,
                rfp_content=json.dumps(rfp_documents),  # Convert to string for processing
                company_context={'context': company_context, 'metadata': rfp_metadata},
                existing_documents=None
            )

            if orchestrator_result.success:
                generated_documents = orchestrator_result.documents
                rfp_status_store[notice_id].architecture_used = "dynamic (5-pass)"
                rfp_status_store[notice_id].quality_scores = {
                    "confidence": orchestrator_result.confidence_score,
                    "requirements_covered": orchestrator_result.total_requirements_covered,
                    "gaps_remaining": orchestrator_result.total_gaps_remaining
                }
                logger.info(f"Multi-pass orchestration completed with {len(generated_documents)} documents")
            else:
                logger.error(f"Multi-pass orchestration failed: {orchestrator_result.errors}")
                # Fallback to old multi-agent orchestrator
                result = await app.state.multi_agent_orchestrator.generate_rfp_response(
                    northstar=northstar_doc,
                    rfp_documents=rfp_documents,
                    notice_id=notice_id,
                    company_context="",
                    strategy='hybrid'
                )
                generated_documents = result['documents']
                rfp_status_store[notice_id].architecture_used = "dynamic (fallback)"
        elif config.mode == ProcessingMode.COMPARISON:
            # Run both architectures for comparison
            processing_result = await app.state.architecture_manager.process_rfp_with_architecture(
                rfp_data=rfp_data,
                config=config,
                northstar_doc=northstar_doc,
                rfp_documents=rfp_documents,
                company_context=""
            )
            # Use the better performing result
            if isinstance(processing_result, dict) and 'comparison' in processing_result:
                # Use dynamic result if quality is better
                if processing_result['comparison']['quality_improvement'] > 0:
                    generated_documents = processing_result['dynamic'].documents
                    rfp_status_store[notice_id].architecture_used = "dynamic (comparison)"
                else:
                    generated_documents = processing_result['classic'].documents
                    rfp_status_store[notice_id].architecture_used = "classic (comparison)"
                rfp_status_store[notice_id].quality_scores = processing_result['comparison']
            else:
                generated_documents = processing_result.documents
        else:
            # Fallback to classic
            generated_documents = await app.state.orchestration.generate_rfp_response(
                northstar_document=northstar_doc,
                rfp_documents=rfp_documents,
                notice_id=notice_id
            )
            rfp_status_store[notice_id].architecture_used = "classic (fallback)"
        
        rfp_status_store[notice_id].progress = 0.7
        rfp_status_store[notice_id].status = "reviewing"
        rfp_status_store[notice_id].message = "Running quality checks and generating review guide..."
        
        review_guide = await app.state.orchestration.generate_review_guide(
            generated_documents=generated_documents,
            northstar_document=northstar_doc
        )
        
        # Save documents locally first
        rfp_status_store[notice_id].progress = 0.85
        rfp_status_store[notice_id].status = "saving"
        rfp_status_store[notice_id].message = "Saving documents locally..."
        
        # Create local output directory
        output_dir = Path(f"output/{notice_id}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all documents locally with empty content validation
        for doc_name, doc_content in generated_documents.items():
            # Validate content is not empty
            content_to_save = doc_content if isinstance(doc_content, str) else json.dumps(doc_content, indent=2)

            if not content_to_save or len(content_to_save.strip()) < 100:
                logger.warning(f"Document {doc_name} appears empty or too small ({len(content_to_save) if content_to_save else 0} bytes), skipping save to prevent data loss")
                # Keep original file if it exists rather than overwriting with empty content
                continue

            # For MP4 files, save as .txt script instead
            if doc_name.endswith('.mp4'):
                file_path = output_dir / doc_name.replace('.mp4', '_script.txt')
                logger.info(f"Saving video script as text file: {file_path}")
            else:
                file_path = output_dir / doc_name

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content_to_save)
        
        # Save Northstar and review guide
        with open(output_dir / "Northstar_Document.md", 'w', encoding='utf-8') as f:
            f.write(northstar_doc)
        with open(output_dir / "Human_Review_Guide.md", 'w', encoding='utf-8') as f:
            f.write(review_guide)
        
        logger.info(f"Documents saved locally to {output_dir}")
        
        rfp_status_store[notice_id].progress = 0.9
        rfp_status_store[notice_id].status = "uploading"
        rfp_status_store[notice_id].message = "Uploading to Google Drive..."
        
        try:
            folder_id = await app.state.drive_service.create_rfp_folder(notice_id, rfp_data['title'])
        except Exception as drive_error:
            logger.error(f"Drive upload failed but documents saved locally: {drive_error}")
            rfp_status_store[notice_id].status = "completed_locally"
            rfp_status_store[notice_id].message = f"Documents saved locally to output/{notice_id}. Drive upload failed."
            rfp_status_store[notice_id].documents_generated = list(generated_documents.keys())
            return
        
        await app.state.drive_service.upload_documents(
            folder_id=folder_id,
            documents={
                **generated_documents,
                "Northstar_Document.md": northstar_doc,
                "Human_Review_Guide.md": review_guide
            }
        )
        
        rfp_status_store[notice_id].progress = 1.0
        rfp_status_store[notice_id].status = "completed"
        rfp_status_store[notice_id].message = f"RFP processing completed. Documents uploaded to Drive."
        rfp_status_store[notice_id].documents_generated = list(generated_documents.keys())

        await app.state.sheets_service.update_rfp_status(
            notice_id=notice_id,
            status="Processed",
            folder_url=f"https://drive.google.com/drive/folders/{folder_id}"
        )

        # Clear checkbox after successful processing
        try:
            row_number = rfp_data.get('row_number')
            if row_number:
                await app.state.sheets_service.clear_checkbox(row_number)
        except Exception as clear_err:
            logger.error(f"Failed to clear checkbox for {notice_id}: {clear_err}")

        # Auto-trigger Human Review Agent after successful completion
        try:
            logger.info(f"Auto-triggering review process for {notice_id}")
            await app.state.human_review_agent.auto_trigger_on_completion(notice_id, "completed")
        except Exception as review_error:
            logger.error(f"Failed to trigger review for {notice_id}: {review_error}")
            # Don't fail the whole process if review fails to start

        # Auto-launch enhanced chatbot with multi-agent system for human interaction
        try:
            logger.info(f"Auto-launching enhanced chatbot for {notice_id}")
            output_folder = f"output/{notice_id}"
            chat_result = await app.state.enhanced_chatbot.start_session_from_output(output_folder, notice_id)
            if chat_result["status"] == "success":
                logger.info(f"Enhanced chatbot launched successfully: {chat_result['chat_url']}")
                # Update status with chat URL
                rfp_status_store[notice_id].message += f" Chat available at: {chat_result['chat_url']}"
        except Exception as chat_error:
            logger.error(f"Failed to launch enhanced chatbot for {notice_id}: {chat_error}")
        
    except Exception as e:
        logger.error(f"Error processing RFP {notice_id}: {e}")
        rfp_status_store[notice_id].status = "failed"
        rfp_status_store[notice_id].message = f"Processing failed: {str(e)}"
        rfp_status_store[notice_id].errors.append(str(e))
        
        await app.state.sheets_service.update_rfp_status(
            notice_id=notice_id,
            status="Failed",
            error=str(e)
        )
        # Clear checkbox even on failure to prevent retry loops
        try:
            row_number = rfp_data.get('row_number')
            if row_number:
                await app.state.sheets_service.clear_checkbox(row_number)
        except Exception as clear_err:
            logger.error(f"Failed to clear checkbox after failure for {notice_id}: {clear_err}")

@app.get("/api/rfps/status/{notice_id}")
async def get_rfp_status(notice_id: str):
    """Get the processing status of a specific RFP"""
    if notice_id not in rfp_status_store:
        raise HTTPException(status_code=404, detail="RFP not found")
    return rfp_status_store[notice_id]

@app.get("/api/rfps/status")
async def get_all_status():
    """Get status of all RFPs being processed"""
    try:
        # Serialize Pydantic models to plain dicts
        rfps_list = []
        for status in rfp_status_store.values():
            try:
                rfps_list.append(status.dict())
            except Exception:
                # Pydantic v2 compatibility
                rfps_list.append(status.model_dump())
        return {"rfps": rfps_list}
    except Exception as e:
        logger.error(f"Error serializing RFP statuses: {e}")
        # Return best-effort minimal info
        fallback = [
            {
                "notice_id": k,
                "title": v.title if hasattr(v, 'title') else "",
                "status": v.status if hasattr(v, 'status') else "unknown",
                "progress": getattr(v, 'progress', 0.0),
                "message": getattr(v, 'message', ""),
                "documents_generated": getattr(v, 'documents_generated', []),
                "errors": getattr(v, 'errors', [])
            }
            for k, v in rfp_status_store.items()
        ]
        return {"rfps": fallback}

@app.post("/api/rag/index")
async def index_company_documents():
    """Index company documents for RAG"""
    try:
        result = await app.state.milvus_rag.index_company_documents()
        return {"message": "Documents indexed successfully", "details": result}
    except Exception as e:
        logger.error(f"Error indexing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint with light, real checks."""
    checks: Dict[str, Any] = {}

    # OpenAI check (tiny prompt, 1 token, fallback if circuit open)
    try:
        oai = OpenAIService()
        _ = await oai.get_completion(
            prompt="pong",
            model=oai.model,
            max_tokens=1,
            temperature=0.0,
        )
        checks["openai_api"] = True
    except Exception as e:
        logger.warning(f"OpenAI health check failed: {e}")
        checks["openai_api"] = False

    # Redis check: prefer chatbot_service client, else try ephemeral
    try:
        redis_ok = False
        if getattr(app.state, "chatbot_service", None) and getattr(app.state.chatbot_service, "redis_client", None):
            try:
                await app.state.chatbot_service.redis_client.ping()
                redis_ok = True
            except Exception:
                redis_ok = False
        else:
            host = os.getenv("REDIS_HOST", "localhost")
            port = int(os.getenv("REDIS_PORT", 6380))
            tmp = await redis.Redis(host=host, port=port, decode_responses=True)
            try:
                await tmp.ping()
                redis_ok = True
            finally:
                try:
                    await tmp.close()
                except Exception:
                    pass
        checks["redis"] = redis_ok
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")
        checks["redis"] = False

    # Socket availability (very lightweight heuristic)
    try:
        import resource
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        # Consider healthy if soft limit is above a minimum baseline
        checks["socket_availability"] = soft_limit >= 256
    except Exception:
        checks["socket_availability"] = True

    # GPT-5 Circuit Breaker
    try:
        checks["gpt5_circuit"] = CIRCUIT_BREAKER.is_healthy()
    except Exception:
        checks["gpt5_circuit"] = True

    return {
        "status": "healthy" if all(checks.values()) else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks,
    }

# Kamiwaza Model Management Endpoints
@app.get("/api/models")
async def list_available_models():
    """List all available models from Kamiwaza deployment."""
    try:
        oai_service = OpenAIService()
        models = await oai_service.list_available_models()
        return {
            "models": models,
            "default_model": oai_service.model,
            "fallback_model": oai_service.fallback_model,
            "total": len(models)
        }
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@app.post("/api/models/select")
async def select_model(request: Dict[str, Any]):
    """Select a model for the current session."""
    try:
        model_name = request.get("model_name")
        capability = request.get("capability")

        if not model_name and not capability:
            raise ValueError("Either model_name or capability must be provided")

        oai_service = OpenAIService()
        selected = await oai_service.select_model(model_name=model_name, capability=capability)

        # Update the default model for this session
        oai_service.model = selected

        return {
            "selected_model": selected,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Failed to select model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to select model: {str(e)}")

@app.get("/api/models/health")
async def check_kamiwaza_health():
    """Check health of Kamiwaza connection and models."""
    try:
        oai_service = OpenAIService()
        health_info = await oai_service.health_check()
        return health_info
    except Exception as e:
        logger.error(f"Kamiwaza health check failed: {e}")
        return {
            "healthy": False,
            "message": f"Health check failed: {str(e)}",
            "models_count": 0,
            "models": []
        }

# Architecture Management Endpoints

@app.post("/api/architecture/config")
async def set_architecture_config(request: ArchitectureRequest):
    """Set architecture configuration for a session"""
    try:
        session_id = request.session_id or f"session_{datetime.utcnow().timestamp()}"

        config = await app.state.architecture_manager.get_architecture_config(
            session_id=session_id,
            user_preference=request.mode
        )

        return {
            "session_id": session_id,
            "config": config.dict(),
            "message": f"Architecture set to {request.mode.value}"
        }
    except Exception as e:
        logger.error(f"Error setting architecture config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/architecture/config/{session_id}")
async def get_architecture_config(session_id: str):
    """Get architecture configuration for a session"""
    try:
        config = await app.state.architecture_manager.get_architecture_config(session_id)
        return {"session_id": session_id, "config": config.dict()}
    except Exception as e:
        logger.error(f"Error getting architecture config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/architecture/metrics")
async def get_architecture_metrics(notice_id: Optional[str] = None):
    """Get processing metrics for architecture comparison"""
    try:
        metrics = await app.state.architecture_manager.get_architecture_metrics(notice_id)
        return metrics
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/architecture/ab-test")
async def configure_ab_test(enabled: bool, rollout_percentage: int = 20):
    """Configure A/B testing for architecture selection"""
    try:
        app.state.architecture_manager.update_ab_test_config(enabled, rollout_percentage)
        return {
            "enabled": enabled,
            "rollout_percentage": rollout_percentage,
            "message": f"A/B testing {'enabled' if enabled else 'disabled'}"
        }
    except Exception as e:
        logger.error(f"Error configuring A/B test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/agents/list")
async def list_available_agents():
    """List all available specialized agents"""
    try:
        agents = app.state.multi_agent_orchestrator.agents
        agent_info = [
            {
                "name": agent.name,
                "specializations": agent.specializations
            }
            for agent in agents
        ]
        return {
            "total_agents": len(agents),
            "agents": agent_info
        }
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/uniqueness/check")
async def check_uniqueness(content: str, document_type: str = "general"):
    """Check uniqueness score of content"""
    try:
        uniqueness_report = await app.state.anti_template_engine.generate_uniqueness_report(
            content, document_type
        )
        return uniqueness_report
    except Exception as e:
        logger.error(f"Error checking uniqueness: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ProcessWithArchitectureRequest(BaseModel):
    notice_id: str
    session_id: str
    northstar_doc: str
    rfp_documents: Dict[str, Any]
    company_context: str = ""

@app.post("/api/rfps/process-with-architecture")
async def process_rfp_with_architecture(
    request: ProcessWithArchitectureRequest
):
    """Process RFP with selected architecture"""
    try:
        # Get the session's architecture config
        if request.session_id not in app.state.architecture_manager.session_configs:
            raise HTTPException(status_code=400, detail="Session not configured. Please set architecture first.")

        config = app.state.architecture_manager.session_configs[request.session_id]

        # Create RFP data structure
        rfp_data = {
            'notice_id': request.notice_id,
            'title': 'Custom RFP Processing',
            'description': request.rfp_documents.get('requirements', ''),
            'agency': 'Custom',
            'type': 'Custom Processing'
        }

        # Process with architecture manager directly
        result = await app.state.architecture_manager.process_rfp_with_architecture(
            rfp_data=rfp_data,
            config=config,
            northstar_doc=request.northstar_doc,
            rfp_documents=request.rfp_documents,
            company_context=request.company_context
        )

        # Return the result directly
        if isinstance(result, ProcessingResult):
            return {
                "notice_id": result.notice_id,
                "architecture_used": result.architecture_used,
                "documents": result.documents,
                "processing_time": result.processing_time,
                "quality_scores": result.quality_scores,
                "uniqueness_score": result.uniqueness_score,
                "agent_assignments": result.agent_assignments,
                "errors": result.errors,
                "documents_generated": len(result.documents)
            }
        else:
            # Comparison mode returns dict
            return result
    except Exception as e:
        logger.error(f"Error starting RFP processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_single_rfp_with_architecture(rfp_data: Dict[str, Any], config: ArchitectureConfig):
    """Process a single RFP with specified architecture"""
    notice_id = rfp_data['notice_id']

    try:
        # Initialize status
        rfp_status_store[notice_id] = RFPStatus(
            notice_id=notice_id,
            title=rfp_data.get('title', 'Unknown'),
            status="processing",
            progress=0.1,
            message=f"Starting with {config.mode.value} architecture",
            architecture_used=config.mode.value
        )

        # Extract RFP data (similar to existing process_single_rfp)
        if rfp_data.get('data_source') == 'google_drive' and rfp_data.get('drive_folder_url'):
            logger.info(f"Using Google Drive extraction for {notice_id}")
            rfp_documents = await app.state.drive_extractor.extract_rfp_data(
                rfp_data['drive_folder_url']
            )
        else:
            # Use existing extraction methods
            rfp_documents = {}

        # Create Northstar
        rfp_status_store[notice_id].progress = 0.3
        rfp_status_store[notice_id].message = "Creating Northstar analysis"

        company_context = await app.state.milvus_rag.get_company_context("")
        northstar_doc = await app.state.orchestration.create_northstar_document(
            rfp_documents, rfp_data, company_context
        )

        # Process with selected architecture
        rfp_status_store[notice_id].progress = 0.5
        rfp_status_store[notice_id].message = f"Generating documents with {config.mode.value} architecture"

        result = await app.state.architecture_manager.process_rfp_with_architecture(
            rfp_data, config, northstar_doc, rfp_documents, company_context
        )

        # Handle results based on mode
        if config.mode == ProcessingMode.COMPARISON:
            # Handle comparison results
            classic_result = result.get("classic")
            dynamic_result = result.get("dynamic")

            rfp_status_store[notice_id].documents_generated = list(dynamic_result.documents.keys())
            rfp_status_store[notice_id].quality_scores = dynamic_result.quality_scores
            rfp_status_store[notice_id].uniqueness_score = dynamic_result.uniqueness_score

            # Save both sets of documents
            await save_comparison_documents(notice_id, result)
        else:
            # Handle single architecture result
            generated_documents = result.documents

            # Apply anti-template engine
            for doc_name, content in generated_documents.items():
                enhanced_content = await app.state.anti_template_engine.ensure_uniqueness(
                    content, doc_name, rfp_data
                )
                generated_documents[doc_name] = enhanced_content

            rfp_status_store[notice_id].documents_generated = list(generated_documents.keys())
            rfp_status_store[notice_id].quality_scores = result.quality_scores
            rfp_status_store[notice_id].uniqueness_score = result.uniqueness_score

            # Save documents (existing logic)
            output_dir = Path(f"output/{notice_id}")
            output_dir.mkdir(parents=True, exist_ok=True)

            for doc_name, doc_content in generated_documents.items():
                file_path = output_dir / doc_name
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(doc_content if isinstance(doc_content, str) else json.dumps(doc_content, indent=2))

        rfp_status_store[notice_id].progress = 1.0
        rfp_status_store[notice_id].status = "completed"
        rfp_status_store[notice_id].message = f"Completed with {config.mode.value} architecture"

    except Exception as e:
        logger.error(f"Error processing RFP {notice_id}: {e}")
        rfp_status_store[notice_id].status = "failed"
        rfp_status_store[notice_id].message = f"Processing failed: {str(e)}"
        rfp_status_store[notice_id].errors.append(str(e))

async def save_comparison_documents(notice_id: str, comparison_results: Dict[str, Any]):
    """Save documents from comparison mode"""
    output_dir = Path(f"output/{notice_id}/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save classic results
    classic_dir = output_dir / "classic"
    classic_dir.mkdir(exist_ok=True)
    for doc_name, content in comparison_results["classic"].documents.items():
        file_path = classic_dir / doc_name
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content if isinstance(content, str) else json.dumps(content, indent=2))

    # Save dynamic results
    dynamic_dir = output_dir / "dynamic"
    dynamic_dir.mkdir(exist_ok=True)
    for doc_name, content in comparison_results["dynamic"].documents.items():
        file_path = dynamic_dir / doc_name
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content if isinstance(content, str) else json.dumps(content, indent=2))

    # Save comparison metrics
    metrics_file = output_dir / "comparison_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(comparison_results.get("comparison", {}), f, indent=2)

class ComparisonRequest(BaseModel):
    notice_id: str
    northstar_doc: str
    rfp_documents: Dict[str, Any]
    company_context: str = ""

class RefinementAnalysisRequest(BaseModel):
    notice_id: str
    documents: Dict[str, str]
    northstar_doc: Optional[str] = None
    company_context: Optional[str] = None

class RefinementAnswerRequest(BaseModel):
    session_id: str
    question_id: str
    answer: str

class RefinementSaveRequest(BaseModel):
    session_id: str
    notice_id: str

@app.post("/api/rfps/process-comparison")
async def process_rfps_comparison(request: ComparisonRequest):
    """Process RFP using comparison mode to run both architectures"""
    try:
        # Create comparison config
        config = ArchitectureConfig(
            mode=ProcessingMode.COMPARISON,
            session_id=f"comparison_{request.notice_id}",
            user_preference=True
        )

        # Create RFP data structure
        rfp_data = {
            'notice_id': request.notice_id,
            'title': 'Comparison Processing',
            'description': request.rfp_documents.get('requirements', ''),
            'agency': 'Comparison',
            'type': 'Comparison Processing'
        }

        # Process with comparison mode
        result = await app.state.architecture_manager.process_rfp_with_architecture(
            rfp_data=rfp_data,
            config=config,
            northstar_doc=request.northstar_doc,
            rfp_documents=request.rfp_documents,
            company_context=request.company_context
        )

        return result
    except Exception as e:
        logger.error(f"Error processing RFPs in comparison mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/refinement/analyze")
async def analyze_for_refinement(request: RefinementAnalysisRequest):
    """Analyze documents and start refinement session"""
    try:
        # First run comprehensive review
        review_result = await app.state.review_agent.review_document_suite(
            request.documents,
            request.northstar_doc,
            request.company_context
        )

        # Start refinement session
        session = await app.state.refinement_analyzer.start_refinement_session(
            request.notice_id,
            request.documents,
            review_result
        )

        return {
            "session_id": session["session_id"],
            "questions": session["questions"],
            "weaknesses": session["weaknesses"],
            "review_scores": review_result["overall_scores"],
            "critical_issues": review_result["critical_issues"]
        }
    except Exception as e:
        logger.error(f"Error analyzing for refinement: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/refinement/answer")
async def answer_refinement_question(request: RefinementAnswerRequest):
    """Process human answer to refinement question"""
    try:
        # Get current session
        session = await app.state.refinement_analyzer.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Process answer
        result = await app.state.refinement_analyzer.process_answer(
            request.session_id,
            request.question_id,
            request.answer
        )

        # Apply refinement to documents if enough answers collected
        if result.get("ready_for_refinement"):
            refinement_result = await app.state.document_refiner.refine_documents_with_feedback(
                request.session_id,
                session["documents"],
                result["answered_questions"],
                session["weaknesses"]
            )

            # Re-review after refinement
            new_review = await app.state.review_agent.review_document_suite(
                refinement_result["refined_documents"],
                session.get("northstar_doc"),
                session.get("company_context")
            )

            return {
                "status": "refined",
                "refined_documents": refinement_result["refined_documents"],
                "quality_metrics": refinement_result["quality_metrics"],
                "new_scores": new_review["overall_scores"],
                "score_improvement": {
                    doc: new_review["overall_scores"].get(doc, 0) - session["initial_scores"].get(doc, 0)
                    for doc in session["documents"].keys()
                },
                "next_questions": result.get("next_questions", [])
            }
        else:
            return {
                "status": "collecting",
                "progress": result["progress"],
                "next_questions": result.get("next_questions", []),
                "answered_count": len(result["answered_questions"])
            }
    except Exception as e:
        logger.error(f"Error processing refinement answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/refinement/review/{notice_id}")
async def get_review_report(notice_id: str):
    """Get comprehensive review report for a notice"""
    try:
        # Get documents from output directory
        output_dir = Path(f"output/{notice_id}")
        if not output_dir.exists():
            raise HTTPException(status_code=404, detail="Documents not found")

        documents = {}
        for file in output_dir.glob("*.md"):
            with open(file, 'r', encoding='utf-8') as f:
                documents[file.name] = f.read()

        for file in output_dir.glob("*.txt"):
            with open(file, 'r', encoding='utf-8') as f:
                documents[file.name] = f.read()

        # Run comprehensive review
        review_result = await app.state.review_agent.review_document_suite(documents)

        return review_result
    except Exception as e:
        logger.error(f"Error getting review report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/refinement/save-to-rag")
async def save_refinement_to_rag(request: RefinementSaveRequest):
    """Save refinement session to RAG for future learning"""
    try:
        # Get session data
        session = await app.state.refinement_analyzer.get_session(request.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Prepare data for RAG
        rag_data = {
            "notice_id": request.notice_id,
            "session_id": request.session_id,
            "timestamp": datetime.now().isoformat(),
            "questions_answered": session.get("answered_questions", []),
            "weaknesses_addressed": session.get("weaknesses", {}),
            "quality_improvements": session.get("quality_metrics", {}),
            "refined_documents": session.get("refined_documents", {})
        }

        # Save to output directory
        output_dir = Path(f"output/{request.notice_id}/refinement_sessions")
        output_dir.mkdir(parents=True, exist_ok=True)

        session_file = output_dir / f"session_{request.session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(rag_data, f, indent=2)

        # Add to Milvus RAG if available
        if app.state.milvus_rag:
            await app.state.milvus_rag.add_refinement_session(rag_data)

        return {
            "status": "saved",
            "session_id": request.session_id,
            "file_path": str(session_file),
            "questions_saved": len(rag_data["questions_answered"])
        }
    except Exception as e:
        logger.error(f"Error saving refinement to RAG: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mission Control SSE Endpoints
@app.get("/api/mission-control/stream/{session_id}")
async def stream_mission_control(session_id: str):
    """Stream real-time logs and AI analysis for a session"""
    async def event_generator():
        """Generate Server-Sent Events"""
        try:
            # Subscribe to Redis channels for this session
            redis_client = await redis.from_url("redis://localhost:6380", decode_responses=True)
            pubsub = redis_client.pubsub()

            # Subscribe to all three streams for this session
            await pubsub.subscribe(
                f"mission_control:{session_id}:logs",
                f"mission_control:{session_id}:ai_analysis",
                f"mission_control:{session_id}:patterns"
            )

            # Send initial connection event
            yield f"data: {json.dumps({'type': 'connected', 'session_id': session_id})}\n\n"

            # Stream events as they come
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    channel = message['channel']
                    data = json.loads(message['data'])

                    # Determine event type based on channel
                    if channel.endswith(':logs'):
                        event_type = 'log'
                    elif channel.endswith(':ai_analysis'):
                        event_type = 'ai_analysis'
                    elif channel.endswith(':patterns'):
                        event_type = 'pattern'
                    else:
                        event_type = 'unknown'

                    # Format SSE event
                    event = {
                        'type': event_type,
                        'session_id': session_id,
                        'timestamp': data.get('timestamp', datetime.now().isoformat()),
                        'data': data
                    }

                    yield f"data: {json.dumps(event)}\n\n"

        except Exception as e:
            logger.error(f"SSE streaming error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            if pubsub:
                await pubsub.unsubscribe()
                await pubsub.close()
            if redis_client:
                await redis_client.close()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable Nginx buffering
        }
    )

@app.post("/api/mission-control/start/{notice_id}")
async def start_mission_control(notice_id: str):
    """Start Mission Control monitoring for an RFP processing session"""
    try:
        session_id = await app.state.mission_control.start_session(
            session_id=f"mission_{notice_id}_{datetime.now().timestamp()}",
            metadata={
                "notice_id": notice_id,
                "rfp_title": rfp_status_store.get(notice_id, {}).get("title", "Unknown"),
                "architecture": rfp_status_store.get(notice_id, {}).get("architecture_used", "dynamic")
            }
        )

        return {
            "session_id": session_id,
            "notice_id": notice_id,
            "status": "monitoring_started",
            "stream_url": f"/api/mission-control/stream/{session_id}"
        }
    except Exception as e:
        logger.error(f"Error starting Mission Control: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/mission-control/stop/{session_id}")
async def stop_mission_control(session_id: str):
    """Stop Mission Control monitoring for a session"""
    try:
        summary = await app.state.mission_control.end_session(session_id)

        return {
            "session_id": session_id,
            "status": "monitoring_stopped",
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error stopping Mission Control: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/mission-control/sessions")
async def get_mission_control_sessions():
    """Get list of active Mission Control sessions"""
    try:
        sessions = app.state.mission_control.get_active_sessions()
        return {
            "sessions": sessions,
            "count": len(sessions)
        }
    except Exception as e:
        logger.error(f"Error getting Mission Control sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Human Review Agent Endpoints

class ReviewFeedbackRequest(BaseModel):
    notice_id: str
    feedback: Dict[str, Any]
    original_documents: Optional[Dict[str, Any]] = None

@app.post("/api/rfps/review/{notice_id}/start")
async def start_review_process(notice_id: str):
    """Manually start the review process for a completed RFP"""
    try:
        # Check if RFP exists and is completed
        if notice_id not in rfp_status_store:
            raise HTTPException(status_code=404, detail="RFP not found")

        if rfp_status_store[notice_id].status != "completed":
            raise HTTPException(status_code=400, detail=f"RFP must be completed first. Current status: {rfp_status_store[notice_id].status}")

        # Get documents for review
        output_dir = Path(f"output/{notice_id}")
        if not output_dir.exists():
            raise HTTPException(status_code=404, detail="RFP documents not found")

        documents = {}
        for file_path in output_dir.glob("*"):
            if file_path.is_file() and file_path.suffix in ['.md', '.txt', '.docx', '.xlsx', '.pdf']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        documents[file_path.name] = f.read()
                except:
                    # For binary files, just note their presence
                    documents[file_path.name] = f"[Binary file: {file_path.name}]"

        # Trigger review process
        review_result = await app.state.human_review_agent.trigger_review_process(notice_id, documents)

        return {
            "success": True,
            "notice_id": notice_id,
            "review_session": review_result,
            "questions_count": len(review_result.get("questions", []))
        }

    except Exception as e:
        logger.error(f"Error starting review process: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rfps/review/{notice_id}/questions")
async def get_review_questions(notice_id: str):
    """Get the review questions for a specific RFP"""
    try:
        # Check if review session exists
        review_file = Path(f"review_sessions/{notice_id}_review.json")
        if not review_file.exists():
            raise HTTPException(status_code=404, detail="Review session not found")

        with open(review_file, 'r') as f:
            review_data = json.load(f)

        return {
            "notice_id": notice_id,
            "questions": review_data.get("questions", []),
            "analysis": review_data.get("analysis", {}),
            "status": review_data.get("status", "unknown")
        }

    except Exception as e:
        logger.error(f"Error getting review questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/rfps/review/{notice_id}/feedback")
async def submit_review_feedback(notice_id: str, request: ReviewFeedbackRequest):
    """Submit human feedback for document improvements"""
    try:
        # Get original documents if not provided
        if not request.original_documents:
            output_dir = Path(f"output/{notice_id}")
            original_documents = {}
            for file_path in output_dir.glob("*"):
                if file_path.is_file() and file_path.suffix in ['.md', '.txt']:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        original_documents[file_path.name] = f.read()
            request.original_documents = original_documents

        # Process feedback through Human Review Agent
        improvement_result = await app.state.human_review_agent.process_review_feedback(
            notice_id,
            {
                **request.feedback,
                "original_documents": request.original_documents
            }
        )

        # Update status
        if notice_id in rfp_status_store:
            rfp_status_store[notice_id].status = "improved"
            rfp_status_store[notice_id].message = "Documents improved based on feedback"

        return {
            "success": True,
            "notice_id": notice_id,
            "improvement_result": improvement_result,
            "status": "improvements_completed"
        }

    except Exception as e:
        logger.error(f"Error submitting review feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rfps/review/{notice_id}/status")
async def get_review_status(notice_id: str):
    """Get the status of the review process for an RFP"""
    try:
        review_file = Path(f"review_sessions/{notice_id}_review.json")
        if not review_file.exists():
            return {
                "notice_id": notice_id,
                "status": "not_started",
                "message": "Review process has not been started for this RFP"
            }

        with open(review_file, 'r') as f:
            review_data = json.load(f)

        return {
            "notice_id": notice_id,
            "status": review_data.get("status", "unknown"),
            "questions_count": len(review_data.get("questions", [])),
            "timestamp": review_data.get("timestamp", ""),
            "analysis_summary": review_data.get("analysis", {}).get("analysis", "")[:500]  # First 500 chars
        }

    except Exception as e:
        logger.error(f"Error getting review status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Chatbot API Endpoints
class ChatStartRequest(BaseModel):
    folder_link: Optional[str] = None
    rfp_id: Optional[str] = None

class ChatMessageRequest(BaseModel):
    message: str

@app.post("/api/chat/start")
async def start_chat_session(request: ChatStartRequest):
    """Start a new chat session either from folder link or RFP ID"""
    try:
        if request.folder_link:
            # Process Google Drive folder link with multi-agent system
            result = await app.state.enhanced_chatbot.start_session_from_folder(request.folder_link)
        elif request.rfp_id:
            # Start chat for existing RFP
            output_folder = f"output/{request.rfp_id}"
            result = await app.state.enhanced_chatbot.start_session_from_output(output_folder, request.rfp_id)
        else:
            raise HTTPException(status_code=400, detail="Either folder_link or rfp_id required")

        return result

    except Exception as e:
        logger.error(f"Error starting chat session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/{session_id}/message")
async def send_chat_message(session_id: str, request: ChatMessageRequest):
    """Send a message in an active chat session"""
    try:
        result = await app.state.enhanced_chatbot.send_message(session_id, request.message)
        return result

    except Exception as e:
        logger.error(f"Error sending message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/{session_id}/status")
async def get_chat_status(session_id: str):
    """Get status of a chat session"""
    try:
        result = await app.state.enhanced_chatbot.get_session_status(session_id)
        return result

    except Exception as e:
        logger.error(f"Error getting chat status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/folder-review")
async def review_folder_directly(request: ChatStartRequest):
    """Process a Google Drive folder link and start human review immediately"""
    try:
        if not request.folder_link:
            raise HTTPException(status_code=400, detail="folder_link required")

        # Process the folder link with multi-agent system
        result = await app.state.enhanced_chatbot.start_session_from_folder(request.folder_link)

        # If successful, return the chat session info
        if result["status"] == "success":
            # Add chat_url if not present
            if "chat_url" not in result:
                result["chat_url"] = f"/chat/{result['session_id']}"

            return {
                "status": "success",
                "message": result.get("message", "Folder processed. Review chat started."),
                "session_id": result["session_id"],
                "chat_url": result.get("chat_url", f"/chat/{result['session_id']}"),
                "total_questions": result["total_questions"],
                "first_question": result["first_question"]
            }
        else:
            return result

    except Exception as e:
        logger.error(f"Error processing folder for review: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
