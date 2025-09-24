"""
Chatbot Service for RFP refinement and human interaction
Handles both automatic launch after document completion and folder link processing
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import re
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class ChatSession(BaseModel):
    """Chat session model"""
    session_id: str
    rfp_id: str
    folder_path: Optional[str] = None
    documents: Dict[str, str] = {}
    human_review_guide: Optional[str] = None
    questions_answered: List[Dict[str, Any]] = []
    current_question_index: int = 0
    status: str = "active"  # active, refining, completed
    started_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    last_activity: str = Field(default_factory=lambda: datetime.now().isoformat())

class ChatbotService:
    """Service for managing RFP refinement chatbot sessions"""

    def __init__(self, document_refiner=None, milvus_rag=None):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=float(os.getenv("OPENAI_TIMEOUT", "180")))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.document_refiner = document_refiner
        self.milvus_rag = milvus_rag

        # Redis for session management
        self.redis_client = None
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", 6380))

        # Session TTL (24 hours)
        self.session_ttl = 86400

    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = await redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            # Fallback to in-memory storage if Redis unavailable
            self.redis_client = None
            self.sessions = {}

    async def auto_launch_chat(self, rfp_id: str, output_folder: str) -> Dict[str, Any]:
        """
        Automatically launch chatbot after document generation completes
        """
        try:
            logger.info(f"Auto-launching chatbot for RFP {rfp_id}")

            # Load generated documents
            documents = await self._load_documents_from_folder(output_folder)

            # Load human review guide
            human_review_guide = await self._load_human_review_guide(output_folder)

            if not human_review_guide:
                logger.warning(f"No human review guide found for RFP {rfp_id}")
                return {
                    "status": "error",
                    "message": "Human review guide not found"
                }

            # Parse questions from human review guide
            questions = self._parse_questions_from_guide(human_review_guide)

            # Create chat session
            session = ChatSession(
                session_id=f"chat_{rfp_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                rfp_id=rfp_id,
                folder_path=output_folder,
                documents=documents,
                human_review_guide=human_review_guide
            )

            # Store session
            await self._store_session(session)

            # Get first question
            first_question = questions[0] if questions else None

            return {
                "status": "success",
                "session_id": session.session_id,
                "rfp_id": rfp_id,
                "total_questions": len(questions),
                "first_question": first_question,
                "message": f"Chat session started. {len(questions)} questions to review.",
                "chat_url": f"/chat/{session.session_id}"
            }

        except Exception as e:
            logger.error(f"Error auto-launching chat for RFP {rfp_id}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def process_folder_link(self, folder_link: str) -> Dict[str, Any]:
        """
        Process a Google Drive folder link and start chat session
        """
        try:
            logger.info(f"Processing folder link: {folder_link}")

            # Extract folder ID from link
            folder_id = self._extract_folder_id(folder_link)
            if not folder_id:
                return {
                    "status": "error",
                    "message": "Invalid folder link format"
                }

            # Download documents from Google Drive
            documents = await self._download_from_drive(folder_id)

            # Find and load human review guide
            human_review_guide = None
            for doc_name, content in documents.items():
                if "Human_Review_Guide" in doc_name:
                    human_review_guide = content
                    logger.info(f"Found Human Review Guide: {doc_name} ({len(content)} chars)")
                    logger.debug(f"Human Review Guide preview: {content[:500]}...")
                    break

            if not human_review_guide:
                return {
                    "status": "error",
                    "message": "No Human Review Guide found in folder"
                }

            # Use standard questions instead of parsing from guide
            from .standard_questions import get_essential_questions
            questions = get_essential_questions()

            logger.info(f"Using {len(questions)} standard essential questions for refinement")

            # Extract RFP ID from folder or generate one
            rfp_id = self._extract_rfp_id_from_documents(documents) or f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Create session
            session = ChatSession(
                session_id=f"chat_{rfp_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                rfp_id=rfp_id,
                folder_path=folder_link,
                documents=documents,
                human_review_guide=human_review_guide
            )

            # Store session
            await self._store_session(session)

            # Get first question
            first_question = questions[0] if questions else None

            return {
                "status": "success",
                "session_id": session.session_id,
                "rfp_id": rfp_id,
                "total_questions": len(questions),
                "first_question": first_question,
                "message": f"Chat session started from folder link. {len(questions)} questions to review.",
                "chat_url": f"/chat/{session.session_id}"
            }

        except Exception as e:
            logger.error(f"Error processing folder link: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def send_message(self, session_id: str, message: str) -> Dict[str, Any]:
        """
        Process a message in the chat session
        """
        try:
            # Load session
            session = await self._load_session(session_id)
            if not session:
                return {
                    "status": "error",
                    "message": "Session not found"
                }

            # Use standard questions
            from .standard_questions import get_essential_questions
            questions = get_essential_questions()

            # Store answer for current question
            if session.current_question_index < len(questions):
                current_question = questions[session.current_question_index]
                session.questions_answered.append({
                    "question": current_question["question"],
                    "answer": message,
                    "timestamp": datetime.now().isoformat()
                })
                session.current_question_index += 1
                session.last_activity = datetime.now().isoformat()

            # Check if all questions answered
            if session.current_question_index >= len(questions):
                # All questions answered, trigger refinement
                session.status = "refining"
                await self._store_session(session)

                # Start refinement process
                refinement_result = await self._trigger_refinement(session)

                return {
                    "status": "complete",
                    "message": "All questions answered. Starting document refinement...",
                    "refinement_status": refinement_result,
                    "next_question": None
                }

            # Get next question
            next_question = questions[session.current_question_index]

            # Store updated session
            await self._store_session(session)

            return {
                "status": "success",
                "message": "Answer recorded",
                "progress": {
                    "current": session.current_question_index,
                    "total": len(questions)
                },
                "next_question": next_question
            }

        except Exception as e:
            logger.error(f"Error processing message for session {session_id}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get current status of a chat session"""
        try:
            session = await self._load_session(session_id)
            if not session:
                return {
                    "status": "error",
                    "message": "Session not found"
                }

            questions = self._parse_questions_from_guide(session.human_review_guide)

            return {
                "status": "success",
                "session": {
                    "id": session.session_id,
                    "rfp_id": session.rfp_id,
                    "status": session.status,
                    "progress": {
                        "current": session.current_question_index,
                        "total": len(questions),
                        "percentage": (session.current_question_index / len(questions) * 100) if questions else 0
                    },
                    "started_at": session.started_at,
                    "last_activity": session.last_activity,
                    "current_question": questions[session.current_question_index] if session.current_question_index < len(questions) else None
                }
            }
        except Exception as e:
            logger.error(f"Error getting session status: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def _trigger_refinement(self, session: ChatSession) -> Dict[str, Any]:
        """Trigger document refinement with collected answers"""
        try:
            if not self.document_refiner:
                logger.warning("Document refiner not configured")
                return {
                    "status": "skipped",
                    "message": "Document refiner not available"
                }

            # Parse weaknesses from human review guide
            weaknesses = self._extract_weaknesses_from_guide(session.human_review_guide)

            # Call document refiner
            result = await self.document_refiner.refine_documents_with_feedback(
                session_id=session.session_id,
                documents=session.documents,
                answered_questions=session.questions_answered,
                weaknesses=weaknesses
            )

            # Update session status
            session.status = "completed"
            await self._store_session(session)

            # Save refined documents if folder path available
            if session.folder_path and not session.folder_path.startswith("http"):
                await self._save_refined_documents(
                    session.folder_path,
                    result['refined_documents']
                )

            return {
                "status": "success",
                "documents_refined": len(result['refined_documents']),
                "quality_metrics": result.get('quality_metrics', {}),
                "message": "Documents successfully refined with human feedback"
            }

        except Exception as e:
            logger.error(f"Error triggering refinement: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    def _parse_questions_from_guide_intelligent(self, guide_content: str) -> List[Dict[str, Any]]:
        """Parse questions intelligently from human review guide using GPT"""
        try:
            # Use GPT to extract actual questions that need human answers
            prompt = f"""Extract all questions that require human input from this Human Review Guide.
Only return questions that:
1. Actually need a human to provide information
2. Are not URLs or links
3. Are asking for specific details about the company or proposal

Human Review Guide Content:
{guide_content[:8000]}

Return as a JSON array with objects containing:
- question: The question text
- category: What category (pricing, technical, compliance, etc.)
- priority: high/medium/low

Only return the JSON array, no other text."""

            import asyncio
            import json

            response = asyncio.run(self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            ))

            questions_json = response.choices[0].message.content
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\[.*\]', questions_json, re.DOTALL)
            if json_match:
                questions_list = json.loads(json_match.group())

                # Convert to our format
                questions = []
                for q in questions_list:
                    questions.append({
                        "question": q.get("question", ""),
                        "category": q.get("category", "General"),
                        "context": "",
                        "type": q.get("priority", "medium")
                    })

                logger.info(f"Intelligently parsed {len(questions)} questions from Human Review Guide")
                return questions
        except Exception as e:
            logger.error(f"Error in intelligent parsing: {e}")

        # Fall back to the original parsing
        return self._parse_questions_from_guide(guide_content)

    def _parse_questions_from_guide(self, guide_content: str) -> List[Dict[str, Any]]:
        """Parse questions from human review guide"""
        questions = []

        # Look for questions section
        lines = guide_content.split('\n')
        in_questions = False
        current_category = ""

        for i, line in enumerate(lines):
            # Check for question sections or direct questions
            if "QUESTIONS" in line.upper() or "REVIEW" in line.upper() or "SPECIFIC ITEMS" in line.upper():
                in_questions = True
                # Get the category from this line if it contains a category name
                if ")" in line and line.startswith(tuple(str(i) for i in range(10))):
                    current_category = line.split(")", 1)[1].strip()
                continue

            # Also look for numbered sections that might contain questions
            if line.strip().startswith(tuple(str(i) + ")" for i in range(1, 10))):
                current_category = line.split(")", 1)[1].strip() if ")" in line else line.strip()
                in_questions = True
                continue

            if in_questions or True:  # Always scan for questions
                # Check for category headers
                if line.startswith('###') or line.startswith('##'):
                    current_category = line.replace('#', '').strip()
                    continue

                # Check for bullet points that look like questions
                if line.strip().startswith('- ') and '?' in line:
                    question_text = line.strip()[2:].strip()
                    # Skip if it's just a URL or doesn't look like a real question
                    if not question_text.startswith('http') and len(question_text) > 10:
                        questions.append({
                            "question": question_text,
                            "category": current_category,
                            "context": "",
                            "type": self._determine_question_type(question_text)
                        })
                    continue

                # Check for numbered questions
                question_match = re.match(r'^\d+\.\s+(.+)', line.strip())
                if question_match:
                    question_text = question_match.group(1)

                    # Extract context if available
                    context = ""
                    for next_line in lines[i+1:]:
                        if re.match(r'^\d+\.', next_line.strip()):
                            break
                        if next_line.strip().startswith('- ') or next_line.strip().startswith('* '):
                            context += next_line.strip() + " "

                    questions.append({
                        "question": question_text,
                        "category": current_category,
                        "context": context.strip(),
                        "type": self._determine_question_type(question_text)
                    })

        # If no questions found with specific format, extract all lines ending with ?
        if not questions:
            logger.info("No structured questions found, extracting all questions marked with ?")
            for line in lines:
                stripped = line.strip()
                # Must end with ?, not be a URL, and be a reasonable length
                if (stripped.endswith('?') and
                    not stripped.startswith('http') and
                    not stripped.startswith('Base:') and
                    10 < len(stripped) < 500):
                    questions.append({
                        "question": stripped,
                        "category": "General",
                        "context": "",
                        "type": "general"
                    })

        logger.info(f"Parsed {len(questions)} questions from Human Review Guide")
        return questions

    def _determine_question_type(self, question: str) -> str:
        """Determine the type of question for better handling"""
        question_lower = question.lower()

        if any(word in question_lower for word in ['specific', 'detail', 'example']):
            return 'specifics'
        elif any(word in question_lower for word in ['verify', 'accurate', 'correct']):
            return 'verification'
        elif any(word in question_lower for word in ['missing', 'add', 'include']):
            return 'addition'
        elif any(word in question_lower for word in ['clarify', 'explain']):
            return 'clarification'
        else:
            return 'general'

    def _extract_weaknesses_from_guide(self, guide_content: str) -> Dict[str, Any]:
        """Extract identified weaknesses from human review guide"""
        weaknesses = {
            "vague_statements": [],
            "missing_details": [],
            "unsubstantiated_claims": []
        }

        lines = guide_content.split('\n')
        current_section = None

        for line in lines:
            if "IDENTIFIED WEAKNESSES" in line.upper():
                current_section = "weaknesses"
            elif current_section == "weaknesses":
                if "vague" in line.lower():
                    weaknesses["vague_statements"].append(line.strip())
                elif "missing" in line.lower():
                    weaknesses["missing_details"].append(line.strip())
                elif "unsubstantiated" in line.lower() or "claim" in line.lower():
                    weaknesses["unsubstantiated_claims"].append(line.strip())

        return weaknesses

    def _extract_folder_id(self, folder_link: str) -> Optional[str]:
        """Extract folder ID from Google Drive link"""
        patterns = [
            r'/folders/([a-zA-Z0-9-_]+)',
            r'id=([a-zA-Z0-9-_]+)',
            r'/d/([a-zA-Z0-9-_]+)'
        ]

        for pattern in patterns:
            match = re.search(pattern, folder_link)
            if match:
                return match.group(1)

        return None

    def _extract_rfp_id_from_documents(self, documents: Dict[str, str]) -> Optional[str]:
        """Extract RFP ID from document content"""
        for doc_name, content in documents.items():
            # Look for notice ID or RFP number
            patterns = [
                r'Notice ID:\s*([A-Z0-9]+)',
                r'Solicitation No\.?\s*([A-Z0-9-]+)',
                r'RFP\s*#?\s*([A-Z0-9-]+)'
            ]

            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    return match.group(1)

        return None

    async def _load_documents_from_folder(self, folder_path: str) -> Dict[str, str]:
        """Load documents from local folder"""
        documents = {}

        try:
            folder = Path(folder_path)
            if folder.exists():
                # Load PDF and MD files
                for file_path in folder.glob('*.pdf'):
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        documents[file_path.name] = f.read()

                for file_path in folder.glob('*.md'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        documents[file_path.name] = f.read()
        except Exception as e:
            logger.error(f"Error loading documents from {folder_path}: {e}")

        return documents

    async def _load_human_review_guide(self, folder_path: str) -> Optional[str]:
        """Load human review guide from folder"""
        try:
            guide_path = Path(folder_path) / "Human_Review_Guide.md"
            if guide_path.exists():
                with open(guide_path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Error loading human review guide: {e}")

        return None

    async def _download_from_drive(self, folder_id: str) -> Dict[str, str]:
        """Download documents from Google Drive folder"""
        from .google_drive import GoogleDriveService

        try:
            # Initialize Google Drive service
            drive_service = GoogleDriveService()
            await drive_service.initialize()

            # Download all files from the folder
            logger.info(f"Downloading files from Drive folder {folder_id}")

            # Create a temporary directory for downloads
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="rfp_chat_")
            logger.info(f"Created temp directory: {temp_dir}")

            # Download files to temp directory
            downloaded_files = await drive_service.download_folder_contents(
                folder_id=folder_id,
                local_path=temp_dir
            )

            # Read all downloaded files into memory
            documents = {}
            for file_path in downloaded_files:
                try:
                    file_name = os.path.basename(file_path)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        documents[file_name] = content
                        logger.info(f"Loaded document: {file_name} ({len(content)} chars)")
                except Exception as e:
                    logger.warning(f"Could not read file {file_path}: {e}")
                    # Try binary read for non-text files
                    try:
                        with open(file_path, 'rb') as f:
                            content = f.read()
                            # For binary files, store as base64
                            import base64
                            documents[file_name] = base64.b64encode(content).decode('utf-8')
                            logger.info(f"Loaded binary document: {file_name}")
                    except Exception as e2:
                        logger.error(f"Failed to read file {file_path}: {e2}")

            logger.info(f"Downloaded {len(documents)} documents from Google Drive")
            return documents

        except Exception as e:
            logger.error(f"Error downloading from Google Drive: {e}")
            # Return empty dict if download fails
            return {}

    async def _save_refined_documents(self, folder_path: str, refined_docs: Dict[str, str]):
        """Save refined documents to folder"""
        try:
            folder = Path(folder_path)
            refined_folder = folder / "refined"
            refined_folder.mkdir(exist_ok=True)

            for doc_name, content in refined_docs.items():
                file_path = refined_folder / doc_name
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                logger.info(f"Saved refined document: {file_path}")

        except Exception as e:
            logger.error(f"Error saving refined documents: {e}")

    async def _store_session(self, session: ChatSession):
        """Store session in Redis or memory"""
        session_data = session.model_dump_json()

        if self.redis_client:
            await self.redis_client.setex(
                f"chat_session:{session.session_id}",
                self.session_ttl,
                session_data
            )
        else:
            # Fallback to memory
            if not hasattr(self, 'sessions'):
                self.sessions = {}
            self.sessions[session.session_id] = session_data

    async def _load_session(self, session_id: str) -> Optional[ChatSession]:
        """Load session from Redis or memory"""
        if self.redis_client:
            session_data = await self.redis_client.get(f"chat_session:{session_id}")
            if session_data:
                return ChatSession.model_validate_json(session_data)
        else:
            # Fallback to memory
            if hasattr(self, 'sessions') and session_id in self.sessions:
                return ChatSession.model_validate_json(self.sessions[session_id])

        return None
