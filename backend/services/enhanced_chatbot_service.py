"""
Enhanced Chatbot Service using Multi-Agent Conversational System
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import hashlib

# Import the multi-agent system
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from agents.conversational import OrchestrationAgent

logger = logging.getLogger(__name__)

class EnhancedChatbotService:
    """Enhanced chatbot service with multi-agent conversational intelligence"""

    def __init__(self):
        # Initialize the orchestration agent
        self.orchestrator = OrchestrationAgent()

        # Session management
        self.active_sessions = {}

        # Configuration
        self.auto_regenerate = True
        self.max_session_duration = 3600  # 1 hour

    async def initialize(self):
        """Initialize the service"""
        logger.info("Initializing Enhanced Chatbot Service with Multi-Agent System")
        return True

    async def start_session_from_folder(self, folder_link: str) -> Dict[str, Any]:
        """Start a chat session from a Google Drive folder link"""
        try:
            logger.info(f"Starting session from folder: {folder_link}")

            # Extract folder ID
            folder_id = self._extract_folder_id(folder_link)
            if not folder_id:
                return {
                    "status": "error",
                    "message": "Invalid Google Drive folder link"
                }

            # Download documents from folder
            documents = await self._download_from_drive(folder_id)

            # Find Human Review Guide
            human_review_guide = None
            for doc_name, content in documents.items():
                if "Human_Review_Guide" in doc_name or "human_review" in doc_name.lower():
                    human_review_guide = content
                    logger.info(f"Found Human Review Guide: {doc_name}")
                    break

            # Initialize session with orchestrator
            session_id = await self.orchestrator.initialize_session(
                documents=documents,
                review_guide=human_review_guide
            )

            # Start conversation
            conversation_data = await self.orchestrator.start_conversation(session_id)

            # Store session
            self.active_sessions[session_id] = {
                "folder_link": folder_link,
                "started_at": datetime.now().isoformat(),
                "documents": list(documents.keys())
            }

            return {
                "status": "success",
                "session_id": session_id,
                "message": conversation_data["greeting"],
                "first_question": conversation_data["first_question"],
                "total_questions": conversation_data["total_questions"],
                "has_review_guide": human_review_guide is not None
            }

        except Exception as e:
            logger.error(f"Error starting session from folder: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def start_session_from_output(self, output_folder: str, rfp_id: str) -> Dict[str, Any]:
        """Auto-launch chatbot after document generation"""
        try:
            logger.info(f"Auto-launching chatbot for RFP {rfp_id}")

            # Load documents from output folder
            documents = await self._load_documents_from_folder(output_folder)

            # Load Human Review Guide
            human_review_guide = await self._load_human_review_guide(output_folder)

            # Initialize session with orchestrator
            session_id = await self.orchestrator.initialize_session(
                documents=documents,
                review_guide=human_review_guide
            )

            # Start conversation
            conversation_data = await self.orchestrator.start_conversation(session_id)

            # Store session
            self.active_sessions[session_id] = {
                "rfp_id": rfp_id,
                "output_folder": output_folder,
                "started_at": datetime.now().isoformat(),
                "documents": list(documents.keys())
            }

            return {
                "status": "success",
                "session_id": session_id,
                "rfp_id": rfp_id,
                "message": conversation_data["greeting"],
                "first_question": conversation_data["first_question"],
                "total_questions": conversation_data["total_questions"],
                "chat_url": f"/chat/{session_id}"
            }

        except Exception as e:
            logger.error(f"Error auto-launching chat: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def send_message(self, session_id: str, message: str) -> Dict[str, Any]:
        """Process user message in conversation"""
        try:
            if session_id not in self.active_sessions:
                # Try to recover session from orchestrator
                try:
                    result = await self.orchestrator.process_user_input(session_id, message)
                except:
                    return {
                        "status": "error",
                        "message": "Session not found or expired"
                    }

            # Process message with orchestrator
            result = await self.orchestrator.process_user_input(session_id, message)

            # Check if conversation is complete
            if result["action"] == "complete":
                # Trigger completion flow
                completion_result = await self.complete_conversation(session_id)
                return {
                    "status": "complete",
                    "response": result["response"],
                    "completion_data": completion_result
                }

            # Return response with next question
            return {
                "status": "success",
                "response": result["response"],
                "next_question": result.get("next_question"),
                "progress": result.get("progress", 0),
                "questions_remaining": result.get("questions_remaining", 0),
                "action": result.get("action")
            }

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            # Try to recover gracefully
            error_result = await self.orchestrator.handle_error(session_id, str(e))
            return error_result

    async def complete_conversation(self, session_id: str) -> Dict[str, Any]:
        """Complete conversation and optionally regenerate documents"""
        try:
            # Get completion data from orchestrator
            completion_data = await self.orchestrator.complete_conversation(session_id)

            # Check if we should auto-regenerate
            if self.auto_regenerate and session_id in self.active_sessions:
                session_info = self.active_sessions[session_id]

                # Load original documents
                if "output_folder" in session_info:
                    documents = await self._load_documents_from_folder(session_info["output_folder"])
                elif "folder_link" in session_info:
                    folder_id = self._extract_folder_id(session_info["folder_link"])
                    documents = await self._download_from_drive(folder_id)
                else:
                    documents = {}

                if documents:
                    # Regenerate documents
                    regeneration_result = await self.orchestrator.regenerate_documents(
                        session_id=session_id,
                        original_documents=documents
                    )

                    # Save regenerated documents
                    output_path = await self._save_regenerated_documents(
                        session_id,
                        regeneration_result["regenerated_documents"],
                        regeneration_result["new_review_guide"]
                    )

                    return {
                        "status": "success",
                        "transcript": completion_data["transcript"],
                        "summary": completion_data["summary"],
                        "regenerated": True,
                        "output_path": output_path,
                        "readiness": regeneration_result["readiness_assessment"],
                        "improvements": regeneration_result["improvements_made"],
                        "gaps_remaining": regeneration_result["gaps_remaining"]
                    }

            # Return completion data without regeneration
            return {
                "status": "success",
                "transcript": completion_data["transcript"],
                "summary": completion_data["summary"],
                "regenerated": False,
                "message": "Conversation complete. Documents not regenerated."
            }

        except Exception as e:
            logger.error(f"Error completing conversation: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get current session status"""
        try:
            if session_id not in self.active_sessions:
                return {
                    "status": "error",
                    "message": "Session not found"
                }

            session_info = self.active_sessions[session_id]

            # Add orchestrator status
            # This would need to be implemented in orchestrator

            return {
                "status": "success",
                "session_id": session_id,
                "started_at": session_info["started_at"],
                "documents": session_info["documents"],
                "active": True
            }

        except Exception as e:
            logger.error(f"Error getting session status: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def skip_question(self, session_id: str) -> Dict[str, Any]:
        """Skip current question"""
        return await self.send_message(session_id, "skip")

    async def end_conversation(self, session_id: str) -> Dict[str, Any]:
        """End conversation early"""
        return await self.send_message(session_id, "done")

    # Helper methods

    def _extract_folder_id(self, folder_link: str) -> Optional[str]:
        """Extract folder ID from Google Drive link"""
        import re
        patterns = [
            r'/folders/([a-zA-Z0-9-_]+)',
            r'id=([a-zA-Z0-9-_]+)',
            r'^([a-zA-Z0-9-_]+)$'
        ]

        for pattern in patterns:
            match = re.search(pattern, folder_link)
            if match:
                return match.group(1)

        return None

    async def _download_from_drive(self, folder_id: str) -> Dict[str, str]:
        """Download documents from Google Drive folder"""
        from .google_drive import GoogleDriveService

        try:
            drive_service = GoogleDriveService()
            await drive_service.initialize()

            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="rfp_chat_")

            # Download files
            downloaded_files = await drive_service.download_folder_contents(
                folder_id=folder_id,
                local_path=temp_dir
            )

            # Load into memory
            documents = {}
            for file_path in downloaded_files:
                file_name = os.path.basename(file_path)
                with open(file_path, 'r', encoding='utf-8') as f:
                    documents[file_name] = f.read()

            # Cleanup temp files
            import shutil
            shutil.rmtree(temp_dir)

            return documents

        except Exception as e:
            logger.error(f"Error downloading from Drive: {e}")
            return {}

    async def _load_documents_from_folder(self, folder_path: str) -> Dict[str, str]:
        """Load documents from local folder"""
        documents = {}

        try:
            folder = Path(folder_path)
            if not folder.exists():
                logger.error(f"Folder not found: {folder_path}")
                return documents

            # Load all text documents
            for file_path in folder.glob("*"):
                if file_path.suffix in ['.txt', '.md', '.pdf', '.docx', '.json']:
                    try:
                        if file_path.suffix == '.json':
                            with open(file_path, 'r') as f:
                                content = json.dumps(json.load(f), indent=2)
                        else:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()

                        documents[file_path.name] = content
                    except Exception as e:
                        logger.error(f"Error loading {file_path}: {e}")

            return documents

        except Exception as e:
            logger.error(f"Error loading documents from folder: {e}")
            return documents

    async def _load_human_review_guide(self, folder_path: str) -> Optional[str]:
        """Load Human Review Guide from folder"""
        try:
            folder = Path(folder_path)

            # Look for Human Review Guide
            for file_path in folder.glob("*Human_Review_Guide*"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()

            return None

        except Exception as e:
            logger.error(f"Error loading human review guide: {e}")
            return None

    async def _save_regenerated_documents(self,
                                         session_id: str,
                                         documents: Dict[str, str],
                                         review_guide: str) -> str:
        """Save regenerated documents to new folder"""
        try:
            # Create output folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_folder = Path(f"output/regenerated_{session_id[:8]}_{timestamp}")
            output_folder.mkdir(parents=True, exist_ok=True)

            # Save documents
            for doc_name, content in documents.items():
                file_path = output_folder / doc_name
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

            # Save new review guide
            review_path = output_folder / "Human_Review_Guide_v2.md"
            with open(review_path, 'w', encoding='utf-8') as f:
                f.write(review_guide)

            logger.info(f"Saved regenerated documents to {output_folder}")
            return str(output_folder)

        except Exception as e:
            logger.error(f"Error saving regenerated documents: {e}")
            return ""