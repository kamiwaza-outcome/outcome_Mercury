import os
import logging
import re
from typing import Dict, Any, Optional, List
from pathlib import Path
import tempfile
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload
import io

logger = logging.getLogger(__name__)

class GoogleDriveService:
    def __init__(self):
        self.output_folder_id = os.getenv("OUTPUT_FOLDER_ID")
        self.template_folder_id = os.getenv("TEMPLATE_FOLDER_ID")
        self.service = None

    async def initialize(self):
        """Async initialization"""
        if not self.service:
            self.service = self._initialize_service()
        return self
    
    def _initialize_service(self):
        """Initialize Google Drive API service"""
        try:
            service_account_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
            
            if not service_account_file or not os.path.exists(service_account_file):
                raise FileNotFoundError(f"Service account file not found: {service_account_file}")
            
            credentials = service_account.Credentials.from_service_account_file(
                service_account_file,
                scopes=['https://www.googleapis.com/auth/drive']
            )
            
            service = build('drive', 'v3', credentials=credentials)
            logger.info("Google Drive service initialized successfully")
            return service
        except Exception as e:
            logger.error(f"Failed to initialize Google Drive service: {e}")
            raise
    
    async def create_rfp_folder(self, notice_id: str, title: str) -> str:
        """Create a folder for the RFP in Google Drive"""
        try:
            # Create folder name with timestamp
            folder_name = f"{notice_id}_{title[:50]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Create main RFP folder
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [self.output_folder_id]
            }

            main_folder = self.service.files().create(
                body=file_metadata,
                fields='id, webViewLink',
                supportsAllDrives=True
            ).execute()

            main_folder_id = main_folder.get('id')
            logger.info(f"Created main folder: {folder_name} with ID: {main_folder_id}")

            # Store folder info for later use (no subfolders anymore)
            self.current_folder_structure = {
                'main': main_folder_id,
                'url': main_folder.get('webViewLink')
            }

            return main_folder_id
            
        except HttpError as e:
            logger.error(f"Google Drive API error creating folder: {e}")
            raise
        except Exception as e:
            logger.error(f"Error creating RFP folder: {e}")
            raise
    
    async def upload_documents(self, folder_id: str, documents: Dict[str, Any]) -> List[Dict[str, str]]:
        """Upload documents to Google Drive folder"""
        try:
            uploaded_files = []
            
            for doc_name, content in documents.items():
                try:
                    # All files go directly in the main folder now
                    target_folder_id = folder_id

                    # Determine file type and MIME type
                    mime_type = self._get_mime_type(doc_name)

                    # Handle different content types
                    if isinstance(content, str):
                        # Text content - determine if it should be converted to Google Doc
                        ext = doc_name.lower().split('.')[-1] if '.' in doc_name else 'txt'

                        # Convert text-based files AND PDFs to Google Docs for editability
                        # PDFs will be created as formatted Google Docs instead
                        if ext in ['txt', 'md', 'markdown', 'rst', 'log', 'pdf', 'docx', 'doc']:
                            file_id = await self._upload_text_as_google_doc(
                                content, doc_name, target_folder_id
                            )
                        else:
                            # For other text content (JSON, XML, etc.), keep as regular files
                            file_id = await self._upload_text_content(
                                content, doc_name, target_folder_id, mime_type
                            )
                    elif isinstance(content, bytes):
                        # Binary content
                        file_id = await self._upload_binary_content(
                            content, doc_name, target_folder_id, mime_type
                        )
                    elif isinstance(content, Path):
                        # File path
                        file_id = await self._upload_file(
                            content, doc_name, target_folder_id, mime_type
                        )
                    else:
                        logger.warning(f"Unsupported content type for {doc_name}")
                        continue
                    
                    if file_id:
                        uploaded_files.append({
                            'name': doc_name,
                            'id': file_id,
                            'folder': target_folder_id
                        })
                        logger.info(f"Uploaded {doc_name} to Google Drive")
                    
                except Exception as e:
                    logger.error(f"Error uploading {doc_name}: {e}")
            
            return uploaded_files
            
        except Exception as e:
            logger.error(f"Error uploading documents: {e}")
            raise
    
    def _get_subfolder_for_document(self, doc_name: str, parent_folder_id: str) -> str:
        """Determine which subfolder a document should go in"""
        doc_name_lower = doc_name.lower()
        
        if hasattr(self, 'current_folder_structure'):
            subfolders = self.current_folder_structure.get('subfolders', {})
            
            if 'northstar' in doc_name_lower or 'analysis' in doc_name_lower:
                return subfolders.get('Review_Documents', parent_folder_id)
            elif 'review' in doc_name_lower or 'guide' in doc_name_lower:
                return subfolders.get('Review_Documents', parent_folder_id)
            elif 'source' in doc_name_lower or 'original' in doc_name_lower:
                return subfolders.get('Source_Documents', parent_folder_id)
            elif any(ext in doc_name_lower for ext in ['.docx', '.pdf', '.xlsx']):
                return subfolders.get('Generated_Responses', parent_folder_id)
            else:
                return subfolders.get('Submission_Package', parent_folder_id)
        
        return parent_folder_id
    
    def _get_mime_type(self, filename: str) -> str:
        """Get MIME type based on file extension"""
        ext = filename.lower().split('.')[-1] if '.' in filename else 'txt'
        
        mime_types = {
            'txt': 'text/plain',
            'md': 'text/markdown',
            'pdf': 'application/pdf',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'doc': 'application/msword',
            'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'xls': 'application/vnd.ms-excel',
            'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'html': 'text/html',
            'json': 'application/json',
            'xml': 'application/xml',
            'csv': 'text/csv'
        }
        
        return mime_types.get(ext, 'application/octet-stream')
    
    async def _upload_text_as_google_doc(self, content: str, filename: str, folder_id: str) -> str:
        """Upload text content as a Google Doc with proper formatting"""
        try:
            # Remove file extension if present and add appropriate title
            doc_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
            ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'txt'

            # Create file metadata for Google Docs
            file_metadata = {
                'name': doc_name,
                'parents': [folder_id],
                'mimeType': 'application/vnd.google-apps.document'  # Google Docs MIME type
            }

            # Convert content to HTML with better formatting based on file type
            if ext == 'md' or ext == 'markdown':
                # Convert Markdown to HTML for better formatting
                html_content = self._markdown_to_html(content, doc_name)
            elif ext in ['pdf', 'docx', 'doc']:
                # For PDF-like documents, use professional formatting
                html_content = self._format_document_html(content, doc_name)
            else:
                # Default formatting for plain text
                html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{doc_name}</title>
</head>
<body>
    <pre style="white-space: pre-wrap; font-family: Arial, sans-serif; line-height: 1.6;">
{content}
    </pre>
</body>
</html>"""

            # Upload as Google Doc using text/html content
            media = MediaIoBaseUpload(
                io.BytesIO(html_content.encode('utf-8')),
                mimetype='text/html',
                resumable=True
            )

            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, webViewLink',
                supportsAllDrives=True
            ).execute()

            logger.info(f"Created Google Doc: {doc_name} (ID: {file.get('id')})")
            return file.get('id')

        except Exception as e:
            logger.error(f"Error creating Google Doc: {e}")
            raise

    async def _upload_text_content(self, content: str, filename: str, folder_id: str, mime_type: str) -> str:
        """Upload text content as a regular file (for JSON, XML, etc.)"""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{filename}', delete=False) as tmp_file:
                tmp_file.write(content)
                tmp_path = tmp_file.name

            try:
                # Upload the temporary file
                file_metadata = {
                    'name': filename,
                    'parents': [folder_id]
                }

                media = MediaFileUpload(
                    tmp_path,
                    mimetype=mime_type,
                    resumable=True
                )

                file = self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id',
                    supportsAllDrives=True
                ).execute()

                return file.get('id')

            finally:
                # Clean up temporary file
                os.unlink(tmp_path)

        except Exception as e:
            logger.error(f"Error uploading text content: {e}")
            raise

    async def _upload_binary_content(self, content: bytes, filename: str, folder_id: str, mime_type: str) -> str:
        """Upload binary content to Google Drive"""
        try:
            file_metadata = {
                'name': filename,
                'parents': [folder_id]
            }
            
            media = MediaIoBaseUpload(
                io.BytesIO(content),
                mimetype=mime_type,
                resumable=True
            )

            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id',
                supportsAllDrives=True
            ).execute()
            
            return file.get('id')
            
        except Exception as e:
            logger.error(f"Error uploading binary content: {e}")
            raise
    
    async def _upload_file(self, file_path: Path, filename: str, folder_id: str, mime_type: str) -> str:
        """Upload a file from disk to Google Drive"""
        try:
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                return None
            
            file_metadata = {
                'name': filename,
                'parents': [folder_id]
            }
            
            media = MediaFileUpload(
                str(file_path),
                mimetype=mime_type,
                resumable=True
            )

            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id',
                supportsAllDrives=True
            ).execute()
            
            return file.get('id')
            
        except Exception as e:
            logger.error(f"Error uploading file {file_path}: {e}")
            raise
    
    async def get_templates(self) -> List[Dict[str, str]]:
        """Get list of templates from template folder"""
        try:
            if not self.template_folder_id:
                return []
            
            results = self.service.files().list(
                q=f"'{self.template_folder_id}' in parents and trashed = false",
                fields="files(id, name, mimeType, modifiedTime)",
                supportsAllDrives=True
            ).execute()
            
            templates = results.get('files', [])
            logger.info(f"Found {len(templates)} templates")
            return templates
            
        except Exception as e:
            logger.error(f"Error getting templates: {e}")
            return []

    def _markdown_to_html(self, markdown_content: str, title: str) -> str:
        """Convert Markdown content to HTML for Google Docs"""
        import re

        # Basic Markdown to HTML conversion
        html = markdown_content

        # Headers
        html = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

        # Bold and Italic
        html = re.sub(r'\*\*\*(.*?)\*\*\*', r'<b><i>\1</i></b>', html)
        html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', html)
        html = re.sub(r'\*(.*?)\*', r'<i>\1</i>', html)

        # Lists
        html = re.sub(r'^\* (.*)$', r'<li>\1</li>', html, flags=re.MULTILINE)
        html = re.sub(r'^\- (.*)$', r'<li>\1</li>', html, flags=re.MULTILINE)
        html = re.sub(r'^\d+\. (.*)$', r'<li>\1</li>', html, flags=re.MULTILINE)

        # Wrap consecutive list items
        html = re.sub(r'(<li>.*?</li>\n)+', lambda m: '<ul>\n' + m.group(0) + '</ul>\n', html)

        # Paragraphs
        html = re.sub(r'\n\n', '</p><p>', html)

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 20px; }}
        h3 {{ color: #7f8c8d; margin-top: 15px; }}
        ul {{ margin: 10px 0; padding-left: 30px; }}
        li {{ margin: 5px 0; }}
        p {{ margin: 10px 0; }}
    </style>
</head>
<body>
    <p>{html}</p>
</body>
</html>"""

    def _format_document_html(self, content: str, title: str) -> str:
        """Format document content as professional HTML for Google Docs"""
        # Split content into sections if it has clear headers
        lines = content.split('\n')
        formatted_html = []

        for line in lines:
            line = line.strip()
            if not line:
                formatted_html.append('<br>')
            elif line.isupper() and len(line) < 100:
                # Likely a header
                formatted_html.append(f'<h2 style="color: #2c3e50; margin-top: 20px;">{line}</h2>')
            elif line.startswith('•') or line.startswith('-') or line.startswith('*'):
                # Bullet point
                formatted_html.append(f'<li>{line[1:].strip()}</li>')
            else:
                # Regular paragraph
                formatted_html.append(f'<p>{line}</p>')

        # Wrap consecutive list items
        html_content = '\n'.join(formatted_html)
        html_content = re.sub(r'(<li>.*?</li>\n)+', lambda m: '<ul style="margin: 10px 0; padding-left: 30px;">\n' + m.group(0) + '</ul>\n', html_content)

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Times New Roman', serif;
            font-size: 12pt;
            line-height: 1.8;
            color: #000;
            margin: 40px;
        }}
        h1 {{
            font-size: 16pt;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
        }}
        h2 {{
            font-size: 14pt;
            font-weight: bold;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        p {{
            text-align: justify;
            margin: 12px 0;
        }}
        ul {{
            margin: 10px 0;
            padding-left: 40px;
        }}
        li {{
            margin: 8px 0;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    {html_content}
</body>
</html>"""
    
    async def download_template(self, template_id: str) -> bytes:
        """Download a template file"""
        try:
            request = self.service.files().get_media(fileId=template_id, supportsAllDrives=True)
            content = request.execute()
            return content
            
        except Exception as e:
            logger.error(f"Error downloading template {template_id}: {e}")
            raise
    
    async def share_folder(self, folder_id: str, email: str, role: str = 'writer') -> bool:
        """Share a folder with a user"""
        try:
            permission = {
                'type': 'user',
                'role': role,
                'emailAddress': email
            }
            
            self.service.permissions().create(
                fileId=folder_id,
                body=permission,
                sendNotificationEmail=True,
                supportsAllDrives=True
            ).execute()
            
            logger.info(f"Shared folder {folder_id} with {email}")
            return True
            
        except Exception as e:
            logger.error(f"Error sharing folder: {e}")
            return False
    
    async def create_shareable_link(self, folder_id: str) -> str:
        """Create a shareable link for a folder"""
        try:
            permission = {
                'type': 'anyone',
                'role': 'reader'
            }
            
            self.service.permissions().create(
                fileId=folder_id,
                body=permission,
                supportsAllDrives=True
            ).execute()
            
            file = self.service.files().get(
                fileId=folder_id,
                fields='webViewLink',
                supportsAllDrives=True
            ).execute()
            
            return file.get('webViewLink', '')
            
        except Exception as e:
            logger.error(f"Error creating shareable link: {e}")
            return ""

    async def download_folder_contents(self, folder_id: str, local_path: str) -> List[str]:
        """Download all files from a Google Drive folder to local path"""
        downloaded_files = []

        try:
            # Ensure service is initialized
            if not self.service:
                await self.initialize()

            # Create local directory if it doesn't exist
            os.makedirs(local_path, exist_ok=True)

            # List all files in the folder
            query = f"'{folder_id}' in parents and trashed = false"
            results = self.service.files().list(
                q=query,
                fields="nextPageToken, files(id, name, mimeType)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()

            files = results.get('files', [])
            logger.info(f"Found {len(files)} files in folder {folder_id}")

            for file in files:
                file_id = file['id']
                file_name = file['name']
                mime_type = file.get('mimeType', '')

                logger.info(f"Downloading file: {file_name} (ID: {file_id}, Type: {mime_type})")

                try:
                    # Handle Google Docs/Sheets/Slides export
                    if mime_type.startswith('application/vnd.google-apps'):
                        content = await self._export_google_doc(file_id, mime_type)

                        # Determine file extension based on export type
                        if 'document' in mime_type:
                            file_name = file_name if file_name.endswith('.txt') else f"{file_name}.txt"
                        elif 'spreadsheet' in mime_type:
                            file_name = file_name if file_name.endswith('.csv') else f"{file_name}.csv"
                        elif 'presentation' in mime_type:
                            file_name = file_name if file_name.endswith('.txt') else f"{file_name}.txt"

                        # Save content to file
                        file_path = os.path.join(local_path, file_name)
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                    else:
                        # Download regular files
                        request = self.service.files().get_media(
                            fileId=file_id,
                            supportsAllDrives=True
                        )
                        content = request.execute()

                        # Save to file
                        file_path = os.path.join(local_path, file_name)
                        mode = 'wb' if isinstance(content, bytes) else 'w'
                        with open(file_path, mode) as f:
                            f.write(content)

                    downloaded_files.append(file_path)
                    logger.info(f"Downloaded: {file_name} to {file_path}")

                except Exception as e:
                    logger.error(f"Error downloading file {file_name}: {e}")
                    continue

            # Also check for subfolders and download recursively
            folder_query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed = false"
            folder_results = self.service.files().list(
                q=folder_query,
                fields="files(id, name)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()

            subfolders = folder_results.get('files', [])
            for subfolder in subfolders:
                subfolder_path = os.path.join(local_path, subfolder['name'])
                os.makedirs(subfolder_path, exist_ok=True)
                subfolder_files = await self.download_folder_contents(subfolder['id'], subfolder_path)
                downloaded_files.extend(subfolder_files)

            logger.info(f"Downloaded {len(downloaded_files)} total files from folder and subfolders")
            return downloaded_files

        except Exception as e:
            logger.error(f"Error downloading folder contents: {e}")
            return downloaded_files

    async def _export_google_doc(self, file_id: str, mime_type: str) -> str:
        """Export Google Docs/Sheets/Slides to text format"""
        try:
            # Determine export MIME type
            if 'document' in mime_type:
                export_mime = 'text/plain'
            elif 'spreadsheet' in mime_type:
                export_mime = 'text/csv'
            elif 'presentation' in mime_type:
                export_mime = 'text/plain'
            else:
                export_mime = 'text/plain'

            # Export the file
            request = self.service.files().export_media(
                fileId=file_id,
                mimeType=export_mime
            )
            content = request.execute()

            # Decode if bytes
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='ignore')

            return content

        except Exception as e:
            logger.error(f"Error exporting Google Doc {file_id}: {e}")
            return ""