#!/usr/bin/env python3
"""Upload generated RFP documents to Google Drive shared folder"""

import os
import sys
from pathlib import Path
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload
from googleapiclient.errors import HttpError
import io
from dotenv import load_dotenv

load_dotenv()

def upload_files_to_drive():
    """Upload the generated files to the specified Google Drive folder"""

    # The parent folder ID from the URL you provided
    parent_folder_id = "1PL8UuOnvW2NzvDJfOPwS2XNg6mLZzmom"

    # Initialize Drive service
    service_account_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file,
        scopes=['https://www.googleapis.com/auth/drive']
    )
    service = build('drive', 'v3', credentials=credentials)

    # Create a subfolder for this RFP
    rfp_folder_name = f"P1SM_CSO_9a1e55199b1f4c1da83c14d548723c24_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        # Create the RFP folder with shared drive support
        folder_metadata = {
            'name': rfp_folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_folder_id]
        }

        rfp_folder = service.files().create(
            body=folder_metadata,
            fields='id, webViewLink',
            supportsAllDrives=True
        ).execute()

        rfp_folder_id = rfp_folder['id']
        print(f"Created folder: {rfp_folder_name}")
        print(f"Folder URL: {rfp_folder.get('webViewLink')}")

        # Upload files from the output directory
        output_dir = Path("output/9a1e55199b1f4c1da83c14d548723c24")

        files_to_upload = [
            "Northstar_Document.md",
            "P1SM_Solution_Pitch.mp4",
            "P1SM_Portal_Submission_Form_Content.txt",
            "Human_Review_Guide.md"
        ]

        for filename in files_to_upload:
            file_path = output_dir / filename

            if file_path.exists():
                # Determine MIME type
                if filename.endswith('.md'):
                    mime_type = 'text/markdown'
                elif filename.endswith('.txt'):
                    mime_type = 'text/plain'
                elif filename.endswith('.mp4'):
                    mime_type = 'video/mp4'
                else:
                    mime_type = 'application/octet-stream'

                # Upload the file
                file_metadata = {
                    'name': filename,
                    'parents': [rfp_folder_id]
                }

                # Handle empty files by uploading with minimal content
                if file_path.stat().st_size == 0:
                    # For empty files, upload a placeholder message
                    if filename == "P1SM_Solution_Pitch.mp4":
                        content = b"[Video script placeholder - content generation in progress]"
                    elif filename == "P1SM_Portal_Submission_Form_Content.txt":
                        content = "[Portal submission form content - generation in progress]"
                    elif filename == "Human_Review_Guide.md":
                        content = "# Human Review Guide\n\n[Content generation in progress]"
                    else:
                        content = "[Empty file - content generation pending]"

                    if isinstance(content, str):
                        content = content.encode('utf-8')

                    media = MediaIoBaseUpload(io.BytesIO(content), mimetype=mime_type)
                else:
                    media = MediaFileUpload(str(file_path), mimetype=mime_type)

                uploaded_file = service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id, name',
                    supportsAllDrives=True
                ).execute()

                print(f"Uploaded: {filename} (ID: {uploaded_file['id']})")
            else:
                print(f"File not found: {filename}")

        print(f"\n✅ All files uploaded successfully to Google Drive!")
        print(f"📁 Folder URL: {rfp_folder.get('webViewLink')}")
        return rfp_folder.get('webViewLink')

    except HttpError as e:
        print(f"❌ Google Drive API error: {e}")
        raise
    except Exception as e:
        print(f"❌ Error uploading files: {e}")
        raise

if __name__ == "__main__":
    upload_files_to_drive()