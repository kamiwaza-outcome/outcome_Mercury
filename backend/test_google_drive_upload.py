#!/usr/bin/env python3
"""Test Google Drive upload with shared drive support"""

import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Load environment variables
load_dotenv()

from services.google_drive import GoogleDriveService

async def test_upload():
    """Test uploading files to Google Drive shared folder"""
    print("\n=== Testing Google Drive Upload with Shared Drive Support ===\n")

    try:
        # Initialize service
        drive_service = GoogleDriveService()
        print(f"✓ Google Drive service initialized")
        print(f"Output folder ID: {drive_service.output_folder_id}")

        # Create test RFP folder
        notice_id = "TEST_SHARED_DRIVE_002"
        title = "Test Shared Drive Upload"

        print(f"\nCreating folder: {notice_id}_{title}")
        folder_id = await drive_service.create_rfp_folder(notice_id, title)
        print(f"✓ Created folder with ID: {folder_id}")

        # Create test documents
        test_documents = {
            "test_text.txt": "This is a test text file uploaded to shared drive.",
            "test_markdown.md": "# Test Markdown\n\nThis is a test markdown document with **bold** text.",
            "test_data.json": '{"test": true, "message": "Shared drive upload test"}'
        }

        print(f"\nUploading {len(test_documents)} test documents...")
        uploaded_files = await drive_service.upload_documents(folder_id, test_documents)

        if uploaded_files:
            print(f"✓ Successfully uploaded {len(uploaded_files)} files:")
            for file_info in uploaded_files:
                print(f"  - {file_info['name']} (ID: {file_info['id']})")
        else:
            print("✗ No files were uploaded")

        # Get folder URL
        if hasattr(drive_service, 'current_folder_structure'):
            folder_url = drive_service.current_folder_structure.get('url', 'N/A')
            print(f"\n✓ Folder URL: {folder_url}")

        print("\n=== Test Complete ===\n")
        return True

    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_upload())
    sys.exit(0 if success else 1)