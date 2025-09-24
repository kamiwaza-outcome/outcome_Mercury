#!/usr/bin/env python3
"""Test if service account can access the specified Google Drive folder"""

import os
import sys
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv

# Load environment variables
load_dotenv('backend/.env')

def extract_folder_id(url):
    """Extract folder ID from Google Drive URL"""
    # Handle different URL formats
    if '/folders/' in url:
        folder_id = url.split('/folders/')[-1].split('?')[0]
    elif 'id=' in url:
        folder_id = url.split('id=')[-1].split('&')[0]
    else:
        folder_id = url
    return folder_id

def test_folder_access():
    """Test access to the specified Google Drive folder"""
    
    # The folder URL provided by user
    folder_url = "https://drive.google.com/drive/folders/1NHwlIk-SV1FZl3gb05WPaVLMFeIYULIF?dmr=1&ec=wgc-drive-hero-goto"
    folder_id = extract_folder_id(folder_url)
    
    print(f"Testing access to folder ID: {folder_id}")
    print("-" * 50)
    
    try:
        # Get service account file path
        service_account_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
        
        if not service_account_file:
            print("ERROR: GOOGLE_SERVICE_ACCOUNT_FILE not set in environment")
            return
        
        if not os.path.exists(service_account_file):
            print(f"ERROR: Service account file not found: {service_account_file}")
            return
        
        print(f"Using service account file: {service_account_file}")
        
        # Load credentials
        credentials = service_account.Credentials.from_service_account_file(
            service_account_file,
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        
        # Get service account email
        with open(service_account_file, 'r') as f:
            import json
            sa_info = json.load(f)
            sa_email = sa_info.get('client_email', 'Unknown')
            print(f"Service account email: {sa_email}")
        
        print("-" * 50)
        
        # Build the Drive service
        service = build('drive', 'v3', credentials=credentials)
        
        # Try to get folder metadata
        print("Attempting to access folder metadata...")
        print("(Trying with shared drive support...)")
        try:
            # Try with supportsAllDrives=True for shared drives
            folder_metadata = service.files().get(
                fileId=folder_id,
                fields='id,name,mimeType,owners,permissions,shared,capabilities,driveId',
                supportsAllDrives=True
            ).execute()
            
            print(f"✓ SUCCESS: Can access folder!")
            print(f"  Folder name: {folder_metadata.get('name', 'Unknown')}")
            print(f"  Folder ID: {folder_metadata.get('id')}")
            print(f"  Is shared: {folder_metadata.get('shared', False)}")
            
            # Check capabilities
            capabilities = folder_metadata.get('capabilities', {})
            print(f"  Can list children: {capabilities.get('canListChildren', False)}")
            print(f"  Can download: {capabilities.get('canDownload', False)}")
            
        except HttpError as e:
            if e.resp.status == 404:
                print(f"✗ ERROR: Folder not found or no access")
                print(f"  The service account ({sa_email}) does not have access to this folder.")
                print(f"  To grant access:")
                print(f"  1. Open the folder in Google Drive")
                print(f"  2. Click 'Share' button")
                print(f"  3. Add this email: {sa_email}")
                print(f"  4. Set permission to 'Viewer' or higher")
            elif e.resp.status == 403:
                print(f"✗ ERROR: Permission denied")
                print(f"  The service account does not have permission to access this folder.")
                print(f"  Please share the folder with: {sa_email}")
            else:
                print(f"✗ ERROR: {e}")
            return
        
        # Try to list folder contents
        print("\n" + "-" * 50)
        print("Listing folder contents...")
        
        try:
            # List files in the folder (with shared drive support)
            results = service.files().list(
                q=f"'{folder_id}' in parents and trashed=false",
                pageSize=100,
                fields="files(id,name,mimeType,size,modifiedTime,createdTime)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()
            
            files = results.get('files', [])
            
            if not files:
                print("  The folder is empty or contains no accessible files.")
            else:
                print(f"  Found {len(files)} items in the folder:\n")
                
                for file in files:
                    file_type = "Folder" if file['mimeType'] == 'application/vnd.google-apps.folder' else "File"
                    size = file.get('size', 'N/A')
                    if size != 'N/A':
                        size = f"{int(size) / (1024*1024):.2f} MB"
                    
                    print(f"  • {file['name']}")
                    print(f"    Type: {file_type}")
                    print(f"    ID: {file['id']}")
                    print(f"    MIME Type: {file['mimeType']}")
                    if file_type == "File":
                        print(f"    Size: {size}")
                    print(f"    Modified: {file.get('modifiedTime', 'Unknown')}")
                    print()
                    
        except HttpError as e:
            print(f"  ERROR listing contents: {e}")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_folder_access()