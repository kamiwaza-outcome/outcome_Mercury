#!/usr/bin/env python3
"""
Test Google Cloud authentication using service account credentials
"""
import os
import json
from google.oauth2 import service_account
from google.auth.transport.requests import Request

def test_service_account():
    """Test that service account credentials are valid and can authenticate"""
    
    # Path to service account file
    service_account_path = "/Users/finnegannorris/code/Mercury_RFP/credentials/service-account.json"
    
    try:
        # Load credentials
        with open(service_account_path, 'r') as f:
            service_account_info = json.load(f)
        
        # Create credentials object
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        # Test authentication by refreshing token
        request = Request()
        credentials.refresh(request)
        
        print("✓ Authentication successful!")
        print(f"  Project ID: {service_account_info['project_id']}")
        print(f"  Service Account: {service_account_info['client_email']}")
        print(f"  Token acquired: {bool(credentials.token)}")
        print(f"  Token expiry: {credentials.expiry}")
        
        # You can also set environment variable for other tools
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path
        print(f"\n✓ GOOGLE_APPLICATION_CREDENTIALS set to: {service_account_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Authentication failed: {e}")
        return False

if __name__ == "__main__":
    success = test_service_account()
    exit(0 if success else 1)