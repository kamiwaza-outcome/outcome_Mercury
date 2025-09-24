#!/usr/bin/env python3
"""Test script to verify backend services are working"""

import asyncio
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from dotenv import load_dotenv
load_dotenv()

async def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from services.google_sheets import GoogleSheetsService
        print("✓ Google Sheets service imported")
    except Exception as e:
        print(f"✗ Google Sheets import failed: {e}")
        
    try:
        from services.browser_scraper import BrowserScraper
        print("✓ Browser scraper imported")
    except Exception as e:
        print(f"✗ Browser scraper import failed: {e}")
        
    try:
        from services.sam_api import SamApiClient
        print("✓ SAM API client imported")
    except Exception as e:
        print(f"✗ SAM API import failed: {e}")
        
    try:
        from services.milvus_rag import MilvusRAG
        print("✓ Milvus RAG imported")
    except Exception as e:
        print(f"✗ Milvus RAG import failed: {e}")
        
    try:
        from services.document_generator import DocumentGenerator
        print("✓ Document generator imported")
    except Exception as e:
        print(f"✗ Document generator import failed: {e}")
        
    try:
        from services.orchestration_agent import OrchestrationAgent
        print("✓ Orchestration agent imported")
    except Exception as e:
        print(f"✗ Orchestration agent import failed: {e}")
        
    try:
        from services.google_drive import GoogleDriveService
        print("✓ Google Drive service imported")
    except Exception as e:
        print(f"✗ Google Drive import failed: {e}")

async def test_milvus():
    """Test Milvus initialization"""
    print("\nTesting Milvus RAG...")
    try:
        from services.milvus_rag import MilvusRAG
        rag = MilvusRAG()
        await rag.initialize()
        print("✓ Milvus initialized successfully")
        
        # Test search
        results = await rag.search_similar("AI capabilities", limit=3)
        print(f"✓ Found {len(results)} results for test query")
        
    except Exception as e:
        print(f"✗ Milvus test failed: {e}")

async def test_openai():
    """Test OpenAI API"""
    print("\nTesting OpenAI API...")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Test with a simple completion
        response = client.chat.completions.create(
            model="gpt-4",  # Use GPT-4 as fallback since GPT-5 might not be available
            messages=[{"role": "user", "content": "Say 'test successful'"}],
            max_tokens=10
        )
        
        result = response.choices[0].message.content
        if "test" in result.lower() or "successful" in result.lower():
            print("✓ OpenAI API working")
        else:
            print(f"? OpenAI API responded: {result}")
            
    except Exception as e:
        print(f"✗ OpenAI API test failed: {e}")

async def test_google_sheets():
    """Test Google Sheets connection"""
    print("\nTesting Google Sheets...")
    try:
        from services.google_sheets import GoogleSheetsService
        sheets = GoogleSheetsService()
        
        # Try to get checked RFPs
        rfps = await sheets.get_checked_rfps()
        print(f"✓ Google Sheets connected, found {len(rfps)} checked RFPs")
        
        if rfps:
            print(f"  First RFP: {rfps[0].get('title', 'No title')}")
            
    except FileNotFoundError as e:
        print(f"⚠ Service account file issue: {e}")
        print("  Please ensure you have a valid service account JSON file")
    except Exception as e:
        print(f"✗ Google Sheets test failed: {e}")

async def main():
    print("=" * 50)
    print("Mercury RFP Backend Test Suite")
    print("=" * 50)
    
    await test_imports()
    await test_openai()
    await test_milvus()
    await test_google_sheets()
    
    print("\n" + "=" * 50)
    print("Test complete!")
    print("\nNext steps:")
    print("1. Add valid Google service account credentials")
    print("2. Ensure Google Sheet ID is correct in .env")
    print("3. Run backend: cd backend && source venv/bin/activate && python main.py")
    print("4. Run frontend: cd frontend && npm install && npm run dev")

if __name__ == "__main__":
    asyncio.run(main())