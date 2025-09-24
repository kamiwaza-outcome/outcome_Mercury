#!/usr/bin/env python3
"""Test SAM.gov API integration"""
import os
import asyncio
import logging
from services.sam_api import SamApiClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_sam_api():
    """Test SAM API functionality"""
    
    # Check if API key is configured
    api_key = os.getenv("SAM_API_KEY")
    
    print("=" * 60)
    print("SAM.gov API Integration Test")
    print("=" * 60)
    
    if not api_key:
        print("\n❌ SAM_API_KEY environment variable not set")
        print("\nTo get a SAM.gov API key:")
        print("1. Create an account at https://sam.gov")
        print("2. Go to Account Details page")
        print("3. Enter password to view/generate API key")
        print("4. Set environment variable: export SAM_API_KEY='your-40-char-key'")
        print("\nWithout API key, system will use Browser Use for all scraping")
        return
    
    print(f"\n✅ SAM_API_KEY configured (length: {len(api_key)} chars)")
    
    if len(api_key) != 40:
        print(f"⚠️  Warning: API key should be 40 chars, got {len(api_key)}")
    
    # Initialize client
    client = SamApiClient()
    
    # Test with NARWHAL RFP
    notice_id = "710637f47916457280ca93dbb73c2b41"
    print(f"\n📋 Testing with notice ID: {notice_id}")
    
    # Test get_opportunity
    print("\n1. Testing get_opportunity()...")
    opportunity = await client.get_opportunity(notice_id)
    
    if opportunity:
        print("✅ Successfully retrieved opportunity")
        print(f"   - Title: {opportunity.get('title', 'N/A')}")
        print(f"   - Sol Number: {opportunity.get('sol_number', 'N/A')}")
        print(f"   - Department: {opportunity.get('department', 'N/A')}")
        print(f"   - Posted Date: {opportunity.get('posted_date', 'N/A')}")
        print(f"   - Response Deadline: {opportunity.get('response_deadline', 'N/A')}")
        print(f"   - UI Link: {opportunity.get('ui_link', 'N/A')}")
        print(f"   - Additional Info Link: {opportunity.get('additional_info_link', 'N/A')}")
        resource_links = opportunity.get('resource_links')
        if resource_links:
            print(f"   - Resource Links: {len(resource_links)} links")
        else:
            print(f"   - Resource Links: None")
    else:
        print("❌ Failed to retrieve opportunity")
        print("   This could mean:")
        print("   - Invalid API key")
        print("   - Notice ID not found")
        print("   - API rate limit exceeded")
        return
    
    # Test download_attachments
    print("\n2. Testing download_attachments()...")
    documents = await client.download_attachments(notice_id)
    
    if documents:
        print(f"✅ Successfully downloaded {len(documents)} documents:")
        for doc_name, content in documents.items():
            content_preview = str(content)[:100] if content else "Empty"
            print(f"   - {doc_name}: {len(str(content))} chars")
    else:
        print("⚠️  No documents downloaded (may not have attachments)")
    
    # Test search_opportunities
    print("\n3. Testing search_opportunities()...")
    results = await client.search_opportunities(
        keywords="software",
        posted_from="01/01/2025",
        posted_to="01/31/2025"
    )
    
    if results:
        print(f"✅ Found {len(results)} opportunities matching 'software'")
        for i, opp in enumerate(results[:3], 1):
            print(f"   {i}. {opp.get('title', 'N/A')[:60]}...")
    else:
        print("⚠️  No opportunities found (try different date range)")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_sam_api())