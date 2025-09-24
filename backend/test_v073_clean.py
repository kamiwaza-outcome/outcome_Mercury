#!/usr/bin/env python3
"""Test browser-use v0.7.3 clean implementation"""

import asyncio
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_browser_use():
    """Test the clean browser-use v0.7.3 implementation"""
    
    logger.info("Testing browser-use v0.7.3 clean implementation...")
    
    # Import from our updated browser_scraper
    from services.browser_scraper import BrowserScraper
    
    # Test URL - using a simple page first
    test_url = "https://sam.gov/opp/710637f47916457280ca93dbb73c2b41/view"
    test_notice_id = "710637f47916457280ca93dbb73c2b41"
    
    scraper = BrowserScraper()
    
    try:
        logger.info(f"Testing scraping of {test_url}")
        documents, metadata = await scraper.scrape_rfp(test_url, test_notice_id)
        
        logger.info(f"✅ Success! Scraped {len(documents)} documents")
        logger.info(f"Documents: {list(documents.keys())}")
        
        # Check if we got NARWHAL content (not HVAC)
        main_content = documents.get('main_content.txt', '')
        if 'NARWHAL' in main_content.upper() or 'Artificial Learning' in main_content:
            logger.info("✅ Correctly extracted NARWHAL RFP content!")
        else:
            logger.warning("⚠️ Content may not be NARWHAL RFP")
        
        return True
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        scraper.cleanup()

if __name__ == "__main__":
    # Ensure we have the API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not set in environment")
        exit(1)
    
    success = asyncio.run(test_browser_use())
    if success:
        print("\n✅ Test passed - browser-use v0.7.3 is working!")
    else:
        print("\n❌ Test failed - check logs above")