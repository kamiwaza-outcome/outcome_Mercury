#!/usr/bin/env python3
"""
Test script for attachment download functionality
"""
import asyncio
import logging
import os
from services.browser_scraper import BrowserScraper

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_attachment_download():
    """Test the attachment download functionality"""
    try:
        # Initialize the browser scraper
        scraper = BrowserScraper()
        
        # Test URL with known attachments (NARWHAL RFP)
        test_url = "https://sam.gov/opp/710637f47916457280ca93dbb73c2b41/view"
        notice_id = "710637f47916457280ca93dbb73c2b41"
        
        logger.info("=" * 80)
        logger.info("Testing Attachment Download for NARWHAL RFP")
        logger.info("=" * 80)
        logger.info(f"URL: {test_url}")
        logger.info(f"Notice ID: {notice_id}")
        
        # Run the scraper
        logger.info("\nStarting Browser Use scraping with attachment extraction...")
        documents, metadata = await scraper.scrape_rfp(test_url, notice_id)
        
        # Check results
        logger.info("\n" + "=" * 80)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 80)
        
        # Count attachment documents
        attachment_docs = [doc for doc in documents.keys() if 'attachment' in doc.lower()]
        logger.info(f"Total documents extracted: {len(documents)}")
        logger.info(f"Attachment documents: {len(attachment_docs)}")
        
        # Show document names and sizes
        logger.info("\nDocument Details:")
        for doc_name, content in documents.items():
            size = len(str(content)) if content else 0
            is_attachment = 'attachment' in doc_name.lower()
            marker = "📎" if is_attachment else "📄"
            logger.info(f"  {marker} {doc_name}: {size:,} chars")
        
        # Check for attachment URLs found
        if 'extraction_stats' in metadata and 'attachment_urls' in metadata['extraction_stats']:
            urls = metadata['extraction_stats']['attachment_urls']
            logger.info(f"\nAttachment URLs found: {len(urls)}")
            for url in urls[:5]:  # Show first 5 URLs
                logger.info(f"  - {url}")
        
        # Check if attachments were actually downloaded
        if attachment_docs:
            logger.info("\n✅ SUCCESS: Attachments were downloaded and extracted!")
            logger.info("\nAttachment Contents Preview:")
            for doc_name in attachment_docs[:3]:  # Show first 3 attachments
                content = documents[doc_name]
                preview = str(content)[:200] if content else "No content"
                logger.info(f"\n{doc_name}:")
                logger.info(f"  {preview}...")
        else:
            logger.info("\n⚠️  WARNING: No attachments were downloaded")
            logger.info("Checking if attachment URLs were found in content...")
            
            # Search for attachment patterns in main content
            all_content = ' '.join(str(doc) for doc in documents.values())
            if '/api/opportunities/v2/download/' in all_content or 'attachment' in all_content.lower():
                logger.info("  - Attachment references found in content")
                logger.info("  - May need to check URL extraction logic")
            else:
                logger.info("  - No attachment references found in scraped content")
        
        # Show total content size
        total_size = sum(len(str(doc)) for doc in documents.values())
        logger.info(f"\nTotal content extracted: {total_size:,} characters")
        
        # Cleanup
        scraper.cleanup()
        
        return len(attachment_docs) > 0
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_attachment_download())
    
    if success:
        logger.info("\n" + "=" * 80)
        logger.info("✅ ATTACHMENT DOWNLOAD TEST PASSED!")
        logger.info("=" * 80)
    else:
        logger.info("\n" + "=" * 80)
        logger.info("❌ ATTACHMENT DOWNLOAD TEST FAILED - Attachments not downloaded")
        logger.info("=" * 80)