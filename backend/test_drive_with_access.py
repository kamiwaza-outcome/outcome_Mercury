#!/usr/bin/env python3
"""
Test Google Drive extraction with shared drive access
The service account already has access to all folders in the shared drive
"""

import asyncio
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from services.google_drive_extractor import GoogleDriveRFPExtractor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def test_drive_extraction_with_access():
    """Test extraction from row 250 Google Drive folder with shared drive access"""

    # Row 250 folder URL from Google Sheets column M
    test_folder_url = "https://drive.google.com/drive/folders/1HHGv1rVIlisRWiRE8-FcO0We7DmJNtrn?usp=drive_link"
    test_notice_id = "row_250_test"

    logger.info("=" * 80)
    logger.info("Testing Google Drive RFP Extraction with Shared Drive Access")
    logger.info("=" * 80)
    logger.info(f"Folder URL: {test_folder_url}")
    logger.info("Service account has access via shared drive membership")
    logger.info("")

    try:
        # Initialize extractor
        extractor = GoogleDriveRFPExtractor()

        # Perform full extraction
        logger.info("🔄 Starting extraction from Google Drive folder...")
        documents, metadata = await extractor.extract_from_folder(test_folder_url, test_notice_id)

        # Display results
        logger.info("\n" + "=" * 80)
        logger.info("📊 EXTRACTION RESULTS")
        logger.info("=" * 80)

        logger.info(f"\n📄 Documents extracted: {len(documents)}")
        assessment_excluded = False

        for doc_name, content in documents.items():
            preview = content[:100].replace('\n', ' ') if content else "[empty]"
            logger.info(f"  - {doc_name}: {len(content)} chars")
            if len(content) > 0:
                logger.info(f"    Preview: {preview}...")

        # Check if assessment was excluded
        logger.info(f"\n🔍 Checking assessment.txt exclusion:")
        assessment_found = any('assessment' in key.lower() for key in documents.keys())
        if assessment_found:
            logger.error("  ❌ WARNING: assessment.txt was NOT excluded properly!")
        else:
            logger.info("  ✅ assessment.txt successfully excluded from LLM context")

        logger.info(f"\n📋 Metadata extracted:")
        for key, value in metadata.items():
            if key != 'extraction_stats':
                logger.info(f"  - {key}: {value}")

        logger.info(f"\n📈 Extraction Statistics:")
        stats = metadata.get('extraction_stats', {})
        for key, value in stats.items():
            logger.info(f"  - {key}: {value}")

        # Quality check
        logger.info("\n" + "=" * 80)
        logger.info("✅ QUALITY VERIFICATION")
        logger.info("=" * 80)

        total_content = sum(len(str(content)) for content in documents.values())

        # Check content volume
        if total_content < 1000:
            logger.warning("⚠️  Low content volume - may need to check file access")
        elif total_content < 10000:
            logger.info("📊 Good content volume - basic RFP data extracted")
        else:
            logger.info("✅ Excellent content volume - comprehensive RFP data extracted")

        # Check for key document types
        has_info = any('info' in key.lower() for key in documents.keys())
        has_main = any('main' in key.lower() or 'rfp' in key.lower() for key in documents.keys())
        has_tech = any('tech' in key.lower() or 'spec' in key.lower() for key in documents.keys())
        has_attachments = any('attachment' in key.lower() for key in documents.keys())

        logger.info(f"\n📁 Document completeness:")
        logger.info(f"  - Has info document: {'✅' if has_info else '❌'}")
        logger.info(f"  - Has main RFP: {'✅' if has_main else '❌'}")
        logger.info(f"  - Has technical specs: {'✅' if has_tech else '❌'}")
        logger.info(f"  - Has attachments: {'✅' if has_attachments else '❌'}")
        logger.info(f"  - Assessment excluded: {'✅' if not assessment_found else '❌'}")

        # Performance metrics
        excluded_count = stats.get('excluded_files', 0)
        processed_count = stats.get('processed_files', 0)
        total_files = stats.get('total_files', 0)

        logger.info(f"\n📊 Processing summary:")
        logger.info(f"  - Total files in folder: {total_files}")
        logger.info(f"  - Files processed: {processed_count}")
        logger.info(f"  - Files excluded (assessment): {excluded_count}")
        if total_files > 0:
            logger.info(f"  - Extraction efficiency: {processed_count}/{total_files} = {(processed_count/total_files*100):.1f}%")
        else:
            logger.info(f"  - No files found in folder to process")

        logger.info("\n" + "=" * 80)
        if total_content > 5000 and not assessment_found:
            logger.info("🎉 SUCCESS - Google Drive extraction working perfectly!")
            logger.info("✅ Assessment.txt excluded as requested")
            logger.info("✅ Ready for production use")
        else:
            logger.info("⚠️  Test completed with issues - review results above")
        logger.info("=" * 80)

        # Clean up
        extractor.cleanup()

        return True

    except Exception as e:
        logger.error(f"\n❌ Test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = asyncio.run(test_drive_extraction_with_access())
    sys.exit(0 if success else 1)