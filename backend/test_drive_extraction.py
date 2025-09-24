#!/usr/bin/env python3
"""
Test script for Google Drive RFP folder extraction
Tests with row 250 folder from Google Sheets
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

async def test_drive_extraction():
    """Test extraction from row 250 Google Drive folder"""

    # Row 250 folder URL from Google Sheets column M
    test_folder_url = "https://drive.google.com/drive/folders/1HHGv1rVIlisRWiRE8-FcO0We7DmJNtrn?usp=drive_link"
    test_notice_id = "row_250_test"

    logger.info("=" * 80)
    logger.info("Testing Google Drive RFP Extraction")
    logger.info("=" * 80)
    logger.info(f"Folder URL: {test_folder_url}")
    logger.info("")

    try:
        # Initialize extractor
        extractor = GoogleDriveRFPExtractor()

        # Extract folder ID
        folder_id = extractor.extract_folder_id(test_folder_url)
        logger.info(f"Extracted folder ID: {folder_id}")

        # List folder contents first
        logger.info("\n📁 Listing folder contents...")
        files = await extractor._list_folder_contents(folder_id)

        if not files:
            logger.error("❌ No files found in folder. Check service account permissions.")
            logger.info("\nTo fix this:")
            logger.info("1. Share the folder with your service account email")
            logger.info("2. The service account email can be found in your JSON key file")
            return

        logger.info(f"✅ Found {len(files)} files:")
        for file in files:
            size_mb = int(file.get('size', 0)) / (1024 * 1024) if file.get('size') else 0
            logger.info(f"  - {file['name']} ({file['mimeType']}) - {size_mb:.2f} MB")

        # Check for assessment.txt
        assessment_files = [f for f in files if 'assessment' in f['name'].lower()]
        if assessment_files:
            logger.info(f"\n⚠️  Found {len(assessment_files)} assessment file(s) that will be excluded:")
            for f in assessment_files:
                logger.info(f"  - {f['name']} (excluded from LLM context)")

        # Perform full extraction
        logger.info("\n🔄 Extracting documents and metadata...")
        documents, metadata = await extractor.extract_from_folder(test_folder_url, test_notice_id)

        # Display results
        logger.info("\n" + "=" * 80)
        logger.info("📊 EXTRACTION RESULTS")
        logger.info("=" * 80)

        logger.info(f"\n📄 Documents extracted: {len(documents)}")
        for doc_name, content in documents.items():
            preview = content[:100].replace('\n', ' ') if content else "[empty]"
            logger.info(f"  - {doc_name}: {len(content)} chars")
            logger.info(f"    Preview: {preview}...")

        logger.info(f"\n📋 Metadata extracted:")
        for key, value in metadata.items():
            if key != 'extraction_stats':
                logger.info(f"  - {key}: {value}")

        logger.info(f"\n📈 Extraction Statistics:")
        stats = metadata.get('extraction_stats', {})
        for key, value in stats.items():
            logger.info(f"  - {key}: {value}")

        # Check data quality
        logger.info("\n" + "=" * 80)
        logger.info("✅ DATA QUALITY CHECK")
        logger.info("=" * 80)

        total_content = sum(len(str(content)) for content in documents.values())

        if total_content < 1000:
            logger.warning("⚠️  Low content extracted - may need to check file processing")
        elif total_content < 5000:
            logger.info("📊 Moderate content extracted - basic RFP data available")
        else:
            logger.info("✅ Substantial content extracted - comprehensive RFP data available")

        # Check for key document types
        has_info = any('info' in key.lower() for key in documents.keys())
        has_main = any('main' in key.lower() or 'rfp' in key.lower() for key in documents.keys())
        has_tech = any('tech' in key.lower() or 'spec' in key.lower() for key in documents.keys())

        logger.info(f"\nDocument completeness:")
        logger.info(f"  - Has info document: {'✅' if has_info else '❌'}")
        logger.info(f"  - Has main RFP: {'✅' if has_main else '❌'}")
        logger.info(f"  - Has technical specs: {'✅' if has_tech else '❌'}")

        # Save extracted content for review
        output_dir = Path("test_extraction_output")
        output_dir.mkdir(exist_ok=True)

        # Save documents
        for doc_name, content in documents.items():
            safe_name = doc_name.replace('/', '_').replace(' ', '_')
            output_file = output_dir / f"{safe_name}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)

        # Save metadata
        import json
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"\n💾 Extracted content saved to: {output_dir.absolute()}")

        logger.info("\n" + "=" * 80)
        logger.info("🎉 TEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        # Clean up
        extractor.cleanup()

    except Exception as e:
        logger.error(f"\n❌ Test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

    return True

if __name__ == "__main__":
    success = asyncio.run(test_drive_extraction())
    sys.exit(0 if success else 1)