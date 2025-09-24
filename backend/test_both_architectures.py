#!/usr/bin/env python3
"""Test both architectures with simplified approach"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
import sys

sys.path.append(str(Path(__file__).parent))
load_dotenv()

from services.architecture_manager import ArchitectureManager, ProcessingMode, ArchitectureConfig
from services.orchestration_agent import OrchestrationAgent
from services.multi_agent_orchestrator import MultiAgentOrchestrator
from services.google_drive import GoogleDriveService

async def test_both_architectures():
    print("\n" + "="*80)
    print("TESTING BOTH ARCHITECTURES")
    print("="*80)

    # Initialize components
    print("\nInitializing components...")
    orchestration_agent = OrchestrationAgent()
    multi_agent_orchestrator = MultiAgentOrchestrator()
    architecture_manager = ArchitectureManager(
        orchestration_agent=orchestration_agent,
        multi_agent_orchestrator=multi_agent_orchestrator
    )
    google_drive = GoogleDriveService()

    # Test data
    test_rfp_data = {
        'notice_id': 'TEST_BOTH_ARCH',
        'title': 'Test Both Architectures',
        'description': 'Test RFP for both architectures'
    }

    northstar_doc = """
    # Test Company Capabilities
    We provide software development services with cloud expertise.
    """

    rfp_documents = {
        'requirements': 'Build a cloud solution'
    }

    company_context = "Test company"

    # TEST 1: Classic Architecture
    print("\n" + "-"*60)
    print("TEST 1: CLASSIC ARCHITECTURE")
    print("-"*60)

    config_classic = await architecture_manager.get_architecture_config(
        "test_classic",
        user_preference=ProcessingMode.CLASSIC
    )

    print(f"Mode: {config_classic.mode}")
    print("Processing with Classic architecture...")

    try:
        classic_result = await architecture_manager.process_rfp_with_architecture(
            rfp_data=test_rfp_data,
            config=config_classic,
            northstar_doc=northstar_doc,
            rfp_documents=rfp_documents,
            company_context=company_context
        )

        print(f"✓ Classic completed in {classic_result.processing_time:.2f}s")
        print(f"  Documents: {len(classic_result.documents)}")
        for doc_name in classic_result.documents:
            print(f"    - {doc_name}")
    except Exception as e:
        print(f"✗ Classic failed: {e}")

    # TEST 2: Dynamic Multi-Agent Architecture
    print("\n" + "-"*60)
    print("TEST 2: DYNAMIC MULTI-AGENT ARCHITECTURE")
    print("-"*60)

    config_dynamic = await architecture_manager.get_architecture_config(
        "test_dynamic",
        user_preference=ProcessingMode.DYNAMIC
    )

    print(f"Mode: {config_dynamic.mode}")
    print("Processing with Dynamic architecture...")

    try:
        dynamic_result = await architecture_manager.process_rfp_with_architecture(
            rfp_data=test_rfp_data,
            config=config_dynamic,
            northstar_doc=northstar_doc,
            rfp_documents=rfp_documents,
            company_context=company_context
        )

        print(f"✓ Dynamic completed in {dynamic_result.processing_time:.2f}s")
        print(f"  Documents: {len(dynamic_result.documents)}")
        print(f"  Uniqueness: {dynamic_result.uniqueness_score:.2f}")

        if dynamic_result.agent_assignments:
            print("  Agents used:")
            for doc, agent in dynamic_result.agent_assignments.items():
                print(f"    - {agent} -> {doc}")
    except Exception as e:
        print(f"✗ Dynamic failed: {e}")

    # TEST 3: Google Drive Upload
    print("\n" + "-"*60)
    print("TEST 3: GOOGLE DRIVE UPLOAD")
    print("-"*60)

    try:
        folder_id = await google_drive.create_rfp_folder(
            "TEST_UPLOAD",
            "Test Upload"
        )
        print(f"✓ Created folder: {folder_id}")

        test_docs = {
            "test.txt": "Test content for upload"
        }

        uploaded = await google_drive.upload_documents(folder_id, test_docs)
        if uploaded:
            print(f"✓ Uploaded {len(uploaded)} files")
            if hasattr(google_drive, 'current_folder_structure'):
                url = google_drive.current_folder_structure.get('url')
                print(f"  URL: {url}")
        else:
            print("✗ Upload failed")
    except Exception as e:
        print(f"✗ Drive error: {e}")

    print("\n" + "="*80)
    print("TESTS COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(test_both_architectures())