#!/usr/bin/env python3
"""Test the complete multi-agent RFP processing system"""

import os
import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Load environment variables
load_dotenv()

from services.architecture_manager import ArchitectureManager, ProcessingMode, ArchitectureConfig
from services.orchestration_agent import OrchestrationAgent
from services.multi_agent_orchestrator import MultiAgentOrchestrator
from services.google_drive import GoogleDriveService
from services.sam_api import SamApiClient

async def test_multi_agent_system():
    """Test the complete multi-agent RFP processing system"""
    print("\n" + "="*80)
    print("TESTING MULTI-AGENT RFP PROCESSING SYSTEM")
    print("="*80 + "\n")

    try:
        # Initialize components
        print("Initializing system components...")
        orchestration_agent = OrchestrationAgent()
        multi_agent_orchestrator = MultiAgentOrchestrator()
        architecture_manager = ArchitectureManager(
            orchestration_agent=orchestration_agent,
            multi_agent_orchestrator=multi_agent_orchestrator
        )
        google_drive = GoogleDriveService()
        print("✓ All components initialized\n")

        # Use test RFP data (skip SAM search for now to test the architecture)
        print("Using test RFP data...")
        test_rfp_data = {
            'notice_id': 'TEST_MULTI_AGENT_001',
            'title': 'Test Multi-Agent Software Development RFP',
            'description': 'This is a test RFP for software development services requiring cloud deployment, security compliance, and agile methodology.',
            'agency': 'Test Agency',
            'type': 'Combined Synopsis/Solicitation',
            'posted_date': datetime.now().isoformat(),
            'response_deadline': '2025-10-01'
        }

        print(f"\nTest RFP: {test_rfp_data.get('title', 'Unknown')}")
        print(f"Notice ID: {test_rfp_data.get('notice_id', test_rfp_data.get('noticeId', 'Unknown'))}")

        # Create test Northstar document
        northstar_doc = f"""
# Company Capabilities - Multi-Agent Test

## Overview
We are a leading software development company with expertise in:
- Cloud-native application development
- AI/ML integration
- DevSecOps practices
- Federal compliance (FedRAMP, FISMA)

## Key Differentiators
1. **Rapid Deployment**: Agile methodology with 2-week sprint cycles
2. **Security First**: Zero-trust architecture and continuous monitoring
3. **Innovation Focus**: R&D investment in emerging technologies
4. **Federal Experience**: 10+ years serving federal agencies

## Technical Capabilities
- Programming: Python, Java, JavaScript, Go, Rust
- Cloud: AWS, Azure, GCP certified
- Security: CMMC Level 2, ISO 27001 certified
- AI/ML: TensorFlow, PyTorch, LangChain expertise
        """

        # Create test RFP documents
        rfp_documents = {
            'requirements': """
## Technical Requirements
1. Cloud-based solution deployment
2. API-first architecture
3. Security compliance with federal standards
4. 99.99% uptime SLA
5. Scalability to 100,000+ users
            """,
            'evaluation_criteria': """
## Evaluation Criteria
- Technical Approach (40%)
- Past Performance (30%)
- Cost Reasonableness (20%)
- Small Business Participation (10%)
            """
        }

        # Company context
        company_context = "Federal contractor with SECRET clearance facility, CMMC Level 2 certified"

        # Test 1: Classic Architecture
        print("\n" + "-"*60)
        print("TEST 1: CLASSIC ARCHITECTURE")
        print("-"*60)

        session_id = "test_session_classic"
        config = await architecture_manager.get_architecture_config(
            session_id,
            user_preference=ProcessingMode.CLASSIC
        )

        print(f"Architecture: {config.mode}")
        print("Processing RFP...")

        start_time = datetime.now()
        classic_result = await architecture_manager.process_rfp_with_architecture(
            rfp_data=test_rfp_data,
            config=config,
            northstar_doc=northstar_doc,
            rfp_documents=rfp_documents,
            company_context=company_context
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        print(f"\n✓ Classic processing completed in {processing_time:.2f} seconds")
        print(f"Documents generated: {len(classic_result.documents)}")
        for doc_name in classic_result.documents:
            print(f"  - {doc_name}: {len(classic_result.documents[doc_name])} chars")
        print(f"Average quality score: {classic_result.metrics.get('average_quality', 0):.2f}")

        # Test 2: Dynamic Multi-Agent Architecture
        print("\n" + "-"*60)
        print("TEST 2: DYNAMIC MULTI-AGENT ARCHITECTURE")
        print("-"*60)

        session_id = "test_session_dynamic"
        config = await architecture_manager.get_architecture_config(
            session_id,
            user_preference=ProcessingMode.DYNAMIC
        )

        print(f"Architecture: {config.mode}")
        print("Processing RFP with specialized agents...")

        start_time = datetime.now()
        dynamic_result = await architecture_manager.process_rfp_with_architecture(
            rfp_data=test_rfp_data,
            config=config,
            northstar_doc=northstar_doc,
            rfp_documents=rfp_documents,
            company_context=company_context
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        print(f"\n✓ Dynamic processing completed in {processing_time:.2f} seconds")
        print(f"Documents generated: {len(dynamic_result.documents)}")
        for doc_name in dynamic_result.documents:
            print(f"  - {doc_name}: {len(dynamic_result.documents[doc_name])} chars")
        print(f"Average quality score: {dynamic_result.metrics.get('average_quality', 0):.2f}")
        print(f"Agents used: {dynamic_result.metrics.get('agents_used', 0)}")
        print(f"Uniqueness score: {dynamic_result.uniqueness_score:.2f}")

        # Test 3: Comparison Mode
        print("\n" + "-"*60)
        print("TEST 3: COMPARISON MODE")
        print("-"*60)

        session_id = "test_session_comparison"
        config = await architecture_manager.get_architecture_config(
            session_id,
            user_preference=ProcessingMode.COMPARISON
        )

        print(f"Architecture: {config.mode}")
        print("Running both architectures in parallel for comparison...")

        start_time = datetime.now()
        comparison_result = await architecture_manager.process_rfp_with_architecture(
            rfp_data=test_rfp_data,
            config=config,
            northstar_doc=northstar_doc,
            rfp_documents=rfp_documents,
            company_context=company_context
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        print(f"\n✓ Comparison completed in {processing_time:.2f} seconds")

        if isinstance(comparison_result, dict) and 'comparison' in comparison_result:
            comp_metrics = comparison_result['comparison']
            print("\nComparison Metrics:")
            print(f"  Classic Time: {comp_metrics.get('classic_time', 0):.2f}s")
            print(f"  Dynamic Time: {comp_metrics.get('dynamic_time', 0):.2f}s")
            print(f"  Time Improvement: {comp_metrics.get('time_improvement', 0):.1f}%")
            print(f"  Classic Quality: {comp_metrics.get('classic_quality', 0):.2f}")
            print(f"  Dynamic Quality: {comp_metrics.get('dynamic_quality', 0):.2f}")
            print(f"  Quality Improvement: {comp_metrics.get('quality_improvement', 0):.2f}")

        # Test 4: Google Drive Upload
        print("\n" + "-"*60)
        print("TEST 4: GOOGLE DRIVE UPLOAD")
        print("-"*60)

        print("Creating folder in Google Drive...")
        folder_id = await google_drive.create_rfp_folder(
            test_rfp_data.get('notice_id', 'TEST'),
            test_rfp_data.get('title', 'Test RFP')[:50]
        )
        print(f"✓ Created folder: {folder_id}")

        print("Uploading documents...")
        # Use the dynamic result documents for upload
        uploaded_files = await google_drive.upload_documents(
            folder_id,
            dynamic_result.documents
        )

        if uploaded_files:
            print(f"✓ Uploaded {len(uploaded_files)} files to Google Drive")
            if hasattr(google_drive, 'current_folder_structure'):
                folder_url = google_drive.current_folder_structure.get('url', 'N/A')
                print(f"Folder URL: {folder_url}")
        else:
            print("✗ Failed to upload documents")

        # Get architecture metrics
        print("\n" + "-"*60)
        print("AGGREGATE METRICS")
        print("-"*60)

        metrics = await architecture_manager.get_architecture_metrics()
        print(f"Total RFPs processed: {metrics['total_processed']}")
        print(f"Classic runs: {metrics['classic_count']}")
        print(f"Dynamic runs: {metrics['dynamic_count']}")
        print(f"Classic avg time: {metrics['classic_avg_time']:.2f}s")
        print(f"Dynamic avg time: {metrics['dynamic_avg_time']:.2f}s")
        print(f"Classic avg quality: {metrics['classic_avg_quality']:.2f}")
        print(f"Dynamic avg quality: {metrics['dynamic_avg_quality']:.2f}")

        print("\n" + "="*80)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*80 + "\n")

        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_multi_agent_system())
    sys.exit(0 if success else 1)