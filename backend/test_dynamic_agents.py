#!/usr/bin/env python3
"""Direct test of Dynamic multi-agent architecture with timeout handling"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent))
load_dotenv()

from services.architecture_manager import ArchitectureManager, ProcessingMode
from services.orchestration_agent import OrchestrationAgent
from services.multi_agent_orchestrator import MultiAgentOrchestrator

async def test_dynamic_agents():
    print("\n" + "="*80)
    print("TESTING DYNAMIC MULTI-AGENT ARCHITECTURE")
    print("="*80 + "\n")

    # Initialize components
    print("Initializing components...")
    orchestration_agent = OrchestrationAgent()
    multi_agent_orchestrator = MultiAgentOrchestrator()
    architecture_manager = ArchitectureManager(
        orchestration_agent=orchestration_agent,
        multi_agent_orchestrator=multi_agent_orchestrator
    )

    # Simple test RFP data
    test_rfp_data = {
        'notice_id': f'DYNAMIC_TEST_{datetime.now().strftime("%H%M%S")}',
        'title': 'Cloud Security Services',
        'description': 'Need cloud security assessment and implementation',
        'agency': 'Test Agency',
        'type': 'RFP'
    }

    # Simple Northstar
    northstar_doc = """
# Company Capabilities for Cloud Security RFP

## Core Competencies
- Cloud Security (AWS, Azure, GCP certified)
- Zero Trust Architecture implementation
- 24/7 Security Operations Center
- FedRAMP High authorization

## Differentiators
- 10+ years federal experience
- Cleared staff available
- Rapid deployment capability
"""

    # Simple requirements
    rfp_documents = {
        'requirements': """
Technical Requirements:
1. Cloud security assessment
2. Zero Trust implementation
3. 24/7 monitoring
4. Incident response

Evaluation:
- Technical Approach (40%)
- Past Performance (30%)
- Cost (30%)
"""
    }

    company_context = "Federal cloud security provider with cleared staff"

    # Get Dynamic configuration
    print("\nSetting Dynamic architecture mode...")
    config = await architecture_manager.get_architecture_config(
        session_id="dynamic_test_session",
        user_preference=ProcessingMode.DYNAMIC
    )
    print(f"✓ Mode set to: {config.mode}")

    # Process with Dynamic architecture
    print("\nProcessing with Dynamic multi-agent architecture...")
    print("This will use specialized agents for different document types...\n")

    try:
        start_time = datetime.now()

        # Set a timeout for the processing
        result = await asyncio.wait_for(
            architecture_manager.process_rfp_with_architecture(
                rfp_data=test_rfp_data,
                config=config,
                northstar_doc=northstar_doc,
                rfp_documents=rfp_documents,
                company_context=company_context
            ),
            timeout=180  # 3 minute timeout
        )

        elapsed = (datetime.now() - start_time).total_seconds()

        print(f"\n✅ DYNAMIC PROCESSING COMPLETED!")
        print(f"  Time: {elapsed:.2f} seconds")
        print(f"  Architecture: {result.architecture_used}")
        print(f"  Documents generated: {len(result.documents)}")
        print(f"  Uniqueness score: {result.uniqueness_score:.2f}")

        # Show which agents handled which documents
        if result.agent_assignments:
            print("\n📋 Agent Assignments:")
            for doc, agent in result.agent_assignments.items():
                print(f"  • {agent} → {doc}")

        # Show generated documents
        if result.documents:
            print("\n📄 Generated Documents:")
            for doc_name, content in result.documents.items():
                print(f"  • {doc_name}: {len(content)} characters")
                # Show first 200 chars of each
                preview = content[:200] + "..." if len(content) > 200 else content
                print(f"    Preview: {preview}\n")

        # Show quality scores
        if result.quality_scores:
            print("\n⭐ Quality Scores:")
            for doc, score in result.quality_scores.items():
                print(f"  • {doc}: {score:.2f}/100")

        # Save to output folder
        output_dir = Path(f"output/{test_rfp_data['notice_id']}")
        output_dir.mkdir(parents=True, exist_ok=True)

        for doc_name, content in result.documents.items():
            file_path = output_dir / doc_name
            with open(file_path, 'w') as f:
                f.write(content)

        print(f"\n💾 Documents saved to: {output_dir}")

    except asyncio.TimeoutError:
        print("\n⏰ Processing timed out after 3 minutes")
        print("The Dynamic architecture may need optimization for speed")
    except Exception as e:
        print(f"\n❌ Error during Dynamic processing: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("DYNAMIC AGENT TEST COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(test_dynamic_agents())