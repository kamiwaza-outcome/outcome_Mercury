#!/usr/bin/env python3
"""Quick test of Dynamic multi-agent architecture with minimal processing"""

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

async def test_dynamic_quick():
    print("\n" + "="*80)
    print("QUICK DYNAMIC MULTI-AGENT TEST")
    print("="*80 + "\n")

    # Initialize components
    print("Initializing components...")
    orchestration_agent = OrchestrationAgent()
    multi_agent_orchestrator = MultiAgentOrchestrator()
    architecture_manager = ArchitectureManager(
        orchestration_agent=orchestration_agent,
        multi_agent_orchestrator=multi_agent_orchestrator
    )

    # Minimal test data for speed
    test_rfp_data = {
        'notice_id': f'DYNAMIC_QUICK_{datetime.now().strftime("%H%M%S")}',
        'title': 'Quick Test',
        'description': 'Quick test',
        'agency': 'Test',
        'type': 'RFP'
    }

    # Minimal Northstar
    northstar_doc = "We provide cloud services."

    # Minimal requirements
    rfp_documents = {
        'requirements': "Need cloud services"
    }

    company_context = "Cloud provider"

    # Get Dynamic configuration
    print("\nSetting Dynamic architecture mode...")
    config = await architecture_manager.get_architecture_config(
        session_id="dynamic_quick_test",
        user_preference=ProcessingMode.DYNAMIC
    )
    print(f"✓ Mode set to: {config.mode}")

    # Test just the agent assignment without full generation
    print("\nTesting agent assignment logic...")

    try:
        # Test confidence scoring
        print("Testing agent confidence assessment...")
        for agent in multi_agent_orchestrator.agents[:3]:  # Test first 3 agents
            confidence = await agent.assess_confidence(
                rfp_data=test_rfp_data,
                document_type="Technical_Proposal"
            )
            print(f"  • {agent.__class__.__name__}: confidence {confidence:.2f}")

        print("\n✅ Agent confidence assessment working!")

        # Now test a quick document generation with timeout
        print("\nTesting quick document generation (30 second timeout)...")

        # Override to generate just one document quickly
        test_documents = ["Executive_Summary"]

        result = await asyncio.wait_for(
            multi_agent_orchestrator.generate_rfp_response(
                rfp_data=test_rfp_data,
                northstar=northstar_doc,
                rfp_documents=rfp_documents,
                company_context=company_context,
                required_documents=test_documents
            ),
            timeout=30  # 30 second timeout for quick test
        )

        print(f"\n✅ DYNAMIC PROCESSING COMPLETED!")
        print(f"  Architecture: Dynamic Multi-Agent")
        print(f"  Documents generated: {len(result.get('documents', {}))}")

        if 'agent_assignments' in result:
            print("\n📋 Agent Assignments:")
            for doc, agent in result['agent_assignments'].items():
                print(f"  • {agent} → {doc}")

        if 'documents' in result:
            print("\n📄 Generated Documents:")
            for doc_name, content in result['documents'].items():
                print(f"  • {doc_name}: {len(content)} characters")

        # Save to output folder
        output_dir = Path(f"output/{test_rfp_data['notice_id']}")
        output_dir.mkdir(parents=True, exist_ok=True)

        if 'documents' in result:
            for doc_name, content in result['documents'].items():
                file_path = output_dir / doc_name
                with open(file_path, 'w') as f:
                    f.write(content)
            print(f"\n💾 Documents saved to: {output_dir}")

    except asyncio.TimeoutError:
        print("\n⏰ Timed out after 30 seconds")
        print("Even the quick test is taking too long - investigating...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("QUICK TEST COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(test_dynamic_quick())