#!/usr/bin/env python3
"""Test architecture switching between Classic and Dynamic modes"""

import asyncio
import httpx
import json
from datetime import datetime

async def test_architecture_switching():
    """Test switching between Classic and Dynamic architectures via API"""
    print("\n" + "="*80)
    print("TESTING ARCHITECTURE SWITCHING")
    print("="*80 + "\n")

    base_url = "http://localhost:8000"

    async with httpx.AsyncClient(timeout=120) as client:
        # Test 1: Set Classic mode
        print("TEST 1: Setting Classic Architecture")
        print("-" * 40)

        response = await client.post(
            f"{base_url}/api/architecture/config",
            json={
                "session_id": "test_classic_001",
                "mode": "classic"
            }
        )

        if response.status_code == 200:
            config = response.json()
            print(f"✓ Mode set to: {config['mode']}")
            print(f"  Session: {config['session_id']}")
            print(f"  User preference: {config['user_preference']}")
        else:
            print(f"✗ Failed to set mode: {response.status_code}")
            print(response.text)

        # Test 2: Set Dynamic mode
        print("\nTEST 2: Setting Dynamic Architecture")
        print("-" * 40)

        response = await client.post(
            f"{base_url}/api/architecture/config",
            json={
                "session_id": "test_dynamic_001",
                "mode": "dynamic"
            }
        )

        if response.status_code == 200:
            config = response.json()
            print(f"✓ Mode set to: {config['mode']}")
            print(f"  Session: {config['session_id']}")
            print(f"  User preference: {config['user_preference']}")
        else:
            print(f"✗ Failed to set mode: {response.status_code}")
            print(response.text)

        # Test 3: Get architecture metrics
        print("\nTEST 3: Getting Architecture Metrics")
        print("-" * 40)

        response = await client.get(f"{base_url}/api/architecture/metrics")

        if response.status_code == 200:
            metrics = response.json()
            print("✓ Retrieved metrics:")
            print(f"  Total processed: {metrics.get('total_processed', 0)}")
            print(f"  Classic runs: {metrics.get('classic_count', 0)}")
            print(f"  Dynamic runs: {metrics.get('dynamic_count', 0)}")
            print(f"  Classic avg time: {metrics.get('classic_avg_time', 0):.2f}s")
            print(f"  Dynamic avg time: {metrics.get('dynamic_avg_time', 0):.2f}s")
        else:
            print(f"✗ Failed to get metrics: {response.status_code}")

        # Test 4: Process test RFP with Classic mode
        print("\nTEST 4: Processing Test RFP with Classic Mode")
        print("-" * 40)

        test_rfp_request = {
            "notice_id": "TEST_ARCH_CLASSIC",
            "session_id": "test_classic_001",
            "northstar_doc": """
# Test Company Capabilities
We are a test company with software development expertise.
## Key Capabilities
- Cloud development
- Security compliance
- Federal experience
            """,
            "rfp_documents": {
                "requirements": "Develop a cloud-based solution with security compliance.",
                "evaluation": "Technical approach 50%, Cost 30%, Past performance 20%"
            },
            "company_context": "Test company with federal experience"
        }

        print("Submitting RFP for Classic processing...")
        response = await client.post(
            f"{base_url}/api/rfps/process-with-architecture",
            json=test_rfp_request,
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✓ Classic processing completed")
            print(f"  Architecture used: {result.get('architecture_used', 'unknown')}")
            print(f"  Processing time: {result.get('processing_time', 0):.2f}s")
            print(f"  Documents generated: {result.get('documents_generated', 0)}")
            if 'quality_scores' in result:
                avg_quality = sum(result['quality_scores'].values()) / len(result['quality_scores']) if result['quality_scores'] else 0
                print(f"  Average quality: {avg_quality:.2f}")
        else:
            print(f"✗ Classic processing failed: {response.status_code}")
            print(response.text[:500])

        # Test 5: Process test RFP with Dynamic mode
        print("\nTEST 5: Processing Test RFP with Dynamic Mode")
        print("-" * 40)

        test_rfp_request["notice_id"] = "TEST_ARCH_DYNAMIC"
        test_rfp_request["session_id"] = "test_dynamic_001"

        print("Submitting RFP for Dynamic processing...")
        response = await client.post(
            f"{base_url}/api/rfps/process-with-architecture",
            json=test_rfp_request,
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✓ Dynamic processing completed")
            print(f"  Architecture used: {result.get('architecture_used', 'unknown')}")
            print(f"  Processing time: {result.get('processing_time', 0):.2f}s")
            print(f"  Documents generated: {result.get('documents_generated', 0)}")
            print(f"  Uniqueness score: {result.get('uniqueness_score', 0):.2f}")
            if 'agent_assignments' in result:
                print(f"  Agents used: {len(set(result['agent_assignments'].values()))}")
        else:
            print(f"✗ Dynamic processing failed: {response.status_code}")
            print(response.text[:500])

        # Test 6: Comparison mode
        print("\nTEST 6: Running Comparison Mode")
        print("-" * 40)

        comparison_request = {
            "notice_id": "TEST_COMPARISON",
            "northstar_doc": test_rfp_request["northstar_doc"],
            "rfp_documents": test_rfp_request["rfp_documents"],
            "company_context": test_rfp_request["company_context"]
        }

        print("Running both architectures in parallel...")
        response = await client.post(
            f"{base_url}/api/rfps/process-comparison",
            json=comparison_request,
            timeout=180
        )

        if response.status_code == 200:
            result = response.json()
            print("✓ Comparison completed")

            if 'classic' in result:
                print("\n  Classic Results:")
                print(f"    Processing time: {result['classic'].get('processing_time', 0):.2f}s")
                print(f"    Documents: {len(result['classic'].get('documents', {}))}")

            if 'dynamic' in result:
                print("\n  Dynamic Results:")
                print(f"    Processing time: {result['dynamic'].get('processing_time', 0):.2f}s")
                print(f"    Documents: {len(result['dynamic'].get('documents', {}))}")
                print(f"    Uniqueness: {result['dynamic'].get('uniqueness_score', 0):.2f}")

            if 'comparison' in result:
                comp = result['comparison']
                print("\n  Comparison Metrics:")
                print(f"    Time improvement: {comp.get('time_improvement', 0):.1f}%")
                print(f"    Quality improvement: {comp.get('quality_improvement', 0):.2f}")
        else:
            print(f"✗ Comparison failed: {response.status_code}")
            print(response.text[:500])

    print("\n" + "="*80)
    print("ARCHITECTURE SWITCHING TESTS COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(test_architecture_switching())