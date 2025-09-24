#!/usr/bin/env python3
"""Direct test of Dynamic multi-agent architecture"""

import asyncio
import httpx

async def test_dynamic():
    print("\n" + "="*80)
    print("TESTING DYNAMIC MULTI-AGENT ARCHITECTURE")
    print("="*80 + "\n")

    base_url = "http://localhost:8000"

    async with httpx.AsyncClient(timeout=300) as client:
        # Set Dynamic mode
        print("1. Setting architecture to Dynamic...")
        response = await client.post(
            f"{base_url}/api/architecture/config",
            json={
                "session_id": "test_dynamic_direct",
                "mode": "dynamic"
            }
        )

        if response.status_code == 200:
            print(f"✓ Set to Dynamic mode")
        else:
            print(f"✗ Failed: {response.text}")
            return

        # Process with Dynamic
        print("\n2. Processing with Dynamic multi-agent architecture...")
        response = await client.post(
            f"{base_url}/api/rfps/process-with-architecture",
            json={
                "notice_id": "TEST_DYNAMIC_DIRECT",
                "session_id": "test_dynamic_direct",
                "northstar_doc": """
# Company Capabilities
We are a cloud solutions provider with:
- 10+ years federal experience
- FedRAMP High certified
- 24/7 SOC operations
- Agile development expertise
                """,
                "rfp_documents": {
                    "requirements": "Need cloud migration services with security focus"
                },
                "company_context": "Federal cloud provider"
            },
            timeout=300
        )

        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ Dynamic processing completed!")
            print(f"  Architecture: {result.get('architecture_used')}")
            print(f"  Processing time: {result.get('processing_time', 0):.2f}s")
            print(f"  Documents generated: {result.get('documents_generated', 0)}")
            print(f"  Uniqueness score: {result.get('uniqueness_score', 0):.2f}")

            if 'agent_assignments' in result and result['agent_assignments']:
                print("\n  Agent assignments:")
                for doc, agent in result['agent_assignments'].items():
                    print(f"    {agent} -> {doc}")

            if 'documents' in result:
                print("\n  Generated documents:")
                for doc_name in result['documents']:
                    print(f"    - {doc_name}")
        else:
            print(f"\n✗ Failed: {response.status_code}")
            print(response.text[:500])

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(test_dynamic())