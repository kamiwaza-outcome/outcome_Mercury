#!/usr/bin/env python3
"""
Test script for Mercury RFP System - Multi-Agent Architecture Integration
Tests both Classic and Dynamic architectures
"""

import asyncio
import aiohttp
import json
from datetime import datetime
import sys

API_URL = "http://localhost:8000"

async def test_architecture_system():
    """Test the complete architecture system"""

    print("=" * 80)
    print("Mercury RFP System - Architecture Integration Test")
    print("=" * 80)

    async with aiohttp.ClientSession() as session:
        # Step 1: Check system health
        print("\n1. Checking system health...")
        try:
            async with session.get(f"{API_URL}/api/architecture/metrics") as resp:
                metrics = await resp.json()
                print(f"   ✓ System online - {metrics['total_processed']} RFPs processed historically")
        except Exception as e:
            print(f"   ✗ System offline: {e}")
            return

        # Step 2: List available agents
        print("\n2. Checking available agents...")
        async with session.get(f"{API_URL}/api/agents/list") as resp:
            agents_data = await resp.json()
            print(f"   ✓ {len(agents_data['agents'])} specialized agents available:")
            for agent in agents_data['agents'][:5]:  # Show first 5
                # Handle different agent data structures
                agent_name = agent if isinstance(agent, str) else agent.get('name', 'Unknown')
                print(f"      - {agent_name}")
            if len(agents_data['agents']) > 5:
                print(f"      ... and {len(agents_data['agents']) - 5} more")

        # Step 3: Test architecture configuration
        print("\n3. Testing architecture configuration...")
        session_id = f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Test setting different modes
        modes = ['classic', 'dynamic', 'comparison']
        for mode in modes:
            config_data = {
                "mode": mode,
                "session_id": session_id,
                "config": {}
            }
            async with session.post(
                f"{API_URL}/api/architecture/config",
                json=config_data
            ) as resp:
                result = await resp.json()
                print(f"   ✓ Set mode to {mode}: {result.get('message', 'Success')}")

        # Step 4: Check pending RFPs
        print("\n4. Checking for pending RFPs...")
        async with session.get(f"{API_URL}/api/rfps/pending") as resp:
            pending_data = await resp.json()
            print(f"   Found {pending_data['count']} pending RFPs")

            if pending_data['count'] > 0:
                rfp = pending_data['rfps'][0]
                print(f"   Sample RFP:")
                print(f"      - Notice ID: {rfp['notice_id']}")
                print(f"      - Title: {rfp['title'][:60]}...")
                print(f"      - Agency: {rfp['agency']}")

                # Step 5: Test processing with different architectures
                print("\n5. Testing RFP processing with different architectures...")

                # Test with Classic architecture
                print("\n   Testing Classic Architecture:")
                process_data = {
                    "force_process_all": False,
                    "architecture_mode": "classic",
                    "session_id": f"{session_id}_classic"
                }
                async with session.post(
                    f"{API_URL}/api/rfps/process",
                    json=process_data
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        print(f"      ✓ Started processing with Classic: {result['message']}")
                    else:
                        print(f"      ✗ Failed to start Classic processing: {resp.status}")

                # Wait a bit
                await asyncio.sleep(3)

                # Test with Dynamic architecture
                print("\n   Testing Dynamic Architecture:")
                process_data = {
                    "force_process_all": False,
                    "architecture_mode": "dynamic",
                    "session_id": f"{session_id}_dynamic"
                }
                async with session.post(
                    f"{API_URL}/api/rfps/process",
                    json=process_data
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        print(f"      ✓ Started processing with Dynamic: {result['message']}")
                    else:
                        print(f"      ✗ Failed to start Dynamic processing: {resp.status}")

                # Check status
                await asyncio.sleep(5)
                print("\n6. Checking processing status...")
                async with session.get(f"{API_URL}/api/rfps/status") as resp:
                    status_data = await resp.json()
                    if status_data['rfps']:
                        for rfp_status in status_data['rfps']:
                            print(f"   RFP: {rfp_status['notice_id'][:8]}...")
                            print(f"      Status: {rfp_status['status']}")
                            print(f"      Progress: {rfp_status['progress'] * 100:.0f}%")
                            print(f"      Architecture: {rfp_status.get('architecture_used', 'Unknown')}")
                            if rfp_status['documents_generated']:
                                print(f"      Documents: {', '.join(rfp_status['documents_generated'])}")
            else:
                print("   No pending RFPs to test processing")

        # Step 7: Test uniqueness check
        print("\n7. Testing anti-template uniqueness check...")
        uniqueness_data = {
            "content": "This is a test document for Platform One Solutions.",
            "document_type": "technical_proposal",
            "rfp_context": {"agency": "DoD", "focus_area": "AI/ML"}
        }
        async with session.post(
            f"{API_URL}/api/uniqueness/check",
            json=uniqueness_data
        ) as resp:
            if resp.status == 200:
                result = await resp.json()
                print(f"   ✓ Uniqueness score: {result['uniqueness_score']:.2f}")
                print(f"   ✓ Modified content preview: {result['modified_content'][:100]}...")
            else:
                print(f"   ✗ Uniqueness check failed: {resp.status}")

        print("\n" + "=" * 80)
        print("Integration test complete!")
        print("=" * 80)

if __name__ == "__main__":
    print("Starting Mercury RFP Architecture Integration Test...")
    print("Make sure the backend server is running on port 8000")
    print("")

    try:
        asyncio.run(test_architecture_system())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        sys.exit(1)