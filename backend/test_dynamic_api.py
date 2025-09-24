#!/usr/bin/env python3
"""Test Dynamic multi-agent architecture via API"""

import asyncio
import httpx
from datetime import datetime
import json

async def test_dynamic_via_api():
    print("\n" + "="*80)
    print("TESTING DYNAMIC ARCHITECTURE VIA API")
    print("="*80 + "\n")

    base_url = "http://localhost:8000"
    session_id = f"dynamic_api_test_{datetime.now().strftime('%H%M%S')}"

    async with httpx.AsyncClient(timeout=300) as client:
        # Step 1: Set architecture to Dynamic
        print("1. Setting architecture to Dynamic...")
        response = await client.post(
            f"{base_url}/api/architecture/config",
            json={
                "session_id": session_id,
                "mode": "dynamic"
            }
        )

        if response.status_code == 200:
            result = response.json()
            config = result.get('config', {})
            print(f"✓ Architecture set to: {config.get('mode', 'Unknown')}")
            print(f"  Message: {result.get('message', 'N/A')}")
        else:
            print(f"✗ Failed to set architecture: {response.text}")
            return

        # Step 2: Process RFP with Dynamic architecture
        print("\n2. Processing RFP with Dynamic multi-agent architecture...")
        print("   This will use specialized agents for different documents...\n")

        test_data = {
            "notice_id": f"DYNAMIC_API_{datetime.now().strftime('%H%M%S')}",
            "session_id": session_id,
            "northstar_doc": """
# Northstar Technologies - Company Capabilities

## Core Competencies
- Cloud Architecture & Migration (AWS, Azure, GCP certified)
- DevSecOps & CI/CD Implementation
- AI/ML Solutions Development
- Cybersecurity & Zero Trust Architecture
- 24/7 SOC Operations

## Federal Experience
- 15+ years supporting federal agencies
- Active facility clearance
- FedRAMP High certified solutions
- CMMC Level 2 compliant

## Key Differentiators
- Proprietary AI-driven threat detection
- 99.99% uptime SLA guarantee
- Cleared staff readily available
- Rapid deployment capability (< 30 days)
""",
            "rfp_documents": {
                "requirements": """
## Statement of Work
The Government requires comprehensive cloud modernization services including:

1. **Cloud Migration Services**
   - Assessment of current infrastructure
   - Migration strategy and roadmap
   - Zero-downtime migration execution
   - Post-migration optimization

2. **Security Requirements**
   - Implement Zero Trust Architecture
   - Continuous monitoring and threat detection
   - Incident response capability
   - Quarterly security assessments

3. **Deliverables**
   - Technical Approach Document
   - Cost Breakdown Structure
   - Past Performance References
   - Security Compliance Matrix

4. **Period of Performance**
   - Base Year + 4 Option Years
   - Start Date: October 1, 2025
"""
            },
            "company_context": "Leading federal cloud solutions provider with focus on security and compliance"
        }

        try:
            response = await client.post(
                f"{base_url}/api/rfps/process-with-architecture",
                json=test_data,
                timeout=120  # 2 minute timeout
            )

            if response.status_code == 200:
                result = response.json()

                print("✅ DYNAMIC PROCESSING COMPLETED!")
                print(f"\n📊 Results:")
                print(f"  • Architecture: {result.get('architecture_used', 'Unknown')}")
                print(f"  • Processing time: {result.get('processing_time', 0):.2f} seconds")
                print(f"  • Documents generated: {result.get('documents_generated', 0)}")
                print(f"  • Uniqueness score: {result.get('uniqueness_score', 0):.2f}/100")

                # Show agent assignments
                if 'agent_assignments' in result and result['agent_assignments']:
                    print(f"\n🤖 Agent Assignments:")
                    for doc, agent in result['agent_assignments'].items():
                        print(f"  • {agent} → {doc}")

                # Show document details
                if 'documents' in result and result['documents']:
                    print(f"\n📄 Generated Documents:")
                    for doc_name, content in result['documents'].items():
                        print(f"  • {doc_name}:")
                        print(f"    - Size: {len(content)} characters")
                        # Show preview
                        preview = content[:150] + "..." if len(content) > 150 else content
                        print(f"    - Preview: {preview}")

                # Show quality scores
                if 'quality_scores' in result:
                    print(f"\n⭐ Quality Scores:")
                    total_score = 0
                    for doc, score in result['quality_scores'].items():
                        print(f"  • {doc}: {score:.1f}/100")
                        total_score += score
                    avg_score = total_score / len(result['quality_scores'])
                    print(f"  • Average: {avg_score:.1f}/100")

                # Show output location
                if 'output_location' in result:
                    print(f"\n💾 Output saved to: {result['output_location']}")

            else:
                print(f"✗ Processing failed: {response.status_code}")
                error_text = response.text[:500] if len(response.text) > 500 else response.text
                print(f"Error: {error_text}")

        except httpx.TimeoutException:
            print("⏰ Request timed out after 2 minutes")
            print("The Dynamic architecture may need performance optimization")
        except Exception as e:
            print(f"❌ Error during processing: {e}")

    print("\n" + "="*80)
    print("API TEST COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(test_dynamic_via_api())