#!/usr/bin/env python3
"""
Production Dynamic Multi-Agent RFP Processing System
Runs full RFP processing with comprehensive Northstar and retry logic
"""

import asyncio
import json
import time
from pathlib import Path
from datetime import datetime
import httpx
from typing import Dict, Any

# Production configuration with retry logic
PRODUCTION_CONFIG = {
    "architecture": "dynamic",
    "quality_mode": "maximum",
    "retry_attempts": 3,
    "retry_delay": 5,
    "timeout": 3600,  # 60 minutes
    "api_settings": {
        "reasoning_effort": "high",
        "verbosity": "medium",
        "temperature": 0.7
    }
}

async def load_comprehensive_context():
    """Load comprehensive Northstar and RFP context"""
    print("\n📚 Loading comprehensive context...")

    # Use the high-quality Northstar from previous run
    northstar_path = Path("output/9a1e55199b1f4c1da83c14d548723c24/Northstar_Document.md")

    if northstar_path.exists():
        with open(northstar_path, 'r') as f:
            northstar = f.read()
        print(f"  ✓ Loaded Northstar: {len(northstar):,} characters")
    else:
        print("  ⚠️ Northstar not found, using default context")
        northstar = """# Mercury Technologies - Enterprise RFP Response Framework

## Company Overview
Mercury Technologies is a leading provider of advanced technology solutions specializing in cloud modernization,
security services, and digital transformation for federal and enterprise clients."""

    # Load RFP requirements
    rfp_requirements = """## Statement of Work - Cloud Modernization and Security Services

### 1. SCOPE OF WORK
The Contractor shall provide comprehensive cloud modernization and security services including:
- Cloud architecture design and implementation
- Security assessment and remediation
- DevSecOps pipeline development
- Zero Trust architecture implementation
- Continuous monitoring and compliance

### 2. TECHNICAL REQUIREMENTS
#### 2.1 Cloud Services
- Multi-cloud strategy (AWS, Azure, GCP)
- Containerization and orchestration (Kubernetes)
- Infrastructure as Code (Terraform, CloudFormation)
- Serverless architecture implementation

#### 2.2 Security Requirements
- NIST 800-53 compliance
- FedRAMP authorization support
- Continuous ATO processes
- Security Operations Center (SOC) services

### 3. DELIVERABLES
- Technical Architecture Document
- Security Assessment Report
- Implementation Roadmap
- Monthly Progress Reports
- Training Materials

### 4. EVALUATION CRITERIA
Technical Approach (40%), Past Performance (30%), Cost (20%), Management Approach (10%)"""

    # Company context
    company_context = """Kamiwaza Technologies specializes in:
- Cloud modernization and migration
- DevSecOps implementation
- Zero Trust security architecture
- AI/ML solutions
- Agile transformation
- Federal compliance expertise (FedRAMP, FISMA, NIST)

Key Differentiators:
- 15+ years federal experience
- Current Top Secret facility clearance
- ISO 27001, CMMI Level 3 certified
- GSA Schedule holder
- 8(a) certified small business"""

    return {
        "northstar_doc": northstar,
        "rfp_requirements": rfp_requirements,
        "company_context": company_context,
        "total_context": len(northstar) + len(rfp_requirements) + len(company_context)
    }

async def process_rfp_with_retry(client: httpx.AsyncClient, context: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    """Process RFP with retry logic for API errors"""

    rfp_id = f"PROD_DYNAMIC_{datetime.now().strftime('%H%M%S')}"

    payload = {
        "notice_id": rfp_id,
        "session_id": session_id,
        "northstar_doc": context["northstar_doc"],
        "rfp_documents": {
            "requirements": context["rfp_requirements"],
            "context": context["company_context"]
        }
    }

    for attempt in range(PRODUCTION_CONFIG["retry_attempts"]):
        try:
            print(f"\n🚀 Processing attempt {attempt + 1}/{PRODUCTION_CONFIG['retry_attempts']}...")

            # Send processing request
            response = await client.post(
                "http://localhost:8000/api/rfps/process-with-architecture",
                json=payload,
                timeout=PRODUCTION_CONFIG["timeout"]
            )

            if response.status_code == 200:
                result = response.json()
                print(f"  ✓ Processing initiated: {result.get('notice_id', rfp_id)}")
                return result
            elif response.status_code == 502:
                print(f"  ⚠️ API Gateway error (502), retrying in {PRODUCTION_CONFIG['retry_delay']}s...")
                await asyncio.sleep(PRODUCTION_CONFIG["retry_delay"])
            else:
                print(f"  ❌ Error: {response.status_code} - {response.text}")
                if attempt < PRODUCTION_CONFIG["retry_attempts"] - 1:
                    await asyncio.sleep(PRODUCTION_CONFIG["retry_delay"])

        except httpx.TimeoutException:
            print(f"  ⚠️ Request timeout, retrying...")
            await asyncio.sleep(PRODUCTION_CONFIG["retry_delay"])
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            if attempt < PRODUCTION_CONFIG["retry_attempts"] - 1:
                await asyncio.sleep(PRODUCTION_CONFIG["retry_delay"])

    raise Exception("Failed to process RFP after all retry attempts")

async def monitor_processing(client: httpx.AsyncClient, rfp_id: str):
    """Monitor RFP processing with detailed status updates"""

    print("\n📊 Monitoring processing status...")
    start_time = time.time()
    last_status = None

    while True:
        try:
            response = await client.get(f"http://localhost:8000/api/rfps/status/{rfp_id}")

            if response.status_code == 200:
                status_data = response.json()
                current_status = status_data.get("status", "unknown")

                if current_status != last_status:
                    elapsed = time.time() - start_time
                    print(f"  [{elapsed:.1f}s] Status: {current_status}")

                    # Print document progress
                    if "documents" in status_data:
                        for doc in status_data["documents"]:
                            print(f"    • {doc['name']}: {doc.get('status', 'pending')}")

                    last_status = current_status

                if current_status in ["completed", "error"]:
                    return status_data

            await asyncio.sleep(5)

        except Exception as e:
            print(f"  ⚠️ Monitoring error: {str(e)}")
            await asyncio.sleep(5)

async def main():
    """Main production processing flow"""

    print("=" * 80)
    print("PRODUCTION DYNAMIC MULTI-AGENT RFP PROCESSING")
    print("=" * 80)

    # Configure architecture
    async with httpx.AsyncClient() as client:
        # First create a unique session ID
        session_id = f"prod_session_{datetime.now().strftime('%H%M%S')}"

        print(f"\n⚙️ Configuring Dynamic Multi-Agent architecture (session: {session_id})...")
        config_response = await client.post(
            "http://localhost:8000/api/architecture/config",
            json={"mode": "dynamic", "session_id": session_id}
        )
        if config_response.status_code == 200:
            print("  ✓ Dynamic architecture configured")
            config_data = config_response.json()
            session_id = config_data.get("session_id", session_id)

        # Load context
        context = await load_comprehensive_context()
        print(f"\n📝 Total context size: {context['total_context']:,} characters")

        # Process with retry logic
        try:
            result = await process_rfp_with_retry(client, context, session_id)
            rfp_id = result.get("notice_id", result.get("rfp_id"))

            # Monitor processing
            final_status = await monitor_processing(client, rfp_id)

            # Display results
            print("\n" + "=" * 80)
            if final_status.get("status") == "completed":
                print("✅ PRODUCTION RFP PROCESSING COMPLETED!")
                print("=" * 80)

                # Display metrics
                if "metrics" in final_status:
                    metrics = final_status["metrics"]
                    print(f"\n📊 Performance Metrics:")
                    print(f"  • Processing time: {metrics.get('processing_time', 'N/A')} seconds")
                    print(f"  • Documents generated: {metrics.get('document_count', 0)}")
                    print(f"  • Average quality: {metrics.get('average_quality', 0):.1f}/100")
                    print(f"  • Total output: {metrics.get('total_output', 0):,} characters")

                # Display document details
                if "documents" in final_status:
                    print(f"\n📄 Generated Documents:")
                    for doc in final_status["documents"]:
                        print(f"  • {doc['name']}: {doc.get('quality_score', 0)}/100 ({doc.get('size', 0):,} chars)")

                print(f"\n💾 Output saved to: output/{rfp_id}")

            else:
                print("❌ PRODUCTION RFP PROCESSING FAILED")
                print("=" * 80)
                print(f"Error: {final_status.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"\n❌ Fatal error: {str(e)}")
            return 1

    print("\n" + "=" * 80)
    print("PRODUCTION RUN COMPLETE")
    print("=" * 80)
    return 0

if __name__ == "__main__":
    exit(asyncio.run(main()))