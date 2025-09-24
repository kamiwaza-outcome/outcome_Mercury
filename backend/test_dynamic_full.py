#!/usr/bin/env python3
"""Full test of Dynamic multi-agent architecture"""

import asyncio
import httpx
from datetime import datetime
import json
from pathlib import Path
import time

async def test_dynamic_full():
    print("\n" + "="*80)
    print("FULL DYNAMIC MULTI-AGENT ARCHITECTURE TEST")
    print("="*80 + "\n")

    base_url = "http://localhost:8000"
    dynamic_session = f"dynamic_full_{datetime.now().strftime('%H%M%S')}"

    # Same test data that was used for Classic
    test_rfp_data = {
        "northstar_doc": """
# Mercury Technologies - Company Capabilities

## Core Competencies
- Cloud Architecture & Migration (AWS, Azure, GCP certified partners)
- DevSecOps & CI/CD Implementation (GitLab, Jenkins, CircleCI)
- AI/ML Solutions Development (TensorFlow, PyTorch, Custom Models)
- Cybersecurity & Zero Trust Architecture (NIST 800-207 compliant)
- 24/7 Security Operations Center (SOC 2 Type II certified)
- Agile Software Development (SAFe 5.0 certified teams)

## Federal Experience
- 15+ years supporting federal agencies (DoD, DHS, VA, HHS)
- Active Top Secret facility clearance
- FedRAMP High certified cloud solutions
- CMMC Level 2 compliant organization
- GSA Schedule 70 contract holder
- CIO-SP3 Small Business prime contractor

## Key Differentiators
- Proprietary AI-driven threat detection platform (PatriotShield™)
- 99.99% uptime SLA guarantee with financial backing
- 500+ cleared professionals readily available
- Rapid deployment capability (operational in < 30 days)
- ISO 9001:2015 and ISO 27001:2013 certified
- Woman-owned small business (WOSB) with 8(a) certification

## Past Performance
- $150M+ in federal contracts successfully delivered
- CPARS ratings averaging 4.8/5.0 across all evaluations
- Zero security incidents in 15 years of operation
- 95% customer retention rate
- Award-winning innovation (3x Federal 100 awards)
""",
        "rfp_documents": {
            "requirements": """
## Statement of Work - Cloud Modernization and Security Services

### 1. SCOPE
The Government requires comprehensive cloud modernization and cybersecurity services for critical infrastructure systems supporting 50,000+ users across 200 locations nationwide.

### 2. TECHNICAL REQUIREMENTS

#### 2.1 Cloud Migration Services
- Assess and document current on-premises infrastructure (1,000+ servers, 500+ applications)
- Develop detailed migration strategy and roadmap with zero-downtime approach
- Execute phased migration to FedRAMP High cloud environment
- Implement auto-scaling, load balancing, and disaster recovery
- Provide post-migration optimization and cost management

#### 2.2 Security Requirements
- Design and implement Zero Trust Architecture across all systems
- Deploy continuous monitoring and threat detection (SIEM/SOAR)
- Establish 24/7 Security Operations Center with 15-minute incident response SLA
- Conduct quarterly penetration testing and vulnerability assessments
- Implement data loss prevention (DLP) and encryption at rest/in transit
- Ensure compliance with FISMA, NIST 800-53, and agency-specific requirements

#### 2.3 DevSecOps Implementation
- Establish CI/CD pipelines with security scanning integration
- Implement Infrastructure as Code (IaC) using Terraform/CloudFormation
- Deploy containerization strategy using Kubernetes
- Automate security compliance checking and remediation
- Provide GitOps-based deployment workflows

### 3. DELIVERABLES
- Technical Architecture Document (30 days post-award)
- Migration Plan and Schedule (45 days post-award)
- Monthly Progress Reports with KPI metrics
- Security Assessment Reports (quarterly)
- Training materials and knowledge transfer documentation
- Source code and documentation for all custom development

### 4. PERFORMANCE REQUIREMENTS
- 99.9% system availability (measured monthly)
- < 200ms application response time for 95% of transactions
- Zero critical security vulnerabilities in production
- 100% compliance with federal security requirements
- Customer satisfaction score > 4.0/5.0

### 5. KEY PERSONNEL
Contractor shall provide:
- Project Manager (PMP certified, Secret clearance)
- Technical Lead (10+ years cloud architecture, Secret clearance)
- Security Lead (CISSP certified, Secret clearance)
- DevOps Engineers (minimum 5, AWS/Azure certified)
- Security Analysts (minimum 3, Security+ certified)

### 6. PERIOD OF PERFORMANCE
- Base Year: October 1, 2025 - September 30, 2026
- Option Year 1: October 1, 2026 - September 30, 2027
- Option Year 2: October 1, 2027 - September 30, 2028
- Option Year 3: October 1, 2028 - September 30, 2029
- Option Year 4: October 1, 2029 - September 30, 2030

### 7. EVALUATION CRITERIA
Technical Approach (40%)
Past Performance (30%)
Price (20%)
Small Business Participation (10%)
"""
        },
        "company_context": "Leading federal technology integrator specializing in secure cloud solutions and zero trust architecture implementation"
    }

    async with httpx.AsyncClient(timeout=1800) as client:  # 30 minute timeout

        # Set Dynamic mode
        print("Setting architecture to Dynamic Multi-Agent...")
        response = await client.post(
            f"{base_url}/api/architecture/config",
            json={
                "session_id": dynamic_session,
                "mode": "dynamic"
            }
        )
        print(f"✓ Dynamic mode configured\n")

        # Process with Dynamic
        print("Processing RFP with Dynamic multi-agent architecture...")
        print("Specialized agents will work on different document types")
        print("This may take 30-45 minutes with GPT-5 high reasoning...\n")

        start_time = time.time()

        try:
            response = await client.post(
                f"{base_url}/api/rfps/process-with-architecture",
                json={
                    "notice_id": f"DYNAMIC_FULL_{datetime.now().strftime('%H%M%S')}",
                    "session_id": dynamic_session,
                    **test_rfp_data
                }
            )

            dynamic_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()

                print(f"\n✅ DYNAMIC PROCESSING COMPLETED!")
                print(f"  Time: {dynamic_time:.2f} seconds ({dynamic_time/60:.1f} minutes)")
                print(f"  Documents generated: {result.get('documents_generated', 0)}")

                if 'uniqueness_score' in result:
                    print(f"  Uniqueness Score: {result.get('uniqueness_score', 0):.2f}/100")

                # Show agent assignments
                if 'agent_assignments' in result:
                    print("\n📤 Agent Assignments:")
                    for doc, agent in result['agent_assignments'].items():
                        print(f"    • {agent} → {doc}")

                # Show quality scores
                if 'quality_scores' in result:
                    print("\n⭐ Quality Scores:")
                    for doc, score in result['quality_scores'].items():
                        print(f"    • {doc}: {score:.1f}/100")
                    avg_score = sum(result['quality_scores'].values()) / len(result['quality_scores'])
                    print(f"    • Average: {avg_score:.1f}/100")

                # Save documents
                if 'documents' in result:
                    output_dir = Path(f"output/dynamic_full_{datetime.now().strftime('%H%M%S')}")
                    output_dir.mkdir(parents=True, exist_ok=True)

                    print(f"\n📄 Generated Documents:")
                    for doc_name, content in result['documents'].items():
                        file_path = output_dir / doc_name
                        with open(file_path, 'w') as f:
                            f.write(content)
                        print(f"    • {doc_name}: {len(content):,} characters")

                    print(f"\n💾 Documents saved to: {output_dir}")

                # Save full result
                with open(output_dir / "result.json", 'w') as f:
                    json.dump(result, f, indent=2)

                print("\n" + "="*80)
                print("DYNAMIC ARCHITECTURE TEST SUMMARY")
                print("="*80)
                print(f"✅ Successfully completed in {dynamic_time/60:.1f} minutes")
                print(f"✅ Generated {result.get('documents_generated', 0)} documents")
                if 'quality_scores' in result:
                    print(f"✅ Average quality score: {avg_score:.1f}/100")

            else:
                print(f"✗ Dynamic processing failed: {response.status_code}")
                print(f"Error: {response.text}")

        except httpx.TimeoutException:
            elapsed = time.time() - start_time
            print(f"\n⏰ Request timed out after {elapsed/60:.1f} minutes")
            print("The Dynamic architecture is still processing on the server")
            print("Check server logs for progress")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("DYNAMIC TEST COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(test_dynamic_full())