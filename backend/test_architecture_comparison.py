#!/usr/bin/env python3
"""Compare Classic vs Dynamic Multi-Agent Architectures"""

import asyncio
import httpx
from datetime import datetime
import json
from pathlib import Path
import time

async def test_architecture_comparison():
    print("\n" + "="*80)
    print("ARCHITECTURE COMPARISON TEST: CLASSIC vs DYNAMIC")
    print("="*80 + "\n")

    base_url = "http://localhost:8000"

    # Common test data for both architectures
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

    results = {}

    async with httpx.AsyncClient(timeout=600) as client:  # 10 minute timeout

        # TEST 1: CLASSIC ARCHITECTURE
        print("=" * 60)
        print("TESTING CLASSIC ARCHITECTURE")
        print("=" * 60 + "\n")

        classic_session = f"classic_test_{datetime.now().strftime('%H%M%S')}"

        # Set Classic mode
        print("Setting architecture to Classic...")
        response = await client.post(
            f"{base_url}/api/architecture/config",
            json={
                "session_id": classic_session,
                "mode": "classic"
            }
        )
        print(f"✓ Classic mode configured\n")

        # Process with Classic
        print("Processing RFP with Classic architecture...")
        print("(Single orchestration agent handling all documents)")
        start_time = time.time()

        try:
            response = await client.post(
                f"{base_url}/api/rfps/process-with-architecture",
                json={
                    "notice_id": f"CLASSIC_COMPARE_{datetime.now().strftime('%H%M%S')}",
                    "session_id": classic_session,
                    **test_rfp_data
                }
            )

            classic_time = time.time() - start_time

            if response.status_code == 200:
                results['classic'] = response.json()
                results['classic']['total_time'] = classic_time

                print(f"\n✅ CLASSIC PROCESSING COMPLETED!")
                print(f"  Time: {classic_time:.2f} seconds")
                print(f"  Documents: {results['classic'].get('documents_generated', 0)}")

                # Save Classic documents
                if 'documents' in results['classic']:
                    classic_dir = Path(f"output/comparison_classic_{datetime.now().strftime('%H%M%S')}")
                    classic_dir.mkdir(parents=True, exist_ok=True)

                    for doc_name, content in results['classic']['documents'].items():
                        with open(classic_dir / doc_name, 'w') as f:
                            f.write(content)
                    print(f"  Saved to: {classic_dir}")
            else:
                print(f"✗ Classic processing failed: {response.status_code}")

        except Exception as e:
            print(f"✗ Classic error: {e}")

        # TEST 2: DYNAMIC MULTI-AGENT ARCHITECTURE
        print("\n" + "=" * 60)
        print("TESTING DYNAMIC MULTI-AGENT ARCHITECTURE")
        print("=" * 60 + "\n")

        dynamic_session = f"dynamic_test_{datetime.now().strftime('%H%M%S')}"

        # Set Dynamic mode
        print("Setting architecture to Dynamic...")
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
        print("(Specialized agents working on different documents)")
        print("This may take longer but should produce higher quality output...\n")
        start_time = time.time()

        try:
            response = await client.post(
                f"{base_url}/api/rfps/process-with-architecture",
                json={
                    "notice_id": f"DYNAMIC_COMPARE_{datetime.now().strftime('%H%M%S')}",
                    "session_id": dynamic_session,
                    **test_rfp_data
                }
            )

            dynamic_time = time.time() - start_time

            if response.status_code == 200:
                results['dynamic'] = response.json()
                results['dynamic']['total_time'] = dynamic_time

                print(f"\n✅ DYNAMIC PROCESSING COMPLETED!")
                print(f"  Time: {dynamic_time:.2f} seconds")
                print(f"  Documents: {results['dynamic'].get('documents_generated', 0)}")
                print(f"  Uniqueness Score: {results['dynamic'].get('uniqueness_score', 0):.2f}")

                # Show agent assignments
                if 'agent_assignments' in results['dynamic']:
                    print("\n  Agent Assignments:")
                    for doc, agent in results['dynamic']['agent_assignments'].items():
                        print(f"    • {agent} → {doc}")

                # Save Dynamic documents
                if 'documents' in results['dynamic']:
                    dynamic_dir = Path(f"output/comparison_dynamic_{datetime.now().strftime('%H%M%S')}")
                    dynamic_dir.mkdir(parents=True, exist_ok=True)

                    for doc_name, content in results['dynamic']['documents'].items():
                        with open(dynamic_dir / doc_name, 'w') as f:
                            f.write(content)
                    print(f"\n  Saved to: {dynamic_dir}")
            else:
                print(f"✗ Dynamic processing failed: {response.status_code}")

        except Exception as e:
            print(f"✗ Dynamic error: {e}")

    # COMPARISON ANALYSIS
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80 + "\n")

    if 'classic' in results and 'dynamic' in results:
        # Performance comparison
        print("📊 PERFORMANCE METRICS:")
        print(f"  Classic Time: {results['classic']['total_time']:.2f}s")
        print(f"  Dynamic Time: {results['dynamic']['total_time']:.2f}s")

        time_diff = results['dynamic']['total_time'] - results['classic']['total_time']
        time_ratio = results['dynamic']['total_time'] / results['classic']['total_time']

        if time_diff > 0:
            print(f"  Dynamic was {time_diff:.2f}s slower ({time_ratio:.1f}x)")
        else:
            print(f"  Dynamic was {abs(time_diff):.2f}s faster ({time_ratio:.1f}x)")

        # Quality comparison
        print("\n📈 QUALITY METRICS:")
        if 'quality_scores' in results['classic']:
            classic_avg = sum(results['classic']['quality_scores'].values()) / len(results['classic']['quality_scores'])
            print(f"  Classic Average Quality: {classic_avg:.1f}/100")

        if 'quality_scores' in results['dynamic']:
            dynamic_avg = sum(results['dynamic']['quality_scores'].values()) / len(results['dynamic']['quality_scores'])
            print(f"  Dynamic Average Quality: {dynamic_avg:.1f}/100")

        if 'uniqueness_score' in results['dynamic']:
            print(f"  Dynamic Uniqueness Score: {results['dynamic']['uniqueness_score']:.2f}/100")

        # Document size comparison
        print("\n📄 DOCUMENT ANALYSIS:")
        if 'documents' in results['classic'] and 'documents' in results['dynamic']:
            for doc_name in results['classic']['documents']:
                if doc_name in results['dynamic']['documents']:
                    classic_len = len(results['classic']['documents'][doc_name])
                    dynamic_len = len(results['dynamic']['documents'][doc_name])
                    diff = dynamic_len - classic_len

                    print(f"\n  {doc_name}:")
                    print(f"    Classic: {classic_len:,} characters")
                    print(f"    Dynamic: {dynamic_len:,} characters")

                    if diff > 0:
                        print(f"    Dynamic is {diff:,} chars longer (+{(diff/classic_len*100):.1f}%)")
                    else:
                        print(f"    Dynamic is {abs(diff):,} chars shorter ({(diff/classic_len*100):.1f}%)")

        # Architecture characteristics
        print("\n🏗️ ARCHITECTURE CHARACTERISTICS:")
        print("\n  Classic:")
        print("    • Single orchestration agent")
        print("    • Sequential processing")
        print("    • Consistent style across documents")
        print("    • Faster execution")

        print("\n  Dynamic:")
        print("    • Multiple specialized agents")
        print("    • Parallel processing capability")
        print("    • Tailored approach per document")
        print("    • Higher uniqueness scores")

        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print("  • Use Classic for: Quick turnaround, standard RFPs, consistent formatting")
        print("  • Use Dynamic for: Complex RFPs, specialized requirements, differentiation")
        print("  • Consider A/B testing for optimal architecture selection per RFP type")

    else:
        print("⚠️ Could not complete comparison - one or both architectures failed")

    print("\n" + "="*80)
    print("COMPARISON TEST COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(test_architecture_comparison())