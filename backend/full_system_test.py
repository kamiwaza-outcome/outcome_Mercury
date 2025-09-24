#!/usr/bin/env python3
"""Full system test - Complete RFP processing with both architectures"""

import asyncio
import httpx
import json
from datetime import datetime
import time

async def full_system_test():
    """Run complete RFP processing through the system"""
    print("\n" + "="*80)
    print("FULL SYSTEM TEST - COMPLETE RFP PROCESSING")
    print("="*80 + "\n")

    base_url = "http://localhost:8000"

    # Comprehensive test data that mimics a real RFP
    test_rfp = {
        "notice_id": "FULL_TEST_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        "title": "Enterprise Cloud Migration and Modernization Services",
        "description": """
        The agency seeks a qualified contractor to provide comprehensive cloud migration
        and modernization services for legacy systems. This includes assessment, planning,
        migration, optimization, and ongoing support for critical infrastructure.

        Key Requirements:
        - Migrate 50+ legacy applications to cloud
        - Implement DevSecOps practices
        - Ensure FedRAMP High compliance
        - Provide 24/7 support and monitoring
        - Complete migration within 18 months
        """,
        "agency": "Department of Digital Transformation",
        "type": "Combined Synopsis/Solicitation",
        "posted_date": datetime.now().isoformat(),
        "response_deadline": "2025-12-31",
        "naics": "541512",
        "set_aside": "Small Business",
        "documents": {
            "requirements": """
            TECHNICAL REQUIREMENTS:
            1. Cloud Architecture and Migration
               - Design and implement multi-cloud architecture (AWS, Azure, GCP)
               - Migrate legacy applications with zero downtime
               - Implement containerization using Kubernetes
               - Establish CI/CD pipelines

            2. Security and Compliance
               - FedRAMP High compliance
               - Zero Trust Architecture implementation
               - Continuous security monitoring
               - Incident response capabilities

            3. Performance Requirements
               - 99.99% uptime SLA
               - Sub-second response times
               - Auto-scaling capabilities
               - Disaster recovery with RTO < 1 hour

            4. Team Requirements
               - Certified cloud architects (AWS/Azure/GCP)
               - Security clearance requirements (Secret)
               - On-site support in Washington DC
               - 24/7 SOC operations
            """,
            "evaluation_criteria": """
            EVALUATION CRITERIA:
            Factor 1: Technical Approach (40%)
            - Cloud migration methodology
            - Security architecture
            - Innovation and automation

            Factor 2: Past Performance (30%)
            - Similar federal projects
            - Cloud migration experience
            - Performance ratings

            Factor 3: Management Approach (20%)
            - Project management
            - Risk mitigation
            - Quality assurance

            Factor 4: Price (10%)
            - Total cost reasonableness
            - Cost breakdown structure
            """
        }
    }

    async with httpx.AsyncClient(timeout=300) as client:
        # TEST 1: Classic Architecture
        print("="*60)
        print("TEST 1: CLASSIC ARCHITECTURE - FULL RFP PROCESSING")
        print("="*60)

        # Set architecture to Classic
        print("\n1. Setting architecture to Classic mode...")
        config_response = await client.post(
            f"{base_url}/api/architecture/config",
            json={
                "session_id": "full_test_classic",
                "mode": "classic"
            }
        )

        if config_response.status_code == 200:
            print(f"✓ Architecture set to Classic mode")
        else:
            print(f"✗ Failed to set architecture: {config_response.text}")
            return

        # Process RFP with Classic architecture
        print("\n2. Processing RFP with Classic architecture...")
        start_time = time.time()

        rfp_request = {
            "notice_id": test_rfp["notice_id"] + "_CLASSIC",
            "session_id": "full_test_classic",
            "northstar_doc": """
# Northstar Document - Company Capabilities

## Company Overview
We are a leading cloud solutions provider with 15+ years of federal experience.
- CMMI Level 5 certified
- ISO 27001/27017/27018 certified
- FedRAMP authorized (High)
- 500+ cloud certified professionals

## Core Competencies

### Cloud Migration Excellence
- Migrated 200+ applications to cloud for federal agencies
- Zero-downtime migration methodology
- Automated migration tools and accelerators
- Multi-cloud expertise (AWS, Azure, GCP)

### Security & Compliance
- Zero Trust implementation experience
- Continuous ATO support
- 24/7 Security Operations Center
- Incident response team with < 15 min response time

### Technical Capabilities
- Kubernetes & containerization
- Infrastructure as Code (Terraform, CloudFormation)
- CI/CD pipeline automation
- AI/ML integration for optimization

### Past Performance
1. DOD Cloud Migration - $50M, 100+ applications, completed on time
2. VA Modernization - $30M, FedRAMP High implementation
3. DHS Security Operations - $25M, 24/7 SOC services

## Differentiators
- Proprietary migration tools reducing timeline by 40%
- Cleared staff readily available
- Existing federal contracts for quick deployment
- Innovation lab for emerging technologies
            """,
            "rfp_documents": test_rfp["documents"],
            "company_context": "Federal contractor with extensive cloud migration experience, cleared facility, existing federal contracts"
        }

        try:
            response = await client.post(
                f"{base_url}/api/rfps/process-with-architecture",
                json=rfp_request,
                timeout=300
            )

            if response.status_code == 200:
                classic_result = response.json()
                elapsed = time.time() - start_time

                print(f"\n✓ Classic processing completed in {elapsed:.2f} seconds")
                print(f"  Architecture: {classic_result.get('architecture_used')}")
                print(f"  Documents generated: {classic_result.get('documents_generated', 0)}")

                if 'documents' in classic_result:
                    print("\n  Generated documents:")
                    for doc_name, content in classic_result['documents'].items():
                        print(f"    - {doc_name}: {len(content)} characters")

                if 'quality_scores' in classic_result:
                    print("\n  Quality scores:")
                    for doc_name, score in classic_result['quality_scores'].items():
                        print(f"    - {doc_name}: {score:.2f}")

                if 'errors' in classic_result and classic_result['errors']:
                    print("\n  ⚠️ Errors encountered:")
                    for error in classic_result['errors']:
                        print(f"    - {error}")
            else:
                print(f"\n✗ Classic processing failed: {response.status_code}")
                print(f"  Error: {response.text[:500]}")

        except Exception as e:
            print(f"\n✗ Exception during Classic processing: {e}")

        # TEST 2: Dynamic Multi-Agent Architecture
        print("\n" + "="*60)
        print("TEST 2: DYNAMIC MULTI-AGENT - FULL RFP PROCESSING")
        print("="*60)

        # Set architecture to Dynamic
        print("\n1. Setting architecture to Dynamic mode...")
        config_response = await client.post(
            f"{base_url}/api/architecture/config",
            json={
                "session_id": "full_test_dynamic",
                "mode": "dynamic"
            }
        )

        if config_response.status_code == 200:
            print(f"✓ Architecture set to Dynamic mode")
        else:
            print(f"✗ Failed to set architecture: {config_response.text}")

        # Process RFP with Dynamic architecture
        print("\n2. Processing RFP with Dynamic multi-agent architecture...")
        start_time = time.time()

        rfp_request["notice_id"] = test_rfp["notice_id"] + "_DYNAMIC"
        rfp_request["session_id"] = "full_test_dynamic"

        try:
            response = await client.post(
                f"{base_url}/api/rfps/process-with-architecture",
                json=rfp_request,
                timeout=300
            )

            if response.status_code == 200:
                dynamic_result = response.json()
                elapsed = time.time() - start_time

                print(f"\n✓ Dynamic processing completed in {elapsed:.2f} seconds")
                print(f"  Architecture: {dynamic_result.get('architecture_used')}")
                print(f"  Documents generated: {dynamic_result.get('documents_generated', 0)}")
                print(f"  Uniqueness score: {dynamic_result.get('uniqueness_score', 0):.2f}")

                if 'agent_assignments' in dynamic_result:
                    print("\n  Agent assignments:")
                    agents_used = {}
                    for doc, agent in dynamic_result['agent_assignments'].items():
                        if agent not in agents_used:
                            agents_used[agent] = []
                        agents_used[agent].append(doc)

                    for agent, docs in agents_used.items():
                        print(f"    - {agent}: {', '.join(docs)}")

                if 'documents' in dynamic_result:
                    print("\n  Generated documents:")
                    for doc_name, content in dynamic_result['documents'].items():
                        print(f"    - {doc_name}: {len(content)} characters")

                if 'quality_scores' in dynamic_result:
                    print("\n  Quality scores:")
                    for doc_name, score in dynamic_result['quality_scores'].items():
                        print(f"    - {doc_name}: {score:.2f}")

                if 'errors' in dynamic_result and dynamic_result['errors']:
                    print("\n  ⚠️ Errors encountered:")
                    for error in dynamic_result['errors']:
                        print(f"    - {error}")

            else:
                print(f"\n✗ Dynamic processing failed: {response.status_code}")
                print(f"  Error: {response.text[:500]}")

        except Exception as e:
            print(f"\n✗ Exception during Dynamic processing: {e}")

        # TEST 3: Comparison Mode
        print("\n" + "="*60)
        print("TEST 3: COMPARISON MODE - BOTH ARCHITECTURES")
        print("="*60)

        print("\nRunning both architectures in parallel for comparison...")
        start_time = time.time()

        comparison_request = {
            "notice_id": test_rfp["notice_id"] + "_COMPARISON",
            "northstar_doc": rfp_request["northstar_doc"],
            "rfp_documents": rfp_request["rfp_documents"],
            "company_context": rfp_request["company_context"]
        }

        try:
            response = await client.post(
                f"{base_url}/api/rfps/process-comparison",
                json=comparison_request,
                timeout=600
            )

            if response.status_code == 200:
                comparison_result = response.json()
                elapsed = time.time() - start_time

                print(f"\n✓ Comparison completed in {elapsed:.2f} seconds")

                if 'classic' in comparison_result:
                    print("\n  Classic Results:")
                    print(f"    Processing time: {comparison_result['classic'].get('processing_time', 0):.2f}s")
                    print(f"    Documents: {len(comparison_result['classic'].get('documents', {}))}")
                    print(f"    Avg quality: {comparison_result['classic'].get('metrics', {}).get('average_quality', 0):.2f}")

                if 'dynamic' in comparison_result:
                    print("\n  Dynamic Results:")
                    print(f"    Processing time: {comparison_result['dynamic'].get('processing_time', 0):.2f}s")
                    print(f"    Documents: {len(comparison_result['dynamic'].get('documents', {}))}")
                    print(f"    Avg quality: {comparison_result['dynamic'].get('metrics', {}).get('average_quality', 0):.2f}")
                    print(f"    Uniqueness: {comparison_result['dynamic'].get('uniqueness_score', 0):.2f}")
                    print(f"    Agents used: {comparison_result['dynamic'].get('metrics', {}).get('agents_used', 0)}")

                if 'comparison' in comparison_result:
                    comp = comparison_result['comparison']
                    print("\n  Comparison Metrics:")
                    print(f"    Time improvement: {comp.get('time_improvement', 0):.1f}%")
                    print(f"    Quality improvement: {comp.get('quality_improvement', 0):.2f} points")

                    if comp.get('time_improvement', 0) > 0:
                        print(f"    ✓ Dynamic is {comp.get('time_improvement', 0):.1f}% faster")
                    else:
                        print(f"    ✓ Classic is {abs(comp.get('time_improvement', 0)):.1f}% faster")

            else:
                print(f"\n✗ Comparison failed: {response.status_code}")
                print(f"  Error: {response.text[:500]}")

        except Exception as e:
            print(f"\n✗ Exception during comparison: {e}")

        # Get final metrics
        print("\n" + "="*60)
        print("SYSTEM METRICS SUMMARY")
        print("="*60)

        try:
            metrics_response = await client.get(f"{base_url}/api/architecture/metrics")

            if metrics_response.status_code == 200:
                metrics = metrics_response.json()
                print(f"\nTotal RFPs processed: {metrics.get('total_processed', 0)}")
                print(f"Classic runs: {metrics.get('classic_count', 0)}")
                print(f"Dynamic runs: {metrics.get('dynamic_count', 0)}")

                if metrics.get('classic_count', 0) > 0:
                    print(f"\nClassic Performance:")
                    print(f"  Avg processing time: {metrics.get('classic_avg_time', 0):.2f}s")
                    print(f"  Avg quality score: {metrics.get('classic_avg_quality', 0):.2f}")

                if metrics.get('dynamic_count', 0) > 0:
                    print(f"\nDynamic Performance:")
                    print(f"  Avg processing time: {metrics.get('dynamic_avg_time', 0):.2f}s")
                    print(f"  Avg quality score: {metrics.get('dynamic_avg_quality', 0):.2f}")

        except Exception as e:
            print(f"\n✗ Failed to get metrics: {e}")

    print("\n" + "="*80)
    print("FULL SYSTEM TEST COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(full_system_test())