#!/usr/bin/env python3
"""High-Quality Dynamic Multi-Agent Test with Maximum Context"""

import asyncio
import httpx
from datetime import datetime
import json
from pathlib import Path
import time

async def test_dynamic_highquality():
    print("\n" + "="*80)
    print("HIGH-QUALITY DYNAMIC MULTI-AGENT ARCHITECTURE TEST")
    print("Prioritizing QUALITY over speed/cost - Maximum context deployment")
    print("="*80 + "\n")

    base_url = "http://localhost:8000"
    session_id = f"dynamic_highquality_{datetime.now().strftime('%H%M%S')}"

    # Read the comprehensive Northstar document
    northstar_path = Path("output/9a1e55199b1f4c1da83c14d548723c24/Northstar_Document.md")
    with open(northstar_path, 'r') as f:
        comprehensive_northstar = f.read()

    # Comprehensive test data with FULL context
    test_rfp_data = {
        "northstar_doc": comprehensive_northstar,  # Full 342-line comprehensive Northstar
        "rfp_documents": {
            "full_rfp": """
DEPARTMENT OF STATE
U.S. EMBASSY MANAMA, KINGDOM OF BAHRAIN

COMBINED SYNOPSIS/SOLICITATION
RFP No.: 19BA3025R0003
PR15563218

MULTIMEDIA MARKETING STRATEGY FOR AMERICA 250

SECTION 1 - THE SCHEDULE

The U.S. Embassy Manama requires comprehensive multimedia marketing services for America 250,
a year-long campaign celebrating America's 250th anniversary in the Kingdom of Bahrain.

STATEMENT OF WORK

IV. SCOPE OF WORK

IV.1 Branding
The Department of State has developed pre-existing "America 250" branding that must be used
with appropriate localization for the Kingdom of Bahrain market.

IV.1.1 Pre-Campaign Assessment & Branding
- Conduct focus groups with target demographics
- Plan regional marketing campaign for Bahrain in English and Arabic
- Identify and profile target audiences
- Develop cultural adaptation strategy

IV.2 Strategy
Design and implement a comprehensive campaign strategy for Bahrain to begin December 2025
and run through July 2026, encompassing all media channels and audience touchpoints.

IV.2.1 Execute Media Campaigns
- Develop and execute media campaigns and outreach strategies
- Ensure Embassy consultation and approval at all stages
- Implement data-driven channel selection

IV.2.2 Integrated Media Planning & Buying
- Traditional media (print, broadcast, outdoor)
- Digital and emerging media platforms
- Direct marketing vehicles
- Performance guarantees and make-good provisions
- Industry-standard measurement techniques

IV.2.3 Audience Understanding
- Deep audience insights and persona development
- Persuasive communications framework
- Cultural sensitivity and localization

IV.3 Budget Development
- Integrated campaign budget across multiple funding streams
- State Department allocations
- Public affairs budget integration
- Stakeholder contribution management

IV.4 Implementation

IV.4.1 Campaign Execution
- One calendar year implementation period
- September 2025 work commencement
- December 2025 public launch
- Data-driven channel selection
- USG approval processes

IV.4.2 Special Events
- Leverage Embassy and local cultural calendar
- REQUIRED: Multimedia display for Embassy's official Independence Day celebration
- Integration with local festivals and holidays

IV.5 Measurement & Reporting

IV.5.1 Performance Metrics
- Monthly status reports covering all performance aspects
- Standard advertising metrics:
  * Reach and frequency
  * CPM (Cost Per Thousand)
  * GRPs (Gross Rating Points)
  * Audience composition
  * Audited circulation
  * Ratings data
- Digital metrics:
  * Site visits and page views
  * Video views and engagement
  * Time on site
  * Click-through rates (CTR)
  * Link-offs and conversions
  * Lead generation
- Post-buy analyses against media plans
- Performance guarantees enforcement
- Make-good placement tracking
- Added-value opportunity capture
- Full data sharing with Embassy

IV.6 Contractor Personnel Requirements

IV.6.1 Public Sector Experience
- Account management team with integrated marketing program experience
- Public sector/public diplomacy advertising expertise
- Demonstrated outcomes and measurement capabilities

IV.6.2 Federal Contract Experience
- Experience with public sector contracts
- Research on U.S. perceptions in Bahrain/Gulf region

IV.6.3 Market Knowledge
- Deep knowledge of Bahraini and Gulf markets
- Cultural understanding and sensitivity

IV.6.4 Technical Capabilities
- Branding expertise
- Traditional and emerging media proficiency
- Interactive marketing
- Publication design
- Production and printing
- Social media management
- Public relations

IV.6.5 Account Management
- Drive branding/advertising/marketing strategy
- Campaign design and implementation leadership

IV.6.6 Government Experience
- U.S. government or foreign government support
- Branding/advertising/marketing for public sector

IV.6.7 Regional Presence
- REQUIRED: Established operational office in the Gulf
- Capability to provide services in Bahrain

IV.6.8 Digital Mastery
- In-house digital/social capabilities
- Online/digital campaign expertise
- Data analytics and optimization

IV.6.9 Partnership Experience
- Public-Private Partnership experience
- Major corporate client management

IV.7 Project Management Plan
Comprehensive PM plan including:
- Phase integration
- Resource allocation
- Database integrity
- Financial reporting
- Partner/subcontractor integration
- Embassy program coordination

IV.8 Transition of Services
- Physical and digital media transfer
- Database management services
- Website management services
- Administrative services handover

IV.9 Post-Award Conference
- Technical alignment
- Management coordination
- Security requirements
- Requirement clarification

DELIVERABLES:
- Branding materials
- Planning and research documents
- Campaign strategy documents
- Implementation materials
- Measurement reports
- Media plans
- Creative/multimedia production
- Administrative support documentation

LANGUAGES: English and Arabic (bilingual requirement)

SECTION 3 - INSTRUCTIONS TO OFFERORS

Evidence Requirements:
1. Named Project Manager fluent in English (written/spoken)
2. Established business with permanent address/telephone in Bahrain
3. Active SAM registration with UEID
4. Client list with past performance (last 3 years) with references
5. Personnel, equipment, and financial resource evidence
6. Local licenses/permits (DOSAR 652.242-73)
7. Vehicle descriptions for transport
8. Warehouse descriptions with safety features
9. Written Quality Assurance Plan
10. Audited financial statements (2022, 2023, 2024)
11. Insurance evidence including workers' compensation
12. Video samples from previous productions

EVALUATION: Lowest Price Technically Acceptable (LPTA)

SUBMISSION REQUIREMENTS:
- Email only: manamagsoprocurement@state.gov
- Subject: "Quotation Enclosed – RFP No.: 19BA3025R0003"
- PDF format only
- 10MB maximum per email
- English language
- Due: September 18, 2025, 4:00 PM Bahrain time
""",
            "compliance_matrix": """
COMPLIANCE REQUIREMENTS MATRIX

1. MANDATORY CERTIFICATIONS
- FAR 52.204-24: Representation Regarding Certain Telecommunications
- FAR 52.204-26: Covered Telecommunications Equipment or Services
- FAR 52.212-3: Offeror Representations and Certifications
- FAR 52.225-4: Buy American-Free Trade Agreements
- FAR 52.225-25: Prohibition on Contracting with Entities Relating to Iran

2. PROHIBITIONS
- FAR 52.204-27: ByteDance/TikTok prohibition
- FAR 52.204-23: Kaspersky prohibition
- FAR 52.204-25: Section 889 Huawei/ZTE prohibition

3. INSURANCE
- DOSAR 652.228-71: Defense Base Act (DBA) Insurance required
- Workers' compensation coverage
- General liability insurance

4. SECURITY
- DOSAR 652.239-71: Security Requirements for Unclassified IT
- Privacy training (FAR 52.224-3) if handling PII
- Contractor identification requirements

5. LABOR STANDARDS
- Service Contract Labor Standards
- Equal opportunity requirements
- Minimum wage (EO 14026)
- Paid sick leave (EO 13706)
- Child labor prohibitions
- Human trafficking prohibitions
"""
        },
        "company_context": """
Kamiwaza Technologies - Federal Contracting Excellence

CORPORATE OVERVIEW:
Kamiwaza is a leading technology integrator specializing in AI-powered marketing, multimedia
production, and public diplomacy campaigns for federal agencies. With 12+ years supporting
State Department, USAID, and DoD public affairs initiatives, we bring proven expertise in
culturally-sensitive communications across the Middle East and Gulf regions.

CORE COMPETENCIES:
- AI-Powered Marketing: Proprietary KamiInsights™ platform for audience analytics, sentiment
  analysis in Arabic/English, predictive modeling, and campaign optimization
- Multimedia Production: Full-service creative studio with 4K/8K capabilities, drone operations,
  AR/VR experiences, and interactive display technologies
- Public Diplomacy: 200+ successful campaigns across 45 countries, specializing in youth
  engagement, cultural exchange promotion, and soft power projection
- Digital Transformation: Cloud-native marketing platforms, real-time dashboards, API-driven
  integrations, and DevSecOps practices

GULF REGION PRESENCE:
- Partnership with Al-Mamlaka Creative (Bahrain's premier advertising agency)
- Office in Dubai Media City supporting regional operations
- 15 Arabic-speaking creative professionals
- Local production crews in Bahrain, Kuwait, UAE, and Saudi Arabia
- Existing relationships with Gulf media outlets and influencers

FEDERAL EXPERIENCE:
- GSA Schedule 541-4G (Advertising and Integrated Marketing)
- SEWP V contract vehicle
- Active Top Secret facility clearance
- ISO 9001:2015 and ISO 27001:2013 certified
- CMMI Level 3 for Services

KEY DIFFERENTIATORS:
- KamiInsights™ AI platform: 40% improvement in campaign ROI through predictive optimization
- 24/7 multilingual social media command center
- Proprietary content management system with Embassy approval workflows
- Real-time performance dashboards with automated reporting
- Cultural advisory board with regional experts
- In-house Arabic localization and transcreation team

PAST PERFORMANCE HIGHLIGHTS:
1. State Department "Discover America" Middle East Campaign ($18M, 2021-2024)
   - 24 countries, 8 languages, 350M impressions
   - 45% increase in favorable U.S. perceptions
   - Won PR Week Global Award for Public Sector Campaign

2. USAID Youth Entrepreneurship Initiative Gulf Region ($12M, 2020-2023)
   - Reached 2M+ youth across 6 Gulf countries
   - 15,000 program applications generated
   - 92% participant satisfaction rate

3. U.S. Embassy Kuwait National Day Celebration ($3.5M, 2022-2023)
   - Multimedia spectacular with 500K live attendees
   - 10M social media engagements
   - Zero security incidents

4. DoD Central Command Public Affairs Support ($8M, 2019-2022)
   - Crisis communications in 12 countries
   - 24/7 monitoring and response
   - 99.9% uptime on all platforms

FINANCIAL STRENGTH:
- Annual revenue: $145M (FY2023)
- D&B Rating: 5A1
- Bonding capacity: $50M
- Self-performing capability: 85%

QUALITY ASSURANCE:
- ISO 9001:2015 certified quality management
- Six Sigma Black Belt program managers
- Automated testing for all digital deliverables
- Third-party security audits quarterly
- Client satisfaction score: 4.8/5.0 average

SECURITY & COMPLIANCE:
- NIST 800-171 compliant
- FedRAMP Ready designation (in process for Moderate)
- SOC 2 Type II certified
- GDPR and CCPA compliant
- Cleared personnel available

INNOVATION LABS:
- AR/VR experience design studio
- AI/ML development center
- Social listening command center
- Multimedia production facility
- User experience testing lab

CONTACT:
Kamiwaza Technologies
8260 Greensboro Drive, Suite 600
McLean, VA 22102
Federal@kamiwaza.com
(703) 555-0200
DUNS: 123456789
CAGE: ABC123
UEID: KAMIWAZA123FED
""",
        "required_documents": [
            "Technical_Proposal.docx",
            "Cost_Proposal.xlsx",
            "Executive_Summary.pdf",
            "Past_Performance.pdf"
        ]
    }

    async with httpx.AsyncClient(timeout=3600) as client:  # 60 minute timeout for quality

        # Set Dynamic mode with quality parameters
        print("Configuring Dynamic Multi-Agent architecture for MAXIMUM QUALITY...")
        print("  • High context window utilization")
        print("  • Maximum reasoning effort")
        print("  • Comprehensive document generation")
        print("  • Quality over speed optimization\n")

        response = await client.post(
            f"{base_url}/api/architecture/config",
            json={
                "session_id": session_id,
                "mode": "dynamic",
                "quality_params": {
                    "reasoning_effort": "high",
                    "max_tokens": 16000,
                    "temperature": 0.7,
                    "verbosity": "high"
                }
            }
        )
        print(f"✓ Dynamic high-quality mode configured\n")

        # Process with Dynamic architecture
        print("Deploying specialized agents with comprehensive context...")
        print(f"  • Northstar document: {len(comprehensive_northstar):,} characters")
        print(f"  • RFP requirements: {len(test_rfp_data['rfp_documents']['full_rfp']):,} characters")
        print(f"  • Company context: {len(test_rfp_data['company_context']):,} characters")
        print(f"  • Total context: {len(comprehensive_northstar) + len(str(test_rfp_data['rfp_documents'])) + len(test_rfp_data['company_context']):,} characters")
        print("\nThis will take 30-60 minutes for maximum quality generation...\n")

        start_time = time.time()

        try:
            response = await client.post(
                f"{base_url}/api/rfps/process-with-architecture",
                json={
                    "notice_id": f"HIGHQUALITY_{datetime.now().strftime('%H%M%S')}",
                    "session_id": session_id,
                    **test_rfp_data
                }
            )

            elapsed_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()

                print(f"\n" + "="*80)
                print("✅ HIGH-QUALITY DYNAMIC PROCESSING COMPLETED!")
                print("="*80)

                print(f"\n📊 Performance Metrics:")
                print(f"  • Processing time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
                print(f"  • Documents generated: {result.get('documents_generated', 0)}")
                print(f"  • Architecture used: {result.get('architecture_used', 'Unknown')}")

                if 'uniqueness_score' in result:
                    print(f"  • Uniqueness Score: {result.get('uniqueness_score', 0):.2f}/100")

                # Agent assignments
                if 'agent_assignments' in result:
                    print(f"\n🤖 Specialized Agent Deployments:")
                    for doc, agent in result['agent_assignments'].items():
                        print(f"    • {agent} → {doc}")

                # Quality scores
                if 'quality_scores' in result:
                    print(f"\n⭐ Document Quality Scores:")
                    total_score = 0
                    for doc, score in result['quality_scores'].items():
                        print(f"    • {doc}: {score:.1f}/100")
                        total_score += score
                    avg_score = total_score / len(result['quality_scores'])
                    print(f"    ━━━━━━━━━━━━━━━━━━━━")
                    print(f"    • AVERAGE QUALITY: {avg_score:.1f}/100")

                # Document details
                if 'documents' in result:
                    output_dir = Path(f"output/highquality_{datetime.now().strftime('%H%M%S')}")
                    output_dir.mkdir(parents=True, exist_ok=True)

                    print(f"\n📄 Generated Documents:")
                    total_chars = 0
                    for doc_name, content in result['documents'].items():
                        file_path = output_dir / doc_name
                        with open(file_path, 'w') as f:
                            f.write(content)
                        doc_size = len(content)
                        total_chars += doc_size
                        print(f"    • {doc_name}: {doc_size:,} characters")

                    print(f"    ━━━━━━━━━━━━━━━━━━━━")
                    print(f"    • TOTAL OUTPUT: {total_chars:,} characters")

                    # Save full result
                    with open(output_dir / "result.json", 'w') as f:
                        json.dump(result, f, indent=2)

                    print(f"\n💾 All documents saved to: {output_dir}")

                    # Quality Analysis
                    print(f"\n📈 QUALITY ANALYSIS:")
                    print(f"  • Context utilization: {(len(comprehensive_northstar) + len(str(test_rfp_data['rfp_documents'])) + len(test_rfp_data['company_context'])) / 1000:.1f}K chars input")
                    print(f"  • Output density: {total_chars / 1000:.1f}K chars generated")
                    print(f"  • Generation rate: {total_chars / elapsed_time:.0f} chars/second")

                    if avg_score >= 90:
                        print(f"  • Quality tier: EXCELLENT (90+)")
                    elif avg_score >= 80:
                        print(f"  • Quality tier: VERY GOOD (80-89)")
                    elif avg_score >= 70:
                        print(f"  • Quality tier: GOOD (70-79)")
                    else:
                        print(f"  • Quality tier: NEEDS IMPROVEMENT (<70)")

                print("\n" + "="*80)
                print("HIGH-QUALITY TEST SUMMARY")
                print("="*80)
                print(f"✅ Successfully deployed Dynamic Multi-Agent architecture")
                print(f"✅ Processed {len(comprehensive_northstar):,} character Northstar")
                print(f"✅ Generated {result.get('documents_generated', 0)} compliant documents")
                if 'quality_scores' in result:
                    print(f"✅ Achieved {avg_score:.1f}/100 average quality score")
                print(f"✅ Completed in {elapsed_time/60:.1f} minutes")

            else:
                print(f"\n❌ Processing failed: {response.status_code}")
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
    print("HIGH-QUALITY DYNAMIC TEST COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(test_dynamic_highquality())