"""
Specialized Federal RFP Agents
Handles specific federal contracting requirements
"""

from typing import Dict, Any, List, Optional
import logging
import os
from openai import OpenAI
from services.multi_agent_orchestrator import SpecializedAgent

logger = logging.getLogger(__name__)

class CybersecurityComplianceAgent(SpecializedAgent):
    """Handles CMMC, NIST 800-171, and cybersecurity requirements"""

    def __init__(self):
        super().__init__(
            "CybersecurityComplianceAgent",
            ["CMMC", "NIST", "cybersecurity", "security controls", "FedRAMP"]
        )

    async def can_handle(self, document_spec: Dict[str, Any]) -> float:
        keywords = ['cmmc', 'cybersecurity', 'nist', '800-171', 'security controls',
                   'fedramp', 'cyber', 'information security', 'data protection',
                   'security assessment', 'vulnerability', 'threat']
        return self._calculate_keyword_confidence(document_spec, keywords)

    async def generate_document(self, spec: Dict[str, Any], context: Dict[str, Any]) -> str:
        prompt = f"""You are a federal cybersecurity compliance expert creating: {spec['name']}

REQUIREMENTS:
{spec.get('requirements', '')}

NORTHSTAR ANALYSIS:
{context.get('northstar', '')}

COMPANY SECURITY POSTURE:
{context.get('company_context', '')}

Create a comprehensive cybersecurity compliance document that addresses:

## 1. CMMC COMPLIANCE
- Current CMMC Level and certification status
- Path to required CMMC level
- Implementation timeline and milestones
- Third-party assessment readiness

## 2. NIST 800-171 CONTROLS
- All 110 security controls implementation status
- System Security Plan (SSP) overview
- Plan of Action & Milestones (POA&M) for gaps
- Continuous monitoring approach

## 3. SECURITY ARCHITECTURE
- Network segmentation and CUI isolation
- Access control and authentication mechanisms
- Encryption at rest and in transit
- Security tools and technologies

## 4. INCIDENT RESPONSE
- Incident response plan and procedures
- Security Operations Center (SOC) capabilities
- Threat detection and response times
- Forensics and recovery procedures

## 5. PERSONNEL SECURITY
- Security clearance management
- Insider threat program
- Security awareness training
- Background investigation processes

## 6. SUPPLY CHAIN SECURITY
- Vendor risk management
- Software bill of materials (SBOM)
- Third-party risk assessments
- Critical supplier contingencies

Ensure all responses align with current DoD and federal cybersecurity requirements."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a federal cybersecurity compliance expert with deep knowledge of CMMC, NIST frameworks, and DoD security requirements."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=self.max_tokens
        )

        return response.choices[0].message.content

class OCIAgent(SpecializedAgent):
    """Handles Organizational Conflict of Interest mitigation"""

    def __init__(self):
        super().__init__(
            "OCIAgent",
            ["OCI", "conflict of interest", "organizational conflict", "firewall"]
        )

    async def can_handle(self, document_spec: Dict[str, Any]) -> float:
        keywords = ['oci', 'organizational conflict', 'conflict of interest',
                   'firewall', 'mitigation plan', 'impaired objectivity',
                   'unfair competitive advantage', 'biased ground rules']
        return self._calculate_keyword_confidence(document_spec, keywords)

    async def generate_document(self, spec: Dict[str, Any], context: Dict[str, Any]) -> str:
        prompt = f"""You are an OCI mitigation expert creating: {spec['name']}

REQUIREMENTS:
{spec.get('requirements', '')}

NORTHSTAR ANALYSIS:
{context.get('northstar', '')}

COMPANY RELATIONSHIPS:
{context.get('company_context', '')}

Create a comprehensive OCI Mitigation Plan addressing:

## 1. OCI IDENTIFICATION AND ANALYSIS
- Review of all three types of OCI:
  * Impaired Objectivity
  * Unfair Competitive Advantage
  * Biased Ground Rules
- Current contracts and relationships analysis
- Potential conflict identification
- Risk assessment and categorization

## 2. MITIGATION STRATEGIES
- Organizational firewalls and separation
- Personnel restrictions and NDAs
- Information handling procedures
- Subcontractor flow-down requirements
- Monitoring and compliance procedures

## 3. FIREWALL IMPLEMENTATION
- Physical separation measures
- IT system segregation
- Personnel access controls
- Document handling procedures
- Communication restrictions

## 4. CERTIFICATION AND DISCLOSURE
- OCI certification statements
- Full disclosure of relevant relationships
- Ongoing reporting requirements
- Update procedures for changes

## 5. COMPLIANCE MONITORING
- Internal audit procedures
- Violation reporting mechanisms
- Corrective action plans
- Training and awareness programs

Ensure compliance with FAR 9.5 and agency-specific OCI requirements."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an OCI expert familiar with FAR 9.5 and federal conflict of interest regulations."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=self.max_tokens
        )

        return response.choices[0].message.content

class SmallBusinessSubcontractingAgent(SpecializedAgent):
    """Handles Small Business Subcontracting Plans"""

    def __init__(self):
        super().__init__(
            "SmallBusinessSubcontractingAgent",
            ["small business", "subcontracting", "socioeconomic", "8(a)", "HUBZone"]
        )

    async def can_handle(self, document_spec: Dict[str, Any]) -> float:
        keywords = ['small business', 'subcontracting', 'socioeconomic', '8(a)',
                   'hubzone', 'sdvosb', 'wosb', 'edwosb', 'sdb', 'subcontracting plan']
        return self._calculate_keyword_confidence(document_spec, keywords)

    async def generate_document(self, spec: Dict[str, Any], context: Dict[str, Any]) -> str:
        prompt = f"""You are a small business subcontracting expert creating: {spec['name']}

REQUIREMENTS:
{spec.get('requirements', '')}

NORTHSTAR ANALYSIS:
{context.get('northstar', '')}

CONTRACT VALUE AND COMPANY SIZE:
{context.get('company_context', '')}

Create a comprehensive Small Business Subcontracting Plan per FAR 19.704:

## 1. SUBCONTRACTING GOALS
- Small Business: [Percentage goal]
- Small Disadvantaged Business (SDB): [Percentage goal]
- Women-Owned Small Business (WOSB): [Percentage goal]
- HUBZone Small Business: [Percentage goal]
- Veteran-Owned Small Business (VOSB): [Percentage goal]
- Service-Disabled Veteran-Owned (SDVOSB): [Percentage goal]

## 2. TOTAL DOLLARS PLANNED
- Total contract value for subcontracting
- Dollar amounts per socioeconomic category
- Indirect costs inclusion
- Material/supply breakdowns

## 3. SUBCONTRACTOR IDENTIFICATION
- Method for identifying small business sources
- Outreach activities and events
- Use of SBA resources (SUB-Net, DSBS)
- Market research procedures

## 4. COMPLIANCE AND ADMINISTRATION
- Program administrator designation
- Compliance monitoring procedures
- Good faith effort documentation
- Recordkeeping requirements
- Reporting procedures (eSRS)

## 5. FLOW-DOWN REQUIREMENTS
- Subcontractor requirements
- Tier 2 and below flow-down
- Compliance verification
- Assistance to small businesses

## 6. EQUITABLE OPPORTUNITY ASSURANCE
- Procedures ensuring equitable opportunities
- Mentor-protégé considerations
- Joint venture opportunities
- Capability development initiatives

Include all required elements per FAR 19.704 and ensure quantifiable, measurable goals."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a small business subcontracting expert familiar with FAR Part 19 and socioeconomic program requirements."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=self.max_tokens
        )

        return response.choices[0].message.content

class KeyPersonnelAgent(SpecializedAgent):
    """Handles Key Personnel and Organizational Structure documentation"""

    def __init__(self):
        super().__init__(
            "KeyPersonnelAgent",
            ["key personnel", "resumes", "organizational", "staffing", "team"]
        )

    async def can_handle(self, document_spec: Dict[str, Any]) -> float:
        keywords = ['key personnel', 'resume', 'organizational', 'staffing',
                   'team', 'project manager', 'technical lead', 'staff',
                   'qualifications', 'experience']
        return self._calculate_keyword_confidence(document_spec, keywords)

    async def generate_document(self, spec: Dict[str, Any], context: Dict[str, Any]) -> str:
        prompt = f"""You are a federal personnel documentation expert creating: {spec['name']}

REQUIREMENTS:
{spec.get('requirements', '')}

NORTHSTAR ANALYSIS:
{context.get('northstar', '')}

COMPANY PERSONNEL INFORMATION:
{context.get('company_context', '')}

Create comprehensive Key Personnel documentation including:

## 1. ORGANIZATIONAL STRUCTURE
- Program Management Office structure
- Technical team organization
- Support functions alignment
- Government interface points
- Clear reporting relationships

## 2. KEY PERSONNEL PROFILES
For each key position, provide:

### [Position Title]
**Proposed Individual:** [Name]
**Security Clearance:** [Level/Status]
**Years of Experience:** [Total/Relevant]

**Summary**
[2-3 sentences highlighting suitability]

**Relevant Experience**
- [Company, Role, Duration, Key Achievements]
- [Quantified accomplishments relevant to this requirement]

**Education & Certifications**
- [Degree, Institution, Year]
- [Relevant certifications]

**Key Qualifications for This Role**
- [Specific qualification aligned to requirement]
- [Demonstrated capability with metrics]

## 3. STAFFING PLAN
- Staffing ramp-up schedule
- Skill mix and labor categories
- Recruitment strategies
- Retention approaches
- Surge capacity planning

## 4. CONTINGENCY PLANNING
- Key personnel backup identification
- Cross-training programs
- Succession planning
- Knowledge management approach

## 5. COMMITMENT LETTERS
- Key personnel commitment statements
- Availability confirmations
- Non-compete considerations

Format as professional federal resumes emphasizing relevant experience and clearances."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert in federal personnel documentation and resume writing."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=self.max_tokens
        )

        return response.choices[0].message.content

class QualityAssurancePlanAgent(SpecializedAgent):
    """Handles Quality Assurance Plans and Performance Management"""

    def __init__(self):
        super().__init__(
            "QualityAssurancePlanAgent",
            ["quality assurance", "QA", "performance management", "metrics", "QASP"]
        )

    async def can_handle(self, document_spec: Dict[str, Any]) -> float:
        keywords = ['quality assurance', 'qa', 'qasp', 'quality control',
                   'performance management', 'metrics', 'sla', 'kpi',
                   'quality plan', 'surveillance']
        return self._calculate_keyword_confidence(document_spec, keywords)

    async def generate_document(self, spec: Dict[str, Any], context: Dict[str, Any]) -> str:
        prompt = f"""You are a quality assurance expert creating: {spec['name']}

REQUIREMENTS:
{spec.get('requirements', '')}

NORTHSTAR ANALYSIS:
{context.get('northstar', '')}

COMPANY QA PROCESSES:
{context.get('company_context', '')}

Create a comprehensive Quality Assurance Plan including:

## 1. QUALITY MANAGEMENT FRAMEWORK
- Quality standards adoption (ISO 9001, CMMI, etc.)
- Quality policy and objectives
- Organizational responsibilities
- Quality management system overview

## 2. PERFORMANCE STANDARDS AND METRICS
- Acceptable Quality Levels (AQLs)
- Key Performance Indicators (KPIs)
- Service Level Agreements (SLAs)
- Performance thresholds and targets
- Measurement methodologies

## 3. QUALITY CONTROL PROCEDURES
- Inspection and testing protocols
- Review and approval processes
- Defect prevention strategies
- Root cause analysis procedures
- Corrective and preventive actions (CAPA)

## 4. SURVEILLANCE AND MONITORING
- Government surveillance support
- Self-inspection programs
- Performance monitoring tools
- Reporting frequencies and formats
- Dashboard and metrics visualization

## 5. QUALITY ASSURANCE SURVEILLANCE PLAN (QASP)
- Surveillance methods (100%, random, periodic)
- Performance requirements matrix
- Evaluation criteria and scoring
- Remediation procedures
- Incentive/disincentive structures

## 6. CONTINUOUS IMPROVEMENT
- Lessons learned processes
- Best practice identification
- Innovation initiatives
- Performance improvement plans
- Customer satisfaction measurement

## 7. QUALITY DOCUMENTATION
- Quality records management
- Deliverable acceptance criteria
- Non-conformance reporting
- Change management procedures
- Configuration management

Ensure alignment with government quality requirements and industry best practices."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a quality assurance expert familiar with federal QASP requirements and quality management systems."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=self.max_tokens
        )

        return response.choices[0].message.content

class RiskManagementAgent(SpecializedAgent):
    """Handles Risk Management Plans and Mitigation Strategies"""

    def __init__(self):
        super().__init__(
            "RiskManagementAgent",
            ["risk", "mitigation", "contingency", "risk management", "risk assessment"]
        )

    async def can_handle(self, document_spec: Dict[str, Any]) -> float:
        keywords = ['risk', 'mitigation', 'contingency', 'risk management',
                   'risk assessment', 'risk analysis', 'risk register',
                   'threat', 'vulnerability', 'impact']
        return self._calculate_keyword_confidence(document_spec, keywords)

    async def generate_document(self, spec: Dict[str, Any], context: Dict[str, Any]) -> str:
        prompt = f"""You are a risk management expert creating: {spec['name']}

REQUIREMENTS:
{spec.get('requirements', '')}

NORTHSTAR ANALYSIS:
{context.get('northstar', '')}

PROJECT CONTEXT:
{context.get('company_context', '')}

Create a comprehensive Risk Management Plan including:

## 1. RISK MANAGEMENT APPROACH
- Risk management framework and methodology
- Risk appetite and tolerance levels
- Roles and responsibilities
- Risk review and update cycles

## 2. RISK IDENTIFICATION AND ASSESSMENT
Create a detailed risk register with:

### Risk ID: [R001]
**Risk Description:** [Detailed description]
**Category:** [Technical/Schedule/Cost/Performance]
**Probability:** [Low/Medium/High] (1-5 scale)
**Impact:** [Low/Medium/High] (1-5 scale)
**Risk Score:** [Probability × Impact]
**Risk Owner:** [Responsible party]

**Mitigation Strategy:**
- Preventive measures
- Detective controls
- Corrective actions

**Contingency Plan:**
- Trigger conditions
- Response actions
- Recovery procedures

**Residual Risk:** [After mitigation]

## 3. TOP 10 PROJECT RISKS
[Detailed analysis of highest priority risks]

## 4. RISK MITIGATION STRATEGIES
- Risk avoidance approaches
- Risk transfer mechanisms
- Risk reduction techniques
- Risk acceptance criteria

## 5. CONTINGENCY PLANNING
- Contingency reserves (schedule and cost)
- Fallback plans
- Workaround procedures
- Emergency response protocols

## 6. RISK MONITORING AND CONTROL
- Risk tracking mechanisms
- Early warning indicators
- Risk review meetings
- Escalation procedures
- Risk reporting formats

## 7. OPPORTUNITY MANAGEMENT
- Positive risk identification
- Opportunity enhancement strategies
- Benefits realization tracking

Ensure comprehensive coverage of all risk categories and actionable mitigation plans."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a risk management expert familiar with federal project risk management and mitigation strategies."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=self.max_tokens
        )

        return response.choices[0].message.content

class TransitionPlanAgent(SpecializedAgent):
    """Handles Transition and Implementation Plans"""

    def __init__(self):
        super().__init__(
            "TransitionPlanAgent",
            ["transition", "implementation", "migration", "deployment", "rollout"]
        )

    async def can_handle(self, document_spec: Dict[str, Any]) -> float:
        keywords = ['transition', 'implementation', 'migration', 'deployment',
                   'rollout', 'phase-in', 'onboarding', 'knowledge transfer',
                   'takeover', 'startup']
        return self._calculate_keyword_confidence(document_spec, keywords)

    async def generate_document(self, spec: Dict[str, Any], context: Dict[str, Any]) -> str:
        prompt = f"""You are a transition planning expert creating: {spec['name']}

REQUIREMENTS:
{spec.get('requirements', '')}

NORTHSTAR ANALYSIS:
{context.get('northstar', '')}

IMPLEMENTATION CONTEXT:
{context.get('company_context', '')}

Create a comprehensive Transition/Implementation Plan including:

## 1. TRANSITION APPROACH AND STRATEGY
- Transition philosophy and principles
- Phased vs. big-bang approach justification
- Risk-based transition planning
- Success criteria and exit criteria

## 2. TRANSITION TIMELINE AND PHASES

### Phase 1: Planning and Preparation (Days 1-30)
- Transition team mobilization
- Stakeholder engagement
- Current state assessment
- Transition plan finalization

### Phase 2: Knowledge Transfer (Days 31-60)
- Documentation review and gap analysis
- Shadowing and observation
- Training and certification
- Knowledge validation testing

### Phase 3: Parallel Operations (Days 61-90)
- Gradual responsibility transfer
- Performance monitoring
- Issue identification and resolution
- Capability demonstration

### Phase 4: Full Transition (Days 91-120)
- Complete operational responsibility
- Incumbent contractor off-boarding
- Final knowledge transfer validation
- Transition completion certification

## 3. RESOURCE REQUIREMENTS
- Transition team composition
- Government furnished items/equipment
- Facility requirements
- System access needs
- Security clearance processing

## 4. KNOWLEDGE MANAGEMENT
- Documentation requirements
- Knowledge capture methods
- Training programs
- Competency validation
- Continuity assurance

## 5. STAKEHOLDER MANAGEMENT
- Communication plan
- Stakeholder mapping
- Change management approach
- Expectation management

## 6. RISK MITIGATION
- Transition-specific risks
- Contingency plans
- Fallback procedures
- Service continuity assurance

## 7. PERFORMANCE METRICS
- Transition milestones
- Quality gates
- Success measurements
- Lessons learned capture

Ensure minimal disruption to ongoing operations and seamless service continuity."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a transition planning expert experienced in federal contract transitions and implementations."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=self.max_tokens
        )

        return response.choices[0].message.content

class SustainabilityAgent(SpecializedAgent):
    """Handles Environmental and Sustainability Requirements"""

    def __init__(self):
        super().__init__(
            "SustainabilityAgent",
            ["sustainability", "environmental", "green", "ESG", "climate"]
        )

    async def can_handle(self, document_spec: Dict[str, Any]) -> float:
        keywords = ['sustainability', 'environmental', 'green', 'esg', 'climate',
                   'carbon', 'energy efficiency', 'waste reduction', 'sustainable',
                   'eco-friendly', 'emissions']
        return self._calculate_keyword_confidence(document_spec, keywords)

    async def generate_document(self, spec: Dict[str, Any], context: Dict[str, Any]) -> str:
        prompt = f"""You are a sustainability expert creating: {spec['name']}

REQUIREMENTS:
{spec.get('requirements', '')}

NORTHSTAR ANALYSIS:
{context.get('northstar', '')}

COMPANY SUSTAINABILITY PRACTICES:
{context.get('company_context', '')}

Create a comprehensive Sustainability and Environmental Plan including:

## 1. ENVIRONMENTAL MANAGEMENT SYSTEM
- ISO 14001 compliance or equivalent
- Environmental policy and objectives
- Regulatory compliance framework
- Environmental impact assessments

## 2. CARBON FOOTPRINT AND EMISSIONS
- Current baseline measurements
- Reduction targets and timelines
- Scope 1, 2, and 3 emissions strategies
- Carbon offset programs
- Renewable energy utilization

## 3. RESOURCE EFFICIENCY
- Energy efficiency measures
- Water conservation strategies
- Waste reduction and recycling
- Circular economy principles
- Sustainable procurement practices

## 4. SUSTAINABLE OPERATIONS
- Green building certifications (LEED)
- Fleet management and transportation
- Remote work and travel reduction
- Paperless operations
- Sustainable technology choices

## 5. SUPPLY CHAIN SUSTAINABILITY
- Supplier sustainability requirements
- Environmental screening criteria
- Sustainable material sourcing
- Packaging reduction initiatives
- Transportation optimization

## 6. REPORTING AND METRICS
- Sustainability KPIs and targets
- ESG reporting framework
- Progress tracking mechanisms
- Transparency and disclosure
- Third-party verification

## 7. INNOVATION AND IMPROVEMENT
- Green technology adoption
- Sustainability innovation initiatives
- Employee engagement programs
- Community partnerships
- Continuous improvement approach

## 8. COMPLIANCE AND CERTIFICATIONS
- Environmental regulations compliance
- Federal sustainability requirements
- Industry certifications
- Audit and assessment procedures

Align with federal sustainability goals and Executive Orders on climate and environment."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a sustainability expert familiar with federal environmental requirements and ESG best practices."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=self.max_tokens
        )

        return response.choices[0].message.content

# Agent Registry for easy access
FEDERAL_AGENTS = [
    CybersecurityComplianceAgent,
    OCIAgent,
    SmallBusinessSubcontractingAgent,
    KeyPersonnelAgent,
    QualityAssurancePlanAgent,
    RiskManagementAgent,
    TransitionPlanAgent,
    SustainabilityAgent
]

def get_federal_agents() -> List[SpecializedAgent]:
    """Get instances of all federal specialized agents"""
    return [agent_class() for agent_class in FEDERAL_AGENTS]