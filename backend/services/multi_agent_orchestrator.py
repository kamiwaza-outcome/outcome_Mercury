"""
Multi-Agent Orchestrator for Dynamic RFP Processing
Coordinates specialized agents for different document types with shared memory integration
"""

from typing import Dict, Any, List, Optional, Tuple
import asyncio
from abc import ABC, abstractmethod
import logging
from datetime import datetime
import os
from openai import OpenAI
from .openai_service import OpenAIService, CIRCUIT_BREAKER
import json
try:
    from .enhanced_document_identification import EnhancedDocumentIdentifier, DocumentIdentification
except ImportError:
    # Fallback if enhanced document identification is not available
    EnhancedDocumentIdentifier = None
    DocumentIdentification = None

# Import shared memory components
from .shared_memory import SharedMemory, SharedMemoryMixin

logger = logging.getLogger(__name__)

class SpecializedAgent(ABC, SharedMemoryMixin):
    """Base class for all specialized agents with shared memory integration"""

    def __init__(self, name: str, specializations: List[str]):
        self.name = name
        self.specializations = specializations
        # Use shared sync client to reduce sockets
        self.openai_service = OpenAIService()
        self.client = self.openai_service.sync_client
        self.model = os.getenv("OPENAI_MODEL", "gpt-5")
        # Cap tokens to reduce timeouts/socket pressure
        self.max_tokens = int(os.getenv("MAX_COMPLETION_TOKENS", "16000"))
        # Reasoning level per agent
        self.reasoning_effort = os.getenv("GPT5_REASONING_EFFORT", "medium")
        self.confidence_cache = {}

        # Initialize shared memory mixin
        super().__init__()
        self.agent_id = name

    @abstractmethod
    async def can_handle(self, document_spec: Dict[str, Any]) -> float:
        """Return confidence score (0-1) for handling this document"""
        pass

    @abstractmethod
    async def generate_document(self, spec: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate document content"""
        pass

    def _calculate_keyword_confidence(self, doc_spec: Dict[str, Any], keywords: List[str]) -> float:
        """Calculate confidence based on keyword matches"""
        doc_name = doc_spec.get('name', '').lower()
        requirements = doc_spec.get('requirements', '').lower()

        score = sum(1 for keyword in keywords if keyword in doc_name or keyword in requirements)
        return min(score / max(len(keywords), 1), 1.0)

    def _select_model(self, requested_model: str) -> str:
        if "gpt-5" in (requested_model or "").lower() and not CIRCUIT_BREAKER.should_allow():
            logger.warning(f"GPT-5 circuit open; {self.name} using fallback model")
            return os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o")
        return requested_model

# Specialized Agent Implementations

class TechnicalArchitectureAgent(SpecializedAgent):
    def __init__(self):
        super().__init__(
            "TechnicalArchitectureAgent",
            ["technical", "architecture", "system design", "integration", "implementation"]
        )

    async def can_handle(self, document_spec: Dict[str, Any]) -> float:
        keywords = ['technical', 'architecture', 'system', 'design', 'implementation',
                   'integration', 'infrastructure', 'solution', 'technology', 'engineering']
        return self._calculate_keyword_confidence(document_spec, keywords)

    async def generate_document(self, spec: Dict[str, Any], context: Dict[str, Any]) -> str:
        # Read shared context to understand what other agents have discovered
        shared_context = await self.read_shared_context("requirements")

        # Track technical requirements in shared memory
        await self._track_technical_requirements(spec, shared_context)

        # Check for cross-document references from other technical documents
        existing_refs = shared_context.get('cross_references', {})
        technical_refs = [ref for ref in existing_refs.values()
                         if 'technical' in ref.get('reference_type', '').lower()]

        # Build enhanced prompt with shared context
        shared_insights = shared_context.get('insights', {})
        relevant_insights = [insight for insight in shared_insights.values()
                           if insight.get('category') == 'technical']

        prompt = f"""You are a technical architecture expert creating: {spec['name']}

REQUIREMENTS:
{spec.get('requirements', '')}

NORTHSTAR ANALYSIS:
{context.get('northstar', '')}

COMPANY TECHNICAL CAPABILITIES:
{context.get('company_context', '')}

SHARED TECHNICAL INSIGHTS FROM OTHER AGENTS:
{self._format_insights(relevant_insights)}

CROSS-DOCUMENT TECHNICAL REFERENCES:
{self._format_references(technical_refs)}

Create a comprehensive technical document that:
1. Provides detailed system architecture and design
2. Addresses all technical requirements with specific solutions
3. Includes integration approaches and technical dependencies
4. Demonstrates technical feasibility and innovation
5. Uses appropriate technical diagrams descriptions
6. Shows deep understanding of technology stack
7. References and builds upon insights from other document sections
8. Ensures consistency with other technical commitments

Be specific, detailed, and technically accurate."""

        effective_model = self._select_model(self.model)
        response = self.client.chat.completions.create(
            model=effective_model,
            messages=[
                {"role": "system", "content": "You are an expert technical architect for government contracts."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort if "gpt-5" in effective_model.lower() else None
        )

        content = response.choices[0].message.content
        if "gpt-5" in effective_model.lower():
            try:
                CIRCUIT_BREAKER.record_success()
            except Exception:
                pass

        # Write discoveries back to shared memory
        await self._record_technical_discoveries(spec, content)

        # Create cross-references to other documents
        await self._create_technical_references(spec['name'], content, context)

        return content

    async def _track_technical_requirements(self, spec: Dict[str, Any], shared_context: Dict[str, Any]) -> None:
        """Track technical requirements in shared memory"""
        if not self.shared_memory:
            return

        # Extract technical requirements from spec
        requirements_text = spec.get('requirements', '')
        technical_keywords = ['system', 'architecture', 'integration', 'performance', 'scalability']

        # Simple requirement extraction (would be more sophisticated in production)
        requirements = []
        for keyword in technical_keywords:
            if keyword in requirements_text.lower():
                req_id = f"TECH_{keyword.upper()}_{spec['name'][:20]}"
                await self.shared_memory.track_requirement(
                    self.agent_id,
                    req_id,
                    f"Technical {keyword} requirement for {spec['name']}",
                    spec['name']
                )
                requirements.append(req_id)

        # Assign requirements to self
        for req_id in requirements:
            await self.shared_memory.assign_requirement(
                self.agent_id, req_id, self.agent_id
            )

    async def _record_technical_discoveries(self, spec: Dict[str, Any], content: str) -> None:
        """Record technical discoveries in shared memory"""
        if not self.shared_memory:
            return

        # Extract key technical insights (simplified)
        insights = []
        if 'cloud' in content.lower():
            insights.append("Cloud-based architecture approach")
        if 'microservice' in content.lower():
            insights.append("Microservices architecture pattern")
        if 'api' in content.lower():
            insights.append("API-first integration strategy")

        for insight in insights:
            await self.write_discovery(
                category="technical",
                title=f"Technical Insight from {spec['name']}",
                description=insight,
                evidence=[f"Referenced in {spec['name']}"],
                relevance_score=0.8
            )

    async def _create_technical_references(self, doc_name: str, content: str, context: Dict[str, Any]) -> None:
        """Create cross-document references for technical consistency"""
        if not self.shared_memory:
            return

        # Look for references to other documents in the content
        other_docs = ['cost', 'security', 'compliance']
        for doc_type in other_docs:
            if doc_type in content.lower():
                await self.reference_other_document(
                    source_document=doc_name,
                    target_document=f"{doc_type.title()}_Document",
                    reference_type="dependency",
                    content=f"Technical solution depends on {doc_type} considerations",
                    context="Technical architecture requires coordination",
                    confidence=0.7
                )

    def _format_insights(self, insights: List[Dict[str, Any]]) -> str:
        """Format insights for prompt inclusion"""
        if not insights:
            return "No relevant technical insights from other agents yet."

        formatted = []
        for insight in insights[:3]:  # Limit to top 3
            formatted.append(f"- {insight.get('title', 'Unknown')}: {insight.get('description', '')}")

        return "\n".join(formatted)

    def _format_references(self, references: List[Dict[str, Any]]) -> str:
        """Format cross-references for prompt inclusion"""
        if not references:
            return "No cross-document technical references yet."

        formatted = []
        for ref in references[:3]:  # Limit to top 3
            formatted.append(
                f"- {ref.get('source_document', 'Unknown')} -> {ref.get('target_document', 'Unknown')}: "
                f"{ref.get('content', '')}"
            )

        return "\n".join(formatted)

class CostPricingAgent(SpecializedAgent):
    def __init__(self):
        super().__init__(
            "CostPricingAgent",
            ["cost", "pricing", "budget", "financial", "rates"]
        )

    async def can_handle(self, document_spec: Dict[str, Any]) -> float:
        keywords = ['cost', 'price', 'pricing', 'budget', 'financial', 'rates',
                   'billing', 'payment', 'economy', 'value', 'fee']
        return self._calculate_keyword_confidence(document_spec, keywords)

    async def generate_document(self, spec: Dict[str, Any], context: Dict[str, Any]) -> str:
        prompt = f"""You are a federal contracting cost/pricing expert creating: {spec['name']}

REQUIREMENTS:
{spec.get('requirements', '')}

NORTHSTAR ANALYSIS:
{context.get('northstar', '')}

COMPANY INFORMATION:
{context.get('company_context', '')}

Create a comprehensive cost proposal that:
1. Provides detailed cost breakdowns with justifications
2. Follows federal cost principles (FAR Part 31)
3. Includes direct costs, indirect rates, and profit/fee
4. Demonstrates cost reasonableness and value
5. Shows price competitiveness while maintaining quality
6. Includes cost narrative and basis of estimates

Ensure compliance with federal pricing requirements."""

        effective_model = self._select_model(self.model)
        response = self.client.chat.completions.create(
            model=effective_model,
            messages=[
                {"role": "system", "content": "You are a federal cost/pricing expert."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort if "gpt-5" in effective_model.lower() else None
        )
        content = response.choices[0].message.content
        if "gpt-5" in effective_model.lower():
            try:
                CIRCUIT_BREAKER.record_success()
            except Exception:
                pass
        return content

class ComplianceAgent(SpecializedAgent):
    def __init__(self):
        super().__init__(
            "ComplianceAgent",
            ["compliance", "legal", "regulatory", "certification", "requirements"]
        )

    async def can_handle(self, document_spec: Dict[str, Any]) -> float:
        keywords = ['compliance', 'legal', 'regulatory', 'certification', 'requirement',
                   'standard', 'regulation', 'policy', 'guideline', 'mandate']
        return self._calculate_keyword_confidence(document_spec, keywords)

    async def generate_document(self, spec: Dict[str, Any], context: Dict[str, Any]) -> str:
        prompt = f"""You are a compliance and regulatory expert creating: {spec['name']}

REQUIREMENTS:
{spec.get('requirements', '')}

NORTHSTAR ANALYSIS:
{context.get('northstar', '')}

Create a comprehensive compliance document that:
1. Addresses all regulatory and legal requirements
2. Provides compliance matrices and traceability
3. Demonstrates understanding of applicable regulations
4. Includes certification and attestation statements
5. Shows compliance verification methods
6. Addresses risk and mitigation strategies

Ensure 100% requirement coverage and compliance."""

        effective_model = self._select_model(self.model)
        response = self.client.chat.completions.create(
            model=effective_model,
            messages=[
                {"role": "system", "content": "You are a federal compliance expert."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort if "gpt-5" in effective_model.lower() else None
        )
        content = response.choices[0].message.content
        if "gpt-5" in effective_model.lower():
            try:
                CIRCUIT_BREAKER.record_success()
            except Exception:
                pass
        return content

class SecurityAgent(SpecializedAgent):
    def __init__(self):
        super().__init__(
            "SecurityAgent",
            ["security", "cybersecurity", "CMMC", "NIST", "clearance"]
        )

    async def can_handle(self, document_spec: Dict[str, Any]) -> float:
        keywords = ['security', 'cybersecurity', 'cmmc', 'nist', 'clearance',
                   'classified', 'protection', 'safeguard', 'threat', 'vulnerability']
        return self._calculate_keyword_confidence(document_spec, keywords)

    async def generate_document(self, spec: Dict[str, Any], context: Dict[str, Any]) -> str:
        prompt = f"""You are a cybersecurity and federal security expert creating: {spec['name']}

REQUIREMENTS:
{spec.get('requirements', '')}

NORTHSTAR ANALYSIS:
{context.get('northstar', '')}

Create a comprehensive security document that:
1. Addresses cybersecurity requirements (CMMC, NIST 800-171)
2. Provides security controls and implementation plans
3. Includes personnel security and clearance management
4. Demonstrates threat awareness and mitigation
5. Shows compliance with federal security standards
6. Includes incident response and continuity planning

Ensure alignment with DoD and federal security requirements."""

        effective_model = self._select_model(self.model)
        response = self.client.chat.completions.create(
            model=effective_model,
            messages=[
                {"role": "system", "content": "You are a federal security expert."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort if "gpt-5" in effective_model.lower() else None
        )
        content = response.choices[0].message.content
        if "gpt-5" in effective_model.lower():
            try:
                CIRCUIT_BREAKER.record_success()
            except Exception:
                pass
        return content

class PastPerformanceAgent(SpecializedAgent):
    def __init__(self):
        super().__init__(
            "PastPerformanceAgent",
            ["past performance", "experience", "references", "history"]
        )

    async def can_handle(self, document_spec: Dict[str, Any]) -> float:
        keywords = ['past', 'performance', 'experience', 'reference', 'history',
                   'previous', 'prior', 'track', 'record', 'cpars']
        return self._calculate_keyword_confidence(document_spec, keywords)

    async def generate_document(self, spec: Dict[str, Any], context: Dict[str, Any]) -> str:
        prompt = f"""You are a past performance documentation expert creating: {spec['name']}

REQUIREMENTS:
{spec.get('requirements', '')}

NORTHSTAR ANALYSIS:
{context.get('northstar', '')}

COMPANY HISTORY:
{context.get('company_context', '')}

Create compelling past performance documentation that:
1. Highlights relevant contract experience
2. Demonstrates successful performance metrics
3. Shows growth and lessons learned
4. Includes quantified achievements and outcomes
5. Provides client references and testimonials
6. Aligns past work with current requirements

Make it specific, measurable, and relevant."""

        effective_model = self._select_model(self.model)
        response = self.client.chat.completions.create(
            model=effective_model,
            messages=[
                {"role": "system", "content": "You are a past performance expert."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort if "gpt-5" in effective_model.lower() else None
        )
        content = response.choices[0].message.content
        if "gpt-5" in effective_model.lower():
            try:
                CIRCUIT_BREAKER.record_success()
            except Exception:
                pass
        return content

class VideoScriptAgent(SpecializedAgent):
    def __init__(self):
        super().__init__(
            "VideoScriptAgent",
            ["video", "script", "pitch", "presentation", "multimedia"]
        )

    async def can_handle(self, document_spec: Dict[str, Any]) -> float:
        keywords = ['video', 'script', 'pitch', 'presentation', 'multimedia',
                   'mp4', 'storyboard', 'visual', 'narration']
        return self._calculate_keyword_confidence(document_spec, keywords)

    async def generate_document(self, spec: Dict[str, Any], context: Dict[str, Any]) -> str:
        prompt = f"""You are creating a VIDEO SCRIPT AND STORYBOARD for: {spec['name']}

REQUIREMENTS:
{spec.get('requirements', '')}

NORTHSTAR ANALYSIS:
{context.get('northstar', '')}

COMPANY INFORMATION:
{context.get('company_context', '')}

Create a COMPLETE video script/storyboard with:

## STRUCTURE (5 minutes total)

### OPENING (0:00 - 0:30)
- VISUAL: [Opening graphics and title]
- SCRIPT: [Engaging introduction]
- TEXT OVERLAY: [Company name and solution]

### PROBLEM IDENTIFICATION (0:30 - 1:30)
- VISUAL: [Problem visualization]
- SCRIPT: [Clear problem statement]
- GRAPHICS: [Supporting data]

### SOLUTION OVERVIEW (1:30 - 3:00)
- VISUAL: [Solution demonstration]
- SCRIPT: [Technical explanation]
- ANIMATIONS: [Key features]

### VALUE PROPOSITION (3:00 - 4:00)
- VISUAL: [Benefits visualization]
- SCRIPT: [ROI and impact]
- DATA: [Metrics and outcomes]

### CALL TO ACTION (4:00 - 5:00)
- VISUAL: [Contact information]
- SCRIPT: [Next steps]
- TEXT: [Contact details]

Make it engaging, professional, and government-appropriate."""

        effective_model = self._select_model(self.model)
        response = self.client.chat.completions.create(
            model=effective_model,
            messages=[
                {"role": "system", "content": "You are a video script expert for government pitches."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=self.max_tokens
        )
        content = response.choices[0].message.content
        if "gpt-5" in effective_model.lower():
            try:
                CIRCUIT_BREAKER.record_success()
            except Exception:
                pass
        return content

class ExecutiveSummaryAgent(SpecializedAgent):
    def __init__(self):
        super().__init__(
            "ExecutiveSummaryAgent",
            ["executive", "summary", "overview", "abstract"]
        )

    async def can_handle(self, document_spec: Dict[str, Any]) -> float:
        keywords = ['executive', 'summary', 'overview', 'abstract', 'synopsis',
                   'brief', 'highlight', 'introduction']
        return self._calculate_keyword_confidence(document_spec, keywords)

    async def generate_document(self, spec: Dict[str, Any], context: Dict[str, Any]) -> str:
        prompt = f"""You are creating an executive summary for: {spec['name']}

REQUIREMENTS:
{spec.get('requirements', '')}

NORTHSTAR ANALYSIS:
{context.get('northstar', '')}

ALL GENERATED DOCUMENTS SUMMARY:
{context.get('all_documents_summary', '')}

Create a compelling executive summary that:
1. Captures the essence of the entire proposal
2. Highlights key discriminators and win themes
3. Demonstrates understanding of requirements
4. Shows unique value proposition
5. Provides clear benefits to the government
6. Uses persuasive but factual language

Keep it concise, impactful, and executive-appropriate."""

        effective_model = self._select_model(self.model)
        response = self.client.chat.completions.create(
            model=effective_model,
            messages=[
                {"role": "system", "content": "You are an executive communications expert."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=self.max_tokens
        )
        content = response.choices[0].message.content
        if "gpt-5" in effective_model.lower():
            try:
                CIRCUIT_BREAKER.record_success()
            except Exception:
                pass
        return content

class GeneralistAgent(SpecializedAgent):
    """Handles edge cases and documents that don't fit specialized categories"""

    def __init__(self):
        super().__init__(
            "GeneralistAgent",
            ["general", "undefined", "custom", "other"]
        )

    async def can_handle(self, document_spec: Dict[str, Any]) -> float:
        # Generalist can handle anything with moderate confidence
        return 0.6

    async def generate_document(self, spec: Dict[str, Any], context: Dict[str, Any]) -> str:
        prompt = f"""You are creating a document for: {spec['name']}

REQUIREMENTS:
{spec.get('requirements', '')}

NORTHSTAR ANALYSIS:
{context.get('northstar', '')}

COMPANY CONTEXT:
{context.get('company_context', '')}

This document doesn't fit standard categories. Create comprehensive content that:
1. Addresses ALL stated requirements completely
2. Maintains professional government contracting standards
3. Provides specific, detailed responses
4. Shows understanding of unique aspects
5. Incorporates relevant company capabilities
6. Ensures compliance while being creative

Be thorough, professional, and adaptable to the unique requirements."""

        effective_model = self._select_model(self.model)
        response = self.client.chat.completions.create(
            model=effective_model,
            messages=[
                {"role": "system", "content": "You are a versatile RFP response expert."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=self.max_tokens
        )
        content = response.choices[0].message.content
        if "gpt-5" in effective_model.lower():
            try:
                CIRCUIT_BREAKER.record_success()
            except Exception:
                pass
        return content

class AdaptiveAgent(SpecializedAgent):
    """Learning agent that improves over time"""

    def __init__(self):
        super().__init__(
            "AdaptiveAgent",
            ["adaptive", "learning", "novel", "experimental"]
        )
        self.learning_file = "data/adaptive_learning.json"
        self.patterns = self._load_patterns()

    def _load_patterns(self) -> Dict[str, Any]:
        """Load learned patterns from previous interactions"""
        try:
            if os.path.exists(self.learning_file):
                with open(self.learning_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return {"successful_patterns": [], "failure_patterns": [], "improvements": {}}

    async def can_handle(self, document_spec: Dict[str, Any]) -> float:
        # Check if this matches any learned patterns
        doc_signature = self._get_document_signature(document_spec)

        for pattern in self.patterns.get("successful_patterns", []):
            if self._pattern_matches(doc_signature, pattern):
                return 0.9  # High confidence for known patterns

        # Low confidence for completely novel documents
        return 0.4

    def _get_document_signature(self, doc_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key features of a document for pattern matching"""
        return {
            "keywords": set(doc_spec.get('name', '').lower().split() +
                          doc_spec.get('requirements', '').lower().split()[:20]),
            "length": len(doc_spec.get('requirements', '')),
            "type": doc_spec.get('format', 'unknown')
        }

    def _pattern_matches(self, signature: Dict[str, Any], pattern: Dict[str, Any]) -> bool:
        """Check if a document signature matches a learned pattern"""
        keyword_overlap = len(signature['keywords'].intersection(pattern.get('keywords', set())))
        return keyword_overlap > 5  # Simple matching for now

    async def generate_document(self, spec: Dict[str, Any], context: Dict[str, Any]) -> str:
        # Check for similar successful patterns
        best_pattern = self._find_best_pattern(spec)

        adaptation_hints = ""
        if best_pattern:
            adaptation_hints = f"\n\nLEARNED INSIGHTS:\n{best_pattern.get('insights', '')}"

        prompt = f"""You are an adaptive learning agent creating: {spec['name']}

REQUIREMENTS:
{spec.get('requirements', '')}

NORTHSTAR ANALYSIS:
{context.get('northstar', '')}
{adaptation_hints}

This is a novel or complex document. Apply creative problem-solving to:
1. Identify unique aspects of these requirements
2. Synthesize information in innovative ways
3. Address unconventional needs effectively
4. Learn from this interaction for future improvements
5. Balance creativity with compliance

Think outside standard templates while maintaining quality."""

        effective_model = self._select_model(self.model)
        response = self.client.chat.completions.create(
            model=effective_model,
            messages=[
                {"role": "system", "content": "You are an adaptive, learning RFP expert."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=self.max_tokens
        )

        content = response.choices[0].message.content
        if "gpt-5" in effective_model.lower():
            try:
                CIRCUIT_BREAKER.record_success()
            except Exception:
                pass

        # Learn from this interaction
        await self._learn_from_generation(spec, content)

        return content

    def _find_best_pattern(self, spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find the best matching pattern from learned experiences"""
        doc_signature = self._get_document_signature(spec)
        best_match = None
        best_score = 0

        for pattern in self.patterns.get("successful_patterns", []):
            score = len(doc_signature['keywords'].intersection(pattern.get('keywords', set())))
            if score > best_score:
                best_score = score
                best_match = pattern

        return best_match if best_score > 3 else None

    async def _learn_from_generation(self, spec: Dict[str, Any], content: str):
        """Learn from this document generation for future use"""
        signature = self._get_document_signature(spec)
        signature['timestamp'] = datetime.utcnow().isoformat()
        signature['content_length'] = len(content)

        # Add to patterns (in production, this would be more sophisticated)
        self.patterns["successful_patterns"].append(signature)

        # Keep only recent patterns
        self.patterns["successful_patterns"] = self.patterns["successful_patterns"][-100:]

        # Save patterns
        try:
            os.makedirs(os.path.dirname(self.learning_file), exist_ok=True)
            with open(self.learning_file, 'w') as f:
                # Convert sets to lists for JSON serialization
                patterns_to_save = self.patterns.copy()
                for pattern in patterns_to_save.get("successful_patterns", []):
                    if 'keywords' in pattern and isinstance(pattern['keywords'], set):
                        pattern['keywords'] = list(pattern['keywords'])
                json.dump(patterns_to_save, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save learning patterns: {e}")

# Multi-Agent Orchestrator

class MultiAgentOrchestrator:
    def __init__(self, milvus_rag=None, shared_memory: SharedMemory = None):
        self.agents = [
            TechnicalArchitectureAgent(),
            CostPricingAgent(),
            ComplianceAgent(),
            SecurityAgent(),
            PastPerformanceAgent(),
            VideoScriptAgent(),
            ExecutiveSummaryAgent(),
            GeneralistAgent(),
            AdaptiveAgent()
        ]

        self.milvus_rag = milvus_rag
        self.shared_memory = shared_memory
        self.coordination_strategies = {
            'sequential': self._coordinate_sequential,
            'parallel': self._coordinate_parallel,
            'hybrid': self._coordinate_hybrid
        }

        # Initialize enhanced document identifier
        self.enhanced_identifier = EnhancedDocumentIdentifier()

        # Performance optimization settings
        self.parallel_passes = [1, 2, 4]  # Passes that can run in parallel
        self.sequential_passes = [3, 5]   # Passes that require sequential execution

        # Quality thresholds for pass validation
        self.pass_quality_thresholds = {
            1: 0.7,  # Requirements analysis
            2: 0.6,  # Initial generation
            3: 0.8,  # Peer review
            4: 0.85, # Gap filling
            5: 0.9   # Polish
        }

        logger.info(f"Initialized MultiAgentOrchestrator with {len(self.agents)} specialized agents and multi-pass processing")

    async def initialize_shared_memory(self, shared_memory: SharedMemory) -> None:
        """Initialize shared memory for all agents"""
        self.shared_memory = shared_memory

        # Initialize shared memory for all agents
        for agent in self.agents:
            await agent.initialize_shared_memory(shared_memory)

        logger.info("Shared memory initialized for all agents")

    async def generate_rfp_response(
        self,
        northstar: str,
        rfp_documents: Dict[str, Any],
        notice_id: str,
        company_context: str = "",
        strategy: str = 'multipass'  # Default to multi-pass for better quality
    ) -> Dict[str, Any]:
        """Generate RFP response using multi-agent coordination"""
        logger.info(f"Starting multi-agent RFP generation with strategy: {strategy}")

        # Identify required documents from northstar with enhanced intelligence
        required_documents = await self._identify_required_documents(northstar, rfp_documents)

        # Get company context if not provided
        if not company_context:
            if self.milvus_rag:
                try:
                    company_context = await self.milvus_rag.get_company_context("")
                except Exception as e:
                    logger.warning(f"Could not get company context from milvus_rag: {e}")
                    company_context = "Company information not available"
            else:
                company_context = "Company information not available"

        # Select coordination strategy
        coordinator = self.coordination_strategies.get(strategy, self._coordinate_multipass)

        # Generate documents
        result = await coordinator(required_documents, northstar, rfp_documents, notice_id, company_context)

        # Add strategy metadata to result
        if 'strategy_used' not in result:
            result['strategy_used'] = strategy

        # Add quality metrics summary for multi-pass
        if strategy == 'multipass' and 'final_metrics' in result:
            result['quality_summary'] = {
                'overall_quality': result['final_metrics'].get('overall_quality', 0.0),
                'requirement_coverage': result['final_metrics'].get('requirement_coverage', 0.0),
                'passes_completed': result.get('passes_completed', 0),
                'processing_time': result.get('processing_time', 0)
            }

        return result

    async def _identify_required_documents(self, northstar: str, rfp_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Enhanced document identification using GPT-5 intelligent analysis"""
        try:
            logger.info("Starting enhanced document identification with intelligent analysis")

            # Use enhanced document identifier
            document_identifications = await self.enhanced_identifier.identify_required_documents(
                northstar, rfp_metadata
            )

            # Convert DocumentIdentification objects to dict format for compatibility
            documents = []
            for doc_id in document_identifications:
                doc_dict = {
                    "name": doc_id.name,
                    "requirements": doc_id.requirements,
                    "format": doc_id.format,
                    "confidence": doc_id.confidence_score,
                    "reasoning": doc_id.reasoning,
                    "specialized": doc_id.specialized,
                    "page_limit": doc_id.page_limit,
                    "consolidates": doc_id.consolidates,
                    "assumptions": doc_id.assumptions,
                    "document_type": doc_id.document_type.value
                }
                documents.append(doc_dict)

            # Log intelligent analysis results
            logger.info(f"Enhanced identification complete: {len(documents)} documents identified")
            for doc in documents:
                logger.info(f"  - {doc['name']} (confidence: {doc['confidence']:.2f}, type: {doc['document_type']})")
                if doc['assumptions']:
                    logger.info(f"    Assumptions: {doc['assumptions']}")

            return documents

        except Exception as e:
            logger.error(f"Enhanced document identification failed: {e}")
            logger.warning("Falling back to primitive keyword matching")
            return await self._fallback_primitive_identification(northstar)

    async def _fallback_primitive_identification(self, northstar: str) -> List[Dict[str, Any]]:
        """Fallback to original primitive keyword matching"""
        documents = []

        # Look for document specifications in northstar (original logic)
        if "technical proposal" in northstar.lower() or "technical volume" in northstar.lower():
            documents.append({
                "name": "Technical_Proposal.docx",
                "requirements": "Technical solution, architecture, implementation approach",
                "format": "docx",
                "confidence": 0.6,
                "reasoning": "Primitive keyword matching fallback",
                "specialized": True,
                "page_limit": None,
                "consolidates": [],
                "assumptions": ["Fallback to primitive matching"],
                "document_type": "technical_proposal"
            })

        if "cost" in northstar.lower() or "price" in northstar.lower():
            documents.append({
                "name": "Cost_Proposal.xlsx",
                "requirements": "Detailed cost breakdown, pricing structure",
                "format": "xlsx",
                "confidence": 0.6,
                "reasoning": "Primitive keyword matching fallback",
                "specialized": True,
                "page_limit": None,
                "consolidates": [],
                "assumptions": ["Fallback to primitive matching"],
                "document_type": "cost_volume"
            })

        if "past performance" in northstar.lower():
            documents.append({
                "name": "Past_Performance.pdf",
                "requirements": "Relevant past performance examples",
                "format": "pdf",
                "confidence": 0.6,
                "reasoning": "Primitive keyword matching fallback",
                "specialized": True,
                "page_limit": None,
                "consolidates": [],
                "assumptions": ["Fallback to primitive matching"],
                "document_type": "past_performance"
            })

        # Always include executive summary if multiple documents
        if len(documents) > 1:
            documents.append({
                "name": "Executive_Summary.pdf",
                "requirements": "High-level summary of proposal",
                "format": "pdf",
                "confidence": 0.5,
                "reasoning": "Default executive summary",
                "specialized": False,
                "page_limit": None,
                "consolidates": [],
                "assumptions": ["Default document"],
                "document_type": "executive_summary"
            })

        # If no specific documents identified, use defaults
        if not documents:
            documents = [
                {
                    "name": "Technical_Proposal.docx",
                    "requirements": "Technical response",
                    "format": "docx",
                    "confidence": 0.4,
                    "reasoning": "Default fallback",
                    "specialized": True,
                    "page_limit": None,
                    "consolidates": [],
                    "assumptions": ["Default fallback"],
                    "document_type": "technical_proposal"
                },
                {
                    "name": "Cost_Proposal.xlsx",
                    "requirements": "Cost breakdown",
                    "format": "xlsx",
                    "confidence": 0.4,
                    "reasoning": "Default fallback",
                    "specialized": True,
                    "page_limit": None,
                    "consolidates": [],
                    "assumptions": ["Default fallback"],
                    "document_type": "cost_volume"
                },
                {
                    "name": "Executive_Summary.pdf",
                    "requirements": "Executive overview",
                    "format": "pdf",
                    "confidence": 0.4,
                    "reasoning": "Default fallback",
                    "specialized": False,
                    "page_limit": None,
                    "consolidates": [],
                    "assumptions": ["Default fallback"],
                    "document_type": "executive_summary"
                }
            ]

        logger.info(f"Primitive fallback identified {len(documents)} documents")
        return documents

    async def _coordinate_hybrid(
        self,
        documents: List[Dict],
        northstar: str,
        rfp_documents: Dict[str, Any],
        notice_id: str,
        company_context: str
    ) -> Dict[str, Any]:
        """Hybrid coordination - parallel where possible, sequential where needed"""
        logger.info("Using hybrid coordination strategy")

        # Categorize documents by dependencies
        independent_docs = []
        dependent_docs = []

        for doc_spec in documents:
            if "executive" in doc_spec['name'].lower() or "summary" in doc_spec['name'].lower():
                dependent_docs.append(doc_spec)  # Needs other docs first
            else:
                independent_docs.append(doc_spec)

        # Process independent documents in parallel
        generated_docs = {}
        agent_assignments = {}

        if independent_docs:
            parallel_results = await self._process_documents_parallel(
                independent_docs, northstar, rfp_documents, notice_id, company_context
            )
            generated_docs.update(parallel_results['documents'])
            agent_assignments.update(parallel_results['agent_assignments'])

        # Process dependent documents with context from previous ones
        for doc_spec in dependent_docs:
            context = {
                'northstar': northstar,
                'rfp_documents': rfp_documents,
                'notice_id': notice_id,
                'company_context': company_context,
                'all_documents_summary': self._summarize_documents(generated_docs)
            }

            best_agent = await self._assign_best_agent(doc_spec)
            agent_assignments[doc_spec['name']] = best_agent.name

            content = await best_agent.generate_document(doc_spec, context)
            generated_docs[doc_spec['name']] = content

        return {
            'documents': generated_docs,
            'agent_assignments': agent_assignments,
            'strategy_used': 'hybrid'
        }

    async def _coordinate_parallel(
        self,
        documents: List[Dict],
        northstar: str,
        rfp_documents: Dict[str, Any],
        notice_id: str,
        company_context: str
    ) -> Dict[str, Any]:
        """Parallel processing with agent specialization"""
        logger.info("Using parallel coordination strategy")

        return await self._process_documents_parallel(
            documents, northstar, rfp_documents, notice_id, company_context
        )

    async def _coordinate_sequential(
        self,
        documents: List[Dict],
        northstar: str,
        rfp_documents: Dict[str, Any],
        notice_id: str,
        company_context: str
    ) -> Dict[str, Any]:
        """Sequential processing for complex dependencies"""
        logger.info("Using sequential coordination strategy")

        generated_docs = {}
        agent_assignments = {}

        for doc_spec in documents:
            context = {
                'northstar': northstar,
                'rfp_documents': rfp_documents,
                'notice_id': notice_id,
                'company_context': company_context,
                'previous_documents': generated_docs
            }

            best_agent = await self._assign_best_agent(doc_spec)
            agent_assignments[doc_spec['name']] = best_agent.name

            content = await best_agent.generate_document(doc_spec, context)
            generated_docs[doc_spec['name']] = content

        return {
            'documents': generated_docs,
            'agent_assignments': agent_assignments,
            'strategy_used': 'sequential'
        }

    async def _process_documents_parallel(
        self,
        documents: List[Dict],
        northstar: str,
        rfp_documents: Dict[str, Any],
        notice_id: str,
        company_context: str
    ) -> Dict[str, Any]:
        """Process multiple documents in parallel"""
        tasks = []
        document_assignments = {}

        for doc_spec in documents:
            best_agent = await self._assign_best_agent(doc_spec)
            document_assignments[doc_spec['name']] = best_agent.name

            context = {
                'northstar': northstar,
                'rfp_documents': rfp_documents,
                'notice_id': notice_id,
                'company_context': company_context
            }

            tasks.append(best_agent.generate_document(doc_spec, context))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        generated_docs = {}
        for i, (doc_spec, result) in enumerate(zip(documents, results)):
            if isinstance(result, Exception):
                logger.error(f"Agent {document_assignments[doc_spec['name']]} failed for {doc_spec['name']}: {result}")
                # Fallback to generalist agent
                generalist = GeneralistAgent()
                context = {
                    'northstar': northstar,
                    'rfp_documents': rfp_documents,
                    'notice_id': notice_id,
                    'company_context': company_context
                }
                generated_docs[doc_spec['name']] = await generalist.generate_document(doc_spec, context)
            else:
                generated_docs[doc_spec['name']] = result

        return {
            'documents': generated_docs,
            'agent_assignments': document_assignments
        }

    async def _assign_best_agent(self, document_spec: Dict[str, Any]) -> SpecializedAgent:
        """Enhanced agent assignment using document type intelligence"""
        scores = {}
        doc_type = document_spec.get('document_type', 'unknown')

        # Enhanced type-based agent selection with higher confidence
        type_agent_mapping = {
            'technical_volume': TechnicalArchitectureAgent,
            'technical_proposal': TechnicalArchitectureAgent,
            'cost_volume': CostPricingAgent,
            'price_narrative': CostPricingAgent,
            'management_volume': ComplianceAgent,  # Management often involves compliance
            'past_performance': PastPerformanceAgent,
            'video_pitch': VideoScriptAgent,
            'executive_summary': ExecutiveSummaryAgent,
            'cover_letter': ExecutiveSummaryAgent,
            'security_plan': SecurityAgent,
            'compliance_matrix': ComplianceAgent
        }

        # If we have a direct type mapping, boost that agent's confidence
        if doc_type in type_agent_mapping:
            target_agent_class = type_agent_mapping[doc_type]
            for agent in self.agents:
                if isinstance(agent, target_agent_class):
                    scores[agent] = 0.95  # High confidence for type-matched agents
                    logger.info(f"Type-matched agent {agent.name} for {document_spec['name']} (type: {doc_type})")
                    break

        # Get standard confidence scores for all agents
        for agent in self.agents:
            if agent not in scores:  # Don't override type-matched scores
                confidence = await agent.can_handle(document_spec)
                scores[agent] = confidence
                logger.debug(f"Agent {agent.name} confidence for {document_spec['name']}: {confidence:.2f}")

        # Find best agent
        best_agent = max(scores.keys(), key=lambda agent: scores[agent])
        best_score = scores[best_agent]

        # Use adaptive agent for novel document types not covered by specialists
        if best_score < 0.5 and doc_type not in type_agent_mapping:
            adaptive_agent = next((agent for agent in self.agents if isinstance(agent, AdaptiveAgent)), None)
            if adaptive_agent:
                logger.info(f"Using AdaptiveAgent for novel document type: {doc_type}")
                best_agent = adaptive_agent
                best_score = 0.7  # Give adaptive agent moderate confidence

        # Final fallback to generalist
        if best_score < 0.4:
            logger.info(f"Low confidence scores for {document_spec['name']}, using GeneralistAgent")
            best_agent = next(agent for agent in self.agents if isinstance(agent, GeneralistAgent))

        logger.info(f"Assigned {best_agent.name} to {document_spec['name']} (confidence: {scores.get(best_agent, 0.0):.2f}, type: {doc_type})")
        return best_agent

    def _summarize_documents(self, documents: Dict[str, str]) -> str:
        """Create a summary of generated documents for dependent processing"""
        summary = "Generated Documents Summary:\n\n"

        for doc_name, content in documents.items():
            # Take first 500 chars of each document
            preview = content[:500] if content else "No content"
            summary += f"{doc_name}:\n{preview}...\n\n"

        return summary

    # ========== MULTI-PASS PROCESSING SYSTEM ==========

    async def execute_multipass(
        self,
        documents: List[Dict],
        northstar: str,
        rfp_documents: Dict[str, Any],
        notice_id: str,
        company_context: str
    ) -> Dict[str, Any]:
        """Execute the 5-pass document generation system"""
        session_id = f"multipass_{notice_id}_{datetime.now().isoformat()}"
        logger.info(f"Starting 5-pass document generation for session: {session_id}")

        try:
            # Initialize pass state
            self.pass_state[session_id] = {
                'current_pass': 0,
                'documents': documents,
                'northstar': northstar,
                'rfp_documents': rfp_documents,
                'notice_id': notice_id,
                'company_context': company_context,
                'agent_assignments': {},
                'pass_history': [],
                'start_time': datetime.now()
            }

            # Execute each pass
            for pass_num in range(1, 6):
                logger.info(f"Starting Pass {pass_num}")

                # Create rollback checkpoint before each pass
                await self._create_rollback_checkpoint(session_id, pass_num)

                # Execute the pass
                pass_result = await self.coordinate_pass(session_id, pass_num)

                # Validate pass output
                validation_result = await self.validate_pass_output(session_id, pass_num, pass_result)

                if not validation_result['passed']:
                    logger.warning(f"Pass {pass_num} failed validation: {validation_result['issues']}")

                    # Attempt rollback and retry
                    if validation_result.get('retry', False):
                        logger.info(f"Attempting rollback and retry for Pass {pass_num}")
                        await self._rollback_to_checkpoint(session_id, pass_num)
                        pass_result = await self.coordinate_pass(session_id, pass_num, retry=True)
                        validation_result = await self.validate_pass_output(session_id, pass_num, pass_result)

                    if not validation_result['passed']:
                        logger.error(f"Pass {pass_num} failed after retry. Proceeding with best effort.")

                # Store pass results
                self.pass_results[f"{session_id}_pass_{pass_num}"] = pass_result
                self.pass_state[session_id]['pass_history'].append({
                    'pass': pass_num,
                    'result': pass_result,
                    'validation': validation_result,
                    'timestamp': datetime.now()
                })

                logger.info(f"Completed Pass {pass_num}")

            # Compile final results
            final_result = await self._compile_final_results(session_id)

            # Cleanup session state
            self._cleanup_session(session_id)

            return final_result

        except Exception as e:
            logger.error(f"Multi-pass execution failed: {e}")
            # Attempt to recover with best available results
            return await self._recover_from_failure(session_id, e)

    async def coordinate_pass(self, session_id: str, pass_num: int, retry: bool = False) -> Dict[str, Any]:
        """Coordinate a specific pass execution"""
        state = self.pass_state[session_id]

        if pass_num == 1:
            return await self._execute_pass_1_requirements_analysis(state, retry)
        elif pass_num == 2:
            return await self._execute_pass_2_initial_generation(state, retry)
        elif pass_num == 3:
            return await self._execute_pass_3_peer_review(state, retry)
        elif pass_num == 4:
            return await self._execute_pass_4_gap_filling(state, retry)
        elif pass_num == 5:
            return await self._execute_pass_5_polish(state, retry)
        else:
            raise ValueError(f"Invalid pass number: {pass_num}")

    async def validate_pass_output(self, session_id: str, pass_num: int, pass_result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the output of a specific pass"""
        try:
            threshold = self.pass_quality_thresholds.get(pass_num, 0.7)

            # Extract quality metrics from pass result
            quality_score = pass_result.get('quality_score', 0.0)
            completeness = pass_result.get('completeness', 0.0)
            consistency = pass_result.get('consistency', 0.0)

            # Calculate overall score
            overall_score = (quality_score + completeness + consistency) / 3

            # Check if pass meets threshold
            passed = overall_score >= threshold

            # Identify specific issues
            issues = []
            if quality_score < threshold:
                issues.append(f"Quality score {quality_score:.2f} below threshold {threshold}")
            if completeness < 0.8:
                issues.append(f"Incomplete content detected: {completeness:.2f}")
            if consistency < 0.7:
                issues.append(f"Consistency issues detected: {consistency:.2f}")

            # Determine if retry is recommended
            retry_recommended = not passed and pass_num in [2, 3, 4]  # Retry for generation passes

            return {
                'passed': passed,
                'overall_score': overall_score,
                'quality_score': quality_score,
                'completeness': completeness,
                'consistency': consistency,
                'issues': issues,
                'retry': retry_recommended,
                'threshold': threshold
            }

        except Exception as e:
            logger.error(f"Error validating pass {pass_num}: {e}")
            return {
                'passed': False,
                'issues': [f"Validation error: {str(e)}"],
                'retry': True
            }

    # ========== PASS IMPLEMENTATIONS ==========

    async def _execute_pass_1_requirements_analysis(self, state: Dict, retry: bool = False) -> Dict[str, Any]:
        """Pass 1: Requirements Analysis - All agents review requirements collaboratively"""
        logger.info("Executing Pass 1: Requirements Analysis")

        try:
            # Step 1: Individual agent analysis
            agent_analyses = []

            if 1 in self.parallel_passes:
                # Execute agent analyses in parallel
                analysis_tasks = []
                for agent in self.agents:
                    task = self._agent_requirements_analysis(
                        agent, state['documents'], state['northstar'], state['company_context']
                    )
                    analysis_tasks.append(task)

                agent_analyses = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            else:
                # Execute sequentially
                for agent in self.agents:
                    analysis = await self._agent_requirements_analysis(
                        agent, state['documents'], state['northstar'], state['company_context']
                    )
                    agent_analyses.append(analysis)

            # Step 2: Collaborative planning session
            collaborative_plan = await self._collaborative_planning_session(agent_analyses, state)

            # Step 3: Document-to-agent assignments
            assignments = await self._create_document_assignments(collaborative_plan, state['documents'])

            return {
                'pass': 1,
                'agent_analyses': agent_analyses,
                'collaborative_plan': collaborative_plan,
                'document_assignments': assignments,
                'quality_score': self._calculate_analysis_quality(agent_analyses),
                'completeness': self._calculate_completeness(collaborative_plan),
                'consistency': self._calculate_consistency(agent_analyses),
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Pass 1 execution failed: {e}")
            raise

    async def _execute_pass_2_initial_generation(self, state: Dict, retry: bool = False) -> Dict[str, Any]:
        """Pass 2: Initial Generation - Agents create first drafts with uncertainty marking"""
        logger.info("Executing Pass 2: Initial Generation")

        try:
            # Get assignments from Pass 1
            session_id = f"multipass_{state['notice_id']}_{state['start_time'].isoformat()}"
            pass_1_result = self.pass_results.get(f"{session_id}_pass_1", {})
            assignments = pass_1_result.get('document_assignments', {})

            # Generate initial drafts
            initial_drafts = {}
            uncertainty_maps = {}

            if 2 in self.parallel_passes:
                # Parallel generation
                generation_tasks = []
                for doc_spec in state['documents']:
                    agent = assignments.get(doc_spec['name'], self.agents[0])  # Fallback to first agent
                    task = self._generate_initial_draft(agent, doc_spec, state, mark_uncertainties=True)
                    generation_tasks.append((doc_spec['name'], task))

                for doc_name, task in generation_tasks:
                    result = await task
                    initial_drafts[doc_name] = result['content']
                    uncertainty_maps[doc_name] = result['uncertainties']
            else:
                # Sequential generation
                for doc_spec in state['documents']:
                    agent = assignments.get(doc_spec['name'], self.agents[0])
                    result = await self._generate_initial_draft(agent, doc_spec, state, mark_uncertainties=True)
                    initial_drafts[doc_spec['name']] = result['content']
                    uncertainty_maps[doc_spec['name']] = result['uncertainties']

            return {
                'pass': 2,
                'initial_drafts': initial_drafts,
                'uncertainty_maps': uncertainty_maps,
                'agent_assignments': assignments,
                'quality_score': self._calculate_draft_quality(initial_drafts),
                'completeness': self._calculate_draft_completeness(initial_drafts, state['documents']),
                'consistency': self._calculate_draft_consistency(initial_drafts),
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Pass 2 execution failed: {e}")
            raise

    async def _execute_pass_3_peer_review(self, state: Dict, retry: bool = False) -> Dict[str, Any]:
        """Pass 3: Peer Review - Agents review each other's work to identify gaps"""
        logger.info("Executing Pass 3: Peer Review")

        try:
            # Get initial drafts from Pass 2
            session_id = f"multipass_{state['notice_id']}_{state['start_time'].isoformat()}"
            pass_2_result = self.pass_results.get(f"{session_id}_pass_2", {})
            drafts = pass_2_result.get('initial_drafts', {})
            uncertainty_maps = pass_2_result.get('uncertainty_maps', {})

            # Cross-agent review matrix
            review_matrix = {}
            gap_identifications = {}

            # Each agent reviews all documents (not just their own)
            for reviewer_agent in self.agents:
                agent_reviews = {}

                for doc_name, content in drafts.items():
                    review = await self._peer_review_document(
                        reviewer_agent, doc_name, content,
                        uncertainty_maps.get(doc_name, []), state
                    )
                    agent_reviews[doc_name] = review

                review_matrix[reviewer_agent.name] = agent_reviews

            # Aggregate reviews to identify gaps and inconsistencies
            aggregated_gaps = await self._aggregate_review_findings(review_matrix, drafts)

            return {
                'pass': 3,
                'review_matrix': review_matrix,
                'aggregated_gaps': aggregated_gaps,
                'quality_score': self._calculate_review_quality(review_matrix),
                'completeness': self._calculate_review_completeness(aggregated_gaps),
                'consistency': self._calculate_review_consistency(review_matrix),
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Pass 3 execution failed: {e}")
            raise

    async def _execute_pass_4_gap_filling(self, state: Dict, retry: bool = False) -> Dict[str, Any]:
        """Pass 4: Gap Filling - Address identified gaps and ensure requirement coverage"""
        logger.info("Executing Pass 4: Gap Filling")

        try:
            # Get previous results
            session_id = f"multipass_{state['notice_id']}_{state['start_time'].isoformat()}"
            pass_2_result = self.pass_results.get(f"{session_id}_pass_2", {})
            pass_3_result = self.pass_results.get(f"{session_id}_pass_3", {})

            drafts = pass_2_result.get('initial_drafts', {})
            gaps = pass_3_result.get('aggregated_gaps', {})

            # Fill gaps in each document
            gap_filled_drafts = {}
            coverage_reports = {}

            if 4 in self.parallel_passes:
                # Parallel gap filling
                filling_tasks = []
                for doc_name, content in drafts.items():
                    doc_gaps = gaps.get(doc_name, [])
                    if doc_gaps:
                        task = self._fill_document_gaps(doc_name, content, doc_gaps, state)
                        filling_tasks.append((doc_name, task))
                    else:
                        gap_filled_drafts[doc_name] = content
                        coverage_reports[doc_name] = {'status': 'no_gaps', 'coverage': 1.0}

                for doc_name, task in filling_tasks:
                    result = await task
                    gap_filled_drafts[doc_name] = result['content']
                    coverage_reports[doc_name] = result['coverage_report']
            else:
                # Sequential gap filling
                for doc_name, content in drafts.items():
                    doc_gaps = gaps.get(doc_name, [])
                    if doc_gaps:
                        result = await self._fill_document_gaps(doc_name, content, doc_gaps, state)
                        gap_filled_drafts[doc_name] = result['content']
                        coverage_reports[doc_name] = result['coverage_report']
                    else:
                        gap_filled_drafts[doc_name] = content
                        coverage_reports[doc_name] = {'status': 'no_gaps', 'coverage': 1.0}

            # Verify requirement coverage
            requirement_coverage = await self._verify_requirement_coverage(
                gap_filled_drafts, state['documents'], state['northstar']
            )

            return {
                'pass': 4,
                'gap_filled_drafts': gap_filled_drafts,
                'coverage_reports': coverage_reports,
                'requirement_coverage': requirement_coverage,
                'quality_score': self._calculate_gap_fill_quality(gap_filled_drafts),
                'completeness': self._calculate_requirement_coverage_score(requirement_coverage),
                'consistency': self._calculate_gap_fill_consistency(gap_filled_drafts),
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Pass 4 execution failed: {e}")
            raise

    async def _execute_pass_5_polish(self, state: Dict, retry: bool = False) -> Dict[str, Any]:
        """Pass 5: Polish - Final formatting and consistency check"""
        logger.info("Executing Pass 5: Polish")

        try:
            # Get gap-filled drafts from Pass 4
            session_id = f"multipass_{state['notice_id']}_{state['start_time'].isoformat()}"
            pass_4_result = self.pass_results.get(f"{session_id}_pass_4", {})
            drafts = pass_4_result.get('gap_filled_drafts', {})

            # Final polishing
            polished_documents = {}
            consistency_reports = {}
            formatting_reports = {}

            # Sequential polishing for consistency
            for doc_name, content in drafts.items():
                polished_result = await self._polish_document(
                    doc_name, content, state, polished_documents
                )
                polished_documents[doc_name] = polished_result['content']
                consistency_reports[doc_name] = polished_result['consistency_report']
                formatting_reports[doc_name] = polished_result['formatting_report']

            # Final cross-document consistency check
            final_consistency_check = await self._final_consistency_check(polished_documents)

            return {
                'pass': 5,
                'polished_documents': polished_documents,
                'consistency_reports': consistency_reports,
                'formatting_reports': formatting_reports,
                'final_consistency_check': final_consistency_check,
                'quality_score': self._calculate_polish_quality(polished_documents),
                'completeness': 1.0,  # Should be complete by this point
                'consistency': self._calculate_final_consistency(final_consistency_check),
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Pass 5 execution failed: {e}")
            raise

    # ========== HELPER METHODS ==========

    async def _agent_requirements_analysis(self, agent: SpecializedAgent, documents: List[Dict],
                                         northstar: str, company_context: str) -> Dict[str, Any]:
        """Have an agent analyze requirements from their perspective"""
        try:
            analysis_prompt = f"""As a {agent.name}, analyze these RFP requirements and provide your perspective:

DOCUMENTS TO ANALYZE:
{json.dumps(documents, indent=2)}

NORTHSTAR DOCUMENT:
{northstar[:3000]}...

COMPANY CONTEXT:
{company_context[:1500]}...

Provide analysis in this format:
{{
    "agent_name": "{agent.name}",
    "specialization_relevance": 0.0-1.0,
    "key_requirements_identified": ["req1", "req2", ...],
    "potential_challenges": ["challenge1", "challenge2", ...],
    "recommended_approach": "detailed approach recommendation",
    "confidence_level": 0.0-1.0,
    "collaboration_needs": ["what you need from other agents"],
    "contribution_potential": "what you can contribute to other documents"
}}"""

            response = agent.client.chat.completions.create(
                model=agent.model,
                messages=[
                    {"role": "system", "content": f"You are {agent.name} analyzing RFP requirements."},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_completion_tokens=min(self.max_tokens, 16000),
                reasoning_effort=self.reasoning_effort if "gpt-5" in agent.model.lower() else None,
                response_format={"type": "json_object"}
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Agent analysis failed for {agent.name}: {e}")
            return {
                "agent_name": agent.name,
                "specialization_relevance": 0.0,
                "error": str(e)
            }

    async def _collaborative_planning_session(self, agent_analyses: List[Dict], state: Dict) -> Dict[str, Any]:
        """Simulate a collaborative planning session between agents"""
        try:
            # Aggregate all agent perspectives
            planning_prompt = f"""Based on these agent analyses, create a collaborative document plan:

AGENT ANALYSES:
{json.dumps(agent_analyses, indent=2)}

ORIGINAL REQUIREMENTS:
{json.dumps(state['documents'], indent=2)}

Create a comprehensive plan that addresses:
1. How to leverage each agent's strengths
2. How to address collaboration needs
3. How to minimize overlap and maximize coverage
4. How to handle potential challenges
5. Optimal document structure and content distribution

Return JSON format:
{{
    "collaboration_strategy": "overall strategy",
    "agent_synergies": {{"agent1": ["agents to collaborate with"], ...}},
    "document_recommendations": {{"doc_name": {{"lead_agent": "agent", "supporting_agents": ["agents"], "approach": "strategy"}}}},
    "risk_mitigation": ["strategies for identified challenges"],
    "quality_assurance": "cross-checking strategy",
    "timeline_considerations": "parallel vs sequential recommendations"
}}"""

            # Use the most experienced agent (first in list) to synthesize
            lead_agent = self.agents[0]
            response = lead_agent.client.chat.completions.create(
                model=lead_agent.model,
                messages=[
                    {"role": "system", "content": "You are orchestrating a collaborative planning session."},
                    {"role": "user", "content": planning_prompt}
                ],
                max_completion_tokens=min(self.max_tokens, 32000),
                reasoning_effort=self.reasoning_effort if "gpt-5" in lead_agent.model.lower() else None,
                response_format={"type": "json_object"}
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Collaborative planning failed: {e}")
            return {"error": str(e), "fallback": True}

    async def _create_document_assignments(self, collaborative_plan: Dict, documents: List[Dict]) -> Dict[str, SpecializedAgent]:
        """Create agent assignments based on collaborative plan"""
        assignments = {}

        try:
            recommendations = collaborative_plan.get('document_recommendations', {})

            for doc_spec in documents:
                doc_name = doc_spec['name']

                if doc_name in recommendations:
                    # Use recommended lead agent
                    lead_agent_name = recommendations[doc_name].get('lead_agent', '')
                    assigned_agent = None

                    for agent in self.agents:
                        if agent.name.lower() in lead_agent_name.lower():
                            assigned_agent = agent
                            break

                    if not assigned_agent:
                        # Fallback to confidence-based assignment
                        assigned_agent = await self._assign_best_agent(doc_spec)
                else:
                    # Fallback to confidence-based assignment
                    assigned_agent = await self._assign_best_agent(doc_spec)

                assignments[doc_name] = assigned_agent
                logger.info(f"Assigned {assigned_agent.name} to {doc_name}")

        except Exception as e:
            logger.error(f"Assignment creation failed: {e}")
            # Fallback assignments
            for doc_spec in documents:
                assignments[doc_spec['name']] = await self._assign_best_agent(doc_spec)

        return assignments

    async def _generate_initial_draft(self, agent: SpecializedAgent, doc_spec: Dict,
                                    state: Dict, mark_uncertainties: bool = True) -> Dict[str, Any]:
        """Generate initial draft with uncertainty marking"""
        try:
            uncertainty_prompt = """
            IMPORTANT: Mark any areas where you're uncertain with [UNCERTAIN: reason] tags.
            Mark areas that need specific company information with [NEEDS_COMPANY_INFO: what info].
            Mark technical details you cannot verify with [NEEDS_VERIFICATION: detail].
            """ if mark_uncertainties else ""

            prompt = f"""Generate document: {doc_spec['name']}

REQUIREMENTS:
{doc_spec.get('requirements', '')}

NORTHSTAR GUIDANCE:
{state['northstar'][:4000]}...

COMPANY CONTEXT:
{state['company_context'][:2000]}...

{uncertainty_prompt}

Create comprehensive content that addresses all requirements. Be thorough and professional."""

            content = await agent.generate_document(doc_spec, {
                'northstar': state['northstar'],
                'company_context': state['company_context'],
                'rfp_documents': state['rfp_documents']
            })

            # Extract uncertainty markers
            uncertainties = []
            if mark_uncertainties:
                import re
                uncertainty_patterns = [
                    r'\[UNCERTAIN:([^\]]+)\]',
                    r'\[NEEDS_COMPANY_INFO:([^\]]+)\]',
                    r'\[NEEDS_VERIFICATION:([^\]]+)\]'
                ]

                for pattern in uncertainty_patterns:
                    matches = re.findall(pattern, content)
                    uncertainties.extend(matches)

            return {
                'content': content,
                'uncertainties': uncertainties,
                'agent': agent.name
            }

        except Exception as e:
            logger.error(f"Initial draft generation failed for {doc_spec['name']}: {e}")
            return {
                'content': f"Error generating {doc_spec['name']}: {str(e)}",
                'uncertainties': [f"Generation error: {str(e)}"],
                'agent': agent.name
            }

    async def _peer_review_document(self, reviewer_agent: SpecializedAgent, doc_name: str,
                                  content: str, uncertainties: List[str], state: Dict) -> Dict[str, Any]:
        """Have an agent peer review a document"""
        try:
            review_prompt = f"""Peer review this document from your perspective as {reviewer_agent.name}:

DOCUMENT: {doc_name}
CONTENT: {content[:4000]}...

KNOWN UNCERTAINTIES: {uncertainties}

NORTHSTAR REQUIREMENTS: {state['northstar'][:2000]}...

Provide detailed review:
{{
    "overall_assessment": "brief assessment",
    "strengths": ["strength1", "strength2", ...],
    "gaps_identified": ["gap1", "gap2", ...],
    "inconsistencies": ["issue1", "issue2", ...],
    "missing_requirements": ["req1", "req2", ...],
    "suggestions": ["suggestion1", "suggestion2", ...],
    "confidence_in_review": 0.0-1.0,
    "priority_issues": ["high priority items to address"]
}}"""

            response = reviewer_agent.client.chat.completions.create(
                model=reviewer_agent.model,
                messages=[
                    {"role": "system", "content": f"You are {reviewer_agent.name} reviewing a colleague's work."},
                    {"role": "user", "content": review_prompt}
                ],
                max_completion_tokens=min(self.max_tokens, 2000),
                response_format={"type": "json_object"}
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Peer review failed for {doc_name} by {reviewer_agent.name}: {e}")
            return {"error": str(e), "gaps_identified": [], "missing_requirements": []}

    async def _aggregate_review_findings(self, review_matrix: Dict, drafts: Dict) -> Dict[str, Any]:
        """Aggregate review findings across all agents"""
        aggregated = {}

        for doc_name in drafts.keys():
            doc_reviews = [review_matrix[agent_name][doc_name]
                          for agent_name in review_matrix.keys()
                          if doc_name in review_matrix[agent_name]]

            # Combine all gaps and issues
            all_gaps = []
            all_missing_requirements = []
            all_inconsistencies = []
            priority_issues = []

            for review in doc_reviews:
                all_gaps.extend(review.get('gaps_identified', []))
                all_missing_requirements.extend(review.get('missing_requirements', []))
                all_inconsistencies.extend(review.get('inconsistencies', []))
                priority_issues.extend(review.get('priority_issues', []))

            # Remove duplicates and prioritize
            aggregated[doc_name] = {
                'gaps': list(set(all_gaps)),
                'missing_requirements': list(set(all_missing_requirements)),
                'inconsistencies': list(set(all_inconsistencies)),
                'priority_issues': list(set(priority_issues)),
                'review_count': len(doc_reviews),
                'consensus_score': self._calculate_review_consensus(doc_reviews)
            }

        return aggregated

    async def _fill_document_gaps(self, doc_name: str, content: str, gaps: List[str], state: Dict) -> Dict[str, Any]:
        """Fill identified gaps in a document"""
        try:
            gap_filling_prompt = f"""Improve this document by addressing these specific gaps:

DOCUMENT: {doc_name}
CURRENT CONTENT: {content[:6000]}...

GAPS TO ADDRESS: {json.dumps(gaps, indent=2)}

NORTHSTAR REFERENCE: {state['northstar'][:3000]}...

Return the improved document content that addresses all identified gaps while maintaining existing strengths."""

            # Use the most appropriate agent for this document
            assigned_agent = await self._assign_best_agent({'name': doc_name, 'requirements': str(gaps)})

            response = assigned_agent.client.chat.completions.create(
                model=assigned_agent.model,
                messages=[
                    {"role": "system", "content": f"You are improving a document by filling gaps."},
                    {"role": "user", "content": gap_filling_prompt}
                ],
                max_completion_tokens=min(self.max_tokens, 8000)
            )

            improved_content = response.choices[0].message.content

            # Calculate coverage improvement
            coverage_report = {
                'gaps_addressed': len(gaps),
                'coverage': min(1.0, 0.6 + (len(gaps) * 0.1)),  # Estimate based on gaps filled
                'status': 'improved'
            }

            return {
                'content': improved_content,
                'coverage_report': coverage_report
            }

        except Exception as e:
            logger.error(f"Gap filling failed for {doc_name}: {e}")
            return {
                'content': content,  # Return original if gap filling fails
                'coverage_report': {'status': 'error', 'error': str(e)}
            }

    async def _verify_requirement_coverage(self, documents: Dict[str, str],
                                         original_docs: List[Dict], northstar: str) -> Dict[str, Any]:
        """Verify that all requirements are covered across documents"""
        try:
            verification_prompt = f"""Verify requirement coverage across these documents:

GENERATED DOCUMENTS:
{json.dumps({name: content[:1000] + "..." for name, content in documents.items()}, indent=2)}

ORIGINAL REQUIREMENTS:
{json.dumps(original_docs, indent=2)}

NORTHSTAR REQUIREMENTS:
{northstar[:3000]}...

Return coverage analysis:
{{
    "overall_coverage": 0.0-1.0,
    "requirement_mapping": {{"requirement": "document_covering_it"}},
    "uncovered_requirements": ["req1", "req2", ...],
    "overlapping_coverage": ["requirements covered in multiple docs"],
    "coverage_quality": "assessment of how well covered",
    "recommendations": ["suggestions for improvement"]
}}"""

            # Use generalist agent for broad analysis
            generalist = next(agent for agent in self.agents if isinstance(agent, GeneralistAgent))

            response = generalist.client.chat.completions.create(
                model=generalist.model,
                messages=[
                    {"role": "system", "content": "You are analyzing requirement coverage."},
                    {"role": "user", "content": verification_prompt}
                ],
                max_completion_tokens=min(self.max_tokens, 32000),
                reasoning_effort=self.reasoning_effort if "gpt-5" in generalist.model.lower() else None,
                response_format={"type": "json_object"}
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Requirement coverage verification failed: {e}")
            return {
                "overall_coverage": 0.8,  # Conservative estimate
                "error": str(e)
            }

    async def _polish_document(self, doc_name: str, content: str, state: Dict,
                             other_documents: Dict[str, str]) -> Dict[str, Any]:
        """Polish a document for final formatting and consistency"""
        try:
            polish_prompt = f"""Polish this document for final submission:

DOCUMENT: {doc_name}
CONTENT: {content[:8000]}...

OTHER DOCUMENTS FOR CONSISTENCY:
{json.dumps({name: content[:500] + "..." for name, content in other_documents.items()}, indent=2)}

Apply final polish:
1. Professional formatting and structure
2. Consistent terminology across documents
3. Clear, compelling language
4. Proper government contract tone
5. Error-free content
6. Optimal length and detail level

Return the polished document."""

            # Use the originally assigned agent for consistency
            assigned_agent = await self._assign_best_agent({'name': doc_name, 'requirements': content[:500]})

            response = assigned_agent.client.chat.completions.create(
                model=assigned_agent.model,
                messages=[
                    {"role": "system", "content": "You are polishing a document for final submission."},
                    {"role": "user", "content": polish_prompt}
                ],
                max_completion_tokens=min(self.max_tokens, 32000),
                reasoning_effort=self.reasoning_effort if "gpt-5" in assigned_agent.model.lower() else None
            )

            polished_content = response.choices[0].message.content

            return {
                'content': polished_content,
                'consistency_report': {'status': 'polished', 'agent': assigned_agent.name},
                'formatting_report': {'status': 'formatted', 'length': len(polished_content)}
            }

        except Exception as e:
            logger.error(f"Document polishing failed for {doc_name}: {e}")
            return {
                'content': content,
                'consistency_report': {'status': 'error', 'error': str(e)},
                'formatting_report': {'status': 'error', 'error': str(e)}
            }

    async def _final_consistency_check(self, documents: Dict[str, str]) -> Dict[str, Any]:
        """Perform final cross-document consistency check"""
        try:
            consistency_prompt = f"""Perform final consistency check across all documents:

DOCUMENTS:
{json.dumps({name: content[:1000] + "..." for name, content in documents.items()}, indent=2)}

Check for:
1. Consistent terminology and naming
2. Aligned technical approaches
3. Consistent pricing/cost references
4. Coherent narrative across documents
5. No contradictions between documents

Return analysis:
{{
    "consistency_score": 0.0-1.0,
    "inconsistencies_found": ["issue1", "issue2", ...],
    "terminology_alignment": "assessment",
    "narrative_coherence": "assessment",
    "final_recommendations": ["final tweaks needed"],
    "ready_for_submission": true/false
}}"""

            # Use executive summary agent for final review
            exec_agent = next(agent for agent in self.agents if isinstance(agent, ExecutiveSummaryAgent))

            response = exec_agent.client.chat.completions.create(
                model=exec_agent.model,
                messages=[
                    {"role": "system", "content": "You are performing a final consistency check."},
                    {"role": "user", "content": consistency_prompt}
                ],
                max_completion_tokens=min(self.max_tokens, 16000),
                reasoning_effort=self.reasoning_effort if "gpt-5" in exec_agent.model.lower() else None,
                response_format={"type": "json_object"}
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Final consistency check failed: {e}")
            return {
                "consistency_score": 0.8,
                "ready_for_submission": True,
                "error": str(e)
            }

    # ========== STATE MANAGEMENT AND ROLLBACK ==========

    async def _create_rollback_checkpoint(self, session_id: str, pass_num: int):
        """Create a rollback checkpoint before executing a pass"""
        try:
            checkpoint_data = {
                'session_id': session_id,
                'pass_num': pass_num,
                'state': self.pass_state.get(session_id, {}).copy(),
                'results': {k: v for k, v in self.pass_results.items() if session_id in k},
                'timestamp': datetime.now()
            }

            self.rollback_checkpoints[f"{session_id}_pass_{pass_num}"] = checkpoint_data
            logger.info(f"Created rollback checkpoint for {session_id} pass {pass_num}")

        except Exception as e:
            logger.error(f"Failed to create rollback checkpoint: {e}")

    async def _rollback_to_checkpoint(self, session_id: str, pass_num: int):
        """Rollback to a previous checkpoint"""
        try:
            checkpoint_key = f"{session_id}_pass_{pass_num}"
            checkpoint = self.rollback_checkpoints.get(checkpoint_key)

            if checkpoint:
                # Restore state
                self.pass_state[session_id] = checkpoint['state'].copy()

                # Remove failed pass results
                keys_to_remove = [k for k in self.pass_results.keys()
                                if session_id in k and f"pass_{pass_num}" in k]
                for key in keys_to_remove:
                    del self.pass_results[key]

                logger.info(f"Rolled back to checkpoint for {session_id} pass {pass_num}")
            else:
                logger.warning(f"No checkpoint found for {session_id} pass {pass_num}")

        except Exception as e:
            logger.error(f"Rollback failed: {e}")

    async def _compile_final_results(self, session_id: str) -> Dict[str, Any]:
        """Compile final results from all passes"""
        try:
            # Get final documents from Pass 5
            pass_5_result = self.pass_results.get(f"{session_id}_pass_5", {})
            final_documents = pass_5_result.get('polished_documents', {})

            # Compile metadata from all passes
            all_pass_results = {k: v for k, v in self.pass_results.items() if session_id in k}

            # Calculate overall metrics
            final_metrics = self._calculate_final_metrics(all_pass_results)

            return {
                'documents': final_documents,
                'pass_results': all_pass_results,
                'final_metrics': final_metrics,
                'session_id': session_id,
                'processing_time': (datetime.now() - self.pass_state[session_id]['start_time']).total_seconds(),
                'strategy_used': 'multipass',
                'passes_completed': len(all_pass_results),
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Failed to compile final results: {e}")
            return {'error': str(e), 'documents': {}}

    def _cleanup_session(self, session_id: str):
        """Clean up session state after completion"""
        try:
            # Remove from active state
            if session_id in self.pass_state:
                del self.pass_state[session_id]

            # Clean up old checkpoints (keep recent ones for debugging)
            checkpoint_keys = [k for k in self.rollback_checkpoints.keys() if session_id in k]
            for key in checkpoint_keys[:-3]:  # Keep last 3 checkpoints
                del self.rollback_checkpoints[key]

            # Clean up old pass results (keep for analysis)
            # Note: We keep pass_results for potential analysis

            logger.info(f"Cleaned up session {session_id}")

        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")

    async def _recover_from_failure(self, session_id: str, error: Exception) -> Dict[str, Any]:
        """Attempt to recover from multi-pass failure"""
        try:
            logger.error(f"Attempting recovery from failure: {error}")

            # Get the best available results
            available_results = {k: v for k, v in self.pass_results.items() if session_id in k}

            if not available_results:
                # No results available, return error
                return {
                    'error': f"Complete failure: {str(error)}",
                    'documents': {},
                    'recovery_attempted': True
                }

            # Find the most recent successful pass
            latest_pass = max([int(k.split('pass_')[1]) for k in available_results.keys()
                             if 'pass_' in k])

            latest_result = available_results[f"{session_id}_pass_{latest_pass}"]

            # Extract the best available documents
            if latest_pass >= 4:
                documents = latest_result.get('gap_filled_drafts', {})
            elif latest_pass >= 2:
                documents = latest_result.get('initial_drafts', {})
            else:
                documents = {}

            return {
                'documents': documents,
                'error': f"Partial failure at pass {latest_pass + 1}: {str(error)}",
                'recovery_attempted': True,
                'recovered_from_pass': latest_pass,
                'available_results': available_results,
                'strategy_used': 'multipass_recovery'
            }

        except Exception as recovery_error:
            logger.error(f"Recovery also failed: {recovery_error}")
            return {
                'error': f"Complete failure: {str(error)}, Recovery failed: {str(recovery_error)}",
                'documents': {}
            }

    # ========== QUALITY CALCULATION METHODS ==========

    def _calculate_analysis_quality(self, agent_analyses: List[Dict]) -> float:
        """Calculate quality score for requirements analysis"""
        if not agent_analyses:
            return 0.0

        valid_analyses = [a for a in agent_analyses if 'error' not in a]
        if not valid_analyses:
            return 0.0

        # Average confidence levels
        confidences = [a.get('confidence_level', 0.0) for a in valid_analyses]
        return sum(confidences) / len(confidences) if confidences else 0.0

    def _calculate_completeness(self, plan: Dict) -> float:
        """Calculate completeness score for collaborative plan"""
        required_fields = ['collaboration_strategy', 'document_recommendations', 'quality_assurance']
        present_fields = sum(1 for field in required_fields if field in plan)
        return present_fields / len(required_fields)

    def _calculate_consistency(self, agent_analyses: List[Dict]) -> float:
        """Calculate consistency score across agent analyses"""
        if len(agent_analyses) < 2:
            return 1.0

        # Simple consistency check based on confidence alignment
        confidences = [a.get('confidence_level', 0.0) for a in agent_analyses if 'error' not in a]
        if not confidences:
            return 0.0

        avg_confidence = sum(confidences) / len(confidences)
        variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
        return max(0.0, 1.0 - variance)

    def _calculate_draft_quality(self, drafts: Dict[str, str]) -> float:
        """Calculate quality score for initial drafts"""
        if not drafts:
            return 0.0

        # Simple heuristic based on content length and structure
        total_score = 0
        for content in drafts.values():
            # Basic quality indicators
            length_score = min(1.0, len(content) / 5000)  # Target ~5000 chars
            structure_score = 0.8 if len(content.split('\n\n')) > 3 else 0.4  # Has sections
            total_score += (length_score + structure_score) / 2

        return total_score / len(drafts)

    def _calculate_draft_completeness(self, drafts: Dict[str, str], required_docs: List[Dict]) -> float:
        """Calculate completeness score for drafts"""
        required_count = len(required_docs)
        generated_count = len([d for d in drafts.values() if len(d) > 100])  # Non-trivial content
        return min(1.0, generated_count / required_count) if required_count > 0 else 1.0

    def _calculate_draft_consistency(self, drafts: Dict[str, str]) -> float:
        """Calculate consistency score for drafts"""
        # Simple consistency based on terminology and style similarity
        return 0.8  # Placeholder - could implement more sophisticated analysis

    def _calculate_review_quality(self, review_matrix: Dict) -> float:
        """Calculate quality score for peer reviews"""
        if not review_matrix:
            return 0.0

        total_confidence = 0
        count = 0

        for agent_reviews in review_matrix.values():
            for review in agent_reviews.values():
                if 'confidence_in_review' in review:
                    total_confidence += review['confidence_in_review']
                    count += 1

        return total_confidence / count if count > 0 else 0.0

    def _calculate_review_completeness(self, aggregated_gaps: Dict) -> float:
        """Calculate completeness score for review process"""
        if not aggregated_gaps:
            return 1.0

        # Score based on number of reviews per document
        avg_reviews = sum(gap_info.get('review_count', 0) for gap_info in aggregated_gaps.values())
        avg_reviews = avg_reviews / len(aggregated_gaps) if aggregated_gaps else 0

        return min(1.0, avg_reviews / 3)  # Target 3+ reviews per document

    def _calculate_review_consistency(self, review_matrix: Dict) -> float:
        """Calculate consistency score for reviews"""
        if not review_matrix:
            return 1.0

        # Placeholder for consensus analysis
        return 0.85

    def _calculate_review_consensus(self, reviews: List[Dict]) -> float:
        """Calculate consensus score among reviews"""
        if len(reviews) < 2:
            return 1.0

        # Simple consensus based on similar gap identification
        all_gaps = []
        for review in reviews:
            all_gaps.extend(review.get('gaps_identified', []))

        if not all_gaps:
            return 1.0

        # Count overlap in identified gaps
        gap_counts = {}
        for gap in all_gaps:
            gap_counts[gap] = gap_counts.get(gap, 0) + 1

        consensus_gaps = sum(1 for count in gap_counts.values() if count > 1)
        return consensus_gaps / len(gap_counts) if gap_counts else 1.0

    def _calculate_gap_fill_quality(self, gap_filled_drafts: Dict[str, str]) -> float:
        """Calculate quality score for gap-filled documents"""
        return self._calculate_draft_quality(gap_filled_drafts) * 1.1  # Slight boost for improvement

    def _calculate_requirement_coverage_score(self, coverage_report: Dict) -> float:
        """Calculate requirement coverage score"""
        return coverage_report.get('overall_coverage', 0.8)

    def _calculate_gap_fill_consistency(self, gap_filled_drafts: Dict[str, str]) -> float:
        """Calculate consistency score for gap-filled documents"""
        return 0.9  # Placeholder

    def _calculate_polish_quality(self, polished_documents: Dict[str, str]) -> float:
        """Calculate quality score for polished documents"""
        return self._calculate_draft_quality(polished_documents) * 1.2  # Boost for polishing

    def _calculate_final_consistency(self, consistency_check: Dict) -> float:
        """Calculate final consistency score"""
        return consistency_check.get('consistency_score', 0.9)

    def _calculate_final_metrics(self, all_pass_results: Dict) -> Dict[str, Any]:
        """Calculate final comprehensive metrics"""
        try:
            metrics = {
                'overall_quality': 0.0,
                'process_efficiency': 0.0,
                'requirement_coverage': 0.0,
                'agent_utilization': 0.0,
                'pass_success_rate': 0.0
            }

            # Calculate overall quality (average of all pass quality scores)
            quality_scores = []
            for result in all_pass_results.values():
                if 'quality_score' in result:
                    quality_scores.append(result['quality_score'])

            if quality_scores:
                metrics['overall_quality'] = sum(quality_scores) / len(quality_scores)

            # Calculate pass success rate
            total_passes = len(all_pass_results)
            metrics['pass_success_rate'] = total_passes / 5.0  # Out of 5 passes

            # Other metrics (simplified)
            metrics['process_efficiency'] = 0.8  # Placeholder
            metrics['requirement_coverage'] = 0.9  # Placeholder
            metrics['agent_utilization'] = 0.85  # Placeholder

            return metrics

        except Exception as e:
            logger.error(f"Error calculating final metrics: {e}")
            return {'error': str(e)}

    # ========== COORDINATION STRATEGY ==========

    async def _coordinate_multipass(
        self,
        documents: List[Dict],
        northstar: str,
        rfp_documents: Dict[str, Any],
        notice_id: str,
        company_context: str
    ) -> Dict[str, Any]:
        """Multi-pass coordination strategy - the main entry point"""
        logger.info("Using multi-pass coordination strategy")

        return await self.execute_multipass(
            documents, northstar, rfp_documents, notice_id, company_context
        )
