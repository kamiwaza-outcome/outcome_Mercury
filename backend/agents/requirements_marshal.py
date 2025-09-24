"""
Requirements Marshal Agent for Mercury RFP System
Ensures comprehensive requirement coverage and optimal agent assignment
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import logging
from datetime import datetime
import asyncio
from openai import AsyncOpenAI
from services.openai_service import CIRCUIT_BREAKER
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RequirementPriority(Enum):
    CRITICAL = "critical"  # Deal-breakers (compliance, mandatory)
    HIGH = "high"         # Major evaluation factors
    MEDIUM = "medium"     # Important but not critical
    LOW = "low"          # Nice-to-have


@dataclass
class Requirement:
    id: str
    description: str
    source_section: str
    priority: RequirementPriority
    document_type_hints: List[str]
    compliance_mandatory: bool
    evaluation_weight: Optional[float]
    dependencies: List[str]
    confidence_score: float
    assigned_document: Optional[str] = None
    coverage_status: str = "unassigned"

    def to_dict(self):
        """Convert to dictionary with serializable types"""
        return {
            'id': self.id,
            'description': self.description,
            'source_section': self.source_section,
            'priority': self.priority.value if isinstance(self.priority, RequirementPriority) else self.priority,
            'document_type_hints': self.document_type_hints,
            'compliance_mandatory': self.compliance_mandatory,
            'evaluation_weight': self.evaluation_weight,
            'dependencies': self.dependencies,
            'confidence_score': self.confidence_score,
            'assigned_document': self.assigned_document,
            'coverage_status': self.coverage_status
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create from dictionary with enum conversion"""
        if 'priority' in data and isinstance(data['priority'], str):
            data['priority'] = RequirementPriority(data['priority'])
        return cls(**data)


@dataclass
class DocumentAssignment:
    document_name: str
    assigned_requirements: List[str]
    agent_recommendations: List[Tuple[str, float]]  # (agent_name, confidence)
    coverage_completeness: float
    risk_factors: List[str]


class RequirementsMarshal:
    """
    Central Requirements Marshal for RFP Processing
    Ensures comprehensive requirement coverage and optimal agent assignment
    """

    CRITICAL_DOCUMENT_PATTERNS = {
        "volume_1": ["volume 1", "volume i", "technical proposal", "technical volume", "technical capability"],
        "cover_letter": ["cover letter", "transmittal letter", "introduction", "executive letter"],
        "price_narrative": ["price narrative", "pricing narrative", "cost narrative", "pricing explanation"],
        "cost_proposal": ["cost proposal", "pricing proposal", "financial proposal", "price volume"],
        "executive_summary": ["executive summary", "executive overview"],
        "past_performance": ["past performance", "experience", "prior work", "contract history"],
        "draft_sow": ["draft sow", "statement of work", "sow", "draft statement"],
    }

    def __init__(self, agents_registry: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents_registry = agents_registry or {}
        self.requirements_registry: Dict[str, Requirement] = {}
        self.document_assignments: Dict[str, DocumentAssignment] = {}
        self.coverage_matrix: Dict[str, Set[str]] = {}  # doc_name -> requirement_ids
        self.critical_documents = {
            "Volume 1", "Volume I", "Technical Volume", "Technical Proposal",
            "Cover Letter", "Price Narrative", "Cost Proposal",
            "Executive Summary", "Past Performance", "Draft SOW"
        }

        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=float(os.getenv("OPENAI_TIMEOUT", "180"))
        )
        # Use GPT-5-mini for faster processing
        self.model = os.getenv("REQUIREMENTS_MODEL", "gpt-5-mini")
        # Constrain outputs to reduce timeouts and failures
        # Requirements extraction should be fast - just need a list, not generation
        self.max_tokens = int(os.getenv("REQUIREMENTS_MAX_TOKENS", "8000"))
        # Balance quality/latency - low is fine for simple extraction
        self.reasoning_effort = os.getenv("REQUIREMENTS_REASONING", "low")

        # Log configuration for debugging
        logger.info(f"RequirementsMarshal initialized with model: {self.model}")
        logger.info(f"Max tokens: {self.max_tokens}")
        logger.info(f"Reasoning effort: {self.reasoning_effort}")

        # Define agent capabilities for intelligent assignment
        # Maps to actual agents in MultiPassOrchestrator
        self.agent_capabilities = {
            "volume_organizer": {
                "expertise": "Federal proposal volumes, document organization, compliance matrices, past performance",
                "best_for": ["volume", "organization", "federal", "compliance", "past_performance"],
                "confidence": 0.95
            },
            "cover_letter": {
                "expertise": "Executive communications, cover letters, certifications, executive summaries",
                "best_for": ["executive", "cover", "letter", "summary", "certification"],
                "confidence": 0.95
            },
            "sow": {
                "expertise": "Statements of work, project planning, work breakdown structures, timelines",
                "best_for": ["sow", "statement_of_work", "project", "work_breakdown", "timeline"],
                "confidence": 0.95
            },
            "artifact_generator": {
                "expertise": "Technical artifacts, sample deliverables, demonstrations, examples",
                "best_for": ["sample", "artifact", "demonstration", "technical", "example"],
                "confidence": 0.90
            },
            "price_narrative": {
                "expertise": "Pricing strategies, cost justifications, CLIN mapping, budget analysis",
                "best_for": ["price", "cost", "budget", "pricing", "financial", "clin"],
                "confidence": 0.95
            }
        }

    def _select_model(self, requested_model: str) -> str:
        if "gpt-5" in (requested_model or "").lower() and not CIRCUIT_BREAKER.should_allow():
            logger.warning("GPT-5 circuit open; RequirementsMarshal using fallback model gpt-4o")
            return os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o")
        return requested_model

    async def parse_northstar_requirements(self, northstar_document: str) -> Dict[str, Requirement]:
        """
        Extract ALL deliverables from Northstar with intelligent parsing
        """
        # If document is very large, use simplified approach immediately to avoid timeouts
        # Increased to 400k chars (approximately 100k tokens) as per user request
        if len(northstar_document) > 400000:
            logger.info(f"Document is large ({len(northstar_document)} chars), using simplified extraction")
            return await self._simplified_requirement_extraction(northstar_document)

        try:
            system_prompt = """You are a requirements extraction expert for federal RFP proposals.
            Analyze the Northstar document and extract EVERY requirement, deliverable, and compliance item.
            Be exhaustive and ensure nothing is missed. Pay special attention to:
            - Volume structures (Volume I, II, III, IV)
            - Cover letters and executive summaries
            - Technical proposals and approaches
            - Cost/price proposals and narratives
            - Past performance documentation
            - Draft statements of work (SOW)
            - Compliance matrices and certifications
            - Any specialized artifacts or examples required"""

            extraction_prompt = f"""
            Extract ALL requirements from this Northstar document:

            {northstar_document[:10000]}  # First 10k chars for context (reduced for stability)

            CRITICAL EXTRACTION RULES:
            1. Parse EVERY section for requirements, especially looking for Volume I/II/III/IV structures
            2. Identify ALL document types mentioned (Cover Letter, Technical Volume, Price Narrative, etc.)
            3. Extract mandatory vs. optional items
            4. Capture ALL evaluation criteria and weights
            5. Note ALL document format and page limit requirements
            6. Identify ALL compliance obligations and certifications
            7. Extract ALL submission requirements and deadlines
            8. Find ALL required examples or artifacts

            Return JSON with this exact structure:
            {{
                "requirements": [
                    {{
                        "id": "REQ_001",
                        "description": "Complete requirement description",
                        "source_section": "Section where found",
                        "priority": "critical|high|medium|low",
                        "document_type_hints": ["technical", "cost", "volume_1", "cover_letter"],
                        "compliance_mandatory": true,
                        "evaluation_weight": 0.25,
                        "dependencies": ["REQ_002"],
                        "confidence_score": 0.95
                    }}
                ],
                "critical_documents": ["Volume I - Technical", "Cover Letter", "Price Narrative"],
                "document_specifications": {{
                    "Volume I": {{"format": "pdf", "page_limit": 40, "mandatory": true}},
                    "Cover Letter": {{"format": "pdf", "page_limit": 2, "mandatory": true}}
                }}
            }}
            """

            try:
                effective_model = self._select_model(self.model)
                logger.info(f"Making OpenAI request with model: {effective_model}, max_tokens: {self.max_tokens}")
                logger.info(f"Prompt length: {len(extraction_prompt)} characters")
                response = await self.client.chat.completions.create(
                    model=effective_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": extraction_prompt}
                    ],
                    response_format={"type": "json_object"},
                    max_completion_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort if "gpt-5" in effective_model else None,
                    # GPT-5 only supports default temperature=1
                )
                logger.debug(f"OpenAI request completed successfully")
                if "gpt-5" in effective_model.lower():
                    try:
                        CIRCUIT_BREAKER.record_success()
                    except Exception:
                        pass
            except Exception as openai_error:
                logger.error(f"OpenAI API call failed: {openai_error}", exc_info=True)
                raise ValueError(f"OpenAI API call failed: {openai_error}") from openai_error

            # Validate response before parsing
            response_content = response.choices[0].message.content
            logger.debug(f"OpenAI response content type: {type(response_content)}")
            logger.debug(f"OpenAI response content length: {len(response_content) if response_content else 0}")

            if not response_content or response_content.strip() == "":
                logger.error("OpenAI returned empty or None content")
                logger.error(f"Full response: {response}")
                raise ValueError("OpenAI API returned empty response content")

            try:
                parsed_data = json.loads(response_content)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                logger.error(f"Response content (first 1000 chars): {response_content[:1000]}")
                raise ValueError(f"Failed to parse OpenAI response as JSON: {e}")

            # Convert to Requirement objects
            requirements = {}
            for req_data in parsed_data.get("requirements", []):
                req = Requirement(
                    id=req_data["id"],
                    description=req_data["description"],
                    source_section=req_data["source_section"],
                    priority=RequirementPriority(req_data["priority"]),
                    document_type_hints=req_data["document_type_hints"],
                    compliance_mandatory=req_data["compliance_mandatory"],
                    evaluation_weight=req_data.get("evaluation_weight"),
                    dependencies=req_data.get("dependencies", []),
                    confidence_score=req_data["confidence_score"]
                )
                requirements[req.id] = req

            self.requirements_registry = requirements

            # Update critical documents list
            self.critical_documents.update(parsed_data.get("critical_documents", []))

            logger.info(f"Extracted {len(requirements)} requirements from Northstar")
            self._log_requirement_summary(requirements)

            return requirements

        except Exception as e:
            logger.error(f"Error parsing Northstar requirements: {e}", exc_info=True)
            logger.error(f"Exception type: {type(e).__name__}")
            if hasattr(e, 'args') and e.args:
                logger.error(f"Exception details: {e.args}")

            # Try a simplified approach with smaller prompt before falling back to pattern matching
            # Increased threshold to 200k chars to allow more comprehensive extraction
            if len(northstar_document) > 200000:
                logger.info("Attempting simplified extraction with truncated document")
                try:
                    simplified_prompt = f"""
                    Extract key requirements from this document summary:

                    {northstar_document[:3000]}

                    Return JSON with this structure:
                    {{
                        "requirements": [
                            {{
                                "id": "REQ_001",
                                "description": "Requirement description",
                                "priority": "critical",
                                "document_type_hints": ["technical"],
                                "compliance_mandatory": true,
                                "confidence_score": 0.8
                            }}
                        ]
                    }}
                    """

                    simplified_response = await self.client.chat.completions.create(
                        model=self._select_model(self.model),
                        messages=[
                            {"role": "system", "content": "You are a requirements extraction expert. Return only valid JSON."},
                            {"role": "user", "content": simplified_prompt}
                        ],
                        response_format={"type": "json_object"},
                        max_completion_tokens=16000,
                        reasoning_effort=self.reasoning_effort if "gpt-5" in self.model else None,
                    )

                    if simplified_response.choices[0].message.content:
                        simplified_data = json.loads(simplified_response.choices[0].message.content)
                        requirements = {}
                        for req_data in simplified_data.get("requirements", []):
                            req = Requirement(
                                id=req_data["id"],
                                description=req_data["description"],
                                source_section="Simplified extraction",
                                priority=RequirementPriority(req_data.get("priority", "medium")),
                                document_type_hints=req_data.get("document_type_hints", []),
                                compliance_mandatory=req_data.get("compliance_mandatory", False),
                                evaluation_weight=req_data.get("evaluation_weight"),
                                dependencies=req_data.get("dependencies", []),
                                confidence_score=req_data.get("confidence_score", 0.5)
                            )
                            requirements[req.id] = req

                        logger.info(f"Simplified extraction succeeded with {len(requirements)} requirements")
                        return requirements

                except Exception as simplified_error:
                    logger.warning(f"Simplified extraction also failed: {simplified_error}")

            # Fall back to pattern-based extraction
            logger.info("Falling back to pattern-based requirement extraction")
            return await self._fallback_requirement_extraction(northstar_document)

    def _log_requirement_summary(self, requirements: Dict[str, Requirement]):
        """Log comprehensive requirements summary for transparency"""
        priority_counts = {}
        for req in requirements.values():
            priority_counts[req.priority.value] = priority_counts.get(req.priority.value, 0) + 1

        logger.info("=== REQUIREMENTS EXTRACTION SUMMARY ===")
        logger.info(f"Total requirements extracted: {len(requirements)}")
        logger.info(f"Priority breakdown: {priority_counts}")
        logger.info(f"Critical documents identified: {list(self.critical_documents)}")

        # Log critical requirements for visibility
        critical_reqs = [req for req in requirements.values() if req.priority == RequirementPriority.CRITICAL]
        logger.info(f"Critical requirements ({len(critical_reqs)}):")
        for req in critical_reqs[:5]:  # Log first 5 critical requirements
            logger.info(f"  - {req.id}: {req.description[:100]}...")

    async def create_requirements_registry(self) -> Dict[str, Any]:
        """
        Create comprehensive requirements tracking registry
        """
        registry = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_requirements": len(self.requirements_registry),
                "critical_count": len([r for r in self.requirements_registry.values()
                                     if r.priority == RequirementPriority.CRITICAL]),
                "compliance_mandatory_count": len([r for r in self.requirements_registry.values()
                                                 if r.compliance_mandatory])
            },
            "requirements": {
                req_id: {
                    "description": req.description,
                    "priority": req.priority.value,
                    "source_section": req.source_section,
                    "compliance_mandatory": req.compliance_mandatory,
                    "evaluation_weight": req.evaluation_weight,
                    "dependencies": req.dependencies,
                    "document_type_hints": req.document_type_hints,
                    "confidence_score": req.confidence_score,
                    "assigned_document": req.assigned_document,
                    "coverage_status": req.coverage_status
                }
                for req_id, req in self.requirements_registry.items()
            },
            "critical_documents": list(self.critical_documents),
            "coverage_matrix": {k: list(v) for k, v in self.coverage_matrix.items()},
            "assignment_log": []
        }

        return registry

    async def assign_requirements_to_agents(self, identified_documents: List[Dict[str, Any]]) -> Dict[str, DocumentAssignment]:
        """
        Optimal assignment logic ensuring comprehensive coverage
        """
        assignments = {}
        unassigned_requirements = set(self.requirements_registry.keys())

        # First pass: Direct assignment based on document specs
        for doc_spec in identified_documents:
            doc_name = doc_spec.get("document_name", doc_spec.get("name", "Unknown Document"))
            doc_requirements = doc_spec.get("requirements", "")

            assigned_reqs = []

            # Match requirements to this document
            for req_id, requirement in self.requirements_registry.items():
                if req_id in unassigned_requirements:
                    assignment_score = self._calculate_assignment_score(
                        requirement, doc_spec, doc_requirements
                    )

                    if assignment_score > 0.2:  # Lowered confidence threshold for better matching
                        assigned_reqs.append(req_id)
                        unassigned_requirements.remove(req_id)
                        requirement.assigned_document = doc_name
                        requirement.coverage_status = "assigned"

            # Find best agent for this document
            agent_scores = []
            if self.agents_registry:
                for agent_name, agent in self.agents_registry.items():
                    try:
                        if hasattr(agent, 'can_handle'):
                            confidence = await agent.can_handle(doc_spec)
                            agent_scores.append((agent_name, confidence))
                    except Exception as e:
                        logger.warning(f"Error getting confidence from {agent_name}: {e}")
                        agent_scores.append((agent_name, 0.0))

            if not agent_scores:
                agent_scores = [("volume_organizer", 0.8)]

            agent_scores.sort(key=lambda x: x[1], reverse=True)

            assignments[doc_name] = DocumentAssignment(
                document_name=doc_name,
                assigned_requirements=assigned_reqs,
                agent_recommendations=agent_scores,
                coverage_completeness=len(assigned_reqs) / max(len(self.requirements_registry), 1),
                risk_factors=self._identify_risk_factors(assigned_reqs, doc_spec)
            )

        # Second pass: Handle unassigned critical requirements
        if unassigned_requirements:
            logger.warning(f"Found {len(unassigned_requirements)} unassigned requirements")
            await self._handle_unassigned_requirements(unassigned_requirements, assignments)

        # Third pass: Ensure critical documents are not missed
        await self._ensure_critical_documents(assignments, identified_documents)

        self.document_assignments = assignments

        # Update coverage matrix
        for doc_name, assignment in assignments.items():
            self.coverage_matrix[doc_name] = set(assignment.assigned_requirements)

        return assignments

    def _calculate_assignment_score(self, requirement: Requirement, doc_spec: Dict[str, Any], doc_content: str) -> float:
        """Calculate how well a requirement fits a document"""
        score = 0.0
        # Fix: Use consistent document name extraction
        doc_name = doc_spec.get("document_name", doc_spec.get("name", ""))
        doc_name_lower = doc_name.lower()
        doc_content_lower = doc_content.lower()

        # Check for exact document type match
        for hint in requirement.document_type_hints:
            hint_lower = hint.lower()
            if hint_lower in doc_name_lower:
                score += 0.4
            elif hint_lower in doc_content_lower:
                score += 0.2

        # Check for critical document patterns
        for doc_type, patterns in self.CRITICAL_DOCUMENT_PATTERNS.items():
            if any(pattern in doc_name_lower for pattern in patterns):
                if doc_type in str(requirement.document_type_hints).lower():
                    score += 0.3

        # Priority-based scoring
        if requirement.priority == RequirementPriority.CRITICAL:
            score += 0.2
        elif requirement.priority == RequirementPriority.HIGH:
            score += 0.1

        # Compliance mandatory boost
        if requirement.compliance_mandatory:
            score += 0.2

        return min(score, 1.0)

    def _identify_risk_factors(self, assigned_reqs: List[str], doc_spec: Dict[str, Any]) -> List[str]:
        """Identify risk factors for a document assignment"""
        risks = []

        # Check for critical requirements
        critical_reqs = [req_id for req_id in assigned_reqs
                        if self.requirements_registry[req_id].priority == RequirementPriority.CRITICAL]
        if len(critical_reqs) > 5:
            risks.append("high_critical_requirement_concentration")

        # Check for missing mandatory compliance
        mandatory_count = sum(1 for req_id in assigned_reqs
                            if self.requirements_registry[req_id].compliance_mandatory)
        if mandatory_count == 0 and len(assigned_reqs) > 0:
            risks.append("no_mandatory_compliance_items")

        # Check for auto-created documents
        if doc_spec.get("emergency_created"):
            risks.append("emergency_document_creation")

        return risks

    async def _handle_unassigned_requirements(self, unassigned: Set[str], assignments: Dict[str, DocumentAssignment]):
        """Handle requirements that weren't assigned to any document"""

        # Group unassigned requirements by priority
        critical_unassigned = [
            req_id for req_id in unassigned
            if self.requirements_registry[req_id].priority == RequirementPriority.CRITICAL
        ]

        if critical_unassigned:
            logger.error(f"CRITICAL: {len(critical_unassigned)} critical requirements unassigned!")

            # Create emergency documents for critical requirements
            for req_id in critical_unassigned:
                req = self.requirements_registry[req_id]

                # Try to find the best existing document
                best_doc = self._find_best_fit_document(req_id, assignments)
                if best_doc:
                    assignments[best_doc].assigned_requirements.append(req_id)
                    req.assigned_document = best_doc
                    req.coverage_status = "assigned"
                else:
                    # Create emergency document
                    emergency_doc = f"Compliance_Addendum_{req_id}.pdf"

                    if emergency_doc not in assignments:
                        assignments[emergency_doc] = DocumentAssignment(
                            document_name=emergency_doc,
                            assigned_requirements=[req_id],
                            agent_recommendations=[("volume_organizer", 0.9), ("cover_letter", 0.7)],
                            coverage_completeness=0.0,
                            risk_factors=["emergency_document", "critical_requirement_missed"]
                        )
                    else:
                        assignments[emergency_doc].assigned_requirements.append(req_id)

                    req.assigned_document = emergency_doc
                    req.coverage_status = "emergency_assigned"

    def _find_best_fit_document(self, req_id: str, assignments: Dict[str, DocumentAssignment]) -> Optional[str]:
        """Find the best existing document for a requirement"""
        requirement = self.requirements_registry[req_id]
        best_doc = None
        best_score = 0.0

        for doc_name, assignment in assignments.items():
            # Skip emergency documents
            if "Compliance_Addendum" in doc_name:
                continue

            # Calculate fit score
            score = 0.0

            # Check document type hints
            for hint in requirement.document_type_hints:
                if hint.lower() in doc_name.lower():
                    score += 0.5

            # Prefer documents with fewer requirements
            if len(assignment.assigned_requirements) < 10:
                score += 0.2

            if score > best_score:
                best_score = score
                best_doc = doc_name

        return best_doc if best_score > 0.1 else None

    async def _ensure_critical_documents(self, assignments: Dict[str, DocumentAssignment], identified_docs: List[Dict[str, Any]]):
        """Ensure critical documents like Volume 1, Cover Letter are not missed"""

        identified_doc_names_lower = {doc.get("document_name", doc.get("name", "")).lower() for doc in identified_docs}
        assignment_names_lower = {name.lower() for name in assignments.keys()}
        all_doc_names_lower = identified_doc_names_lower | assignment_names_lower

        missing_critical = []

        # Check each critical document pattern
        for doc_type, patterns in self.CRITICAL_DOCUMENT_PATTERNS.items():
            found = False
            for pattern in patterns:
                if any(pattern in doc_name for doc_name in all_doc_names_lower):
                    found = True
                    break

            if not found and doc_type in ["volume_1", "cover_letter", "price_narrative"]:
                missing_critical.append(doc_type)

        # Create missing critical documents
        for doc_type in missing_critical:
            doc_name = self._get_document_name_for_type(doc_type)
            logger.warning(f"CRITICAL DOCUMENT MISSING: {doc_name}")

            # Find requirements that should go in this document
            relevant_reqs = []
            for req_id, req in self.requirements_registry.items():
                if req.coverage_status == "unassigned":
                    for hint in req.document_type_hints:
                        if doc_type in hint.lower() or hint.lower() in doc_type:
                            relevant_reqs.append(req_id)
                            req.assigned_document = doc_name
                            req.coverage_status = "recovered"
                            break

            assignments[doc_name] = DocumentAssignment(
                document_name=doc_name,
                assigned_requirements=relevant_reqs,
                agent_recommendations=self._get_agent_for_document_type(doc_type),
                coverage_completeness=len(relevant_reqs) / max(len(self.requirements_registry), 1),
                risk_factors=["critical_document_recovery", "auto_created"]
            )

            logger.info(f"Created critical document: {doc_name} with {len(relevant_reqs)} requirements")

    def _get_document_name_for_type(self, doc_type: str) -> str:
        """Get standard document name for a document type"""
        name_map = {
            "volume_1": "Volume_I_Technical_Proposal.pdf",
            "cover_letter": "Cover_Letter.pdf",
            "price_narrative": "Price_Narrative.pdf",
            "cost_proposal": "Cost_Proposal.xlsx",
            "executive_summary": "Executive_Summary.pdf",
            "past_performance": "Past_Performance.pdf",
            "draft_sow": "Draft_SOW.docx"
        }
        return name_map.get(doc_type, f"{doc_type.replace('_', ' ').title()}.pdf")

    def _get_agent_for_document_type(self, doc_type: str) -> List[Tuple[str, float]]:
        """
        Get appropriate agent recommendations for document type.
        Returns agents that actually exist in MultiPassOrchestrator.
        """
        # Map to actual orchestrator agent keys
        agent_map = {
            # Volume documents
            "volume_1": [("volume_organizer", 0.95), ("artifact_generator", 0.70)],
            "volume_2": [("price_narrative", 0.95), ("volume_organizer", 0.85)],
            "volume_3": [("volume_organizer", 0.95), ("sow", 0.80)],
            "volume_4": [("volume_organizer", 0.90)],
            "technical_volume": [("volume_organizer", 0.95)],
            "technical_proposal": [("volume_organizer", 0.95)],

            # Executive/Cover documents
            "cover_letter": [("cover_letter", 0.98)],
            "executive_summary": [("cover_letter", 0.95)],
            "transmittal": [("cover_letter", 0.90)],

            # Pricing/Cost documents
            "price_narrative": [("price_narrative", 0.98)],
            "cost_proposal": [("price_narrative", 0.95)],
            "pricing": [("price_narrative", 0.95)],
            "cost_volume": [("price_narrative", 0.95)],
            "financial": [("price_narrative", 0.90)],

            # SOW/Project documents
            "draft_sow": [("sow", 0.95)],
            "statement_of_work": [("sow", 0.98)],
            "work_breakdown": [("sow", 0.90)],
            "project_plan": [("sow", 0.85)],

            # Past Performance
            "past_performance": [("volume_organizer", 0.85), ("artifact_generator", 0.75)],
            "experience": [("volume_organizer", 0.80)],
            "capability": [("artifact_generator", 0.85)],

            # Technical/Samples
            "technical": [("volume_organizer", 0.90), ("artifact_generator", 0.85)],
            "sample": [("artifact_generator", 0.95)],
            "artifact": [("artifact_generator", 0.95)],
            "demonstration": [("artifact_generator", 0.90)],

            # Compliance/Management
            "compliance": [("volume_organizer", 0.85)],
            "management": [("volume_organizer", 0.85), ("sow", 0.75)],
            "security": [("volume_organizer", 0.85)],
            "quality": [("sow", 0.80), ("artifact_generator", 0.70)],

            # Default fallback
            "default": [("volume_organizer", 0.70)]
        }

        # Normalize document type
        normalized = doc_type.lower().replace("-", "_").replace(" ", "_")

        # Try exact match
        if normalized in agent_map:
            return agent_map[normalized]

        # Try partial matches
        for key, agents in agent_map.items():
            if key in normalized or normalized in key:
                return agents

        # Use agent capabilities for intelligent matching
        if hasattr(self, 'agent_capabilities'):
            best_agents = []
            for agent_name, caps in self.agent_capabilities.items():
                for keyword in caps["best_for"]:
                    if keyword in normalized:
                        best_agents.append((agent_name, caps["confidence"]))
                        break
            if best_agents:
                return sorted(best_agents, key=lambda x: x[1], reverse=True)[:2]

        # Default to volume_organizer
        return [("volume_organizer", 0.70)]

    async def identify_required_documents(
        self,
        northstar_document: str,
        rfp_content: str
    ) -> List[Dict[str, Any]]:
        """
        Identify all required documents based on Northstar analysis and RFP content

        Args:
            northstar_document: Strategic guidance document
            rfp_content: Original RFP content

        Returns:
            List of required document specifications
        """
        self.logger.info("Identifying required documents from Northstar and RFP")

        try:
            prompt = f"""
            Based on the Northstar analysis and RFP content, identify ALL required documents.

            NORTHSTAR ANALYSIS (first 30k chars):
            {northstar_document[:30000]}

            RFP CONTENT (first 20k chars):
            {rfp_content[:20000]}

            Identify and list ALL required documents with these details:
            1. Document type (e.g., volume_1_technical, cover_letter, price_narrative)
            2. Document name
            3. Key requirements for the document
            4. Estimated page count
            5. Critical sections that must be included

            Return as JSON list.
            """

            try:
                logger.debug(f"Making document identification OpenAI request with model: {self.model}")
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert at identifying federal proposal requirements."},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=min(self.max_tokens, 16000),
                    # GPT-5 only supports default temperature=1
                    # Note: reasoning_effort parameter causes empty responses in some cases
                    reasoning_effort=self.reasoning_effort if "gpt-5" in self.model else None
                )
                logger.debug(f"Document identification OpenAI request completed successfully")
            except Exception as openai_error:
                logger.error(f"Document identification OpenAI API call failed: {openai_error}", exc_info=True)
                # Return default set of critical documents
                return [
                    {"type": "volume_1_technical", "name": "Volume 1 - Technical Capability", "critical": True},
                    {"type": "cover_letter", "name": "Cover Letter", "critical": True},
                    {"type": "price_narrative", "name": "Price Narrative", "critical": True},
                    {"type": "statement_of_work", "name": "Statement of Work", "critical": True}
                ]

            content = response.choices[0].message.content

            # Validate response before parsing
            logger.debug(f"Document identification response content type: {type(content)}")
            logger.debug(f"Document identification response content length: {len(content) if content else 0}")

            if not content or content.strip() == "":
                logger.error("OpenAI returned empty or None content for document identification")
                logger.error(f"Full response: {response}")
                # Use fallback extraction with empty content
                documents = self._extract_documents_from_text("")
            else:
                # Parse JSON response
                try:
                    documents = json.loads(content)
                    if not isinstance(documents, list):
                        documents = [documents]
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error in document identification: {e}")
                    logger.warning(f"Response content (first 1000 chars): {content[:1000]}")
                    # Fallback to pattern-based extraction
                    documents = self._extract_documents_from_text(content)

            # Ensure critical documents are included
            critical_docs = {
                "volume_1_technical": False,
                "cover_letter": False,
                "price_narrative": False,
                "statement_of_work": False
            }

            for doc in documents:
                doc_type = doc.get("type", "").lower()
                for critical_type in critical_docs:
                    if critical_type in doc_type:
                        critical_docs[critical_type] = True

            # Add missing critical documents
            for doc_type, found in critical_docs.items():
                if not found:
                    documents.append({
                        "type": doc_type,
                        "name": doc_type.replace("_", " ").title(),
                        "requirements": f"Standard {doc_type.replace('_', ' ')} for federal proposal",
                        "page_count": 10,
                        "critical": True
                    })

            self.logger.info(f"Identified {len(documents)} required documents")
            return documents

        except Exception as e:
            self.logger.error(f"Failed to identify documents: {str(e)}")
            # Return default set of critical documents
            return [
                {"type": "volume_1_technical", "name": "Volume 1 - Technical Capability", "critical": True},
                {"type": "cover_letter", "name": "Cover Letter", "critical": True},
                {"type": "price_narrative", "name": "Price Narrative", "critical": True},
                {"type": "statement_of_work", "name": "Statement of Work", "critical": True}
            ]

    def _extract_documents_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract document information from text response"""
        documents = []
        lines = text.split('\n')

        current_doc = {}
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ["volume", "document", "attachment", "exhibit"]):
                if current_doc:
                    documents.append(current_doc)
                current_doc = {"name": line, "type": "document", "requirements": ""}
            elif current_doc and line:
                current_doc["requirements"] += f" {line}"

        if current_doc:
            documents.append(current_doc)

        return documents if documents else [
            {"type": "general", "name": "RFP Response", "requirements": "Complete response"}
        ]

    async def validate_completeness(self) -> Dict[str, Any]:
        """
        Comprehensive validation ensuring nothing is missed
        """
        validation_report = {
            "overall_status": "unknown",
            "timestamp": datetime.now().isoformat(),
            "total_requirements": len(self.requirements_registry),
            "assigned_requirements": 0,
            "unassigned_requirements": [],
            "critical_gaps": [],
            "compliance_coverage": {"mandatory": 0, "total_mandatory": 0},
            "document_coverage": {},
            "risk_assessment": {"high": [], "medium": [], "low": []},
            "recommendations": []
        }

        # Count assigned requirements
        all_assigned = set()
        for doc_assignment in self.document_assignments.values():
            all_assigned.update(doc_assignment.assigned_requirements)

        validation_report["assigned_requirements"] = len(all_assigned)

        # Find unassigned requirements
        all_req_ids = set(self.requirements_registry.keys())
        unassigned = all_req_ids - all_assigned
        validation_report["unassigned_requirements"] = list(unassigned)

        # Check critical gaps
        critical_unassigned = [
            req_id for req_id in unassigned
            if self.requirements_registry[req_id].priority == RequirementPriority.CRITICAL
        ]
        validation_report["critical_gaps"] = critical_unassigned

        # Compliance coverage analysis
        mandatory_reqs = [req for req in self.requirements_registry.values() if req.compliance_mandatory]
        assigned_mandatory = [req_id for req_id in all_assigned
                             if req_id in self.requirements_registry and
                             self.requirements_registry[req_id].compliance_mandatory]

        validation_report["compliance_coverage"] = {
            "mandatory": len(assigned_mandatory),
            "total_mandatory": len(mandatory_reqs),
            "coverage_percentage": (len(assigned_mandatory) / max(len(mandatory_reqs), 1)) * 100
        }

        # Document coverage analysis
        for doc_name, assignment in self.document_assignments.items():
            total_weight = sum(
                self.requirements_registry[req_id].evaluation_weight or 0
                for req_id in assignment.assigned_requirements
                if req_id in self.requirements_registry
            )
            validation_report["document_coverage"][doc_name] = {
                "requirement_count": len(assignment.assigned_requirements),
                "evaluation_weight": total_weight,
                "agent": assignment.agent_recommendations[0][0] if assignment.agent_recommendations else "Unassigned",
                "risk_factors": assignment.risk_factors
            }

        # Risk assessment
        if critical_unassigned:
            validation_report["risk_assessment"]["high"].append(
                f"{len(critical_unassigned)} critical requirements unassigned"
            )

        if validation_report["compliance_coverage"]["coverage_percentage"] < 100:
            validation_report["risk_assessment"]["medium"].append(
                f"Only {validation_report['compliance_coverage']['coverage_percentage']:.1f}% mandatory compliance coverage"
            )

        # Check for critical documents
        found_critical_docs = set()
        for assignment in self.document_assignments.values():
            doc_name_lower = assignment.document_name.lower()
            for doc_type, patterns in self.CRITICAL_DOCUMENT_PATTERNS.items():
                if any(pattern in doc_name_lower for pattern in patterns):
                    found_critical_docs.add(doc_type)

        missing_critical = set(self.CRITICAL_DOCUMENT_PATTERNS.keys()) - found_critical_docs
        critical_must_have = {"volume_1", "cover_letter", "price_narrative"}
        missing_must_have = critical_must_have & missing_critical

        if missing_must_have:
            for doc in missing_must_have:
                validation_report["risk_assessment"]["high"].append(
                    f"Missing critical document type: {doc}"
                )

        # Overall status determination
        if validation_report["risk_assessment"]["high"]:
            validation_report["overall_status"] = "FAILED"
            validation_report["recommendations"].append("IMMEDIATE ACTION REQUIRED: Critical gaps identified")
        elif validation_report["risk_assessment"]["medium"]:
            validation_report["overall_status"] = "WARNING"
            validation_report["recommendations"].append("Review medium-priority issues before submission")
        elif len(unassigned) > 0:
            validation_report["overall_status"] = "WARNING"
            validation_report["recommendations"].append("Review unassigned requirements")
        else:
            validation_report["overall_status"] = "PASSED"
            validation_report["recommendations"].append("All requirements appear to be covered")

        # Add specific recommendations
        if missing_must_have:
            validation_report["recommendations"].append(
                f"Create missing critical documents: {', '.join(missing_must_have)}"
            )

        if validation_report["compliance_coverage"]["coverage_percentage"] < 100:
            validation_report["recommendations"].append(
                "Review and assign all mandatory compliance requirements"
            )

        # Log validation results
        self._log_validation_results(validation_report)

        return validation_report

    def _log_validation_results(self, report: Dict[str, Any]):
        """Log comprehensive validation results"""
        logger.info("=== REQUIREMENTS VALIDATION REPORT ===")
        logger.info(f"Overall Status: {report['overall_status']}")
        logger.info(f"Coverage: {report['assigned_requirements']}/{report['total_requirements']} requirements assigned")
        logger.info(f"Compliance: {report['compliance_coverage']['coverage_percentage']:.1f}% mandatory requirements covered")

        if report["critical_gaps"]:
            logger.error(f"CRITICAL GAPS: {len(report['critical_gaps'])} critical requirements unassigned")
            for gap in report["critical_gaps"][:3]:  # Show first 3
                if gap in self.requirements_registry:
                    logger.error(f"  - {gap}: {self.requirements_registry[gap].description[:100]}...")

        if report["risk_assessment"]["high"]:
            logger.error("HIGH RISKS IDENTIFIED:")
            for risk in report["risk_assessment"]["high"]:
                logger.error(f"  - {risk}")

    async def generate_compliance_matrix(self) -> str:
        """
        Generate automatic compliance tracking matrix
        """
        matrix_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_requirements": len(self.requirements_registry),
                "total_documents": len(self.document_assignments)
            },
            "compliance_matrix": {},
            "coverage_gaps": []
        }

        # Build matrix
        for doc_name, assignment in self.document_assignments.items():
            matrix_data["compliance_matrix"][doc_name] = {
                "assigned_agent": assignment.agent_recommendations[0][0] if assignment.agent_recommendations else "Unassigned",
                "requirements": {},
                "risk_factors": assignment.risk_factors,
                "coverage_score": assignment.coverage_completeness
            }

            for req_id in assignment.assigned_requirements:
                if req_id in self.requirements_registry:
                    req = self.requirements_registry[req_id]
                    matrix_data["compliance_matrix"][doc_name]["requirements"][req_id] = {
                        "description": req.description[:200],
                        "priority": req.priority.value,
                        "mandatory": req.compliance_mandatory,
                        "status": req.coverage_status
                    }

        # Find gaps
        for req_id, req in self.requirements_registry.items():
            if req.coverage_status == "unassigned":
                matrix_data["coverage_gaps"].append({
                    "id": req_id,
                    "description": req.description[:200],
                    "priority": req.priority.value
                })

        return json.dumps(matrix_data, indent=2)

    async def _fallback_requirement_extraction(self, northstar_document: str) -> Dict[str, Requirement]:
        """Fallback extraction using pattern matching"""
        import re

        requirements = {}
        req_id = 1

        # Look for requirement patterns
        patterns = [
            (r"(?i)volume\s+([IVX\d]+)[:\-\s]+([^.]+)", "volume"),
            (r"(?i)(must|shall|will|required?)\s+([^.]{20,200})", "mandatory"),
            (r"(?i)(deliverable|document|submission)\s*:?\s*([^.]{20,200})", "deliverable"),
            (r"(?i)cover\s+letter[:\-\s]+([^.]+)", "cover_letter"),
            (r"(?i)technical\s+(proposal|volume)[:\-\s]+([^.]+)", "technical"),
            (r"(?i)cost\s+(proposal|volume)[:\-\s]+([^.]+)", "cost"),
            (r"(?i)price\s+narrative[:\-\s]+([^.]+)", "price_narrative"),
        ]

        for pattern, doc_type in patterns:
            matches = re.findall(pattern, northstar_document[:50000], re.MULTILINE)
            for match in matches:
                desc = ' '.join(match) if isinstance(match, tuple) else match

                req = Requirement(
                    id=f"FALLBACK_{req_id:03d}",
                    description=desc.strip()[:500],
                    source_section="Pattern extraction",
                    priority=RequirementPriority.HIGH if "must" in desc.lower() or "shall" in desc.lower() else RequirementPriority.MEDIUM,
                    document_type_hints=[doc_type],
                    compliance_mandatory="must" in desc.lower() or "shall" in desc.lower(),
                    evaluation_weight=None,
                    dependencies=[],
                    confidence_score=0.6
                )
                requirements[req.id] = req
                req_id += 1

        # Add default critical requirements if too few found
        if len(requirements) < 5:
            defaults = self._get_default_critical_requirements()
            requirements.update(defaults)

        logger.info(f"Fallback extraction found {len(requirements)} requirements")
        return requirements

    def _get_default_critical_requirements(self) -> Dict[str, Requirement]:
        """Emergency default requirements for government RFPs"""
        defaults = {
            "DEFAULT_001": Requirement(
                id="DEFAULT_001",
                description="Volume I - Technical approach and solution description",
                source_section="Default critical requirements",
                priority=RequirementPriority.CRITICAL,
                document_type_hints=["technical", "volume_1"],
                compliance_mandatory=True,
                evaluation_weight=0.4,
                dependencies=[],
                confidence_score=0.3
            ),
            "DEFAULT_002": Requirement(
                id="DEFAULT_002",
                description="Cover Letter with compliance certifications",
                source_section="Default critical requirements",
                priority=RequirementPriority.CRITICAL,
                document_type_hints=["cover_letter"],
                compliance_mandatory=True,
                evaluation_weight=0.0,
                dependencies=[],
                confidence_score=0.3
            ),
            "DEFAULT_003": Requirement(
                id="DEFAULT_003",
                description="Cost proposal with detailed pricing",
                source_section="Default critical requirements",
                priority=RequirementPriority.CRITICAL,
                document_type_hints=["cost", "pricing"],
                compliance_mandatory=True,
                evaluation_weight=0.3,
                dependencies=[],
                confidence_score=0.3
            ),
            "DEFAULT_004": Requirement(
                id="DEFAULT_004",
                description="Price Narrative explaining pricing approach",
                source_section="Default critical requirements",
                priority=RequirementPriority.HIGH,
                document_type_hints=["price_narrative"],
                compliance_mandatory=True,
                evaluation_weight=0.0,
                dependencies=["DEFAULT_003"],
                confidence_score=0.3
            ),
            "DEFAULT_005": Requirement(
                id="DEFAULT_005",
                description="Past performance and relevant experience",
                source_section="Default critical requirements",
                priority=RequirementPriority.HIGH,
                document_type_hints=["past_performance"],
                compliance_mandatory=True,
                evaluation_weight=0.3,
                dependencies=[],
                confidence_score=0.3
            )
        }

        logger.warning("Using emergency default requirements - manual review required")
        return defaults

    async def _simplified_requirement_extraction(self, northstar_document: str) -> Dict[str, Requirement]:
        """
        Simplified requirement extraction for large documents to avoid timeouts
        """
        try:
            # Use only the first part of the document
            document_summary = northstar_document[:3000]

            simplified_prompt = f"""
            Extract key requirements from this document:

            {document_summary}

            Return JSON with requirements in this format:
            {{
                "requirements": [
                    {{
                        "id": "REQ_001",
                        "description": "Brief requirement description",
                        "priority": "critical",
                        "document_type_hints": ["technical", "volume_1"],
                        "compliance_mandatory": true,
                        "confidence_score": 0.8
                    }}
                ]
            }}
            """

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a requirements extraction expert. Return only valid JSON."},
                    {"role": "user", "content": simplified_prompt}
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=16000,
                reasoning_effort=self.reasoning_effort if "gpt-5" in self.model else None,
            )

            response_content = response.choices[0].message.content
            if not response_content:
                logger.error("Simplified extraction returned empty content")
                return await self._fallback_requirement_extraction(northstar_document)

            parsed_data = json.loads(response_content)
            requirements = {}

            for req_data in parsed_data.get("requirements", []):
                req = Requirement(
                    id=req_data["id"],
                    description=req_data["description"],
                    source_section="Simplified extraction",
                    priority=RequirementPriority(req_data.get("priority", "medium")),
                    document_type_hints=req_data.get("document_type_hints", []),
                    compliance_mandatory=req_data.get("compliance_mandatory", False),
                    evaluation_weight=req_data.get("evaluation_weight"),
                    dependencies=req_data.get("dependencies", []),
                    confidence_score=req_data.get("confidence_score", 0.5)
                )
                requirements[req.id] = req

            # Add some default critical requirements to ensure completeness
            if len(requirements) < 3:
                defaults = self._get_default_critical_requirements()
                requirements.update(defaults)

            logger.info(f"Simplified extraction completed with {len(requirements)} requirements")
            return requirements

        except Exception as e:
            logger.error(f"Simplified extraction failed: {e}")
            return await self._fallback_requirement_extraction(northstar_document)
