"""
Multi-Pass Orchestrator for Federal Proposal Generation

Implements a 5-pass processing system for comprehensive RFP response generation:
1. Analysis Pass - Requirements analysis and distribution
2. Generation Pass - Parallel content generation by specialized agents
3. Review Pass - Cross-validation and consistency checking
4. Gap-Filling Pass - Identify and fill remaining gaps
5. Polish Pass - Final refinement and formatting

This replaces the primitive keyword-based document identification with intelligent
multi-pass processing that ensures complete requirement coverage.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
import os
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from enum import Enum

import openai
from .openai_service import OpenAIService, CIRCUIT_BREAKER

# Import core components
try:
    from ..agents.requirements_marshal import RequirementsMarshal
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from agents.requirements_marshal import RequirementsMarshal

try:
    from ..agents.specialized import (
        VolumeOrganizerAgent,
        CoverLetterAgent,
        SOWAgent,
        ArtifactGeneratorAgent,
        PriceNarrativeAgent,
        AGENT_REGISTRY
    )
except ImportError:
    from agents.specialized import (
        VolumeOrganizerAgent,
        CoverLetterAgent,
        SOWAgent,
        ArtifactGeneratorAgent,
        PriceNarrativeAgent,
        AGENT_REGISTRY
    )

try:
    from .shared_memory import SharedMemory, RequirementCoverage
except ImportError:
    from services.shared_memory import SharedMemory, RequirementCoverage

try:
    from .enhanced_rag_system import EnhancedRAGSystem
except ImportError:
    # If enhanced_rag_system doesn't exist, use a simple fallback
    class EnhancedRAGSystem:
        async def initialize(self):
            pass

logger = logging.getLogger(__name__)


class ProcessingPass(Enum):
    """Processing pass types"""
    ANALYSIS = "analysis"
    GENERATION = "generation"
    REVIEW = "review"
    GAP_FILLING = "gap_filling"
    POLISH = "polish"


@dataclass
class PassResult:
    """Result from a processing pass"""
    pass_type: ProcessingPass
    success: bool
    documents_generated: List[str] = field(default_factory=list)
    requirements_covered: List[str] = field(default_factory=list)
    gaps_identified: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    duration: float = 0.0


@dataclass
class OrchestratorResult:
    """Final result from multi-pass orchestration"""
    success: bool
    total_documents: int
    documents: Dict[str, Any] = field(default_factory=dict)
    pass_results: List[PassResult] = field(default_factory=list)
    total_requirements_covered: int = 0
    total_gaps_remaining: int = 0
    confidence_score: float = 0.0
    total_duration: float = 0.0
    errors: List[str] = field(default_factory=list)


class MultiPassOrchestrator:
    """
    Orchestrates multi-pass RFP processing

    Replaces primitive keyword matching with intelligent 5-pass processing
    that ensures comprehensive requirement coverage and high-quality output.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = openai.AsyncOpenAI(timeout=float(os.getenv("OPENAI_TIMEOUT", "180")))
        # Model and token config
        self.model = os.getenv("OPENAI_MODEL", "gpt-5")
        self.fallback_model = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o")
        self.max_completion_tokens = int(os.getenv("MAX_COMPLETION_TOKENS", "16000"))
        self.reasoning_effort = os.getenv("GPT5_REASONING_EFFORT", "medium")
        self.shared_memory = SharedMemory()

        # Initialize specialized agents first
        self.agents = {
            'volume_organizer': VolumeOrganizerAgent(),
            'cover_letter': CoverLetterAgent(),
            'sow': SOWAgent(),
            'artifact_generator': ArtifactGeneratorAgent(),
            'price_narrative': PriceNarrativeAgent()
        }

        # Pass agents registry to RequirementsMarshal
        self.requirements_marshal = RequirementsMarshal(agents_registry=self.agents)
        # Only initialize EnhancedRAGSystem if available
        try:
            self.rag_system = EnhancedRAGSystem()
        except Exception as e:
            self.logger.warning(f"EnhancedRAGSystem not available: {e}")
            self.rag_system = None

        # Pass configuration
        self.pass_config = {
            ProcessingPass.ANALYSIS: {
                "max_duration": 120,  # seconds
                "parallel": False,
                "required_confidence": 0.8
            },
            ProcessingPass.GENERATION: {
                "max_duration": 600,
                "parallel": True,
                "required_confidence": 0.75
            },
            ProcessingPass.REVIEW: {
                "max_duration": 180,
                "parallel": True,
                "required_confidence": 0.85
            },
            ProcessingPass.GAP_FILLING: {
                "max_duration": 300,
                "parallel": True,
                "required_confidence": 0.7
            },
            ProcessingPass.POLISH: {
                "max_duration": 120,
                "parallel": False,
                "required_confidence": 0.9
            }
        }

    def _select_model(self, requested_model: str) -> str:
        if "gpt-5" in (requested_model or "").lower() and not CIRCUIT_BREAKER.should_allow():
            self.logger.warning("GPT-5 circuit open; MultiPass using fallback model")
            return self.fallback_model
        return requested_model

    async def orchestrate(
        self,
        northstar_document: str,
        rfp_content: str,
        company_context: Dict[str, Any],
        existing_documents: Optional[Dict[str, Any]] = None
    ) -> OrchestratorResult:
        """
        Execute 5-pass orchestration

        Args:
            northstar_document: Strategic guidance document
            rfp_content: Original RFP content
            company_context: Company capabilities and past performance
            existing_documents: Any existing generated documents

        Returns:
            OrchestratorResult with all generated documents and metrics
        """
        start_time = time.time()
        result = OrchestratorResult(success=False, total_documents=0)

        try:
            # Initialize shared memory
            await self.shared_memory.initialize()

            # Initialize EnhancedRAGSystem if available
            if self.rag_system:
                await self.rag_system.initialize()

            self.logger.info("Starting 5-pass orchestration")

            # Prepare context
            context = {
                "northstar": northstar_document,
                "rfp": rfp_content,
                "company": company_context,
                "existing_documents": existing_documents or {},
                "timestamp": datetime.now().isoformat()
            }

            # Execute passes in sequence
            pass_functions = [
                (ProcessingPass.ANALYSIS, self._execute_analysis_pass),
                (ProcessingPass.GENERATION, self._execute_generation_pass),
                (ProcessingPass.REVIEW, self._execute_review_pass),
                (ProcessingPass.GAP_FILLING, self._execute_gap_filling_pass),
                (ProcessingPass.POLISH, self._execute_polish_pass)
            ]

            for pass_type, pass_func in pass_functions:
                self.logger.info(f"Executing {pass_type.value} pass")
                pass_result = await pass_func(context)
                result.pass_results.append(pass_result)

                if not pass_result.success:
                    self.logger.error(f"{pass_type.value} pass failed: {pass_result.errors}")
                    if pass_type in [ProcessingPass.ANALYSIS, ProcessingPass.GENERATION]:
                        # Critical passes - stop if they fail
                        result.errors.extend(pass_result.errors)
                        break
                else:
                    # Update context with results
                    self._update_context(context, pass_result)

            # Compile final results
            result = await self._compile_final_results(context, result)
            result.total_duration = time.time() - start_time
            result.success = True

            self.logger.info(f"Orchestration completed in {result.total_duration:.2f}s")
            self.logger.info(f"Generated {result.total_documents} documents")
            self.logger.info(f"Covered {result.total_requirements_covered} requirements")
            self.logger.info(f"Remaining gaps: {result.total_gaps_remaining}")
            self.logger.info(f"Overall confidence: {result.confidence_score:.2%}")

        except Exception as e:
            self.logger.error(f"Orchestration failed: {str(e)}", exc_info=True)
            result.errors.append(str(e))
            result.total_duration = time.time() - start_time

        return result

    async def _execute_analysis_pass(self, context: Dict[str, Any]) -> PassResult:
        """
        Pass 1: Analysis
        Requirements Marshal analyzes Northstar and distributes requirements
        """
        pass_result = PassResult(pass_type=ProcessingPass.ANALYSIS, success=False)
        start_time = time.time()

        try:
            # Parse requirements from Northstar
            requirements = await self.requirements_marshal.parse_northstar_requirements(
                context["northstar"]
            )
            self.logger.info(f"Identified {len(requirements)} requirements")

            # Identify required documents
            documents = await self.requirements_marshal.identify_required_documents(
                context["northstar"],
                context["rfp"]
            )
            self.logger.info(f"Identified {len(documents)} required documents")

            # Assign requirements to agents
            assignments = await self.requirements_marshal.assign_requirements_to_agents(
                documents
            )

            # Store in shared memory
            for req_id, requirement in requirements.items():
                coverage = RequirementCoverage(
                    requirement_id=req_id,
                    description=requirement.description,
                    assigned_agents=[agent_rec[0] for doc_assignment in assignments.values()
                                   for agent_rec in doc_assignment.agent_recommendations
                                   if req_id in doc_assignment.assigned_requirements],
                    status="identified",
                    coverage_percentage=0.0,
                    evidence=[],
                    confidence_score=0.0,
                    last_updated=datetime.now().isoformat()
                )
                await self.shared_memory.track_requirement(coverage)

            # Update pass result
            pass_result.success = True
            pass_result.requirements_covered = list(requirements.keys())
            pass_result.documents_generated = [doc.get("type", "unknown") for doc in documents]
            pass_result.metrics = {
                "total_requirements": len(requirements),
                "total_documents": len(documents),
                "assignments_made": len(assignments)
            }

            # Store in context for next passes
            context["requirements"] = requirements
            context["document_assignments"] = assignments
            context["identified_documents"] = documents

        except Exception as e:
            self.logger.error(f"Analysis pass failed: {str(e)}")
            pass_result.errors.append(str(e))

        pass_result.duration = time.time() - start_time
        return pass_result

    async def _execute_generation_pass(self, context: Dict[str, Any]) -> PassResult:
        """
        Pass 2: Generation
        Specialized agents generate content in parallel
        """
        pass_result = PassResult(pass_type=ProcessingPass.GENERATION, success=False)
        start_time = time.time()

        try:
            assignments = context.get("document_assignments", {})
            if not assignments:
                raise ValueError("No document assignments found")

            # Execute agents in parallel
            generation_tasks = []
            # Fix: assignments is a dict, not a list
            if isinstance(assignments, dict):
                for doc_name, assignment in assignments.items():
                    # Get the first recommended agent from the assignment
                    if hasattr(assignment, 'agent_recommendations') and assignment.agent_recommendations:
                        agent_type = assignment.agent_recommendations[0][0]  # Get agent name from tuple
                    else:
                        agent_type = "GeneralistAgent"  # fallback

                    if agent_type in self.agents:
                        agent = self.agents[agent_type]
                    elif agent_type == "GeneralistAgent":
                        # Default to volume_organizer for GeneralistAgent fallback
                        agent = self.agents['volume_organizer']
                        self.logger.info(f"Using volume_organizer for GeneralistAgent fallback on {doc_name}")
                    else:
                        # Default to volume_organizer for unknown agents
                        agent = self.agents['volume_organizer']
                        self.logger.warning(f"Unknown agent type '{agent_type}', using volume_organizer for {doc_name}")

                    # Create proper assignment dict for the agent
                    # Convert Requirement objects to dicts for JSON serialization
                    requirements = assignment.assigned_requirements if hasattr(assignment, 'assigned_requirements') else []
                    serializable_requirements = []
                    for req in requirements:
                        if hasattr(req, 'to_dict'):
                            # Requirement object with to_dict method
                            serializable_requirements.append(req.to_dict())
                        elif isinstance(req, dict):
                            # Already a dict
                            serializable_requirements.append(req)
                        else:
                            # Try to convert to string as fallback
                            serializable_requirements.append(str(req))

                    assignment_dict = {
                        "document_name": doc_name,
                        "document_type": doc_name.lower().replace(" ", "_"),
                        "requirements": serializable_requirements,
                        "agent_recommendations": assignment.agent_recommendations if hasattr(assignment, 'agent_recommendations') else [],
                        "coverage_completeness": assignment.coverage_completeness if hasattr(assignment, 'coverage_completeness') else 0.0,
                        "risk_factors": assignment.risk_factors if hasattr(assignment, 'risk_factors') else []
                    }
                    task = self._execute_agent(agent, assignment_dict, context)
                    generation_tasks.append(task)
            else:
                # Fallback for old format (shouldn't happen but just in case)
                self.logger.error(f"Unexpected assignment format: {type(assignments)}")
                raise ValueError("Document assignments in wrong format")

            # Wait for all agents to complete
            results = await asyncio.gather(*generation_tasks, return_exceptions=True)

            # Process results with better error tracking
            generated_documents = {}
            covered_requirements = set()
            errors = []
            successful_generations = 0
            failed_generations = 0

            # Map results back to document assignments
            assignment_items = list(assignments.items()) if isinstance(assignments, dict) else []

            for i, result in enumerate(results):
                # Get document name from assignment
                doc_name = assignment_items[i][0] if i < len(assignment_items) else f"Unknown_Document_{i}"

                if isinstance(result, Exception):
                    failed_generations += 1
                    error_msg = f"Document '{doc_name}' failed: {str(result)}"
                    errors.append(error_msg)
                    self.logger.error(f"Agent failed for {doc_name}: {result}")
                elif result and hasattr(result, 'success') and result.success:
                    successful_generations += 1
                    doc_type = result.document_type
                    generated_documents[doc_type] = {
                        "content": result.content,
                        "artifacts": getattr(result, 'artifacts', {}),
                        "confidence": result.confidence.to_dict() if hasattr(result, 'confidence') and result.confidence else {},
                        "metadata": getattr(result, 'metadata', {})
                    }
                    covered_requirements.update(
                        result.metadata.get("covered_requirements", []) if hasattr(result, 'metadata') else []
                    )
                    self.logger.info(f"Successfully generated {doc_type} (was {doc_name})")
                    if hasattr(result, 'confidence') and result.confidence and hasattr(result.confidence, 'overall_confidence'):
                        self.logger.info(f"Confidence: {result.confidence.overall_confidence:.2%}")
                else:
                    failed_generations += 1
                    error_msg = f"Document '{doc_name}' failed: Invalid result format"
                    errors.append(error_msg)
                    self.logger.error(f"Invalid result for {doc_name}: {result}")

            # Log summary
            self.logger.info(f"Generation pass complete: {successful_generations} succeeded, {failed_generations} failed")
            if failed_generations > 0:
                self.logger.warning(f"Generation failures: {errors}")

            # Update shared memory with coverage
            for req_id in covered_requirements:
                try:
                    coverage = await self.shared_memory.get_requirement(req_id)
                    if coverage:
                        coverage.coverage_status = "generated"
                        coverage.confidence_score = 0.75  # Base generation confidence
                        await self.shared_memory.track_requirement(coverage)
                except Exception as e:
                    self.logger.warning(f"Failed to update requirement {req_id}: {e}")

            # Store generated documents
            context["generated_documents"] = generated_documents

            # Update pass result with detailed metrics
            pass_result.success = successful_generations > 0
            pass_result.documents_generated = list(generated_documents.keys())
            pass_result.requirements_covered = list(covered_requirements)
            pass_result.errors = errors
            pass_result.metrics = {
                "documents_generated": len(generated_documents),
                "successful_generations": successful_generations,
                "failed_generations": failed_generations,
                "requirements_covered": len(covered_requirements),
                "agent_failures": len(errors),
                "success_rate": f"{(successful_generations / max(len(results), 1)) * 100:.1f}%"
            }

        except Exception as e:
            self.logger.error(f"Generation pass failed: {str(e)}")
            pass_result.errors.append(str(e))

        pass_result.duration = time.time() - start_time
        return pass_result

    async def _execute_review_pass(self, context: Dict[str, Any]) -> PassResult:
        """
        Pass 3: Review
        Cross-validation and consistency checking
        """
        pass_result = PassResult(pass_type=ProcessingPass.REVIEW, success=False)
        start_time = time.time()

        try:
            generated_docs = context.get("generated_documents", {})
            if not generated_docs:
                raise ValueError("No documents to review")

            # Perform cross-document validation
            validation_results = await self._validate_cross_references(generated_docs)

            # Check consistency
            consistency_issues = await self._check_consistency(generated_docs)

            # Identify gaps
            gaps = await self._identify_gaps(context)

            # Update documents with review feedback
            for doc_type, issues in consistency_issues.items():
                if doc_type in generated_docs:
                    if "review_notes" not in generated_docs[doc_type]["metadata"]:
                        generated_docs[doc_type]["metadata"]["review_notes"] = []
                    generated_docs[doc_type]["metadata"]["review_notes"].extend(issues)

            # Update pass result
            pass_result.success = True
            pass_result.gaps_identified = gaps
            pass_result.warnings = consistency_issues.get("warnings", [])
            pass_result.metrics = {
                "validation_score": validation_results.get("score", 0),
                "consistency_issues": len(consistency_issues),
                "gaps_found": len(gaps)
            }

            # Store review results
            context["validation_results"] = validation_results
            context["consistency_issues"] = consistency_issues
            context["identified_gaps"] = gaps

        except Exception as e:
            self.logger.error(f"Review pass failed: {str(e)}")
            pass_result.errors.append(str(e))

        pass_result.duration = time.time() - start_time
        return pass_result

    async def _execute_gap_filling_pass(self, context: Dict[str, Any]) -> PassResult:
        """
        Pass 4: Gap-Filling
        Generate content for identified gaps
        """
        pass_result = PassResult(pass_type=ProcessingPass.GAP_FILLING, success=False)
        start_time = time.time()

        try:
            gaps = context.get("identified_gaps", [])
            if not gaps:
                self.logger.info("No gaps to fill")
                pass_result.success = True
                return pass_result

            self.logger.info(f"Filling {len(gaps)} identified gaps")

            # Use GPT-5 to fill gaps directly
            filled_gaps = {}
            for gap in gaps:
                filled_content = await self._fill_gap(gap, context)
                if filled_content:
                    filled_gaps[gap] = filled_content

                    # Update relevant document
                    doc_type = self._determine_document_for_gap(gap)
                    if doc_type and doc_type in context.get("generated_documents", {}):
                        if "gap_fills" not in context["generated_documents"][doc_type]:
                            context["generated_documents"][doc_type]["gap_fills"] = []
                        context["generated_documents"][doc_type]["gap_fills"].append({
                            "gap": gap,
                            "content": filled_content
                        })

            # Update pass result
            pass_result.success = True
            pass_result.gaps_identified = [g for g in gaps if g not in filled_gaps]
            pass_result.metrics = {
                "gaps_filled": len(filled_gaps),
                "gaps_remaining": len(gaps) - len(filled_gaps)
            }

            context["filled_gaps"] = filled_gaps

        except Exception as e:
            self.logger.error(f"Gap-filling pass failed: {str(e)}")
            pass_result.errors.append(str(e))

        pass_result.duration = time.time() - start_time
        return pass_result

    async def _execute_polish_pass(self, context: Dict[str, Any]) -> PassResult:
        """
        Pass 5: Polish
        Final refinement and formatting
        """
        pass_result = PassResult(pass_type=ProcessingPass.POLISH, success=False)
        start_time = time.time()

        try:
            generated_docs = context.get("generated_documents", {})
            if not generated_docs:
                raise ValueError("No documents to polish")

            polished_docs = {}
            for doc_type, doc_data in generated_docs.items():
                # Polish content
                polished_content = await self._polish_document(
                    doc_type,
                    doc_data,
                    context
                )

                if polished_content:
                    polished_docs[doc_type] = {
                        **doc_data,
                        "content": polished_content,
                        "polished": True
                    }

            # Update context
            context["final_documents"] = polished_docs

            # Final validation
            final_validation = await self.requirements_marshal.validate_completeness()

            # Update pass result
            pass_result.success = True
            pass_result.documents_generated = list(polished_docs.keys())
            pass_result.metrics = {
                "documents_polished": len(polished_docs),
                "final_validation_score": final_validation.get("score", 0),
                "ready_for_submission": final_validation.get("ready", False)
            }

        except Exception as e:
            self.logger.error(f"Polish pass failed: {str(e)}")
            pass_result.errors.append(str(e))

        pass_result.duration = time.time() - start_time
        return pass_result

    async def _execute_agent(
        self,
        agent,
        assignment: Dict,
        context: Dict[str, Any]
    ) -> Any:
        """Execute a specialized agent"""
        try:
            # Prepare agent context
            # Convert rfp_requirements to JSON-serializable structures
            rfp_requirements = context.get("requirements", {})
            if isinstance(rfp_requirements, list):
                # List of Requirement objects or dicts
                serialized_rfp_requirements = []
                for req in rfp_requirements:
                    if hasattr(req, 'to_dict'):
                        serialized_rfp_requirements.append(req.to_dict())
                    elif isinstance(req, dict):
                        serialized_rfp_requirements.append(req)
                    else:
                        serialized_rfp_requirements.append(str(req))
            elif isinstance(rfp_requirements, dict):
                # Dict of id -> Requirement or id -> dict
                serialized_rfp_requirements = {}
                for k, v in rfp_requirements.items():
                    if hasattr(v, 'to_dict'):
                        serialized_rfp_requirements[k] = v.to_dict()
                    elif isinstance(v, dict):
                        serialized_rfp_requirements[k] = v
                    else:
                        serialized_rfp_requirements[k] = str(v)
            else:
                serialized_rfp_requirements = {}

            agent_context = {
                "northstar": context["northstar"],
                "rfp": context["rfp"],
                "company_context": context["company"],
                "requirements": assignment.get("requirements", []),  # Already serialized
                "document_type": assignment.get("document_type"),
                "shared_memory": await self.shared_memory.read_shared_context(),
                # Add fields expected by agents
                "generated_documents": context.get("generated_documents", {}),
                "rfp_requirements": serialized_rfp_requirements
            }

            # Execute agent
            result = await agent.execute(agent_context)

            # Validate result format
            if result and hasattr(result, 'success'):
                return result
            else:
                self.logger.warning(f"Agent returned invalid result format: {type(result)}")
                # Return a mock failed result
                from types import SimpleNamespace
                return SimpleNamespace(
                    success=False,
                    document_type=assignment.get("document_type", "unknown"),
                    content="Agent execution failed - invalid result format",
                    confidence=SimpleNamespace(overall_confidence=0.0, to_dict=lambda: {}),
                    metadata={"error": "Invalid result format"}
                )

        except Exception as e:
            self.logger.error(f"Agent execution failed: {str(e)}")
            # Return a mock failed result instead of the exception
            from types import SimpleNamespace
            return SimpleNamespace(
                success=False,
                document_type=assignment.get("document_type", "unknown"),
                content=f"Agent execution failed: {str(e)}",
                confidence=SimpleNamespace(overall_confidence=0.0, to_dict=lambda: {}),
                metadata={"error": str(e)}
            )

    async def _validate_cross_references(self, documents: Dict) -> Dict:
        """Validate cross-references between documents"""
        validation_results = {
            "score": 0.85,  # Default score
            "issues": [],
            "validated_references": []
        }

        # Check for cross-references
        for doc_type, doc_data in documents.items():
            content = doc_data.get("content", "")

            # Look for references to other documents
            for other_doc in documents:
                if other_doc != doc_type and other_doc in content:
                    validation_results["validated_references"].append({
                        "from": doc_type,
                        "to": other_doc,
                        "valid": True
                    })

        return validation_results

    async def _check_consistency(self, documents: Dict) -> Dict:
        """Check consistency across documents"""
        issues = defaultdict(list)

        # Check for consistent terminology
        key_terms = {}
        for doc_type, doc_data in documents.items():
            content = doc_data.get("content", "").lower()

            # Extract key terms (simplified)
            if "price" in content:
                if "pricing" not in key_terms:
                    key_terms["pricing"] = doc_type
                elif key_terms["pricing"] != doc_type:
                    issues["warnings"].append(
                        f"Pricing terminology inconsistent between {key_terms['pricing']} and {doc_type}"
                    )

        return dict(issues)

    async def _identify_gaps(self, context: Dict[str, Any]) -> List[str]:
        """Identify remaining gaps in coverage"""
        gaps = []

        # Check requirement coverage
        shared_context = await self.shared_memory.read_shared_context()
        requirements = shared_context.get("requirements", [])

        for req in requirements:
            if req.get("coverage_status") != "generated":
                gaps.append(f"Requirement not covered: {req.get('requirement_id')}")

        # Check for missing critical documents
        critical_docs = [
            "volume_1_technical",
            "cover_letter",
            "price_narrative",
            "statement_of_work"
        ]

        generated = context.get("generated_documents", {})
        for critical_doc in critical_docs:
            if critical_doc not in generated:
                gaps.append(f"Missing critical document: {critical_doc}")

        return gaps

    async def _fill_gap(self, gap: str, context: Dict[str, Any]) -> Optional[str]:
        """Fill a specific gap using GPT-5"""
        try:
            prompt = f"""
            Fill the following gap in the RFP response:
            Gap: {gap}

            Context:
            - RFP: {context['rfp'][:2000]}
            - Northstar guidance: {context['northstar'][:2000]}

            Generate appropriate content to fill this gap.
            """

            effective_model = self._select_model(self.model)
            response = await self.client.chat.completions.create(
                model=effective_model,
                messages=[
                    {"role": "system", "content": "You are an expert federal proposal writer."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=self.max_completion_tokens,
                # Gap-filling benefits from deliberate but not maximal reasoning
                reasoning_effort="medium" if "gpt-5" in effective_model.lower() else None
            )

            if "gpt-5" in effective_model.lower():
                try:
                    CIRCUIT_BREAKER.record_success()
                except Exception:
                    pass
            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"Failed to fill gap '{gap}': {str(e)}")
            return None

    def _determine_document_for_gap(self, gap: str) -> Optional[str]:
        """Determine which document a gap belongs to"""
        gap_lower = gap.lower()

        if "volume 1" in gap_lower or "technical" in gap_lower:
            return "volume_1_technical"
        elif "cover letter" in gap_lower:
            return "cover_letter"
        elif "price" in gap_lower:
            return "price_narrative"
        elif "sow" in gap_lower or "statement of work" in gap_lower:
            return "statement_of_work"

        return None

    async def _polish_document(
        self,
        doc_type: str,
        doc_data: Dict,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Polish and format a document"""
        try:
            content = doc_data.get("content", "")
            gap_fills = doc_data.get("gap_fills", [])

            # Incorporate gap fills
            if gap_fills:
                for gap_fill in gap_fills:
                    content += f"\n\n{gap_fill['content']}"

            # Polish with GPT-5
            prompt = f"""
            Polish and format the following {doc_type} document for federal submission:

            {content[:10000]}

            Requirements:
            1. Ensure professional federal contracting language
            2. Verify FAR/DFARS compliance statements
            3. Format according to federal proposal standards
            4. Ensure consistency and clarity
            5. Add any missing standard sections

            Return the polished document.
            """

            effective_model = self._select_model(self.model)
            response = await self.client.chat.completions.create(
                model=effective_model,
                messages=[
                    {"role": "system", "content": "You are an expert federal proposal editor."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=self.max_completion_tokens,
                # Polishing should use highest reasoning for quality
                reasoning_effort="high" if "gpt-5" in effective_model.lower() else None
            )

            if "gpt-5" in effective_model.lower():
                try:
                    CIRCUIT_BREAKER.record_success()
                except Exception:
                    pass
            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"Failed to polish {doc_type}: {str(e)}")
            return None

    def _update_context(self, context: Dict, pass_result: PassResult):
        """Update context with pass results"""
        pass_name = pass_result.pass_type.value
        context[f"{pass_name}_result"] = {
            "success": pass_result.success,
            "documents": pass_result.documents_generated,
            "requirements": pass_result.requirements_covered,
            "gaps": pass_result.gaps_identified,
            "metrics": pass_result.metrics
        }

    async def _compile_final_results(
        self,
        context: Dict[str, Any],
        result: OrchestratorResult
    ) -> OrchestratorResult:
        """Compile final orchestration results"""
        # Get final documents
        final_docs = context.get("final_documents", context.get("generated_documents", {}))
        result.documents = final_docs
        result.total_documents = len(final_docs)

        # Calculate coverage
        shared_context = await self.shared_memory.read_shared_context()
        all_requirements = shared_context.get("requirements", [])
        covered_requirements = [
            r for r in all_requirements
            if r.get("coverage_status") in ["generated", "verified"]
        ]
        result.total_requirements_covered = len(covered_requirements)

        # Calculate remaining gaps
        gaps = context.get("identified_gaps", [])
        filled_gaps = context.get("filled_gaps", {})
        result.total_gaps_remaining = len(gaps) - len(filled_gaps)

        # Calculate overall confidence
        confidences = []
        for doc_data in final_docs.values():
            if "confidence" in doc_data and "overall_confidence" in doc_data["confidence"]:
                confidences.append(doc_data["confidence"]["overall_confidence"])

        result.confidence_score = sum(confidences) / len(confidences) if confidences else 0.0

        return result
