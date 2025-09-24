"""
Architecture Manager for Mercury RFP System
Handles mode selection between Classic (monolithic) and Dynamic (multi-agent) architectures
"""

from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pydantic import BaseModel
import logging
import asyncio
from datetime import datetime
import json
import random
from pathlib import Path

logger = logging.getLogger(__name__)

class ProcessingMode(str, Enum):
    CLASSIC = "classic"
    DYNAMIC = "dynamic"
    COMPARISON = "comparison"
    AB_TEST = "ab_test"

class ArchitectureConfig(BaseModel):
    mode: ProcessingMode
    session_id: Optional[str] = None
    user_preference: bool = False
    ab_test_group: Optional[str] = None
    classic_config: Optional[Dict[str, Any]] = None
    dynamic_config: Optional[Dict[str, Any]] = None
    comparison_config: Optional[Dict[str, Any]] = None

class ProcessingResult(BaseModel):
    notice_id: str
    architecture_used: str
    documents: Dict[str, str]
    metrics: Dict[str, Any]
    processing_time: float
    quality_scores: Dict[str, float]
    errors: List[str] = []
    uniqueness_score: float = 0.0
    agent_assignments: Optional[Dict[str, str]] = None

class ArchitectureManager:
    def __init__(self, orchestration_agent=None, multi_agent_orchestrator=None):
        self.session_configs: Dict[str, ArchitectureConfig] = {}
        self.processing_results: Dict[str, List[ProcessingResult]] = {}
        self.ab_test_config = self._load_ab_test_config()
        self.orchestration_agent = orchestration_agent
        self.multi_agent_orchestrator = multi_agent_orchestrator
        self.metrics_file = Path("data/architecture_metrics.jsonl")
        self.metrics_file.parent.mkdir(exist_ok=True)

    def _load_ab_test_config(self) -> Dict[str, Any]:
        """Load A/B testing configuration"""
        return {
            'enabled': False,
            'rollout_percentage': 20,
            'control_group_size': 50,
            'seed': 42
        }

    async def get_architecture_config(self, session_id: str, user_preference: Optional[ProcessingMode] = None) -> ArchitectureConfig:
        """Get architecture configuration for a session"""
        if session_id in self.session_configs:
            # Update existing config if user preference changes
            if user_preference:
                self.session_configs[session_id].mode = user_preference
                self.session_configs[session_id].user_preference = True
            return self.session_configs[session_id]

        # Determine mode based on priority: user preference > A/B test > default
        if user_preference:
            mode = user_preference
            logger.info(f"Using user-selected mode: {mode}")
        elif self.ab_test_config['enabled']:
            mode = self._assign_ab_test_group(session_id)
            logger.info(f"Assigned A/B test mode: {mode}")
        else:
            mode = ProcessingMode.CLASSIC
            logger.info(f"Using default mode: {mode}")

        config = ArchitectureConfig(
            mode=mode,
            session_id=session_id,
            user_preference=bool(user_preference)
        )

        self.session_configs[session_id] = config
        return config

    def _assign_ab_test_group(self, session_id: str) -> ProcessingMode:
        """Assign A/B test group based on session ID"""
        random.seed(hash(session_id) % 1000000)
        if random.random() * 100 < self.ab_test_config['rollout_percentage']:
            return ProcessingMode.DYNAMIC
        return ProcessingMode.CLASSIC

    async def process_rfp_with_architecture(
        self,
        rfp_data: Dict[str, Any],
        config: ArchitectureConfig,
        northstar_doc: str,
        rfp_documents: Dict[str, Any],
        company_context: str
    ) -> Union[ProcessingResult, Dict[str, ProcessingResult]]:
        """Process RFP using specified architecture"""
        notice_id = rfp_data.get('notice_id', 'unknown')

        logger.info(f"Processing RFP {notice_id} with architecture: {config.mode}")

        if config.mode == ProcessingMode.CLASSIC:
            return await self._process_classic(rfp_data, config, northstar_doc, rfp_documents, company_context)
        elif config.mode == ProcessingMode.DYNAMIC:
            return await self._process_dynamic(rfp_data, config, northstar_doc, rfp_documents, company_context)
        elif config.mode == ProcessingMode.COMPARISON:
            return await self._process_comparison(rfp_data, config, northstar_doc, rfp_documents, company_context)
        elif config.mode == ProcessingMode.AB_TEST:
            # A/B test mode randomly assigns to classic or dynamic
            test_mode = self._assign_ab_test_group(notice_id)
            if test_mode == ProcessingMode.DYNAMIC:
                return await self._process_dynamic(rfp_data, config, northstar_doc, rfp_documents, company_context)
            else:
                return await self._process_classic(rfp_data, config, northstar_doc, rfp_documents, company_context)
        else:
            raise ValueError(f"Unsupported processing mode: {config.mode}")

    async def _process_classic(
        self,
        rfp_data: Dict[str, Any],
        config: ArchitectureConfig,
        northstar_doc: str,
        rfp_documents: Dict[str, Any],
        company_context: str
    ) -> ProcessingResult:
        """Process using classic monolithic orchestration agent"""
        start_time = datetime.utcnow()
        notice_id = rfp_data.get('notice_id', 'unknown')
        errors = []

        try:
            # Use existing orchestration agent
            if not self.orchestration_agent:
                raise ValueError("Classic orchestration agent not available")

            # Generate RFP response using classic approach
            generated_documents = await self.orchestration_agent.generate_rfp_response(
                northstar_doc, rfp_documents, notice_id
            )

            # Calculate metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()

            # Calculate quality scores (simplified for now)
            quality_scores = {}
            for doc_name in generated_documents:
                # Basic quality scoring based on document completeness
                content = generated_documents[doc_name]
                if content and len(content) > 1000:
                    quality_scores[doc_name] = min(95, 70 + len(content) / 1000)
                else:
                    quality_scores[doc_name] = 50

            result = ProcessingResult(
                notice_id=notice_id,
                architecture_used="classic",
                documents=generated_documents,
                metrics={
                    "total_processing_time": processing_time,
                    "documents_generated": len(generated_documents),
                    "average_quality": sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0
                },
                processing_time=processing_time,
                quality_scores=quality_scores,
                errors=errors
            )

            # Save metrics
            await self._save_metrics(result)

            return result

        except Exception as e:
            logger.error(f"Error in classic processing: {e}")
            errors.append(str(e))

            return ProcessingResult(
                notice_id=notice_id,
                architecture_used="classic",
                documents={},
                metrics={"error": str(e)},
                processing_time=(datetime.utcnow() - start_time).total_seconds(),
                quality_scores={},
                errors=errors
            )

    async def _process_dynamic(
        self,
        rfp_data: Dict[str, Any],
        config: ArchitectureConfig,
        northstar_doc: str,
        rfp_documents: Dict[str, Any],
        company_context: str
    ) -> ProcessingResult:
        """Process using dynamic multi-agent system"""
        start_time = datetime.utcnow()
        notice_id = rfp_data.get('notice_id', 'unknown')
        errors = []

        try:
            # Use multi-agent orchestrator
            if not self.multi_agent_orchestrator:
                # Fallback to classic if multi-agent not available
                logger.warning("Multi-agent orchestrator not available, falling back to classic")
                return await self._process_classic(rfp_data, config, northstar_doc, rfp_documents, company_context)

            # Generate RFP response using multi-agent approach
            result = await self.multi_agent_orchestrator.generate_rfp_response(
                northstar=northstar_doc,
                rfp_documents=rfp_documents,
                notice_id=notice_id,
                company_context=company_context,
                strategy=config.dynamic_config.get('coordination_strategy', 'hybrid') if config.dynamic_config else 'hybrid'
            )

            generated_documents = result['documents']
            agent_assignments = result.get('agent_assignments', {})

            # Calculate metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()

            # Calculate quality scores with agent-specific weighting
            quality_scores = {}
            for doc_name in generated_documents:
                content = generated_documents[doc_name]
                # Higher base quality for specialized agents
                if content and len(content) > 1000:
                    quality_scores[doc_name] = min(98, 80 + len(content) / 800)
                else:
                    quality_scores[doc_name] = 60

            # Calculate uniqueness score
            uniqueness_score = await self._calculate_uniqueness_score(generated_documents)

            result = ProcessingResult(
                notice_id=notice_id,
                architecture_used="dynamic",
                documents=generated_documents,
                metrics={
                    "total_processing_time": processing_time,
                    "documents_generated": len(generated_documents),
                    "average_quality": sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0,
                    "agents_used": len(set(agent_assignments.values())) if agent_assignments else 0,
                    "parallel_processing": True
                },
                processing_time=processing_time,
                quality_scores=quality_scores,
                errors=errors,
                uniqueness_score=uniqueness_score,
                agent_assignments=agent_assignments
            )

            # Save metrics
            await self._save_metrics(result)

            return result

        except Exception as e:
            logger.error(f"Error in dynamic processing: {e}")
            errors.append(str(e))

            # Try fallback to classic
            logger.info("Attempting fallback to classic processing")
            return await self._process_classic(rfp_data, config, northstar_doc, rfp_documents, company_context)

    async def _process_comparison(
        self,
        rfp_data: Dict[str, Any],
        config: ArchitectureConfig,
        northstar_doc: str,
        rfp_documents: Dict[str, Any],
        company_context: str
    ) -> Dict[str, ProcessingResult]:
        """Run both architectures in parallel for comparison"""
        logger.info(f"Running comparison mode for RFP {rfp_data.get('notice_id', 'unknown')}")

        # Run both architectures in parallel
        classic_task = self._process_classic(rfp_data, config, northstar_doc, rfp_documents, company_context)
        dynamic_task = self._process_dynamic(rfp_data, config, northstar_doc, rfp_documents, company_context)

        classic_result, dynamic_result = await asyncio.gather(
            classic_task, dynamic_task, return_exceptions=True
        )

        # Handle exceptions
        if isinstance(classic_result, Exception):
            logger.error(f"Classic processing failed: {classic_result}")
            classic_result = self._error_result(rfp_data, "classic", classic_result)

        if isinstance(dynamic_result, Exception):
            logger.error(f"Dynamic processing failed: {dynamic_result}")
            dynamic_result = self._error_result(rfp_data, "dynamic", dynamic_result)

        # Save comparison metrics
        comparison_metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "notice_id": rfp_data.get('notice_id', 'unknown'),
            "classic_time": classic_result.processing_time,
            "dynamic_time": dynamic_result.processing_time,
            "classic_quality": classic_result.metrics.get('average_quality', 0),
            "dynamic_quality": dynamic_result.metrics.get('average_quality', 0),
            "time_improvement": (classic_result.processing_time - dynamic_result.processing_time) / classic_result.processing_time * 100 if classic_result.processing_time > 0 else 0,
            "quality_improvement": (dynamic_result.metrics.get('average_quality', 0) - classic_result.metrics.get('average_quality', 0))
        }

        await self._save_comparison_metrics(comparison_metrics)

        return {
            "classic": classic_result,
            "dynamic": dynamic_result,
            "comparison": comparison_metrics
        }

    def _error_result(self, rfp_data: Dict[str, Any], architecture: str, error: Exception) -> ProcessingResult:
        """Create error result for failed processing"""
        return ProcessingResult(
            notice_id=rfp_data.get('notice_id', 'unknown'),
            architecture_used=architecture,
            documents={},
            metrics={"error": str(error)},
            processing_time=0,
            quality_scores={},
            errors=[str(error)]
        )

    async def _calculate_uniqueness_score(self, documents: Dict[str, str]) -> float:
        """Calculate uniqueness score for generated documents"""
        # Simplified uniqueness calculation
        # In production, this would compare against historical documents
        total_length = sum(len(doc) for doc in documents.values())
        unique_words = set()

        for content in documents.values():
            words = content.lower().split()
            unique_words.update(words)

        if total_length > 0:
            # Higher ratio of unique words = higher uniqueness
            uniqueness = len(unique_words) / (total_length / 10)  # Normalize by average word length
            return min(1.0, uniqueness)
        return 0.0

    async def _save_metrics(self, result: ProcessingResult):
        """Save processing metrics to file"""
        try:
            with open(self.metrics_file, 'a') as f:
                metrics_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "notice_id": result.notice_id,
                    "architecture": result.architecture_used,
                    "processing_time": result.processing_time,
                    "documents_generated": len(result.documents),
                    "average_quality": sum(result.quality_scores.values()) / len(result.quality_scores) if result.quality_scores else 0,
                    "uniqueness_score": result.uniqueness_score,
                    "errors": result.errors
                }
                f.write(json.dumps(metrics_data) + '\n')
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    async def _save_comparison_metrics(self, metrics: Dict[str, Any]):
        """Save comparison metrics"""
        try:
            comparison_file = Path("data/comparison_metrics.jsonl")
            comparison_file.parent.mkdir(exist_ok=True)
            with open(comparison_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
        except Exception as e:
            logger.error(f"Failed to save comparison metrics: {e}")

    async def get_architecture_metrics(self, notice_id: Optional[str] = None) -> Dict[str, Any]:
        """Get processing metrics for architecture comparison"""
        if notice_id:
            return {
                "notice_id": notice_id,
                "results": self.processing_results.get(notice_id, [])
            }

        # Return aggregated metrics
        all_metrics = []
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                for line in f:
                    try:
                        all_metrics.append(json.loads(line))
                    except:
                        continue

        # Calculate aggregates
        classic_metrics = [m for m in all_metrics if m.get('architecture') == 'classic']
        dynamic_metrics = [m for m in all_metrics if m.get('architecture') == 'dynamic']

        return {
            "total_processed": len(all_metrics),
            "classic_count": len(classic_metrics),
            "dynamic_count": len(dynamic_metrics),
            "classic_avg_time": sum(m.get('processing_time', 0) for m in classic_metrics) / len(classic_metrics) if classic_metrics else 0,
            "dynamic_avg_time": sum(m.get('processing_time', 0) for m in dynamic_metrics) / len(dynamic_metrics) if dynamic_metrics else 0,
            "classic_avg_quality": sum(m.get('average_quality', 0) for m in classic_metrics) / len(classic_metrics) if classic_metrics else 0,
            "dynamic_avg_quality": sum(m.get('average_quality', 0) for m in dynamic_metrics) / len(dynamic_metrics) if dynamic_metrics else 0
        }

    def update_ab_test_config(self, enabled: bool, rollout_percentage: int = 20):
        """Update A/B testing configuration"""
        self.ab_test_config['enabled'] = enabled
        self.ab_test_config['rollout_percentage'] = rollout_percentage
        logger.info(f"A/B testing {'enabled' if enabled else 'disabled'} with {rollout_percentage}% rollout")