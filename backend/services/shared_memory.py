"""
Shared Memory System for Multi-Agent Collaboration
Provides persistent, scalable memory for agent coordination
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import redis.asyncio as redis
from collections import defaultdict
import hashlib
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import redis configuration
from config.redis_config import (
    RedisConfig,
    RedisConnectionManager,
    RedisOperations,
    create_redis_manager
)

logger = logging.getLogger(__name__)


class ConsistencyLevel(Enum):
    """Consistency levels for shared memory operations"""
    WEAK = "weak"  # Allow temporary inconsistencies
    STRONG = "strong"  # Enforce immediate consistency
    EVENTUAL = "eventual"  # Eventually consistent


@dataclass
class RequirementCoverage:
    """Track requirement coverage across agents"""
    requirement_id: str
    description: str
    assigned_agents: List[str]
    status: str  # UNASSIGNED, ASSIGNED, IN_PROGRESS, COMPLETED
    coverage_percentage: float
    evidence: List[str]
    confidence_score: float
    last_updated: str


@dataclass
class CrossDocumentReference:
    """Track references between documents"""
    source_document: str
    target_document: str
    reference_type: str  # citation, dependency, contradiction, support
    context: str
    created_by: str
    created_at: str
    confidence: float


@dataclass
class AgentDiscovery:
    """Track insights discovered by agents"""
    agent_id: str
    discovery_type: str  # technical, compliance, risk, opportunity
    title: str
    description: str
    evidence: List[str]
    related_requirements: List[str]
    relevance_score: float
    timestamp: str


@dataclass
class AgentCommitment:
    """Track agent commitments and dependencies"""
    agent_id: str
    commitment_type: str  # deliverable, timeline, quality
    description: str
    dependencies: List[str]
    target_date: Optional[str]
    status: str  # pending, in_progress, completed, failed
    confidence: float


class SharedMemory:
    """
    Redis-backed shared memory for multi-agent collaboration
    Provides persistent storage and real-time synchronization
    """

    def __init__(self, redis_client: Optional[redis.Redis] = None, session_id: str = None):
        self.redis = redis_client
        self.connection_manager: Optional[RedisConnectionManager] = None
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.local_cache = {}  # In-memory cache for performance
        self.cache_ttl = timedelta(seconds=60)
        self.last_cache_update = {}
        self._initialized = False
        self._initialization_lock = asyncio.Lock()

        # Key prefixes for different data types
        self.KEY_PREFIX = f"mercury:rfp:{self.session_id}"
        self.REQUIREMENT_KEY = f"{self.KEY_PREFIX}:requirements"
        self.REFERENCE_KEY = f"{self.KEY_PREFIX}:references"
        self.DISCOVERY_KEY = f"{self.KEY_PREFIX}:discoveries"
        self.COMMITMENT_KEY = f"{self.KEY_PREFIX}:commitments"
        self.CONTEXT_KEY = f"{self.KEY_PREFIX}:context"
        self.LOCK_KEY = f"{self.KEY_PREFIX}:locks"

        # Consistency rules
        self.consistency_rules = []
        self.consistency_level = ConsistencyLevel.STRONG

        # Event subscribers
        self.event_subscribers = defaultdict(list)

        logger.info(f"Initialized SharedMemory with session {self.session_id}")

    async def initialize(self):
        """Initialize connection and create indexes"""
        async with self._initialization_lock:
            if self._initialized:
                return

            if not self.redis:
                try:
                    # Use connection manager for proper pooling
                    config = RedisConfig(
                        url="redis://localhost:6380",
                        db=0,
                        max_connections=20,
                        retry_on_timeout=True,
                        health_check_interval=30,
                        socket_keepalive=True
                    )

                    self.connection_manager = RedisConnectionManager(config)
                    await self.connection_manager.initialize()
                    self.redis = await self.connection_manager.get_client()

                    # Verify connection health
                    health_status = await self.connection_manager.health_check()
                    if health_status.get("status") != "healthy":
                        raise Exception(f"Redis unhealthy: {health_status.get('error', 'Unknown error')}")

                    logger.info(f"Connected to Redis via connection manager - Health: {health_status}")
                    self._initialized = True

                except Exception as e:
                    logger.warning(f"Redis connection failed: {e}. Using in-memory only")
                    self.redis = None
                    self.connection_manager = None
                    self._initialized = True

    async def close(self):
        """Clean up connections"""
        if self.connection_manager:
            await self.connection_manager.close()
            self.connection_manager = None
            self.redis = None
            self._initialized = False
        elif self.redis:
            await self.redis.close()
            self.redis = None
            self._initialized = False

    # ==================== REQUIREMENT TRACKING ====================

    async def track_requirement(self, requirement: RequirementCoverage) -> bool:
        """Track a requirement's coverage across agents"""
        try:
            # Initialize Redis if not already connected
            if not self._initialized:
                await self.initialize()

            req_key = f"{self.REQUIREMENT_KEY}:{requirement.requirement_id}"
            req_data = asdict(requirement)
            req_data['last_updated'] = datetime.now().isoformat()

            # Store in Redis if available with proper enum handling
            if self.redis:
                # Serialize data with enum support
                serialized_data = {}
                for k, v in req_data.items():
                    if hasattr(v, 'value'):  # Handle enums
                        serialized_data[k] = str(v.value)
                    elif isinstance(v, (list, dict)):
                        serialized_data[k] = json.dumps(v)
                    else:
                        serialized_data[k] = str(v)

                await self.redis.hset(req_key, mapping=serialized_data)
                await self.redis.expire(req_key, 86400)  # 24 hour TTL

            # Update local cache
            self.local_cache[req_key] = req_data
            self.last_cache_update[req_key] = datetime.now()

            # Emit event
            await self._emit_event("requirement_updated", req_data)

            return True
        except Exception as e:
            logger.error(f"Error tracking requirement: {e}")
            return False

    async def get_requirement_coverage(self, requirement_id: str) -> Optional[RequirementCoverage]:
        """Get coverage status for a specific requirement"""
        req_key = f"{self.REQUIREMENT_KEY}:{requirement_id}"

        # Check cache first
        if self._is_cache_valid(req_key):
            data = self.local_cache[req_key]
            return RequirementCoverage(**data)

        # Fetch from Redis
        if self.redis:
            try:
                data = await self.redis.hgetall(req_key)
                if data:
                    # Parse JSON fields
                    for field in ['assigned_agents', 'evidence', 'related_requirements']:
                        if field in data and data[field]:
                            data[field] = json.loads(data[field])

                    # Convert string to float
                    data['coverage_percentage'] = float(data.get('coverage_percentage', 0))
                    data['confidence_score'] = float(data.get('confidence_score', 0))

                    # Update cache
                    self.local_cache[req_key] = data
                    self.last_cache_update[req_key] = datetime.now()

                    return RequirementCoverage(**data)
            except Exception as e:
                logger.error(f"Error fetching requirement coverage: {e}")

        return None

    async def get_all_requirements(self) -> List[RequirementCoverage]:
        """Get all tracked requirements"""
        # Initialize Redis if not already connected
        if not self._initialized:
            await self.initialize()

        requirements = []

        if self.redis:
            try:
                # Get all requirement keys
                pattern = f"{self.REQUIREMENT_KEY}:*"
                keys = await self.redis.keys(pattern)

                for key in keys:
                    # Handle both bytes and string keys (defensive programming)
                    if isinstance(key, bytes):
                        key = key.decode('utf-8')
                    req_id = key.split(":")[-1]
                    req = await self.get_requirement_coverage(req_id)
                    if req:
                        requirements.append(req)
            except Exception as e:
                logger.error(f"Error fetching all requirements: {e}")

        # Fallback to local cache
        if not requirements:
            for key, data in self.local_cache.items():
                if key.startswith(self.REQUIREMENT_KEY):
                    requirements.append(RequirementCoverage(**data))

        return requirements

    async def check_requirement_coverage(self) -> Dict[str, Any]:
        """Check overall requirement coverage status"""
        requirements = await self.get_all_requirements()

        total = len(requirements)
        completed = sum(1 for r in requirements if r.status == "COMPLETED")
        in_progress = sum(1 for r in requirements if r.status == "IN_PROGRESS")
        unassigned = sum(1 for r in requirements if r.status == "UNASSIGNED")

        coverage_percentage = (completed / max(total, 1)) * 100

        return {
            "total_requirements": total,
            "completed": completed,
            "in_progress": in_progress,
            "unassigned": unassigned,
            "coverage_percentage": coverage_percentage,
            "at_risk": [r.requirement_id for r in requirements
                       if r.status == "UNASSIGNED" and r.confidence_score > 0.8]
        }

    # ==================== CROSS-DOCUMENT REFERENCES ====================

    async def add_reference(self, reference: CrossDocumentReference) -> bool:
        """Add a cross-document reference"""
        try:
            ref_id = self._generate_reference_id(reference)
            ref_key = f"{self.REFERENCE_KEY}:{ref_id}"
            ref_data = asdict(reference)
            ref_data['created_at'] = datetime.now().isoformat()

            # Check for circular references
            if await self._check_circular_reference(reference):
                logger.warning(f"Circular reference detected: {reference.source_document} -> {reference.target_document}")
                ref_data['warning'] = "circular_reference"

            # Store in Redis
            if self.redis:
                await self.redis.hset(ref_key, mapping={
                    k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                    for k, v in ref_data.items()
                })
                await self.redis.expire(ref_key, 86400)

            # Update local cache
            self.local_cache[ref_key] = ref_data
            self.last_cache_update[ref_key] = datetime.now()

            # Emit event
            await self._emit_event("reference_added", ref_data)

            return True
        except Exception as e:
            logger.error(f"Error adding reference: {e}")
            return False

    async def get_document_references(self, document_name: str) -> List[CrossDocumentReference]:
        """Get all references for a document"""
        references = []

        if self.redis:
            try:
                pattern = f"{self.REFERENCE_KEY}:*"
                keys = await self.redis.keys(pattern)

                for key in keys:
                    data = await self.redis.hgetall(key)
                    if data and (data.get('source_document') == document_name or
                               data.get('target_document') == document_name):
                        references.append(CrossDocumentReference(**data))
            except Exception as e:
                logger.error(f"Error fetching document references: {e}")

        # Fallback to local cache
        if not references:
            for key, data in self.local_cache.items():
                if key.startswith(self.REFERENCE_KEY):
                    if (data.get('source_document') == document_name or
                        data.get('target_document') == document_name):
                        references.append(CrossDocumentReference(**data))

        return references

    async def _check_circular_reference(self, reference: CrossDocumentReference) -> bool:
        """Check for circular references"""
        visited = set()

        async def has_cycle(source: str, target: str) -> bool:
            if source in visited:
                return True
            visited.add(source)

            # Check if target references back to any visited node
            refs = await self.get_document_references(target)
            for ref in refs:
                if ref.source_document == target:
                    if ref.target_document in visited:
                        return True
                    if await has_cycle(target, ref.target_document):
                        return True

            return False

        return await has_cycle(reference.source_document, reference.target_document)

    # ==================== AGENT DISCOVERIES ====================

    async def write_discovery(self, discovery: AgentDiscovery) -> bool:
        """Record an agent discovery"""
        try:
            discovery_id = self._generate_discovery_id(discovery)
            disc_key = f"{self.DISCOVERY_KEY}:{discovery_id}"
            disc_data = asdict(discovery)
            disc_data['timestamp'] = datetime.now().isoformat()

            # Store in Redis
            if self.redis:
                await self.redis.hset(disc_key, mapping={
                    k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                    for k, v in disc_data.items()
                })
                await self.redis.expire(disc_key, 86400)

                # Add to sorted set for relevance ranking
                await self.redis.zadd(
                    f"{self.DISCOVERY_KEY}:ranked",
                    {discovery_id: discovery.relevance_score}
                )

            # Update local cache
            self.local_cache[disc_key] = disc_data
            self.last_cache_update[disc_key] = datetime.now()

            # Emit event
            await self._emit_event("discovery_added", disc_data)

            logger.info(f"Agent {discovery.agent_id} recorded discovery: {discovery.title}")
            return True
        except Exception as e:
            logger.error(f"Error writing discovery: {e}")
            return False

    async def get_discoveries(self, agent_id: Optional[str] = None,
                            discovery_type: Optional[str] = None,
                            min_relevance: float = 0.0) -> List[AgentDiscovery]:
        """Get discoveries with optional filters"""
        discoveries = []

        if self.redis:
            try:
                # Get top discoveries by relevance
                discovery_ids = await self.redis.zrange(
                    f"{self.DISCOVERY_KEY}:ranked",
                    0, -1,
                    desc=True,
                    withscores=True
                )

                for disc_id, score in discovery_ids:
                    if score < min_relevance:
                        continue

                    disc_key = f"{self.DISCOVERY_KEY}:{disc_id}"
                    data = await self.redis.hgetall(disc_key)

                    if data:
                        # Parse JSON fields
                        for field in ['evidence', 'related_requirements']:
                            if field in data and data[field]:
                                data[field] = json.loads(data[field])

                        data['relevance_score'] = float(data.get('relevance_score', 0))

                        # Apply filters
                        if agent_id and data.get('agent_id') != agent_id:
                            continue
                        if discovery_type and data.get('discovery_type') != discovery_type:
                            continue

                        discoveries.append(AgentDiscovery(**data))
            except Exception as e:
                logger.error(f"Error fetching discoveries: {e}")

        return discoveries

    # ==================== AGENT COMMITMENTS ====================

    async def make_commitment(self, commitment: AgentCommitment) -> bool:
        """Record an agent commitment"""
        try:
            commit_id = self._generate_commitment_id(commitment)
            commit_key = f"{self.COMMITMENT_KEY}:{commit_id}"
            commit_data = asdict(commitment)

            # Validate dependencies
            if commitment.dependencies:
                for dep in commitment.dependencies:
                    dep_commit = await self.get_commitment(dep)
                    if dep_commit and dep_commit.status == "failed":
                        logger.warning(f"Commitment depends on failed commitment: {dep}")
                        commit_data['warning'] = "dependency_failed"

            # Store in Redis
            if self.redis:
                await self.redis.hset(commit_key, mapping={
                    k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                    for k, v in commit_data.items()
                })
                await self.redis.expire(commit_key, 86400)

            # Update local cache
            self.local_cache[commit_key] = commit_data
            self.last_cache_update[commit_key] = datetime.now()

            logger.info(f"Agent {commitment.agent_id} made commitment: {commitment.description}")
            return True
        except Exception as e:
            logger.error(f"Error making commitment: {e}")
            return False

    async def get_commitment(self, commitment_id: str) -> Optional[AgentCommitment]:
        """Get a specific commitment"""
        commit_key = f"{self.COMMITMENT_KEY}:{commitment_id}"

        # Check cache first
        if self._is_cache_valid(commit_key):
            data = self.local_cache[commit_key]
            return AgentCommitment(**data)

        # Fetch from Redis
        if self.redis:
            try:
                data = await self.redis.hgetall(commit_key)
                if data:
                    # Parse JSON fields
                    if 'dependencies' in data and data['dependencies']:
                        data['dependencies'] = json.loads(data['dependencies'])
                    data['confidence'] = float(data.get('confidence', 0))

                    return AgentCommitment(**data)
            except Exception as e:
                logger.error(f"Error fetching commitment: {e}")

        return None

    async def get_agent_commitments(self, agent_id: str) -> List[AgentCommitment]:
        """Get all commitments for an agent"""
        commitments = []

        if self.redis:
            try:
                pattern = f"{self.COMMITMENT_KEY}:*"
                keys = await self.redis.keys(pattern)

                for key in keys:
                    data = await self.redis.hgetall(key)
                    if data and data.get('agent_id') == agent_id:
                        # Parse JSON fields
                        if 'dependencies' in data and data['dependencies']:
                            data['dependencies'] = json.loads(data['dependencies'])
                        data['confidence'] = float(data.get('confidence', 0))

                        commitments.append(AgentCommitment(**data))
            except Exception as e:
                logger.error(f"Error fetching agent commitments: {e}")

        return commitments

    # ==================== SHARED CONTEXT ====================

    async def read_shared_context(self, context_type: str = "full") -> Dict[str, Any]:
        """Read comprehensive shared context"""
        context = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "context_type": context_type
        }

        if context_type in ["full", "requirements"]:
            coverage = await self.check_requirement_coverage()
            context["requirement_coverage"] = coverage

            if context_type == "full":
                context["requirements"] = [
                    asdict(r) for r in await self.get_all_requirements()
                ]

        if context_type in ["full", "discoveries"]:
            discoveries = await self.get_discoveries(min_relevance=0.5)
            context["discoveries"] = [asdict(d) for d in discoveries[:10]]  # Top 10
            context["discovery_count"] = len(discoveries)

        if context_type in ["full", "commitments"]:
            # Get all commitments
            commitments = []
            if self.redis:
                try:
                    pattern = f"{self.COMMITMENT_KEY}:*"
                    keys = await self.redis.keys(pattern)
                    for key in keys[:20]:  # Limit to 20 for performance
                        data = await self.redis.hgetall(key)
                        if data:
                            commitments.append(data)
                except:
                    pass

            context["commitments"] = commitments
            context["commitment_count"] = len(commitments)

        # Add consistency status
        context["consistency_status"] = await self.validate_consistency()

        return context

    async def write_shared_context(self, context_data: Dict[str, Any]) -> bool:
        """Write to shared context"""
        try:
            context_key = f"{self.CONTEXT_KEY}:{context_data.get('context_type', 'general')}"
            context_data['updated_at'] = datetime.now().isoformat()

            if self.redis:
                await self.redis.hset(context_key, mapping={
                    k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                    for k, v in context_data.items()
                })
                await self.redis.expire(context_key, 3600)  # 1 hour TTL

            # Update local cache
            self.local_cache[context_key] = context_data
            self.last_cache_update[context_key] = datetime.now()

            return True
        except Exception as e:
            logger.error(f"Error writing shared context: {e}")
            return False

    # ==================== CONSISTENCY & VALIDATION ====================

    async def validate_consistency(self) -> Dict[str, Any]:
        """Validate consistency across shared memory"""
        issues = []

        # Check for orphaned requirements
        requirements = await self.get_all_requirements()
        for req in requirements:
            if req.status == "COMPLETED" and req.coverage_percentage < 100:
                issues.append({
                    "type": "inconsistent_completion",
                    "requirement_id": req.requirement_id,
                    "message": "Requirement marked complete but coverage < 100%"
                })

        # Check commitment dependencies
        if self.redis:
            try:
                pattern = f"{self.COMMITMENT_KEY}:*"
                keys = await self.redis.keys(pattern)

                for key in keys:
                    data = await self.redis.hgetall(key)
                    if data and data.get('status') == 'completed':
                        deps = json.loads(data.get('dependencies', '[]'))
                        for dep in deps:
                            dep_commit = await self.get_commitment(dep)
                            if dep_commit and dep_commit.status != 'completed':
                                issues.append({
                                    "type": "broken_dependency",
                                    "commitment_id": key.split(":")[-1],
                                    "dependency": dep,
                                    "message": "Completed commitment has incomplete dependency"
                                })
            except Exception as e:
                logger.error(f"Error validating commitments: {e}")

        # Apply consistency rules
        for rule in self.consistency_rules:
            rule_issues = await rule(self)
            issues.extend(rule_issues)

        return {
            "is_consistent": len(issues) == 0,
            "issue_count": len(issues),
            "issues": issues[:10],  # Limit to first 10 issues
            "consistency_level": self.consistency_level.value
        }

    def add_consistency_rule(self, rule_func):
        """Add a custom consistency rule"""
        self.consistency_rules.append(rule_func)

    # ==================== DISTRIBUTED LOCKING ====================

    async def acquire_lock(self, resource: str, timeout: int = 10) -> bool:
        """Acquire a distributed lock for coordinated operations"""
        if not self.redis:
            return True  # No locking in local mode

        lock_key = f"{self.LOCK_KEY}:{resource}"
        lock_id = f"{self.session_id}:{datetime.now().timestamp()}"

        try:
            # Try to acquire lock with timeout
            result = await self.redis.set(
                lock_key,
                lock_id,
                nx=True,  # Only set if not exists
                ex=timeout  # Expire after timeout seconds
            )
            return bool(result)
        except Exception as e:
            logger.error(f"Error acquiring lock: {e}")
            return False

    async def release_lock(self, resource: str) -> bool:
        """Release a distributed lock"""
        if not self.redis:
            return True

        lock_key = f"{self.LOCK_KEY}:{resource}"

        try:
            await self.redis.delete(lock_key)
            return True
        except Exception as e:
            logger.error(f"Error releasing lock: {e}")
            return False

    # ==================== HELPER METHODS ====================

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self.local_cache:
            return False

        last_update = self.last_cache_update.get(key)
        if not last_update:
            return False

        return datetime.now() - last_update < self.cache_ttl

    def _generate_reference_id(self, reference: CrossDocumentReference) -> str:
        """Generate unique ID for reference"""
        data = f"{reference.source_document}:{reference.target_document}:{reference.reference_type}"
        return hashlib.md5(data.encode()).hexdigest()[:8]

    def _generate_discovery_id(self, discovery: AgentDiscovery) -> str:
        """Generate unique ID for discovery"""
        data = f"{discovery.agent_id}:{discovery.title}:{discovery.timestamp}"
        return hashlib.md5(data.encode()).hexdigest()[:8]

    def _generate_commitment_id(self, commitment: AgentCommitment) -> str:
        """Generate unique ID for commitment"""
        data = f"{commitment.agent_id}:{commitment.description}:{datetime.now().timestamp()}"
        return hashlib.md5(data.encode()).hexdigest()[:8]

    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to subscribers"""
        for callback in self.event_subscribers[event_type]:
            try:
                await callback(data)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")

    def subscribe_to_event(self, event_type: str, callback):
        """Subscribe to memory events"""
        self.event_subscribers[event_type].append(callback)

    # ==================== CLEANUP & MAINTENANCE ====================

    async def cleanup_expired(self):
        """Clean up expired data"""
        if self.redis:
            try:
                # Redis handles TTL automatically
                # Just clear old local cache entries
                current_time = datetime.now()
                keys_to_remove = []

                for key, last_update in self.last_cache_update.items():
                    if current_time - last_update > timedelta(hours=1):
                        keys_to_remove.append(key)

                for key in keys_to_remove:
                    del self.local_cache[key]
                    del self.last_cache_update[key]

                logger.info(f"Cleaned up {len(keys_to_remove)} expired cache entries")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")

    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        stats = {
            "session_id": self.session_id,
            "cache_size": len(self.local_cache),
            "cache_memory_bytes": sum(len(str(v)) for v in self.local_cache.values()),
        }

        if self.redis:
            try:
                # Include connection health check
                if self.connection_manager:
                    health = await self.connection_manager.health_check()
                    stats["redis_health"] = health

                info = await self.redis.info("memory")
                stats["redis_memory_used"] = info.get("used_memory_human", "N/A")
                stats["redis_connected"] = True

                # Count different data types
                req_count = len(await self.redis.keys(f"{self.REQUIREMENT_KEY}:*"))
                ref_count = len(await self.redis.keys(f"{self.REFERENCE_KEY}:*"))
                disc_count = len(await self.redis.keys(f"{self.DISCOVERY_KEY}:*"))
                commit_count = len(await self.redis.keys(f"{self.COMMITMENT_KEY}:*"))

                stats["requirement_count"] = req_count
                stats["reference_count"] = ref_count
                stats["discovery_count"] = disc_count
                stats["commitment_count"] = commit_count
            except Exception as e:
                stats["redis_connected"] = False
                stats["redis_error"] = str(e)
        else:
            stats["redis_connected"] = False

        return stats


# ==================== FACTORY & HELPERS ====================

async def create_shared_memory(session_id: Optional[str] = None) -> SharedMemory:
    """Factory function to create and initialize shared memory"""
    memory = SharedMemory(session_id=session_id)
    await memory.initialize()
    return memory


class SharedMemoryMixin:
    """Mixin for agents to easily integrate shared memory"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_memory: Optional[SharedMemory] = None

    async def initialize_shared_memory(self, shared_memory: SharedMemory):
        """Initialize shared memory for this agent"""
        self.shared_memory = shared_memory
        logger.info(f"Initialized shared memory for {self.__class__.__name__}")

    async def read_shared_context(self, context_type: str = "full") -> Dict[str, Any]:
        """Read from shared memory"""
        if not self.shared_memory:
            logger.warning("Shared memory not initialized")
            return {}

        return await self.shared_memory.read_shared_context(context_type)

    async def write_discovery(self, title: str, description: str,
                             discovery_type: str = "general",
                             evidence: List[str] = None,
                             related_requirements: List[str] = None,
                             relevance_score: float = 0.5):
        """Write a discovery to shared memory"""
        if not self.shared_memory:
            logger.warning("Shared memory not initialized")
            return False

        discovery = AgentDiscovery(
            agent_id=self.__class__.__name__,
            discovery_type=discovery_type,
            title=title,
            description=description,
            evidence=evidence or [],
            related_requirements=related_requirements or [],
            relevance_score=relevance_score,
            timestamp=datetime.now().isoformat()
        )

        return await self.shared_memory.write_discovery(discovery)

    async def check_requirement_coverage(self) -> Dict[str, Any]:
        """Check requirement coverage status"""
        if not self.shared_memory:
            logger.warning("Shared memory not initialized")
            return {}

        return await self.shared_memory.check_requirement_coverage()

    async def reference_other_document(self, source: str, target: str,
                                      reference_type: str = "citation",
                                      context: str = "",
                                      confidence: float = 0.8):
        """Create a reference to another document"""
        if not self.shared_memory:
            logger.warning("Shared memory not initialized")
            return False

        reference = CrossDocumentReference(
            source_document=source,
            target_document=target,
            reference_type=reference_type,
            context=context,
            created_by=self.__class__.__name__,
            created_at=datetime.now().isoformat(),
            confidence=confidence
        )

        return await self.shared_memory.add_reference(reference)

    async def make_commitment(self, commitment_type: str, description: str,
                             dependencies: List[str] = None,
                             target_date: Optional[str] = None,
                             confidence: float = 0.8):
        """Make a commitment that other agents can depend on"""
        if not self.shared_memory:
            logger.warning("Shared memory not initialized")
            return False

        commitment = AgentCommitment(
            agent_id=self.__class__.__name__,
            commitment_type=commitment_type,
            description=description,
            dependencies=dependencies or [],
            target_date=target_date,
            status="pending",
            confidence=confidence
        )

        return await self.shared_memory.make_commitment(commitment)