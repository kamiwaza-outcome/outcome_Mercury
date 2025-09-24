"""
Mission Control: Real-time logging and AI-powered monitoring system
"""

import os
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, AsyncGenerator
from collections import deque
from enum import Enum
import logging
from dataclasses import dataclass, asdict
from openai import AsyncOpenAI
import redis.asyncio as redis
import hashlib

logger = logging.getLogger(__name__)

class LogLevel(Enum):
    """Story-based log levels for narrative mode"""
    EPIC = "epic"       # Major phases
    CHAPTER = "chapter" # Sub-phases
    SCENE = "scene"     # Individual operations
    DETAIL = "detail"   # Debug info
    ERROR = "error"     # Errors
    WARNING = "warning" # Warnings
    SUCCESS = "success" # Success events

class EventType(Enum):
    """Types of events in the system"""
    SYSTEM_START = "system_start"
    PHASE_CHANGE = "phase_change"
    AGENT_ACTION = "agent_action"
    API_CALL = "api_call"
    DOCUMENT_EVENT = "document_event"
    ERROR_OCCURRED = "error_occurred"
    USER_ACTION = "user_action"
    COMPLETION = "completion"

@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: str
    level: LogLevel
    event_type: EventType
    service: str
    message: str
    metadata: Dict[str, Any]
    session_id: str
    correlation_id: str
    phase: Optional[str] = None
    confidence: Optional[float] = None

    def to_dict(self) -> Dict:
        data = asdict(self)
        data['level'] = self.level.value
        data['event_type'] = self.event_type.value
        return data

class LogBuffer:
    """Intelligent log buffering with memory management"""

    def __init__(self, max_size: int = 1000, max_memory_mb: int = 50):
        self.buffer = deque(maxlen=max_size)
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_size = 0
        self.statistics = {
            'total_logs': 0,
            'dropped_logs': 0,
            'error_count': 0,
            'warning_count': 0
        }

    def add(self, entry: LogEntry) -> bool:
        """Add log entry with memory management"""
        entry_size = len(json.dumps(entry.to_dict()).encode('utf-8'))

        # Check memory limit
        while self.current_size + entry_size > self.max_memory_bytes and self.buffer:
            old_entry = self.buffer.popleft()
            old_size = len(json.dumps(old_entry.to_dict()).encode('utf-8'))
            self.current_size -= old_size
            self.statistics['dropped_logs'] += 1

        self.buffer.append(entry)
        self.current_size += entry_size
        self.statistics['total_logs'] += 1

        # Track error/warning counts
        if entry.level == LogLevel.ERROR:
            self.statistics['error_count'] += 1
        elif entry.level == LogLevel.WARNING:
            self.statistics['warning_count'] += 1

        return True

    def get_recent(self, count: int = 100) -> List[LogEntry]:
        """Get most recent log entries"""
        return list(self.buffer)[-count:]

    def get_by_level(self, level: LogLevel, count: int = 50) -> List[LogEntry]:
        """Get logs by level"""
        return [e for e in list(self.buffer) if e.level == level][-count:]

    def clear_old_entries(self, age_seconds: int = 3600):
        """Remove entries older than specified age"""
        current_time = datetime.now()
        self.buffer = deque(
            (e for e in self.buffer
             if (current_time - datetime.fromisoformat(e.timestamp)).total_seconds() < age_seconds),
            maxlen=self.buffer.maxlen
        )

class AIStreamAnalyzer:
    """AI-powered log analysis with streaming capabilities"""

    def __init__(self, api_key: str = None):
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), timeout=float(os.getenv("OPENAI_TIMEOUT", "180")))
        self.context_window = deque(maxlen=100)  # Keep last 100 logs for context
        self.analysis_interval = 10  # Analyze every N logs
        self.log_counter = 0
        self.current_phase = "initialization"
        self.session_summary = ""
        self.patterns_detected = []
        self.model = "gpt-5-mini"  # Using GPT-5 mini for speed as requested

    async def analyze_stream(self, log_entries: List[LogEntry]) -> AsyncGenerator[str, None]:
        """Stream AI analysis of log entries"""

        # Add to context window
        for entry in log_entries:
            self.context_window.append(entry)
            self.log_counter += 1

        # Only analyze at intervals or on errors
        should_analyze = (
            self.log_counter % self.analysis_interval == 0 or
            any(e.level == LogLevel.ERROR for e in log_entries) or
            any(e.event_type == EventType.PHASE_CHANGE for e in log_entries)
        )

        if not should_analyze and len(log_entries) < 5:
            return

        # Prepare context for AI
        recent_logs = list(self.context_window)[-20:]  # Last 20 logs

        # Build the analysis prompt
        prompt = self._build_analysis_prompt(recent_logs, log_entries)

        try:
            # Stream response from GPT-5
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
                temperature=0.3,  # Lower temperature for consistent analysis
                max_tokens=500
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            yield f"⚠️ Analysis temporarily unavailable: {str(e)}"

    def _get_system_prompt(self) -> str:
        """System prompt for the AI analyzer"""
        return """You are Mission Control AI, monitoring an RFP generation system in real-time.
        Your role is to:
        1. Translate technical logs into clear, human-friendly explanations
        2. Identify patterns, issues, and opportunities for optimization
        3. Provide predictive insights based on current progress
        4. Alert on critical issues that need attention
        5. Maintain a narrative flow that tells the story of the RFP generation

        Be concise but insightful. Use emojis sparingly for important alerts.
        Focus on what matters to the user - progress, problems, and predictions."""

    def _build_analysis_prompt(self, recent_logs: List[LogEntry], new_logs: List[LogEntry]) -> str:
        """Build the analysis prompt with context"""

        # Format recent context
        context_summary = f"""
        Current Phase: {self.current_phase}
        Session Summary: {self.session_summary}
        Error Count: {sum(1 for l in recent_logs if l.level == LogLevel.ERROR)}
        Time Elapsed: {self._calculate_elapsed_time(recent_logs)}
        """

        # Format new logs for analysis
        new_logs_formatted = "\n".join([
            f"[{l.level.value}] {l.service}: {l.message}"
            for l in new_logs[-10:]  # Last 10 new logs
        ])

        return f"""{context_summary}

        New Events:
        {new_logs_formatted}

        Analyze these events and provide:
        1. What's happening now (in plain English)
        2. Any concerns or unusual patterns
        3. Predicted next steps
        4. Overall confidence in the process

        Keep response under 3 sentences unless there's a critical issue."""

    def _calculate_elapsed_time(self, logs: List[LogEntry]) -> str:
        """Calculate elapsed time from logs"""
        if not logs:
            return "0 seconds"

        try:
            start = datetime.fromisoformat(logs[0].timestamp)
            end = datetime.fromisoformat(logs[-1].timestamp)
            elapsed = (end - start).total_seconds()

            if elapsed < 60:
                return f"{int(elapsed)} seconds"
            elif elapsed < 3600:
                return f"{int(elapsed / 60)} minutes"
            else:
                return f"{elapsed / 3600:.1f} hours"
        except:
            return "unknown"

    async def detect_patterns(self, logs: List[LogEntry]) -> List[Dict[str, Any]]:
        """Detect patterns in log data"""
        patterns = []

        # Error clustering
        error_logs = [l for l in logs if l.level == LogLevel.ERROR]
        if len(error_logs) > 3:
            # Group errors by service
            error_by_service = {}
            for log in error_logs:
                if log.service not in error_by_service:
                    error_by_service[log.service] = []
                error_by_service[log.service].append(log)

            for service, errors in error_by_service.items():
                if len(errors) > 2:
                    patterns.append({
                        'type': 'error_cluster',
                        'service': service,
                        'count': len(errors),
                        'severity': 'high',
                        'message': f"Multiple errors detected in {service}",
                        'recommendation': f"Investigate {service} service stability"
                    })

        # Performance patterns
        api_logs = [l for l in logs if l.event_type == EventType.API_CALL]
        if api_logs:
            response_times = []
            for log in api_logs:
                if 'response_time' in log.metadata:
                    response_times.append(log.metadata['response_time'])

            if response_times and max(response_times) > 5000:  # 5 seconds
                patterns.append({
                    'type': 'slow_api',
                    'max_time': max(response_times),
                    'severity': 'medium',
                    'message': "Slow API responses detected",
                    'recommendation': "Consider caching or optimization"
                })

        # Retry patterns
        retry_count = sum(1 for l in logs if 'retry' in l.message.lower())
        if retry_count > 5:
            patterns.append({
                'type': 'excessive_retries',
                'count': retry_count,
                'severity': 'medium',
                'message': f"High retry count detected ({retry_count} retries)",
                'recommendation': "Check external service availability"
            })

        return patterns

class MissionControl:
    """Main Mission Control orchestrator"""

    def __init__(self):
        self.buffer = LogBuffer(max_size=5000)
        self.ai_analyzer = AIStreamAnalyzer()
        self.redis_client = None
        self.sessions: Dict[str, Dict] = {}
        self.subscribers: List[asyncio.Queue] = []
        self.is_running = False
        self.patterns_cache = deque(maxlen=100)

    async def initialize(self):
        """Initialize Mission Control systems"""
        try:
            # Initialize Redis for distributed logging
            self.redis_client = await redis.from_url(
                "redis://localhost:6380",
                encoding="utf-8",
                decode_responses=True
            )
            logger.info("Mission Control initialized with Redis")
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory only: {e}")

        self.is_running = True

        # Start background tasks
        asyncio.create_task(self._pattern_detection_loop())
        asyncio.create_task(self._cleanup_loop())

    async def log_event(
        self,
        level: LogLevel,
        event_type: EventType,
        service: str,
        message: str,
        session_id: str,
        metadata: Dict[str, Any] = None,
        phase: str = None,
        confidence: float = None
    ) -> LogEntry:
        """Log an event to Mission Control"""

        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            event_type=event_type,
            service=service,
            message=message,
            metadata=metadata or {},
            session_id=session_id,
            correlation_id=self._generate_correlation_id(),
            phase=phase,
            confidence=confidence
        )

        # Add to buffer
        self.buffer.add(entry)

        # Store in Redis if available
        if self.redis_client:
            try:
                await self.redis_client.lpush(
                    f"logs:{session_id}",
                    json.dumps(entry.to_dict())
                )
                await self.redis_client.expire(f"logs:{session_id}", 86400)  # 24 hours
            except Exception as e:
                logger.error(f"Redis storage error: {e}")

        # Broadcast to subscribers
        await self._broadcast_to_subscribers(entry)

        # Update session info
        if session_id in self.sessions:
            self.sessions[session_id]['last_activity'] = datetime.now()
            self.sessions[session_id]['log_count'] += 1

        return entry

    async def start_session(self, session_id: str, metadata: Dict[str, Any] = None) -> Dict:
        """Start a new monitoring session"""

        session = {
            'session_id': session_id,
            'start_time': datetime.now().isoformat(),
            'metadata': metadata or {},
            'log_count': 0,
            'error_count': 0,
            'phase': 'initialization',
            'last_activity': datetime.now()
        }

        self.sessions[session_id] = session

        # Log session start
        await self.log_event(
            LogLevel.EPIC,
            EventType.SYSTEM_START,
            "mission_control",
            f"Session started: {metadata.get('rfp_title', 'Unknown RFP')}",
            session_id,
            metadata
        )

        return session

    async def subscribe_to_logs(self) -> asyncio.Queue:
        """Subscribe to real-time log stream"""
        queue = asyncio.Queue()
        self.subscribers.append(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue):
        """Unsubscribe from log stream"""
        if queue in self.subscribers:
            self.subscribers.remove(queue)

    async def _broadcast_to_subscribers(self, entry: LogEntry):
        """Broadcast log entry to all subscribers"""
        for queue in self.subscribers:
            try:
                await queue.put(entry.to_dict())
            except:
                # Remove broken subscribers
                self.subscribers.remove(queue)

    async def get_ai_analysis(self, session_id: str, count: int = 20) -> AsyncGenerator[str, None]:
        """Get streaming AI analysis for recent logs"""

        # Get recent logs for session
        recent_logs = [
            log for log in self.buffer.get_recent(count * 2)
            if log.session_id == session_id
        ][-count:]

        if not recent_logs:
            yield "No recent activity to analyze."
            return

        # Stream AI analysis
        async for chunk in self.ai_analyzer.analyze_stream(recent_logs):
            yield chunk

    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session summary"""

        if session_id not in self.sessions:
            return {'error': 'Session not found'}

        session = self.sessions[session_id]

        # Get all logs for session
        session_logs = [
            log for log in self.buffer.buffer
            if log.session_id == session_id
        ]

        # Calculate statistics
        stats = {
            'total_logs': len(session_logs),
            'error_count': sum(1 for l in session_logs if l.level == LogLevel.ERROR),
            'warning_count': sum(1 for l in session_logs if l.level == LogLevel.WARNING),
            'success_count': sum(1 for l in session_logs if l.level == LogLevel.SUCCESS),
            'duration': self.ai_analyzer._calculate_elapsed_time(session_logs),
            'phases_completed': list(set(l.phase for l in session_logs if l.phase)),
            'services_involved': list(set(l.service for l in session_logs)),
            'average_confidence': sum(l.confidence for l in session_logs if l.confidence) / max(1, sum(1 for l in session_logs if l.confidence))
        }

        # Detect patterns
        patterns = await self.ai_analyzer.detect_patterns(session_logs)

        return {
            'session': session,
            'statistics': stats,
            'patterns': patterns,
            'recent_logs': [l.to_dict() for l in session_logs[-50:]],
            'buffer_stats': self.buffer.statistics
        }

    async def _pattern_detection_loop(self):
        """Background task for pattern detection"""
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Run every 30 seconds

                for session_id in list(self.sessions.keys()):
                    session_logs = [
                        log for log in self.buffer.buffer
                        if log.session_id == session_id
                    ]

                    if session_logs:
                        patterns = await self.ai_analyzer.detect_patterns(session_logs[-100:])
                        if patterns:
                            # Log detected patterns
                            for pattern in patterns:
                                await self.log_event(
                                    LogLevel.WARNING if pattern['severity'] == 'high' else LogLevel.SCENE,
                                    EventType.AGENT_ACTION,
                                    "pattern_detector",
                                    pattern['message'],
                                    session_id,
                                    pattern
                                )
            except Exception as e:
                logger.error(f"Pattern detection error: {e}")

    async def _cleanup_loop(self):
        """Background task for cleanup"""
        while self.is_running:
            await asyncio.sleep(300)  # Run every 5 minutes

            # Clean old log entries
            self.buffer.clear_old_entries(3600)  # Remove logs older than 1 hour

            # Clean inactive sessions
            current_time = datetime.now()
            for session_id in list(self.sessions.keys()):
                last_activity = self.sessions[session_id]['last_activity']
                if (current_time - last_activity).total_seconds() > 7200:  # 2 hours
                    del self.sessions[session_id]

    def _generate_correlation_id(self) -> str:
        """Generate unique correlation ID"""
        return hashlib.sha256(
            f"{datetime.now().isoformat()}_{os.urandom(8).hex()}".encode()
        ).hexdigest()[:16]

    async def export_session_logs(self, session_id: str, format: str = "json") -> str:
        """Export session logs for analysis"""

        session_logs = [
            log.to_dict() for log in self.buffer.buffer
            if log.session_id == session_id
        ]

        if format == "json":
            return json.dumps(session_logs, indent=2)
        elif format == "csv":
            import csv
            import io

            output = io.StringIO()
            if session_logs:
                writer = csv.DictWriter(output, fieldnames=session_logs[0].keys())
                writer.writeheader()
                writer.writerows(session_logs)

            return output.getvalue()
        else:
            return str(session_logs)

    async def end_session(self, session_id: str) -> Dict[str, Any]:
        """End a monitoring session and return summary"""
        if session_id not in self.sessions:
            return {'error': 'Session not found'}

        summary = await self.get_session_summary(session_id)

        # Clean up session
        del self.sessions[session_id]

        return summary

    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get list of active sessions"""
        return [
            {
                'session_id': session_id,
                'start_time': session_data['start_time'],
                'phase': session_data.get('phase', 'unknown'),
                'log_count': session_data.get('log_count', 0),
                'error_count': session_data.get('error_count', 0)
            }
            for session_id, session_data in self.sessions.items()
        ]

# Global Mission Control instance
mission_control = MissionControl()
