"""
Mission Control Logger Integration
Provides decorators and utilities for integrating Mission Control logging
into existing services with minimal code changes.
"""

import logging
import asyncio
import functools
from typing import Any, Callable, Optional, Dict
from datetime import datetime
import json
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class MissionControlLogger:
    """Logger that integrates with Mission Control"""

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self.redis_client = None
        self.connected = False

    async def connect(self):
        """Connect to Redis for publishing logs"""
        try:
            self.redis_client = await redis.from_url(
                "redis://localhost:6380",
                decode_responses=True
            )
            self.connected = True
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}")
            self.connected = False

    async def log(
        self,
        level: str,
        message: str,
        service: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Send log to Mission Control"""
        if not self.session_id:
            return

        if not self.connected:
            await self.connect()

        if not self.connected:
            return

        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": level,
                "service": service,
                "message": message,
                "metadata": metadata or {}
            }

            # Publish to Redis channel
            channel = f"mission_control:{self.session_id}:logs"
            await self.redis_client.publish(
                channel,
                json.dumps(log_entry)
            )
        except Exception as e:
            logger.error(f"Failed to send log to Mission Control: {e}")

    async def close(self):
        """Close Redis connection"""
        if self.redis_client:
            await self.redis_client.close()
            self.connected = False

# Global logger instance
_mission_logger: Optional[MissionControlLogger] = None

def init_mission_logging(session_id: str):
    """Initialize Mission Control logging for a session"""
    global _mission_logger
    _mission_logger = MissionControlLogger(session_id)
    asyncio.create_task(_mission_logger.connect())

def get_mission_logger() -> Optional[MissionControlLogger]:
    """Get the current Mission Control logger"""
    return _mission_logger

# Decorator for logging function calls
def mission_log(
    service: str,
    level: str = "scene",
    extract_metadata: Optional[Callable] = None
):
    """
    Decorator to automatically log function calls to Mission Control

    Args:
        service: Name of the service/component
        level: Log level (epic, chapter, scene, detail)
        extract_metadata: Optional function to extract metadata from args/kwargs
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            mc_logger = get_mission_logger()

            # Log function entry
            if mc_logger:
                metadata = {}
                if extract_metadata:
                    try:
                        metadata = extract_metadata(*args, **kwargs)
                    except:
                        pass

                await mc_logger.log(
                    level=level,
                    message=f"Starting {func.__name__}",
                    service=service,
                    metadata=metadata
                )

            try:
                # Execute function
                result = await func(*args, **kwargs)

                # Log success
                if mc_logger:
                    await mc_logger.log(
                        level=level,
                        message=f"Completed {func.__name__}",
                        service=service,
                        metadata={"status": "success"}
                    )

                return result

            except Exception as e:
                # Log error
                if mc_logger:
                    await mc_logger.log(
                        level="epic",  # Errors are always epic level
                        message=f"Error in {func.__name__}: {str(e)}",
                        service=service,
                        metadata={"error": str(e), "status": "failed"}
                    )
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, just pass through
            # (Mission Control logging is async-only)
            return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

# Convenience decorators for different services
def log_orchestration(level: str = "chapter"):
    """Decorator for orchestration agent logging"""
    return mission_log(service="orchestration", level=level)

def log_document_generation(level: str = "scene"):
    """Decorator for document generation logging"""
    return mission_log(service="document_generation", level=level)

def log_sam_api(level: str = "detail"):
    """Decorator for SAM API logging"""
    return mission_log(service="sam_api", level=level)

def log_browser_scraper(level: str = "detail"):
    """Decorator for browser scraper logging"""
    return mission_log(service="browser_scraper", level=level)

def log_milvus(level: str = "detail"):
    """Decorator for Milvus RAG logging"""
    return mission_log(service="milvus_rag", level=level)

def log_agent(agent_name: str, level: str = "scene"):
    """Decorator for specific agent logging"""
    return mission_log(service=f"agent_{agent_name}", level=level)

# Context manager for logging blocks
class MissionLogContext:
    """Context manager for logging code blocks"""

    def __init__(
        self,
        service: str,
        operation: str,
        level: str = "scene",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.service = service
        self.operation = operation
        self.level = level
        self.metadata = metadata or {}
        self.mc_logger = None

    async def __aenter__(self):
        self.mc_logger = get_mission_logger()
        if self.mc_logger:
            await self.mc_logger.log(
                level=self.level,
                message=f"Starting {self.operation}",
                service=self.service,
                metadata=self.metadata
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.mc_logger:
            if exc_type:
                await self.mc_logger.log(
                    level="epic",
                    message=f"Error in {self.operation}: {exc_val}",
                    service=self.service,
                    metadata={"error": str(exc_val), "status": "failed"}
                )
            else:
                await self.mc_logger.log(
                    level=self.level,
                    message=f"Completed {self.operation}",
                    service=self.service,
                    metadata={"status": "success"}
                )
        return False

# Quick logging functions
async def log_epic(service: str, message: str, **metadata):
    """Log an epic-level event"""
    mc_logger = get_mission_logger()
    if mc_logger:
        await mc_logger.log("epic", message, service, metadata)

async def log_chapter(service: str, message: str, **metadata):
    """Log a chapter-level event"""
    mc_logger = get_mission_logger()
    if mc_logger:
        await mc_logger.log("chapter", message, service, metadata)

async def log_scene(service: str, message: str, **metadata):
    """Log a scene-level event"""
    mc_logger = get_mission_logger()
    if mc_logger:
        await mc_logger.log("scene", message, service, metadata)

async def log_detail(service: str, message: str, **metadata):
    """Log a detail-level event"""
    mc_logger = get_mission_logger()
    if mc_logger:
        await mc_logger.log("detail", message, service, metadata)