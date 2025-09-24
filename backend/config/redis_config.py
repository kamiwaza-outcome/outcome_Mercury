"""Redis configuration for the application."""
import os
import redis
from typing import Optional, Any
import json
import logging

logger = logging.getLogger(__name__)

# Redis configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

class RedisConfig:
    """Redis configuration class."""
    def __init__(self):
        self.host = REDIS_HOST
        self.port = REDIS_PORT
        self.db = REDIS_DB
        self.password = REDIS_PASSWORD
        self.decode_responses = True

    def get_url(self) -> str:
        """Get Redis connection URL."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"

    def get_config(self) -> dict:
        """Get Redis configuration dictionary."""
        config = {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "decode_responses": self.decode_responses,
        }
        if self.password:
            config["password"] = self.password
        return config

class RedisConnectionManager:
    """Manages Redis connections."""
    def __init__(self, config: RedisConfig):
        self.config = config
        self._client = None

    def get_client(self):
        """Get or create Redis client."""
        if self._client is None:
            try:
                self._client = redis.Redis(**self.config.get_config())
                self._client.ping()
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                # Return a mock client for development
                self._client = MockRedisClient()
        return self._client

    def close(self):
        """Close Redis connection."""
        if self._client:
            self._client.close()
            self._client = None

class RedisOperations:
    """Redis operations wrapper."""
    def __init__(self, client):
        self.client = client

    def set_json(self, key: str, value: Any, ex: Optional[int] = None):
        """Set JSON value in Redis."""
        try:
            json_value = json.dumps(value)
            return self.client.set(key, json_value, ex=ex)
        except Exception as e:
            logger.error(f"Failed to set JSON in Redis: {e}")
            return False

    def get_json(self, key: str) -> Optional[Any]:
        """Get JSON value from Redis."""
        try:
            value = self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Failed to get JSON from Redis: {e}")
            return None

class MockRedisClient:
    """Mock Redis client for development when Redis is not available."""
    def __init__(self):
        self._data = {}

    def ping(self):
        return True

    def set(self, key, value, ex=None):
        self._data[key] = value
        return True

    def get(self, key):
        return self._data.get(key)

    def delete(self, *keys):
        for key in keys:
            self._data.pop(key, None)
        return len(keys)

    def exists(self, key):
        return key in self._data

    def close(self):
        pass

def create_redis_manager() -> RedisConnectionManager:
    """Create a Redis connection manager."""
    config = RedisConfig()
    return RedisConnectionManager(config)

# Legacy functions for backward compatibility
def get_redis_url() -> str:
    """Get Redis connection URL."""
    config = RedisConfig()
    return config.get_url()

def get_redis_config() -> dict:
    """Get Redis configuration dictionary."""
    config = RedisConfig()
    return config.get_config()