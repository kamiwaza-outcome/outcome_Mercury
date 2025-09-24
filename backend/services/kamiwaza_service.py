"""
Kamiwaza Service - Wrapper for Kamiwaza SDK to provide local model access
Provides OpenAI-compatible interface for seamless migration from OpenAI APIs
"""
import os
import logging
from functools import cached_property
from typing import List, Dict, Any, Optional
from kamiwaza_client import KamiwazaClient

logger = logging.getLogger(__name__)


class KamiwazaService:
    """Service wrapper for Kamiwaza SDK integration.

    Provides:
    - Model discovery and listing from local Kamiwaza deployment
    - OpenAI-compatible client interface for easy migration
    - Health check and monitoring capabilities
    """

    def __init__(self):
        """Initialize Kamiwaza service with configuration from environment."""
        # Kamiwaza endpoint configuration
        self._endpoint = os.getenv("KAMIWAZA_ENDPOINT", "http://host.docker.internal:7777/api/")
        self._verify_ssl = os.getenv("KAMIWAZA_VERIFY_SSL", "false").lower() == "true"

        # Default model selection
        self.default_model = os.getenv("KAMIWAZA_DEFAULT_MODEL")
        self.fallback_model = os.getenv("KAMIWAZA_FALLBACK_MODEL")

        logger.info(f"Initializing Kamiwaza service with endpoint: {self._endpoint}")

    @cached_property
    def _client(self) -> KamiwazaClient:
        """Lazy-load and cache Kamiwaza client."""
        try:
            client = KamiwazaClient(self._endpoint)
            client.session.verify = self._verify_ssl
            logger.info("Kamiwaza client initialized successfully")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Kamiwaza client: {e}")
            raise

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available AI models from Kamiwaza deployment.

        Returns:
            List of available models with id, name, status, and endpoint
        """
        try:
            deployments = self._client.serving.list_active_deployments()
            logger.info(f"Found {len(deployments)} active model deployments")

            models = []
            for deployment in deployments:
                model_info = {
                    "id": str(deployment.id),
                    "name": deployment.m_name,
                    "status": deployment.status,
                    "endpoint": deployment.endpoint,
                    "model_type": getattr(deployment, "model_type", "unknown"),
                }
                models.append(model_info)
                logger.debug(f"Model available: {model_info['name']} (status: {model_info['status']})")

            return models
        except Exception as e:
            logger.error(f"Failed to list models from Kamiwaza: {e}")
            raise

    def get_openai_client(self, model_name: Optional[str] = None):
        """Get OpenAI-compatible client for the specified model.

        This provides a drop-in replacement for OpenAI client, allowing
        seamless migration from OpenAI APIs to Kamiwaza local models.

        Args:
            model_name: Name of the model to use. If None, uses default model.

        Returns:
            OpenAI-compatible client instance
        """
        model = model_name or self.default_model
        if not model:
            raise ValueError("No model specified and no default model configured")

        logger.info(f"Getting OpenAI-compatible client for model: {model}")
        try:
            return self._client.openai.get_client(model)
        except Exception as e:
            logger.error(f"Failed to get OpenAI client for model {model}: {e}")
            # Try fallback model if configured
            if self.fallback_model and model != self.fallback_model:
                logger.info(f"Attempting fallback model: {self.fallback_model}")
                return self._client.openai.get_client(self.fallback_model)
            raise

    async def get_model_by_capability(self, capability: str) -> Optional[str]:
        """Get a model name that matches the requested capability.

        Args:
            capability: Type of capability needed (e.g., 'chat', 'completion', 'embedding')

        Returns:
            Model name that supports the capability, or None if not found
        """
        models = await self.list_models()

        # Map capabilities to model name patterns
        capability_patterns = {
            "chat": ["gpt", "llama", "mistral", "chat"],
            "completion": ["gpt", "llama", "mistral", "completion"],
            "embedding": ["embed", "e5", "bge"],
            "vision": ["vision", "multimodal", "clip"],
        }

        patterns = capability_patterns.get(capability.lower(), [])

        for model in models:
            model_name_lower = model["name"].lower()
            for pattern in patterns:
                if pattern in model_name_lower and model["status"] == "active":
                    logger.info(f"Found model {model['name']} for capability {capability}")
                    return model["name"]

        logger.warning(f"No model found for capability: {capability}")
        return None

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Kamiwaza connection.

        Returns:
            Health status including connection status and available models
        """
        try:
            models = await self.list_models()
            return {
                "healthy": True,
                "message": f"Connected to Kamiwaza, {len(models)} models available",
                "models_count": len(models),
                "endpoint": self._endpoint,
                "models": [m["name"] for m in models]
            }
        except Exception as e:
            return {
                "healthy": False,
                "message": f"Kamiwaza connection failed: {str(e)}",
                "endpoint": self._endpoint,
                "models_count": 0,
                "models": []
            }

    def create_chat_completion(
        self,
        model_name: Optional[str] = None,
        messages: List[Dict[str, str]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ):
        """Create a chat completion using Kamiwaza models.

        This method provides a convenience wrapper that matches OpenAI's API.

        Args:
            model_name: Model to use for completion
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the model

        Returns:
            Chat completion response
        """
        client = self.get_openai_client(model_name)

        # Build request parameters
        request_params = {
            "model": model_name or self.default_model,
            "messages": messages or [],
            "temperature": temperature,
            "stream": stream,
            **kwargs
        }

        if max_tokens:
            request_params["max_tokens"] = max_tokens

        logger.debug(f"Creating chat completion with params: {request_params}")

        try:
            response = client.chat.completions.create(**request_params)
            logger.info(f"Chat completion successful for model: {model_name}")
            return response
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise


# Singleton instance for shared usage
_kamiwaza_service_instance: Optional[KamiwazaService] = None


def get_kamiwaza_service() -> KamiwazaService:
    """Get or create singleton Kamiwaza service instance.

    Returns:
        Shared KamiwazaService instance
    """
    global _kamiwaza_service_instance
    if _kamiwaza_service_instance is None:
        _kamiwaza_service_instance = KamiwazaService()
    return _kamiwaza_service_instance