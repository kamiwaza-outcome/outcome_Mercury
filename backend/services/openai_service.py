"""
OpenAI Service - Now powered by Kamiwaza SDK for local model deployment
Provides OpenAI-compatible interface using Kamiwaza local models
"""
import os
import logging
from typing import Optional, List, Dict, Any
import asyncio
from .kamiwaza_service import get_kamiwaza_service, KamiwazaService

logger = logging.getLogger(__name__)


class ModelCircuitBreaker:
    """Circuit breaker for model failures."""
    def __init__(self, threshold: int = 3, cooldown: float = 300.0):
        self.threshold = threshold
        self.cooldown = cooldown
        self.failures = 0
        self.last_failure: float = 0.0

    def should_allow(self) -> bool:
        import time
        # Allow if under threshold
        if self.failures < self.threshold:
            return True
        # If over threshold, only allow after cooldown elapsed
        return (time.time() - self.last_failure) > self.cooldown

    def record_failure(self) -> None:
        import time
        self.failures += 1
        self.last_failure = time.time()

    def record_success(self) -> None:
        self.failures = 0
        self.last_failure = 0.0

    def is_healthy(self) -> bool:
        return self.failures == 0 or self.should_allow()


CIRCUIT_BREAKER = ModelCircuitBreaker()


class OpenAIService:
    """OpenAI-compatible service now powered by Kamiwaza for local model deployment.

    This class maintains the same interface as the original OpenAI service
    but routes all calls through Kamiwaza SDK to use locally deployed models.
    """

    # Shared Kamiwaza service instance
    _kamiwaza_service: Optional[KamiwazaService] = None
    _available_models: List[Dict[str, Any]] = []
    _last_model_refresh: float = 0.0

    def __init__(self):
        """Initialize with Kamiwaza service instead of OpenAI."""
        # Get or create Kamiwaza service
        if OpenAIService._kamiwaza_service is None:
            OpenAIService._kamiwaza_service = get_kamiwaza_service()
        self.kamiwaza = OpenAIService._kamiwaza_service

        # Model configuration - now uses Kamiwaza models
        self.model = os.getenv("KAMIWAZA_DEFAULT_MODEL", "llama3")
        self.fallback_model = os.getenv("KAMIWAZA_FALLBACK_MODEL", "mistral")

        # Legacy compatibility - map OpenAI model names to Kamiwaza equivalents
        self._model_mapping = {
            "gpt-5": self.model,  # Map GPT-5 to default Kamiwaza model
            "gpt-5-mini": self.fallback_model,
            "gpt-4o": self.model,
            "gpt-4o-mini": self.fallback_model,
            "gpt-4": self.model,
            "gpt-3.5-turbo": self.fallback_model,
        }

        # Config knobs
        self.enable_streaming = os.getenv("KAMIWAZA_STREAMING", "false").lower() == "true"
        self.max_retries = int(os.getenv("KAMIWAZA_MAX_RETRIES", "3"))
        self.retry_delay = float(os.getenv("KAMIWAZA_RETRY_DELAY", "5"))

        # Create OpenAI-compatible clients for both async and sync
        try:
            self.client = self.kamiwaza.get_openai_client(self.model)
            self.sync_client = self.kamiwaza.get_openai_client(self.model)
            logger.info(f"Kamiwaza-powered OpenAI service initialized with model: {self.model}")
        except Exception as e:
            logger.warning(f"Failed to initialize default model {self.model}, will select on-demand: {e}")
            self.client = None
            self.sync_client = None

        # Refresh available models on init
        asyncio.create_task(self._refresh_models())

    async def _refresh_models(self) -> None:
        """Refresh the list of available models from Kamiwaza."""
        import time
        try:
            current_time = time.time()
            # Only refresh if more than 60 seconds have passed
            if current_time - OpenAIService._last_model_refresh > 60:
                OpenAIService._available_models = await self.kamiwaza.list_models()
                OpenAIService._last_model_refresh = current_time
                logger.info(f"Refreshed model list: {len(OpenAIService._available_models)} models available")

                # Auto-select default model if not set
                if not self.model and OpenAIService._available_models:
                    self.model = OpenAIService._available_models[0]["name"]
                    logger.info(f"Auto-selected model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to refresh model list: {e}")

    def _map_model_name(self, model_name: str) -> str:
        """Map OpenAI model names to Kamiwaza model names.

        Args:
            model_name: Original model name (possibly OpenAI format)

        Returns:
            Mapped Kamiwaza model name
        """
        # Check explicit mapping
        if model_name in self._model_mapping:
            mapped = self._model_mapping[model_name]
            logger.debug(f"Mapped model {model_name} -> {mapped}")
            return mapped

        # Check if it's already a valid Kamiwaza model
        for model in OpenAIService._available_models:
            if model["name"] == model_name:
                return model_name

        # Default to the configured default model
        logger.warning(f"Unknown model {model_name}, using default: {self.model}")
        return self.model

    def _calculate_timeout(self, tokens: int, model: str, reasoning_effort: Optional[str]) -> float:
        """Calculate timeout for request (kept for compatibility)."""
        base = 120.0 if "llama" in (model or "").lower() else 30.0
        effort_multiplier = {"low": 1.0, "medium": 2.0, "high": 4.0}.get((reasoning_effort or "medium").lower(), 2.0)
        dynamic = base * effort_multiplier + (max(tokens or 0, 0) / 100.0)
        return min(dynamic, 600.0)

    async def list_available_models(self) -> List[Dict[str, Any]]:
        """List all available models from Kamiwaza deployment.

        Returns:
            List of available models with their details
        """
        await self._refresh_models()
        return OpenAIService._available_models

    async def select_model(self, model_name: Optional[str] = None, capability: Optional[str] = None) -> str:
        """Select a model by name or capability.

        Args:
            model_name: Specific model name to use
            capability: Type of capability needed (e.g., 'chat', 'completion', 'embedding')

        Returns:
            Selected model name
        """
        if model_name:
            return self._map_model_name(model_name)

        if capability:
            selected = await self.kamiwaza.get_model_by_capability(capability)
            if selected:
                return selected

        return self.model

    async def get_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """Get a completion from Kamiwaza models with OpenAI-compatible interface.

        Args:
            prompt: The prompt to send to the model
            model: Model name (will be mapped to Kamiwaza model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            reasoning_effort: Effort level (kept for compatibility)

        Returns:
            Generated text response
        """
        # Map model name to Kamiwaza equivalent
        chosen_model = self._map_model_name(model or self.model)

        # Circuit breaker check
        if not CIRCUIT_BREAKER.should_allow():
            logger.warning(f"Circuit breaker open for model {chosen_model}; using fallback")
            chosen_model = self.fallback_model

        # Respect token cap
        env_cap = int(os.getenv("MAX_COMPLETION_TOKENS", "16000"))
        token_param_value = min(max_tokens or env_cap, env_cap)

        # Build request parameters
        params = {
            "model": chosen_model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for RFP analysis and document improvement."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": token_param_value,
        }

        # Get OpenAI-compatible client for the selected model
        try:
            client = self.kamiwaza.get_openai_client(chosen_model)
        except Exception as e:
            logger.error(f"Failed to get client for model {chosen_model}: {e}")
            # Try fallback model
            client = self.kamiwaza.get_openai_client(self.fallback_model)
            params["model"] = self.fallback_model
            chosen_model = self.fallback_model

        # Retry logic with backoff
        attempt = 0
        last_err = None

        while attempt < max(1, self.max_retries):
            try:
                # Try streaming if enabled
                if self.enable_streaming:
                    try:
                        stream = await client.chat.completions.create(
                            **{**params, "stream": True}
                        )
                        chunks = []
                        async for chunk in stream:
                            delta = getattr(getattr(chunk.choices[0], "delta", {}), "content", None)
                            if delta:
                                chunks.append(delta)
                        CIRCUIT_BREAKER.record_success()
                        return "".join(chunks) if chunks else ""
                    except Exception as stream_err:
                        logger.info(f"Streaming failed, retrying non-streaming: {stream_err}")

                # Non-streaming request
                response = await client.chat.completions.create(**params)
                CIRCUIT_BREAKER.record_success()
                return response.choices[0].message.content

            except Exception as e:
                last_err = e
                CIRCUIT_BREAKER.record_failure()
                attempt += 1

                # Try fallback model on last attempt
                if attempt >= self.max_retries and chosen_model != self.fallback_model:
                    logger.warning(f"Model {chosen_model} failed; trying fallback {self.fallback_model}: {e}")
                    try:
                        fallback_client = self.kamiwaza.get_openai_client(self.fallback_model)
                        params["model"] = self.fallback_model
                        response = await fallback_client.chat.completions.create(**params)
                        return response.choices[0].message.content
                    except Exception as e2:
                        last_err = e2
                        break

                # Backoff delay
                await asyncio.sleep(self.retry_delay)

        logger.error(f"Error getting completion from Kamiwaza: {last_err}")
        return f"Unable to generate response. Error: {str(last_err)}"

    async def health_check(self) -> Dict[str, Any]:
        """Check health of Kamiwaza connection and models.

        Returns:
            Health status information
        """
        health_info = await self.kamiwaza.health_check()
        health_info["circuit_breaker_healthy"] = CIRCUIT_BREAKER.is_healthy()
        return health_info