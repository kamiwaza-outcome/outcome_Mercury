"""
OpenAI Service - Wrapper for OpenAI API calls
Consolidates client creation, sane defaults, and light resilience.
"""
import os
import logging
from typing import Optional
from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)


class GPT5CircuitBreaker:
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


CIRCUIT_BREAKER = GPT5CircuitBreaker()


class OpenAIService:
    """Shared OpenAI async service with sensible defaults and retries.

    Fixes the missing `model` attribute bug and centralizes defaults for
    model selection, timeouts, streaming, and fallback model.
    """

    # Module-level shared client to reduce socket exhaustion
    _shared_client: Optional[AsyncOpenAI] = None
    _shared_sync_client: Optional[OpenAI] = None

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")

        # Public attributes expected by agents
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.fallback_model = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o-mini")

        # Config knobs
        self.enable_streaming = os.getenv("OPENAI_STREAMING", "false").lower() == "true"
        self.max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
        self.retry_delay = float(os.getenv("OPENAI_RETRY_DELAY", "5"))

        # Timeout: default to 180s unless overridden
        timeout_env = float(os.getenv("OPENAI_TIMEOUT", "180"))

        # Create or reuse shared client
        if OpenAIService._shared_client is None:
            OpenAIService._shared_client = AsyncOpenAI(api_key=api_key, timeout=timeout_env)
        self.client = OpenAIService._shared_client

        if OpenAIService._shared_sync_client is None:
            OpenAIService._shared_sync_client = OpenAI(api_key=api_key, timeout=timeout_env)
        self.sync_client = OpenAIService._shared_sync_client

    def _calculate_timeout(self, tokens: int, model: str, reasoning_effort: Optional[str]) -> float:
        """Heuristic timeout calculation with 600s cap.

        Aligns with recommendations to scale by model and effort.
        """
        base = 120.0 if "gpt-5" in (model or "").lower() else 30.0
        effort_multiplier = {"low": 1.0, "medium": 2.0, "high": 4.0}.get((reasoning_effort or "medium").lower(), 2.0)
        # +1s per 100 tokens as a small linear component
        dynamic = base * effort_multiplier + (max(tokens or 0, 0) / 100.0)
        return min(dynamic, 600.0)

    async def get_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """Get a completion from OpenAI with light retry and optional streaming.

        - Uses `self.model` by default
        - Caps tokens by env MAX_COMPLETION_TOKENS (default 16000)
        - Applies heuristic timeout when possible (client has a hard cap)
        """
        chosen_model = model or self.model
        # Circuit breaker: avoid GPT-5 if tripped
        if "gpt-5" in chosen_model.lower() and not CIRCUIT_BREAKER.should_allow():
            logger.warning("GPT-5 circuit open; using fallback model instead")
            chosen_model = self.fallback_model
        # Respect env cap
        env_cap = int(os.getenv("MAX_COMPLETION_TOKENS", "16000"))
        token_param_value = None
        if max_tokens is not None:
            token_param_value = min(max_tokens, env_cap)
        else:
            token_param_value = env_cap

        # Build params according to model family
        is_gpt5 = "gpt-5" in chosen_model.lower()
        token_param_name = "max_completion_tokens" if (is_gpt5 or "gpt-4" in chosen_model.lower()) else "max_tokens"

        params = {
            "model": chosen_model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for RFP analysis and document improvement."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            token_param_name: token_param_value,
        }

        if is_gpt5:
            # Default to medium unless caller overrides
            params["reasoning_effort"] = (reasoning_effort or os.getenv("GPT5_REASONING_EFFORT", "medium"))

        # Heuristic timeout (documented; client still enforces its own)
        _ = self._calculate_timeout(tokens=token_param_value or 0, model=chosen_model, reasoning_effort=params.get("reasoning_effort"))

        # Try request with small backoff and fallback model on last attempt
        attempt = 0
        last_err = None
        while attempt < max(1, self.max_retries):
            try:
                # Streaming optional (only if enabled and supported)
                if self.enable_streaming:
                    try:
                        stream = await self.client.chat.completions.create(
                            **{**params, "stream": True}
                        )
                        # Some SDKs return an async iterator, others a wrapper – handle conservatively
                        chunks = []
                        async for chunk in stream:  # type: ignore
                            delta = getattr(getattr(chunk.choices[0], "delta", {}), "content", None)
                            if delta:
                                chunks.append(delta)
                        # Streaming completed successfully
                        if "gpt-5" in chosen_model.lower():
                            CIRCUIT_BREAKER.record_success()
                        return "".join(chunks) if chunks else ""
                    except Exception as stream_err:
                        logger.info(f"Streaming failed, retrying non-streaming: {stream_err}")
                        # fall through to non-streaming request

                response = await self.client.chat.completions.create(**params)
                if "gpt-5" in chosen_model.lower():
                    CIRCUIT_BREAKER.record_success()
                return response.choices[0].message.content
            except Exception as e:
                last_err = e
                # Update circuit breaker on GPT-5 failures
                if "gpt-5" in chosen_model.lower():
                    CIRCUIT_BREAKER.record_failure()
                attempt += 1
                # On last attempt, try fallback model
                if attempt >= self.max_retries and chosen_model != self.fallback_model:
                    logger.warning(f"Primary model failed; retrying with fallback model {self.fallback_model}: {e}")
                    params["model"] = self.fallback_model
                    chosen_model = self.fallback_model
                    # reset attempt window for fallback model (single try)
                    try:
                        response = await self.client.chat.completions.create(**params)
                        # mark healthy if fallback works (breaker pertains to gpt-5 only)
                        return response.choices[0].message.content
                    except Exception as e2:
                        last_err = e2
                        break
                # basic backoff delay
                try:
                    import asyncio as _asyncio
                    await _asyncio.sleep(self.retry_delay)
                except Exception:
                    pass

        # Success path would have early returned; if we used GPT-5 and got here with no exception raised above,
        # the breaker remains as previously set. If a GPT-5 attempt succeeded earlier, we should mark success.
        # As we only reach here on failure, nothing to reset. Success calls should call record_success explicitly.

        logger.error(f"Error getting OpenAI completion: {last_err}")
        return f"Unable to generate response at this time. Error: {str(last_err)}"
