"""
GPT Client - Base client for OpenAI API integration

Provides async communication with OpenAI's GPT models with:
- Automatic retry with exponential backoff
- Token usage tracking
- Error handling
- Response caching
"""

import asyncio
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from openai import AsyncOpenAI
from openai import APIError, RateLimitError, APIConnectionError
from config import config

logger = logging.getLogger(__name__)


class GPTClient:
    """
    Base client for OpenAI GPT API with async support and intelligent caching
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5-mini",
        temperature: float = 0.7,
        max_tokens: int = 2000,
        timeout: float = 60.0
    ):
        """
        Initialize GPT Client

        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-5-mini, gpt-5.1)
            temperature: Creativity level (0.0-1.0)
            max_tokens: Maximum response tokens
            timeout: Request timeout in seconds
        """
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Token usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0
        self.total_cost = 0.0

        # Response cache (5 minute TTL)
        self._cache: Dict[str, Dict] = {}
        self._cache_ttl = timedelta(minutes=5)

        # Rate limiting
        self._last_request_time = None
        self._min_request_interval = 0.1  # 100ms between requests

        # Daily cost limit from config
        self._daily_cost_limit = getattr(config, 'GPT_MAX_DAILY_COST_USD', 10.0)
        self._daily_cost_reset_date = datetime.now().date()

        # Model pricing (per 1M tokens) - Updated Dec 2024
        self._pricing = {
            # GPT-5 Models (Primary)
            "gpt-5-mini": {"input": 0.20, "output": 0.80},   # Frequent calls
            "gpt-5.1": {"input": 5.00, "output": 15.00},     # Premium calls
            # Legacy GPT-4 Models (backward compatibility)
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-4": {"input": 30.00, "output": 60.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        }

        # Model routing: frequent (cheap) vs premium (powerful)
        self.model_frequent = model  # Default model for frequent calls
        self.model_premium = "gpt-5.1"  # Premium model for critical decisions

        logger.info(f"GPT Client initialized with model: {model}")

    def _get_cache_key(self, messages: List[Dict], **kwargs) -> str:
        """Generate cache key from messages and params"""
        content = json.dumps({"messages": messages, "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()

    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        if not cache_entry:
            return False
        cached_time = cache_entry.get("timestamp")
        if not cached_time:
            return False
        return datetime.now() - cached_time < self._cache_ttl

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: Optional[str] = None) -> float:
        """Calculate cost based on token usage"""
        model_to_use = model or self.model
        pricing = self._pricing.get(model_to_use, self._pricing["gpt-5-mini"])
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    async def _rate_limit(self):
        """Apply rate limiting between requests"""
        if self._last_request_time:
            elapsed = (datetime.now() - self._last_request_time).total_seconds()
            if elapsed < self._min_request_interval:
                await asyncio.sleep(self._min_request_interval - elapsed)
        self._last_request_time = datetime.now()

    def _check_daily_cost_limit(self) -> bool:
        """
        Check if daily cost limit has been reached.
        Resets counter at midnight.

        Returns:
            True if under limit, False if limit exceeded
        """
        today = datetime.now().date()

        # Reset daily cost if new day
        if today > self._daily_cost_reset_date:
            logger.info(f"🔄 Resetting daily GPT cost counter (new day)")
            self.total_cost = 0.0
            self._daily_cost_reset_date = today

        if self.total_cost >= self._daily_cost_limit:
            logger.warning(
                f"⚠️ Daily GPT cost limit reached: ${self.total_cost:.2f} >= ${self._daily_cost_limit:.2f}"
            )
            return False

        return True

    def set_models(self, frequent: str, premium: str):
        """
        Configure model routing

        Args:
            frequent: Model for frequent/cheap calls (gpt-5-mini)
            premium: Model for critical decisions (gpt-5.1)
        """
        self.model_frequent = frequent
        self.model_premium = premium
        logger.info(f"Model routing configured: frequent={frequent}, premium={premium}")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
        json_mode: bool = False,
        use_premium: bool = False
    ) -> Dict[str, Any]:
        """
        Send chat completion request to GPT

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            use_cache: Whether to use response caching
            json_mode: Whether to request JSON response format
            use_premium: Whether to use premium model for this request

        Returns:
            Dict with 'content', 'usage', 'cost', and 'cached' keys
        """
        # Check cache
        cache_key = self._get_cache_key(messages, temp=temperature, tokens=max_tokens)
        if use_cache and cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if self._is_cache_valid(cache_entry):
                logger.debug("Returning cached GPT response")
                return {**cache_entry["response"], "cached": True}

        # Check daily cost limit (CRÍTICO: prevent runaway API costs)
        if not self._check_daily_cost_limit():
            raise Exception(
                f"Daily GPT cost limit exceeded (${self.total_cost:.2f} >= ${self._daily_cost_limit:.2f}). "
                f"Increase GPT_MAX_DAILY_COST_USD in config or wait until tomorrow."
            )

        # Apply rate limiting
        await self._rate_limit()

        # Select model based on use_premium flag
        selected_model = self.model_premium if use_premium else self.model_frequent
        if use_premium:
            logger.info(f"🧠 Using PREMIUM model: {selected_model}")

        # Prepare request params
        params = {
            "model": selected_model,
            "messages": messages,
        }

        # Reasoning models (o1, o1-mini, o1-preview) do NOT support custom temperature
        # They only accept the default value (1). For these models, omit temperature entirely.
        # GPT-5 and GPT-4o models DO support temperature (0.0-1.0)
        reasoning_models = ['o1', 'o1-mini', 'o1-preview']
        model_lower = selected_model.lower()

        is_reasoning_model = any(rm == model_lower or model_lower.startswith(f"{rm}-") for rm in reasoning_models)

        if not is_reasoning_model:
            # Normal models support temperature
            params["temperature"] = temperature if temperature is not None else self.temperature
        else:
            logger.debug(f"Reasoning model {selected_model} detected - omitting temperature parameter")

        # Use max_completion_tokens for newer models (gpt-4o, o1, etc.), max_tokens for legacy
        # OpenAI changed the parameter name for newer models
        tokens_value = max_tokens if max_tokens is not None else self.max_tokens
        newer_models = ['gpt-4o', 'gpt-4o-mini', 'o1', 'o1-mini', 'o1-preview', 'gpt-4-turbo', 'gpt-5', 'chatgpt-4o']
        if any(model in model_lower for model in newer_models):
            params["max_completion_tokens"] = tokens_value
        else:
            params["max_tokens"] = tokens_value

        if json_mode:
            params["response_format"] = {"type": "json_object"}

        # Retry logic with exponential backoff
        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(**params)

                # Extract response data
                content = response.choices[0].message.content
                usage = response.usage

                # Update tracking
                self.total_prompt_tokens += usage.prompt_tokens
                self.total_completion_tokens += usage.completion_tokens
                self.total_requests += 1
                cost = self._calculate_cost(usage.prompt_tokens, usage.completion_tokens, selected_model)
                self.total_cost += cost

                result = {
                    "content": content,
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens
                    },
                    "cost": cost,
                    "cached": False,
                    "model_used": selected_model,
                    "is_premium": use_premium
                }

                # Cache response
                if use_cache:
                    self._cache[cache_key] = {
                        "response": result,
                        "timestamp": datetime.now()
                    }

                logger.debug(
                    f"GPT request successful: {usage.total_tokens} tokens, ${cost:.4f}"
                )

                return result

            except RateLimitError as e:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Rate limit hit, retrying in {delay}s: {e}")
                await asyncio.sleep(delay)

            except APIConnectionError as e:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Connection error, retrying in {delay}s: {e}")
                await asyncio.sleep(delay)

            except APIError as e:
                logger.error(f"OpenAI API error: {e}")
                raise

            except Exception as e:
                logger.error(f"Unexpected error in GPT request: {e}")
                raise

        raise Exception(f"Failed after {max_retries} retries")

    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Send chat request expecting JSON response

        Args:
            messages: List of message dicts
            temperature: Override temperature
            max_tokens: Override max_tokens

        Returns:
            Parsed JSON response
        """
        response = await self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True
        )

        try:
            content = response["content"]
            parsed = json.loads(content)
            return {
                "data": parsed,
                "usage": response["usage"],
                "cost": response["cost"],
                "cached": response["cached"]
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw content: {response['content'][:500]}")
            raise

    async def analyze(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_response: bool = False,
        use_premium: bool = False
    ) -> Dict[str, Any]:
        """
        Simplified analysis request

        Args:
            system_prompt: System instructions
            user_prompt: User query/data
            temperature: Override temperature
            max_tokens: Override max_tokens
            json_response: Whether to expect JSON
            use_premium: Whether to use premium model (gpt-5.1) for critical decisions

        Returns:
            Response dict
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if json_response:
            # For JSON responses, we need to pass use_premium to the underlying chat call
            response = await self.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=True,
                use_premium=use_premium
            )
            try:
                content = response["content"]
                parsed = json.loads(content)
                return {
                    "data": parsed,
                    "usage": response["usage"],
                    "cost": response["cost"],
                    "cached": response["cached"],
                    "model_used": response.get("model_used"),
                    "is_premium": response.get("is_premium", use_premium)
                }
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                raise
        else:
            return await self.chat(messages, temperature, max_tokens, use_premium=use_premium)

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get token usage statistics"""
        return {
            "total_requests": self.total_requests,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_cost": self.total_cost,
            "avg_cost_per_request": self.total_cost / max(1, self.total_requests),
            "model": self.model
        }

    def clear_cache(self):
        """Clear response cache"""
        self._cache.clear()
        logger.info("GPT response cache cleared")

    def cleanup_cache(self):
        """Remove expired cache entries"""
        now = datetime.now()
        expired = [
            key for key, entry in self._cache.items()
            if now - entry.get("timestamp", now) > self._cache_ttl
        ]
        for key in expired:
            del self._cache[key]
        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired cache entries")
