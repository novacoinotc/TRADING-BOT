"""
GPT Client - Advanced OpenAI API integration for Trading

Based on GPT-5 Trading Integration Guide:
- Dual endpoint support: /v1/responses (recommended) and /v1/chat/completions (legacy)
- Reasoning effort control for cost optimization
- JSON Schema strict validation
- Automatic retry with exponential backoff
- Token usage tracking and daily cost limits
"""

import asyncio
import logging
import json
import hashlib
import httpx
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from openai import AsyncOpenAI
from openai import APIError, RateLimitError, APIConnectionError
from config import config

logger = logging.getLogger(__name__)


class GPTClient:
    """
    Advanced GPT Client with support for:
    - /v1/responses endpoint (GPT-5 recommended)
    - /v1/chat/completions endpoint (legacy compatibility)
    - Reasoning effort control (none/low/medium/high)
    - JSON Schema strict validation
    """

    # Reasoning effort levels for /v1/responses
    REASONING_NONE = "none"      # Fastest, cheapest - for simple decisions
    REASONING_LOW = "low"        # Light reasoning - for normal trading decisions
    REASONING_MEDIUM = "medium"  # More depth - for complex analysis
    REASONING_HIGH = "high"      # Maximum reasoning - for critical decisions

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5-mini",
        temperature: float = 0.2,
        max_tokens: int = 100000,
        timeout: float = 60.0
    ):
        """
        Initialize GPT Client

        Args:
            api_key: OpenAI API key
            model: Model to use (gpt-5-mini, gpt-5.1)
            temperature: Creativity level (0.0-1.0) - trading recommends 0.0-0.2
            max_tokens: Maximum response tokens
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

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
            # GPT-5 Models (Latest - recommended for trading)
            "gpt-5-mini": {"input": 0.20, "output": 0.80},
            "gpt-5.1": {"input": 5.00, "output": 15.00},
            # GPT-4 Models (Legacy)
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-4": {"input": 30.00, "output": 60.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
            # Reasoning models (NO temperature support - only default 1.0!)
            "o1": {"input": 15.00, "output": 60.00},
            "o1-mini": {"input": 3.00, "output": 12.00},
            "o1-preview": {"input": 15.00, "output": 60.00},
        }

        # Model routing: frequent (cheap) vs premium (powerful)
        # gpt-5-mini for 95% of calls (economical), gpt-5.1 for critical analysis
        self.model_frequent = model
        self.model_premium = "gpt-5.1"

        # Endpoint preference - use /v1/responses for GPT-5 models
        self.use_responses_endpoint = True

        logger.info(f"GPT Client initialized with model: {model} (temperature={temperature})")

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
        # Find matching pricing (handle model variants)
        pricing = None
        for model_key in self._pricing:
            if model_key in model_to_use.lower():
                pricing = self._pricing[model_key]
                break
        if not pricing:
            # Default to gpt-5-mini pricing as fallback
            pricing = self._pricing.get("gpt-5-mini", {"input": 0.20, "output": 0.80})

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
        """Check if daily cost limit has been reached."""
        today = datetime.now().date()

        if today > self._daily_cost_reset_date:
            logger.info("🔄 Resetting daily GPT cost counter (new day)")
            self.total_cost = 0.0
            self._daily_cost_reset_date = today

        if self.total_cost >= self._daily_cost_limit:
            logger.warning(
                f"⚠️ Daily GPT cost limit reached: ${self.total_cost:.2f} >= ${self._daily_cost_limit:.2f}"
            )
            return False
        return True

    def _is_reasoning_model(self, model: str) -> bool:
        """
        Check if model is a reasoning-only model - these only support temperature=1.0

        Reasoning-only models (NO temperature support):
        - o1, o1-mini, o1-preview

        Models that DO support temperature:
        - gpt-5-mini, gpt-5.1 (GPT-5 family - supports 0.0-2.0)
        - gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
        """
        model_lower = model.lower()
        # O1 family ONLY - these don't support temperature
        o1_models = ['o1', 'o1-mini', 'o1-preview']
        return any(rm == model_lower or model_lower.startswith(f"{rm}-") for rm in o1_models)

    def _supports_temperature(self, model: str) -> bool:
        """
        Check if model supports custom temperature values.

        Models that DON'T support custom temperature (only default 1.0):
        - o1, o1-mini, o1-preview (reasoning-only models)
        - gpt-5-mini, gpt-5.1 (GPT-5 family) - confirmed via API error

        Models that DO support custom temperature (0.0-2.0):
        - gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
        """
        # Reasoning models and GPT-5 don't support custom temperature
        if self._is_reasoning_model(model):
            return False
        # GPT-5 models also don't support custom temperature
        if self._is_gpt5_model(model):
            return False
        # GPT-4 and older models support custom temperature
        return True

    def _is_gpt5_model(self, model: str) -> bool:
        """Check if model is GPT-5 family (supports /v1/responses endpoint)"""
        model_lower = model.lower()
        return 'gpt-5' in model_lower or 'gpt5' in model_lower

    def set_models(self, frequent: str, premium: str):
        """Configure model routing"""
        self.model_frequent = frequent
        self.model_premium = premium
        logger.info(f"Model routing configured: frequent={frequent}, premium={premium}")

    async def chat_with_responses_endpoint(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_schema: Optional[Dict] = None,
        reasoning_effort: str = "low",
        use_premium: bool = False
    ) -> Dict[str, Any]:
        """
        Send request using /v1/responses endpoint (GPT-5 recommended).

        This endpoint provides:
        - Better reasoning control
        - JSON Schema support
        - More consistent responses

        Args:
            messages: List of message dicts
            temperature: Override temperature (0.0-1.0)
            max_tokens: Max output tokens
            json_schema: Optional JSON schema for structured output
            reasoning_effort: none/low/medium/high
            use_premium: Use premium model

        Returns:
            Response dict with content, usage, cost
        """
        await self._rate_limit()

        if not self._check_daily_cost_limit():
            raise Exception(f"Daily GPT cost limit exceeded (${self.total_cost:.2f})")

        selected_model = self.model_premium if use_premium else self.model_frequent

        # Build request body for /v1/responses
        # Note: This endpoint uses 'input' instead of 'messages'
        # IMPORTANT: /v1/responses does NOT support temperature - only default (1)
        request_body = {
            "model": selected_model,
            "input": messages,
            "max_output_tokens": max_tokens or self.max_tokens,
        }
        # NOTE: Do NOT add temperature for /v1/responses - it only supports default (1)

        # Add reasoning effort for GPT-5 models
        if self._is_gpt5_model(selected_model) or self._is_reasoning_model(selected_model):
            request_body["reasoning"] = {"effort": reasoning_effort}

        # Add JSON schema using text.format for /v1/responses
        # CORRECT structure: text.format = {type, name, strict, schema}
        if json_schema:
            request_body["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": json_schema.get("name", "response"),
                    "strict": json_schema.get("strict", True),
                    "schema": json_schema.get("schema", json_schema)
                }
            }

        # Make HTTP request directly (openai library may not support this endpoint yet)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as http_client:
                    response = await http_client.post(
                        "https://api.openai.com/v1/responses",
                        headers=headers,
                        json=request_body
                    )

                    if response.status_code == 404:
                        # Endpoint not available, fallback to chat completions
                        logger.warning("/v1/responses not available, falling back to /v1/chat/completions")
                        return await self.chat(
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            json_mode=json_schema is not None,
                            use_premium=use_premium
                        )

                    if response.status_code == 429:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Rate limit hit, retrying in {delay}s")
                        await asyncio.sleep(delay)
                        continue

                    if response.status_code != 200:
                        try:
                            error_data = response.json() if response.content else {}
                        except json.JSONDecodeError:
                            error_data = {}
                        error_msg = error_data.get("error", {}).get("message", response.text[:500])
                        raise Exception(f"API error {response.status_code}: {error_msg}")

                    # Parse JSON response with error handling
                    try:
                        data = response.json()
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in /v1/responses: {e}")
                        logger.error(f"Response content: {response.text[:500]}")
                        # Fallback to chat/completions
                        return await self.chat(
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            json_mode=json_schema is not None,
                            use_premium=use_premium
                        )

                # Check response status - handle incomplete/truncated responses
                response_status = data.get("status", "completed")
                if response_status == "incomplete":
                    incomplete_reason = data.get("incomplete_details", {}).get("reason", "unknown")
                    logger.warning(f"⚠️ /v1/responses status=incomplete, reason={incomplete_reason}")
                    # If max_tokens was the issue, fallback to chat/completions with higher tokens
                    if incomplete_reason == "max_output_tokens":
                        logger.warning(f"⚠️ Response truncated due to max_output_tokens limit - falling back to chat/completions with higher limit")
                        # Fallback to chat/completions which may handle this better
                        return await self.chat(
                            messages=messages,
                            temperature=temperature,
                            max_tokens=100000,  # Use very high token limit for fallback
                            json_mode=json_schema is not None,
                            use_premium=use_premium
                        )

                # Extract response content
                # /v1/responses format may differ from chat completions
                output = data.get("output", [])
                content = ""
                if output:
                    for item in output:
                        if item.get("type") == "message":
                            for content_item in item.get("content", []):
                                if content_item.get("type") == "output_text" or content_item.get("type") == "text":
                                    content = content_item.get("text", "")
                                    break
                            if content:
                                break

                # If no content found in new format, try legacy format
                if not content and "choices" in data:
                    try:
                        content = data["choices"][0]["message"]["content"]
                    except (IndexError, KeyError, TypeError) as e:
                        logger.warning(f"Could not extract content from legacy format: {e}")

                # Validate content is not empty
                if not content or content.strip() == "":
                    logger.error(f"❌ /v1/responses returned empty content. Data keys: {data.keys()}")
                    logger.error(f"   Output: {output}")
                    logger.error(f"   Status: {response_status}")
                    raise Exception(f"GPT /v1/responses returned empty content (status={response_status})")

                # If response was incomplete but we got partial content, log warning
                if response_status == "incomplete" and content:
                    logger.warning(f"⚠️ Using partial content from incomplete response (length={len(content)})")

                usage = data.get("usage", {})
                prompt_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
                completion_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))

                # Update tracking
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_requests += 1
                cost = self._calculate_cost(prompt_tokens, completion_tokens, selected_model)
                self.total_cost += cost

                return {
                    "content": content,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    },
                    "cost": cost,
                    "cached": False,
                    "model_used": selected_model,
                    "is_premium": use_premium,
                    "reasoning_effort": reasoning_effort
                }

            except httpx.TimeoutException:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Request timeout, retrying in {delay}s")
                await asyncio.sleep(delay)

            except Exception as e:
                if "404" in str(e) or "not found" in str(e).lower():
                    # Fallback to chat completions
                    return await self.chat(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        json_mode=json_schema is not None,
                        use_premium=use_premium
                    )
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Error: {e}, retrying in {delay}s")
                    await asyncio.sleep(delay)
                else:
                    raise

        raise Exception(f"Failed after {max_retries} retries")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
        json_mode: bool = False,
        json_schema: Optional[Dict] = None,
        use_premium: bool = False,
        reasoning_effort: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send chat completion request to GPT using /v1/chat/completions.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            use_cache: Whether to use response caching
            json_mode: Whether to request JSON response format
            json_schema: Optional strict JSON schema
            use_premium: Whether to use premium model
            reasoning_effort: Optional reasoning effort for compatible models

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

        if not self._check_daily_cost_limit():
            raise Exception(
                f"Daily GPT cost limit exceeded (${self.total_cost:.2f} >= ${self._daily_cost_limit:.2f})"
            )

        await self._rate_limit()

        selected_model = self.model_premium if use_premium else self.model_frequent
        if use_premium:
            logger.info(f"🧠 Using PREMIUM model: {selected_model}")

        # Prepare request params
        params = {
            "model": selected_model,
            "messages": messages,
        }

        model_lower = selected_model.lower()
        supports_temp = self._supports_temperature(selected_model)

        # Log the model being used for debugging
        logger.info(f"🤖 GPT Request: model={selected_model}, supports_temperature={supports_temp}")

        # Temperature handling - CRITICAL: some models only support default (1.0)
        if supports_temp:
            # GPT-5, GPT-4 models support custom temperature (0.0-2.0)
            temp_value = temperature if temperature is not None else self.temperature
            params["temperature"] = temp_value
            logger.debug(f"✅ Using temperature={temp_value} (model supports custom temperature)")
        else:
            # Reasoning models (o1, o1-mini, o1-preview) only support default temperature (1.0)
            # Do NOT send temperature parameter - API will reject any value except default
            logger.warning(f"⚠️ Model {selected_model} only supports default temperature - OMITTING parameter")

        # Token parameter
        tokens_value = max_tokens if max_tokens is not None else self.max_tokens
        newer_models = ['gpt-4o', 'gpt-4o-mini', 'o1', 'o1-mini', 'o1-preview', 'gpt-4-turbo', 'gpt-5', 'chatgpt-4o']
        if any(model in model_lower for model in newer_models):
            params["max_completion_tokens"] = tokens_value
        else:
            params["max_tokens"] = tokens_value

        # Response format
        if json_schema:
            # Use strict JSON schema
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": json_schema
            }
        elif json_mode:
            params["response_format"] = {"type": "json_object"}

        # Add reasoning_effort for chat/completions if supported
        is_reasoning = self._is_reasoning_model(selected_model)
        if reasoning_effort and (self._is_gpt5_model(selected_model) or is_reasoning):
            params["reasoning_effort"] = reasoning_effort

        # Retry logic with intelligent temperature handling
        max_retries = 3
        base_delay = 1.0
        temperature_retry_done = False

        for attempt in range(max_retries):
            try:
                logger.debug(f"📤 GPT Request attempt {attempt + 1}: model={selected_model}, params_keys={list(params.keys())}")
                response = await self.client.chat.completions.create(**params)

                # Check finish_reason for truncation
                finish_reason = response.choices[0].finish_reason
                if finish_reason == "length":
                    logger.warning(f"⚠️ Response truncated (finish_reason=length) - max_tokens may be too low")

                content = response.choices[0].message.content

                # Validate content is not None/empty
                if content is None or content.strip() == "":
                    logger.error(f"❌ GPT returned empty/None content. finish_reason={finish_reason}")
                    logger.error(f"   Response: {response}")
                    raise Exception(f"GPT returned empty response content (finish_reason={finish_reason})")

                usage = response.usage

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

                if use_cache:
                    self._cache[cache_key] = {
                        "response": result,
                        "timestamp": datetime.now()
                    }

                logger.debug(f"GPT request successful: {usage.total_tokens} tokens, ${cost:.4f}")
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
                error_str = str(e)
                logger.error(f"OpenAI API error: {e}")

                # Handle temperature not supported error - retry without temperature
                if "temperature" in error_str.lower() and "unsupported" in error_str.lower():
                    if not temperature_retry_done and "temperature" in params:
                        logger.warning(f"⚠️ Model {selected_model} rejected temperature parameter - retrying WITHOUT temperature")
                        del params["temperature"]
                        temperature_retry_done = True
                        continue  # Retry immediately without temperature
                    else:
                        logger.error(f"Temperature retry already done, raising error")
                raise

            except Exception as e:
                error_str = str(e)
                logger.error(f"Unexpected error in GPT request: {e}")

                # Also catch temperature errors from generic exceptions
                if "temperature" in error_str.lower() and "unsupported" in error_str.lower():
                    if not temperature_retry_done and "temperature" in params:
                        logger.warning(f"⚠️ Model {selected_model} rejected temperature - retrying WITHOUT temperature")
                        del params["temperature"]
                        temperature_retry_done = True
                        continue  # Retry immediately without temperature
                raise

        raise Exception(f"Failed after {max_retries} retries")

    async def analyze(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_response: bool = False,
        json_schema: Optional[Dict] = None,
        use_premium: bool = False,
        reasoning_effort: str = "low"
    ) -> Dict[str, Any]:
        """
        Simplified analysis request with full GPT-5 features.

        Args:
            system_prompt: System instructions
            user_prompt: User query/data
            temperature: Override temperature
            max_tokens: Override max_tokens
            json_response: Whether to expect JSON
            json_schema: Optional strict JSON schema
            use_premium: Use premium model (gpt-5.1)
            reasoning_effort: none/low/medium/high

        Returns:
            Response dict with parsed data if JSON
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Try /v1/responses first for GPT-5 models
        selected_model = self.model_premium if use_premium else self.model_frequent
        if self.use_responses_endpoint and self._is_gpt5_model(selected_model):
            try:
                response = await self.chat_with_responses_endpoint(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_schema=json_schema if json_response else None,
                    reasoning_effort=reasoning_effort,
                    use_premium=use_premium
                )
            except Exception as e:
                logger.warning(f"/v1/responses failed, using /v1/chat/completions: {e}")
                response = await self.chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_response,
                    json_schema=json_schema,
                    use_premium=use_premium,
                    reasoning_effort=reasoning_effort
                )
        else:
            response = await self.chat(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_response,
                json_schema=json_schema,
                use_premium=use_premium,
                reasoning_effort=reasoning_effort
            )

        if json_response:
            try:
                content = response["content"]
                parsed = json.loads(content)
                return {
                    "data": parsed,
                    "usage": response["usage"],
                    "cost": response["cost"],
                    "cached": response.get("cached", False),
                    "model_used": response.get("model_used"),
                    "is_premium": response.get("is_premium", use_premium),
                    "reasoning_effort": response.get("reasoning_effort", reasoning_effort)
                }
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Raw content: {response['content'][:500]}")
                raise

        return response

    async def analyze_with_schema(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Dict,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_premium: bool = False,
        reasoning_effort: str = "low"
    ) -> Dict[str, Any]:
        """
        Analyze with strict JSON schema validation.

        Args:
            system_prompt: System instructions
            user_prompt: User query
            schema: JSON schema dict (from trading_schemas.py)
            temperature: Override temperature
            max_tokens: Override max tokens
            use_premium: Use premium model
            reasoning_effort: Reasoning effort level

        Returns:
            Response with validated data
        """
        return await self.analyze(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_response=True,
            json_schema=schema,
            use_premium=use_premium,
            reasoning_effort=reasoning_effort
        )

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get token usage statistics"""
        return {
            "total_requests": self.total_requests,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "total_cost": self.total_cost,
            "avg_cost_per_request": self.total_cost / max(1, self.total_requests),
            "model": self.model,
            "daily_limit": self._daily_cost_limit,
            "daily_remaining": self._daily_cost_limit - self.total_cost
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
