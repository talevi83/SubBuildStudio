"""
Gemini generate_content with an optional Flex service tier.

Flex (https://ai.google.dev/gemini-api/docs/flex-inference) bills tokens at half the Standard
price, but Google may take minutes to answer and sheds Flex traffic first when it is busy
(429 / 503). Google does not fall back to Standard on its own, so this module does:

  1. send the request with service_tier="flex" and a client timeout of FLEX_TIMEOUT_MS (2 min),
  2. on a capacity error, retry with exponential backoff (5s, 10s),
  3. if Flex is still busy (or a request timed out), send it once on the Standard tier so the
     job finishes; the usage tracker counts these fallbacks so the job can warn about them.

  4. after FLEX_FALLBACK_LIMIT "bad" Flex requests in a row for a model - a fallback, or an answer
     slower than FLEX_SLOW_SECONDS - Flex enters a temporary cooldown (FLEX_COOLDOWN_REQUESTS calls
     or FLEX_COOLDOWN_SECONDS) followed by a periodic single-attempt re-probe via FlexState so
     discounted Flex pricing resumes automatically once Google capacity recovers.

Errors that are not about capacity (bad key, unknown model, invalid request) are raised at once
so the callers' own error handling stays unchanged.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from app.core.model_catalog import is_capacity_error, is_timeout_error

logger = logging.getLogger(__name__)

# Flex answers slower than FLEX_SLOW_SECONDS already count as overloaded; job_5b4c9f644a stalled
# 15 min for a 504 DEADLINE_EXCEEDED while Standard answered in 8.6s.
FLEX_TIMEOUT_MS = 120_000
# Asyncio safety net in case the SDK ignores http_options.timeout on a hung connection.
FLEX_CLIENT_GUARD_SECONDS = FLEX_TIMEOUT_MS / 1000 + 15.0
FLEX_ATTEMPTS = 3
FLEX_BASE_DELAY = 5.0
FLEX_FALLBACK_LIMIT = 2
FLEX_SLOW_SECONDS = 60.0
FLEX_COOLDOWN_REQUESTS = 12
FLEX_COOLDOWN_SECONDS = 90.0


@dataclass
class FlexState:
    """Per-job Flex bookkeeping with temporary cooldown and periodic single-shot re-probe.

    After `limit` consecutive slow/busy Flex requests on `model`, Flex enters a temporary cooldown
    rather than permanently disabling discounted Flex pricing for the remainder of the job. Once
    `cooldown_requests` calls have skipped Flex OR `cooldown_seconds` have elapsed, the next call
    performs a fast single-attempt re-probe on Flex: if Google capacity has recovered, full Flex
    resumes automatically; if still overloaded, cooldown restarts without multi-retry delay.
    """

    limit: int = FLEX_FALLBACK_LIMIT
    cooldown_requests: int = FLEX_COOLDOWN_REQUESTS
    cooldown_seconds: float = FLEX_COOLDOWN_SECONDS
    consecutive: Dict[str, int] = field(default_factory=dict)
    _cooldown_skips_left: Dict[str, int] = field(default_factory=dict)
    _cooldown_until: Dict[str, float] = field(default_factory=dict)
    _reprobing: Set[str] = field(default_factory=set)

    @property
    def disabled(self) -> Set[str]:
        """Models currently in active Flex cooldown (not yet ready to re-probe)."""
        now = time.monotonic()
        return {
            m
            for m, skips in self._cooldown_skips_left.items()
            if skips > 0 and now < self._cooldown_until.get(m, 0.0)
        }

    def is_reprobing(self, model: str) -> bool:
        return model in self._reprobing

    def use_flex(self, model: str) -> bool:
        if model not in self._cooldown_skips_left:
            return True
        now = time.monotonic()
        skips = self._cooldown_skips_left.get(model, 0)
        until = self._cooldown_until.get(model, 0.0)
        if skips <= 0 or now >= until:
            # Cooldown expired: allow one fast re-probe on Flex
            self._cooldown_skips_left.pop(model, None)
            self._cooldown_until.pop(model, None)
            self._reprobing.add(model)
            logger.info(f"Re-probing Flex tier availability for {model} after temporary cooldown")
            return True
        self._cooldown_skips_left[model] = skips - 1
        return False

    def succeeded(self, model: str, seconds: float = 0.0) -> None:
        was_reprobing = model in self._reprobing
        self._reprobing.discard(model)
        if seconds > FLEX_SLOW_SECONDS:
            self._bad(model, f"answered slowly ({seconds:.0f}s)", force_cooldown=was_reprobing)
        else:
            self.consecutive[model] = 0
            self._cooldown_skips_left.pop(model, None)
            self._cooldown_until.pop(model, None)
            if was_reprobing:
                logger.info(f"Flex re-probe succeeded for {model} ({seconds:.1f}s); resuming discounted Flex tier")

    def fell_back(self, model: str) -> None:
        was_reprobing = model in self._reprobing
        self._reprobing.discard(model)
        self._bad(model, "was busy", force_cooldown=was_reprobing)

    def _bad(self, model: str, what: str, *, force_cooldown: bool = False) -> None:
        self.consecutive[model] = self.consecutive.get(model, 0) + 1
        if force_cooldown or self.consecutive[model] >= self.limit:
            self._cooldown_skips_left[model] = max(1, int(self.cooldown_requests))
            self._cooldown_until[model] = time.monotonic() + max(0.0, float(self.cooldown_seconds))
            logger.warning(
                f"Flex {what} for {self.consecutive[model]} request(s) in a row to {model}; "
                f"cooling down to Standard for the next {self.cooldown_requests} request(s) or {self.cooldown_seconds:.0f}s"
            )


def flex_config(config: Any) -> Any:
    """Copy of a GenerateContentConfig that asks for the Flex tier."""
    from google.genai import types

    return config.model_copy(update={
        "service_tier": types.ServiceTier.FLEX,
        "http_options": types.HttpOptions(timeout=FLEX_TIMEOUT_MS),
    })


async def generate_content(
    client: Any,
    *,
    model: str,
    contents: Any,
    config: Any,
    flex: bool = False,
    usage: Any = None,
    stage: str = "",
    flex_state: Optional[FlexState] = None,
    timeout: Optional[float] = None,
) -> Any:
    """Run client.models.generate_content in a worker thread (the SDK call is blocking) and
    record its token usage. With `flex`, use the Flex tier with retries and a Standard fallback."""

    async def call(cfg: Any, *, flex_guard: bool = False) -> Any:
        coro = asyncio.to_thread(client.models.generate_content, model=model, contents=contents, config=cfg)
        if flex_guard:
            return await asyncio.wait_for(coro, timeout=FLEX_CLIENT_GUARD_SECONDS)
        if timeout is not None and timeout > 0:
            return await asyncio.wait_for(coro, timeout=timeout)
        return await coro

    wanted_flex = bool(flex)
    skipped_by_cooldown = False
    if flex and flex_state is not None and not flex_state.use_flex(model):
        flex = False  # Flex is in temporary cooldown; route straight to Standard
        skipped_by_cooldown = True
    if flex:
        reprobing = bool(flex_state is not None and flex_state.is_reprobing(model))
        max_attempts = 1 if reprobing else FLEX_ATTEMPTS
        cfg = flex_config(config)
        last_error: Optional[Exception] = None
        for attempt in range(max_attempts):
            began = time.monotonic()
            try:
                response = await call(cfg, flex_guard=True)
                _record(usage, stage, model, response, flex=True)
                took = time.monotonic() - began
                logger.info(f"Flex request to {model} ({stage}) answered in {took:.0f}s")
                if flex_state is not None:
                    flex_state.succeeded(model, took)
                return response
            except Exception as e:
                timed_out = (
                    is_timeout_error(e)
                    or isinstance(e, (asyncio.TimeoutError, TimeoutError))
                    or "DEADLINE_EXCEEDED" in str(e).upper()
                )
                if not (timed_out or is_capacity_error(e)):
                    if flex_state is not None:
                        flex_state._reprobing.discard(model)
                    raise
                last_error = e
                logger.warning(f"Flex request to {model} failed (attempt {attempt + 1}/{max_attempts}): {e}")
                if timed_out:
                    break  # already waited the full Flex timeout; don't wait it out again
                if attempt + 1 < max_attempts:
                    await asyncio.sleep(FLEX_BASE_DELAY * (2 ** attempt))
        logger.warning(f"Flex unavailable for {model} ({last_error}); sending this request on the Standard tier")
        response = await call(config)
        _record(usage, stage, model, response, flex=False, fell_back=True)
        if flex_state is not None:
            flex_state.fell_back(model)
        return response

    response = await call(config)
    _record(usage, stage, model, response, flex=False, fell_back=bool(wanted_flex and skipped_by_cooldown))
    return response


def _record(usage: Any, stage: str, model: str, response: Any, flex: bool = False, fell_back: bool = False) -> None:
    if usage is not None:
        usage.record(stage, model, getattr(response, "usage_metadata", None), flex=flex, fell_back=fell_back)
