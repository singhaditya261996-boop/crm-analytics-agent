"""
agent/llm_client.py — Unified LLM abstraction (async-primary) for Ollama and Groq.

Async-primary API
-----------------
await client.complete(system, user) -> str
await client.complete_with_schema(schema_context, question) -> str
await client.startup_check() -> None

Sync backward-compat wrappers
------------------------------
client.chat(messages, temperature) -> str
client.vision_chat(messages, image_bytes) -> str
client.health_check() -> bool

Provider strategy: Ollama (local) first → Groq (cloud) fallback on any failure.
Every successful and failed call is logged to exports/llm_log.jsonl (stays local).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"

_SCHEMA_SYSTEM_TEMPLATE = """\
You are a CRM data analyst. You have access to the following table schemas:

{schema_context}

Answer questions about this data clearly and concisely.
If you need to reference specific columns or tables, use the exact names shown above.
"""


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# LLMClient
# ─────────────────────────────────────────────────────────────────────────────


class LLMClient:
    """
    Async-primary unified interface for Ollama (local) and Groq (cloud fallback).

    Usage
    -----
    client = LLMClient()
    await client.startup_check()                           # recommended on boot
    response = await client.complete("You are…", "Q?")    # primary async API
    response = client.chat([{"role": "user", …}])         # sync backward-compat
    """

    def __init__(self, provider: str | None = None) -> None:
        self.config = _load_config()
        self.provider = (
            provider
            or os.getenv("LLM_PROVIDER")
            or self.config.get("llm_provider", "ollama")
        ).lower()

        ollama_cfg: dict = self.config.get("ollama", {})
        groq_cfg: dict = self.config.get("groq", {})
        exports_cfg: dict = self.config.get("exports", {})

        # Ollama settings
        self._ollama_base_url: str = (
            os.getenv("OLLAMA_BASE_URL") or ollama_cfg.get("base_url", "http://localhost:11434")
        )
        self._ollama_model: str = ollama_cfg.get("model", "llama3.1:8b")
        self._vision_model: str = ollama_cfg.get("vision_model", "llava")
        self._timeout: int = ollama_cfg.get("timeout_seconds", 60)
        self._max_retries: int = ollama_cfg.get("max_retries", 3)
        self._temperature: float = ollama_cfg.get("temperature", 0.1)

        # Groq settings
        self._groq_api_key: str = (
            os.getenv("GROQ_API_KEY") or groq_cfg.get("api_key", "")
        )
        self._groq_model: str = groq_cfg.get("model", "llama-3.1-70b-versatile")
        self._groq_enabled: bool = bool(groq_cfg.get("enabled", True))

        # Logging
        self._log_calls: bool = bool(exports_cfg.get("log_llm_calls", True))
        self._log_path: Path = (
            Path(exports_cfg.get("output_folder", "exports")) / "llm_log.jsonl"
        )

        # Availability flag (optimistic; confirmed by startup_check)
        self._ollama_available: bool = True

        # Dedicated thread pool for running async code from sync callers
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm_sync")

        # Legacy alias used by backward-compat sync methods
        self._model = self._ollama_model if self.provider == "ollama" else self._groq_model

    # ── Startup ──────────────────────────────────────────────────────────────

    async def startup_check(self) -> None:
        """
        Probe Ollama at startup.

        Sets ``self._ollama_available`` and prints a clear user-facing message
        if Ollama is not reachable, including instructions to start it.
        """
        try:
            import ollama as _ollama  # type: ignore

            probe = _ollama.AsyncClient(host=self._ollama_base_url)
            await asyncio.wait_for(probe.list(), timeout=5.0)
            self._ollama_available = True
            logger.info(
                "Ollama is reachable at %s (model: %s)",
                self._ollama_base_url,
                self._ollama_model,
            )
        except Exception as exc:
            self._ollama_available = False
            url_field = self._ollama_base_url[:40].ljust(40)
            msg = (
                "\n"
                "╔══════════════════════════════════════════════════════════╗\n"
                "║  Ollama is not running.                                  ║\n"
                "║                                                          ║\n"
                "║  Please start it:  ollama serve                          ║\n"
                f"║  Expected URL:     {url_field}║\n"
                "║                                                          ║\n"
                "║  Groq fallback will be used if GROQ_API_KEY is set.      ║\n"
                "╚══════════════════════════════════════════════════════════╝\n"
            )
            print(msg)
            logger.warning("Ollama unavailable: %s", exc)

    # ── Primary async API ─────────────────────────────────────────────────────

    async def complete(
        self,
        system: str,
        user: str,
        temperature: float | None = None,
    ) -> str:
        """
        Send a system + user prompt pair and return the assistant reply.

        Tries Ollama first; falls back to Groq on any error or when Ollama
        has been marked unavailable by ``startup_check``.
        Every call is logged to ``exports/llm_log.jsonl``.
        """
        temp = temperature if temperature is not None else self._temperature
        t0 = time.monotonic()
        provider_used = "ollama"
        response = ""

        try:
            if self._ollama_available:
                response = await self._ollama_complete(system, user, temp)
            else:
                raise RuntimeError("Ollama marked unavailable — skipping to Groq.")
        except Exception as ollama_exc:
            logger.warning("Ollama failed (%s); trying Groq fallback.", ollama_exc)
            provider_used = "groq"
            try:
                response = await self._groq_complete(system, user, temp)
            except Exception as groq_exc:
                error_str = f"Ollama: {ollama_exc} | Groq: {groq_exc}"
                latency_ms = int((time.monotonic() - t0) * 1000)
                await self._log_call(
                    provider="both_failed",
                    system=system,
                    user=user,
                    response="",
                    latency_ms=latency_ms,
                    error=error_str,
                )
                raise RuntimeError(
                    f"Both providers failed.\n  Ollama: {ollama_exc}\n  Groq: {groq_exc}"
                ) from groq_exc

        latency_ms = int((time.monotonic() - t0) * 1000)
        await self._log_call(
            provider=provider_used,
            system=system,
            user=user,
            response=response,
            latency_ms=latency_ms,
            error=None,
        )
        return response

    async def complete_with_schema(self, schema_context: str, question: str) -> str:
        """
        Inject table schemas into the system prompt then call ``complete()``.

        Parameters
        ----------
        schema_context : plaintext / markdown description of available tables/columns.
        question       : the user's natural-language question.
        """
        system = _SCHEMA_SYSTEM_TEMPLATE.format(schema_context=schema_context)
        return await self.complete(system, question)

    # ── Ollama internals ──────────────────────────────────────────────────────

    async def _ollama_complete(self, system: str, user: str, temperature: float) -> str:
        import ollama as _ollama  # type: ignore

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        client = _ollama.AsyncClient(host=self._ollama_base_url)
        result: str = ""

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                resp = await asyncio.wait_for(
                    client.chat(
                        model=self._ollama_model,
                        messages=messages,
                        options={"temperature": temperature},
                    ),
                    timeout=self._timeout,
                )
                result = resp["message"]["content"]

        return result

    # ── Groq internals ────────────────────────────────────────────────────────

    async def _groq_complete(self, system: str, user: str, temperature: float) -> str:
        if not self._groq_enabled:
            raise RuntimeError("Groq is disabled in settings.yaml.")
        if not self._groq_api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Add it to .env or config/settings.yaml."
            )

        try:
            from groq import AsyncGroq  # type: ignore
        except ImportError as exc:
            raise ImportError("Install the 'groq' package: pip install groq") from exc

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        client = AsyncGroq(api_key=self._groq_api_key)
        result: str = ""

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                resp = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=self._groq_model,
                        messages=messages,
                        temperature=temperature,
                    ),
                    timeout=self._timeout,
                )
                result = resp.choices[0].message.content

        return result

    # ── JSONL call logger ─────────────────────────────────────────────────────

    async def _log_call(
        self,
        *,
        provider: str,
        system: str,
        user: str,
        response: str,
        latency_ms: int,
        error: str | None,
    ) -> None:
        if not self._log_calls:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": self._ollama_model if "ollama" in provider else self._groq_model,
            "system_snippet": system[:200],
            "user_snippet": user[:200],
            "response_snippet": response[:400],
            "latency_ms": latency_ms,
            "error": error,
        }
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._append_log, json.dumps(record))
        except Exception as exc:
            logger.warning("Failed to write LLM log: %s", exc)

    def _append_log(self, line: str) -> None:
        with open(self._log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # ── Sync helper ───────────────────────────────────────────────────────────

    def _run_sync(self, coro: Any) -> Any:
        """
        Run an async coroutine from a synchronous context.

        If there is already a running event loop (e.g. inside Streamlit), the
        coroutine is submitted to a dedicated ``ThreadPoolExecutor`` so it runs
        in a fresh loop on a worker thread, avoiding nested-loop errors.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            future = self._executor.submit(asyncio.run, coro)
            return future.result()
        return asyncio.run(coro)

    # ── Sync backward-compat wrappers ─────────────────────────────────────────

    def chat(self, messages: list[dict], temperature: float | None = None) -> str:
        """
        Sync wrapper — converts a messages list to system/user and calls complete().

        Parameters
        ----------
        messages : list of {"role": str, "content": str}
        temperature : float | None — overrides settings.yaml if provided
        """
        system = ""
        user_parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system = content
            else:
                user_parts.append(content)
        user = "\n".join(user_parts) or "Hello"
        return self._run_sync(self.complete(system, user, temperature))

    def vision_chat(self, messages: list[dict], image_bytes: bytes) -> str:
        """
        Sync vision chat — only supported with Ollama (llava model).

        Parameters
        ----------
        messages    : list of {"role": str, "content": str}
        image_bytes : raw image bytes (PNG / JPEG)
        """
        import base64
        import ollama as _ollama  # type: ignore

        encoded = base64.b64encode(image_bytes).decode()
        augmented = list(messages)
        if augmented and augmented[-1]["role"] == "user":
            augmented[-1] = {**augmented[-1], "images": [encoded]}
        else:
            augmented.append({"role": "user", "content": "", "images": [encoded]})

        client = _ollama.Client(host=self._ollama_base_url)
        resp = client.chat(model=self._vision_model, messages=augmented)
        return resp["message"]["content"]

    def health_check(self) -> bool:
        """Return True if at least one provider is reachable."""
        try:
            self._run_sync(self.startup_check())
            return self._ollama_available or bool(self._groq_api_key)
        except Exception as exc:
            logger.warning("Health check failed: %s", exc)
            return False
