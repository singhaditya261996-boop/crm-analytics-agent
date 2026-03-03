"""
tests/test_llm_client.py — Unit tests for agent/llm_client.py (Module 5).

All LLM calls are fully mocked; no real Ollama or Groq connections are made.
"""
from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ─── Lightweight stubs for optional heavy packages ────────────────────────────

# --- ollama stub ---
_ollama_stub = types.ModuleType("ollama")


class _FakeOllamaSync:
    """Synchronous Ollama client stub (used by vision_chat)."""

    def __init__(self, host: str | None = None) -> None:
        self.host = host

    def chat(self, model: str, messages: list, options: dict | None = None) -> dict:
        return {"message": {"content": "ollama_sync_reply"}}


class _FakeOllamaAsync:
    """Asynchronous Ollama client stub (used by complete / startup_check)."""

    def __init__(self, host: str | None = None) -> None:
        self.host = host

    async def chat(
        self, model: str, messages: list, options: dict | None = None
    ) -> dict:
        return {"message": {"content": "ollama_async_reply"}}

    async def list(self) -> dict:
        return {"models": []}


_ollama_stub.Client = _FakeOllamaSync  # type: ignore[attr-defined]
_ollama_stub.AsyncClient = _FakeOllamaAsync  # type: ignore[attr-defined]
sys.modules.setdefault("ollama", _ollama_stub)

# --- groq stub ---
_groq_stub = types.ModuleType("groq")


class _FakeGroqMessage:
    content = "groq_async_reply"


class _FakeGroqChoice:
    message = _FakeGroqMessage()


class _FakeGroqResponse:
    choices = [_FakeGroqChoice()]


class _FakeGroqCompletions:
    async def create(
        self, model: str, messages: list, temperature: float | None = None
    ) -> _FakeGroqResponse:
        return _FakeGroqResponse()


class _FakeGroqChat:
    completions = _FakeGroqCompletions()


class _FakeAsyncGroq:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self.chat = _FakeGroqChat()


_groq_stub.AsyncGroq = _FakeAsyncGroq  # type: ignore[attr-defined]
sys.modules.setdefault("groq", _groq_stub)

# ─── Import module under test ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from agent.llm_client import LLMClient, _SCHEMA_SYSTEM_TEMPLATE  # noqa: E402

# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """LLMClient wired to Ollama with exports redirected to tmp_path."""
    monkeypatch.setenv("GROQ_API_KEY", "sk-test")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    c = LLMClient(provider="ollama")
    c._log_path = tmp_path / "llm_log.jsonl"
    return c


# ─── TestStartupCheck ─────────────────────────────────────────────────────────


class TestStartupCheck:
    def test_available_sets_flag_true(self, client):
        asyncio.run(client.startup_check())
        assert client._ollama_available is True

    def test_unavailable_sets_flag_false(self, client):
        mock_probe = AsyncMock()
        mock_probe.list.side_effect = ConnectionRefusedError("refused")
        with patch.object(sys.modules["ollama"], "AsyncClient", return_value=mock_probe):
            asyncio.run(client.startup_check())
        assert client._ollama_available is False

    def test_unavailable_prints_serve_instruction(self, client, capsys):
        mock_probe = AsyncMock()
        mock_probe.list.side_effect = Exception("unreachable")
        with patch.object(sys.modules["ollama"], "AsyncClient", return_value=mock_probe):
            asyncio.run(client.startup_check())
        out = capsys.readouterr().out
        assert "ollama serve" in out

    def test_unavailable_mentions_groq_fallback(self, client, capsys):
        mock_probe = AsyncMock()
        mock_probe.list.side_effect = Exception("down")
        with patch.object(sys.modules["ollama"], "AsyncClient", return_value=mock_probe):
            asyncio.run(client.startup_check())
        out = capsys.readouterr().out
        assert "Groq" in out

    def test_available_does_not_print_warning(self, client, capsys):
        asyncio.run(client.startup_check())
        out = capsys.readouterr().out
        assert "ollama serve" not in out


# ─── TestOllamaComplete ───────────────────────────────────────────────────────


class TestOllamaComplete:
    def test_returns_content_string(self, client):
        result = asyncio.run(client._ollama_complete("sys", "usr", 0.1))
        assert result == "ollama_async_reply"

    def test_passes_system_role(self, client):
        captured: dict = {}
        mock_chat_client = AsyncMock()

        async def _fake_chat(model, messages, options=None):
            captured["messages"] = messages
            return {"message": {"content": "ok"}}

        mock_chat_client.chat = _fake_chat
        with patch.object(sys.modules["ollama"], "AsyncClient", return_value=mock_chat_client):
            asyncio.run(client._ollama_complete("my system", "my user", 0.1))

        roles = {m["role"]: m["content"] for m in captured["messages"]}
        assert roles.get("system") == "my system"

    def test_passes_user_role(self, client):
        captured: dict = {}
        mock_client = AsyncMock()

        async def _fake_chat(model, messages, options=None):
            captured["messages"] = messages
            return {"message": {"content": "ok"}}

        mock_client.chat = _fake_chat
        with patch.object(sys.modules["ollama"], "AsyncClient", return_value=mock_client):
            asyncio.run(client._ollama_complete("sys", "my question", 0.1))

        roles = {m["role"]: m["content"] for m in captured["messages"]}
        assert roles.get("user") == "my question"

    def test_passes_temperature_in_options(self, client):
        captured: dict = {}
        mock_client = AsyncMock()

        async def _fake_chat(model, messages, options=None):
            captured["options"] = options
            return {"message": {"content": "ok"}}

        mock_client.chat = _fake_chat
        with patch.object(sys.modules["ollama"], "AsyncClient", return_value=mock_client):
            asyncio.run(client._ollama_complete("s", "u", 0.77))

        assert captured["options"]["temperature"] == pytest.approx(0.77)

    def test_uses_configured_model(self, client):
        captured: dict = {}
        mock_client = AsyncMock()

        async def _fake_chat(model, messages, options=None):
            captured["model"] = model
            return {"message": {"content": "ok"}}

        mock_client.chat = _fake_chat
        with patch.object(sys.modules["ollama"], "AsyncClient", return_value=mock_client):
            asyncio.run(client._ollama_complete("s", "u", 0.1))

        assert captured["model"] == client._ollama_model


# ─── TestGroqComplete ─────────────────────────────────────────────────────────


class TestGroqComplete:
    def test_returns_content_string(self, client):
        result = asyncio.run(client._groq_complete("sys", "usr", 0.1))
        assert result == "groq_async_reply"

    def test_raises_environment_error_without_key(self, client):
        client._groq_api_key = ""
        with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
            asyncio.run(client._groq_complete("s", "u", 0.1))

    def test_raises_runtime_error_when_disabled(self, client):
        client._groq_enabled = False
        with pytest.raises(RuntimeError, match="disabled"):
            asyncio.run(client._groq_complete("s", "u", 0.1))

    def test_uses_configured_groq_model(self, client):
        captured: dict = {}

        async def _fake_create(model, messages, temperature=None):
            captured["model"] = model
            return _FakeGroqResponse()

        mock_completions = MagicMock()
        mock_completions.create = _fake_create
        mock_chat = MagicMock()
        mock_chat.completions = mock_completions
        mock_groq_instance = MagicMock()
        mock_groq_instance.chat = mock_chat

        with patch.object(sys.modules["groq"], "AsyncGroq", return_value=mock_groq_instance):
            asyncio.run(client._groq_complete("s", "u", 0.1))

        assert captured["model"] == client._groq_model

    def test_passes_messages_to_groq(self, client):
        captured: dict = {}

        async def _fake_create(model, messages, temperature=None):
            captured["messages"] = messages
            return _FakeGroqResponse()

        mock_completions = MagicMock()
        mock_completions.create = _fake_create
        mock_chat = MagicMock()
        mock_chat.completions = mock_completions
        mock_groq_instance = MagicMock()
        mock_groq_instance.chat = mock_chat

        with patch.object(sys.modules["groq"], "AsyncGroq", return_value=mock_groq_instance):
            asyncio.run(client._groq_complete("sys prompt", "user question", 0.1))

        roles = {m["role"]: m["content"] for m in captured["messages"]}
        assert roles["system"] == "sys prompt"
        assert roles["user"] == "user question"


# ─── TestComplete (Ollama → Groq fallback) ────────────────────────────────────


class TestComplete:
    def test_ollama_success_returns_reply(self, client):
        result = asyncio.run(client.complete("sys", "usr"))
        assert result == "ollama_async_reply"

    def test_falls_back_to_groq_on_ollama_error(self, client):
        client._ollama_available = True

        mock_ollama = AsyncMock()
        mock_ollama.chat.side_effect = ConnectionError("refused")

        with patch.object(sys.modules["ollama"], "AsyncClient", return_value=mock_ollama):
            result = asyncio.run(client.complete("sys", "usr"))

        assert result == "groq_async_reply"

    def test_skips_ollama_when_flagged_unavailable(self, client):
        client._ollama_available = False
        result = asyncio.run(client.complete("sys", "usr"))
        assert result == "groq_async_reply"

    def test_raises_when_both_providers_fail(self, client):
        client._ollama_available = True
        client._groq_api_key = ""  # triggers EnvironmentError in _groq_complete

        mock_ollama = AsyncMock()
        mock_ollama.chat.side_effect = ConnectionError("refused")

        with patch.object(sys.modules["ollama"], "AsyncClient", return_value=mock_ollama):
            with pytest.raises(RuntimeError, match="Both providers failed"):
                asyncio.run(client.complete("sys", "usr"))

    def test_custom_temperature_forwarded(self, client):
        captured: dict = {}
        mock_ollama = AsyncMock()

        async def _fake_chat(model, messages, options=None):
            captured["options"] = options
            return {"message": {"content": "ok"}}

        mock_ollama.chat = _fake_chat
        with patch.object(sys.modules["ollama"], "AsyncClient", return_value=mock_ollama):
            asyncio.run(client.complete("s", "u", temperature=0.42))

        assert captured["options"]["temperature"] == pytest.approx(0.42)

    def test_default_temperature_from_config(self, client):
        captured: dict = {}
        mock_ollama = AsyncMock()

        async def _fake_chat(model, messages, options=None):
            captured["options"] = options
            return {"message": {"content": "ok"}}

        mock_ollama.chat = _fake_chat
        with patch.object(sys.modules["ollama"], "AsyncClient", return_value=mock_ollama):
            asyncio.run(client.complete("s", "u"))

        assert captured["options"]["temperature"] == pytest.approx(client._temperature)

    def test_returns_non_empty_string(self, client):
        result = asyncio.run(client.complete("s", "u"))
        assert isinstance(result, str)
        assert len(result) > 0


# ─── TestCompleteWithSchema ───────────────────────────────────────────────────


class TestCompleteWithSchema:
    def test_schema_appears_in_system_prompt(self, client):
        captured: dict = {}
        mock_ollama = AsyncMock()

        async def _fake_chat(model, messages, options=None):
            captured["messages"] = messages
            return {"message": {"content": "ok"}}

        mock_ollama.chat = _fake_chat
        with patch.object(sys.modules["ollama"], "AsyncClient", return_value=mock_ollama):
            asyncio.run(
                client.complete_with_schema(
                    "TABLE customers (id INT, name VARCHAR)", "How many rows?"
                )
            )

        system_content = next(
            m["content"] for m in captured["messages"] if m["role"] == "system"
        )
        assert "TABLE customers" in system_content

    def test_question_passed_as_user_message(self, client):
        captured: dict = {}
        mock_ollama = AsyncMock()

        async def _fake_chat(model, messages, options=None):
            captured["messages"] = messages
            return {"message": {"content": "ok"}}

        mock_ollama.chat = _fake_chat
        with patch.object(sys.modules["ollama"], "AsyncClient", return_value=mock_ollama):
            asyncio.run(client.complete_with_schema("schema", "What is total revenue?"))

        user_content = next(
            m["content"] for m in captured["messages"] if m["role"] == "user"
        )
        assert "What is total revenue?" in user_content

    def test_returns_non_empty_string(self, client):
        result = asyncio.run(client.complete_with_schema("schema text", "question"))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_uses_schema_system_template(self, client):
        captured: dict = {}
        mock_ollama = AsyncMock()

        async def _fake_chat(model, messages, options=None):
            captured["messages"] = messages
            return {"message": {"content": "ok"}}

        mock_ollama.chat = _fake_chat
        schema = "TABLE orders (id, amount)"
        with patch.object(sys.modules["ollama"], "AsyncClient", return_value=mock_ollama):
            asyncio.run(client.complete_with_schema(schema, "q"))

        expected_system = _SCHEMA_SYSTEM_TEMPLATE.format(schema_context=schema)
        actual_system = next(
            m["content"] for m in captured["messages"] if m["role"] == "system"
        )
        assert actual_system == expected_system


# ─── TestLogging ──────────────────────────────────────────────────────────────


class TestLogging:
    def test_log_file_created_after_call(self, client):
        asyncio.run(client.complete("sys", "usr"))
        assert client._log_path.exists()

    def test_log_contains_valid_json_lines(self, client):
        asyncio.run(client.complete("sys", "usr"))
        raw = client._log_path.read_text().strip()
        assert raw  # not empty
        record = json.loads(raw.split("\n")[0])
        assert isinstance(record, dict)

    def test_log_has_timestamp_field(self, client):
        asyncio.run(client.complete("s", "u"))
        record = json.loads(client._log_path.read_text().strip().split("\n")[-1])
        assert "timestamp" in record

    def test_log_has_provider_field(self, client):
        asyncio.run(client.complete("s", "u"))
        record = json.loads(client._log_path.read_text().strip().split("\n")[-1])
        assert record["provider"] == "ollama"

    def test_log_has_latency_ms_integer(self, client):
        asyncio.run(client.complete("s", "u"))
        record = json.loads(client._log_path.read_text().strip().split("\n")[-1])
        assert isinstance(record["latency_ms"], int)
        assert record["latency_ms"] >= 0

    def test_log_has_system_snippet(self, client):
        asyncio.run(client.complete("my system prompt", "u"))
        record = json.loads(client._log_path.read_text().strip().split("\n")[-1])
        assert "my system prompt" in record["system_snippet"]

    def test_log_has_user_snippet(self, client):
        asyncio.run(client.complete("s", "my user message"))
        record = json.loads(client._log_path.read_text().strip().split("\n")[-1])
        assert "my user message" in record["user_snippet"]

    def test_multiple_calls_append_separate_lines(self, client):
        asyncio.run(client.complete("s", "first"))
        asyncio.run(client.complete("s", "second"))
        lines = [l for l in client._log_path.read_text().strip().split("\n") if l]
        assert len(lines) == 2

    def test_no_log_file_when_logging_disabled(self, client):
        client._log_calls = False
        asyncio.run(client.complete("s", "u"))
        assert not client._log_path.exists()

    def test_groq_fallback_logged_as_groq(self, client):
        client._ollama_available = False
        asyncio.run(client.complete("s", "u"))
        record = json.loads(client._log_path.read_text().strip().split("\n")[-1])
        assert record["provider"] == "groq"

    def test_failed_call_logged_with_both_failed(self, client):
        client._ollama_available = True
        client._groq_api_key = ""

        mock_ollama = AsyncMock()
        mock_ollama.chat.side_effect = ConnectionError("refused")

        with patch.object(sys.modules["ollama"], "AsyncClient", return_value=mock_ollama):
            with pytest.raises(RuntimeError):
                asyncio.run(client.complete("s", "u"))

        record = json.loads(client._log_path.read_text().strip().split("\n")[-1])
        assert record["provider"] == "both_failed"
        assert record["error"] is not None


# ─── TestSyncWrappers ─────────────────────────────────────────────────────────


class TestSyncWrappers:
    def test_chat_returns_string(self, client):
        result = client.chat([{"role": "user", "content": "hello"}])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_chat_extracts_system_role(self, client):
        captured: dict = {}

        async def _fake_complete(system, user, temperature=None):
            captured["system"] = system
            return "ok"

        with patch.object(client, "complete", side_effect=_fake_complete):
            client.chat([
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "hi"},
            ])

        assert captured["system"] == "be helpful"

    def test_chat_joins_multiple_user_messages(self, client):
        captured: dict = {}

        async def _fake_complete(system, user, temperature=None):
            captured["user"] = user
            return "ok"

        with patch.object(client, "complete", side_effect=_fake_complete):
            client.chat([
                {"role": "user", "content": "first"},
                {"role": "user", "content": "second"},
            ])

        assert "first" in captured["user"]
        assert "second" in captured["user"]

    def test_chat_passes_temperature(self, client):
        captured: dict = {}

        async def _fake_complete(system, user, temperature=None):
            captured["temperature"] = temperature
            return "ok"

        with patch.object(client, "complete", side_effect=_fake_complete):
            client.chat([{"role": "user", "content": "q"}], temperature=0.9)

        assert captured["temperature"] == pytest.approx(0.9)

    def test_health_check_returns_bool(self, client):
        result = client.health_check()
        assert isinstance(result, bool)

    def test_health_check_true_when_ollama_reachable(self, client):
        result = client.health_check()
        assert result is True

    def test_health_check_false_when_both_unavailable(self, client):
        client._groq_api_key = ""
        mock_probe = AsyncMock()
        mock_probe.list.side_effect = Exception("down")
        with patch.object(sys.modules["ollama"], "AsyncClient", return_value=mock_probe):
            result = client.health_check()
        assert result is False

    def test_vision_chat_returns_string(self, client):
        messages = [{"role": "user", "content": "describe this image"}]
        image_bytes = b"\x89PNG\r\n\x1a\n"  # fake PNG header
        result = client.vision_chat(messages, image_bytes)
        assert result == "ollama_sync_reply"

    def test_vision_chat_appends_image_to_last_user_message(self, client):
        captured: dict = {}
        mock_sync = MagicMock()

        def _fake_sync_chat(model, messages, options=None):
            captured["messages"] = messages
            return {"message": {"content": "described"}}

        mock_sync.chat = _fake_sync_chat
        with patch.object(sys.modules["ollama"], "Client", return_value=mock_sync):
            client.vision_chat(
                [{"role": "user", "content": "what do you see?"}], b"imagedata"
            )

        last = captured["messages"][-1]
        assert "images" in last
        assert len(last["images"]) == 1


# ─── TestConfiguration ────────────────────────────────────────────────────────


class TestConfiguration:
    def test_default_provider_is_ollama(self, monkeypatch):
        monkeypatch.delenv("LLM_PROVIDER", raising=False)
        c = LLMClient()
        assert c.provider == "ollama"

    def test_env_var_sets_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "sk-test")
        c = LLMClient()
        assert c.provider == "groq"

    def test_explicit_provider_arg_overrides_env(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        c = LLMClient(provider="ollama")
        assert c.provider == "ollama"

    def test_timeout_loaded_from_config(self, client):
        assert client._timeout == 60

    def test_max_retries_loaded_from_config(self, client):
        assert client._max_retries == 3

    def test_temperature_loaded_from_config(self, client):
        assert client._temperature == pytest.approx(0.1)

    def test_ollama_model_loaded_from_config(self, client):
        assert "llama" in client._ollama_model

    def test_groq_model_set(self, client):
        assert client._groq_model  # non-empty string


# ─── TestRunSync ──────────────────────────────────────────────────────────────


class TestRunSync:
    def test_runs_coroutine_and_returns_value(self, client):
        async def add(a: int, b: int) -> int:
            return a + b

        assert client._run_sync(add(3, 7)) == 10

    def test_propagates_exception_from_coroutine(self, client):
        async def boom() -> None:
            raise ValueError("test error from coroutine")

        with pytest.raises(ValueError, match="test error from coroutine"):
            client._run_sync(boom())

    def test_handles_string_return(self, client):
        async def greet() -> str:
            return "hello"

        assert client._run_sync(greet()) == "hello"
