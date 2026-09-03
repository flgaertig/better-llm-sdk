"""Universal LLM API wrapper for OpenAI-compatible APIs.

Features:
- Structured outputs with automatic schema generation (OpenAI strict-compatible)
- Vision model support
- Tool definitions with automatic function introspection
- Streaming with reasoning-token handling (safe for tags split across chunks)

Requirements: Python >= 3.10 and openai >= 2.0. Pillow is optional and only
needed for PIL images or image formats that must be transcoded.

Message content formats (vision):

    {"role": "user", "content": [
        {"type": "text", "text": "..."},
        {"type": "image", "image_url": "https://..."},
        {"type": "image", "image_path": pathlib.Path("photo.png")},
        {"type": "image", "image_base64": "..."},
        {"type": "image", "image_pil": PIL.Image.Image},
        {"type": "image", "image_path": "photo.png", "detail": "high"},
    ]}

Message content formats (audio/video/file, input only):

    {"role": "user", "content": [
        {"type": "audio", "audio_path": "clip.wav"},
        {"type": "video", "video_url": "https://.../clip.mp4"},
        {"type": "video", "video_path": "clip.mp4", "processing": "agentic"},
        {"type": "file", "file_path": "doc.pdf"},
        {"type": "file", "file_id": "file-abc123", "detail": "high"},
    ]}

``audio`` also accepts ``audio_base64``/``audio_url`` (URLs pass through),
``video`` accepts ``video_base64`` (remote URLs pass through) plus optional
``processing`` (provider extension, passed through),
``file`` accepts ``file_base64``/``file_url``/``file_id`` plus optional
``filename``/``mime_type``/``detail`` (detail is Responses-only; chat warns
and drops it).

``image_path`` accepts ``pathlib.Path`` objects and plain strings.

Stream events are small dicts with ``type`` and ``content`` keys:
``answer``, ``reasoning``, ``refusal``, ``tool_call``, ``tool_call_part``,
``verbose``, ``final``, ``done``. Inference generators are lazy: the HTTP
request starts at the first ``next()``. Breaking out of a stream early aborts
the request and releases the connection.

A ``LLM`` instance keeps no per-request state (parsers and handlers are
created per call), so instances are safe to share between threads and tasks.
The async client is created lazily per event loop, so one instance can be
reused across repeated ``asyncio.run()`` calls. Threads that each run their
own event loop should use separate instances.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import copy
import functools
import inspect
import io
import json
import logging
import math
import mimetypes
import re
import threading
import time
import unicodedata
import uuid
import weakref
from collections.abc import AsyncGenerator, Callable, Generator, Mapping
from dataclasses import MISSING, dataclass, field, is_dataclass
from dataclasses import fields as dataclass_fields
from datetime import date as dt_date
from datetime import datetime as dt_datetime
from datetime import time as dt_time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Final,
    Literal,
    Optional,
    TypedDict,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)
from urllib.parse import parse_qsl, urlencode, urlparse
from uuid import UUID

from openai import APIError, AsyncOpenAI, OpenAI

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

# ============================================================================
# Logging
# ============================================================================

logger = logging.getLogger(__name__)


def configure_quiet_logging() -> None:
    """Silence the chatty httpx/openai/httpcore loggers (opt-in).

    Importing this module does not touch global logging configuration.
    """
    for logger_name in ("httpx", "openai", "httpcore"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def configure_debug_logging() -> None:
    """Enable debug logs for this SDK (one-liner for real programmers).

    Sets only the SDK logger to DEBUG; third-party HTTP loggers are
    untouched here (``LLM(..., debug=True)`` additionally pins
    httpx/openai/httpcore to WARNING to avoid leaking credentials).
    """
    logger.setLevel(logging.DEBUG)


# ============================================================================
# Constants
# ============================================================================

DEFAULT_API_KEY: Final[str] = "lm-studio"
DEFAULT_BASE_URL: Final[str] = "http://localhost:1234/v1"
DEFAULT_TIMEOUT: Final[float] = 300.0
__version__: Final[str] = "2.0.0"

MessageList = list[dict[str, Any]]

# ============================================================================
# Async resource helpers
# ============================================================================

async def _aclose_async_resource(resource: Any) -> None:
    """Close an async resource from within a running event loop."""
    close = getattr(resource, "aclose", None) or getattr(resource, "close", None)
    if close is None:
        return
    close_result = close()
    if inspect.isawaitable(close_result):
        await close_result


# Keeps references to fire-and-forget close tasks so they are not garbage
# collected before finishing.
_pending_async_closes: set[asyncio.Task] = set()
_pending_async_closes_lock = threading.Lock()


def _log_async_close_done(done_task: asyncio.Task) -> None:
    """Drop a finished close task; log failures instead of losing them."""
    with _pending_async_closes_lock:
        _pending_async_closes.discard(done_task)
    if done_task.cancelled():
        return
    error = done_task.exception()
    if error is not None:
        logger.warning("Failed to close async resource: %s", _redact_url_credentials(str(error)))


def _schedule_async_close(resource: Any) -> None:
    """Best-effort close of an async resource from inside a running loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    task = loop.create_task(_aclose_async_resource(resource))
    with _pending_async_closes_lock:
        _pending_async_closes.add(task)
    task.add_done_callback(_log_async_close_done)


# ============================================================================
# Small helpers
# ============================================================================

def _merge_dict_layers(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = _merge_dict_layers(current, dict(value))
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _deep_merge_dicts(
    base: Optional[dict[str, Any]],
    override: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Deep-merge two extra_body dicts; returns None only when both are empty."""
    if not base and not override:
        return None
    return _merge_dict_layers(base or {}, override or {})


def _dump_json(value: Any, *, what: str) -> str:
    """json.dumps with default=str; unserializable payloads raise ConfigurationError."""
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError, RecursionError) as e:
        raise ConfigurationError(f"{what} is not JSON-serializable: {e}") from e


def _copy_messages_shallow(messages: list[dict]) -> list[dict]:
    """Copy message list/dicts/content lists without touching payloads.

    process_messages only *replaces* content items in the copied lists, so
    caller data (including PIL objects, which deepcopy would duplicate or
    choke on) is never mutated and never duplicated.
    """
    copied = []
    for msg in messages:
        if isinstance(msg, dict):
            new_msg = dict(msg)
            content = new_msg.get("content")
            if isinstance(content, list):
                new_msg["content"] = list(content)
            copied.append(new_msg)
        else:
            copied.append(msg)
    return copied


def _drop_response_items_for_chat(messages: list[dict]) -> None:
    """Strip Responses-only replay state from chat messages (copy-on-write).

    ``assistant_message()`` attaches ``response_items`` for Responses tool
    loops; providers behind chat completions reject the unknown key, so it
    is removed here with a one-time warning instead of failing downstream.
    """
    dropped = False
    for index, msg in enumerate(messages):
        if isinstance(msg, dict) and "response_items" in msg:
            new_msg = dict(msg)
            del new_msg["response_items"]
            messages[index] = new_msg
            dropped = True
    if dropped:
        _warn_response_items_dropped_for_chat()


def _drop_original_detail_for_chat(messages: list[dict]) -> None:
    """Strip image detail 'original' from chat parts (copy-on-write).

    ``low``/``high``/``auto`` are valid OpenAI chat detail levels; only
    ``original`` is Responses-only, so just that value is dropped (warn once).
    """
    dropped = False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for index, part in enumerate(content):
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            url_data = part.get("image_url")
            if isinstance(url_data, dict) and url_data.get("detail") == "original":
                new_url_data = dict(url_data)
                del new_url_data["detail"]
                content[index] = {**part, "image_url": new_url_data}
                dropped = True
    if dropped:
        _warn_original_dropped_for_chat()


# Keys the SDK sets itself: extra_body is flattened to the top level of the
# request by the OpenAI client, so these would silently override the call.
_RESERVED_EXTRA_BODY_KEYS: Final[frozenset] = frozenset({"model", "messages", "input", "stream"})


def _reject_reserved_extra_body_keys(extra_body: Optional[dict[str, Any]]) -> None:
    """Fail fast when extra_body would override SDK-managed request fields."""
    if not extra_body:
        return
    clashes = _RESERVED_EXTRA_BODY_KEYS.intersection(extra_body)
    if clashes:
        raise ConfigurationError(
            f"extra_body must not contain {sorted(clashes)}; "
            "use the explicit parameters instead"
        )
    # Strict wire check (no default=str): caller-controlled payloads must
    # be real JSON; tool/result paths keep the lenient _dump_json.
    try:
        json.dumps(extra_body, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as e:
        raise ConfigurationError(
            f"extra_body is not JSON-serializable: {e}"
        ) from e


def _contains_confusable_controls(value: str) -> bool:
    """True for control/format/invisible characters (Cc/Cf/Zl/Zp/C0/DEL).

    Catches what strip()/isspace() miss: zero-width (U+200B), bidi marks
    (U+202E), BOM (U+FEFF) and C1 controls — all invisible in logs and a
    classic log-injection / homograph vector.
    """
    return any(
        unicodedata.category(char) in ("Cc", "Cf", "Zl", "Zp")
        or ord(char) < 32
        or ord(char) == 127
        for char in value
    )


def _reject_confusable_controls(value: Any, *, what: str) -> None:
    """Fail fast on invisible/control characters in user-supplied strings."""
    if isinstance(value, str) and _contains_confusable_controls(value):
        raise ConfigurationError(f"{what} must not contain control characters")


def _validate_identity_options(
    model: Any,
    api_key: Any,
    default_headers: Any,
    use_responses_api: Any,
    normalize_base_url: Any,
    debug: Any,
) -> str:
    """Shared fail-fast checks for identity/flag options; returns stripped model."""
    if not isinstance(model, str) or not model.strip():
        raise ConfigurationError("model must be a non-empty string")
    _reject_confusable_controls(model, what="model")
    model = model.strip()
    if api_key is not None and (
        not isinstance(api_key, str) or not api_key.strip()
    ):
        raise ConfigurationError("api_key must be a non-empty string or None")
    if default_headers is not None and (
        not isinstance(default_headers, dict)
        or not all(
            isinstance(key, str)
            and key.strip()
            and key.strip().lower() not in _AUTH_HEADER_BLOCKLIST
            and isinstance(value, str)
            and not _contains_confusable_controls(key)
            and not _contains_confusable_controls(value)
            for key, value in default_headers.items()
        )
    ):
        raise ConfigurationError(
            "default_headers must be a dict of non-empty str to str or None "
            "(authorization headers are rejected; use api_key)"
        )
    for name, flag in (
        ("use_responses_api", use_responses_api),
        ("normalize_base_url", normalize_base_url),
        ("debug", debug),
    ):
        if not isinstance(flag, bool):
            raise ConfigurationError(f"{name} must be a bool")
    return model


def _validate_connection_options(
    timeout: Any,
    max_retries: Any,
    extra_body: Optional[dict[str, Any]],
) -> None:
    """Shared fail-fast checks for __init__ and direct LLMConfig use."""
    if extra_body is not None and not isinstance(extra_body, dict):
        raise ConfigurationError("extra_body must be a dict of provider-specific fields")
    _reject_reserved_extra_body_keys(extra_body)
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ConfigurationError(
            "timeout must be a positive finite number or None (no timeout)"
        )
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ConfigurationError("max_retries must be an int >= 0")


def _validate_generation_options(
    temperature: Any,
    top_p: Any,
    max_tokens: Any,
    seed: Any,
    stop: Any,
    user: Any,
    store: Any,
    reasoning_effort: Any = None,
    reasoning_budget: Any = None,
) -> None:
    """Shared fail-fast checks for per-call generation options (both APIs)."""
    for name, value in (("temperature", temperature), ("top_p", top_p)):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ConfigurationError(f"{name} must be a finite number or None")
    if max_tokens is not None and (
        isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0
    ):
        raise ConfigurationError("max_tokens must be a positive int or None")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ConfigurationError("seed must be an int or None")
    if stop is not None and (
        not isinstance(stop, (str, list, tuple))
        or (isinstance(stop, str) and not stop.strip())
        or (isinstance(stop, (list, tuple)) and (len(stop) == 0 or not all(isinstance(s, str) and s.strip() for s in stop)))
    ):
        raise ConfigurationError(
            "stop must be a non-empty string or a non-empty list of non-empty strings"
        )
    if user is not None and (not isinstance(user, str) or not user.strip()):
        raise ConfigurationError("user must be a non-empty string or None")
    _reject_confusable_controls(user, what="user")
    if store is not None and not isinstance(store, bool):
        raise ConfigurationError("store must be a bool or None")
    if reasoning_budget is not None and (
        isinstance(reasoning_budget, bool)
        or not isinstance(reasoning_budget, int)
        or reasoning_budget <= 0
    ):
        raise ConfigurationError("reasoning_budget must be a positive int or None")
    if reasoning_effort is not None and (
        not isinstance(reasoning_effort, str) or not reasoning_effort.strip()
    ):
        raise ConfigurationError(
            "reasoning_effort must be a non-empty string or None"
        )
    if reasoning_effort is not None and reasoning_budget is not None:
        raise ConfigurationError(
            "reasoning_effort and reasoning_budget are mutually exclusive; set only one"
        )


def _resolve_messages(
    messages: Optional[list] = None,
    input: Optional[str] = None,
    system: Optional[str] = None,
) -> MessageList:
    if system is not None and not isinstance(system, str):
        raise ConfigurationError("system must be a string")
    if input is not None and not isinstance(input, str):
        raise ConfigurationError("input must be a string")
    if messages is not None and input is not None:
        raise ConfigurationError("Cannot specify both 'messages' and 'input'")

    if messages is not None:
        if not isinstance(messages, list):
            raise ConfigurationError("messages must be a list of dicts")
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ConfigurationError(f"messages[{index}] must be a dict, got {type(message).__name__}")
        resolved = list(messages)
        if system is not None:
            if resolved and isinstance(resolved[0], dict) and resolved[0].get("role") == "system":
                # Replace an existing system prompt instead of stacking two.
                logger.debug("Replacing existing system message with new system prompt")
                resolved[0] = {"role": "system", "content": system}
            else:
                resolved.insert(0, {"role": "system", "content": system})
        return resolved

    if input is not None:
        resolved = []
        if system is not None:
            resolved.append({"role": "system", "content": system})
        resolved.append({"role": "user", "content": input})
        return resolved

    raise ConfigurationError("Must specify either 'messages' or 'input'")


def _extract_reasoning(delta: Any) -> str:
    """Return streamed reasoning content from supported delta fields (strings only)."""
    for attr in ("reasoning_content", "reasoning"):
        value = getattr(delta, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _normalize_stop_reason(reason: Optional[str]) -> Optional[str]:
    if reason == "function_call":
        return "tool_calls"
    return reason


def _resolve_token_metrics(
    completion_tokens: Optional[int],
    prompt_tokens: Optional[int],
    total_tokens: Optional[int],
    chunks: int,
) -> tuple[int, Optional[int]]:
    """Resolve token counts; falls back to the chunk count only for `tokens`."""
    tokens = completion_tokens if completion_tokens is not None else chunks
    if total_tokens is None and prompt_tokens is not None:
        total_tokens = prompt_tokens + tokens
    return tokens, total_tokens


def _decode_tokens_per_second(
    t_first: Optional[float],
    t_last: Optional[float],
    elapsed: float,
    tokens: int,
) -> float:
    """Decode throughput over the chunk window (excludes TTFT and consumer time)."""
    decode_seconds = (
        t_last - t_first
        if t_first is not None and t_last is not None and t_last > t_first
        else None
    )
    if decode_seconds:
        return tokens / decode_seconds
    return tokens / elapsed if elapsed > 0 else 0


_JSON_FENCE_RE = re.compile(r"^\s*```(?:[a-zA-Z0-9_-]*)\s*\n?|\n?\s*```\s*$")


def _strip_json_fences(text: str) -> str:
    return _JSON_FENCE_RE.sub("", text)


def _parse_structured_output(
    answer: str,
    *,
    stop_reason: Optional[str],
    strict: bool,
    refusal: str = "",
    has_tool_calls: bool = False,
) -> Any:
    if refusal:
        if strict:
            raise StructuredOutputError(
                f"Model refused to produce structured output: {_redact_url_credentials(refusal)}",
                raw=answer,
                stop_reason="refusal",
            )
        logger.warning(
            "Model refused to produce structured output: %s",
            _redact_url_credentials(refusal[:500]),
        )
        return answer
    if not answer.strip() and (
        # Tools-only turn ("structured answer OR tool call" router pattern) or
        # a stream that never produced content for a non-JSON reason: the
        # answer is legitimately absent, not invalid JSON.
        has_tool_calls or stop_reason in ("tool_calls", "content_filter", "cancelled", "failed")
    ):
        return None
    cleaned = _strip_json_fences(answer)
    try:
        return json.loads(cleaned)
    except (ValueError, TypeError, RecursionError) as e:
        if strict:
            hint = " (output was truncated; increase the token limit)" if stop_reason == "length" else ""
            raise StructuredOutputError(
                f"Model did not return valid JSON for the requested output format: {e}{hint}",
                raw=answer,
                stop_reason=stop_reason,
            ) from e
        logger.warning("Structured output was not valid JSON, returning raw string: %s", e)
        return answer


def _resolve_api_base(base_url: str, *, normalize: bool = True) -> str:
    """Normalize an API base URL.

    ``/v1`` is appended only when the URL has no path component at all, so
    providers with their own paths (Gemini ``/v1beta/openai``, Cloudflare AI
    Gateway, Azure, proxies) keep working. Pass ``normalize=False`` to use
    the URL exactly as given.
    """
    base = base_url.rstrip("/")
    if not normalize:
        return base
    parsed = urlparse(base)
    if parsed.path in ("", "/"):
        # Insert before query/fragment: "host?api-version=x" becomes
        # "host/v1?api-version=x", not "host?api-version=x/v1".
        return parsed._replace(path="/v1").geturl()
    return base


def _norm_api_base(url: str) -> str:
    return _resolve_api_base(url)


def _validate_base_url(url: Any, *, what: str = "base_url") -> None:
    """Shared fail-fast check: http(s) URL with host, no userinfo.

    Query strings are allowed (e.g. Azure-style ``?api-version=``); only
    credentials in the URL are rejected. Secrets must go in ``api_key``.
    """
    if not isinstance(url, str) or not url.strip():
        raise ConfigurationError(f"{what} must be a non-empty URL string")
    _reject_confusable_controls(url, what=what)
    if len(url) > 8192:
        raise ConfigurationError(f"{what} exceeds the 8KB URL length limit")
    if any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in url):
        raise ConfigurationError(
            f"{what} must not contain whitespace or control characters"
        )
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ConfigurationError(
            f"{what} must be an http(s) URL with a host, got "
            f"{_redact_url_credentials(url)!r}"
        )
    try:
        port = parsed.port
    except ValueError:
        raise ConfigurationError(f"{what} has an invalid port") from None
    if port is not None and not 1 <= port <= 65535:
        raise ConfigurationError(f"{what} has an invalid port (1-65535)")
    if parsed.fragment:
        raise ConfigurationError(f"{what} must not contain a fragment")
    if parsed.username or parsed.password:
        raise ConfigurationError(f"{what} must not contain credentials (userinfo)")


def _validate_http_url(url: Any, *, what: str) -> str:
    """Fail-fast for remote media/file inputs: http(s) only, no userinfo.

    The SDK never fetches these (URL in, URL out), but the provider might —
    so non-web schemes (file://, ftp://, javascript:, data:), embedded
    credentials and absurd lengths are rejected client-side.
    """
    if not isinstance(url, str) or not url.strip():
        raise ConfigurationError(f"{what} must be a non-empty URL string")
    _reject_confusable_controls(url, what=what)
    if len(url) > 8192:
        raise ConfigurationError(f"{what} exceeds the 8KB URL length limit")
    if any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in url):
        raise ConfigurationError(
            f"{what} must not contain whitespace or control characters"
        )
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ConfigurationError(
            f"{what} must be an http(s) URL, got {_redact_url_credentials(url)!r}"
        )
    try:
        port = parsed.port
    except ValueError:
        raise ConfigurationError(f"{what} has an invalid port") from None
    if port is not None and not 1 <= port <= 65535:
        raise ConfigurationError(f"{what} has an invalid port (1-65535)")
    if parsed.username or parsed.password:
        raise ConfigurationError(f"{what} must not contain credentials (userinfo)")
    return url


_KNOWN_DETAIL_LEVELS: frozenset[str] = frozenset({"low", "high", "auto", "original"})


@functools.lru_cache(maxsize=128)
def _warn_unknown_detail(value: str, what: str) -> None:
    logger.warning(
        "%s %r is not a commonly supported detail level; passing it through",
        what,
        value,
    )


@functools.lru_cache(maxsize=1)
def _warn_detail_dropped_for_chat() -> None:
    logger.warning(
        "file detail is not supported by chat completions; ignoring it "
        "(use the Responses API for PDF detail levels)"
    )


@functools.lru_cache(maxsize=1)
def _warn_response_items_dropped_for_chat() -> None:
    logger.warning(
        "response_items are Responses-API replay state; ignoring them in chat"
    )


@functools.lru_cache(maxsize=1)
def _warn_original_dropped_for_chat() -> None:
    logger.warning(
        "image detail 'original' is not supported by chat completions; "
        "ignoring it (use the Responses API)"
    )


def _validate_detail(detail: Any, *, what: str) -> Optional[str]:
    """Validate a detail knob (image/file inputs); returns the stripped value.

    Known levels pass through; unknown short strings warn once (new providers
    keep adding values) instead of raising. Non-strings, empties and absurdly
    long values fail fast.
    """
    if detail is None:
        return None
    if not isinstance(detail, str) or not detail.strip():
        raise ConfigurationError(f"{what} must be a detail string or None")
    detail = detail.strip()
    _reject_confusable_controls(detail, what=what)
    if any(ord(c) < 32 or ord(c) == 127 for c in detail):
        raise ConfigurationError(f"{what} must not contain control characters")
    if len(detail) > 64:
        raise ConfigurationError(f"{what} exceeds the 64-character limit")
    if detail not in _KNOWN_DETAIL_LEVELS:
        _warn_unknown_detail(detail, what)
    return detail


def _validate_max_image_side(value: Any) -> Optional[int]:
    """Fail fast on invalid max_image_side knobs (all image entry points)."""
    if value is not None and (
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
    ):
        raise ConfigurationError("max_image_side must be a positive int or None")
    return value


_B64_WHITESPACE_RE: re.Pattern = re.compile(r"\s")

_EMBEDDED_URL_CREDENTIALS_RE: re.Pattern = re.compile(r"(https?://)[^/\s?#]*@")

# Any scheme inside longer texts, so embedded URLs get redacted too.
_URL_IN_TEXT_RE: Final[re.Pattern] = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s\"'<>]+")

# Query keys treated as secrets and stripped by _redact_url_credentials.
_SECRET_QUERY_KEYS: Final[frozenset] = frozenset({
    "api_key", "apikey", "key", "token", "access_token", "auth_token",
    "secret", "client_secret", "authorization", "auth", "password", "passwd",
    "access_key", "api_secret", "secret_key", "session_token", "id_token",
    "refresh_token", "app_key", "app_secret", "client_token", "api_token",
})

# Authorization headers must never ride in default_headers (use api_key).
_AUTH_HEADER_BLOCKLIST: Final[frozenset] = frozenset({
    "authorization", "x-api-key", "api-key", "x-openai-api-key",
    "openai-api-key",
})


def _is_secret_query_key(key: str) -> bool:
    """True for secret-ish query keys (exact match or _key/_token/_secret suffix)."""
    normalized = key.strip().lower().replace("-", "_")
    if normalized in _SECRET_QUERY_KEYS:
        return True
    return normalized.endswith(("_key", "_token", "_secret", "_password"))


def _ftyp_brand(raw: bytes) -> Optional[bytes]:
    """Brand of a plausible ftyp box near the header (mp4-family media).

    Walks leading padding atoms (wide/free/skip); a coincidental ``ftyp``
    substring anywhere else is not a container signal.
    """
    offset = 0
    limit = min(len(raw), 64)
    while offset + 8 <= limit:
        size = int.from_bytes(raw[offset:offset + 4], "big")
        boxtype = raw[offset + 4:offset + 8]
        if boxtype == b"ftyp":
            return raw[offset + 8:offset + 12]
        if boxtype not in (b"wide", b"free", b"skip") or size < 8:
            return None
        offset += size
    return None


def _strip_b64_whitespace(body: str) -> str:
    """Strip base64 whitespace without copying clean inputs.

    Large base64 payloads are usually already clean; the regex copy would
    transiently double memory, so only copy when whitespace is present.
    """
    if _B64_WHITESPACE_RE.search(body) is None:
        return body
    return re.sub(r"\s+", "", body)


def _redact_pure_url(url: str) -> str:
    """Strip userinfo, secret query keys and fragments from one URL."""
    try:
        parsed = urlparse(url)
    except (TypeError, ValueError):
        return url
    if not parsed.scheme or not parsed.netloc:
        return url
    if "@" not in parsed.netloc and not parsed.query and not parsed.fragment:
        return url
    query = parsed.query
    if query:
        kept = [
            (key, value)
            for key, value in parse_qsl(query, keep_blank_values=True)
            if not _is_secret_query_key(key)
        ]
        query = urlencode(kept)
    return parsed._replace(
        netloc=parsed.netloc.rsplit("@", 1)[-1], query=query, fragment=""
    ).geturl()


def _redact_url_credentials(url: Any) -> Any:
    """Strip userinfo, secret query keys and fragments for safe logging.

    Works on pure URLs and on URLs embedded in longer texts (e.g. server
    error echoes); pure sentences without URL structure pass through
    untouched (no query-mangling of non-URLs). Harmless query keys such as
    ``api-version`` are kept so logs stay debuggable.
    """
    if not isinstance(url, str):
        return url
    # Embedded credentials first: "…https://user:pass@host/v1…" in prose.
    url = _EMBEDDED_URL_CREDENTIALS_RE.sub(r"\1", url)
    if "://" not in url:
        return url
    return _URL_IN_TEXT_RE.sub(lambda m: _redact_pure_url(m.group(0)), url)




def _redact_json_urls(value: Any, _depth: int = 0) -> Any:
    """Recursively redact URLs inside provider-echoed bodies.

    Handles dicts (keys and values), lists, tuples (type-preserving) and
    sets; bytes pass through (they never hold str URLs); anything else is
    returned as-is.
    """
    if isinstance(value, str):
        return _redact_url_credentials(value)
    if _depth > 10:
        return "[redacted]"
    if isinstance(value, dict):
        return {
            _redact_url_credentials(key) if isinstance(key, str) else key:
            _redact_json_urls(item, _depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_redact_json_urls(item, _depth + 1) for item in value)
    if isinstance(value, list):
        return [_redact_json_urls(item, _depth + 1) for item in value]
    if isinstance(value, (set, frozenset)):
        return type(value)(_redact_json_urls(item, _depth + 1) for item in value)
    return value


# ============================================================================
# Enums
# ============================================================================

class EventType(str, Enum):
    """Event types emitted during streaming."""
    ANSWER = "answer"
    REASONING = "reasoning"
    REFUSAL = "refusal"
    TOOL_CALL = "tool_call"
    TOOL_CALL_PART = "tool_call_part"
    VERBOSE = "verbose"
    FINAL = "final"
    DONE = "done"

    def __str__(self) -> str:
        return self.value


class SchemaType(str, Enum):
    """JSON Schema type mappings."""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"

# ============================================================================
# Type Definitions
# ============================================================================

class StreamEvent(TypedDict):
    """A single stream event; ``content`` is ``None`` only for ``done``."""
    type: str
    content: Any

class ToolCall(TypedDict):
    """Typed dictionary for tool calls."""
    id: str
    name: str
    arguments: dict[str, Any]
    callable: Optional[Callable]

class ToolResultMessage(TypedDict, total=False):
    role: str
    tool_call_id: str
    content: str
    name: str

class AssistantMessage(TypedDict, total=False):
    role: str
    content: Any
    tool_calls: list[dict[str, Any]]
    reasoning_content: str
    response_items: list[dict[str, Any]]

class UserMessage(TypedDict):
    role: str
    content: str | list[dict[str, Any]]

class VerboseInfo(TypedDict, total=False):
    """Typed dictionary for verbose information."""
    tokens: int
    chunks: int
    tokens_per_second: float
    latency: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    stop_reason: Optional[str]


class FinalResponse(TypedDict, total=False):
    """Typed dictionary for final response."""
    answer: Any
    reasoning: str
    refusal: str
    reasoning_unterminated: bool
    tool_calls: list[ToolCall]
    response_items: list[dict[str, Any]]
    verbose: VerboseInfo
    stop_reason: Optional[str]

# ============================================================================
# Exceptions
# ============================================================================

class LLMError(Exception):
    """Base exception for LLM errors."""
    pass


class ConfigurationError(LLMError, ValueError):
    """Raised when configuration is invalid."""
    pass


class SchemaConversionError(LLMError):
    """Raised when schema conversion fails."""
    pass


class ModelRequestError(LLMError):
    """Raised when a model request fails.

    Carries API error details (status code, body, request id) when the
    underlying cause was an ``openai.APIError``.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        body: Any = None,
        request_id: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.request_id = request_id


class StructuredOutputError(LLMError):
    """Raised when structured output cannot be parsed (always strict)."""

    def __init__(self, message: str, *, raw: Any = None, stop_reason: Optional[str] = None) -> None:
        super().__init__(message)
        self.raw = raw
        self.stop_reason = stop_reason


class ImageProcessingError(LLMError, ValueError):
    """Raised when image processing fails."""
    pass


class AudioProcessingError(LLMError, ValueError):
    """Raised when audio input processing fails."""
    pass


class VideoProcessingError(LLMError, ValueError):
    """Raised when video input processing fails."""
    pass


class FileProcessingError(LLMError, ValueError):
    """Raised when generic file input processing fails."""
    pass

# ============================================================================
# Data Classes
# ============================================================================

@dataclass(frozen=True)
class CustomReasoningPattern:
    """Configuration for custom reasoning token patterns.

    Attributes:
        from_beginning: Whether content starts inside reasoning mode.
        start_token: Custom start token (regex escaped internally).
        end_token: Custom end token (regex escaped internally).
    """
    from_beginning: bool = False
    start_token: Optional[str] = None
    end_token: Optional[str] = None

    def __post_init__(self):
        for token_name in ("start_token", "end_token"):
            token = getattr(self, token_name)
            if token is not None and not isinstance(token, str):
                raise ConfigurationError(f"{token_name} must be a string or None")
            if isinstance(token, str) and len(token) > 64:
                raise ConfigurationError(f"{token_name} must be at most 64 characters")
        if self.start_token and not self.end_token:
            raise ConfigurationError("end_token required when start_token is specified")
        if self.end_token and not self.start_token:
            raise ConfigurationError("start_token required when end_token is specified")


@dataclass
class LLMConfig:
    """Configuration for LLM instance (see ``LLM`` for field meanings)."""
    model: str
    api_key: str = field(default=DEFAULT_API_KEY, repr=False)
    base_url: str = field(default=DEFAULT_BASE_URL, repr=False)
    reasoning_pattern: Optional[CustomReasoningPattern] = None
    default_stop_sequences: Optional[list[str]] = None
    timeout: Optional[float] = DEFAULT_TIMEOUT
    extra_body: Optional[dict[str, Any]] = field(default=None, repr=False)
    use_responses_api: bool = False
    default_headers: Optional[dict[str, str]] = field(default=None, repr=False)
    max_retries: int = 3
    normalize_base_url: bool = True
    debug: bool = False
    max_image_side: Optional[int] = 8192

    def __post_init__(self):
        _validate_connection_options(self.timeout, self.max_retries, self.extra_body)
        self.model = _validate_identity_options(
            self.model, self.api_key, self.default_headers,
            self.use_responses_api, self.normalize_base_url, self.debug,
        )
        if isinstance(self.api_key, str):
            self.api_key = self.api_key.strip()
        self.extra_body = copy.deepcopy(self.extra_body)
        self.default_headers = copy.deepcopy(self.default_headers)
        if self.default_stop_sequences is not None:
            self.default_stop_sequences = list(self.default_stop_sequences)
        if isinstance(self.base_url, str):
            self.base_url = self.base_url.rstrip("/")
        _validate_base_url(self.base_url)
        if self.default_stop_sequences is not None and (
            not isinstance(self.default_stop_sequences, (list, tuple))
            or len(self.default_stop_sequences) == 0
            or not all(isinstance(s, str) and s.strip() for s in self.default_stop_sequences)
        ):
            raise ConfigurationError(
                "default_stop_sequences must be a non-empty list of non-empty strings"
            )
        side = self.max_image_side
        if side is not None and (
            isinstance(side, bool) or not isinstance(side, int) or side <= 0
        ):
            raise ConfigurationError("max_image_side must be a positive int or None")

# ============================================================================
# Reasoning Parser
# ============================================================================

class ReasoningParser:
    """Separates reasoning tags from answer content in streamed text.

    Tags are matched case-insensitively and may arrive split across any
    number of stream chunks: a carry-over buffer holds back the longest
    suffix that could be the start of a tag (in the current mode) until the
    next chunk — or ``flush()`` at stream end — resolves it.
    """

    _BASE_START_TAGS: ClassVar[tuple[str, ...]] = (
        "<think>", "<thinking>", "[THINK]", "<thought>",
    )
    _BASE_END_TAGS: ClassVar[tuple[str, ...]] = (
        "</think>", "</thinking>", "[/THINK]", "</thought>",
    )

    def __init__(self, custom_token: Optional[CustomReasoningPattern] = None):
        self._custom_token = custom_token
        start_tags = list(self._BASE_START_TAGS)
        end_tags = list(self._BASE_END_TAGS)
        if custom_token:
            if custom_token.start_token:
                start_tags.append(custom_token.start_token)
            if custom_token.end_token:
                end_tags.append(custom_token.end_token)
        self._start_tags = [tag.lower() for tag in start_tags]
        self._end_tags = [tag.lower() for tag in end_tags]
        self._start_pattern = self._build_pattern(start_tags)
        self._end_pattern = self._build_pattern(end_tags)
        self._inside_reasoning = bool(custom_token and custom_token.from_beginning)
        self._carry = ""
        self._max_hold = max(
            (len(tag) for tag in self._start_tags + self._end_tags), default=1
        ) - 1

    @staticmethod
    def _build_pattern(tags: tuple[str, ...]) -> re.Pattern:
        return re.compile("|".join(re.escape(tag) for tag in tags), flags=re.IGNORECASE)

    def parse(self, content: str) -> tuple[str, str]:
        """Parse one chunk; returns (reasoning_part, answer_part).

        Text held back because it might be a split tag is prepended to the
        next chunk, so ''.join(reasoning) and ''.join(answer) over all chunks
        are identical to parsing the whole text at once.
        """
        if not isinstance(content, str):
            content = str(content)
        data = self._carry + content
        self._carry = ""
        reasoning_part = ""
        answer_part = ""

        while data:
            pattern = self._end_pattern if self._inside_reasoning else self._start_pattern
            match = pattern.search(data)
            if match:
                segment, data = data[: match.start()], data[match.end():]
            else:
                hold = self._partial_tag_suffix(data)
                split_at = len(data) - len(hold)
                segment, self._carry, data = data[:split_at], data[split_at:], ""
            if self._inside_reasoning:
                reasoning_part += segment
            else:
                answer_part += segment
            if match:
                self._inside_reasoning = not self._inside_reasoning

        return reasoning_part, answer_part

    def _partial_tag_suffix(self, text: str) -> str:
        tags = self._end_tags if self._inside_reasoning else self._start_tags
        limit = min(len(text), self._max_hold)
        if not limit:
            return ""
        window = text[-limit:].lower()
        for index in range(len(window)):
            candidate = window[index:]
            if any(tag.startswith(candidate) for tag in tags):
                return text[len(text) - len(candidate):]
        return ""

    def flush(self) -> tuple[str, str]:
        """Emit any held-back text in the current mode (call at stream end)."""
        rest, self._carry = self._carry, ""
        return (rest, "") if self._inside_reasoning else ("", rest)

    @property
    def is_inside_reasoning(self) -> bool:
        """Whether the stream ended inside an unterminated reasoning block."""
        return self._inside_reasoning

# ============================================================================
# Schema Converter
# ============================================================================

@dataclass
class _SchemaField:
    annotation: Any
    optional: bool
    description: Optional[str] = None


class SchemaConverter:
    """Converts Python types and classes to JSON Schema (OpenAI-compatible)."""

    _PRIMITIVE_TYPE_MAP: ClassVar[dict[type, SchemaType]] = {
        bool: SchemaType.BOOLEAN,
        int: SchemaType.INTEGER,
        float: SchemaType.NUMBER,
        str: SchemaType.STRING,
    }

    _KNOWN_TYPE_SCHEMAS: ClassVar[dict[type, dict[str, Any]]] = {
        dt_datetime: {"type": "string", "format": "date-time"},
        dt_date: {"type": "string", "format": "date"},
        dt_time: {"type": "string", "format": "time"},
        UUID: {"type": "string", "format": "uuid"},
        Decimal: {"type": "number"},
        Path: {"type": "string"},
        bytes: {"type": "string"},
    }

    _JSON_VALUE_TYPES: ClassVar[dict[type, str]] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }

    @staticmethod
    def _is_union_type(python_type: Any) -> bool:
        return get_origin(python_type) in (Union, UnionType)

    @classmethod
    def is_optional_type(cls, python_type: Any) -> bool:
        """Return whether an annotation is a union that includes None."""
        return cls._is_union_type(python_type) and type(None) in get_args(python_type)

    @staticmethod
    def _ordered_object_schema(
        required: Optional[list[str]] = None,
        properties: Optional[dict[str, Any]] = None,
        additional_properties: Any = False,
    ) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": SchemaType.OBJECT.value}
        if required:
            schema["required"] = required
        if properties is not None:
            schema["properties"] = properties
        if additional_properties is not None:
            schema["additionalProperties"] = additional_properties
        return schema

    def python_type_to_json_schema(
        self,
        python_type: Any,
        seen_models: Optional[set] = None,
        *,
        strict: bool = False,
    ) -> dict[str, Any]:
        """Convert a Python type annotation to JSON Schema."""
        seen_models = seen_models if seen_models is not None else set()
        return self._type_to_schema(python_type, seen_models, strict)

    def _type_to_schema(self, python_type: Any, seen_models: set, strict: bool) -> dict[str, Any]:
        if python_type is type(None):
            return {"type": SchemaType.NULL.value}
        if python_type is Any:
            return {}

        # Annotated[X, "description", ...]
        metadata = getattr(python_type, "__metadata__", None)
        if metadata is not None:
            args = get_args(python_type)
            if args:
                schema = dict(self._type_to_schema(args[0], seen_models, strict))
                for extra in args[1:]:
                    if isinstance(extra, str) and "description" not in schema:
                        schema["description"] = extra
                return schema

        origin = get_origin(python_type)
        args = get_args(python_type)

        if origin is list:
            schema: dict[str, Any] = {"type": SchemaType.ARRAY.value}
            schema["items"] = self._type_to_schema(args[0], seen_models, strict) if args else {}
            return schema

        if origin in (set, frozenset):
            schema = {
                "type": SchemaType.ARRAY.value,
                "items": self._type_to_schema(args[0], seen_models, strict) if args else {},
                "uniqueItems": True,
            }
            return schema

        if origin is tuple:
            if not args:
                return {"type": SchemaType.ARRAY.value, "items": {}}
            if len(args) == 2 and args[1] is Ellipsis:
                return {
                    "type": SchemaType.ARRAY.value,
                    "items": self._type_to_schema(args[0], seen_models, strict),
                }
            return {
                "type": SchemaType.ARRAY.value,
                "prefixItems": [self._type_to_schema(arg, seen_models, strict) for arg in args],
                "minItems": len(args),
                "maxItems": len(args),
            }

        if origin is dict:
            if strict:
                raise SchemaConversionError(
                    "dict/Dict fields cannot be represented in strict JSON schema "
                    "(additionalProperties must be false); use an annotated class instead"
                )
            schema = {"type": SchemaType.OBJECT.value}
            if len(args) == 2:
                schema["additionalProperties"] = self._type_to_schema(args[1], seen_models, strict)
            return schema

        # Bare collection classes: get_origin() is None for unsubscripted types.
        if python_type is list or python_type is tuple:
            return {"type": SchemaType.ARRAY.value, "items": {}}
        if python_type in (set, frozenset):
            return {"type": SchemaType.ARRAY.value, "items": {}, "uniqueItems": True}
        if python_type is dict:
            if strict:
                raise SchemaConversionError(
                    "dict/Dict fields cannot be represented in strict JSON schema "
                    "(additionalProperties must be false); use an annotated class instead"
                )
            return {"type": SchemaType.OBJECT.value}

        if self._is_union_type(python_type):
            non_none = [arg for arg in args if arg is not type(None)]
            variants = [self._type_to_schema(arg, seen_models, strict) for arg in non_none]
            if len(non_none) != len(args):
                variants.append({"type": SchemaType.NULL.value})
            return {"anyOf": variants}

        if origin is Literal:
            return self._enum_schema([self._literal_value(value) for value in args])

        if isinstance(python_type, type) and issubclass(python_type, Enum):
            return self._enum_schema([member.value for member in python_type])

        if (
            isinstance(python_type, type)
            and issubclass(python_type, tuple)
            and hasattr(python_type, "_fields")
        ):
            # NamedTuple -> typed array (matches OpenAI/pydantic tuple mapping)
            try:
                hints = get_type_hints(python_type)
            except Exception as e:
                raise SchemaConversionError(
                    f"Could not resolve type hints for NamedTuple {python_type.__name__}: {e}"
                ) from e
            return {
                "type": SchemaType.ARRAY.value,
                "prefixItems": [
                    self._type_to_schema(hints.get(field_name, Any), seen_models, strict)
                    for field_name in python_type._fields
                ],
                "minItems": len(python_type._fields),
                "maxItems": len(python_type._fields),
            }

        # Known types before the annotated-class check: a str/int/datetime
        # subclass that happens to carry class annotations must stay a
        # primitive, not become a nested object. Subclasses are matched via
        # the MRO (identity map lookups miss them).
        if isinstance(python_type, type):
            for base in python_type.__mro__:
                if base in self._PRIMITIVE_TYPE_MAP:
                    return {"type": self._PRIMITIVE_TYPE_MAP[base].value}
                if base in self._KNOWN_TYPE_SCHEMAS:
                    return dict(self._KNOWN_TYPE_SCHEMAS[base])

        if self._is_annotated_class(python_type):
            if python_type in seen_models:
                raise SchemaConversionError(
                    f"Circular dependency detected for class {python_type.__name__}. "
                    "Recursive schemas are not supported."
                )
            nested_schema = self.convert_class_to_schema(
                python_type, seen_models=seen_models, strict=strict
            )
            return nested_schema["json_schema"]["schema"]

        raise SchemaConversionError(
            f"Cannot map {python_type!r} to JSON schema. Supported: bool/int/float/str, "
            "list/set/frozenset/tuple, dict (non-strict only), Literal/Enum, Optional "
            "unions, datetime/date/time, UUID, Decimal, Path, bytes, Any, NamedTuple, "
            "and annotated classes."
        )

    def _is_annotated_class(self, python_type: Any) -> bool:
        if not isinstance(python_type, type):
            return False
        if issubclass(python_type, Enum):
            return False
        if issubclass(python_type, tuple) and hasattr(python_type, "_fields"):
            return False
        try:
            return bool(get_type_hints(python_type))
        except Exception:
            return False

    @staticmethod
    def _literal_value(value: Any) -> Any:
        return value.value if isinstance(value, Enum) else value

    @classmethod
    def _enum_schema(cls, values: list[Any]) -> dict[str, Any]:
        non_none = [value for value in values if value is not None]
        parts: list[dict[str, Any]] = []
        if non_none:
            parts.append(cls._typed_enum_schema(non_none))
        if len(non_none) != len(values):
            # Literal['a', None] / Literal[None] are valid Python: model the
            # None values as an explicit null variant.
            parts.append({"type": SchemaType.NULL.value})
        if len(parts) == 1:
            return parts[0]
        return {"anyOf": parts}

    @classmethod
    def _typed_enum_schema(cls, values: list[Any]) -> dict[str, Any]:
        kinds = sorted({type(value) for value in values}, key=lambda kind: kind.__name__)
        if not kinds:
            return {"type": SchemaType.NULL.value}
        if len(kinds) == 1:
            kind = kinds[0]
            json_type = cls._JSON_VALUE_TYPES.get(kind)
            if json_type is None:
                raise SchemaConversionError(
                    f"Enum values of type {kind.__name__} cannot be represented in JSON schema"
                )
            return {"type": json_type, "enum": list(values)}
        parts = []
        for kind in kinds:
            json_type = cls._JSON_VALUE_TYPES.get(kind)
            if json_type is None:
                raise SchemaConversionError(
                    f"Enum values of type {kind.__name__} cannot be represented in JSON schema"
                )
            parts.append({
                "type": json_type,
                "enum": [value for value in values if type(value) is kind],
            })
        return {"anyOf": parts}

    def is_llm_supported_type(self, python_type: Any) -> bool:
        """Check whether a Python type can be converted to a JSON schema."""
        if python_type is None or python_type is type(None):
            return True
        try:
            self.python_type_to_json_schema(python_type, strict=False)
        except SchemaConversionError:
            return False
        return True

    # -- class introspection ---------------------------------------------------

    def _class_type_hints(self, schema_class: type) -> dict[str, Any]:
        # TypedDicts: plain hints (Required/NotRequired are unwrapped and
        # optionality comes from __required_keys__/__optional_keys__);
        # everything else keeps Annotated metadata for descriptions.
        include_extras = not is_typeddict(schema_class)
        try:
            hints = get_type_hints(schema_class, include_extras=include_extras)
        except Exception as e:
            raise SchemaConversionError(
                f"Could not resolve type hints for class {schema_class.__name__}: {e}"
            ) from e
        return {
            name: annotation
            for name, annotation in hints.items()
            if not name.startswith("_") and get_origin(annotation) is not ClassVar
        }

    def _class_fields(self, schema_class: type, hints: dict[str, Any]) -> dict[str, _SchemaField]:
        fields_map: dict[str, _SchemaField] = {}

        if is_typeddict(schema_class):
            required_keys = set(getattr(schema_class, "__required_keys__", ()) or ())
            for name, annotation in hints.items():
                fields_map[name] = _SchemaField(
                    annotation=annotation, optional=name not in required_keys
                )
            return fields_map

        if is_dataclass(schema_class):
            defaults = {
                f.name: f for f in dataclass_fields(schema_class) if f.name in hints
            }
            for name, annotation in hints.items():
                field_info = defaults.get(name)
                has_default = field_info is not None and (
                    field_info.default is not MISSING or field_info.default_factory is not MISSING
                )
                description = None
                if field_info is not None and isinstance(field_info.metadata, Mapping):
                    description = field_info.metadata.get("description")
                fields_map[name] = _SchemaField(
                    annotation=annotation,
                    optional=has_default,
                    description=description if isinstance(description, str) else None,
                )
            return fields_map

        # Pydantic v2 models keep defaults/descriptions in model_fields.
        model_fields = getattr(schema_class, "model_fields", None)
        if isinstance(model_fields, dict) and not is_dataclass(schema_class):
            for name, hint_annotation in hints.items():
                field_info = model_fields.get(name)
                annotation = hint_annotation
                optional = False
                description = None
                if field_info is not None:
                    # Prefer the resolved field annotation (handles aliases
                    # and forward refs better than get_type_hints alone).
                    field_annotation = getattr(field_info, "annotation", None)
                    if field_annotation is not None:
                        annotation = field_annotation
                    try:
                        optional = not field_info.is_required()
                    except Exception:
                        optional = False
                    description = getattr(field_info, "description", None)
                fields_map[name] = _SchemaField(
                    annotation=annotation,
                    optional=optional,
                    description=description if isinstance(description, str) else None,
                )
            return fields_map

        # Plain class: walk the MRO so inherited defaults are found.
        defaults: dict[str, Any] = {}
        for base in reversed(schema_class.__mro__):
            for name, value in vars(base).items():
                if not name.startswith("_") and not callable(value):
                    defaults[name] = value
        for name, annotation in hints.items():
            fields_map[name] = _SchemaField(
                annotation=annotation, optional=name in defaults
            )
        return fields_map

    @staticmethod
    def _class_docstring(schema_class: type) -> Optional[str]:
        # Only the class's own docstring: getdoc() would inherit dict's or
        # BaseModel's docstring, and dataclasses generate a "Name(...)" signature.
        doc = schema_class.__dict__.get("__doc__")
        if not isinstance(doc, str) or not doc.strip():
            return None
        if doc.startswith(schema_class.__name__ + "("):
            return None
        cleaned = inspect.cleandoc(doc)
        return cleaned or None

    @classmethod
    def _validate_strict_schema(cls, schema: Any, *, path: str = "$", context: str = "") -> None:
        """Validate OpenAI strict-mode rules (better errors than the server)."""
        if not isinstance(schema, dict):
            return
        properties = schema.get("properties")
        if isinstance(properties, dict):
            required = schema.get("required")
            if not isinstance(required, list) or set(required) != set(properties):
                raise SchemaConversionError(
                    f"Strict schema{f' for {context}' if context else ''}: 'required' must be "
                    f"an array including every key in 'properties' (at {path})"
                )
            if schema.get("additionalProperties") is not False:
                raise SchemaConversionError(
                    f"Strict schema{f' for {context}' if context else ''}: objects must set "
                    f"additionalProperties: false (at {path})"
                )
            for key, sub_schema in properties.items():
                cls._validate_strict_schema(sub_schema, path=f"{path}.{key}", context=context)
        items = schema.get("items")
        if isinstance(items, dict):
            cls._validate_strict_schema(items, path=f"{path}.items", context=context)
        for sub_schema in schema.get("prefixItems", []) or []:
            if isinstance(sub_schema, dict):
                cls._validate_strict_schema(sub_schema, path=f"{path}.prefixItems[]", context=context)
        for sub_schema in schema.get("anyOf", []) or []:
            if isinstance(sub_schema, dict):
                cls._validate_strict_schema(sub_schema, path=f"{path}.anyOf[]", context=context)

    def convert_class_to_schema(
        self,
        schema_class: type,
        name: Optional[str] = None,
        seen_models: Optional[set] = None,
        *,
        strict: bool = True,
    ) -> dict[str, Any]:
        """Convert a plain/data/TypedDict class into an OpenAI JSON schema payload.

        In strict mode every field is listed in ``required`` (OpenAI requires
        this); optionality is expressed via nullable ``anyOf`` unions instead.
        """
        seen_models = seen_models if seen_models is not None else set()

        if not isinstance(schema_class, type):
            raise SchemaConversionError(
                f"output_format class must be a class, got {schema_class!r}"
            )

        hints = self._class_type_hints(schema_class)
        if not hints:
            raise SchemaConversionError(
                f"Class {schema_class.__name__} has no type annotations."
            )

        seen_models.add(schema_class)
        try:
            class_fields = self._class_fields(schema_class, hints)
            properties: dict[str, Any] = {}
            required: list[str] = []

            for field_name, field_info in class_fields.items():
                prop = self._type_to_schema(field_info.annotation, seen_models, strict)
                if field_info.description:
                    prop = dict(prop)
                    prop.setdefault("description", field_info.description)
                if strict:
                    # Strict mode: every key is required; defaults become nullable.
                    if field_info.optional and not self.is_optional_type(field_info.annotation):
                        prop = {"anyOf": [prop, {"type": SchemaType.NULL.value}]}
                    required.append(field_name)
                elif not field_info.optional and not self.is_optional_type(field_info.annotation):
                    required.append(field_name)
                properties[field_name] = prop

            schema = self._ordered_object_schema(
                required=required,
                properties=properties,
                additional_properties=False,
            )

            if strict:
                self._validate_strict_schema(schema, context=schema_class.__name__)

            doc = self._class_docstring(schema_class)
            if doc:
                schema["description"] = doc

            return {
                "type": "json_schema",
                "json_schema": {
                    "name": name or schema_class.__name__,
                    "strict": strict,
                    "schema": schema,
                },
            }
        finally:
            seen_models.discard(schema_class)

# ============================================================================
# Tool Preparator
# ============================================================================

@dataclass
class PreparedTools:
    """Result of tool preparation."""
    definitions: list[dict[str, Any]]
    callables: dict[str, Callable] = field(default_factory=dict)


class ToolPreparator:
    """Prepares tools for LLM consumption."""

    def __init__(self, schema_converter: SchemaConverter):
        self._converter = schema_converter

    def prepare(self, tools: Optional[list[Any]]) -> PreparedTools:
        """Convert callables / tool dicts to OpenAI tool definitions."""
        if not tools:
            return PreparedTools([])

        definitions: list[dict[str, Any]] = []
        callables: dict[str, Callable] = {}
        seen_names: dict[str, int] = {}

        for index, tool in enumerate(tools):
            if callable(tool):
                definition = self._prepare_callable(tool, index)
            elif isinstance(tool, dict):
                self._validate_tool_dict(tool, index)
                definition = self._normalize_tool_dict(tool)
            else:
                raise ConfigurationError(
                    f"Tool at index {index} must be callable or dict, got {type(tool).__name__}"
                )

            definitions.append(definition)

            function = definition.get("function") if isinstance(definition, dict) else None
            function_name = function.get("name") if isinstance(function, dict) else None
            if function_name:
                if function_name in seen_names:
                    raise ConfigurationError(
                        f"Duplicate tool name {function_name!r} (tools at index "
                        f"{seen_names[function_name]} and {index}); tool names must be unique"
                    )
                seen_names[function_name] = index
                if callable(tool):
                    callables[function_name] = tool

        return PreparedTools(definitions, callables)

    def _prepare_callable(self, func: Callable, index: int) -> dict:
        """Prepare a callable for LLM consumption."""
        underlying = self._unwrap_callable(func)

        name = (
            getattr(func, "__name__", None)
            or getattr(underlying, "__name__", None)
            or ""
        ).strip()
        if not name or not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
            raise ConfigurationError(
                f"Tool at index {index}: could not derive a valid tool name (got {name!r}); "
                "use a named function, functools.partial, or a dict tool definition"
            )

        doc = (
            getattr(underlying, "__doc__", None)
            or getattr(func, "__doc__", None)
            or ""
        ).strip()
        doc = inspect.cleandoc(doc) if doc else ""
        # Only the summary goes into the tool description; parameter details
        # live on the parameters themselves (avoids duplicate prompt tokens).
        summary_doc = self._strip_param_sections(doc)

        try:
            annotations = get_type_hints(underlying, include_extras=True)
        except Exception as e:
            raise ConfigurationError(
                f"Could not resolve type hints for tool {name!r}: {e}. "
                "Make sure forward references are resolvable at runtime."
            ) from e

        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError) as e:
            raise ConfigurationError(
                f"Could not inspect signature of tool {name!r}: {e}"
            ) from e

        param_docs = self._parse_docstring_params(doc)
        parameters: dict[str, Any] = {}
        required: list[str] = []
        saw_parameter = False

        for param_name, param in signature.parameters.items():
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                logger.debug(
                    "Ignoring *%s of tool %r; only named parameters become schema fields",
                    param_name, name,
                )
                continue
            if param.kind is inspect.Parameter.POSITIONAL_ONLY:
                raise ConfigurationError(
                    f"Tool {name!r} has positional-only parameter {param_name!r}; "
                    "tool arguments are always passed by keyword"
                )
            saw_parameter = True

            param_type = annotations.get(param_name)
            has_default = param.default is not inspect.Parameter.empty

            try:
                param_schema = (
                    dict(
                        self._converter.python_type_to_json_schema(param_type, strict=False)
                    )
                    if param_type is not None
                    else {"type": SchemaType.STRING.value}
                )
            except SchemaConversionError as e:
                if has_default:
                    logger.debug(
                        "Skipping non-representable parameter %r of tool %r",
                        param_name, name,
                    )
                    continue
                raise ConfigurationError(
                    f"Tool {name!r} parameter {param_name!r}: {e} "
                    "Annotate it with a supported type, or give it a default "
                    "to exclude it from the schema."
                ) from e

            description_parts: list[str] = []
            existing_description = param_schema.get("description")
            if isinstance(existing_description, str) and existing_description:
                description_parts.append(existing_description)
            if param_name in param_docs:
                description_parts.append(param_docs[param_name])
            if has_default:
                description_parts.append(f"Default: {self._format_default(param.default)}")
            if description_parts:
                param_schema["description"] = " ".join(
                    part for part in description_parts if part
                )

            if not has_default:
                required.append(param_name)
            parameters[param_name] = param_schema

        if not saw_parameter and any(
            param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for param in signature.parameters.values()
        ):
            logger.warning(
                "Tool %r only has *args/**kwargs parameters; its schema will be empty", name
            )

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": summary_doc,
                "parameters": self._converter._ordered_object_schema(
                    required=required,
                    properties=parameters,
                    additional_properties=False,
                ),
            },
        }

    @staticmethod
    def _unwrap_callable(func: Callable) -> Callable:
        underlying = inspect.unwrap(func)
        while isinstance(underlying, functools.partial):
            underlying = inspect.unwrap(underlying.func)
        return underlying

    @staticmethod
    def _format_default(value: Any) -> str:
        if isinstance(value, str):
            return f'"{value}"'
        if value is None:
            return "null"
        return repr(value)

    @staticmethod
    def _parse_docstring_params(doc: str) -> dict[str, str]:
        """Extract parameter descriptions from Google/NumPy-style and Sphinx :param: docstrings."""
        if not doc:
            return {}
        params: dict[str, str] = {}
        lines = doc.splitlines()
        index = 0
        total = len(lines)
        while index < total:
            stripped = lines[index].strip()
            if ToolPreparator._is_param_section_header(stripped, lines, index):
                index += 2 if re.match(r"^-{3,}\s*$", lines[index + 1].strip()) else 1
                current: Optional[str] = None
                while index < total:
                    entry = lines[index].strip()
                    if not entry:
                        index += 1
                        continue
                    # The next section header ends this section (entries may
                    # be flush left in NumPy style, so indentation alone
                    # cannot decide).
                    if ToolPreparator._is_param_section_header(entry, lines, index):
                        break
                    # NumPy entries use " : " (space before the colon); their
                    # description follows on indented lines, not inline.
                    numpy_entry = re.match(r"^(\*?\*?\w+)\s+:\s+(.*)$", entry)
                    google_entry = re.match(r"^(\*?\*?\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$", entry)
                    if google_entry:
                        current = google_entry.group(1).lstrip("*")
                        description = google_entry.group(2).strip()
                        if (
                            numpy_entry
                            and description
                            and index + 1 < total
                            and lines[index + 1].startswith((" ", "\t"))
                        ):
                            # "name : type" — the type is not the description.
                            description = ""
                        params.setdefault(current, description)
                    elif current is not None:
                        params[current] = (params[current] + " " + entry).strip()
                    index += 1
                continue
            sphinx = re.match(r"^:param\s+(\w+)\s*:\s*(.*)$", stripped)
            if sphinx:
                params[sphinx.group(1)] = sphinx.group(2).strip()
            index += 1
        return params

    @staticmethod
    def _is_param_section_header(stripped: str, lines: list[str], index: int) -> bool:
        lowered = stripped.lower().rstrip(":")
        if stripped.endswith(":") and lowered in (
            "args", "arguments", "parameters", "params",
            "returns", "raises", "yields", "examples", "notes",
        ):
            return True
        # NumPy/RST style: any underlined heading ("Parameters", "Returns", …).
        if (
            stripped
            and index + 1 < len(lines)
            and re.match(r"^-{3,}\s*$", lines[index + 1].strip())
        ):
            return True
        return bool(re.match(r"^:(?:param|arg|type)\b", stripped))

    @classmethod
    def _strip_param_sections(cls, doc: str) -> str:
        """Return the docstring summary before the first parameter section."""
        if not doc:
            return ""
        lines = doc.splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if cls._is_param_section_header(stripped, lines, index):
                return "\n".join(lines[:index]).rstrip()
        return doc

    @staticmethod
    def _validate_tool_dict(tool: dict, index: int) -> None:
        if "type" not in tool:
            raise ConfigurationError(f"Tool at index {index} must have a 'type' key")
        if tool.get("type") != "function":
            return  # Responses-native / built-in tools pass through unchanged
        function = tool.get("function")
        if isinstance(function, dict):
            if not function.get("name"):
                raise ConfigurationError(
                    f"Function tool at index {index} missing 'name' in function definition"
                )
            return
        if "function" not in tool and tool.get("name"):
            return  # already a Responses-shaped function tool
        raise ConfigurationError(
            f"Function tool at index {index} must have a 'function' object with a 'name'"
        )

    @staticmethod
    def _normalize_tool_dict(tool: dict) -> dict:
        """Flatten Responses-shaped function tools into Chat-Completions shape.

        The chat path needs the nested ``function`` object; the Responses
        conversion re-flattens later, so one shape works for both modes.
        """
        if tool.get("type") == "function" and "function" not in tool and tool.get("name"):
            return {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description") or "",
                    "parameters": tool.get("parameters") or {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                    **({"strict": tool["strict"]} if "strict" in tool else {}),
                },
            }
        return tool

# ============================================================================
# Message Helpers
# ============================================================================

def tool_result(tool_call: ToolCall, result: Any) -> ToolResultMessage:
    """Format a tool call result for LLM consumption."""
    if not isinstance(tool_call, dict):
        raise ConfigurationError("tool_result needs a tool call dict")
    call_id = tool_call.get("id", "")
    if not isinstance(call_id, str) or not call_id:
        raise ConfigurationError(
            "tool_result needs a tool call with a non-empty string 'id'"
        )
    if not isinstance(result, str):
        result = _dump_json(result, what="tool result")
    message: ToolResultMessage = {
        "role": "tool",
        "tool_call_id": call_id,
        "content": result,
    }
    name = tool_call.get("name")
    if name:
        message["name"] = name
    return message


def assistant_message(
    final_response: FinalResponse,
    *,
    include_reasoning: bool = False,
) -> AssistantMessage:
    """Format an assistant message (with optional tool calls) for LLM consumption."""
    tool_calls = []
    for tc in final_response.get("tool_calls") or []:
        if not isinstance(tc, dict):
            raise ConfigurationError("assistant tool_calls must be tool call dicts")
        args_val = tc.get("arguments", {})
        raw = ""
        if isinstance(args_val, dict) and set(args_val) == {"_raw"}:
            raw = args_val["_raw"]
        if isinstance(raw, str) and raw:
            args_str = raw
        elif isinstance(args_val, (dict, list)):
            args_str = _dump_json(args_val, what="tool arguments")
        else:
            args_str = args_val or "{}"
        if not isinstance(args_str, str):
            args_str = _dump_json(args_str, what="tool arguments")
        tool_calls.append({
            "id": tc.get("id", ""),
            "type": "function",
            "function": {
                "name": tc.get("name", ""),
                "arguments": args_str,
            },
        })

    answer = final_response.get("answer")
    if answer is not None and not isinstance(answer, str):
        answer = _dump_json(answer, what="assistant answer")

    message: dict[str, Any] = {"role": "assistant"}
    if answer not in (None, ""):
        # Structured answers like 0 or False must survive (a falsy check
        # silently dropped them).
        message["content"] = answer
    elif not tool_calls:
        message["content"] = ""
    if tool_calls:
        # Only set the key when there are calls: "tool_calls": null is
        # rejected by several strict OpenAI-compatible servers.
        message["tool_calls"] = tool_calls
    if include_reasoning and final_response.get("reasoning"):
        message["reasoning_content"] = final_response["reasoning"]
    response_items = final_response.get("response_items")
    if response_items:
        # Responses-API reasoning items must be replayed alongside the tool
        # calls on the next turn (see _to_responses_input); copied so later
        # chat-side stripping never mutates the caller's list.
        message["response_items"] = [
            dict(item) if isinstance(item, dict) else item for item in response_items
        ]
    return message


def user_message(content: str | list[dict[str, Any]]) -> UserMessage:
    """Format a user message for LLM consumption."""
    return {
        "role": "user",
        "content": content,
    }


def system_message(content: str) -> dict[str, str]:
    """Format a system message for LLM consumption."""
    return {
        "role": "system",
        "content": content,
    }

# ============================================================================
# Request Transformer
# ============================================================================

class RequestTransformer:
    """Provider/model-specific request normalizer (mutates kwargs in place)."""

    # Known values for warning purposes only — new models keep adding values
    # ('none' for gpt-5.1, 'xhigh' for codex-max) and providers accept their
    # own, so unknown strings pass through with a warning instead of raising.
    _KNOWN_REASONING_EFFORTS: ClassVar[frozenset[str]] = frozenset(
        {"minimal", "low", "medium", "high", "none", "xhigh"}
    )
    _REASONING_MODEL_RE: ClassVar[re.Pattern] = re.compile(r"^(o\d|gpt-5)", re.IGNORECASE)

    def __init__(self, model: str, api_base: str):
        self._model = model
        self._api_base = api_base.lower()
        self._warned: set[str] = set()

    def transform(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        self._validate_reasoning_effort(kwargs)
        self._normalize_max_tokens(kwargs)
        self._drop_reasoning_model_conflicts(kwargs)
        self._normalize_reasoning(kwargs)
        return kwargs

    def _warn_once(self, key: str, message: str, *args: Any) -> None:
        if len(self._warned) > 512:
            self._warned.clear()
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(message, *args)

    def _on_openai_host(self) -> bool:
        return urlparse(self._api_base).hostname == "api.openai.com"

    def _validate_reasoning_effort(self, kwargs: dict[str, Any]) -> None:
        effort = kwargs.get("reasoning_effort")
        if effort is None:
            return
        if not isinstance(effort, str) or not effort.strip():
            raise ConfigurationError(
                f"reasoning_effort must be a non-empty string, got {type(effort).__name__}"
            )
        _reject_confusable_controls(effort, what="reasoning_effort")
        effort = effort.strip()
        if effort not in self._KNOWN_REASONING_EFFORTS:
            self._warn_once(
                f"effort:{effort}",
                "reasoning_effort %r is not a commonly supported value; passing it through",
                effort,
            )

    def _normalize_max_tokens(self, kwargs: dict[str, Any]) -> None:
        """Map max_tokens to max_completion_tokens where required (o-series/gpt-5, OpenAI)."""
        if "max_tokens" not in kwargs:
            return
        if self._on_openai_host() or self._REASONING_MODEL_RE.match(self._model):
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")

    def _drop_reasoning_model_conflicts(self, kwargs: dict[str, Any]) -> None:
        # Host-gated: a local vLLM model merely named "o3-..." keeps its
        # parameters; only api.openai.com enforces the reasoning-model rules.
        if not self._on_openai_host():
            return
        if not self._REASONING_MODEL_RE.match(self._model):
            return
        if kwargs.pop("stop", None) is not None:
            self._warn_once(
                "stop",
                "Dropping 'stop': not supported by reasoning model %r on api.openai.com",
                self._model,
            )
        for key in ("temperature", "top_p"):
            if kwargs.pop(key, None) is not None:
                self._warn_once(
                    f"drop:{key}",
                    "Dropping %r: not supported by reasoning model %r on api.openai.com",
                    key,
                    self._model,
                )

    def _normalize_reasoning(self, kwargs: dict[str, Any]) -> None:
        """Map the reasoning pair onto the single Chat wire format.

        Exactly one mapping, no host detection: effort becomes top-level
        ``reasoning_effort`` (OpenAI-native), budget becomes a top-level
        ``reasoning`` object with ``max_tokens``. Both set is rejected.
        """
        effort = kwargs.pop("reasoning_effort", None)
        budget = kwargs.pop("reasoning_budget", None)
        if effort is not None and budget is not None:
            raise ConfigurationError(
                "reasoning_effort and reasoning_budget are mutually exclusive; set only one"
            )
        if budget is not None and (
            isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0
        ):
            raise ConfigurationError("reasoning_budget must be a positive int or None")
        if effort is not None:
            # Validated non-empty upstream; normalize padding so padded
            # values can't reach the wire and 400 downstream.
            kwargs["reasoning_effort"] = effort.strip() if isinstance(effort, str) else effort
        elif budget is not None:
            kwargs["reasoning"] = {"max_tokens": budget}

# ============================================================================
# Image Processor
# ============================================================================

class ImageProcessor:
    """Processes images in messages for API consumption.

    Sources: ``image_path``/``image_pil``/``image_url``/``image_base64`` plus
    optional top-level ``detail`` (injected into the ``image_url`` part;
    explicit dict-level ``detail`` wins). Small images pass through
    byte-identical, oversized ones shrink toward the send budget.
    Bad content raises ``ImageProcessingError``, bad types/URLs raise
    ``ConfigurationError``.
    """

    _pil_image = None

    _RAW_PASS_THROUGH_FORMATS: ClassVar[frozenset[str]] = frozenset(
        {"image/jpeg", "image/png", "image/gif", "image/webp"}
    )
    """MIME types sent byte-identical when small enough (mirrors _FORMAT_MIME)."""

    # Read cap (OOM guard) and send budget (adaptive downscale target).
    _MAX_IMAGE_READ_BYTES: ClassVar[int] = 50 * 1024 * 1024
    _MAX_IMAGE_SEND_BYTES: ClassVar[int] = 15 * 1024 * 1024
    # Hard ceiling for inline base64 (~150MB decoded); above it, reject.
    _MAX_BASE64_HARD_CHARS: ClassVar[int] = 200 * 1024 * 1024
    # Header check before any full decode (compressed size says little).
    _MAX_IMAGE_PIXELS: ClassVar[int] = 100 * 1024 * 1024
    _FORMAT_MIME: ClassVar[dict[str, str]] = {
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "PNG": "image/png",
        "GIF": "image/gif",
        "WEBP": "image/webp",
    }

    @classmethod
    def _get_pil(cls):
        if cls._pil_image is None:
            try:
                from PIL import Image
            except ImportError:
                raise ImportError(
                    "PIL/Pillow required. Install with: pip install Pillow"
                ) from None
            from PIL import __version__ as pillow_version
            version_match = re.match(r"(\d+)\.(\d+)", pillow_version)
            if version_match is None or (
                int(version_match.group(1)), int(version_match.group(2))
            ) < (10, 0):
                raise ImageProcessingError(
                    f"Pillow >= 10 is required, found {pillow_version}; "
                    "upgrade with: pip install -U Pillow"
                )
            cls._pil_image = Image
            # Defense in depth: Pillow's own bomb guard matches our 100MP
            # header check, so races/quirks in header parsing can't allocate
            # gigapixel buffers (raised as DecompressionBombError, wrapped
            # into ImageProcessingError by the callers below).
            Image.MAX_IMAGE_PIXELS = ImageProcessor._MAX_IMAGE_PIXELS
        return cls._pil_image

    @staticmethod
    def process_messages(messages: list[dict], *, max_image_side: Optional[int] = 8192) -> None:
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for index, item in enumerate(content):
                if not isinstance(item, dict) or item.get("type") != "image":
                    continue
                msg["content"][index] = ImageProcessor._convert_image_item(
                    item, max_image_side=max_image_side
                )

    @staticmethod
    def _convert_image_item(item: dict, *, max_image_side: Optional[int] = 8192) -> dict:
        detail = _validate_detail(item.get("detail"), what="image detail")
        if "image_path" in item:
            converted = ImageProcessor._from_path(item["image_path"], max_image_side=max_image_side)
        elif "image_pil" in item:
            converted = ImageProcessor._from_pil(item["image_pil"], max_image_side=max_image_side)
        elif "image_url" in item:
            converted = ImageProcessor._from_url(
                item["image_url"], max_image_side=max_image_side
            )
        elif "image_base64" in item:
            converted = ImageProcessor._from_base64(
                item["image_base64"], max_image_side=max_image_side
            )
        else:
            raise ConfigurationError(
                "image items need one of image_path, image_pil, "
                "image_url or image_base64"
            )
        if detail is not None and converted.get("type") == "image_url":
            url_data = converted.get("image_url")
            if isinstance(url_data, dict):
                # Explicit dict-level detail wins over the top-level knob.
                url_data.setdefault("detail", detail)
        return converted

    @staticmethod
    def _from_path(path: Any, *, max_image_side: Optional[int] = 8192) -> dict:
        _validate_max_image_side(max_image_side)
        if not isinstance(path, (str, Path)):
            raise ConfigurationError(
                f"image_path must be a file path string or pathlib.Path, got {type(path).__name__}"
            )
        try:
            if not Path(path).is_file():
                raise ImageProcessingError("image_path is not a regular file")
        except ImageProcessingError:
            raise
        except (OSError, ValueError) as e:
            detail = getattr(e, "strerror", None) or getattr(e, "errno", None) or "invalid path"
            raise ImageProcessingError(f"Cannot access image ({detail})") from e
        try:
            with open(path, "rb") as handle:
                raw = handle.read(ImageProcessor._MAX_IMAGE_READ_BYTES + 1)
        except (OSError, ValueError) as e:
            detail = getattr(e, "strerror", None) or getattr(e, "errno", None) or "invalid path"
            raise ImageProcessingError(f"Failed to read image ({detail})") from e
        if len(raw) > ImageProcessor._MAX_IMAGE_READ_BYTES:
            raise ImageProcessingError(
                "Image is larger than the 50MB read limit; crop or shrink it first"
            )

        # Must decode as a real image: text files or keys are rejected here
        # instead of being sent to the server disguised as images.
        Image = ImageProcessor._get_pil()
        try:
            with Image.open(io.BytesIO(raw)) as probe:
                dimensions = probe.size
                if dimensions[0] * dimensions[1] > ImageProcessor._MAX_IMAGE_PIXELS:
                    raise ImageProcessingError(
                        "Image dimensions exceed the 100MP limit; crop or shrink it first"
                    )
                probe.load()
                frames = getattr(probe, "n_frames", 1) or 1
                if dimensions[0] * dimensions[1] * frames > ImageProcessor._MAX_IMAGE_PIXELS:
                    raise ImageProcessingError(
                        "Image dimensions exceed the 100MP limit; crop or shrink it first"
                    )
                image_format = (probe.format or "").upper()
                mode = probe.mode
                try:
                    orientation: Optional[int] = probe.getexif().get(274)
                except Exception:
                    orientation = None
        except (ImageProcessingError, ConfigurationError):
            raise
        except Exception as e:
            raise ImageProcessingError("File is not a readable image") from e

        mime = ImageProcessor._FORMAT_MIME.get(image_format)
        side_ok = max_image_side is None or max(dimensions) <= max_image_side
        # Modes _encode would alter (CMYK, 16-bit, ...) must transcode even
        # when small: byte-passthrough would ship unrenderable pixels.
        mode_ok = mode in ("RGB", "RGBA", "L", "LA", "P")
        if (
            mime in ImageProcessor._RAW_PASS_THROUGH_FORMATS
            and side_ok
            and mode_ok
            and len(raw) <= ImageProcessor._MAX_IMAGE_SEND_BYTES
            and orientation in (None, 1)
        ):
            encoded = base64.b64encode(raw).decode("utf-8")
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{encoded}"},
            }
        # Too large for passthrough or needs rotation: transcode (and shrink
        # toward the side cap and send budget if needed).

        try:
            with Image.open(io.BytesIO(raw)) as img:
                img.load()
                return ImageProcessor._encode_pil_image(img, max_image_side=max_image_side)
        except (ImageProcessingError, ConfigurationError):
            raise
        except Exception as e:
            raise ImageProcessingError("Failed to process image") from e

    @staticmethod
    def _from_pil(img: "PILImage", *, max_image_side: Optional[int] = 8192) -> dict:
        ImageProcessor._get_pil()  # friendly error when Pillow is missing
        try:
            return ImageProcessor._encode_pil_image(img, max_image_side=max_image_side)
        except (ImageProcessingError, ConfigurationError):
            raise
        except Exception as e:
            raise ImageProcessingError(f"Failed to encode PIL image: {e}") from e

    @staticmethod
    def _validate_data_url(url: str, *, source: str) -> None:
        """Reject data: URLs whose MIME type is not an image (cheap, no decode).

        An empty MIME type defaults to text/plain per RFC 2397, so it is
        rejected too. The scheme check is case-insensitive.
        """
        if url[:5].lower() != "data:":
            return
        # Parse the header from a short prefix only: splitting the whole URL
        # would copy megabytes of base64 payload.
        comma = url.find(",", 5)
        head = url[5:comma] if 0 < comma <= 5 + 256 else url[5:5 + 256]
        header = head.split(";")[0].strip().lower()
        if not header.startswith("image/"):
            raise ImageProcessingError(
                f"{source} data URL is not an image (MIME: {header or 'missing'})"
            )
        if header == "image/svg+xml" or header.endswith("+xml"):
            raise ImageProcessingError(
                f"{source} data URL is a vector image ({header}); "
                "supply a raster image (jpeg/png/gif/webp) instead"
            )

    @staticmethod
    def _transcode_base64_if_needed(
        cleaned: str, *, max_image_side: Optional[int] = 8192
    ) -> Optional[dict]:
        _validate_max_image_side(max_image_side)
        """Validate base64 image bytes; transcode only if caps/format demand it.

        Returns a transcoded item when the image needs rotation, is no
        passthrough format, or exceeds the side cap / send budget — else
        None (caller keeps the original string verbatim). Raises
        ImageProcessingError for non-images and oversized dimensions.
        Single header-only open in the common case (no pixel buffer).
        """
        raw = base64.b64decode(cleaned, validate=True)
        if len(raw) > ImageProcessor._MAX_IMAGE_READ_BYTES:
            raise ImageProcessingError(
                "Image is larger than the 50MB read limit; crop or shrink it first"
            )
        Image = ImageProcessor._get_pil()
        with Image.open(io.BytesIO(raw)) as img:
            width, height = img.size
            image_format = (img.format or "").upper()
            mode = img.mode
            if width * height > ImageProcessor._MAX_IMAGE_PIXELS:
                raise ImageProcessingError(
                    "Image dimensions exceed the 100MP limit; crop or shrink it first"
                )
            try:
                orientation: Optional[int] = img.getexif().get(274)
            except Exception:
                orientation = None
        mime = ImageProcessor._FORMAT_MIME.get(image_format)
        side_ok = max_image_side is None or max(width, height) <= max_image_side
        mode_ok = mode in ("RGB", "RGBA", "L", "LA", "P")
        if (
            mime in ImageProcessor._RAW_PASS_THROUGH_FORMATS
            and side_ok
            and mode_ok
            and len(raw) <= ImageProcessor._MAX_IMAGE_SEND_BYTES
            and orientation in (None, 1)
        ):
            # Full decode (not just header open + verify): catches truncated
            # files the header check misses, and re-checks dimensions plus
            # the frames*budget for animated images after decoding.
            with Image.open(io.BytesIO(raw)) as check:
                check.load()
                width, height = check.size
                frames = getattr(check, "n_frames", 1) or 1
                if width * height * frames > ImageProcessor._MAX_IMAGE_PIXELS:
                    raise ImageProcessingError(
                        "Image dimensions exceed the 100MP limit; crop or shrink it first"
                    )
            return None
        with Image.open(io.BytesIO(raw)) as img:
            img.load()
            return ImageProcessor._encode_pil_image(img, max_image_side=max_image_side)

    @staticmethod
    def _from_url(url_data: Union[str, dict], *, max_image_side: Optional[int] = 8192) -> dict:
        _validate_max_image_side(max_image_side)
        if isinstance(url_data, str):
            url_data = {"url": url_data}
        url = url_data.get("url") if isinstance(url_data, dict) else None
        if not isinstance(url, str) or not url:
            raise ConfigurationError(
                "image_url must be a URL string or a dict with a 'url' key"
            )
        if isinstance(url_data, dict) and "detail" in url_data:
            # Dict-inherent knobs get the same validation as top-level ones
            # (warn-passthrough, never silent raw); explicit null is dropped
            # so providers never see "detail": null.
            validated = _validate_detail(url_data["detail"], what="image detail")
            url_data = dict(url_data)
            if validated is None:
                del url_data["detail"]
            else:
                url_data["detail"] = validated
        if url[:5].lower() == "data:":
            ImageProcessor._validate_data_url(url, source="image_url")
            _, _, body = url.partition(",")
            if not body:
                raise ImageProcessingError("image data URL is malformed")
            # Raw length overestimates (whitespace is stripped below), so this
            # rejects gross oversize before the expensive regex copy.
            if len(body) > ImageProcessor._MAX_BASE64_HARD_CHARS:
                raise ImageProcessingError(
                    "image data URL exceeds the size limit; crop or shrink it first"
                )
            cleaned = _strip_b64_whitespace(body)
            if len(cleaned) > ImageProcessor._MAX_BASE64_HARD_CHARS:
                raise ImageProcessingError(
                    "image data URL exceeds the size limit; crop or shrink it first"
                )
            try:
                transcoded = ImageProcessor._transcode_base64_if_needed(
                    cleaned, max_image_side=max_image_side
                )
            except (ImageProcessingError, ConfigurationError):
                raise
            except Exception as e:
                raise ImageProcessingError(
                    "image data URL is not a readable image"
                ) from e
            if transcoded is not None:
                new_url = transcoded["image_url"]["url"]
            else:
                new_url = ImageProcessor._normalize_data_url_mime(url, cleaned)
            return {"type": "image_url", "image_url": {**url_data, "url": new_url}}
        else:
            url = _validate_http_url(url, what="image_url")
            url_data = {**url_data, "url": url}
        # Copy: url_data may be the caller's dict (data: URLs fall through
        # to this return without transcoding).
        return {"type": "image_url", "image_url": dict(url_data)}

    @staticmethod
    def _from_base64(data: str, *, max_image_side: Optional[int] = 8192) -> dict:
        _validate_max_image_side(max_image_side)
        if not isinstance(data, str):
            raise ImageProcessingError("image_base64 must be a base64 string")
        body = data
        verbatim_url: Optional[str] = None
        if data[:5].lower() == "data:":
            ImageProcessor._validate_data_url(data, source="image_base64")
            comma = data.find(",")
            if comma == -1:
                raise ImageProcessingError("image_base64 data URL is malformed")
            body = data[comma + 1:]
            verbatim_url = data
        # Raw length overestimates (whitespace is stripped below), so this
        # rejects gross oversize before the expensive regex copy.
        if len(body) > ImageProcessor._MAX_BASE64_HARD_CHARS:
            raise ImageProcessingError(
                "image_base64 exceeds the size limit; crop or shrink it first"
            )
        cleaned = _strip_b64_whitespace(body)
        if len(cleaned) > ImageProcessor._MAX_BASE64_HARD_CHARS:
            raise ImageProcessingError(
                "image_base64 exceeds the size limit; crop or shrink it first"
            )
        try:
            transcoded = ImageProcessor._transcode_base64_if_needed(
                cleaned, max_image_side=max_image_side
            )
        except (ImageProcessingError, ConfigurationError):
            raise
        except Exception as e:
            raise ImageProcessingError(
                "image_base64 is not a readable image"
            ) from e
        if transcoded is not None:
            return transcoded
        if verbatim_url is not None:
            return {
                "type": "image_url",
                "image_url": {
                    "url": ImageProcessor._normalize_data_url_mime(
                        verbatim_url, cleaned
                    )
                },
            }
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{ImageProcessor._sniff_image_mime(cleaned)};base64,{cleaned}"
            },
        }

    @staticmethod
    def _sniff_image_mime(data: str) -> str:
        # Decode only a short prefix; whitespace is stripped first so
        # line-wrapped base64 (common when read from files) still sniffs.
        prefix = re.sub(r"\s+", "", data[:64])[:32]
        padded = prefix + "=" * (-len(prefix) % 4)
        try:
            raw = base64.b64decode(padded, validate=True)[:16]
        except Exception as e:
            raise ImageProcessingError(
                f"Cannot determine image type: {e}"
            ) from e
        mime = ImageProcessor._mime_from_magic(raw)
        if mime is None:
            # Never mislabel: callers use this for unvalidated raw base64
            # (Responses translate path); unknown signatures fail loudly.
            raise ImageProcessingError(
                "Cannot determine image type from base64 data"
            )
        return mime

    @staticmethod
    def _mime_from_magic(raw: bytes) -> Optional[str]:
        if raw.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
            return "image/webp"
        if raw.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        if raw.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if raw.startswith(b"BM"):
            return "image/bmp"
        return None

    @staticmethod
    def _normalize_data_url_mime(url: str, cleaned: str) -> str:
        """Relabel mismatched data: MIME types from the sniffed bytes.

        Declarations like ``image/x-png`` (or a wrong canonical label)
        pass validation but confuse providers; the bytes are already
        verified readable, so only the label is fixed. When the bytes
        cannot be sniffed the original URL is kept (never mislabel).
        """
        comma = url.find(",")
        header_mime = url[5:comma].split(";")[0].strip().lower() if comma != -1 else ""
        try:
            sniffed = ImageProcessor._sniff_image_mime(cleaned)
        except (ImageProcessingError, ConfigurationError, ValueError):
            return url
        if sniffed == header_mime:
            return url
        return f"data:{sniffed};base64,{cleaned}"

    @staticmethod
    def _encode_pil_image(img: "PILImage", *, max_image_side: Optional[int] = 8192) -> dict:
        """Encode a PIL image without needless PNG re-encoding.

        Applies EXIF orientation, keeps alpha via PNG, and uses JPEG for
        opaque images; modes PNG cannot store (CMYK, 16-bit) are converted.
        Animated images are reduced to their first frame.
        Images larger than the side cap or send budget are shrunk stepwise
        (8192 → 4096 → 2048 → 1024 → 512 longest side) instead of failing.
        """
        try:
            from PIL import ImageOps
        except ImportError:
            raise ImportError(
                "PIL/Pillow required. Install with: pip install Pillow"
            ) from None

        if max_image_side is not None and (
            isinstance(max_image_side, bool)
            or not isinstance(max_image_side, int)
            or max_image_side <= 0
        ):
            raise ConfigurationError("max_image_side must be a positive int or None")
        try:
            frames = getattr(img, "n_frames", 1) or 1
            if img.size[0] * img.size[1] * frames > ImageProcessor._MAX_IMAGE_PIXELS:
                raise ImageProcessingError(
                    "Image dimensions exceed the 100MP limit; crop or shrink it first"
                )
            if frames > 1:
                logger.warning(
                    "Animated image will be reduced to its first frame"
                )
            Image = ImageProcessor._get_pil()
            resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS", 1)
            img = ImageOps.exif_transpose(img)
            has_alpha = img.mode in ("RGBA", "LA", "PA") or (
                img.mode == "P" and "transparency" in getattr(img, "info", {})
            )
            if has_alpha:
                if img.mode != "RGBA":
                    img = img.convert("RGBA")
                format_name, mime = "PNG", "image/png"
            else:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                format_name, mime = "JPEG", "image/jpeg"
            save_kwargs: dict[str, Any] = (
                {} if format_name == "PNG" else {"quality": 85}
            )
            original_size = img.size
            working = img
            max_side = max(working.size)
            steps = [step for step in (8192, 4096, 2048, 1024, 512) if step < max_side]
            if max_image_side is not None and max_image_side < max_side:
                steps.append(max_image_side)
            sizes = [max_side] + sorted(set(steps), reverse=True)
            payload = b""
            for index, side in enumerate(sizes):
                if index:
                    shrunken = working.copy()
                    shrunken.thumbnail((side, side), resample=resampling)
                    working = shrunken
                buffer = io.BytesIO()
                working.save(buffer, format=format_name, **save_kwargs)
                payload = buffer.getvalue()
                side_ok = max_image_side is None or max(working.size) <= max_image_side
                if side_ok and len(payload) <= ImageProcessor._MAX_IMAGE_SEND_BYTES:
                    break
            if max_image_side is not None and max(working.size) > max_image_side:
                shrunken = working.copy()
                shrunken.thumbnail(
                    (max_image_side, max_image_side), resample=resampling
                )
                working = shrunken
                buffer = io.BytesIO()
                working.save(buffer, format=format_name, **save_kwargs)
                payload = buffer.getvalue()
            if working.size != original_size:
                logger.debug(
                    "Downscaled image from %s to %s",
                    original_size, working.size,
                )
            if len(payload) > ImageProcessor._MAX_IMAGE_SEND_BYTES:
                logger.debug(
                    "Image payload still exceeds send budget after downscaling"
                )
            encoded = base64.b64encode(payload).decode("utf-8")
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{encoded}"},
            }
        except (ImageProcessingError, ConfigurationError):
            raise
        except Exception as e:
            raise ImageProcessingError(f"Failed to encode PIL image: {e}") from e

# ============================================================================
# Audio Processor
# ============================================================================

class AudioProcessor:
    """Processes audio in messages for API consumption (input only).

    Local files and base64 become ``input_audio`` parts (format sniffed from
    magic bytes; wav/mp3/aiff/aac/ogg/flac/m4a incl. aifc/rifx/adif/rf64
    variants, 25MB cap); remote URLs pass through untouched (URL in, URL out) — the
    provider decides. Bad content raises ``AudioProcessingError``.
    """

    _MAX_AUDIO_READ_BYTES: ClassVar[int] = 25 * 1024 * 1024
    """Read cap (OOM guard); matches common provider upload limits."""
    _MAX_AUDIO_BASE64_CHARS: ClassVar[int] = 40 * 1024 * 1024
    """Inline base64 ceiling (~30MB decoded); above it, reject."""

    _AUDIO_FORMATS: ClassVar[frozenset[str]] = frozenset(
        {"wav", "mp3", "aiff", "aac", "ogg", "flac", "m4a"}
    )

    # Canonical MIME subtypes seen in the wild, mapped to _AUDIO_FORMATS.
    _AUDIO_MIME_ALIASES: ClassVar[dict[str, str]] = {
        "mpeg": "mp3",
        "wave": "wav",
        "x-wav": "wav",
        "mp4": "m4a",
        "x-m4a": "m4a",
        "x-aiff": "aiff",
        "x-aifc": "aiff",
        "x-flac": "flac",
    }

    @staticmethod
    def _sniff_audio_format(raw: bytes) -> Optional[str]:
        """Detect audio container from magic bytes; None when not audio."""
        if len(raw) < 12:
            return None
        if raw.startswith((b"RIFF", b"RF64", b"RIFX")) and raw[8:12] == b"WAVE":
            return "wav"
        if raw.startswith(b"BW64"):
            # Broadcast Wave 64-bit variant: the magic alone is unambiguous.
            return "wav"
        if raw.startswith(b"FORM") and raw[8:12] in (b"AIFF", b"AIFC"):
            return "aiff"
        if raw.startswith(b"OggS"):
            return "ogg"
        if raw.startswith(b"fLaC"):
            return "flac"
        if raw.startswith(b"ADIF"):
            return "aac"
        if raw.startswith(b"ID3"):
            return "mp3"
        if raw[0] == 0xFF and raw[1] & 0xE0 == 0xE0:
            # ADTS/AAC (12-bit sync) vs MPEG (11-bit sync): ADTS layer bits
            # are always 00, which is a reserved (invalid) MPEG layer, so a
            # 00 layer with a 12-bit sync unambiguously means AAC.
            if raw[1] & 0xF0 == 0xF0 and raw[1] & 0x06 == 0x00:
                return "aac"
            return "mp3"
        if _ftyp_brand(raw) is not None:
            # Any MP4-family brand (M4A/mp41/isom/...); the extension-less
            # sniff cannot tell audio-only files from video, so providers
            # receive them labeled m4a and decide.
            return "m4a"
        return None

    @staticmethod
    def _encode_input_audio(raw: bytes, audio_format: str) -> dict:
        encoded = base64.b64encode(raw).decode("utf-8")
        return {
            "type": "input_audio",
            "input_audio": {"data": encoded, "format": audio_format},
        }

    @staticmethod
    def process_messages(messages: list[dict]) -> None:
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for index, item in enumerate(content):
                if not isinstance(item, dict) or item.get("type") != "audio":
                    continue
                msg["content"][index] = AudioProcessor._convert_audio_item(item)

    @staticmethod
    def _convert_audio_item(item: dict) -> dict:
        if "audio_path" in item:
            return AudioProcessor._from_path(item["audio_path"])
        if "audio_url" in item:
            return AudioProcessor._from_url(item["audio_url"])
        if "audio_base64" in item:
            return AudioProcessor._from_base64(item["audio_base64"])
        raise ConfigurationError(
            "audio items need one of audio_path, audio_url or audio_base64"
        )

    @staticmethod
    def _from_path(path: Any) -> dict:
        if not isinstance(path, (str, Path)):
            raise ConfigurationError(
                f"audio_path must be a file path string or pathlib.Path, got {type(path).__name__}"
            )
        try:
            if not Path(path).is_file():
                raise AudioProcessingError("audio_path is not a regular file")
        except AudioProcessingError:
            raise
        except (OSError, ValueError) as e:
            detail = getattr(e, "strerror", None) or getattr(e, "errno", None) or "invalid path"
            raise AudioProcessingError(f"Cannot access audio ({detail})") from e
        try:
            with open(path, "rb") as handle:
                raw = handle.read(AudioProcessor._MAX_AUDIO_READ_BYTES + 1)
        except (OSError, ValueError) as e:
            detail = getattr(e, "strerror", None) or getattr(e, "errno", None) or "invalid path"
            raise AudioProcessingError(f"Failed to read audio ({detail})") from e
        if len(raw) > AudioProcessor._MAX_AUDIO_READ_BYTES:
            raise AudioProcessingError(
                "Audio is larger than the 25MB read limit; trim or compress it first"
            )
        audio_format = AudioProcessor._sniff_audio_format(raw)
        if audio_format is None:
            raise AudioProcessingError(
                "File is not a recognized audio file (wav/mp3/aiff/aac/ogg/flac/m4a)"
            )
        return AudioProcessor._encode_input_audio(raw, audio_format)

    @staticmethod
    def _from_base64(data: Any) -> dict:
        if not isinstance(data, str):
            raise AudioProcessingError("audio_base64 must be a base64 string")
        declared: Optional[str] = None
        body = data
        if data[:5].lower() == "data:":
            comma = data.find(",")
            if comma == -1:
                raise AudioProcessingError("audio_base64 data URL is malformed")
            header = data[5:comma].split(";")[0].strip().lower()
            if not header.startswith("audio/"):
                raise AudioProcessingError(
                    f"audio_base64 data URL is not audio (MIME: {header or 'missing'})"
                )
            declared = header.split("/", 1)[1]
            declared = AudioProcessor._AUDIO_MIME_ALIASES.get(declared, declared)
            if declared not in AudioProcessor._AUDIO_FORMATS:
                raise AudioProcessingError(
                    f"audio_base64 audio format {declared!r} is not supported"
                )
            body = data[comma + 1:]
        if len(body) > AudioProcessor._MAX_AUDIO_BASE64_CHARS:
            raise AudioProcessingError(
                "audio_base64 exceeds the size limit; trim or compress it first"
            )
        cleaned = _strip_b64_whitespace(body)
        if len(cleaned) > AudioProcessor._MAX_AUDIO_BASE64_CHARS:
            raise AudioProcessingError(
                "audio_base64 exceeds the size limit; trim or compress it first"
            )
        try:
            raw = base64.b64decode(cleaned, validate=True)
        except ValueError as e:
            raise AudioProcessingError(f"audio_base64 is not valid base64: {e}") from e
        if len(raw) > AudioProcessor._MAX_AUDIO_READ_BYTES:
            raise AudioProcessingError(
                "Audio is larger than the 25MB read limit; trim or compress it first"
            )
        audio_format = AudioProcessor._sniff_audio_format(raw)
        if audio_format is None:
            raise AudioProcessingError(
                "audio_base64 is not a recognized audio file (wav/mp3/aiff/aac/ogg/flac/m4a)"
            )
        return AudioProcessor._encode_input_audio(raw, audio_format)

    @staticmethod
    def _from_url(url: Any) -> dict:
        """Pass remote URLs through untouched; the provider decides."""
        url = _validate_http_url(url, what="audio_url")
        return {"type": "audio_url", "audio_url": {"url": url}}

# ============================================================================
# Video Processor
# ============================================================================

class VideoProcessor:
    """Processes video in messages for API consumption (input only).

    Local files and base64 become ``video_url`` parts with data URLs
    (mp4/mov/webm, 100MB cap); remote URLs pass through untouched (videos
    are large and URLs are the documented norm there). Optional top-level
    ``processing`` merges into the ``video_url`` part (provider extension,
    passed through). Bad content raises ``VideoProcessingError``.
    """

    _MAX_VIDEO_READ_BYTES: ClassVar[int] = 100 * 1024 * 1024
    """Read cap (OOM guard) for local video files."""
    _MAX_VIDEO_BASE64_CHARS: ClassVar[int] = 150 * 1024 * 1024
    """Inline base64 ceiling (~112MB decoded); above it, reject."""

    _VIDEO_MIME: ClassVar[dict[str, str]] = {
        "mp4": "video/mp4",
        "mov": "video/quicktime",
        "webm": "video/webm",
    }

    @staticmethod
    def _sniff_video_container(raw: bytes) -> Optional[str]:
        """Detect video container; returns mp4/mov/webm key or None."""
        if len(raw) < 12:
            return None
        # ftyp may sit behind padding atoms (wide/free/skip): only
        # box-aligned positions count (see _ftyp_brand).
        brand = _ftyp_brand(raw)
        if brand is not None:
            # QuickTime brand means .mov; everything else MP4-family is mp4.
            # mp4/mov stay interchangeable downstream (same wire shape).
            if brand == b"qt  ":
                return "mov"
            return "mp4"
        if raw.startswith(b"\x1a\x45\xdf\xa3"):
            return "webm"
        return None

    @staticmethod
    def _encode_video_url(url: str, url_data: Optional[dict] = None) -> dict:
        if isinstance(url_data, dict):
            return {"type": "video_url", "video_url": {**url_data, "url": url}}
        return {"type": "video_url", "video_url": {"url": url}}

    @staticmethod
    def process_messages(messages: list[dict]) -> None:
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for index, item in enumerate(content):
                if not isinstance(item, dict) or item.get("type") != "video":
                    continue
                msg["content"][index] = VideoProcessor._convert_video_item(item)

    @staticmethod
    def _convert_video_item(item: dict) -> dict:
        processing = item.get("processing")
        if processing is not None and (
            not isinstance(processing, str) or not processing.strip()
        ):
            raise ConfigurationError(
                "video processing must be a non-empty string or None"
            )
        if isinstance(processing, str):
            processing = processing.strip()
            _reject_confusable_controls(processing, what="video processing")
        if "video_path" in item:
            converted = VideoProcessor._from_path(item["video_path"])
        elif "video_url" in item:
            converted = VideoProcessor._from_url(item["video_url"])
        elif "video_base64" in item:
            converted = VideoProcessor._from_base64(item["video_base64"])
        else:
            raise ConfigurationError(
                "video items need one of video_path, video_url or video_base64"
            )
        if processing is not None and converted.get("type") == "video_url":
            url_data = converted.get("video_url")
            if isinstance(url_data, dict):
                # Explicit dict-level processing wins over the top-level knob.
                url_data.setdefault("processing", processing)
        return converted

    @staticmethod
    def _from_path(path: Any) -> dict:
        if not isinstance(path, (str, Path)):
            raise ConfigurationError(
                f"video_path must be a file path string or pathlib.Path, got {type(path).__name__}"
            )
        try:
            if not Path(path).is_file():
                raise VideoProcessingError("video_path is not a regular file")
        except VideoProcessingError:
            raise
        except (OSError, ValueError) as e:
            detail = getattr(e, "strerror", None) or getattr(e, "errno", None) or "invalid path"
            raise VideoProcessingError(f"Cannot access video ({detail})") from e
        extension = Path(path).suffix.lower().lstrip(".")
        if extension == "mkv":
            # MKV is EBML like WebM: sniff-verified as webm below, sent as webm.
            extension = "webm"
        if extension and extension not in VideoProcessor._VIDEO_MIME:
            raise VideoProcessingError(
                f"video container {extension or 'missing'!r} is not supported (mp4/mov/webm)"
            )
        try:
            with open(path, "rb") as handle:
                raw = handle.read(VideoProcessor._MAX_VIDEO_READ_BYTES + 1)
        except (OSError, ValueError) as e:
            detail = getattr(e, "strerror", None) or getattr(e, "errno", None) or "invalid path"
            raise VideoProcessingError(f"Failed to read video ({detail})") from e
        if len(raw) > VideoProcessor._MAX_VIDEO_READ_BYTES:
            raise VideoProcessingError(
                "Video is larger than the 100MB read limit; trim or compress it first"
            )
        sniffed = VideoProcessor._sniff_video_container(raw)
        if sniffed is None or (extension and (extension == "webm") != (sniffed == "webm")):
            raise VideoProcessingError(
                "File is not a recognized video file (mp4/mov/webm)"
            )
        mime = VideoProcessor._VIDEO_MIME[extension or sniffed]
        encoded = base64.b64encode(raw).decode("utf-8")
        return VideoProcessor._encode_video_url(f"data:{mime};base64,{encoded}")

    @staticmethod
    def _from_base64(data: Any) -> dict:
        if not isinstance(data, str):
            raise VideoProcessingError("video_base64 must be a base64 string")
        if data[:5].lower() == "data:":
            comma = data.find(",")
            if comma == -1:
                raise VideoProcessingError("video_base64 data URL is malformed")
            header = data[5:comma].split(";")[0].strip().lower()
            if not header.startswith("video/"):
                raise VideoProcessingError(
                    f"video_base64 data URL is not video (MIME: {header or 'missing'})"
                )
            if header not in VideoProcessor._VIDEO_MIME.values():
                raise VideoProcessingError(
                    f"video_base64 video format {header!r} is not supported (mp4/mov/webm)"
                )
            body = data[comma + 1:]
            if len(body) > VideoProcessor._MAX_VIDEO_BASE64_CHARS:
                raise VideoProcessingError(
                    "video_base64 exceeds the size limit; trim or compress it first"
                )
            cleaned = _strip_b64_whitespace(body)
            if len(cleaned) > VideoProcessor._MAX_VIDEO_BASE64_CHARS:
                raise VideoProcessingError(
                    "video_base64 exceeds the size limit; trim or compress it first"
                )
            try:
                raw = base64.b64decode(cleaned, validate=True)
            except ValueError as e:
                raise VideoProcessingError(
                    f"video_base64 is not valid base64: {e}"
                ) from e
            if len(raw) > VideoProcessor._MAX_VIDEO_READ_BYTES:
                raise VideoProcessingError(
                    "Video is larger than the 100MB read limit; trim or compress it first"
                )
            sniffed = VideoProcessor._sniff_video_container(raw)
            if sniffed is None:
                raise VideoProcessingError(
                    "video_base64 data URL bytes are not a recognized video "
                    "file (mp4/mov/webm)"
                )
            if (header == "video/webm") != (sniffed == "webm"):
                raise VideoProcessingError(
                    f"video_base64 data URL declares {header!r} but the bytes "
                    f"are {sniffed}"
                )
            return VideoProcessor._encode_video_url(f"data:{header};base64,{cleaned}")
        if len(data) > VideoProcessor._MAX_VIDEO_BASE64_CHARS:
            raise VideoProcessingError(
                "video_base64 exceeds the size limit; trim or compress it first"
            )
        cleaned = _strip_b64_whitespace(data)
        if len(cleaned) > VideoProcessor._MAX_VIDEO_BASE64_CHARS:
            raise VideoProcessingError(
                "video_base64 exceeds the size limit; trim or compress it first"
            )
        try:
            raw = base64.b64decode(cleaned, validate=True)
        except ValueError as e:
            raise VideoProcessingError(f"video_base64 is not valid base64: {e}") from e
        if len(raw) > VideoProcessor._MAX_VIDEO_READ_BYTES:
            raise VideoProcessingError(
                "Video is larger than the 100MB read limit; trim or compress it first"
            )
        sniffed = VideoProcessor._sniff_video_container(raw)
        if sniffed is None:
            raise VideoProcessingError(
                "video_base64 is not a recognized video file (mp4/mov/webm); "
                "declare the format explicitly with a data:video/... URL"
            )
        mime = VideoProcessor._VIDEO_MIME[sniffed]
        return VideoProcessor._encode_video_url(f"data:{mime};base64,{cleaned}")

    @staticmethod
    def _from_url(url_data: Union[str, dict]) -> dict:
        if isinstance(url_data, str):
            url_data = {"url": url_data}
        url = url_data.get("url") if isinstance(url_data, dict) else None
        url = _validate_http_url(url, what="video_url")
        if isinstance(url_data, dict) and "processing" in url_data:
            # Same strictness as the top-level knob; explicit null dropped.
            processing = url_data["processing"]
            if processing is not None and (
                not isinstance(processing, str) or not processing.strip()
            ):
                raise ConfigurationError(
                    "video processing must be a non-empty string or None"
                )
            url_data = dict(url_data)
            if processing is None:
                del url_data["processing"]
            else:
                url_data["processing"] = processing.strip()
                _reject_confusable_controls(
                    url_data["processing"], what="video processing"
                )
        return VideoProcessor._encode_video_url(url, url_data)

# ============================================================================
# File Processor
# ============================================================================

class FileProcessor:
    """Processes generic files in messages for API consumption (input only).

    Local files and base64 become ``file`` parts (filename + data URL);
    remote URLs pass through untouched (URL in, URL out); ``file_id``
    references provider-uploaded files. Optional ``detail`` rides along
    for the Responses API (dropped with a warning in chat mode).
    """

    _MAX_FILE_READ_BYTES: ClassVar[int] = 50 * 1024 * 1024
    """Read cap (OOM guard) for local files."""
    _MAX_FILE_BASE64_CHARS: ClassVar[int] = 75 * 1024 * 1024
    """Inline base64 ceiling (~56MB decoded); above it, reject."""

    _MIME_RE: ClassVar[re.Pattern] = re.compile(r"[\w.+-]+/[\w.+-]+")

    @staticmethod
    def _guess_mime(name: Optional[str]) -> str:
        if name:
            guessed = mimetypes.guess_type(name)[0]
            if guessed:
                return guessed
        return "application/octet-stream"

    @staticmethod
    def _encode_file_part(
        filename: str, mime: str, raw: bytes, *, detail: Optional[str] = None
    ) -> dict:
        encoded = base64.b64encode(raw).decode("utf-8")
        part: dict[str, Any] = {
            "filename": filename,
            "file_data": f"data:{mime};base64,{encoded}",
        }
        if detail is not None:
            # Responses-only knob (PDF detail); chat completions ignore it.
            part["detail"] = detail
        return {"type": "file", "file": part}

    @staticmethod
    def process_messages(messages: list[dict]) -> None:
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for index, item in enumerate(content):
                if not isinstance(item, dict) or item.get("type") != "file":
                    continue
                msg["content"][index] = FileProcessor._convert_file_item(item)

    @staticmethod
    def _convert_file_item(item: dict) -> dict:
        detail = _validate_detail(item.get("detail"), what="file detail")
        if "file" in item:
            # Already converted (converters share the "file" type): keep
            # builders idempotent instead of reprocessing wire parts.
            return item
        if "file_path" in item:
            return FileProcessor._from_path(
                item["file_path"],
                filename=item.get("filename"),
                mime_type=item.get("mime_type"),
                detail=detail,
            )
        if "file_url" in item:
            return FileProcessor._from_url(
                item["file_url"],
                filename=item.get("filename"),
                mime_type=item.get("mime_type"),
                detail=detail,
            )
        if "file_base64" in item:
            return FileProcessor._from_base64(
                item["file_base64"],
                filename=item.get("filename"),
                mime_type=item.get("mime_type"),
                detail=detail,
            )
        if "file_id" in item:
            if item.get("mime_type") is not None:
                logger.warning(
                    "file mime_type is ignored for file_id references "
                    "(there are no bytes to label)"
                )
            return FileProcessor._from_file_id(
                item["file_id"],
                filename=item.get("filename"),
                detail=detail,
            )
        raise ConfigurationError(
            "file items need one of file_path, file_url, file_base64 or file_id"
        )

    @staticmethod
    def _drop_detail_for_chat(messages: list[dict]) -> None:
        """Strip the Responses-only detail knob from chat file parts.

        Chat file parts have no detail field; dropping it silently would
        hide mode-dependent behavior, so warn once instead. Expects
        builder-owned copies (copy-on-write inside, never touches payloads).
        """
        dropped = False
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for index, part in enumerate(content):
                if not isinstance(part, dict) or part.get("type") != "file":
                    continue
                inner = part.get("file")
                # Copy-on-write: content items may be caller-owned
                # (_copy_messages_shallow shares payloads).
                if isinstance(inner, dict) and "detail" in inner:
                    new_inner = dict(inner)
                    new_inner.pop("detail", None)
                    content[index] = {**part, "file": new_inner}
                    dropped = True
        if dropped:
            _warn_detail_dropped_for_chat()

    @staticmethod
    def _safe_basename(value: Any) -> str:
        """Basename for provider-visible filenames; never raises."""
        if isinstance(value, Path):
            candidate: Any = value
        elif isinstance(value, str) and value:
            candidate = value
        else:
            return ""
        try:
            return Path(candidate).name
        except (ValueError, OSError):
            return ""

    @staticmethod
    def _resolve_name_and_mime(
        name_hint: Any, filename: Any, mime_type: Any
    ) -> tuple[str, str]:
        if filename is not None:
            if not isinstance(filename, str):
                raise ConfigurationError("file filename must be a string")
            if len(filename) > 1024:
                raise ConfigurationError("file filename exceeds the 1024-character limit")
            _reject_confusable_controls(filename, what="file filename")
        if mime_type is not None:
            if not isinstance(mime_type, str):
                raise ConfigurationError("file mime_type must be a string")
            if len(mime_type) > 256:
                raise ConfigurationError("file mime_type exceeds the 256-character limit")
            maintype = mime_type.split(";", 1)[0].strip().lower()
            if FileProcessor._MIME_RE.fullmatch(maintype) is None:
                # Caller error (bad knob), hence ConfigurationError — while a
                # malformed data: declaration raises FileProcessingError
                # (bad content). Both fail fast before any bytes move.
                raise ConfigurationError(
                    f"file mime_type is not a valid MIME type: {mime_type!r}"
                )
            # Store the normalized type, not the raw header: parameters like
            # "; charset=..." would otherwise corrupt the data: URL scheme.
            mime_type = maintype
        name = (
            FileProcessor._safe_basename(filename)
            or FileProcessor._safe_basename(name_hint)
            or "file"
        )
        mime = mime_type or FileProcessor._guess_mime(name if "." in name else None)
        return name, mime

    @staticmethod
    def _from_path(
        path: Any,
        *,
        filename: Any = None,
        mime_type: Any = None,
        detail: Optional[str] = None,
    ) -> dict:
        if not isinstance(path, (str, Path)):
            raise ConfigurationError(
                f"file_path must be a file path string or pathlib.Path, got {type(path).__name__}"
            )
        try:
            if not Path(path).is_file():
                raise FileProcessingError("file_path is not a regular file")
        except FileProcessingError:
            raise
        except (OSError, ValueError) as e:
            reason = getattr(e, "strerror", None) or getattr(e, "errno", None) or "invalid path"
            raise FileProcessingError(f"Cannot access file ({reason})") from e
        try:
            with open(path, "rb") as handle:
                raw = handle.read(FileProcessor._MAX_FILE_READ_BYTES + 1)
        except (OSError, ValueError) as e:
            reason = getattr(e, "strerror", None) or getattr(e, "errno", None) or "invalid path"
            raise FileProcessingError(f"Failed to read file ({reason})") from e
        if len(raw) > FileProcessor._MAX_FILE_READ_BYTES:
            raise FileProcessingError(
                "File is larger than the 50MB read limit"
            )
        if len(raw) == 0:
            raise FileProcessingError("file_path is empty")
        name, mime = FileProcessor._resolve_name_and_mime(path, filename, mime_type)
        return FileProcessor._encode_file_part(name, mime, raw, detail=detail)

    @staticmethod
    def _from_base64(
        data: Any,
        *,
        filename: Any = None,
        mime_type: Any = None,
        detail: Optional[str] = None,
    ) -> dict:
        if not isinstance(data, str):
            raise FileProcessingError("file_base64 must be a base64 string")
        declared: Optional[str] = None
        body = data
        if data[:5].lower() == "data:":
            comma = data.find(",")
            if comma == -1:
                raise FileProcessingError("file_base64 data URL is malformed")
            header = data[5:comma].split(";")[0].strip().lower()
            if "/" not in header:
                raise FileProcessingError(
                    f"file_base64 data URL has no MIME type (got {header or 'missing'!r})"
                )
            declared = header
            body = data[comma + 1:]
        if len(body) > FileProcessor._MAX_FILE_BASE64_CHARS:
            raise FileProcessingError("file_base64 exceeds the size limit")
        cleaned = _strip_b64_whitespace(body)
        if len(cleaned) > FileProcessor._MAX_FILE_BASE64_CHARS:
            raise FileProcessingError("file_base64 exceeds the size limit")
        try:
            raw = base64.b64decode(cleaned, validate=True)
        except ValueError as e:
            raise FileProcessingError(f"file_base64 is not valid base64: {e}") from e
        if len(raw) > FileProcessor._MAX_FILE_READ_BYTES:
            raise FileProcessingError("File is larger than the 50MB read limit")
        if len(raw) == 0:
            raise FileProcessingError("file_base64 is empty")
        name, mime = FileProcessor._resolve_name_and_mime(None, filename, mime_type)
        if mime_type is None and declared is not None:
            if len(declared) > 256:
                raise FileProcessingError(
                    "file_base64 data URL MIME exceeds the 256-character limit"
                )
            if FileProcessor._MIME_RE.fullmatch(declared) is None:
                raise FileProcessingError(
                    f"file_base64 data URL MIME is not valid: {declared!r}"
                )
            mime = declared
        return FileProcessor._encode_file_part(name, mime, raw, detail=detail)

    @staticmethod
    def _from_url(
        url: Any,
        *,
        filename: Any = None,
        mime_type: Any = None,
        detail: Optional[str] = None,
    ) -> dict:
        """Pass remote URLs through untouched; the provider decides.

        Unlike local files, URLs stay URLs: some providers fetch them,
        others only accept inline content — that choice is theirs.
        """
        url = _validate_http_url(url, what="file_url")
        # Strip query/fragment so they don't leak into the filename.
        name_hint = url.split("?", 1)[0].split("#", 1)[0]
        name, _ = FileProcessor._resolve_name_and_mime(name_hint, filename, mime_type)
        part: dict[str, Any] = {"filename": name, "file_data": url}
        if detail is not None:
            part["detail"] = detail
        return {"type": "file", "file": part}

    @staticmethod
    def _from_file_id(
        file_id: Any,
        *,
        filename: Any = None,
        detail: Optional[str] = None,
    ) -> dict:
        """Reference a file uploaded via the provider's Files API."""
        if not isinstance(file_id, str) or not file_id.strip():
            raise ConfigurationError("file_id must be a non-empty string")
        file_id = file_id.strip()
        _reject_confusable_controls(file_id, what="file_id")
        if len(file_id) > 512:
            raise ConfigurationError("file_id exceeds the 512-character limit")
        if filename is not None:
            if not isinstance(filename, str):
                raise ConfigurationError("file filename must be a string")
            if len(filename) > 1024:
                raise ConfigurationError(
                    "file filename exceeds the 1024-character limit"
                )
            _reject_confusable_controls(filename, what="file filename")
        part: dict[str, Any] = {"file_id": file_id}
        name = FileProcessor._safe_basename(filename)
        if name:
            part["filename"] = name
        if detail is not None:
            part["detail"] = detail
        return {"type": "file", "file": part}

class EventBuilder:
    """Builds standardized stream events.

    Internal helper (no semver guarantee for direct use); the stable
    surface is the event dicts yielded by ``stream_response``.
    """

    @staticmethod
    def _build(event_type: EventType, content: Any) -> StreamEvent:
        return {
            "type": event_type.value,
            "content": content,
        }

    @staticmethod
    def answer(content: Any) -> StreamEvent:
        return EventBuilder._build(EventType.ANSWER, content)

    @staticmethod
    def reasoning(content: str) -> StreamEvent:
        return EventBuilder._build(EventType.REASONING, content)

    @staticmethod
    def refusal(content: str) -> StreamEvent:
        return EventBuilder._build(EventType.REFUSAL, content)

    @staticmethod
    def tool_call(content: ToolCall) -> StreamEvent:
        return EventBuilder._build(EventType.TOOL_CALL, content)

    @staticmethod
    def tool_call_part(content: dict[str, str]) -> StreamEvent:
        return EventBuilder._build(EventType.TOOL_CALL_PART, content)

    @staticmethod
    def verbose(content: VerboseInfo) -> StreamEvent:
        return EventBuilder._build(EventType.VERBOSE, content)

    @staticmethod
    def final(content: FinalResponse) -> StreamEvent:
        return EventBuilder._build(EventType.FINAL, content)

    @staticmethod
    def done() -> StreamEvent:
        return EventBuilder._build(EventType.DONE, None)

# ============================================================================
# ToolCallStreamHandler
# ============================================================================

Key = tuple[str, Any]


class ToolCallStreamHandler:
    """Accumulates streamed tool-call deltas and emits incremental events.

    ``tool_call_part`` events stream argument deltas as they arrive. Complete
    ``tool_call`` events are emitted from ``finalize()`` (stream end) and,
    conservatively, when the server switches to another call index while the
    accumulated arguments already parse as complete JSON — never merely
    because a chunk carried no tool calls.
    """

    def __init__(self, event_builder: EventBuilder, tools_dict: Optional[dict[str, Callable]] = None):
        self._event_builder = event_builder
        self._tools_dict = tools_dict or {}
        self._pending: dict[Key, dict[str, Any]] = {}
        self._active_key: Optional[Key] = None
        self._emitted_keys: set[Key] = set()
        self._id_to_key: dict[str, Key] = {}
        self._index_redirect: dict[Any, Key] = {}
        self._next_fallback_index = 0

    def _resolve_key(self, tc: Any) -> tuple[Key, bool]:
        """Resolve a stable key for a streamed tool call.

        Some OpenAI-compatible servers omit ``index``; an id is then the only
        reliable identity. Servers that send a constant ``index`` for every
        call are detected via an id change on the same index.
        """
        idx = getattr(tc, "index", None)
        tid = getattr(tc, "id", None)
        # Normalize provider-controlled indices before keying: bools/floats
        # never silently become ints, integer-like strings unify with ints
        # so "0" and 0 don't split one call into two.
        if isinstance(idx, (bool, float)):
            idx = None
        elif isinstance(idx, str):
            with contextlib.suppress(TypeError, ValueError, OverflowError):
                idx = int(idx)
        # Provider-controlled garbage must never crash keying: unhashable
        # values (list/dict) fall back to id-matching. Non-numeric strings
        # keep their stable key on purpose (quirky but working providers).
        if idx is not None:
            try:
                hash(idx)
            except TypeError:
                idx = None
        if tid is not None:
            try:
                hash(tid)
            except TypeError:
                tid = None

        if idx is not None:
            # After an id change redirected a constant index to a new call,
            # later id-less deltas with the same index must follow the
            # redirect instead of reopening the first call.
            key: Key = self._index_redirect.get(idx, ("index", idx))
            pending = self._pending.get(key)
            if pending is not None and tid and pending["id"] and tid != pending["id"]:
                # Constant index for parallel calls: open a new key by id.
                key = ("id", tid)
                self._id_to_key[tid] = key
                self._index_redirect[idx] = key
                return key, False
            if tid:
                self._id_to_key[tid] = key
            return key, True

        if tid:
            if tid in self._id_to_key:
                return self._id_to_key[tid], False
            if (
                self._active_key is not None
                and self._active_key in self._pending
                and not self._pending[self._active_key]["id"]
                and self._active_key not in self._emitted_keys
            ):
                self._id_to_key[tid] = self._active_key
                return self._active_key, False
            key = ("id", tid)
            self._id_to_key[tid] = key
            return key, False

        if self._active_key is not None and self._active_key in self._pending:
            active = self._pending[self._active_key]
            function = getattr(tc, "function", None)
            incoming_name = getattr(function, "name", None) if function is not None else None
            if incoming_name and active["name"] and active["arguments"]:
                try:
                    json.loads(active["arguments"])
                except (ValueError, TypeError, RecursionError):
                    pass
                else:
                    key = ("fallback", self._next_fallback_index)
                    self._next_fallback_index += 1
                    return key, False
            return self._active_key, False

        key = ("fallback", self._next_fallback_index)
        self._next_fallback_index += 1
        return key, False

    def process_chunk(self, tool_calls: Optional[list[Any]]) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        if not tool_calls:
            return events

        for tc in tool_calls:
            key, has_explicit_index = self._resolve_key(tc)

            if (
                self._active_key is not None
                and key != self._active_key
                and self._pending.get(self._active_key, {}).get("emit_on_switch", False)
                and self._active_key not in self._emitted_keys
                and self._arguments_complete(self._active_key)
                and (event := self._emit_complete_tool_call(self._active_key))
            ):
                events.append(event)

            self._active_key = key

            if key not in self._pending:
                self._pending[key] = {
                    "id": "",
                    "name": "",
                    "arguments": "",
                    "buffer": "",
                    "fallback_id": f"call_{uuid.uuid4().hex[:9]}",
                    "emit_on_switch": has_explicit_index,
                }

            pending = self._pending[key]

            if tid := getattr(tc, "id", None):
                pending["id"] = tid
                self._id_to_key[tid] = key

            function = getattr(tc, "function", None)
            if function is not None:
                function_name = getattr(function, "name", None)
                if function_name:
                    pending["name"] = function_name
                chunk = getattr(function, "arguments", None)
                if chunk:
                    if isinstance(chunk, dict):
                        chunk = json.dumps(chunk)
                    elif not isinstance(chunk, str):
                        chunk = str(chunk)
                    # Only treat chunks as cumulative argument resends once
                    # the buffer is long enough that a coincidental prefix
                    # match is implausible.
                    if (
                        len(pending["arguments"]) >= 8
                        and chunk.startswith(pending["arguments"])
                    ):
                        # Server resends cumulative arguments instead of deltas.
                        pending["buffer"] = chunk[len(pending["arguments"]):]
                        pending["arguments"] = chunk
                    else:
                        pending["arguments"] += chunk
                        pending["buffer"] += chunk

            if pending["name"] and pending["buffer"]:
                events.append(
                    self._event_builder.tool_call_part(
                        content={
                            "id": pending["id"] or pending["fallback_id"],
                            "name": pending["name"],
                            "args_delta": pending["buffer"],
                        }
                    )
                )
                pending["buffer"] = ""

        return events

    def _arguments_complete(self, key: Key) -> bool:
        pending = self._pending.get(key)
        if not pending or not pending["arguments"]:
            return False
        try:
            json.loads(pending["arguments"])
        except (ValueError, TypeError, RecursionError):
            return False
        return True

    def _emit_complete_tool_call(self, key: Key) -> Optional[StreamEvent]:
        if key not in self._pending:
            return None
        pending = self._pending[key]
        name = pending["name"]
        if not name:
            return None

        try:
            args = json.loads(pending["arguments"] or "{}")
        except (ValueError, TypeError, RecursionError):
            args = {"_raw": pending["arguments"] or ""}
        if not isinstance(args, dict):
            args = {"_raw": json.dumps(args, ensure_ascii=False, default=str)}

        self._emitted_keys.add(key)
        return self._event_builder.tool_call({
            "id": pending["id"] or pending["fallback_id"],
            "name": name,
            "arguments": args,
            "callable": self._tools_dict.get(name) or None,
        })

    def finalize(self) -> list[ToolCall]:
        new_calls: list[ToolCall] = []
        for key in self._pending:
            if key in self._emitted_keys:
                continue
            event = self._emit_complete_tool_call(key)
            if event:
                content = event.get("content")
                if content is not None:
                    new_calls.append(content)
        self._active_key = None
        return new_calls

    def get_all_calls(self) -> list[ToolCall]:
        result: list[ToolCall] = []
        for key in self._pending:
            pending = self._pending[key]
            name = pending["name"]
            if not name:
                continue

            try:
                args = json.loads(pending["arguments"] or "{}")
            except (ValueError, TypeError, RecursionError):
                args = {"_raw": pending["arguments"] or ""}
            if not isinstance(args, dict):
                args = {"_raw": json.dumps(args, ensure_ascii=False, default=str)}

            result.append({
                "id": pending["id"] or pending["fallback_id"],
                "name": name,
                "arguments": args,
                "callable": self._tools_dict.get(name) or None,
            })
        return result

# ============================================================================
# Chat Stream State (shared by sync/async chat loops)
# ============================================================================

class _ChatStreamState:
    """Per-request, I/O-free state shared by the sync/async chat stream loops."""

    def __init__(
        self,
        *,
        parser: ReasoningParser,
        tool_handler: ToolCallStreamHandler,
        structured_output: bool,
        include_reasoning: bool,
        event_builder: EventBuilder,
        start_time: float,
    ):
        self._parser = parser
        self._tool_handler = tool_handler
        self._structured_output = structured_output
        self._include_reasoning = include_reasoning
        self._event_builder = event_builder
        self._start_time = start_time

        self.reasoning = ""
        self.answer = ""
        self.refusal = ""
        self.stop_reason: Optional[str] = None
        self.chunks = 0
        self.latency: Optional[float] = None
        self.prompt_tokens: Optional[int] = None
        self.completion_tokens: Optional[int] = None
        self.total_tokens: Optional[int] = None
        self._t_first: Optional[float] = None
        self._t_last: Optional[float] = None

    def handle_chunk(self, chunk: Any) -> list[StreamEvent]:
        events: list[StreamEvent] = []
        now = time.perf_counter()
        if self._t_first is None:
            self._t_first = now
            self.latency = now - self._start_time
        self._t_last = now

        usage = getattr(chunk, "usage", None)
        if usage is not None:
            for attr in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = getattr(usage, attr, None)
                if value is not None:
                    converted = LLM._safe_index(value, what=attr)
                    if converted is not None:
                        setattr(self, attr, converted)

        choices = getattr(chunk, "choices", None)
        if not choices:
            return events
        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason:
            self.stop_reason = _normalize_stop_reason(finish_reason)
        delta = getattr(choice, "delta", None)
        if delta is None:
            return events

        self.chunks += 1

        reasoning = _extract_reasoning(delta)
        if reasoning:
            self.reasoning += reasoning
            if self._include_reasoning:
                events.append(self._event_builder.reasoning(reasoning))

        refusal = getattr(delta, "refusal", None)
        if isinstance(refusal, str) and refusal:
            self.refusal += refusal
            events.append(self._event_builder.refusal(refusal))

        content = getattr(delta, "content", None)
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            # Some providers send content parts instead of a plain string;
            # only text parts contribute to the answer.
            parts = []
            for part in content:
                if isinstance(part, dict):
                    part_type, part_text = part.get("type"), part.get("text")
                else:
                    part_type = getattr(part, "type", None)
                    part_text = getattr(part, "text", None)
                if part_type in (None, "text") and isinstance(part_text, str) and part_text:
                    parts.append(part_text)
            text = "".join(parts)
        if text:
            reasoning_part, answer_part = self._parser.parse(text)
            if reasoning_part:
                self.reasoning += reasoning_part
                if self._include_reasoning:
                    events.append(self._event_builder.reasoning(reasoning_part))
            if answer_part:
                self.answer += answer_part
                if not self._structured_output:
                    events.append(self._event_builder.answer(answer_part))

        tool_calls = getattr(delta, "tool_calls", None)
        events.extend(self._tool_handler.process_chunk(tool_calls))
        return events

    def finish(self, *, elapsed: float, verbose: bool, final: bool) -> list[StreamEvent]:
        events: list[StreamEvent] = []

        reasoning_part, answer_part = self._parser.flush()
        if reasoning_part:
            self.reasoning += reasoning_part
            if self._include_reasoning:
                events.append(self._event_builder.reasoning(reasoning_part))
        if answer_part:
            self.answer += answer_part
            if not self._structured_output:
                events.append(self._event_builder.answer(answer_part))

        if self.refusal:
            self.stop_reason = "refusal"

        # Finalize tools first: a tools-only turn with output_format must not
        # fail JSON parsing of an (correctly) empty answer.
        pending_calls = self._tool_handler.finalize()

        answer: Any = self.answer
        if self._structured_output:
            answer = _parse_structured_output(
                self.answer,
                stop_reason=self.stop_reason,
                strict=True,
                refusal=self.refusal,
                has_tool_calls=bool(pending_calls or self._tool_handler.get_all_calls()),
            )
            events.append(self._event_builder.answer(answer))

        for tool_call in pending_calls:
            events.append(self._event_builder.tool_call(tool_call))

        tokens, total_tokens = _resolve_token_metrics(
            self.completion_tokens, self.prompt_tokens, self.total_tokens, self.chunks
        )
        tokens_per_second = _decode_tokens_per_second(
            self._t_first, self._t_last, elapsed, tokens
        )

        verbose_info: VerboseInfo = {
            "tokens": tokens,
            "chunks": self.chunks,
            "tokens_per_second": tokens_per_second,
            "latency": self.latency,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": total_tokens,
            "stop_reason": self.stop_reason,
        }

        if verbose:
            events.append(self._event_builder.verbose(verbose_info))

        if final:
            final_response: FinalResponse = {"answer": answer}
            if self._include_reasoning and self.reasoning:
                final_response["reasoning"] = self.reasoning
            if self.refusal:
                final_response["refusal"] = self.refusal
            if self._parser.is_inside_reasoning and self._include_reasoning:
                final_response["reasoning_unterminated"] = True
            all_completed_calls = self._tool_handler.get_all_calls()
            if all_completed_calls:
                final_response["tool_calls"] = all_completed_calls
            if self.stop_reason:
                final_response["stop_reason"] = self.stop_reason
            if verbose:
                final_response["verbose"] = dict(verbose_info)
            events.append(self._event_builder.final(final_response))

        events.append(self._event_builder.done())
        return events

# ============================================================================
# Main LLM Class
# ============================================================================

class LLM:
    """Universal API wrapper for LLM models with OpenAI-compatible API.

    One instance per event loop for async use: the async client is bound
    to the loop that first uses it, so sharing one ``LLM`` across
    ``asyncio.run()`` calls (or threads with separate loops) drops and
    re-creates the async client instead of reusing connections.

    Args:
        model: Model id sent to the provider.
        api_key: Bearer key (never logged).
        base_url: http(s) URL with host, no credentials; path-less URLs
            get ``/v1`` unless ``normalize_base_url=False``.
        reasoning_pattern: Custom ``CustomReasoningPattern`` for
            non-standard thinking tags.
        default_stop_sequences: Global non-empty stop strings.
        timeout: Request timeout in seconds (positive finite number;
            ``None`` disables it — not recommended).
        extra_body: Provider-specific fields merged last (bypasses SDK
            validation; may override request fields, except
            ``model``/``messages``/``input``/``stream`` which are rejected).
        use_responses_api: Use the Responses API shape instead of chat.
        default_headers: Extra headers (never logged).
        max_retries: Default retry count (``int >= 0``), per-call
            overridable.
        normalize_base_url: Append ``/v1`` to path-less base URLs.
        debug: Enable this SDK's debug logs (third-party HTTP chatter
            stays at WARNING to avoid leaking ``Authorization`` headers).
            Note: this sets the SDK logger level process-wide, so one
            ``debug=True`` instance affects all instances. The WARNING pin
            for httpx/openai/httpcore is applied on every construction
            (even ``debug=False``), so other libraries' debug logging for
            those packages is muted while this SDK is in use.
        max_image_side: Longest image side in px (``None`` disables;
            byte budget still applies).

    Example:
        >>> llm = LLM("qwen2.5-coder-7b")
        >>> response = llm.response([{"role": "user", "content": "Hello!"}])
        >>> print(response["answer"])
    """

    def __init__(
        self,
        model: str,
        api_key: str = DEFAULT_API_KEY,
        base_url: str = DEFAULT_BASE_URL,
        reasoning_pattern: Optional[CustomReasoningPattern] = None,
        default_stop_sequences: Optional[list[str]] = None,
        timeout: Optional[float] = DEFAULT_TIMEOUT,
        extra_body: Optional[dict[str, Any]] = None,
        use_responses_api: bool = False,
        default_headers: Optional[dict[str, str]] = None,
        max_retries: int = 3,
        normalize_base_url: bool = True,
        debug: bool = False,
        max_image_side: Optional[int] = 8192,
    ):
        _validate_connection_options(timeout, max_retries, extra_body)
        model = _validate_identity_options(
            model, api_key, default_headers,
            use_responses_api, normalize_base_url, debug,
        )
        if isinstance(api_key, str):
            api_key = api_key.strip()
        if default_stop_sequences is not None and (
            not isinstance(default_stop_sequences, (list, tuple))
            or len(default_stop_sequences) == 0
            or not all(isinstance(s, str) and s.strip() for s in default_stop_sequences)
        ):
            raise ConfigurationError(
                "default_stop_sequences must be a non-empty list of non-empty strings"
            )
        _validate_base_url(base_url)

        self._config = LLMConfig(
            model=model,
            api_key=api_key,
            base_url=base_url.rstrip("/") if base_url else base_url,
            reasoning_pattern=reasoning_pattern,
            default_stop_sequences=list(default_stop_sequences) if default_stop_sequences else None,
            timeout=timeout,
            extra_body=copy.deepcopy(extra_body),
            use_responses_api=use_responses_api,
            default_headers=copy.deepcopy(default_headers),
            max_retries=max_retries,
            normalize_base_url=normalize_base_url,
            debug=debug,
            max_image_side=max_image_side,
        )
        if debug:
            logger.setLevel(logging.DEBUG)
        # Third-party HTTP chatter stays at WARNING in both modes: at DEBUG
        # httpx would log headers including the Authorization key.
        for logger_name in ("httpx", "openai", "httpcore"):
            logging.getLogger(logger_name).setLevel(logging.WARNING)

        self._api_base = _resolve_api_base(
            self._config.base_url, normalize=normalize_base_url
        )

        self._client = self._new_client()
        # Async client is created lazily, per event loop (A-01/A-02); the
        # weakref detects loops that were closed and garbage collected even
        # when a new loop reuses the same memory address.
        self._async_client: Any = None
        self._async_client_loop_ref: Any = None
        # None = unknown, False = server rejected stream_options once.
        self._stream_options_supported: Optional[bool] = None

        self._schema_converter = SchemaConverter()
        self._tool_preparator = ToolPreparator(self._schema_converter)
        self._event_builder = EventBuilder()
        self._request_transformer = RequestTransformer(model, self._api_base)

        if use_responses_api and default_stop_sequences:
            logger.warning(
                "default_stop_sequences is ignored in Responses API mode "
                "(the Responses API has no stop parameter)"
            )

        logger.debug(
            "LLM initialized: model=%s, base_url=%s", model, _redact_url_credentials(self._api_base)
        )

    @classmethod
    def from_config(cls, config: LLMConfig) -> "LLM":
        """Create an LLM instance from an LLMConfig."""
        return cls(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            reasoning_pattern=config.reasoning_pattern,
            default_stop_sequences=config.default_stop_sequences,
            timeout=config.timeout,
            extra_body=config.extra_body,
            use_responses_api=config.use_responses_api,
            default_headers=config.default_headers,
            max_retries=config.max_retries,
            normalize_base_url=config.normalize_base_url,
            debug=config.debug,
            max_image_side=config.max_image_side,
        )

    def _new_client(
        self,
        max_retries: Optional[int] = None,
        *,
        async_client: bool = False,
    ) -> Any:
        """Create an independent client with its own HTTP transport."""
        if max_retries is not None and (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 0
        ):
            raise ConfigurationError("max_retries must be an int >= 0")
        client_kwargs: dict[str, Any] = {
            "base_url": self._api_base,
            "api_key": self._config.api_key,
            "timeout": self._config.timeout,
            "default_headers": self._config.default_headers,
            "max_retries": self._config.max_retries if max_retries is None else max_retries,
        }
        if async_client:
            return AsyncOpenAI(**client_kwargs)
        return OpenAI(**client_kwargs)

    def _get_async_client(self) -> Any:
        """Return the async client, creating it lazily for the running loop.

        Clients created here are tracked via a weakref to their event loop
        and retired when a different loop runs, so one LLM instance survives
        repeated asyncio.run() calls (httpx pools bind connections to their
        loop). Note: sharing one instance across threads that each run their
        own loop swaps the client back and forth — give each thread its own
        LLM instance in that case.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        loop_ref = weakref.ref(loop) if loop is not None else None

        client = self._async_client
        if client is not None and self._async_client_loop_ref is not None:
            old_ref = self._async_client_loop_ref
            same_loop = old_ref() is loop if loop is not None else old_ref() is None
            if not same_loop:
                self._retire_async_client(client, old_ref)
                client = None
        if client is None:
            client = self._new_client(async_client=True)
            self._async_client = client
            self._async_client_loop_ref = loop_ref
        return client

    @staticmethod
    def _retire_async_client(client: Any, loop_ref: Any) -> None:
        """Best-effort close of a client bound to another event loop."""
        old_loop = loop_ref() if loop_ref is not None else None
        if old_loop is not None and not old_loop.is_closed():
            def _close_on_old_loop() -> None:
                task = old_loop.create_task(_aclose_async_resource(client))
                with _pending_async_closes_lock:
                    _pending_async_closes.add(task)
                task.add_done_callback(_log_async_close_done)

            try:
                old_loop.call_soon_threadsafe(_close_on_old_loop)
                return
            except RuntimeError:
                pass
        # Loop already closed (typical asyncio.run() case): its connections
        # cannot be closed from here; drop the client and let GC reclaim it.
        logger.debug("Dropping async client bound to a closed event loop")

    def _client_for(self, max_retries: Optional[int]) -> Any:
        """Client for one request; a with_options copy shares the transport.

        with_options() copies share the main client's transport, so they must
        never be closed — closing one would close the shared pool. They are
        cheap and safely dropped.
        """
        if max_retries is None:
            return self._client
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ConfigurationError("max_retries must be an int >= 0")
        return self._client.with_options(max_retries=max_retries)

    def _async_client_for(self, max_retries: Optional[int]) -> Any:
        """Async client (lazily created per loop) with optional retry override."""
        client = self._get_async_client()
        if max_retries is None:
            return client
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ConfigurationError("max_retries must be an int >= 0")
        return client.with_options(max_retries=max_retries)

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def base_url(self) -> str:
        """The effective API base URL (after normalization)."""
        return self._api_base

    def list_models(
        self,
        fallback: Optional[Union[list[str], tuple[str, ...]]] = None,
        max_retries: Optional[int] = None,
        raise_on_error: bool = False,
    ) -> list[str]:
        """Return model IDs from the configured API, or fallback/[] on failure or empty results."""
        if fallback is not None and (
            not isinstance(fallback, (list, tuple))
            or not all(isinstance(m, str) and m.strip() for m in fallback)
        ):
            raise ConfigurationError(
                "fallback must be a list of non-empty model id strings or None"
            )
        try:
            client = self._client_for(max_retries)
            models = client.models.list()
            return sorted({model.id for model in models.data}) or list(fallback or [])
        except Exception as e:
            if raise_on_error:
                if isinstance(e, APIError):
                    raise self._wrap_request_error(e, "list_models") from e
                raise
            logger.warning("list_models failed, falling back: %s", _redact_url_credentials(str(e)))
            return list(fallback or [])

    async def async_list_models(
        self,
        fallback: Optional[Union[list[str], tuple[str, ...]]] = None,
        max_retries: Optional[int] = None,
        raise_on_error: bool = False,
    ) -> list[str]:
        """Async return model IDs from the configured API, or fallback/[] on failure or empty results."""
        if fallback is not None and (
            not isinstance(fallback, (list, tuple))
            or not all(isinstance(m, str) and m.strip() for m in fallback)
        ):
            raise ConfigurationError(
                "fallback must be a list of non-empty model id strings or None"
            )
        try:
            client = self._async_client_for(max_retries)
            models = await client.models.list()
            return sorted({model.id for model in models.data}) or list(fallback or [])
        except Exception as e:
            if raise_on_error:
                if isinstance(e, APIError):
                    raise self._wrap_request_error(e, "async_list_models") from e
                raise
            logger.warning(
                "async_list_models failed, falling back: %s",
                _redact_url_credentials(str(e)),
            )
            return list(fallback or [])

    # ========================================================================
    # Output Format Handling
    # ========================================================================

    def _prepare_output_format(
        self, output_format: Union[dict, type, None], *, strict: bool = True
    ) -> Optional[dict]:
        if output_format is None:
            return None
        if isinstance(output_format, dict):
            if output_format.get("type") == "json_schema":
                inner = output_format.get("json_schema")
                if isinstance(inner, dict):
                    schema = inner.get("schema")
                    if isinstance(schema, dict) and inner.get("strict", True):
                        SchemaConverter._validate_strict_schema(
                            schema, context=str(inner.get("name", ""))
                        )
            return output_format
        if isinstance(output_format, type):
            return self._schema_converter.convert_class_to_schema(
                output_format, strict=strict
            )
        raise ConfigurationError(
            f"output_format must be dict, type, or None, got {type(output_format).__name__}"
        )

    @staticmethod
    def _is_structured_output_format(output_format: Optional[dict]) -> bool:
        if not isinstance(output_format, dict):
            return False
        return output_format.get("type") in ("json_schema", "json_object")

    # ========================================================================
    # Request Builder (Chat Completions)
    # ========================================================================

    def _build_request(
        self,
        messages: list[dict],
        output_format: Optional[dict],
        tools: Optional[list],
        reasoning_effort: Optional[str],
        max_tokens: Optional[int],
        extra_body: Optional[dict],
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, list[str]]] = None,
        seed: Optional[int] = None,
        user: Optional[str] = None,
        tool_choice: Optional[Any] = None,
        store: Optional[bool] = None,
        reasoning_budget: Optional[int] = None,
    ) -> tuple[dict[str, Any], PreparedTools, bool]:
        """Build API request kwargs. Returns (kwargs, prepared_tools, structured_output)."""
        if extra_body is not None and not isinstance(extra_body, dict):
            raise ConfigurationError("extra_body must be a dict of provider-specific fields")
        _reject_reserved_extra_body_keys(extra_body)
        _validate_generation_options(
            temperature, top_p, max_tokens, seed, stop, user, store,
            reasoning_effort, reasoning_budget,
        )

        request_messages = _copy_messages_shallow(messages)
        prepared_tools = self._tool_preparator.prepare(tools)
        ImageProcessor.process_messages(
            request_messages, max_image_side=self._config.max_image_side
        )
        AudioProcessor.process_messages(request_messages)
        VideoProcessor.process_messages(request_messages)
        FileProcessor.process_messages(request_messages)
        FileProcessor._drop_detail_for_chat(request_messages)
        _drop_response_items_for_chat(request_messages)
        _drop_original_detail_for_chat(request_messages)
        structured_output = self._is_structured_output_format(output_format)

        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": request_messages,
            "stream": True,
        }
        if self._stream_options_supported is not False:
            kwargs["stream_options"] = {"include_usage": True}
        if prepared_tools.definitions:
            kwargs["tools"] = prepared_tools.definitions
        merged_extra_body = _deep_merge_dicts(self._config.extra_body, extra_body)
        if merged_extra_body:
            kwargs["extra_body"] = merged_extra_body
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if reasoning_budget is not None:
            kwargs["reasoning_budget"] = reasoning_budget
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if stop is not None:
            kwargs["stop"] = list(stop) if isinstance(stop, (list, tuple)) else [stop]
        elif self._config.default_stop_sequences:
            kwargs["stop"] = list(self._config.default_stop_sequences)
        if seed is not None:
            kwargs["seed"] = seed
        if user is not None:
            kwargs["user"] = user
        if store is not None:
            kwargs["store"] = store
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if structured_output:
            kwargs["response_format"] = output_format

        return self._request_transformer.transform(kwargs), prepared_tools, structured_output

    @staticmethod
    def _extract_reasoning(delta: Any) -> str:
        """Return streamed reasoning content from supported delta fields."""
        return _extract_reasoning(delta)

    # ========================================================================
    # Responses API – Request Builder & Helpers
    # ========================================================================

    @staticmethod
    def _translate_content_for_responses_api(
        content: Any, *, output: bool = False, max_image_side: Optional[int] = 8192
    ) -> Any:
        _validate_max_image_side(max_image_side)
        """Translate message content from Chat Completions to Responses format.

        Assistant content lists use ``output_text``; everyone else uses
        ``input_text``. Image ``detail`` is preserved. File parts map to
        ``input_file`` exactly (``file_data``/``file_url``/``file_id`` plus
        ``filename`` and ``detail``); ``audio_url``/``video_url`` have no
        OpenAI equivalent and raise ``ConfigurationError``.
        """
        if not isinstance(content, list):
            return content

        text_key = "output_text" if output else "input_text"
        translated: list[dict] = []
        for item in content:
            if not isinstance(item, dict):
                translated.append(item)
                continue
            item_type = item.get("type", "")
            if item_type == "text":
                translated.append({"type": text_key, "text": item.get("text", "")})
            elif item_type == "image_url":
                url_data = item.get("image_url", {})
                if isinstance(url_data, dict):
                    raw_url = url_data.get("url", "")
                    if not isinstance(raw_url, str) or not raw_url:
                        raise ConfigurationError(
                            "image_url must contain a non-empty url string"
                        )
                    validated = None
                    if url_data.get("detail") is not None:
                        validated = _validate_detail(
                            url_data["detail"], what="image detail"
                        )
                    if raw_url[:5].lower() == "data:":
                        # Same caps/transcode as the chat path, then reattach
                        # the Responses-only detail knob.
                        nested = LLM._translate_content_for_responses_api(
                            [{"type": "image_base64", "image_base64": raw_url}],
                            max_image_side=max_image_side,
                        )
                        image_item = dict(nested[0])
                        if validated is not None:
                            image_item["detail"] = validated
                    else:
                        image_item = {
                            "type": "input_image",
                            "image_url": _validate_http_url(raw_url, what="image_url"),
                        }
                        if validated is not None:
                            image_item["detail"] = validated
                else:
                    if not isinstance(url_data, str) or not url_data:
                        raise ConfigurationError(
                            "image_url must be a non-empty URL string or object"
                        )
                    if url_data[:5].lower() == "data:":
                        nested = LLM._translate_content_for_responses_api(
                            [{"type": "image_base64", "image_base64": url_data}],
                            max_image_side=max_image_side,
                        )
                        image_item = dict(nested[0])
                    else:
                        image_item = {
                            "type": "input_image",
                            "image_url": _validate_http_url(url_data, what="image_url"),
                        }
                translated.append(image_item)
            elif item_type == "image_base64":
                data = item.get("image_base64", "")
                if not isinstance(data, str):
                    raise ConfigurationError("image_base64 must be a base64 string")
                body = data
                if data[:5].lower() == "data:":
                    _, _, after = data.partition(",")
                    if not after:
                        raise ConfigurationError("image_base64 data URL is malformed")
                    body = after
                if len(body) > ImageProcessor._MAX_BASE64_HARD_CHARS:
                    # Pre-strip reject: whitespace-bloated payloads must not
                    # force a transient regex copy before failing.
                    raise ImageProcessingError(
                        "image_base64 exceeds the size limit"
                    )
                cleaned = _strip_b64_whitespace(body)
                if len(cleaned) > ImageProcessor._MAX_BASE64_HARD_CHARS:
                    raise ImageProcessingError(
                        "image_base64 exceeds the size limit"
                    )
                # Same caps/transcode as the chat path: no uncapped payloads
                # or oversized dimensions reach the wire from either mode.
                try:
                    transcoded = ImageProcessor._transcode_base64_if_needed(
                        cleaned, max_image_side=max_image_side
                    )
                except (ImageProcessingError, ConfigurationError):
                    raise
                except Exception as e:
                    raise ImageProcessingError(
                        "image_base64 is not a readable image"
                    ) from e
                if transcoded is not None:
                    url = transcoded["image_url"]["url"]
                elif data[:5].lower() == "data:":
                    url = ImageProcessor._normalize_data_url_mime(data, cleaned)
                else:
                    mime = ImageProcessor._sniff_image_mime(cleaned)
                    url = f"data:{mime};base64,{cleaned}"
                translated_item = {"type": "input_image", "image_url": url}
                base_detail = _validate_detail(item.get("detail"), what="image detail")
                if base_detail is not None:
                    translated_item["detail"] = base_detail
                translated.append(translated_item)
            elif item_type == "input_audio":
                audio_data = item.get("input_audio", {})
                if not isinstance(audio_data, dict):
                    raise ConfigurationError(
                        "input_audio needs an 'input_audio' object"
                    )
                audio_format = audio_data.get("format")
                if audio_format not in AudioProcessor._AUDIO_FORMATS:
                    raise ConfigurationError(
                        "input_audio format must be one of "
                        f"{sorted(AudioProcessor._AUDIO_FORMATS)}"
                    )
                payload = audio_data.get("data", "")
                if not isinstance(payload, str) or not payload:
                    raise ConfigurationError(
                        "input_audio needs non-empty base64 'data'"
                    )
                if len(payload) > AudioProcessor._MAX_AUDIO_BASE64_CHARS:
                    raise ConfigurationError(
                        "input_audio data exceeds the size limit"
                    )
                cleaned_audio = _strip_b64_whitespace(payload)
                if len(cleaned_audio) > AudioProcessor._MAX_AUDIO_BASE64_CHARS:
                    raise ConfigurationError(
                        "input_audio data exceeds the size limit"
                    )
                try:
                    decoded_audio = base64.b64decode(cleaned_audio, validate=True)
                except ValueError as e:
                    raise ConfigurationError(
                        f"input_audio data is not valid base64: {e}"
                    ) from e
                if len(decoded_audio) > AudioProcessor._MAX_AUDIO_READ_BYTES:
                    raise ConfigurationError(
                        "input_audio data exceeds the 25MB limit"
                    )
                sniffed_audio = AudioProcessor._sniff_audio_format(decoded_audio)
                if sniffed_audio is None:
                    raise ConfigurationError(
                        "input_audio data is not a recognized audio file "
                        "(wav/mp3/aiff/aac/ogg/flac/m4a)"
                    )
                if sniffed_audio != audio_format:
                    raise ConfigurationError(
                        f"input_audio declares {audio_format!r} but the bytes "
                        f"are {sniffed_audio}"
                    )
                translated.append(item)
            elif item_type == "audio_url":
                # The Responses API has input_audio (base64 only) and no URL
                # audio input — fail fast instead of a provider 400.
                raise ConfigurationError(
                    "audio URLs are not supported by the Responses API "
                    "(it accepts base64 input_audio only); use chat completions"
                )
            elif item_type == "video_url":
                raise ConfigurationError(
                    "video is not supported by the Responses API; use chat completions"
                )
            elif item_type == "file":
                file_data = item.get("file")
                if not isinstance(file_data, dict):
                    raise ConfigurationError(
                        "file items need a 'file' object with filename and file_data"
                    )
                file_id_value = file_data.get("file_id")
                file_data_value = file_data.get("file_data", "")
                detail_value = file_data.get("detail")
                filename_value = file_data.get("filename", "file")
                if filename_value is None:
                    filename_value = "file"
                if filename_value is not None and (
                    not isinstance(filename_value, str)
                    or len(filename_value) > 1024
                ):
                    raise ConfigurationError(
                        "file filename must be a string of at most 1024 characters"
                    )
                if not filename_value:
                    filename_value = "file"
                _reject_confusable_controls(filename_value, what="file filename")
                file_item: dict[str, Any] = {
                    "type": "input_file",
                    "filename": filename_value,
                }
                if file_id_value is not None:
                    if not isinstance(file_id_value, str) or not file_id_value.strip():
                        raise ConfigurationError(
                            "file file_id must be a non-empty string"
                        )
                    file_id_value = file_id_value.strip()
                    _reject_confusable_controls(file_id_value, what="file file_id")
                    if len(file_id_value) > 512:
                        raise ConfigurationError(
                            "file file_id exceeds the 512-character limit"
                        )
                    file_item["file_id"] = file_id_value
                elif isinstance(file_data_value, str) and file_data_value.lower().startswith(
                    ("http://", "https://")
                ):
                    # input_file accepts file_data, file_id or file_url —
                    # remote URLs map to file_url, never into file_data.
                    file_item["file_url"] = _validate_http_url(
                        file_data_value, what="file file_url"
                    )
                elif isinstance(file_data_value, str) and file_data_value:
                    body = file_data_value
                    if body[:5].lower() == "data:":
                        comma_at = body.find(",")
                        header = body[5:comma_at].split(";")[0].strip().lower()
                        if "/" not in header:
                            raise ConfigurationError(
                                "file file_data data URL has no MIME type"
                            )
                        if len(header) > 256:
                            raise ConfigurationError(
                                "file file_data data URL MIME exceeds the "
                                "256-character limit"
                            )
                        if FileProcessor._MIME_RE.fullmatch(header) is None:
                            raise ConfigurationError(
                                "file file_data data URL MIME is not valid: "
                                f"{header!r}"
                            )
                        after = body[comma_at + 1:]
                        if not after:
                            raise ConfigurationError(
                                "file file_data data URL is malformed"
                            )
                        body = after
                    if len(body) > FileProcessor._MAX_FILE_BASE64_CHARS:
                        raise ConfigurationError(
                            "file file_data exceeds the size limit"
                        )
                    try:
                        decoded_file = base64.b64decode(
                            _strip_b64_whitespace(body), validate=True
                        )
                    except ValueError as e:
                        raise ConfigurationError(
                            f"file file_data is not valid base64: {e}"
                        ) from e
                    if len(decoded_file) > FileProcessor._MAX_FILE_READ_BYTES:
                        raise ConfigurationError(
                            "file file_data exceeds the 50MB limit"
                        )
                    if len(decoded_file) == 0:
                        raise ConfigurationError("file file_data is empty")
                    file_item["file_data"] = file_data_value
                else:
                    raise ConfigurationError(
                        "file items need one of file_id, file_data or a file_url"
                    )
                if detail_value is not None:
                    file_item["detail"] = _validate_detail(
                        detail_value, what="file detail"
                    )
                translated.append(file_item)
            elif item_type in ("image", "audio", "video", "audio_base64", "video_base64", "file_base64"):
                # Pre-normalization shapes must never reach the wire: the
                # builders run the media processors first, so these only
                # arrive via direct calls — fail fast, not provider 400.
                raise ConfigurationError(
                    f"Unprocessed {item_type!r} item in Responses input; "
                    "pass messages through response()/stream_response()"
                )
            else:
                translated.append(item)
        return translated

    def _to_responses_input(self, messages: list[dict]) -> list[dict]:
        """Translate a Chat Completions message list into Responses API input items.

        ``role: tool`` messages become ``function_call_output`` items and
        assistant ``tool_calls`` become ``function_call`` items — without this
        translation the second turn of any tool loop fails with a 400.
        """
        items: list[dict] = []
        for message in messages:
            if not isinstance(message, dict):
                raise ConfigurationError(
                    f"Each message must be a dict, got {type(message).__name__}"
                )
            role = message.get("role")

            if role == "tool":
                output = message.get("content")
                if not isinstance(output, str):
                    output = json.dumps(output, ensure_ascii=False, default=str)
                items.append({
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id", ""),
                    "output": output,
                })
                continue

            if role == "assistant":
                # Reasoning items must be replayed before their function_call
                # items on reasoning models (gpt-5/o-series) or the next
                # request fails with "provided without its required
                # reasoning item".
                for response_item in message.get("response_items") or []:
                    if isinstance(response_item, dict):
                        items.append(response_item)
                content = message.get("content")
                if content:
                    items.append({
                        "role": "assistant",
                        "content": self._translate_content_for_responses_api(
                            content, output=True,
                            max_image_side=self._config.max_image_side,
                        ),
                    })
                for tool_call in message.get("tool_calls") or []:
                    function = tool_call.get("function") or {}
                    arguments = function.get("arguments", "{}")
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments, ensure_ascii=False, default=str)
                    items.append({
                        "type": "function_call",
                        "call_id": tool_call.get("id", ""),
                        "name": function.get("name", ""),
                        "arguments": arguments,
                    })
                continue

            items.append({
                "role": role or "user",
                "content": self._translate_content_for_responses_api(
                    message.get("content"),
                    max_image_side=self._config.max_image_side,
                ),
            })
        return items

    @staticmethod
    def _convert_output_format_for_responses_api(output_format: dict) -> dict:
        """Convert Chat Completions response_format to Responses API text.format."""
        if output_format.get("type") == "json_schema":
            json_schema = output_format.get("json_schema", {})
            return {
                "type": "json_schema",
                "name": json_schema.get("name", "response"),
                "strict": json_schema.get("strict", True),
                "schema": json_schema.get("schema", {}),
            }
        return output_format

    @staticmethod
    def _convert_tools_for_responses_api(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Chat Completions function tools to Responses API function tools."""
        converted: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") != "function" or "function" not in tool:
                converted.append(tool)
                continue

            function = tool.get("function") or {}
            parameters = copy.deepcopy(function.get("parameters") or {})
            properties = parameters.get("properties")
            if isinstance(properties, dict):
                parameters.setdefault("additionalProperties", False)
            converted_item: dict[str, Any] = {
                "type": "function",
                "name": function.get("name", ""),
                "description": function.get("description") or "",
                "parameters": parameters,
            }
            # Only forward an explicit strict flag: injecting strict: true for
            # non-strict-conformant schemas turns every optional tool parameter
            # into a 400 (the chat path never sets it either).
            if "strict" in function:
                converted_item["strict"] = function["strict"]
            converted.append(converted_item)
        return converted

    @staticmethod
    def _convert_tool_choice_for_responses(tool_choice: Any) -> Any:
        """Chat tool_choice -> Responses tool_choice.

        The Responses API accepts only the strings 'none' | 'auto' |
        'required'; forcing a specific function requires the object form.
        """
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            function = tool_choice.get("function")
            if isinstance(function, dict) and function.get("name"):
                return {"type": "function", "name": function["name"]}
        return tool_choice

    @staticmethod
    def _safe_index(value: Any, *, what: str = "output_index") -> Optional[int]:
        """Strict int-or-None for provider-controlled numeric fields; never raises.

        Only ints (no bool) and integer strings are accepted; floats, bools
        and anything else are ignored so counts are never silently truncated
        (True -> 1) or rounded (3.7 -> 3).
        """
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value >= 0 else None
        if not isinstance(value, str):
            logger.debug(
                "Ignoring non-numeric %s: %s", what,
                _redact_url_credentials(str(value))[:200],
            )
            return None
        try:
            parsed_value = int(value)
        except (TypeError, ValueError, OverflowError):
            logger.debug(
                "Ignoring non-numeric %s: %s", what,
                _redact_url_credentials(str(value))[:200],
            )
            return None
        return parsed_value if parsed_value >= 0 else None

    @staticmethod
    def _safe_text(value: Any) -> str:
        """Provider-controlled text deltas; non-strings are ignored, never stringified."""
        return value if isinstance(value, str) else ""

    @staticmethod
    def _read_usage(usage: Any) -> dict[str, Optional[int]]:
        if not usage:
            return {}

        def first_present(*names: str) -> Optional[int]:
            for name in names:
                value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
                if value is not None:
                    converted = LLM._safe_index(value, what=name)
                    if converted is not None:
                        return converted
            return None

        return {
            "prompt_tokens": first_present("input_tokens", "prompt_tokens"),
            "completion_tokens": first_present("output_tokens", "completion_tokens"),
            "total_tokens": first_present("total_tokens"),
        }

    @staticmethod
    def _new_responses_state() -> dict[str, Any]:
        return {
            "reasoning": "",
            "answer": "",
            "refusal": "",
            "tokens": 0,
            "api_usage": {},
            "reasoning_parser": None,
            "pending_tool_items": {},
            "pending_tool_indexes": {},
            "completed_tool_calls": [],
            "completed_call_ids": set(),
            "completed_item_ids": set(),
            "response_items": [],
            "stop_reason": None,
            "t_first": None,
            "t_last": None,
        }

    @staticmethod
    def _dump_response_item(item: Any) -> Optional[dict[str, Any]]:
        """Serialize a Responses output item verbatim for later replay."""
        model_dump = getattr(item, "model_dump", None)
        if callable(model_dump):
            try:
                dumped = model_dump(exclude_none=True, exclude_unset=False)
                return dumped if isinstance(dumped, dict) and dumped.get("type") else None
            except Exception as e:
                logger.debug("Failed to dump Responses output item: %r", e)
        item_type = getattr(item, "type", "")
        if item_type == "reasoning":
            return {
                "type": "reasoning",
                "id": getattr(item, "id", ""),
                "summary": getattr(item, "summary", None) or [],
                **({"encrypted_content": item.encrypted_content} if getattr(item, "encrypted_content", None) else {}),
            }
        return None

    def _handle_responses_event(
        self,
        event: Any,
        state: dict[str, Any],
        structured_output: bool,
        include_reasoning: bool,
        tools_dict: dict[str, Callable],
    ) -> list[StreamEvent]:
        emitted: list[StreamEvent] = []
        event_type = getattr(event, "type", "")

        if event_type in ("response.completed", "response.incomplete"):
            response = getattr(event, "response", None)
            state["api_usage"] = self._read_usage(getattr(response, "usage", None))
            state["stop_reason"] = self._derive_responses_stop_reason(response, state)
            return emitted

        if event_type in ("response.failed", "error"):
            response = getattr(event, "response", None)
            error = getattr(response, "error", None) or event
            message = getattr(error, "message", None) or str(error) or "unknown error"
            code = getattr(error, "code", None)
            raise ModelRequestError(
                f"Responses API request failed: {_redact_url_credentials(message)}",
                status_code=getattr(error, "status_code", None)
                or getattr(response, "status_code", None),
                body=_redact_json_urls({"code": code}) if code else None,
                request_id=getattr(error, "request_id", None)
                or getattr(response, "request_id", None),
            )

        if event_type == "response.output_text.delta":
            chunk = self._safe_text(getattr(event, "delta", ""))
            if chunk:
                state["tokens"] += 1
                parser = state["reasoning_parser"]
                if parser is not None:
                    reasoning_part, answer_part = parser.parse(chunk)
                else:
                    reasoning_part, answer_part = "", chunk
                if reasoning_part:
                    state["reasoning"] += reasoning_part
                    if include_reasoning:
                        emitted.append(self._event_builder.reasoning(reasoning_part))
                if answer_part:
                    state["answer"] += answer_part
                    if not structured_output:
                        emitted.append(self._event_builder.answer(answer_part))
            return emitted

        if event_type in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            chunk = self._safe_text(getattr(event, "delta", ""))
            if chunk:
                state["tokens"] += 1
                state["reasoning"] += chunk
                if include_reasoning:
                    emitted.append(self._event_builder.reasoning(chunk))
            return emitted

        if event_type == "response.refusal.delta":
            chunk = self._safe_text(getattr(event, "delta", ""))
            if chunk:
                state["refusal"] += chunk
                emitted.append(self._event_builder.refusal(chunk))
            return emitted

        if event_type == "response.output_item.added":
            item = getattr(event, "item", None)
            if item and getattr(item, "type", "") == "function_call":
                output_index = getattr(event, "output_index", None)
                pending = {
                    "call_id": getattr(item, "call_id", None) or getattr(item, "id", ""),
                    "name": getattr(item, "name", "") or "",
                    "arguments": "",
                }
                state["pending_tool_items"][getattr(item, "id", "")] = pending
                index = self._safe_index(output_index)
                if index is not None:
                    state["pending_tool_indexes"][index] = pending
            return emitted

        if event_type == "response.output_item.done":
            # Authoritative completion: emits the call when the server sent
            # no function_call_arguments.done event.
            item = getattr(event, "item", None)
            if item and getattr(item, "type", "") == "function_call":
                item_id = getattr(item, "id", "") or ""
                call_id = getattr(item, "call_id", None) or item_id
                already_completed = (
                    (call_id and call_id in state["completed_call_ids"])
                    or (item_id and item_id in state["completed_item_ids"])
                )
                if already_completed:
                    state["completed_item_ids"].add(item_id)
                    return emitted
                pending = state["pending_tool_items"].get(item_id) or {}
                name = getattr(item, "name", "") or pending.get("name", "")
                arguments = getattr(item, "arguments", None)
                if arguments is None or arguments == "":
                    arguments = pending.get("arguments", "") or "{}"
                if name:
                    tool_call = self._complete_responses_tool_call(
                        call_id, name, arguments, tools_dict
                    )
                    state["completed_tool_calls"].append(tool_call)
                    state["completed_call_ids"].add(call_id)
                    state["completed_item_ids"].add(item_id)
                    emitted.append(self._event_builder.tool_call(tool_call))
                state["pending_tool_items"].pop(item_id, None)
                output_index = getattr(event, "output_index", None)
                index = self._safe_index(output_index)
                if index is not None:
                    state["pending_tool_indexes"].pop(index, None)
                return emitted

            if item and getattr(item, "type", "") == "reasoning":
                # Reasoning items must be replayed with their function_call
                # items on the next turn (reasoning models require it).
                dumped = self._dump_response_item(item)
                if dumped is not None:
                    state["response_items"].append(dumped)
            return emitted

        if event_type == "response.function_call_arguments.delta":
            item_id = getattr(event, "item_id", None)
            output_index = getattr(event, "output_index", None)
            index = self._safe_index(output_index)
            delta = self._safe_text(getattr(event, "delta", ""))

            if item_id and item_id in state["pending_tool_items"]:
                state["pending_tool_items"][item_id].setdefault("arguments", "")
                state["pending_tool_items"][item_id]["arguments"] += delta
            elif index is not None and index in state["pending_tool_indexes"]:
                indexed = state["pending_tool_indexes"][index]
                indexed.setdefault("arguments", "")
                indexed["arguments"] += delta

            pending = None
            if item_id and item_id in state["pending_tool_items"]:
                pending = state["pending_tool_items"][item_id]
            elif index is not None:
                pending = state["pending_tool_indexes"].get(index)

            if pending and pending.get("call_id") and pending.get("name") and delta:
                state["tokens"] += 1
                emitted.append(
                    self._event_builder.tool_call_part(
                        content={
                            "id": pending["call_id"],
                            "name": pending["name"],
                            "args_delta": delta,
                        }
                    )
                )
            return emitted

        if event_type == "response.function_call_arguments.done":
            item_id = getattr(event, "item_id", None)
            output_index = getattr(event, "output_index", None)

            args_str = getattr(event, "arguments", None)

            pending = state["pending_tool_items"].pop(item_id, {}) if item_id else {}
            index = self._safe_index(output_index)
            if not pending and index is not None:
                pending = state["pending_tool_indexes"].pop(index, {})

            if args_str is None or args_str == "":
                args_str = pending.get("arguments", "{}") or "{}"

            call_id = pending.get("call_id") or item_id or ""
            if (call_id and call_id in state["completed_call_ids"]) or (
                item_id and item_id in state["completed_item_ids"]
            ):
                return emitted

            function_name = getattr(event, "name", "") or pending.get("name", "")
            if not function_name:
                return emitted

            tool_call = self._complete_responses_tool_call(
                call_id, function_name, args_str, tools_dict
            )
            state["completed_tool_calls"].append(tool_call)
            state["completed_call_ids"].add(call_id)
            if item_id:
                state["completed_item_ids"].add(item_id)
            emitted.append(self._event_builder.tool_call(tool_call))
            return emitted

        return emitted

    @staticmethod
    def _complete_responses_tool_call(
        call_id: str,
        name: str,
        args_str: Any,
        tools_dict: dict[str, Callable],
    ) -> ToolCall:
        try:
            args = json.loads(args_str)
        except (ValueError, TypeError, RecursionError):
            args = {"_raw": args_str if isinstance(args_str, str) else json.dumps(args_str, default=str)}
        if not isinstance(args, dict):
            args = {"_raw": json.dumps(args, ensure_ascii=False, default=str)}
        return {
            "id": call_id,
            "name": name,
            "arguments": args,
            "callable": tools_dict.get(name) or None,
        }

    @staticmethod
    def _derive_responses_stop_reason(response: Any, state: dict[str, Any]) -> Optional[str]:
        status = getattr(response, "status", None)
        if status == "failed":
            return "failed"
        if status == "cancelled":
            return "cancelled"
        if status == "incomplete":
            details = getattr(response, "incomplete_details", None)
            reason = (
                details.get("reason")
                if isinstance(details, dict)
                else getattr(details, "reason", None)
            )
            if reason == "max_output_tokens":
                return "length"
            if reason == "content_filter":
                return "content_filter"
            return "incomplete"
        if status == "completed":
            return "tool_calls" if state.get("completed_tool_calls") else "stop"
        return None

    @staticmethod
    def _normalize_stop_reason(reason: Optional[str]) -> Optional[str]:
        return _normalize_stop_reason(reason)

    def _finalize_responses_stream(
        self,
        state: dict[str, Any],
        structured_output: bool,
        verbose: bool,
        final: bool,
        include_reasoning: bool,
        latency: Optional[float],
        elapsed: float,
    ) -> Generator[StreamEvent, None, None]:
        parser = state["reasoning_parser"]
        if parser is not None:
            reasoning_part, answer_part = parser.flush()
            if reasoning_part:
                state["reasoning"] += reasoning_part
                if include_reasoning:
                    yield self._event_builder.reasoning(reasoning_part)
            if answer_part:
                state["answer"] += answer_part
                if not structured_output:
                    yield self._event_builder.answer(answer_part)

        if state["refusal"]:
            state["stop_reason"] = "refusal"

        answer: Any = state["answer"]
        if structured_output:
            answer = _parse_structured_output(
                state["answer"],
                stop_reason=state["stop_reason"],
                strict=True,
                refusal=state["refusal"],
                has_tool_calls=bool(state["completed_tool_calls"]),
            )
            yield self._event_builder.answer(answer)

        completed_tool_calls = state["completed_tool_calls"]

        api_usage = state["api_usage"]
        stream_tokens = self._safe_index(state["tokens"], what="tokens") or 0
        completion_tokens = self._safe_index(
            api_usage.get("completion_tokens"), what="completion_tokens"
        )
        prompt_tokens = self._safe_index(
            api_usage.get("prompt_tokens"), what="prompt_tokens"
        )
        tokens, total_tokens = _resolve_token_metrics(
            completion_tokens,
            prompt_tokens,
            self._safe_index(api_usage.get("total_tokens"), what="total_tokens"),
            stream_tokens,
        )
        tokens_per_second = _decode_tokens_per_second(
            state["t_first"], state["t_last"], elapsed, tokens
        )

        verbose_info: VerboseInfo = {
            "tokens": tokens,
            "chunks": stream_tokens,
            "tokens_per_second": tokens_per_second,
            "latency": latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "stop_reason": state["stop_reason"],
        }

        if verbose:
            yield self._event_builder.verbose(dict(verbose_info))

        if final:
            final_response: FinalResponse = {"answer": answer}
            if include_reasoning and state["reasoning"]:
                final_response["reasoning"] = state["reasoning"]
            if state["refusal"]:
                final_response["refusal"] = state["refusal"]
            if parser is not None and parser.is_inside_reasoning and include_reasoning:
                final_response["reasoning_unterminated"] = True
            if completed_tool_calls:
                final_response["tool_calls"] = completed_tool_calls
            if state["response_items"]:
                # Reasoning items for the next turn (reasoning models require
                # them to be replayed alongside function_call items).
                final_response["response_items"] = state["response_items"]
            if state["stop_reason"]:
                final_response["stop_reason"] = state["stop_reason"]
            if verbose:
                final_response["verbose"] = dict(verbose_info)
            yield self._event_builder.final(final_response)

        yield self._event_builder.done()

    def _build_responses_request(
        self,
        messages: list[dict],
        output_format: Optional[dict],
        tools: Optional[list],
        reasoning_effort: Optional[str],
        max_tokens: Optional[int],
        extra_body: Optional[dict],
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, list[str]]] = None,
        seed: Optional[int] = None,
        user: Optional[str] = None,
        tool_choice: Optional[Any] = None,
        store: Optional[bool] = None,
        include_reasoning: bool = True,
        reasoning_budget: Optional[int] = None,
    ) -> tuple[dict[str, Any], PreparedTools, bool]:
        """Build a request payload for the Responses API (/v1/responses)."""
        if extra_body is not None and not isinstance(extra_body, dict):
            raise ConfigurationError("extra_body must be a dict of provider-specific fields")
        _reject_reserved_extra_body_keys(extra_body)
        _validate_generation_options(
            temperature, top_p, max_tokens, seed, stop, user, store,
            reasoning_effort, reasoning_budget,
        )
        # seed/stop have no Responses equivalent (verified against the SDK
        # surface); dropping them silently would hide mode-dependent behavior.
        if seed is not None:
            logger.warning("seed is not supported by the Responses API; ignoring it")
        if stop is not None:
            logger.warning("stop sequences are not supported by the Responses API; ignoring them")

        raw_messages = _copy_messages_shallow(messages)
        ImageProcessor.process_messages(
            raw_messages, max_image_side=self._config.max_image_side
        )
        AudioProcessor.process_messages(raw_messages)
        VideoProcessor.process_messages(raw_messages)
        FileProcessor.process_messages(raw_messages)
        input_items = self._to_responses_input(raw_messages)

        prepared_tools = self._tool_preparator.prepare(tools)
        structured_output = self._is_structured_output_format(output_format)

        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "input": input_items,
            "stream": True,
        }

        if prepared_tools.definitions:
            kwargs["tools"] = self._convert_tools_for_responses_api(
                prepared_tools.definitions
            )
        if reasoning_effort:
            effort = reasoning_effort.strip()
            _reject_confusable_controls(effort, what="reasoning_effort")
            if effort not in RequestTransformer._KNOWN_REASONING_EFFORTS:
                # Same warn-passthrough as the chat path (validated
                # non-empty upstream; stripped below before sending).
                self._request_transformer._warn_once(
                    f"effort:{effort}",
                    "reasoning_effort %r is not a commonly supported value; passing it through",
                    effort,
                )
            # Reasoning summaries require a verified organization on some
            # reasoning models — only request them when the caller actually
            # wants to see reasoning, not on every reasoning_effort.
            reasoning: dict[str, Any] = {"effort": effort}
            if include_reasoning:
                reasoning["summary"] = "auto"
            kwargs["reasoning"] = reasoning
        elif reasoning_budget is not None:
            reasoning = {"max_tokens": reasoning_budget}
            if include_reasoning:
                reasoning["summary"] = "auto"
            kwargs["reasoning"] = reasoning
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if user is not None:
            kwargs["user"] = user
        if tool_choice is not None:
            kwargs["tool_choice"] = self._convert_tool_choice_for_responses(tool_choice)
        if store is not None:
            kwargs["store"] = store
            if store is False and "include" not in kwargs:
                # Unstored reasoning must be requested explicitly to stay
                # replayable on the next tool turn.
                kwargs["include"] = ["reasoning.encrypted_content"]
        if structured_output:
            kwargs["text"] = {
                "format": self._convert_output_format_for_responses_api(output_format)
            }

        merged_extra_body = _deep_merge_dicts(self._config.extra_body, extra_body)
        if merged_extra_body:
            kwargs["extra_body"] = merged_extra_body

        return kwargs, prepared_tools, structured_output

    def _should_retry_without_stream_options(self, error: APIError, kwargs: dict[str, Any]) -> bool:
        """Retry only when the server explicitly rejected stream_options."""
        if "stream_options" not in kwargs:
            return False
        status_code = getattr(error, "status_code", None)
        if status_code not in (400, 422):
            return False
        body = getattr(error, "body", None)
        try:
            body_text = json.dumps(body) if isinstance(body, dict) else str(body)
        except (TypeError, ValueError):
            body_text = str(body)
        return "stream_options" in f"{error} {body_text}"

    @staticmethod
    def _wrap_request_error(error: APIError, context: str) -> ModelRequestError:
        """Wrap an openai.APIError, preserving status/body/request id."""
        return ModelRequestError(
            f"{context}: {_redact_url_credentials(str(error))}",
            status_code=getattr(error, "status_code", None),
            body=_redact_json_urls(getattr(error, "body", None)),
            request_id=getattr(error, "request_id", None),
        )

    # ========================================================================
    # Responses API – Streams
    # ========================================================================

    def _stream_responses_sync(
        self,
        messages: list[dict],
        output_format: Optional[dict],
        tools: Optional[list],
        reasoning_effort: Optional[str],
        max_tokens: Optional[int],
        verbose: bool,
        include_reasoning: bool,
        final: bool,
        extra_body: Optional[dict],
        max_retries: Optional[int] = None,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, list[str]]] = None,
        seed: Optional[int] = None,
        user: Optional[str] = None,
        tool_choice: Optional[Any] = None,
        store: Optional[bool] = None,
        reasoning_budget: Optional[int] = None,
    ) -> Generator[StreamEvent, None, None]:
        """Sync streaming via the Responses API (/v1/responses)."""
        kwargs, prepared_tools, structured_output = self._build_responses_request(
            messages, output_format, tools, reasoning_effort, max_tokens, extra_body,
            temperature=temperature, top_p=top_p, stop=stop, seed=seed, user=user,
            tool_choice=tool_choice, store=store, include_reasoning=include_reasoning,
            reasoning_budget=reasoning_budget,
        )
        tools_dict = prepared_tools.callables

        state = self._new_responses_state()
        state["reasoning_parser"] = ReasoningParser(self._config.reasoning_pattern)
        start_time = time.perf_counter()
        latency: Optional[float] = None

        client = self._client_for(max_retries)

        try:
            stream = client.responses.create(**kwargs)
        except APIError as e:
            raise self._wrap_request_error(e, "Responses API request failed") from e

        try:
            for event in stream:
                now = time.perf_counter()
                if state["t_first"] is None:
                    state["t_first"] = now
                    latency = now - start_time
                state["t_last"] = now
                for emitted in self._handle_responses_event(
                    event, state, structured_output, include_reasoning, tools_dict
                ):
                    yield emitted
        except APIError as e:
            raise self._wrap_request_error(e, "Responses API stream failed") from e
        finally:
            if hasattr(stream, "close"):
                try:
                    stream.close()
                except Exception as e:
                    logger.warning("Failed to close stream: %s", _redact_url_credentials(str(e)))

        elapsed = time.perf_counter() - start_time
        yield from self._finalize_responses_stream(
            state, structured_output, verbose, final, include_reasoning, latency, elapsed,
        )

    async def _stream_responses_async(
        self,
        messages: list[dict],
        output_format: Optional[dict],
        tools: Optional[list],
        reasoning_effort: Optional[str],
        max_tokens: Optional[int],
        verbose: bool,
        include_reasoning: bool,
        final: bool,
        extra_body: Optional[dict],
        max_retries: Optional[int] = None,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, list[str]]] = None,
        seed: Optional[int] = None,
        user: Optional[str] = None,
        tool_choice: Optional[Any] = None,
        store: Optional[bool] = None,
        reasoning_budget: Optional[int] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Async streaming via the Responses API (/v1/responses)."""
        kwargs, prepared_tools, structured_output = self._build_responses_request(
            messages, output_format, tools, reasoning_effort, max_tokens, extra_body,
            temperature=temperature, top_p=top_p, stop=stop, seed=seed, user=user,
            tool_choice=tool_choice, store=store, include_reasoning=include_reasoning,
            reasoning_budget=reasoning_budget,
        )
        tools_dict = prepared_tools.callables

        state = self._new_responses_state()
        state["reasoning_parser"] = ReasoningParser(self._config.reasoning_pattern)
        start_time = time.perf_counter()
        latency: Optional[float] = None

        client = self._async_client_for(max_retries)

        async def _acreate():
            call = client.responses.create(**kwargs)
            return await call if asyncio.iscoroutine(call) else call

        try:
            stream = await _acreate()
        except APIError as e:
            raise self._wrap_request_error(e, "Async Responses API request failed") from e

        try:
            async for event in stream:
                now = time.perf_counter()
                if state["t_first"] is None:
                    state["t_first"] = now
                    latency = now - start_time
                state["t_last"] = now
                for emitted in self._handle_responses_event(
                    event, state, structured_output, include_reasoning, tools_dict
                ):
                    yield emitted
        except APIError as e:
            raise self._wrap_request_error(e, "Async Responses API stream failed") from e
        finally:
            try:
                await _aclose_async_resource(stream)
            except Exception as e:
                logger.warning("Failed to close async stream: %s", _redact_url_credentials(str(e)))

        elapsed = time.perf_counter() - start_time
        for emitted in self._finalize_responses_stream(
            state, structured_output, verbose, final, include_reasoning, latency, elapsed,
        ):
            yield emitted

    # ========================================================================
    # Public API – Sync
    # ========================================================================

    def response(
        self,
        messages: Optional[list[dict]] = None,
        *,
        input: Optional[str] = None,
        system: Optional[str] = None,
        output_format: Union[dict, type, None] = None,
        tools: Optional[list] = None,
        reasoning_effort: Optional[str] = None,
        reasoning_budget: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, list[str]]] = None,
        seed: Optional[int] = None,
        user: Optional[str] = None,
        tool_choice: Optional[Any] = None,
        store: Optional[bool] = None,
        verbose: bool = False,
        include_reasoning: bool = True,
        schema_strict: bool = True,
        extra_body: Optional[dict] = None,
        max_retries: Optional[int] = None,
    ) -> FinalResponse:
        """Request model inference (non-streaming).

        Pass ``input="..."`` for a single user message or ``messages=[...]``
        (optionally with ``system=...``). Use ``extra_body`` only for
        provider-specific fields — it is merged last and can override
        request fields, so prefer the explicit parameters. The keys
        ``model``, ``messages``, ``input`` and ``stream`` are rejected
        outright (they would silently break the call).

        Args:
            messages: Chat message list (exclusive with ``input``).
            input: Single user message shorthand (exclusive with
                ``messages``).
            system: System prompt (combinable with either).
            output_format: Dict schema or typed class for strict JSON
                output (invalid JSON raises ``StructuredOutputError``).
            tools: Python callables or OpenAI tool definitions.
            reasoning_effort: Effort level string (mutually exclusive
                with ``reasoning_budget``).
            reasoning_budget: Thinking-token budget int (mutually
                exclusive with ``reasoning_effort``).
            max_tokens/temperature/top_p/stop/seed/user/tool_choice/store:
                Per-call generation options, validated fail-fast
                (``seed``/``stop`` are ignored with a warning in
                Responses mode).
            verbose: Include a verbose stats dict in the final response.
            include_reasoning: Keep reasoning in output (default True).
            schema_strict: Relax the *generated* schema when False;
                parsing stays strict regardless.
            extra_body: Provider-specific extra fields (must be a dict).
            max_retries: Per-call retry override (``int >= 0``).
        """
        resolved_messages = _resolve_messages(messages, input, system)
        output_format = self._prepare_output_format(output_format, strict=schema_strict)

        final_content: Optional[FinalResponse] = None

        gen = self.stream_response(
            messages=resolved_messages,
            output_format=output_format,
            final=True,
            tools=tools,
            include_reasoning=include_reasoning,
            reasoning_effort=reasoning_effort,
            reasoning_budget=reasoning_budget,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            seed=seed,
            user=user,
            tool_choice=tool_choice,
            store=store,
            verbose=verbose,
            extra_body=extra_body,
            max_retries=max_retries,
        )
        try:
            for event in gen:
                if event.get("type") == EventType.FINAL.value:
                    final_content = event.get("content")
                    break
        finally:
            close_gen = getattr(gen, "close", None)
            if callable(close_gen):
                close_gen()

        if final_content is None:
            raise LLMError("No final response received from stream")

        return final_content

    def stream_response(
        self,
        messages: Optional[list[dict]] = None,
        *,
        input: Optional[str] = None,
        system: Optional[str] = None,
        output_format: Union[dict, type, None] = None,
        tools: Optional[list] = None,
        final: bool = False,
        include_reasoning: bool = True,
        reasoning_effort: Optional[str] = None,
        reasoning_budget: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, list[str]]] = None,
        seed: Optional[int] = None,
        user: Optional[str] = None,
        tool_choice: Optional[Any] = None,
        store: Optional[bool] = None,
        verbose: bool = False,
        schema_strict: bool = True,
        extra_body: Optional[dict] = None,
        max_retries: Optional[int] = None,
    ) -> Generator[StreamEvent, None, None]:
        """Request model inference with streaming.

        Yields answer/reasoning/refusal/tool_call events as they arrive and,
        at stream end, tool_call events for completed calls, an optional
        ``verbose`` event, a ``final`` event (``final=True``) and ``done``.
        With ``output_format`` the answer is parsed once at the end (not
        streamed incrementally).

        If you stop consuming early (``break``), close the generator
        (``gen.close()``) so the underlying HTTP response is released
        promptly instead of at garbage collection. Same parameters as
        ``response``, plus ``final=True`` to also yield the aggregated
        final event.
        """
        resolved_messages = _resolve_messages(messages, input, system)
        output_format = self._prepare_output_format(output_format, strict=schema_strict)
        if self._config.use_responses_api:
            yield from self._stream_responses_sync(
                messages=resolved_messages,
                output_format=output_format,
                tools=tools,
                reasoning_effort=reasoning_effort,
                reasoning_budget=reasoning_budget,
                max_tokens=max_tokens,
                verbose=verbose,
                include_reasoning=include_reasoning,
                final=final,
                extra_body=extra_body,
                max_retries=max_retries,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                seed=seed,
                user=user,
                tool_choice=tool_choice,
                store=store,
            )
            return

        kwargs, prepared_tools, structured_output = self._build_request(
            resolved_messages, output_format, tools, reasoning_effort, max_tokens, extra_body,
            temperature=temperature, top_p=top_p, stop=stop, seed=seed, user=user,
            tool_choice=tool_choice, store=store, reasoning_budget=reasoning_budget,
        )

        parser = ReasoningParser(self._config.reasoning_pattern)
        tool_handler = ToolCallStreamHandler(
            self._event_builder, prepared_tools.callables
        )
        start_time = time.perf_counter()
        state = _ChatStreamState(
            parser=parser,
            tool_handler=tool_handler,
            structured_output=structured_output,
            include_reasoning=include_reasoning,
            event_builder=self._event_builder,
            start_time=start_time,
        )

        client = self._client_for(max_retries)
        retried = False
        completion = None

        def _create():
            return client.chat.completions.create(**kwargs)

        while completion is None:
            try:
                completion = _create()
            except APIError as e:
                if not retried and self._should_retry_without_stream_options(e, kwargs):
                    kwargs.pop("stream_options", None)
                    self._stream_options_supported = False
                    retried = True
                    continue
                raise self._wrap_request_error(e, "Model request failed") from e

        try:
            for chunk in completion:
                for stream_event in state.handle_chunk(chunk):
                    yield stream_event
        except APIError as e:
            raise self._wrap_request_error(e, "Model stream failed") from e
        finally:
            if hasattr(completion, "close"):
                try:
                    completion.close()
                except Exception as e:
                    logger.warning("Failed to close stream: %s", _redact_url_credentials(str(e)))

        elapsed = time.perf_counter() - start_time
        for stream_event in state.finish(elapsed=elapsed, verbose=verbose, final=final):
            yield stream_event

    # ========================================================================
    # Public API – Async
    # ========================================================================

    async def async_response(
        self,
        messages: Optional[list[dict]] = None,
        *,
        input: Optional[str] = None,
        system: Optional[str] = None,
        output_format: Union[dict, type, None] = None,
        tools: Optional[list] = None,
        reasoning_effort: Optional[str] = None,
        reasoning_budget: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, list[str]]] = None,
        seed: Optional[int] = None,
        user: Optional[str] = None,
        tool_choice: Optional[Any] = None,
        store: Optional[bool] = None,
        verbose: bool = False,
        include_reasoning: bool = True,
        schema_strict: bool = True,
        extra_body: Optional[dict] = None,
        max_retries: Optional[int] = None,
    ) -> FinalResponse:
        """Async request for model inference (same parameters as ``response``)."""
        resolved_messages = _resolve_messages(messages, input, system)
        output_format = self._prepare_output_format(output_format, strict=schema_strict)

        final_content: Optional[FinalResponse] = None

        agen = self.async_stream_response(
            messages=resolved_messages,
            output_format=output_format,
            final=True,
            tools=tools,
            include_reasoning=include_reasoning,
            reasoning_effort=reasoning_effort,
            reasoning_budget=reasoning_budget,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            seed=seed,
            user=user,
            tool_choice=tool_choice,
            store=store,
            verbose=verbose,
            extra_body=extra_body,
            max_retries=max_retries,
        )
        try:
            async for event in agen:
                if event.get("type") == EventType.FINAL.value:
                    final_content = event.get("content")
                    break
        finally:
            aclose_gen = getattr(agen, "aclose", None)
            if callable(aclose_gen):
                await aclose_gen()

        if final_content is None:
            raise LLMError("No final response received from stream")

        return final_content

    async def async_stream_response(
        self,
        messages: Optional[list[dict]] = None,
        *,
        input: Optional[str] = None,
        system: Optional[str] = None,
        output_format: Union[dict, type, None] = None,
        tools: Optional[list] = None,
        final: bool = False,
        include_reasoning: bool = True,
        reasoning_effort: Optional[str] = None,
        reasoning_budget: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, list[str]]] = None,
        seed: Optional[int] = None,
        user: Optional[str] = None,
        tool_choice: Optional[Any] = None,
        store: Optional[bool] = None,
        verbose: bool = False,
        schema_strict: bool = True,
        extra_body: Optional[dict] = None,
        max_retries: Optional[int] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Async streaming model inference (see ``stream_response``).

        On early exit, call ``await agen.aclose()`` to release the HTTP
        response promptly.
        """
        resolved_messages = _resolve_messages(messages, input, system)
        output_format = self._prepare_output_format(output_format, strict=schema_strict)
        if self._config.use_responses_api:
            async for event in self._stream_responses_async(
                messages=resolved_messages,
                output_format=output_format,
                tools=tools,
                reasoning_effort=reasoning_effort,
                reasoning_budget=reasoning_budget,
                max_tokens=max_tokens,
                verbose=verbose,
                include_reasoning=include_reasoning,
                final=final,
                extra_body=extra_body,
                max_retries=max_retries,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                seed=seed,
                user=user,
                tool_choice=tool_choice,
                store=store,
            ):
                yield event
            return

        kwargs, prepared_tools, structured_output = self._build_request(
            resolved_messages, output_format, tools, reasoning_effort, max_tokens, extra_body,
            temperature=temperature, top_p=top_p, stop=stop, seed=seed, user=user,
            tool_choice=tool_choice, store=store, reasoning_budget=reasoning_budget,
        )

        parser = ReasoningParser(self._config.reasoning_pattern)
        tool_handler = ToolCallStreamHandler(
            self._event_builder, prepared_tools.callables
        )
        start_time = time.perf_counter()
        state = _ChatStreamState(
            parser=parser,
            tool_handler=tool_handler,
            structured_output=structured_output,
            include_reasoning=include_reasoning,
            event_builder=self._event_builder,
            start_time=start_time,
        )

        client = self._async_client_for(max_retries)

        async def _acreate():
            call = client.chat.completions.create(**kwargs)
            return await call if asyncio.iscoroutine(call) else call

        retried = False
        completion = None
        while completion is None:
            try:
                completion = await _acreate()
            except APIError as e:
                if not retried and self._should_retry_without_stream_options(e, kwargs):
                    kwargs.pop("stream_options", None)
                    self._stream_options_supported = False
                    retried = True
                    continue
                raise self._wrap_request_error(e, "Async model request failed") from e

        try:
            iterator = completion.__aiter__()
            while True:
                try:
                    chunk = await anext(iterator)
                except StopAsyncIteration:
                    break
                for stream_event in state.handle_chunk(chunk):
                    yield stream_event
        except APIError as e:
            raise self._wrap_request_error(e, "Async model stream failed") from e
        finally:
            try:
                await _aclose_async_resource(completion)
            except Exception as e:
                logger.warning("Failed to close async stream: %s", _redact_url_credentials(str(e)))

        elapsed = time.perf_counter() - start_time
        for stream_event in state.finish(elapsed=elapsed, verbose=verbose, final=final):
            yield stream_event

    # ========================================================================
    # Context Manager
    # ========================================================================

    def close(self) -> None:
        """Close clients; never raises, even inside a running event loop."""
        try:
            if hasattr(self._client, "close"):
                self._client.close()
        except Exception as e:
            logger.warning("Failed to close sync client: %s", _redact_url_credentials(str(e)))
        finally:
            self._close_async_client_best_effort()

    def _close_async_client_best_effort(self) -> None:
        async_client = self._async_client
        self._async_client = None
        self._async_client_loop_ref = None
        if async_client is None:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                asyncio.run(_aclose_async_resource(async_client))
            except Exception as e:
                logger.warning("Failed to close async client: %s", _redact_url_credentials(str(e)))
        else:
            logger.warning(
                "LLM.close() called inside a running event loop; "
                "scheduling best-effort close (prefer 'await llm.aclose()')"
            )
            _schedule_async_close(async_client)

    async def aclose(self) -> None:
        """Close clients; never raises (failures are logged)."""
        try:
            if hasattr(self._client, "close"):
                self._client.close()
        except Exception as e:
            logger.warning("Failed to close sync client: %s", _redact_url_credentials(str(e)))
        async_client = self._async_client
        self._async_client = None
        self._async_client_loop_ref = None
        if async_client is not None:
            try:
                await _aclose_async_resource(async_client)
            except Exception as e:
                logger.warning("Failed to close async client: %s", _redact_url_credentials(str(e)))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()
        return False


# ============================================================================
# Module-Level Model Listing
# ============================================================================

def list_models(
    fallback: Optional[Union[list[str], tuple[str, ...]]] = None,
    max_retries: Optional[int] = None,
    client: Optional[LLM] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    raise_on_error: bool = False,
    normalize_base_url: bool = True,
    timeout: Optional[float] = None,
) -> list[str]:
    """Return model IDs from the configured API, or fallback/[] on failure.

    When ``client`` is given, the call delegates to it (reusing its headers,
    timeout, transport and retry configuration); ``api_key``/``base_url``/
    ``normalize_base_url``/``timeout`` only apply without one. API failures
    raise ``ModelRequestError`` when ``raise_on_error`` is true.
    ``timeout=None`` (default) leaves the OpenAI client default in place
    rather than applying ``DEFAULT_TIMEOUT``.
    """
    if max_retries is not None and (
        isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0
    ):
        raise ConfigurationError("max_retries must be an int >= 0")
    if fallback is not None and (
        not isinstance(fallback, (list, tuple))
        or not all(isinstance(m, str) and m.strip() for m in fallback)
    ):
        raise ConfigurationError(
            "fallback must be a list of non-empty model id strings or None"
        )
    if client is not None:
        if (
            api_key is not None
            or base_url is not None
            or timeout is not None
        ):
            logger.warning(
                "api_key/base_url/normalize_base_url/timeout are ignored "
                "when a client is provided"
            )
        return client.list_models(
            fallback=fallback,
            max_retries=max_retries,
            raise_on_error=raise_on_error,
        )

    resolved_key = api_key if api_key is not None else DEFAULT_API_KEY
    if base_url is not None:
        _validate_base_url(base_url)
    resolved_base = (
        _resolve_api_base(base_url, normalize=normalize_base_url)
        if base_url is not None
        else DEFAULT_BASE_URL
    )
    standalone_kwargs: dict[str, Any] = {
        "api_key": resolved_key,
        "base_url": resolved_base,
        "max_retries": 3 if max_retries is None else max_retries,
    }
    if timeout is not None:
        _validate_connection_options(timeout, 0, None)
        standalone_kwargs["timeout"] = timeout
    standalone = OpenAI(**standalone_kwargs)
    try:
        models = standalone.models.list()
        return sorted({model.id for model in models.data}) or list(fallback or [])
    except Exception as e:
        if raise_on_error:
            if isinstance(e, APIError):
                raise LLM._wrap_request_error(e, "list_models") from e
            raise
        logger.warning("list_models failed, falling back: %s", _redact_url_credentials(str(e)))
        return list(fallback or [])
    finally:
        try:
            standalone.close()
        except Exception as e:
            logger.warning(
                "list_models failed to close client: %s",
                _redact_url_credentials(str(e)),
            )


async def async_list_models(
    fallback: Optional[Union[list[str], tuple[str, ...]]] = None,
    max_retries: Optional[int] = None,
    client: Optional[LLM] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    raise_on_error: bool = False,
    normalize_base_url: bool = True,
    timeout: Optional[float] = None,
) -> list[str]:
    """Async version of list_models (same fallback/error/timeout semantics)."""
    if max_retries is not None and (
        isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0
    ):
        raise ConfigurationError("max_retries must be an int >= 0")
    if fallback is not None and (
        not isinstance(fallback, (list, tuple))
        or not all(isinstance(m, str) and m.strip() for m in fallback)
    ):
        raise ConfigurationError(
            "fallback must be a list of non-empty model id strings or None"
        )
    if client is not None:
        if (
            api_key is not None
            or base_url is not None
            or timeout is not None
        ):
            logger.warning(
                "api_key/base_url/normalize_base_url/timeout are ignored "
                "when a client is provided"
            )
        return await client.async_list_models(
            fallback=fallback,
            max_retries=max_retries,
            raise_on_error=raise_on_error,
        )

    resolved_key = api_key if api_key is not None else DEFAULT_API_KEY
    if base_url is not None:
        _validate_base_url(base_url)
    resolved_base = (
        _resolve_api_base(base_url, normalize=normalize_base_url)
        if base_url is not None
        else DEFAULT_BASE_URL
    )
    standalone_kwargs: dict[str, Any] = {
        "api_key": resolved_key,
        "base_url": resolved_base,
        "max_retries": 3 if max_retries is None else max_retries,
    }
    if timeout is not None:
        _validate_connection_options(timeout, 0, None)
        standalone_kwargs["timeout"] = timeout
    standalone = AsyncOpenAI(**standalone_kwargs)
    try:
        models = await standalone.models.list()
        return sorted({model.id for model in models.data}) or list(fallback or [])
    except Exception as e:
        if raise_on_error:
            if isinstance(e, APIError):
                raise LLM._wrap_request_error(e, "async_list_models") from e
            raise
        logger.warning("async_list_models failed, falling back: %s", _redact_url_credentials(str(e)))
        return list(fallback or [])
    finally:
        try:
            await _aclose_async_resource(standalone)
        except Exception as e:
            logger.warning(
                "async_list_models failed to close client: %s",
                _redact_url_credentials(str(e)),
            )


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "LLM",
    "LLMConfig",
    "CustomReasoningPattern",
    "StreamEvent",
    "ToolCall",
    "ToolResultMessage",
    "UserMessage",
    "AssistantMessage",
    "assistant_message",
    "user_message",
    "system_message",
    "tool_result",
    "FinalResponse",
    "VerboseInfo",
    "EventType",
    "LLMError",
    "ConfigurationError",
    "SchemaConversionError",
    "ModelRequestError",
    "StructuredOutputError",
    "ImageProcessingError",
    "AudioProcessingError",
    "VideoProcessingError",
    "FileProcessingError",
    "list_models",
    "async_list_models",
    "configure_quiet_logging",
    "configure_debug_logging",
    "__version__",
]
