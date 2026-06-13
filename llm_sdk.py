"""
Universal LLM API Wrapper with OpenAI-compatible API support.

Features:
- Structured outputs with automatic schema generation
- Vision model support
- Tool definitions with automatic function introspection
- Streaming with thinking token handling
"""

from __future__ import annotations

import json
import base64
import re
import io
import time
import sys
import inspect
import logging
import copy
import asyncio
import threading
import functools
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any, AsyncGenerator, Dict, Optional, List, Generator,
    Callable, Union, get_type_hints, get_origin, get_args,
    Literal, TypedDict, TypeVar, Final, ClassVar, Set,
    TYPE_CHECKING
)

from openai import OpenAI, AsyncOpenAI, APIStatusError

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

# ============================================================================
# Version Compatibility
# ============================================================================

if sys.version_info < (3, 10):
    _DEFAULT = object()

    async def anext(async_iterator, default=_DEFAULT):
        """Polyfill for anext() in Python < 3.10."""
        try:
            return await async_iterator.__anext__()
        except StopAsyncIteration:
            if default is _DEFAULT:
                raise
            return default

# ============================================================================
# Logging Configuration
# ============================================================================

logger = logging.getLogger(__name__)

for _logger_name in ("httpx", "openai", "httpcore"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)

# ============================================================================
# Constants
# ============================================================================

DEFAULT_API_KEY: Final[str] = "lm-studio"
DEFAULT_BASE_URL: Final[str] = "http://localhost:1234/v1"
DEFAULT_TIMEOUT: Final[float] = 300.0

MessageList = List[Dict[str, Any]]

def _close_async_resource(resource: Any) -> None:
    if not hasattr(resource, "close"):
        return

    async def _close() -> None:
        close_result = resource.close()
        if inspect.isawaitable(close_result):
            await close_result

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_close())
        return

    error_box: List[BaseException] = []

    def _runner() -> None:
        try:
            asyncio.run(_close())
        except BaseException as exc:  # pragma: no cover - defensive cleanup path
            error_box.append(exc)

    worker = threading.Thread(target=_runner, daemon=True, name="llm-close")
    worker.start()
    worker.join()
    if error_box:
        raise error_box[0]


def _deep_merge_dicts(
    base: Optional[Dict[str, Any]],
    override: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not base and not override:
        return None

    merged = copy.deepcopy(base or {})
    for key, value in (override or {}).items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge_dicts(current, dict(value))
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_messages(
    messages: Optional[list] = None,
    input: Optional[str] = None,
    system: Optional[str] = None,
) -> MessageList:
    if system:
        if not isinstance(system, str):
            raise ValueError("system must be a string")
    if messages is not None and input is not None:
        raise ValueError("Cannot specify both 'messages' and 'input'")
    elif messages is not None and system is not None:
        return [({"role": "system", "content": system})] + messages
    elif messages is not None:
        if not isinstance(messages, list):
            raise ValueError("messages must be a list of dicts")
        return messages
    elif input is not None and system is not None:
        return [{"role": "system", "content": system}, {"role": "user", "content": input}]
    elif input is not None:
        if not isinstance(input, str):
            raise ValueError("input must be a string")
        return [{"role": "user", "content": input}]
    else:
        raise ValueError("Must specify either 'messages' or 'input'")

# ============================================================================
# Enums
# ============================================================================

class EventType(str, Enum):
    """Event types emitted during streaming."""
    ANSWER = "answer"
    REASONING = "reasoning"
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

T = TypeVar('T')


class StreamEvent(TypedDict, total=False):
    type: str
    content: Any
class ToolCall(TypedDict):
    """Typed dictionary for tool calls."""
    id: str
    name: str
    arguments: Dict[str, Any]
    callable: Optional[Callable]
    
class ToolCallMessage(TypedDict):
    id: str
    name: str
    arguments: Dict[str, Any]
    
class ToolResultMessage(TypedDict):
    role: str
    tool_call_id: str
    content: Dict[str, Any] | str
    
class AssistantMessage(TypedDict):
    role: str
    content: Any
    tool_calls: Optional[List[Dict[str, Any]]]
    
class UserMessage(TypedDict):
    role: str
    content: str | List[Dict[str, Any]]
    
class VerboseInfo(TypedDict):
    """Typed dictionary for verbose information."""
    tokens: int
    tokens_per_second: float
    latency: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]


class FinalResponse(TypedDict, total=False):
    """Typed dictionary for final response."""
    answer: Any
    reasoning: str
    tool_calls: List[ToolCall]
    verbose: VerboseInfo

# ============================================================================
# Exceptions
# ============================================================================

class LLMError(Exception):
    """Base exception for LLM errors."""
    pass


class ConfigurationError(LLMError):
    """Raised when configuration is invalid."""
    pass


class SchemaConversionError(LLMError):
    """Raised when schema conversion fails."""
    pass


class ModelRequestError(LLMError):
    """Raised when model request fails."""
    pass

# ============================================================================
# Data Classes
# ============================================================================

@dataclass(frozen=True)
class CustomThinkingToken:
    """Configuration for custom thinking token patterns.

    Attributes:
        from_beginning: Whether content starts inside thinking mode.
        start_token: Custom start token pattern (regex escaped internally).
        end_token: Custom end token pattern (regex escaped internally).
    """
    from_beginning: bool = False
    start_token: Optional[str] = None
    end_token: Optional[str] = None

    def __post_init__(self):
        if self.start_token and not self.end_token:
            raise ConfigurationError("end_token required when start_token is specified")
        if self.end_token and not self.start_token:
            raise ConfigurationError("start_token required when end_token is specified")


@dataclass
class LLMConfig:
    """Configuration for LLM instance."""
    model: str
    api_key: str = DEFAULT_API_KEY
    base_url: str = DEFAULT_BASE_URL
    custom_thinking_token: Optional[CustomThinkingToken] = None
    default_stop_sequences: Optional[List[str]] = None
    timeout: float = DEFAULT_TIMEOUT
    extra_body: Optional[Dict[str, Any]] = None
    use_responses_api: bool = False
    default_headers: Optional[Dict[str, str]] = None
    max_retries: int = 3

# ============================================================================
# Thinking Parser
# ============================================================================

class ThinkingParser:
    """Parses thinking tokens from streamed content.

    Supports multiple thinking tag formats:
    - XML-style: <think>, <thinking>
    - Bracket-style: [THINK]
    - Custom patterns via CustomThinkingToken
    """

    _BASE_START_PATTERNS: ClassVar[tuple[str, ...]] = (
        r'<think>', r'<thinking>', r'\[THINK\]', r'<thought>'
    )
    _BASE_END_PATTERNS: ClassVar[tuple[str, ...]] = (
        r'</think>', r'</thinking>', r'\[/THINK\]', r'</thought>'
    )

    def __init__(self, custom_token: Optional[CustomThinkingToken] = None):
        self._custom_token = custom_token
        self._start_pattern = self._build_pattern(
            self._BASE_START_PATTERNS,
            custom_token.start_token if custom_token else None
        )
        self._end_pattern = self._build_pattern(
            self._BASE_END_PATTERNS,
            custom_token.end_token if custom_token else None
        )
        self._inside_think = custom_token.from_beginning if custom_token else False

    @staticmethod
    def _build_pattern(base_patterns: tuple[str, ...], custom: Optional[str]) -> re.Pattern:
        """Build compiled regex pattern from base patterns and optional custom pattern."""
        patterns = list(base_patterns)
        if custom:
            patterns.append(re.escape(custom))
        return re.compile('|'.join(patterns), flags=re.IGNORECASE)

    def reset(self, inside_think: Optional[bool] = None) -> None:
        """Reset parser state."""
        if inside_think is not None:
            self._inside_think = inside_think
        elif self._custom_token:
            self._inside_think = self._custom_token.from_beginning
        else:
            self._inside_think = False

    def parse(self, content: str) -> tuple[str, str]:
        """Parse content and separate thinking from answer.

        Returns:
            Tuple of (thinking_part, answer_part).
        """
        thinking_part = ""
        answer_part = ""
        remaining = content

        while remaining:
            if self._inside_think:
                match = self._end_pattern.search(remaining)
                if match:
                    thinking_part += remaining[:match.start()]
                    self._inside_think = False
                    remaining = remaining[match.end():]
                else:
                    thinking_part += remaining
                    remaining = ""
            else:
                match = self._start_pattern.search(remaining)
                if match:
                    answer_part += remaining[:match.start()]
                    self._inside_think = True
                    remaining = remaining[match.end():]
                else:
                    answer_part += remaining
                    remaining = ""

        return thinking_part, answer_part

    @property
    def is_inside_thinking(self) -> bool:
        """Whether parser is currently inside a thinking block."""
        return self._inside_think

# ============================================================================
# Schema Converter
# ============================================================================

class SchemaConverter:
    """Converts Python types and classes to JSON Schema format."""

    _TYPE_MAP: ClassVar[Dict[str, SchemaType]] = {
        "str": SchemaType.STRING,
        "int": SchemaType.INTEGER,
        "float": SchemaType.NUMBER,
        "bool": SchemaType.BOOLEAN,
        "list": SchemaType.ARRAY,
        "dict": SchemaType.OBJECT,
    }

    _LLM_SUPPORTED_TYPES: ClassVar[frozenset] = frozenset({str, int, float, bool, list, dict})

    @staticmethod
    def _ordered_object_schema(
        required: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        additional_properties: Any = False,
    ) -> Dict[str, Any]:
        schema: Dict[str, Any] = {"type": SchemaType.OBJECT.value}
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
        seen_models: Optional[set] = None
    ) -> Dict[str, Any]:
        """Convert Python type annotation to JSON Schema."""
        seen_models = seen_models or set()

        if python_type is type(None):
            return {"type": SchemaType.NULL.value}

        origin = get_origin(python_type)
        args = get_args(python_type)

        if origin is list:
            schema: Dict[str, Any] = {"type": SchemaType.ARRAY.value}
            if args:
                schema["items"] = self.python_type_to_json_schema(args[0], seen_models)
            return schema

        if origin is dict:
            schema = self._ordered_object_schema(required=None, properties=None, additional_properties=None)
            if len(args) == 2:
                schema["additionalProperties"] = self.python_type_to_json_schema(args[1], seen_models)
            return schema

        if origin is Union:
            non_none_types = [t for t in args if t is not type(None)]
            if len(non_none_types) == 1:
                return {
                    "anyOf": [
                        self.python_type_to_json_schema(non_none_types[0], seen_models),
                        {"type": SchemaType.NULL.value}
                    ]
                }
            return {
                "anyOf": [self.python_type_to_json_schema(t, seen_models) for t in args]
            }

        if origin is Literal:
            return {"enum": list(args)}

        if self._is_annotated_class(python_type):
            if python_type in seen_models:
                raise SchemaConversionError(
                    f"Circular dependency detected for class {python_type.__name__}. "
                    "Recursive schemas are not supported."
                )
            nested_schema = self.convert_class_to_schema(python_type, seen_models=seen_models)
            return nested_schema["json_schema"]["schema"]

        return {"type": self._get_json_type(python_type).value}

    def _is_annotated_class(self, python_type: Any) -> bool:
        return (
            hasattr(python_type, "__annotations__")
            and python_type.__annotations__
            and isinstance(python_type, type)
        )

    def _get_json_type(self, python_type: Any) -> SchemaType:
        type_name = getattr(python_type, "__name__", str(python_type)).lower()
        return self._TYPE_MAP.get(type_name, SchemaType.STRING)

    def is_llm_supported_type(self, python_type: Any) -> bool:
        """Check if a Python type can be meaningfully provided by an LLM."""
        if python_type is None or python_type is type(None):
            return True

        origin = get_origin(python_type)
        args = get_args(python_type)

        if origin is list:
            return not args or self.is_llm_supported_type(args[0])
        if origin is dict:
            return len(args) != 2 or self.is_llm_supported_type(args[1])
        if origin is Union:
            non_none = [t for t in args if t is not type(None)]
            return all(self.is_llm_supported_type(t) for t in non_none)
        if origin is Literal:
            return True

        return python_type in self._LLM_SUPPORTED_TYPES

    def convert_class_to_schema(
        self,
        schema_class: type,
        name: Optional[str] = None,
        seen_models: Optional[set] = None
    ) -> Dict[str, Any]:
        """Convert plain class with __annotations__ to OpenAI JSON schema."""
        seen_models = seen_models or set()

        if not hasattr(schema_class, "__annotations__") or not schema_class.__annotations__:
            raise SchemaConversionError(
                f"Class {schema_class.__name__} has no type annotations."
            )

        seen_models.add(schema_class)

        try:
            hints = get_type_hints(schema_class)
            properties = {}
            required = []

            class_defaults = {
                k: v for k, v in schema_class.__dict__.items()
                if not k.startswith("_") and not callable(v)
            }

            for field_name, field_type in hints.items():
                properties[field_name] = self.python_type_to_json_schema(field_type, seen_models)

                is_optional = (
                    get_origin(field_type) is Union
                    and type(None) in get_args(field_type)
                )

                if field_name not in class_defaults and not is_optional:
                    required.append(field_name)

            schema = self._ordered_object_schema(
                required=required,
                properties=properties,
                additional_properties=False,
            )

            if doc := inspect.getdoc(schema_class):
                schema["description"] = doc

            return {
                "type": "json_schema",
                "json_schema": {
                    "name": name or schema_class.__name__,
                    "strict": True,
                    "schema": schema
                }
            }
        finally:
            seen_models.discard(schema_class)

# ============================================================================
# Tool Preparator
# ============================================================================

@dataclass
class PreparedTools:
    """Result of tool preparation."""
    definitions: List[Dict[str, Any]]

class ToolPreparator:
    """Prepares tools for LLM consumption."""

    def __init__(self, schema_converter: SchemaConverter):
        self._converter = schema_converter

    def prepare(self, tools: Optional[List[Any]]) -> PreparedTools:
        """Convert callable functions to OpenAI tool format."""
        if not tools:
            return PreparedTools([])

        definitions = []

        for idx, tool in enumerate(tools):
            if callable(tool):
                definitions.append(self._prepare_callable(tool))
            elif isinstance(tool, dict):
                self._validate_tool_dict(tool, idx)
                definitions.append(tool)
            else:
                raise ConfigurationError(
                    f"Tool at index {idx} must be callable or dict, got {type(tool).__name__}"
                )

        return PreparedTools(definitions)

    def _prepare_callable(self, func: Callable) -> Dict:
        """Prepare a callable for LLM consumption."""
        underlying = self._unwrap_callable(func)

        name = (getattr(func, '__name__', None) or underlying.__name__).strip()
        doc = (getattr(underlying, '__doc__', None) or getattr(func, '__doc__', None) or "").strip()

        try:
            annotations = get_type_hints(underlying)
        except Exception:
            annotations = getattr(underlying, "__annotations__", {})

        sig = inspect.signature(func)

        parameters = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name == "return":
                continue

            param_type = annotations.get(param_name)

            if param_type is not None and not self._converter.is_llm_supported_type(param_type):
                continue

            param_schema = (
                self._converter.python_type_to_json_schema(param_type)
                if param_type else {"type": SchemaType.STRING.value}
            )

            if param.default != inspect.Parameter.empty:
                default_repr = self._format_default(param.default)
                existing = param_schema.get("description", "")
                param_schema["description"] = (
                    f"{existing} (Default: {default_repr})" if existing
                    else f"Default: {default_repr}"
                )
            else:
                required.append(param_name)

            parameters[param_name] = param_schema

        return {
            "type": "function",
            "function": {
                "name": name,
                "description": doc,
                "parameters": self._converter._ordered_object_schema(
                    required=required,
                    properties=parameters,
                    additional_properties=False,
                )
            }
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
    def _validate_tool_dict(tool: Dict, index: int) -> None:
        if "type" not in tool or "function" not in tool:
            raise ConfigurationError(
                f"Tool at index {index} must have 'type' and 'function' keys"
            )
        if "name" not in tool.get("function", {}):
            raise ConfigurationError(
                f"Tool at index {index} missing 'name' in function definition"
            )
    
def tool_result(tool_call: ToolCall, result: Any) -> ToolResultMessage:
    """Format a tool call result for LLM consumption."""
    return {
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": result
    }
    
def assistant_message(final_response: FinalResponse) -> AssistantMessage:
    """Format an assistant message with tool calls for LLM consumption."""
    tool_calls = []
    for tc in final_response.get("tool_calls", []):
        args_val = tc.get("arguments", {})
        args_str = json.dumps(args_val) if isinstance(args_val, (dict, list)) else (args_val or "{}")
        tool_calls.append({
            "id": tc.get("id", ""),
            "type": "function",
            "function": {
                "name": tc.get("name", ""),
                "arguments": args_str
            }
        })
    return {
        "role": "assistant",
        "content": final_response.get("answer"),
        "tool_calls": tool_calls if tool_calls else None
    }

def user_message(content: str | List[Dict[str, Any]]) -> UserMessage:
    """Format a user message for LLM consumption."""
    return {
        "role": "user",
        "content": content
    }

class RequestTransformer:
    """Provider/model-specific request normalizer."""

    def __init__(self, model: str, api_base: str):
        self._model = model
        self._api_base = api_base.lower()

    def transform(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        transformed = copy.deepcopy(kwargs)
        transformed = self._normalize_extra_body(transformed)
        transformed = self._normalize_reasoning(transformed)
        transformed = self._normalize_parallel_tool_calls(transformed)
        return transformed

    def _normalize_extra_body(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        extra_body = kwargs.get("extra_body")
        if extra_body is None:
            return kwargs
        if not isinstance(extra_body, dict):
            kwargs["extra_body"] = {"value": extra_body}
            return kwargs
        if not extra_body:
            kwargs.pop("extra_body", None)
        return kwargs

    def _normalize_reasoning(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        effort = kwargs.pop("reasoning_effort", None)
        if effort is None:
            return kwargs

        # OpenRouter-style providers usually accept reasoning controls inside extra_body.
        if "openrouter" in self._api_base:
            extra_body = kwargs.setdefault("extra_body", {})
            reasoning = extra_body.get("reasoning")
            if isinstance(reasoning, dict):
                reasoning.setdefault("effort", effort)
            else:
                extra_body["reasoning"] = {"effort": effort}
            return kwargs

        kwargs["reasoning_effort"] = effort
        return kwargs

    def _normalize_parallel_tool_calls(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if not kwargs.get("tools"):
            return kwargs
        model_name = self._model.lower()
        if "gpt-5" in model_name or "gpt-4.1" in model_name:
            kwargs.setdefault("parallel_tool_calls", True)
        return kwargs

# ============================================================================
# Image Processor
# ============================================================================

class ImageProcessor:
    """Processes images in messages for API consumption."""

    _pil_image = None

    @classmethod
    def _get_pil(cls):
        if cls._pil_image is None:
            try:
                from PIL import Image
                cls._pil_image = Image
            except ImportError:
                raise ImportError("PIL/Pillow required. Install with: pip install Pillow")
        return cls._pil_image

    @staticmethod
    def process_messages(messages: List[Dict]) -> None:
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for i, item in enumerate(content):
                if not isinstance(item, dict) or item.get("type") != "image":
                    continue
                msg["content"][i] = ImageProcessor._convert_image_item(item)

    @staticmethod
    def _convert_image_item(item: Dict) -> Dict:
        if "image_path" in item:
            return ImageProcessor._from_path(item["image_path"])
        if "image_pil" in item:
            return ImageProcessor._from_pil(item["image_pil"])
        if "image_url" in item:
            return ImageProcessor._from_url(item["image_url"])
        if "image_base64" in item:
            return ImageProcessor._from_base64(item["image_base64"])
        return item

    @staticmethod
    def _from_path(path: str) -> Dict:
        Image = ImageProcessor._get_pil()
        try:
            with Image.open(path) as img:
                return ImageProcessor._encode_pil_image(img)
        except Exception as e:
            raise ValueError(f"Failed to process image from path '{path}': {e}")

    @staticmethod
    def _from_pil(img: "PILImage") -> Dict:
        return ImageProcessor._encode_pil_image(img)

    @staticmethod
    def _from_url(url_data: Union[str, Dict]) -> Dict:
        if isinstance(url_data, str):
            url_data = {"url": url_data}
        return {"type": "image_url", "image_url": url_data}

    @staticmethod
    def _from_base64(data: str) -> Dict:
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{data}"}
        }

    @staticmethod
    def _encode_pil_image(img: "PILImage") -> Dict:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        }

# ============================================================================
# Event Builder
# ============================================================================

class EventBuilder:
    """Builds standardized stream events."""

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
    def tool_call(content: ToolCall) -> StreamEvent:
        return EventBuilder._build(EventType.TOOL_CALL, content)

    @staticmethod
    def tool_call_part(content: Dict[str, str]) -> StreamEvent:
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
class ToolCallStreamHandler:
    """Accumulates tool calls AND emits incremental tool_call_part events."""

    def __init__(self, event_builder: EventBuilder, tools_dict: Dict[str, Callable] = {}):
        self._event_builder = event_builder
        self._tools_dict = tools_dict
        self._pending: Dict[int, Dict[str, Any]] = {}
        self._active_index: Optional[int] = None
        self._emitted_indices: Set[int] = set()

    def process_chunk(self, tool_calls: Optional[List[Any]]) -> List[StreamEvent]:
        events: List[StreamEvent] = []

        if not tool_calls:
            if self._active_index is not None:
                if self._active_index not in self._emitted_indices:
                    if event := self._emit_complete_tool_call(self._active_index):
                        events.append(event)
                self._active_index = None
            return events

        for tc in tool_calls:
            idx = getattr(tc, "index", 0)

            if self._active_index is not None and idx != self._active_index:
                if self._active_index not in self._emitted_indices:
                    if event := self._emit_complete_tool_call(self._active_index):
                        events.append(event)

            self._active_index = idx

            if idx not in self._pending:
                self._pending[idx] = {
                    "id": "",
                    "name": "",
                    "arguments": "",
                    "buffer": "",
                }

            p = self._pending[idx]

            if tid := getattr(tc, "id", None):
                p["id"] = tid

            func = getattr(tc, "function", None)
            if func is not None:
                if func.name:
                    p["name"] = func.name
                if func.arguments:
                    chunk = func.arguments
                    if isinstance(chunk, dict):
                        chunk = json.dumps(chunk)
                    p["arguments"] += chunk
                    p["buffer"] += chunk

            if p["name"] and p["id"] and p["buffer"]:
                events.append(
                    self._event_builder.tool_call_part(
                        content={
                            "id": p["id"],
                            "name": p["name"],
                            "args_delta": p["buffer"],
                        }
                    )
                )
                p["buffer"] = ""

        return events

    def _emit_complete_tool_call(self, idx: int) -> Optional[StreamEvent]:
        if idx not in self._pending:
            return None
        p = self._pending[idx]
        if not p["id"] or not p["name"]:
            return None

        try:
            args = json.loads(p["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {"_raw": p["arguments"] or ""}

        self._emitted_indices.add(idx)
        return self._event_builder.tool_call({
            "id": p["id"],
            "name": p["name"],
            "arguments": args,
            "callable": self._tools_dict.get(p["name"]) or None,
        })

    def finalize(self) -> List[ToolCall]:
        new_calls: List[ToolCall] = []
        if self._active_index is not None and self._active_index not in self._emitted_indices:
            event = self._emit_complete_tool_call(self._active_index)
            if event:
                content = event.get("content")
                if content is not None:
                    new_calls.append(content)
        self._active_index = None
        return new_calls

    def get_all_calls(self) -> List[ToolCall]:
        result: List[ToolCall] = []
        for idx in sorted(self._pending.keys()):
            p = self._pending[idx]
            if not p["id"] or not p["name"]:
                continue

            try:
                args = json.loads(p["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {"_raw": p["arguments"] or ""}

            result.append({
                "id": p["id"],
                "name": p["name"],
                "arguments": args,
                "callable": self._tools_dict.get(p["name"]) or None,
            })
        return result

    def clear(self) -> None:
        self._pending.clear()
        self._active_index = None
        self._emitted_indices.clear()

# ============================================================================
# Main LLM Class
# ============================================================================

class LLM:
    """Universal API wrapper for LLM models with OpenAI-compatible API.

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
        custom_thinking_token: Optional[CustomThinkingToken] = None,
        default_stop_sequences: Optional[List[str]] = None,
        timeout: float = DEFAULT_TIMEOUT,
        extra_body: Optional[Dict[str, Any]] = None,
        use_responses_api: bool = False,
        default_headers: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
    ):
        self._config = LLMConfig(
            model=model,
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            custom_thinking_token=custom_thinking_token,
            default_stop_sequences=default_stop_sequences,
            timeout=timeout,
            extra_body=extra_body,
            use_responses_api=use_responses_api,
            default_headers=default_headers,
            max_retries=max_retries,
        )

        self._api_base = self._compute_api_base()

        self._client = OpenAI(
            base_url=self._api_base,
            api_key=api_key,
            timeout=self._config.timeout,
            default_headers=default_headers,
            max_retries=self._config.max_retries,
        )
        self._async_client = AsyncOpenAI(
            base_url=self._api_base,
            api_key=api_key,
            timeout=self._config.timeout,
            default_headers=default_headers,
            max_retries=self._config.max_retries,
        )

        self._schema_converter = SchemaConverter()
        self._tool_preparator = ToolPreparator(self._schema_converter)
        self._event_builder = EventBuilder()
        self._request_transformer = RequestTransformer(model, self._api_base)

        logger.debug(f"LLM initialized: model={model}, base_url={self._api_base}")

    def _compute_api_base(self) -> str:
        base = self._config.base_url.rstrip("/")
        if re.search(r"/v\d+$", base):
            return base
        return f"{base}/v1"

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def base_url(self) -> str:
        return self._config.base_url

    def list_models(self, fallback: Optional[List[str]] = None, max_retries: Optional[int] = None,) -> List[str]:
        """Return model IDs from the configured API, or fallback/[] on failure."""
        try:
            client = self._client if max_retries is None else self._client.with_options(max_retries=max_retries)
            return sorted({model.id for model in client.models.list().data})
        except Exception:
            return list(fallback or [])
        
    async def async_list_models(
        self,
        fallback: Optional[List[str]] = None,
        max_retries: Optional[int] = None,
    ) -> List[str]:
        """Async return model IDs from the configured API, or fallback/[] on failure."""
        try:
            client = self._async_client if max_retries is None else self._async_client.with_options(max_retries=max_retries)
            models = await client.models.list()
            return sorted({model.id for model in models.data})
        except Exception:
            return list(fallback or [])

    # ========================================================================
    # Output Format Handling
    # ========================================================================

    def _prepare_output_format(self, output_format: Union[Dict, type, None]) -> Optional[Dict]:
        if output_format is None:
            return None
        if isinstance(output_format, dict):
            return output_format
        if isinstance(output_format, type):
            return self._schema_converter.convert_class_to_schema(output_format)
        raise ConfigurationError(
            f"output_format must be dict, type, or None, got {type(output_format).__name__}"
        )

    # ========================================================================
    # Request Builder
    # ========================================================================

    def _build_request(
        self,
        messages: List[Dict],
        output_format: Optional[Dict],
        tools: Optional[List],
        reasoning_effort: Optional[str],
        max_tokens: Optional[int],
        extra_body: Optional[Dict],
    ) -> tuple[Dict[str, Any], PreparedTools, bool]:
        """Build API request kwargs. Returns (kwargs, prepared_tools, structured_output)."""
        request_messages = copy.deepcopy(messages)
        prepared_tools = self._tool_preparator.prepare(tools)
        ImageProcessor.process_messages(request_messages)
        structured_output = output_format is not None

        kwargs: Dict[str, Any] = {
            "model": self._config.model,
            "messages": request_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if prepared_tools.definitions:
            kwargs["tools"] = prepared_tools.definitions
        merged_extra_body = _deep_merge_dicts(self._config.extra_body, extra_body)
        if merged_extra_body:
            kwargs["extra_body"] = merged_extra_body
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if self._config.default_stop_sequences:
            kwargs["stop"] = list(self._config.default_stop_sequences)
        if structured_output:
            kwargs["response_format"] = output_format

        return self._request_transformer.transform(kwargs), prepared_tools, structured_output

    def _build_tools_dict(self, tools: list) -> Dict[str, Callable]:
        result = {}
        for tool in tools:
            if isinstance(tool, dict):
                continue
            underlying = ToolPreparator._unwrap_callable(tool)
            name = getattr(tool, '__name__', None) or getattr(underlying, '__name__', None)
            if name:
                result[name] = tool
        return result

    @staticmethod
    def _extract_reasoning(delta: Any) -> str:
        """Return streamed reasoning content from supported delta fields."""
        reasoning_content = getattr(delta, "reasoning_content", None)
        if reasoning_content:
            return str(reasoning_content)
        reasoning = getattr(delta, "reasoning", None)
        return str(reasoning) if reasoning else ""

    # ========================================================================
    # Responses API – Request Builder & Helpers
    # ========================================================================

    @staticmethod
    def _translate_content_for_responses_api(content: Any) -> Any:
        """
        Translate a message content value from Chat Completions format
        to Responses API format.

        - String content: unchanged (compatible with both APIs)
        - Array items:
            {"type": "text", "text": "..."}
                → {"type": "input_text", "text": "..."}
            {"type": "image_url", "image_url": {"url": "..."}}
                → {"type": "input_image", "image_url": "<url-string>"}
        """
        if not isinstance(content, list):
            return content

        translated: List[Dict] = []
        for item in content:
            if not isinstance(item, dict):
                translated.append(item)
                continue
            t = item.get("type", "")
            if t == "text":
                translated.append({"type": "input_text", "text": item.get("text", "")})
            elif t == "image_url":
                url_data = item.get("image_url", {})
                url = url_data.get("url", "") if isinstance(url_data, dict) else str(url_data)
                translated.append({"type": "input_image", "image_url": url})
            elif t == "image_base64":
                translated.append({
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{item.get('image_base64', '')}",
                })
            else:
                translated.append(item)
        return translated

    @staticmethod
    def _convert_output_format_for_responses_api(output_format: Dict) -> Dict:
        """
        Convert from Chat Completions response_format to Responses API text.format.

        Input:  {"type": "json_schema", "json_schema": {"name": "...", "strict": True, "schema": {...}}}
        Output: {"type": "json_schema", "name": "...", "strict": True, "schema": {...}}
        """
        if output_format.get("type") == "json_schema":
            js = output_format.get("json_schema", {})
            return {
                "type": "json_schema",
                "name": js.get("name", "response"),
                "strict": js.get("strict", True),
                "schema": js.get("schema", {}),
            }
        return output_format

    @staticmethod
    def _convert_tools_for_responses_api(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Chat Completions function tools to Responses API function tools."""
        converted: List[Dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") != "function" or "function" not in tool:
                converted.append(tool)
                continue

            function = tool.get("function") or {}
            parameters = copy.deepcopy(function.get("parameters") or {})
            properties = parameters.get("properties")
            if isinstance(properties, dict):
                parameters.setdefault("additionalProperties", False)
            converted.append({
                "type": "function",
                "name": function.get("name", ""),
                "description": function.get("description") or "",
                "parameters": parameters,
                "strict": function.get("strict", True),
            })
        return converted

    @staticmethod
    def _read_usage(usage: Any) -> Dict[str, Optional[int]]:
        if not usage:
            return {}

        def first_present(*names: str) -> Optional[int]:
            for name in names:
                value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
                if value is not None:
                    return int(value)
            return None

        return {
            "prompt_tokens": first_present("input_tokens", "prompt_tokens"),
            "completion_tokens": first_present("output_tokens", "completion_tokens"),
            "total_tokens": first_present("total_tokens"),
        }

    @staticmethod
    def _new_responses_state() -> Dict[str, Any]:
        return {
            "thinking": "",
            "answer": "",
            "tokens": 0,
            "api_usage": {},
            "pending_tool_items": {},
            "pending_tool_indexes": {},
            "completed_tool_calls": [],
        }

    def _handle_responses_event(
        self,
        event: Any,
        state: Dict[str, Any],
        structured_output: bool,
        hide_thinking: bool,
        tools_dict: Dict[str,Callable],
    ) -> List[StreamEvent]:
        state["tokens"] += 1
        emitted: List[StreamEvent] = []
        etype = getattr(event, "type", "")

        if etype == "response.completed":
            response = getattr(event, "response", None)
            state["api_usage"] = self._read_usage(getattr(response, "usage", None))
            return emitted

        if etype == "response.output_text.delta":
            chunk = getattr(event, "delta", "") or ""
            if chunk:
                state["answer"] += chunk
                if not structured_output:
                    emitted.append(self._event_builder.answer(chunk))
            return emitted

        if etype in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            chunk = getattr(event, "delta", "") or ""
            if chunk:
                state["thinking"] += chunk
                if not hide_thinking:
                    emitted.append(self._event_builder.reasoning(chunk))
            return emitted

        if etype == "response.output_item.added":
            item = getattr(event, "item", None)
            if item and getattr(item, "type", "") == "function_call":
                output_index = getattr(event, "output_index", None)
                pending = {
                    "call_id": getattr(item, "call_id", None) or getattr(item, "id", ""),
                    "name": getattr(item, "name", "") or "",
                    "arguments": "",
                }
                state["pending_tool_items"][getattr(item, "id", "")] = pending
                if output_index is not None:
                    state["pending_tool_indexes"][int(output_index)] = pending
            return emitted

        if etype == "response.output_item.done":
            item = getattr(event, "item", None)
            if item and getattr(item, "type", "") == "function_call":
                output_index = getattr(event, "output_index", None)
                pending = state["pending_tool_items"].setdefault(getattr(item, "id", ""), {})
                pending["call_id"] = (
                    getattr(item, "call_id", None)
                    or pending.get("call_id", "")
                    or getattr(item, "id", "")
                )
                pending["name"] = getattr(item, "name", "") or pending.get("name", "")
                pending.setdefault("arguments", "")
                if output_index is not None:
                    state["pending_tool_indexes"][int(output_index)] = pending
            return emitted
        
        if etype == "response.function_call_arguments.delta":
            item_id = getattr(event, "item_id", None)
            output_index = getattr(event, "output_index", None)
            delta = getattr(event, "delta", "") or ""

            if item_id and item_id in state["pending_tool_items"]:
                state["pending_tool_items"][item_id].setdefault("arguments", "")
                state["pending_tool_items"][item_id]["arguments"] += delta
            elif output_index is not None and int(output_index) in state["pending_tool_indexes"]:
                p = state["pending_tool_indexes"][int(output_index)]
                p.setdefault("arguments", "")
                p["arguments"] += delta

            pending = None
            if item_id and item_id in state["pending_tool_items"]:
                pending = state["pending_tool_items"][item_id]
            elif output_index is not None:
                pending = state["pending_tool_indexes"].get(int(output_index))

            if pending and pending.get("call_id") and pending.get("name") and delta:
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

        if etype == "response.function_call_arguments.done":
            item_id = getattr(event, "item_id", None)
            output_index = getattr(event, "output_index", None)
            
            args_str = getattr(event, "arguments", None)
            
            pending = state["pending_tool_items"].pop(item_id, {}) if item_id else {}
            if not pending and output_index is not None:
                pending = state["pending_tool_indexes"].pop(int(output_index), {})

            if args_str is None:
                args_str = pending.get("arguments", "{}") or "{}"
            
            fn_name = getattr(event, "name", "") or pending.get("name", "")

            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {"_raw": args_str}

            tc: ToolCall = {
                "id": pending.get("call_id", item_id or ""),
                "name": fn_name,
                "arguments": args,
                "callable": tools_dict.get(fn_name) or None,
            }
            state["completed_tool_calls"].append(tc)
            emitted.append(self._event_builder.tool_call(tc))
            return emitted

        return emitted

    def _finalize_responses_stream(
        self,
        state: Dict[str, Any],
        structured_output: bool,
        verbose: bool,
        final: bool,
        hide_thinking: bool,
        latency: Optional[float],
        elapsed: float,
    ) -> Generator[StreamEvent, None, None]:
        answer = state["answer"]
        if structured_output:
            try:
                answer = json.loads(answer)
            except json.JSONDecodeError:
                pass
            yield self._event_builder.answer(answer)

        completed_tool_calls = state["completed_tool_calls"]

        api_usage = state["api_usage"]
        stream_tokens = int(state["tokens"])
        completion_tokens = api_usage.get("completion_tokens")
        tokens = int(completion_tokens) if completion_tokens is not None else stream_tokens
        total_tokens = api_usage.get("total_tokens")
        prompt_tokens = api_usage.get("prompt_tokens")
        if total_tokens is None and prompt_tokens is not None:
            total_tokens = int(prompt_tokens) + tokens

        verbose_info: VerboseInfo = {
            "tokens": tokens,
            "tokens_per_second": tokens / elapsed if elapsed > 0 else 0,
            "latency": latency,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens if completion_tokens is not None else stream_tokens,
            "total_tokens": total_tokens,
        }

        if verbose:
            yield self._event_builder.verbose(verbose_info)

        if final:
            final_response: FinalResponse = {
                "answer": answer.strip() if isinstance(answer, str) else answer
            }
            thinking = state["thinking"]
            if not hide_thinking and thinking.strip():
                final_response["reasoning"] = thinking.strip()
            if completed_tool_calls:
                final_response["tool_calls"] = completed_tool_calls
            if verbose:
                final_response["verbose"] = verbose_info
            yield self._event_builder.final(final_response)

        yield self._event_builder.done()

    def _build_responses_request(
        self,
        messages: List[Dict],
        output_format: Optional[Dict],
        tools: Optional[List],
        reasoning_effort: Optional[str],
        max_tokens: Optional[int],
        extra_body: Optional[Dict],
    ) -> tuple[Dict[str, Any], PreparedTools, bool]:
        """
        Build a request payload for the Responses API (/v1/responses).
        Returns (kwargs, prepared_tools, structured_output).
        """
        # Deep-copy and run image processing (same as Chat Completions path)
        raw_messages = copy.deepcopy(messages)
        ImageProcessor.process_messages(raw_messages)

        # Translate message content format
        input_messages: List[Dict] = []
        for msg in raw_messages:
            translated = dict(msg)
            translated["content"] = self._translate_content_for_responses_api(msg.get("content"))
            input_messages.append(translated)

        prepared_tools = self._tool_preparator.prepare(tools)
        structured_output = output_format is not None

        kwargs: Dict[str, Any] = {
            "model": self._config.model,
            "input": input_messages,
            "stream": True,
        }

        if prepared_tools.definitions:
            kwargs["tools"] = self._convert_tools_for_responses_api(prepared_tools.definitions)
            model_name = self._config.model.lower()
            if "gpt-5" in model_name or "gpt-4.1" in model_name:
                kwargs.setdefault("parallel_tool_calls", True)

        # Reasoning effort has a different key in the Responses API
        if reasoning_effort:
            kwargs["reasoning"] = {"effort": reasoning_effort}

        # max_output_tokens replaces max_tokens
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens

        # Structured output via text.format instead of response_format
        if structured_output:
            kwargs["text"] = {"format": self._convert_output_format_for_responses_api(output_format)}

        merged_extra_body = _deep_merge_dicts(self._config.extra_body, extra_body)
        if merged_extra_body:
            kwargs["extra_body"] = merged_extra_body

        return kwargs, prepared_tools, structured_output

    def _stream_responses_sync(
        self,
        messages: List[Dict],
        output_format: Optional[Dict],
        tools: Optional[List],
        reasoning_effort: Optional[str],
        max_tokens: Optional[int],
        verbose: bool,
        hide_thinking: bool,
        final: bool,
        extra_body: Optional[Dict],
        max_retries: Optional[int] = None,
    ) -> Generator[StreamEvent, None, None]:
        """
        Sync streaming via the Responses API (/v1/responses).
        Yields the same StreamEvent format as the Chat Completions path.
        """
        kwargs, _, structured_output = self._build_responses_request(
            messages, output_format, tools, reasoning_effort, max_tokens, extra_body
        )
        tools_dict = self._build_tools_dict(tools or [])

        state = self._new_responses_state()
        start_time = time.perf_counter()
        latency: Optional[float] = None

        client = self._client if max_retries is None else self._client.with_options(max_retries=max_retries)

        try:
            stream = client.responses.create(**kwargs)
        except Exception as e:
            raise ModelRequestError(f"Responses API request failed: {e}")

        try:
            for event in stream:
                if latency is None:
                    latency = time.perf_counter() - start_time
                for emitted in self._handle_responses_event(
                    event, state, structured_output, hide_thinking,tools_dict
                ):
                    yield emitted

        except Exception as e:
            raise ModelRequestError(f"Responses API stream failed: {e}") from e

        elapsed = time.perf_counter() - start_time
        yield from self._finalize_responses_stream(
            state, structured_output, verbose, final, hide_thinking, latency, elapsed
        )

    async def _stream_responses_async(
        self,
        messages: List[Dict],
        output_format: Optional[Dict],
        tools: Optional[List],
        reasoning_effort: Optional[str],
        max_tokens: Optional[int],
        verbose: bool,
        hide_thinking: bool,
        final: bool,
        extra_body: Optional[Dict],
        max_retries: Optional[int] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Async streaming via the Responses API (/v1/responses).
        Yields the same StreamEvent format as the Chat Completions path.
        """
        kwargs, _, structured_output = self._build_responses_request(
            messages, output_format, tools, reasoning_effort, max_tokens, extra_body
        )
        tools_dict = self._build_tools_dict(tools or [])

        state = self._new_responses_state()
        start_time = time.perf_counter()
        latency: Optional[float] = None

        client = self._async_client if max_retries is None else self._async_client.with_options(max_retries=max_retries)

        async def _acreate(c):
            call = c.responses.create(**kwargs)
            return await call if asyncio.iscoroutine(call) else call

        try:
            stream = await _acreate(client)
        except Exception as e:
            raise ModelRequestError(f"Async Responses API request failed: {e}")

        try:
            async for event in stream:
                if latency is None:
                    latency = time.perf_counter() - start_time
                for emitted in self._handle_responses_event(
                    event, state, structured_output, hide_thinking,tools_dict
                ):
                    yield emitted

        except Exception as e:
            raise ModelRequestError(f"Async Responses API stream failed: {e}") from e

        elapsed = time.perf_counter() - start_time
        for emitted in self._finalize_responses_stream(
            state, structured_output, verbose, final, hide_thinking, latency, elapsed
        ):
            yield emitted

    def response(
        self,
        messages: Optional[List[Dict]] = None,
        output_format: Union[Dict, type, None] = None,
        tools: Optional[List] = None,
        verbose: bool = False,
        hide_thinking: bool = True,
        reasoning_effort: Optional[str] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict] = None,
        input: Optional[str] = None,
        system: Optional[str] = None,
        max_retries: Optional[int] = None,
    ) -> FinalResponse:
        """Request model inference (non-streaming)."""
        resolved_messages = _resolve_messages(messages, input,system)
        output_format = self._prepare_output_format(output_format)

        final_content = None

        for event in self.stream_response(
            messages=resolved_messages,
            output_format=output_format,
            final=True,
            tools=tools,
            hide_thinking=hide_thinking,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            verbose=verbose,
            extra_body=extra_body,
            max_retries=max_retries,
        ):
            if event.get("type") == EventType.FINAL.value:
                final_content = event.get("content")
                break

        if final_content is None:
            raise RuntimeError("No final response received")

        return final_content

    def stream_response(
        self,
        messages: Optional[List[Dict]] = None,
        output_format: Union[Dict, type, None] = None,
        final: bool = False,
        tools: Optional[List] = None,
        hide_thinking: bool = True,
        reasoning_effort: Optional[str] = None,
        max_tokens: Optional[int] = None,
        verbose: bool = False,
        extra_body: Optional[Dict] = None,
        max_retries: Optional[int] = None,
        input: Optional[str] = None,
        system: Optional[str] = None,
    ) -> Generator[StreamEvent, None, None]:
        """Request model inference with streaming."""
        resolved_messages = _resolve_messages(messages, input, system)
        output_format = self._prepare_output_format(output_format)
        if self._config.use_responses_api:
            yield from self._stream_responses_sync(
                messages=resolved_messages,
                output_format=output_format,
                tools=tools,
                reasoning_effort=reasoning_effort,
                max_tokens=max_tokens,
                verbose=verbose,
                hide_thinking=hide_thinking,
                final=final,
                extra_body=extra_body,
                max_retries=max_retries,
            )
            return

        kwargs, _, structured_output = self._build_request(
            resolved_messages, output_format, tools, reasoning_effort, max_tokens, extra_body
        )
        
        thinking_parser = ThinkingParser(self._config.custom_thinking_token)
        tool_handler = ToolCallStreamHandler(self._event_builder,self._build_tools_dict(tools or []))

        thinking = ""
        answer = ""
        start_time = time.perf_counter()
        latency: Optional[float] = None
        tokens = 0

        client = self._client if max_retries is None else self._client.with_options(max_retries=max_retries)

        def _create(c):
            try:
                return c.chat.completions.create(**kwargs)
            except APIStatusError as e:
                if "stream_options" in kwargs and e.status_code in (400, 422):
                    kwargs_copy = dict(kwargs)
                    kwargs_copy.pop("stream_options", None)
                    return c.chat.completions.create(**kwargs_copy)
                raise

        try:
            completion = _create(client)
        except Exception as e:
            raise ModelRequestError(f"Model request failed: {e}") from e

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        try:
            for chunk in completion:
                if latency is None:
                    latency = time.perf_counter() - start_time

                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    if getattr(usage, "prompt_tokens", None) is not None:
                        prompt_tokens = int(usage.prompt_tokens)
                    if getattr(usage, "completion_tokens", None) is not None:
                        completion_tokens = int(usage.completion_tokens)
                    if getattr(usage, "total_tokens", None) is not None:
                        total_tokens = int(usage.total_tokens)

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                if not delta:
                    continue

                tokens += 1

                if reasoning := self._extract_reasoning(delta):
                    thinking += reasoning
                    if not hide_thinking:
                        yield self._event_builder.reasoning(reasoning)

                if content := getattr(delta, "content", None):
                    thinking_part, answer_part = thinking_parser.parse(str(content))

                    if thinking_part:
                        thinking += thinking_part
                        if not hide_thinking:
                            yield self._event_builder.reasoning(thinking_part)

                    if answer_part:
                        answer += answer_part
                        if not structured_output:
                            yield self._event_builder.answer(answer_part)

                tool_calls = getattr(delta, "tool_calls", None)
                for event in tool_handler.process_chunk(tool_calls):
                    yield event
        except Exception as e:
            raise ModelRequestError(f"Model stream failed: {e}") from e

        elapsed = time.perf_counter() - start_time
        if completion_tokens > 0:
            tokens = completion_tokens
        tokens_per_second = tokens / elapsed if elapsed > 0 else 0

        if structured_output:
            try:
                answer = json.loads(answer)
            except json.JSONDecodeError:
                pass
            yield self._event_builder.answer(answer)

        final_tool_calls = tool_handler.finalize()
        for tc in final_tool_calls:
            yield self._event_builder.tool_call(tc)

        verbose_info: VerboseInfo = {
            "tokens": tokens,
            "tokens_per_second": tokens_per_second,
            "latency": latency,
            "prompt_tokens": prompt_tokens if prompt_tokens > 0 else None,
            "completion_tokens": completion_tokens if completion_tokens > 0 else None,
            "total_tokens": total_tokens if total_tokens > 0 else None,
        }

        if verbose:
            yield self._event_builder.verbose(verbose_info)

        if final:
            final_response: FinalResponse = {
                "answer": answer.strip() if isinstance(answer, str) else answer
            }
            if not hide_thinking and thinking.strip():
                final_response["reasoning"] = thinking.strip()
            all_completed_calls = tool_handler.get_all_calls()
            if all_completed_calls:
                final_response["tool_calls"] = all_completed_calls
            if verbose:
                final_response["verbose"] = verbose_info

            yield self._event_builder.final(final_response)

        yield self._event_builder.done()

    # ========================================================================
    # Asynchronous Methods
    # ========================================================================

    async def async_response(
        self,
        messages: Optional[List[Dict]] = None,
        output_format: Union[Dict, type, None] = None,
        tools: Optional[List] = None,
        verbose: bool = False,
        hide_thinking: bool = True,
        reasoning_effort: Optional[str] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict] = None,
        input: Optional[str] = None,
        system: Optional[str] = None,
        max_retries: Optional[int] = None,
    ) -> FinalResponse:
        """Async request for model inference."""
        resolved_messages = _resolve_messages(messages, input, system)
        output_format = self._prepare_output_format(output_format)

        final_content = None
        async for event in self.async_stream_response(
            messages=resolved_messages,
            output_format=output_format,
            final=True,
            tools=tools,
            hide_thinking=hide_thinking,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            verbose=verbose,
            extra_body=extra_body,
            max_retries=max_retries,
        ):
            if event.get("type") == EventType.FINAL.value:
                final_content = event.get("content")
                break

        if final_content is None:
            raise RuntimeError("No final response received")

        return final_content

    async def async_stream_response(
        self,
        messages: Optional[List[Dict]] = None,
        output_format: Union[Dict, type, None] = None,
        final: bool = False,
        tools: Optional[List] = None,
        verbose: bool = False,
        hide_thinking: bool = True,
        reasoning_effort: Optional[str] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict] = None,
        input: Optional[str] = None,
        system: Optional[str] = None,
        max_retries: Optional[int] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Async streaming model inference."""

        resolved_messages = _resolve_messages(messages, input, system)
        output_format = self._prepare_output_format(output_format)
        if self._config.use_responses_api:
            async for event in self._stream_responses_async(
                messages=resolved_messages,
                output_format=output_format,
                tools=tools,
                reasoning_effort=reasoning_effort,
                max_tokens=max_tokens,
                verbose=verbose,
                hide_thinking=hide_thinking,
                final=final,
                extra_body=extra_body,
                max_retries=max_retries,
            ):
                yield event
            return

        kwargs, _, structured_output = self._build_request(
            resolved_messages, output_format, tools, reasoning_effort, max_tokens, extra_body
        )

        thinking_parser = ThinkingParser(self._config.custom_thinking_token)
        tool_handler = ToolCallStreamHandler(self._event_builder, self._build_tools_dict(tools or []))

        thinking = ""
        answer = ""
        start_time = time.perf_counter()
        latency: Optional[float] = None
        tokens = 0

        client = self._async_client if max_retries is None else self._async_client.with_options(max_retries=max_retries)

        async def _acreate(c):
            try:
                call = c.chat.completions.create(**kwargs)
                return await call if asyncio.iscoroutine(call) else call
            except APIStatusError as e:
                if "stream_options" in kwargs and e.status_code in (400, 422):
                    kwargs_copy = dict(kwargs)
                    kwargs_copy.pop("stream_options", None)
                    call = c.chat.completions.create(**kwargs_copy)
                    return await call if asyncio.iscoroutine(call) else call
                raise

        try:
            completion = await _acreate(client)
        except Exception as e:
            raise ModelRequestError(f"Async model request failed: {e}") from e

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0

        try:
            async for chunk in completion:
                if latency is None:
                    latency = time.perf_counter() - start_time

                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    if getattr(usage, "prompt_tokens", None) is not None:
                        prompt_tokens = int(usage.prompt_tokens)
                    if getattr(usage, "completion_tokens", None) is not None:
                        completion_tokens = int(usage.completion_tokens)
                    if getattr(usage, "total_tokens", None) is not None:
                        total_tokens = int(usage.total_tokens)

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                if not delta:
                    continue

                tokens += 1

                if reasoning := self._extract_reasoning(delta):
                    thinking += reasoning
                    if not hide_thinking:
                        yield self._event_builder.reasoning(reasoning)

                if content := getattr(delta, "content", None):
                    thinking_part, answer_part = thinking_parser.parse(str(content))

                    if thinking_part:
                        thinking += thinking_part
                        if not hide_thinking:
                            yield self._event_builder.reasoning(thinking_part)

                    if answer_part:
                        answer += answer_part
                        if not structured_output:
                            yield self._event_builder.answer(answer_part)

                tool_calls = getattr(delta, "tool_calls", None)
                for event in tool_handler.process_chunk(tool_calls):
                    yield event
        except Exception as e:
            raise ModelRequestError(f"Async model stream failed: {e}") from e

        elapsed = time.perf_counter() - start_time
        if completion_tokens > 0:
            tokens = completion_tokens
        tokens_per_second = tokens / elapsed if elapsed > 0 else 0

        if structured_output:
            try:
                answer = json.loads(answer)
            except json.JSONDecodeError:
                pass
            yield self._event_builder.answer(answer)

        final_tool_calls = tool_handler.finalize()
        for tc in final_tool_calls:
            yield self._event_builder.tool_call(tc)

        verbose_info: VerboseInfo = {
            "tokens": tokens,
            "tokens_per_second": tokens_per_second,
            "latency": latency,
            "prompt_tokens": prompt_tokens if prompt_tokens > 0 else None,
            "completion_tokens": completion_tokens if completion_tokens > 0 else None,
            "total_tokens": total_tokens if total_tokens > 0 else None,
        }

        if verbose:
            yield self._event_builder.verbose(verbose_info)

        if final:
            final_response: FinalResponse = {
                "answer": answer.strip() if isinstance(answer, str) else answer
            }
            if not hide_thinking and thinking.strip():
                final_response["reasoning"] = thinking.strip()
            
            all_completed_calls = tool_handler.get_all_calls()
            if all_completed_calls:
                final_response["tool_calls"] = all_completed_calls
            if verbose:
                final_response["verbose"] = verbose_info

            yield self._event_builder.final(final_response)

        yield self._event_builder.done()

    # ========================================================================
    # Context Manager
    # ========================================================================

    def close(self) -> None:
        if hasattr(self._client, "close"):
            self._client.close()
        _close_async_resource(self._async_client)

    async def aclose(self) -> None:
        if hasattr(self._client, "close"):
            self._client.close()
        close_result = self._async_client.close()
        if inspect.isawaitable(close_result):
            await close_result

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
    
def list_models(
    fallback: Optional[List[str]] = None,
    max_retries: int = 3,
    client: Optional[LLM] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> List[str]:
    """Return model IDs from the configured API, or fallback/[] on failure."""
    a_k = ""
    b_u = ""
    manually = base_url is not None or api_key is not None
    if client and not manually:
        a_k = client._client.api_key
        b_u = client._api_base
    elif manually and not client:
        if not (base_url and api_key):
            raise ValueError("If client is not provided, both api_key and base_url must be provided.")
        a_k = api_key
        b_u = base_url
    elif not manually and not client:
        raise ValueError("Either a client must be provided, or both api_key and base_url must be provided.")
    elif client and manually:
        if base_url is not None and base_url != client._api_base:
            raise ValueError("If client is provided, base_url must match the client's configuration or be None.")
        if api_key is not None and api_key != client._client.api_key:
            raise ValueError("If client is provided, api_key must match the client's configuration or be None.")
        a_k = client._client.api_key
        b_u = client._api_base
    try:
        c = OpenAI(api_key=a_k, base_url=b_u,max_retries=max_retries)
        models = c.models.list()
        return sorted({model.id for model in models.data}) or list(fallback or [])
    except Exception:
        return list(fallback or [])
        
        
async def async_list_models(
    fallback: Optional[List[str]] = None,
    max_retries: int = 3,
    client: Optional[LLM] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> List[str]:
    """Async version of list_models."""
    a_k = ""
    b_u = ""
    manually = base_url is not None or api_key is not None
    if client and not manually:
        a_k = client._client.api_key
        b_u = client._api_base
    elif manually and not client:
        if not (base_url and api_key):
            raise ValueError("If client is not provided, both api_key and base_url must be provided.")
        a_k = api_key
        b_u = base_url
    elif not manually and not client:
        raise ValueError("Either a client must be provided, or both api_key and base_url must be provided.")
    elif client and manually:
        if base_url is not None and base_url != client._api_base:
            raise ValueError("If client is provided, base_url must match the client's configuration or be None.")
        if api_key is not None and api_key != client._client.api_key:
            raise ValueError("If client is provided, api_key must match the client's configuration or be None.")
        a_k = client._client.api_key
        b_u = client._api_base
    try:
        c = AsyncOpenAI(api_key=a_k, base_url=b_u,max_retries=max_retries)
        models = await c.models.list()
        return sorted({model.id for model in models.data}) or list(fallback or [])
    except Exception:
        return list(fallback or [])
        
        

# ============================================================================
# Public API
# ============================================================================

__all__ = [
    "LLM",
    "LLMConfig",
    "CustomThinkingToken",
    "StreamEvent",
    "ToolCall",
    "ToolCallMessage",
    "ToolResultMessage",
    "UserMessage",
    "AssistantMessage",
    "assistant_message",
    "user_message",
    "tool_result",
    "FinalResponse",
    "VerboseInfo",
    "EventType",
    "LLMError",
    "ConfigurationError",
    "SchemaConversionError",
    "ModelRequestError",
    "list_models",
    "async_list_models",
]
