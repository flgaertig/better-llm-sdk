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
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any, AsyncGenerator, Dict, Optional, List, Generator,
    Callable, Union, get_type_hints, get_origin, get_args,
    Literal, TypedDict, TypeVar, Final, ClassVar,
    TYPE_CHECKING
)

from openai import OpenAI, AsyncOpenAI

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


def _close_async_resource(resource: Any) -> None:
    if not hasattr(resource, "close"):
        return

    async def _close() -> None:
        await resource.close()

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

# ============================================================================
# Enums
# ============================================================================

class EventType(str, Enum):
    """Event types emitted during streaming."""
    ANSWER = "answer"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
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
    """Unified event format for all stream events."""
    type: str
    content: Any
    source: Optional[str]
    tool_id: Optional[str]
    job: Optional[int]
    depth: int


class ToolCall(TypedDict):
    """Typed dictionary for tool calls."""
    id: str
    name: str
    arguments: Dict[str, Any]


class VerboseInfo(TypedDict):
    """Typed dictionary for verbose information."""
    tokens: int
    tokens_per_second: float
    latency: Optional[float]


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
        underlying = func
        while hasattr(underlying, 'func'):
            underlying = underlying.func

        name = (getattr(func, '__name__', None) or underlying.__name__).strip()
        doc = (getattr(func, '__doc__', None) or underlying.__doc__ or "").strip()

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
    def _build(
        event_type: EventType,
        content: Any,
        source: Optional[str] = None,
        tool_id: Optional[str] = None,
        job: Optional[int] = None,
        depth: int = 0
    ) -> StreamEvent:
        event: StreamEvent = {
            "type": event_type.value,
            "content": content,
            "source": source,
            "depth": depth,
        }
        if tool_id is not None:
            event["tool_id"] = tool_id
        if job is not None:
            event["job"] = job
        return event

    @staticmethod
    def answer(content: Any, depth: int = 0) -> StreamEvent:
        return EventBuilder._build(EventType.ANSWER, content, depth=depth)

    @staticmethod
    def reasoning(content: str, depth: int = 0) -> StreamEvent:
        return EventBuilder._build(EventType.REASONING, content, depth=depth)

    @staticmethod
    def tool_call(content: ToolCall, source: Optional[str] = None, job: Optional[int] = None, depth: int = 0) -> StreamEvent:
        return EventBuilder._build(EventType.TOOL_CALL, content, source, content.get("id"), job, depth)

    @staticmethod
    def verbose(content: VerboseInfo, depth: int = 0) -> StreamEvent:
        return EventBuilder._build(EventType.VERBOSE, content, depth=depth)

    @staticmethod
    def final(content: FinalResponse, depth: int = 0) -> StreamEvent:
        return EventBuilder._build(EventType.FINAL, content, depth=depth)

    @staticmethod
    def done(depth: int = 0) -> StreamEvent:
        return EventBuilder._build(EventType.DONE, None, depth=depth)

# ============================================================================
# Tool Call Accumulator
# ============================================================================

class ToolCallAccumulator:
    """Accumulates streaming tool call chunks into complete calls."""

    def __init__(self):
        self._calls: Dict[str, Dict[str, str]] = {}
        self._index_to_id: Dict[int, str] = {}

    def add_chunk(self, tool_call: Any) -> None:
        idx = getattr(tool_call, "index", 0)

        if tool_id := getattr(tool_call, "id", None):
            self._index_to_id[idx] = tool_id

        tool_id = self._index_to_id.get(idx, f"_idx_{idx}")

        if tool_id not in self._calls:
            self._calls[tool_id] = {"name": "", "arguments": ""}

        func = tool_call.function
        if func.name:
            self._calls[tool_id]["name"] = func.name
        if func.arguments:
            args = func.arguments
            if isinstance(args, dict):
                args = json.dumps(args)
            self._calls[tool_id]["arguments"] += args or ""

    def get_completed_calls(self) -> List[ToolCall]:
        result: List[ToolCall] = []
        for tool_id, data in self._calls.items():
            try:
                args = json.loads(data["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {"_raw": data["arguments"] or ""}
            result.append({
                "id": tool_id,
                "name": data["name"],
                "arguments": args
            })
        return result

    def clear(self) -> None:
        self._calls.clear()
        self._index_to_id.clear()

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
    ):
        self._config = LLMConfig(
            model=model,
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            custom_thinking_token=custom_thinking_token,
            default_stop_sequences=default_stop_sequences,
            timeout=timeout,
            extra_body=extra_body,
        )

        self._api_base = self._compute_api_base()

        self._client = OpenAI(
            base_url=self._api_base,
            api_key=api_key,
            timeout=self._config.timeout,
        )
        self._async_client = AsyncOpenAI(
            base_url=self._api_base,
            api_key=api_key,
            timeout=self._config.timeout,
        )

        self._schema_converter = SchemaConverter()
        self._tool_preparator = ToolPreparator(self._schema_converter)
        self._event_builder = EventBuilder()
        self._request_transformer = RequestTransformer(model, self._api_base)

        logger.debug(f"LLM initialized: model={model}, base_url={self._api_base}")

    def _compute_api_base(self) -> str:
        base = self._config.base_url
        if not base.endswith("/v1") and "openai" not in base.lower():
            return base + "/v1"
        return base

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def base_url(self) -> str:
        return self._config.base_url

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
        }
        if prepared_tools.definitions:
            kwargs["tools"] = prepared_tools.definitions
        if extra_body:
            kwargs["extra_body"] = extra_body
        elif self._config.extra_body:
            kwargs["extra_body"] = self._config.extra_body
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if self._config.default_stop_sequences:
            kwargs["stop"] = list(self._config.default_stop_sequences)
        if structured_output:
            kwargs["response_format"] = output_format

        return self._request_transformer.transform(kwargs), prepared_tools, structured_output

    @staticmethod
    def _extract_reasoning(delta: Any) -> str:
        """Return streamed reasoning content from supported delta fields."""
        reasoning_content = getattr(delta, "reasoning_content", None)
        if reasoning_content:
            return str(reasoning_content)
        reasoning = getattr(delta, "reasoning", None)
        return str(reasoning) if reasoning else ""

    # ========================================================================
    # Synchronous Methods
    # ========================================================================

    def response(
        self,
        messages: List[Dict[str, Any]],
        output_format: Union[Dict, type, None] = None,
        tools: Optional[List] = None,
        verbose: bool = False,
        hide_thinking: bool = True,
        reasoning_effort: Optional[str] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict] = None,
    ) -> FinalResponse:
        """Request model inference (non-streaming)."""
        if messages is None:
            raise ValueError("messages must be provided")

        output_format = self._prepare_output_format(output_format)

        final_content = None
        last_answer = ""

        for event in self.stream_response(
            messages=messages,
            output_format=output_format,
            final=True,
            tools=tools,
            hide_thinking=hide_thinking,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            verbose=verbose,
            extra_body=extra_body or self._config.extra_body,
        ):
            if event.get("type") == EventType.ANSWER.value:
                content = event.get("content")
                if isinstance(content, str):
                    last_answer += content
            elif event.get("type") == EventType.FINAL.value:
                final_content = event.get("content")
                break

        if final_content is None:
            return {"answer": last_answer}

        return final_content

    def stream_response(
        self,
        messages: List[Dict],
        output_format: Union[Dict, type, None] = None,
        final: bool = False,
        tools: Optional[List] = None,
        hide_thinking: bool = True,
        reasoning_effort: Optional[str] = None,
        max_tokens: Optional[int] = None,
        verbose: bool = False,
        extra_body: Optional[Dict] = None,
    ) -> Generator[StreamEvent, None, None]:
        """Request model inference with streaming."""
        if messages is None:
            raise ValueError("messages must be provided")

        output_format = self._prepare_output_format(output_format)
        kwargs, _, structured_output = self._build_request(
            messages, output_format, tools, reasoning_effort, max_tokens, extra_body
        )

        thinking_parser = ThinkingParser(self._config.custom_thinking_token)
        tool_accumulator = ToolCallAccumulator()

        thinking = ""
        answer = ""
        start_time = time.perf_counter()
        latency: Optional[float] = None
        tokens = 0

        try:
            completion = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            raise ModelRequestError(f"Model request failed: {e}")

        try:
            for chunk in completion:
                if latency is None:
                    latency = time.perf_counter() - start_time

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

                if tool_calls := getattr(delta, "tool_calls", None):
                    for tc in tool_calls:
                        tool_accumulator.add_chunk(tc)
        except Exception as e:
            raise ModelRequestError(f"Model stream failed: {e}") from e

        elapsed = time.perf_counter() - start_time
        tokens_per_second = tokens / elapsed if elapsed > 0 else 0

        if structured_output:
            try:
                answer = json.loads(answer)
            except json.JSONDecodeError:
                pass
            yield self._event_builder.answer(answer)

        final_tool_calls = tool_accumulator.get_completed_calls()
        for idx, tc in enumerate(final_tool_calls):
            yield self._event_builder.tool_call(tc, source=tc["name"], job=idx + 1)

        verbose_info: VerboseInfo = {
            "tokens": tokens,
            "tokens_per_second": tokens_per_second,
            "latency": latency
        }

        if verbose:
            yield self._event_builder.verbose(verbose_info)

        if final:
            final_response: FinalResponse = {
                "answer": answer.strip() if isinstance(answer, str) else answer
            }
            if not hide_thinking and thinking.strip():
                final_response["reasoning"] = thinking.strip()
            if final_tool_calls:
                final_response["tool_calls"] = final_tool_calls
            if verbose:
                final_response["verbose"] = verbose_info

            yield self._event_builder.final(final_response)

        yield self._event_builder.done()

    # ========================================================================
    # Asynchronous Methods
    # ========================================================================

    async def async_response(
        self,
        messages: List[Dict[str, Any]],
        output_format: Union[Dict, type, None] = None,
        tools: Optional[List] = None,
        verbose: bool = False,
        hide_thinking: bool = True,
        reasoning_effort: Optional[str] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict] = None,
    ) -> FinalResponse:
        """Async request for model inference."""
        if messages is None:
            raise ValueError("messages must be provided")

        output_format = self._prepare_output_format(output_format)

        final_content = None
        async for event in self.async_stream_response(
            messages=messages,
            output_format=output_format,
            final=True,
            tools=tools,
            hide_thinking=hide_thinking,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            verbose=verbose,
            extra_body=extra_body or self._config.extra_body,
        ):
            if event.get("type") == EventType.FINAL.value:
                final_content = event.get("content")
                break

        if final_content is None:
            raise RuntimeError("No final response received")

        return final_content

    async def async_stream_response(
        self,
        messages: List[Dict],
        output_format: Union[Dict, type, None] = None,
        final: bool = False,
        tools: Optional[List] = None,
        verbose: bool = False,
        hide_thinking: bool = True,
        reasoning_effort: Optional[str] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Async streaming model inference."""
        if messages is None:
            raise ValueError("messages must be provided")

        import asyncio

        output_format = self._prepare_output_format(output_format)
        kwargs, _, structured_output = self._build_request(
            messages, output_format, tools, reasoning_effort, max_tokens, extra_body
        )

        thinking_parser = ThinkingParser(self._config.custom_thinking_token)
        tool_accumulator = ToolCallAccumulator()

        thinking = ""
        answer = ""
        start_time = time.perf_counter()
        latency: Optional[float] = None
        tokens = 0

        try:
            create_call = self._async_client.chat.completions.create(**kwargs)
            completion = await create_call if asyncio.iscoroutine(create_call) else create_call
        except Exception as e:
            raise ModelRequestError(f"Async model request failed: {e}")

        try:
            async for chunk in completion:
                if latency is None:
                    latency = time.perf_counter() - start_time

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

                if tool_calls := getattr(delta, "tool_calls", None):
                    for tc in tool_calls:
                        tool_accumulator.add_chunk(tc)
        except Exception as e:
            raise ModelRequestError(f"Async model stream failed: {e}") from e

        elapsed = time.perf_counter() - start_time
        tokens_per_second = tokens / elapsed if elapsed > 0 else 0

        if structured_output:
            try:
                answer = json.loads(answer)
            except json.JSONDecodeError:
                pass
            yield self._event_builder.answer(answer)

        final_tool_calls = tool_accumulator.get_completed_calls()
        for idx, tc in enumerate(final_tool_calls):
            yield self._event_builder.tool_call(tc, source=tc["name"], job=idx + 1)

        verbose_info: VerboseInfo = {
            "tokens": tokens,
            "tokens_per_second": tokens_per_second,
            "latency": latency
        }

        if verbose:
            yield self._event_builder.verbose(verbose_info)

        if final:
            final_response: FinalResponse = {
                "answer": answer.strip() if isinstance(answer, str) else answer
            }
            if not hide_thinking and thinking.strip():
                final_response["reasoning"] = thinking.strip()
            if final_tool_calls:
                final_response["tool_calls"] = final_tool_calls
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
        await self._async_client.close()

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
# Public API
# ============================================================================

__all__ = [
    "LLM",
    "LLMConfig",
    "CustomThinkingToken",
    "StreamEvent",
    "ToolCall",
    "FinalResponse",
    "VerboseInfo",
    "EventType",
    "LLMError",
    "ConfigurationError",
    "SchemaConversionError",
    "ModelRequestError",
]
