"""
Universal LLM API Wrapper with OpenAI-compatible API support.

Features:
- Structured outputs with automatic schema generation
- Vision model support
- Tool calling (sync/async)
- Streaming with thinking token handling
- Tool interceptors for human-in-the-loop workflows
"""

from __future__ import annotations

import json
import base64
import re
import io
import time
import sys
import queue
import threading
import asyncio
import inspect
import logging
from dataclasses import dataclass
from enum import Enum
from typing import (
    Any, AsyncGenerator, Dict, Optional, List, Generator,
    Callable, Union, get_type_hints, get_origin, get_args, 
    Literal, Awaitable, TypedDict, TypeVar, Final, ClassVar,
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

# Silence verbose third-party logging
for _logger_name in ("httpx", "openai", "httpcore"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)

# ============================================================================
# Constants
# ============================================================================

DEFAULT_API_KEY: Final[str] = "lm-studio"
DEFAULT_BASE_URL: Final[str] = "http://localhost:1234/v1"
DEFAULT_TIMEOUT: Final[float] = 300.0

# ============================================================================
# Enums
# ============================================================================

class EventType(str, Enum):
    """Event types emitted during streaming."""
    ANSWER = "answer"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_ERROR = "tool_error"
    TOOL_STREAM = "tool_stream"      # NEW: streaming output from tool
    REVIEW_REQUEST = "review_request"
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

ToolInterceptor = Callable[[str, Dict[str, Any]], Union[bool, Awaitable[bool]]]
ToolFunction = Callable[..., Any]

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


class ToolResult(TypedDict, total=False):
    """Typed dictionary for tool results."""
    name: str
    result: Any
    id: str
    is_error: bool


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
    tool_results: List[ToolResult]
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


class ToolExecutionError(LLMError):
    """Raised when tool execution fails."""
    def __init__(self, tool_name: str, message: str, original_error: Optional[Exception] = None):
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"Tool '{tool_name}' failed: {message}")


class ModelRequestError(LLMError):
    """Raised when model request fails."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when operation times out."""
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
    """Configuration for LLM instance.
    
    Attributes:
        model: Model identifier.
        api_key: API key for authentication.
        base_url: Base URL for API endpoint.
        custom_thinking_token: Custom thinking token configuration.
        default_stop_sequences: Default stop sequences for generation.
        timeout: Default timeout for synchronous operations.
        tool_exec: Whether to execute tools (True) or only stream tool calls (False).
    """
    model: str
    api_key: str = DEFAULT_API_KEY
    base_url: str = DEFAULT_BASE_URL
    custom_thinking_token: Optional[CustomThinkingToken] = None
    default_stop_sequences: Optional[List[str]] = None
    timeout: float = DEFAULT_TIMEOUT
    extra_body: Optional[Dict[str, Any]] = None
    tool_exec: bool = True

# ============================================================================
# Thinking Parser
# ============================================================================

class ThinkingParser:
    """Parses thinking tokens from streamed content.
    
    Supports multiple thinking tag formats:
    - XML-style: <think>, <thought>, <thinking>
    - Bracket-style: [THINK]
    - Custom patterns via CustomThinkingToken
    """
    
    # Pre-compiled base patterns for performance
    _BASE_START_PATTERNS: ClassVar[tuple[str, ...]] = (
        r'<think>', r'<thinking>', r'\[THINK\]'
    )
    _BASE_END_PATTERNS: ClassVar[tuple[str, ...]] = (
        r'</think>', r'</thinking>', r'\[/THINK\]'
    )
    
    def __init__(self, custom_token: Optional[CustomThinkingToken] = None):
        self._custom_token = custom_token
        self._start_pattern = self._build_pattern(self._BASE_START_PATTERNS, 
                                                   custom_token.start_token if custom_token else None)
        self._end_pattern = self._build_pattern(self._BASE_END_PATTERNS,
                                                 custom_token.end_token if custom_token else None)
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
        
        Args:
            content: Content to parse.
            
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
    
    # Type mapping from Python to JSON Schema
    _TYPE_MAP: ClassVar[Dict[str, SchemaType]] = {
        "str": SchemaType.STRING,
        "int": SchemaType.INTEGER,
        "float": SchemaType.NUMBER,
        "bool": SchemaType.BOOLEAN,
        "list": SchemaType.ARRAY,
        "dict": SchemaType.OBJECT,
    }
    
    # Types that LLMs can meaningfully generate
    _LLM_SUPPORTED_TYPES: ClassVar[frozenset] = frozenset({str, int, float, bool, list, dict})
    
    def python_type_to_json_schema(
        self, 
        python_type: Any, 
        seen_models: Optional[set] = None
    ) -> Dict[str, Any]:
        """Convert Python type annotation to JSON Schema.
        
        Supports:
        - Basic types: str, int, float, bool
        - list, List[T]
        - dict, Dict[str, T]
        - Optional[T], Union[T, None]
        - Literal[...] (as enum)
        - Nested classes with __annotations__
        
        Args:
            python_type: Python type to convert.
            seen_models: Set of already-seen models (for recursion detection).
            
        Returns:
            JSON Schema dictionary.
            
        Raises:
            SchemaConversionError: If type cannot be converted or circular dependency detected.
        """
        seen_models = seen_models or set()
        
        # Handle None type
        if python_type is type(None):
            return {"type": SchemaType.NULL.value}
        
        origin = get_origin(python_type)
        args = get_args(python_type)
        
        # Handle List[T]
        if origin is list:
            schema: Dict[str, Any] = {"type": SchemaType.ARRAY.value}
            if args:
                schema["items"] = self.python_type_to_json_schema(args[0], seen_models)
            return schema
        
        # Handle Dict[K, V]
        if origin is dict:
            schema = {"type": SchemaType.OBJECT.value}
            if len(args) == 2:
                schema["additionalProperties"] = self.python_type_to_json_schema(args[1], seen_models)
            return schema
        
        # Handle Optional[T] / Union[T, None]
        if origin is Union:
            non_none_types = [t for t in args if t is not type(None)]
            
            if len(non_none_types) == 1:
                # This is Optional[T]
                return {
                    "anyOf": [
                        self.python_type_to_json_schema(non_none_types[0], seen_models),
                        {"type": SchemaType.NULL.value}
                    ]
                }
            
            # Multiple non-None types
            return {
                "anyOf": [self.python_type_to_json_schema(t, seen_models) for t in args]
            }
        
        # Handle Literal (as enum)
        if origin is Literal:
            return {"enum": list(args)}
        
        # Handle nested class with __annotations__
        if self._is_annotated_class(python_type):
            if python_type in seen_models:
                raise SchemaConversionError(
                    f"Circular dependency detected for class {python_type.__name__}. "
                    "Recursive schemas are not supported."
                )
            
            nested_schema = self.convert_class_to_schema(python_type, seen_models=seen_models)
            return nested_schema["json_schema"]["schema"]
        
        # Basic types
        return {"type": self._get_json_type(python_type).value}
    
    def _is_annotated_class(self, python_type: Any) -> bool:
        """Check if type is a class with annotations."""
        return (
            hasattr(python_type, "__annotations__") 
            and python_type.__annotations__ 
            and isinstance(python_type, type)
        )
    
    def _get_json_type(self, python_type: Any) -> SchemaType:
        """Get JSON Schema type for a basic Python type."""
        type_name = getattr(python_type, "__name__", str(python_type)).lower()
        return self._TYPE_MAP.get(type_name, SchemaType.STRING)
    
    def is_llm_supported_type(self, python_type: Any) -> bool:
        """Check if a Python type can be meaningfully provided by an LLM.
        
        Returns False for complex objects that should be injected, not generated.
        
        Args:
            python_type: Type to check.
            
        Returns:
            True if LLM can generate this type.
        """
        if python_type is None or python_type is type(None):
            return True
        
        origin = get_origin(python_type)
        args = get_args(python_type)
        
        # List[T] - check inner type
        if origin is list:
            return not args or self.is_llm_supported_type(args[0])
        
        # Dict[K, V] - check value type
        if origin is dict:
            return len(args) != 2 or self.is_llm_supported_type(args[1])
        
        # Optional[T] / Union[T, None]
        if origin is Union:
            non_none = [t for t in args if t is not type(None)]
            return all(self.is_llm_supported_type(t) for t in non_none)
        
        # Literal is always supported
        if origin is Literal:
            return True
        
        # Check against supported types
        return python_type in self._LLM_SUPPORTED_TYPES
    
    def convert_class_to_schema(
        self, 
        schema_class: type, 
        name: Optional[str] = None,
        seen_models: Optional[set] = None
    ) -> Dict[str, Any]:
        """Convert plain class with __annotations__ to OpenAI JSON schema.
        
        Args:
            schema_class: Class to convert.
            name: Optional name for the schema.
            seen_models: Set of already-seen models (for recursion detection).
            
        Returns:
            OpenAI-compatible schema dictionary.
            
        Raises:
            SchemaConversionError: If class has no annotations.
        """
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
            
            # Get class-level defaults
            class_defaults = {
                k: v for k, v in schema_class.__dict__.items()
                if not k.startswith("_") and not callable(v)
            }
            
            for field_name, field_type in hints.items():
                properties[field_name] = self.python_type_to_json_schema(field_type, seen_models)
                
                # Check if optional
                is_optional = (
                    get_origin(field_type) is Union 
                    and type(None) in get_args(field_type)
                )
                
                if field_name not in class_defaults and not is_optional:
                    required.append(field_name)
            
            schema: Dict[str, Any] = {
                "type": SchemaType.OBJECT.value,
                "properties": properties,
                "required": required,
                "additionalProperties": False
            }
            
            # Add docstring as description
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
    callables: Dict[str, ToolFunction]
    injected_params: Dict[str, Dict[str, Any]]


class ToolPreparator:
    """Prepares tools for LLM consumption."""
    
    def __init__(self, schema_converter: SchemaConverter):
        self._converter = schema_converter
    
    def prepare(self, tools: Optional[List[Any]]) -> PreparedTools:
        """Convert callable functions to OpenAI tool format.
        
        Args:
            tools: List of callables or tool definition dicts.
            
        Returns:
            PreparedTools with definitions, callables mapping, and injected params.
            
        Raises:
            ConfigurationError: If tool format is invalid.
        """
        if not tools:
            return PreparedTools([], {}, {})
        
        definitions = []
        callables: Dict[str, ToolFunction] = {}
        injected_params: Dict[str, Dict[str, Any]] = {}
        
        for idx, tool in enumerate(tools):
            if callable(tool):
                definition, name, injected = self._prepare_callable(tool)
                definitions.append(definition)
                callables[name] = tool
                if injected:
                    injected_params[name] = injected
            elif isinstance(tool, dict):
                self._validate_tool_dict(tool, idx)
                definitions.append(tool)
            else:
                raise ConfigurationError(
                    f"Tool at index {idx} must be callable or dict, got {type(tool).__name__}"
                )
        
        return PreparedTools(definitions, callables, injected_params)
    
    def _prepare_callable(self, func: Callable) -> tuple[Dict, str, Dict[str, Any]]:
        """Prepare a callable for LLM consumption."""
        # Unwrap partials to get original function
        underlying = func
        while hasattr(underlying, 'func'):
            underlying = underlying.func
        
        name = (getattr(func, '__name__', None) or underlying.__name__).strip()
        doc = (getattr(func, '__doc__', None) or underlying.__doc__ or "").strip()
        
        # Get type hints from underlying function
        try:
            annotations = get_type_hints(underlying)
        except Exception:
            annotations = getattr(underlying, "__annotations__", {})
        
        # Get signature from actual callable
        sig = inspect.signature(func)
        
        parameters = {}
        required = []
        injected: Dict[str, Any] = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == "return":
                continue
            
            param_type = annotations.get(param_name)
            
            # Check if type should be hidden from LLM
            if param_type is not None and not self._converter.is_llm_supported_type(param_type):
                default = param.default if param.default != inspect.Parameter.empty else None
                injected[param_name] = default
                continue
            
            # Build parameter schema
            param_schema = (
                self._converter.python_type_to_json_schema(param_type)
                if param_type else {"type": SchemaType.STRING.value}
            )
            
            # Add default value to description
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
        
        definition = {
            "type": "function",
            "function": {
                "name": name,
                "description": doc,
                "parameters": {
                    "type": SchemaType.OBJECT.value,
                    "properties": parameters,
                    "required": required
                }
            }
        }
        
        return definition, name, injected
    
    @staticmethod
    def _format_default(value: Any) -> str:
        """Format a default value for display."""
        if isinstance(value, str):
            return f'"{value}"'
        if value is None:
            return "null"
        return repr(value)
    
    @staticmethod
    def _validate_tool_dict(tool: Dict, index: int) -> None:
        """Validate a tool definition dictionary."""
        if "type" not in tool or "function" not in tool:
            raise ConfigurationError(
                f"Tool at index {index} must have 'type' and 'function' keys"
            )
        if "name" not in tool.get("function", {}):
            raise ConfigurationError(
                f"Tool at index {index} missing 'name' in function definition"
            )

# ============================================================================
# Image Processor
# ============================================================================

class ImageProcessor:
    """Processes images in messages for API consumption."""
    
    # 1. Create a cache variable (starts empty)
    _pil_image = None
    
    @classmethod
    def _get_pil(cls):
        """Helper to import PIL only once."""
        if cls._pil_image is None:
            try:
                from PIL import Image
                cls._pil_image = Image
            except ImportError:
                raise ImportError("PIL/Pillow required. Install with: pip install Pillow")
        return cls._pil_image

    @staticmethod
    def process_messages(messages: List[Dict]) -> None:
        """Run this every time. It's safe."""
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
        # 2. Use the helper instead of importing directly
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
# Async Runner
# ============================================================================

class AsyncRunner:
    """Runs async code from sync contexts."""
    
    @staticmethod
    def run_sync(coro: Awaitable[T], timeout: float = DEFAULT_TIMEOUT) -> T:
        """Run a coroutine synchronously with timeout.
        
        Args:
            coro: Coroutine to run.
            timeout: Maximum wait time in seconds.
            
        Returns:
            Result of the coroutine.
            
        Raises:
            TimeoutError: If timeout is exceeded.
            RuntimeError: If execution fails.
        """
        try:
            asyncio.get_running_loop()
            loop_running = True
        except RuntimeError:
            loop_running = False
        
        if not loop_running:
            return asyncio.run(coro)
        
        # Thread-based execution for nested event loops
        result_queue: queue.Queue = queue.Queue()
        exception_tb: List[Optional[str]] = [None]
        
        def worker():
            try:
                result = asyncio.run(coro)
                result_queue.put(("success", result))
            except Exception as e:
                import traceback
                exception_tb[0] = traceback.format_exc()
                result_queue.put(("error", e))
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        
        try:
            status, payload = result_queue.get(timeout=timeout)
        except queue.Empty:
            raise LLMTimeoutError(
                f"Coroutine execution timed out after {timeout}s. "
                "Consider using async methods."
            )
        
        if status == "success":
            return payload
        
        raise RuntimeError(
            f"Coroutine execution failed: {payload}\n"
            f"Original traceback:\n{exception_tb[0]}"
        ) from payload

# ============================================================================
# Event Builder
# ============================================================================

class EventBuilder:
    """Builds standardized stream events with unified format."""
    
    @staticmethod
    def _build(
        event_type: EventType,
        content: Any,
        source: Optional[str] = None,
        tool_id: Optional[str] = None,
        job: Optional[int] = None,
        depth: int = 0  # NEW
    ) -> StreamEvent:
        """Build event with all metadata."""
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
    def answer(
        content: Any,
        source: Optional[str] = None,
        tool_id: Optional[str] = None,
        job: Optional[int] = None,
        depth: int = 0
    ) -> StreamEvent:
        return EventBuilder._build(EventType.ANSWER, content, source, tool_id, job, depth)
    
    @staticmethod
    def reasoning(
        content: str,
        source: Optional[str] = None,
        tool_id: Optional[str] = None,
        job: Optional[int] = None,
        depth: int = 0
    ) -> StreamEvent:
        return EventBuilder._build(EventType.REASONING, content, source, tool_id, job, depth)
    
    @staticmethod
    def tool_call(
        content: ToolCall,
        source: Optional[str] = None,
        job: Optional[int] = None,
        depth: int = 0
    ) -> StreamEvent:
        return EventBuilder._build(EventType.TOOL_CALL, content, source, content.get("id"), job, depth)
    
    @staticmethod
    def tool_result(
        content: ToolResult,
        source: Optional[str] = None,
        job: Optional[int] = None,
        depth: int = 0
    ) -> StreamEvent:
        return EventBuilder._build(EventType.TOOL_RESULT, content, source, content.get("id"), job, depth)
    
    @staticmethod
    def tool_error(
        content: Dict,
        source: Optional[str] = None,
        job: Optional[int] = None,
        depth: int = 0
    ) -> StreamEvent:
        return EventBuilder._build(EventType.TOOL_ERROR, content, source, content.get("id"), job, depth)
    
    @staticmethod
    def tool_stream(
        content: Any,
        source: str,
        tool_id: str,
        job: Optional[int] = None,
        depth: int = 1
    ) -> StreamEvent:
        """NEW: Wrap streaming tool output."""
        return EventBuilder._build(EventType.TOOL_STREAM, content, source, tool_id, job, depth)
    
    @staticmethod
    def review_request(
        content: ToolCall,
        source: Optional[str] = None,
        job: Optional[int] = None,
        depth: int = 0
    ) -> StreamEvent:
        return EventBuilder._build(EventType.REVIEW_REQUEST, content, source, content.get("id"), job, depth)
    
    @staticmethod
    def verbose(content: VerboseInfo, depth: int = 0) -> StreamEvent:
        return EventBuilder._build(EventType.VERBOSE, content, None, None, None, depth)
    
    @staticmethod
    def final(content: FinalResponse, depth: int = 0) -> StreamEvent:
        return EventBuilder._build(EventType.FINAL, content, None, None, None, depth)
    
    @staticmethod
    def done(depth: int = 0) -> StreamEvent:
        return EventBuilder._build(EventType.DONE, None, None, None, None, depth)

# ============================================================================
# Tool Call Accumulator
# ============================================================================

class ToolCallAccumulator:
    """Accumulates streaming tool call chunks into complete calls."""
    
    def __init__(self):
        self._calls: Dict[str, Dict[str, str]] = {}
        self._index_to_id: Dict[int, str] = {}
    
    def add_chunk(self, tool_call: Any) -> None:
        """Add a streaming tool call chunk."""
        idx = getattr(tool_call, "index", 0)
        
        # Store ID when provided
        if tool_id := getattr(tool_call, "id", None):
            self._index_to_id[idx] = tool_id
        
        # Get or create tool ID
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
        """Get list of completed tool calls."""
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
        """Clear accumulated calls."""
        self._calls.clear()
        self._index_to_id.clear()

# ============================================================================
# Main LLM Class
# ============================================================================

class LLM:
    """Universal API wrapper for LLM models with OpenAI-compatible API.
    
    Features:
    - Structured outputs with automatic schema generation from Python classes
    - Vision model support with automatic image encoding
    - Tool calling with automatic function introspection
    - Streaming with thinking token handling
    - Both synchronous and asynchronous operation modes
    
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
        tool_exec: bool = True,
    ):
        """Initialize the LLM wrapper.
        
        Args:
            model: Model identifier to use.
            api_key: API key for authentication.
            base_url: Base URL for API endpoint.
            custom_thinking_token: Custom thinking token configuration.
            default_stop_sequences: Default stop sequences for generation.
            timeout: Default timeout for sync operations.
            tool_exec: Whether to execute tools (True) or only stream tool calls (False).
        """
        self._config = LLMConfig(
            model=model,
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            custom_thinking_token=custom_thinking_token,
            default_stop_sequences=default_stop_sequences,
            timeout=timeout,
            extra_body=extra_body,
            tool_exec=tool_exec,
        )
        
        # Compute API base URL
        self._api_base = self._compute_api_base()
        
        # Initialize clients
        self._client = OpenAI(base_url=self._api_base, api_key=api_key)
        self._async_client = AsyncOpenAI(base_url=self._api_base, api_key=api_key)
        
        # Initialize components
        self._schema_converter = SchemaConverter()
        self._tool_preparator = ToolPreparator(self._schema_converter)
        self._event_builder = EventBuilder()
        
        logger.debug(f"LLM initialized: model={model}, base_url={self._api_base.split('@')[-1]}")
    
    def _compute_api_base(self) -> str:
        """Compute the API base URL."""
        base = self._config.base_url
        if not base.endswith("/v1") and "openai" not in base.lower():
            return base + "/v1"
        return base
    
    @property
    def model(self) -> str:
        """Current model identifier."""
        return self._config.model
    
    @property
    def base_url(self) -> str:
        """Base URL for API."""
        return self._config.base_url
    
    # ========================================================================
    # Output Format Handling
    # ========================================================================
    
    def _prepare_output_format(
        self, 
        output_format: Union[Dict, type, None]
    ) -> Optional[Dict]:
        """Convert output_format to OpenAI schema format.
        
        Args:
            output_format: None, dict (passthrough), or type (convert).
            
        Returns:
            OpenAI-compatible schema dict or None.
            
        Raises:
            ConfigurationError: If format is unsupported.
        """
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
    # LM Studio Integration
    # ========================================================================
    
    def _unload_other_models(self) -> None:
        """Unload all models except current one in LM Studio."""
        try:
            import lmstudio as lms
            try:
                lms.configure_default_client(self._config.base_url)
            except Exception:
                pass
            
            for model in (lms.list_loaded_models() or []):
                if model.identifier != self._config.model:
                    model.unload()
                    logger.debug(f"Unloaded model: {model.identifier}")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to unload models: {e}")
    
    def lm_studio_count_tokens(self, input_text: str) -> int:
        """Count tokens for text using LM Studio.
        
        Args:
            input_text: Text to tokenize.
            
        Returns:
            Token count.
            
        Raises:
            RuntimeError: If tokenization fails.
        """
        try:
            import lmstudio as lms
            try:
                lms.configure_default_client(self._config.base_url)
            except Exception:
                pass
            
            model = lms.llm(self._config.model)
            return len(model.tokenize(input_text))
        except ImportError:
            raise RuntimeError("lmstudio package not installed")
        except Exception as e:
            raise RuntimeError(f"Token counting failed: {e}")
    
    def lm_studio_get_context_length(self) -> int:
        """Get model context length from LM Studio.
        
        Returns:
            Context length in tokens.
            
        Raises:
            RuntimeError: If lookup fails.
        """
        try:
            import lmstudio as lms
            try:
                lms.configure_default_client(self._config.base_url)
            except Exception:
                pass
            
            return lms.llm(self._config.model).get_context_length()
        except ImportError:
            raise RuntimeError("lmstudio package not installed")
        except Exception as e:
            raise RuntimeError(f"Context length lookup failed: {e}")
    
    # ========================================================================
    # Synchronous Methods
    # ========================================================================
    
    def response(
        self, 
        messages: List[Dict[str, Any]],
        output_format: Union[Dict, type, None] = None,
        tools: Optional[List] = None,
        lm_studio_unload_model: bool = False,
        verbose: bool = False,
        hide_thinking: bool = True,
        reasoning_effort: Optional[str] = None,
        max_tokens: Optional[int] = None,
        tool_interceptor: Optional[ToolInterceptor] = None,
        extra_body: Optional[Dict] = None,
        tool_exec: Optional[bool] = None,
    ) -> FinalResponse:
        """Request model inference (non-streaming).
        
        Args:
            messages: Conversation messages.
            output_format: JSON schema or class for structured output.
            tools: List of callable functions or tool definitions.
            lm_studio_unload_model: Unload other models in LM Studio.
            verbose: Include timing/token information.
            hide_thinking: Hide reasoning tokens from output.
            reasoning_effort: Reasoning effort level (model-specific).
            max_tokens: Maximum tokens to generate.
            tool_interceptor: Callback for tool review.
            extra_body: Extra body for the request.
            tool_exec: Override tool execution (None uses config default).
            
        Returns:
            Final response with answer and optional tool results.
            
        Raises:
            ValueError: If messages is None.
            RuntimeError: If no response received.
        """
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
            lm_studio_unload_model=lm_studio_unload_model,
            hide_thinking=hide_thinking,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            verbose=verbose,
            tool_interceptor=tool_interceptor,
            extra_body=extra_body or self._config.extra_body,
            tool_exec=tool_exec,
        ):
            if event["type"] == EventType.ANSWER.value:
                content = event["content"]
                if isinstance(content, str):
                    last_answer += content
            elif event["type"] == EventType.FINAL.value:
                final_content = event["content"]
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
        lm_studio_unload_model: bool = False,
        hide_thinking: bool = True,
        reasoning_effort: Optional[str] = None,
        max_tokens: Optional[int] = None,
        verbose: bool = False,
        tool_interceptor: Optional[ToolInterceptor] = None,
        extra_body: Optional[Dict] = None,
        tool_exec: Optional[bool] = None,
    ) -> Generator[StreamEvent, Optional[str], None]:
        """Request model inference with streaming.
        
        Args:
            messages: Conversation messages.
            output_format: JSON schema or class for structured output.
            final: Yield final aggregated response.
            tools: List of callable functions or tool definitions.
            lm_studio_unload_model: Unload other models in LM Studio.
            hide_thinking: Hide reasoning tokens.
            reasoning_effort: Reasoning effort level.
            max_tokens: Maximum tokens to generate.
            verbose: Include timing information.
            tool_interceptor: Callback for tool review.
            extra_body: Extra body for the request.
            tool_exec: Override tool execution (None uses config default).
            
        Yields:
            Stream events with type and content.
        """
        if messages is None:
            raise ValueError("messages must be provided")
        
        # Prepare
        output_format = self._prepare_output_format(output_format)
        prepared_tools = self._tool_preparator.prepare(tools)
        ImageProcessor.process_messages(messages)
        
        # Determine effective tool_exec value
        effective_tool_exec = tool_exec if tool_exec is not None else self._config.tool_exec
        
        if lm_studio_unload_model:
            self._unload_other_models()
        
        # Initialize state
        thinking_parser = ThinkingParser(self._config.custom_thinking_token)
        tool_accumulator = ToolCallAccumulator()
        structured_output = output_format is not None
        
        thinking = ""
        answer = ""
        
        # Metrics
        start_time = time.perf_counter()
        latency: Optional[float] = None
        tokens = 0
        
        # Build request
        kwargs: Dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
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
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if structured_output:
            kwargs["response_format"] = output_format
        
        # Execute request
        try:
            completion = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            raise ModelRequestError(f"Model request failed: {e}")
        
        # Process stream
        for chunk in completion:
            if latency is None:
                latency = time.perf_counter() - start_time
            
            if not chunk.choices:
                continue
            
            delta = chunk.choices[0].delta
            if not delta:
                continue
            
            tokens += 1
            
            # Handle reasoning
            if reasoning := getattr(delta, "reasoning", None):
                thinking += reasoning
                if not hide_thinking:
                    yield self._event_builder.reasoning(reasoning)
            
            # Handle content
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
            
            # Handle tool calls
            if tool_calls := getattr(delta, "tool_calls", None):
                for tc in tool_calls:
                    tool_accumulator.add_chunk(tc)
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        tokens_per_second = tokens / elapsed if elapsed > 0 else 0
        
        # Handle structured output
        if structured_output:
            try:
                answer = json.loads(answer)
            except json.JSONDecodeError:
                pass
            yield self._event_builder.answer(answer)
        
        # Execute tools
        final_tool_calls = tool_accumulator.get_completed_calls()
        executed_results: List[ToolResult] = []

        for idx, tool_call in enumerate(final_tool_calls):
            job = idx + 1
            tool_name = tool_call["name"]
            tool_id = tool_call["id"]
            
            yield self._event_builder.tool_call(tool_call, source=tool_name, job=job)
            
            # Skip execution if disabled
            if not effective_tool_exec:
                continue
            
            if tool_name not in prepared_tools.callables:
                continue
            
            try:
                if tool_interceptor:
                    needs_review = tool_interceptor(tool_name, tool_call["arguments"])
                    if asyncio.iscoroutine(needs_review):
                        needs_review = AsyncRunner.run_sync(needs_review)
                    
                    if needs_review:
                        decision = yield self._event_builder.review_request(
                            tool_call, source=tool_name, job=job
                        )
                        if decision != "approve":
                            error = {"name": tool_name, "error": f"Rejected: {decision}", "id": tool_id}
                            executed_results.append({
                                "name": tool_name, 
                                "result": f"REJECTED: {decision}", 
                                "id": tool_id, 
                                "is_error": True
                            })
                            yield self._event_builder.tool_error(error, source=tool_name, job=job)
                            continue
                
                # Execute tool
                result = self._execute_tool_sync(
                    tool_name, 
                    tool_call,
                    prepared_tools.callables[tool_name],
                    prepared_tools.injected_params.get(tool_name, {})
                )
                
                # Handle generators
                if inspect.isgenerator(result):
                    result = yield from self._consume_generator_sync(
                        result, tool_name, tool_id, job
                    )
                
                tool_result: ToolResult = {"name": tool_name, "result": result, "id": tool_id}
                executed_results.append(tool_result)
                yield self._event_builder.tool_result(tool_result, source=tool_name, job=job)
                
            except Exception as e:
                error = {"name": tool_name, "error": str(e), "id": tool_id}
                executed_results.append({
                    "name": tool_name, 
                    "result": f"ERROR: {e}", 
                    "id": tool_id, 
                    "is_error": True
                })
                yield self._event_builder.tool_error(error, source=tool_name, job=job)
        
        # Verbose info
        verbose_info: VerboseInfo = {
            "tokens": tokens,
            "tokens_per_second": tokens_per_second,
            "latency": latency
        }
        
        if verbose:
            yield self._event_builder.verbose(verbose_info)
        
        # Final response
        if final:
            final_response: FinalResponse = {
                "answer": answer.strip() if isinstance(answer, str) else answer
            }
            if not hide_thinking and thinking.strip():
                final_response["reasoning"] = thinking.strip()
            if final_tool_calls:
                final_response["tool_calls"] = final_tool_calls
            if executed_results:
                final_response["tool_results"] = executed_results
            if verbose:
                final_response["verbose"] = verbose_info
            
            yield self._event_builder.final(final_response)
        
        yield self._event_builder.done()
    
    def _execute_tool_sync(
        self, 
        name: str, 
        call: ToolCall,
        func: ToolFunction,
        injected: Dict[str, Any]
    ) -> Any:
        """Execute a tool synchronously."""
        args = dict(call["arguments"])
        for k, v in injected.items():
            args.setdefault(k, v)
        
        result = func(**args)
        
        # Handle coroutines
        if asyncio.iscoroutine(result) or inspect.isawaitable(result):
            if not inspect.isgenerator(result) and not inspect.isasyncgen(result):
                result = AsyncRunner.run_sync(result)
        
        # Block async generators in sync mode
        if inspect.isasyncgen(result):
            raise RuntimeError(
                f"Async generator tool '{name}' cannot be used in sync mode"
            )
        
        return result
    
    def _consume_generator_sync(
        self,
        gen: Generator,
        tool_name: str,
        tool_id: str,
        job: Optional[int] = None
    ) -> Generator[StreamEvent, Optional[str], Any]:
        """Consume a generator tool and yield TOOL_STREAM events."""
        final_result = None
        
        def compute_depth(chunk: Any) -> int:
            if isinstance(chunk, dict) and "depth" in chunk:
                return chunk["depth"] + 1
            return 1
        
        def wrap_chunk(chunk: Any) -> StreamEvent:
            return self._event_builder.tool_stream(
                content=chunk,
                source=tool_name,
                tool_id=tool_id,
                job=job,
                depth=compute_depth(chunk)
            )
        
        def is_event_type(chunk: Any, event_type: EventType) -> bool:
            return isinstance(chunk, dict) and chunk.get("type") == event_type.value
        
        def needs_send_handling(chunk: Any) -> bool:
            """Check if chunk needs send() handling (direct or nested REVIEW_REQUEST)."""
            if not isinstance(chunk, dict):
                return False
            chunk_type = chunk.get("type")
            if chunk_type == EventType.REVIEW_REQUEST.value:
                return True
            # Recursively check nested TOOL_STREAM content
            if chunk_type == EventType.TOOL_STREAM.value:
                return needs_send_handling(chunk.get("content"))
            return False
        
        try:
            chunk = next(gen)
            while True:
                # FINAL - capture result
                if is_event_type(chunk, EventType.FINAL):
                    final_result = chunk.get("content")
                    yield wrap_chunk(chunk)
                    try:
                        chunk = next(gen)
                    except StopIteration:
                        break
                    continue
                
                # REVIEW_REQUEST (direct or nested) - needs send() propagation
                if needs_send_handling(chunk):
                    wrapped = wrap_chunk(chunk)
                    decision = yield wrapped
                    try:
                        chunk = gen.send(decision)
                    except StopIteration:
                        break
                    continue
                
                # Normal chunk
                yield wrap_chunk(chunk)
                try:
                    chunk = next(gen)
                except StopIteration:
                    break
                    
        except StopIteration:
            pass
        
        return final_result
    
    # ========================================================================
    # Asynchronous Methods
    # ========================================================================
    
    async def async_response(
        self,
        messages: List[Dict[str, Any]],
        output_format: Union[Dict, type, None] = None,
        tools: Optional[List] = None,
        lm_studio_unload_model: bool = False,
        verbose: bool = False,
        hide_thinking: bool = True,
        reasoning_effort: Optional[str] = None,
        max_tokens: Optional[int] = None,
        tool_interceptor: Optional[ToolInterceptor] = None,
        extra_body: Optional[Dict] = None,
        tool_exec: Optional[bool] = None,
    ) -> FinalResponse:
        """Async request for model inference.
        
        Args:
            messages: Conversation messages.
            output_format: JSON schema or class for structured output.
            tools: List of callable functions or tool definitions.
            lm_studio_unload_model: Unload other models.
            verbose: Include timing information.
            hide_thinking: Hide reasoning tokens.
            reasoning_effort: Reasoning effort level.
            max_tokens: Maximum tokens to generate.
            tool_interceptor: Callback for tool review.
            extra_body: Extra body for the request.
            tool_exec: Override tool execution (None uses config default).
            
        Returns:
            Final response dictionary.
        """
        if messages is None:
            raise ValueError("messages must be provided")
        
        output_format = self._prepare_output_format(output_format)
        
        final_content = None
        async for event in self.async_stream_response(
            messages=messages,
            output_format=output_format,
            final=True,
            tools=tools,
            lm_studio_unload_model=lm_studio_unload_model,
            hide_thinking=hide_thinking,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
            verbose=verbose,
            tool_interceptor=tool_interceptor,
            extra_body=extra_body or self._config.extra_body,
            tool_exec=tool_exec,
        ):
            if event["type"] == EventType.FINAL.value:
                final_content = event["content"]
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
        lm_studio_unload_model: bool = False,
        verbose: bool = False,
        hide_thinking: bool = True,
        reasoning_effort: Optional[str] = None,
        max_tokens: Optional[int] = None,
        tool_interceptor: Optional[ToolInterceptor] = None,
        extra_body: Optional[Dict] = None,
        tool_exec: Optional[bool] = None,
    ) -> AsyncGenerator[StreamEvent, Optional[str]]:
        """Async streaming model inference.
        
        Args:
            messages: Conversation messages.
            output_format: JSON schema or class for structured output.
            final: Yield final aggregated response.
            tools: List of callable functions or tool definitions.
            lm_studio_unload_model: Unload other models.
            verbose: Include timing information.
            hide_thinking: Hide reasoning tokens.
            reasoning_effort: Reasoning effort level.
            max_tokens: Maximum tokens to generate.
            tool_interceptor: Callback for tool review.
            extra_body: Extra body for the request.
            tool_exec: Override tool execution (None uses config default).
            
        Yields:
            Stream events with type and content.
        """
        if messages is None:
            raise ValueError("messages must be provided")
        
        # Prepare
        output_format = self._prepare_output_format(output_format)
        prepared_tools = self._tool_preparator.prepare(tools)
        ImageProcessor.process_messages(messages)
        
        # Determine effective tool_exec value
        effective_tool_exec = tool_exec if tool_exec is not None else self._config.tool_exec
        
        if lm_studio_unload_model:
            self._unload_other_models()
        
        # Initialize state
        thinking_parser = ThinkingParser(self._config.custom_thinking_token)
        tool_accumulator = ToolCallAccumulator()
        structured_output = output_format is not None
        
        thinking = ""
        answer = ""
        
        # Metrics
        start_time = time.perf_counter()
        latency: Optional[float] = None
        tokens = 0
        
        # Build request
        kwargs: Dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
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
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if structured_output:
            kwargs["response_format"] = output_format
        
        # Execute request
        try:
            create_call = self._async_client.chat.completions.create(**kwargs)
            completion = await create_call if asyncio.iscoroutine(create_call) else create_call
        except Exception as e:
            raise ModelRequestError(f"Async model request failed: {e}")
        
        # Process stream
        async for chunk in completion:
            if latency is None:
                latency = time.perf_counter() - start_time
            
            if not chunk.choices:
                continue
            
            delta = chunk.choices[0].delta
            if not delta:
                continue
            
            tokens += 1
            
            # Handle reasoning
            if reasoning := getattr(delta, "reasoning", None):
                thinking += reasoning
                if not hide_thinking:
                    yield self._event_builder.reasoning(reasoning)
            
            # Handle content
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
            
            # Handle tool calls
            if tool_calls := getattr(delta, "tool_calls", None):
                for tc in tool_calls:
                    tool_accumulator.add_chunk(tc)
        
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        tokens_per_second = tokens / elapsed if elapsed > 0 else 0
        
        # Handle structured output
        if structured_output:
            try:
                answer = json.loads(answer)
            except json.JSONDecodeError:
                pass
            yield self._event_builder.answer(answer)
        
        # Execute tools (parallel)
        final_tool_calls = tool_accumulator.get_completed_calls()
        executed_results: List[ToolResult] = []
        
        # Separate into approved and rejected
        approved_tools: List[tuple[ToolCall, int]] = []
        
        for idx, tool_call in enumerate(final_tool_calls):
            job = idx + 1
            tool_name = tool_call["name"]
            
            yield self._event_builder.tool_call(tool_call, source=tool_name, job=job)
            
            # Skip execution if disabled
            if not effective_tool_exec:
                continue
            
            if tool_name not in prepared_tools.callables:
                continue
            
            # Check interceptor
            if tool_interceptor:
                if inspect.iscoroutinefunction(tool_interceptor):
                    needs_review = await tool_interceptor(tool_name, tool_call["arguments"])
                else:
                    needs_review = tool_interceptor(tool_name, tool_call["arguments"])
                
                if needs_review:
                    decision = yield self._event_builder.review_request(tool_call, source=tool_name, job=job)
                    if decision != "approve":
                        error = {"name": tool_name, "error": f"Rejected: {decision}", "id": tool_call["id"]}
                        executed_results.append({"name": tool_name, "result": f"REJECTED: {decision}", "id": tool_call["id"], "is_error": True})
                        yield self._event_builder.tool_error(error, source=tool_name, job=job)
                        continue
            
            approved_tools.append((tool_call, job))
        
        # Run approved tools in parallel
        if approved_tools:
            event_queue: asyncio.Queue = asyncio.Queue()
            
            async def run_tool(tc: ToolCall, job: int):
                tool_name = tc["name"]
                tool_id = tc["id"]
                try:
                    result = await self._execute_tool_async(
                        tool_name, tc,
                        prepared_tools.callables[tool_name],
                        prepared_tools.injected_params.get(tool_name, {}),
                        event_queue, job
                    )
                    await event_queue.put(("result", {"name": tool_name, "result": result, "id": tool_id}, job))
                except Exception as e:
                    await event_queue.put(("error", {"name": tool_name, "error": str(e), "id": tool_id}, job))
                finally:
                    await event_queue.put(("done", None, None))
            
            tasks = [asyncio.create_task(run_tool(tc, j)) for tc, j in approved_tools]
            remaining = len(tasks)
            
            while remaining > 0:
                kind, payload, job = await event_queue.get()
                
                if kind == "done":
                    remaining -= 1
                elif kind == "review":
                    future = payload["future"]
                    decision = yield payload["event"]
                    future.set_result(decision)
                elif kind == "event":
                    yield payload
                elif kind == "result":
                    executed_results.append(payload)
                    yield self._event_builder.tool_result(payload, source=payload["name"], job=job)
                elif kind == "error":
                    executed_results.append({"name": payload["name"], "result": f"ERROR: {payload['error']}", "id": payload["id"], "is_error": True})
                    yield self._event_builder.tool_error(payload, source=payload["name"], job=job)
            
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verbose info
        verbose_info: VerboseInfo = {
            "tokens": tokens,
            "tokens_per_second": tokens_per_second,
            "latency": latency
        }
        
        if verbose:
            yield self._event_builder.verbose(verbose_info)
        
        # Final response
        if final:
            final_response: FinalResponse = {
                "answer": answer.strip() if isinstance(answer, str) else answer
            }
            if not hide_thinking and thinking.strip():
                final_response["reasoning"] = thinking.strip()
            if final_tool_calls:
                final_response["tool_calls"] = final_tool_calls
            if executed_results:
                final_response["tool_results"] = executed_results
            if verbose:
                final_response["verbose"] = verbose_info
            
            yield self._event_builder.final(final_response)
        
        yield self._event_builder.done()
    
    async def _execute_tool_async(
        self,
        name: str,
        call: ToolCall,
        func: ToolFunction,
        injected: Dict[str, Any],
        event_queue: asyncio.Queue,
        job: int
    ) -> Any:
        """Execute a tool asynchronously with generator support."""
        args = dict(call["arguments"])
        for k, v in injected.items():
            args.setdefault(k, v)
        
        result = func(**args)
        
        # Handle coroutines
        if asyncio.iscoroutine(result) or inspect.isawaitable(result):
            if not inspect.isgenerator(result) and not inspect.isasyncgen(result):
                result = await result
        
        # Handle async generators
        if inspect.isasyncgen(result):
            return await self._consume_async_generator(result, name, call["id"], event_queue, job)
        
        # Handle sync generators
        if inspect.isgenerator(result):
            loop = asyncio.get_running_loop()
            return await self._consume_sync_generator_async(result, name, call["id"], event_queue, job, loop)
        
        return result
    
    async def _consume_async_generator(
        self,
        gen: AsyncGenerator,
        tool_name: str,
        tool_id: str,
        event_queue: asyncio.Queue,
        job: int
    ) -> Any:
        """Consume an async generator tool with TOOL_STREAM events."""
        final_result = None
        
        def compute_depth(chunk: Any) -> int:
            if isinstance(chunk, dict) and "depth" in chunk:
                return chunk["depth"] + 1
            return 1
        
        def wrap_chunk(chunk: Any) -> StreamEvent:
            return self._event_builder.tool_stream(
                content=chunk,
                source=tool_name,
                tool_id=tool_id,
                job=job,
                depth=compute_depth(chunk)
            )
        
        def is_event_type(chunk: Any, event_type: EventType) -> bool:
            return isinstance(chunk, dict) and chunk.get("type") == event_type.value

        def needs_send_handling(chunk: Any) -> bool:
            if not isinstance(chunk, dict):
                return False
            chunk_type = chunk.get("type")
            if chunk_type == EventType.REVIEW_REQUEST.value:
                return True
            if chunk_type == EventType.TOOL_STREAM.value:
                return needs_send_handling(chunk.get("content"))
            return False
        
        try:
            chunk = await gen.__anext__()
            while True:
                if is_event_type(chunk, EventType.FINAL):
                    final_result = chunk.get("content")
                    await event_queue.put(("event", wrap_chunk(chunk), job))
                    try:
                        chunk = await gen.__anext__()
                    except StopAsyncIteration:
                        break
                    continue
                
                # Handle direct or nested REVIEW_REQUEST
                if needs_send_handling(chunk):
                    future = asyncio.get_running_loop().create_future()
                    await event_queue.put(("review", {
                        "event": wrap_chunk(chunk),
                        "future": future
                    }, job))
                    decision = await future
                    try:
                        chunk = await gen.asend(decision)
                    except StopAsyncIteration:
                        break
                    continue
                
                await event_queue.put(("event", wrap_chunk(chunk), job))
                try:
                    chunk = await gen.__anext__()
                except StopAsyncIteration:
                    break
        except StopAsyncIteration:
            pass
        
        return final_result
    
    async def _consume_sync_generator_async(
        self,
        gen: Generator,
        tool_name: str,
        tool_id: str,
        event_queue: asyncio.Queue,
        job: int,
        loop: asyncio.AbstractEventLoop
    ) -> Any:
        """Consume a sync generator in async context with TOOL_STREAM events."""
        final_result = None
        
        def compute_depth(chunk: Any) -> int:
            if isinstance(chunk, dict) and "depth" in chunk:
                return chunk["depth"] + 1
            return 1
        
        def wrap_chunk(chunk: Any) -> StreamEvent:
            return self._event_builder.tool_stream(
                content=chunk,
                source=tool_name,
                tool_id=tool_id,
                job=job,
                depth=compute_depth(chunk)
            )
        
        def is_event_type(chunk: Any, event_type: EventType) -> bool:
            return isinstance(chunk, dict) and chunk.get("type") == event_type.value
        
        def needs_send_handling(chunk: Any) -> bool:
            if not isinstance(chunk, dict):
                return False
            chunk_type = chunk.get("type")
            if chunk_type == EventType.REVIEW_REQUEST.value:
                return True
            if chunk_type == EventType.TOOL_STREAM.value:
                return needs_send_handling(chunk.get("content"))
            return False
        
        try:
            chunk = await loop.run_in_executor(None, next, gen)
            while True:
                if is_event_type(chunk, EventType.FINAL):
                    final_result = chunk.get("content")
                    await event_queue.put(("event", wrap_chunk(chunk), job))
                    try:
                        chunk = await loop.run_in_executor(None, next, gen)
                    except StopIteration:
                        break
                    continue
                
                if needs_send_handling(chunk):
                    future = loop.create_future()
                    await event_queue.put(("review", {
                        "event": wrap_chunk(chunk),
                        "future": future
                    }, job))
                    decision = await future
                    try:
                        chunk = await loop.run_in_executor(None, lambda: gen.send(decision))
                    except StopIteration:
                        break
                    continue
                
                await event_queue.put(("event", wrap_chunk(chunk), job))
                try:
                    chunk = await loop.run_in_executor(None, next, gen)
                except StopIteration:
                    break
        except StopIteration:
            pass
        
        return final_result

    # ============================================================================
    # Context Manager
    # ============================================================================

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self._client, 'close'):
            self._client.close()
        if hasattr(self._async_client, 'close'):
            # Note: async close is tricky in __exit__
            pass
        return False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._async_client.close()
        return False

# ============================================================================
# Tool Interceptor
# ============================================================================

def tool_interceptor(*tools: Union[str, Callable]) -> ToolInterceptor:
    review_set: set[str] = set()
    
    for tool in tools:
        if isinstance(tool, str):
            review_set.add(tool)
        elif callable(tool):
            underlying = tool
            while hasattr(underlying, 'func'):
                underlying = underlying.func
            review_set.add(getattr(tool, '__name__', None) or underlying.__name__)
    
    def interceptor(tool_name: str, arguments: Dict[str, Any]) -> bool:
        return tool_name in review_set
    
    return interceptor

# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Main class
    "LLM",
    
    # Configuration
    "LLMConfig",
    "CustomThinkingToken",
    
    # Types
    "ToolInterceptor",
    "StreamEvent",
    "ToolCall",
    "ToolResult",
    "FinalResponse",
    "VerboseInfo",
    
    # Enums
    "EventType",
    
    # Exceptions
    "LLMError",
    "ConfigurationError",
    "SchemaConversionError",
    "ToolExecutionError",
    "ModelRequestError",
    "LLMTimeoutError",

    # Tool Interceptor
    "tool_interceptor",
]
