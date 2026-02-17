# LLM SDK

A universal, production-ready Python SDK for OpenAI-compatible LLM APIs with advanced features including streaming, tool calling, thinking token parsing, generator tools, tool interceptors, and multi-modal support.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Features

- 🔄 **Streaming & Non-Streaming**: Both sync and async implementations with unified event format
- 🛠️ **Tool Calling**: Automatic function introspection, execution, and parallel async execution
- 🔁 **Generator Tools**: Tools can yield streaming output via sync/async generators
- 🛡️ **Tool Interceptors**: Human-in-the-loop approval for tool execution
- 🧠 **Thinking Token Parsing**: Built-in and custom patterns for reasoning tokens
- 🖼️ **Multi-Modal Support**: Image processing from file path, PIL, URL, or base64
- 📦 **Structured Output**: JSON schema from Python classes with nested type support
- ⚡ **Async/Await**: Full async support with parallel tool execution
- 🔒 **Context Manager**: Resource cleanup via `with` / `async with`
- 📊 **Verbose Mode**: Token counts, latency, and throughput metrics

---

## 📦 Installation

### Prerequisites

```bash
pip install openai
```

### Optional Dependencies

```bash
pip install lmstudio pillow
```

### Install from Source

```bash
git clone https://github.com/flgaertig/better-llm-sdk.git
cd better-llm-sdk
```

---

## 🚀 Quick Start

### Basic Usage

```python
from llm import LLM

llm = LLM(
    model="qwen2.5-coder-7b",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

messages = [
    {"role": "user", "content": "Hello, how are you?"}
]

response = llm.response(messages)
print(response["answer"])
```

### Streaming Response

```python
for event in llm.stream_response(messages):
    if event["type"] == "answer":
        print(event["content"], end="", flush=True)
```

### Async Usage

```python
import asyncio

async def main():
    llm = LLM(model="qwen2.5-coder-7b")
    messages = [{"role": "user", "content": "Tell me a joke"}]

    response = await llm.async_response(messages)
    print(response["answer"])

asyncio.run(main())
```

---

## 📖 Documentation

### Initialization

```python
LLM(
    model: str,
    api_key: str = "lm-studio",
    base_url: str = "http://localhost:1234/v1",
    custom_thinking_token: Optional[CustomThinkingToken] = None,
    default_stop_sequences: Optional[List[str]] = None,
    timeout: float = 300.0,
    extra_body: Optional[Dict[str, Any]] = None,
    tool_exec: bool = True,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | *(required)* | Model identifier (e.g. `"qwen2.5-coder-7b"`) |
| `api_key` | `str` | `"lm-studio"` | API authentication key |
| `base_url` | `str` | `"http://localhost:1234/v1"` | API endpoint URL |
| `custom_thinking_token` | `CustomThinkingToken` | `None` | Custom thinking token configuration |
| `default_stop_sequences` | `List[str]` | `None` | Default stop sequences for generation |
| `timeout` | `float` | `300.0` | Default timeout for sync operations (seconds) |
| `extra_body` | `Dict[str, Any]` | `None` | Extra body fields included in every request |
| `tool_exec` | `bool` | `True` | Whether to execute tools (`True`) or only stream tool calls (`False`) |

### Context Manager

```python
with LLM(model="qwen2.5-coder-7b") as llm:
    response = llm.response(messages)

# Async
async with LLM(model="qwen2.5-coder-7b") as llm:
    response = await llm.async_response(messages)
```

---

### Methods

#### `response()`

Non-streaming synchronous inference.

```python
response = llm.response(
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
) -> FinalResponse
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` | `List[Dict]` | *(required)* | Conversation messages |
| `output_format` | `Dict`, `type`, or `None` | `None` | JSON schema or class for structured output |
| `tools` | `List` | `None` | List of callable functions or tool definition dicts |
| `lm_studio_unload_model` | `bool` | `False` | Unload other models in LM Studio |
| `verbose` | `bool` | `False` | Include timing/token information |
| `hide_thinking` | `bool` | `True` | Hide reasoning tokens from output |
| `reasoning_effort` | `str` | `None` | Reasoning effort level (model-specific) |
| `max_tokens` | `int` | `None` | Maximum tokens to generate |
| `tool_interceptor` | `ToolInterceptor` | `None` | Callback for tool review |
| `extra_body` | `Dict` | `None` | Extra body for this request (overrides instance default) |
| `tool_exec` | `bool` | `None` | Override tool execution (`None` uses instance config) |

**Returns:**

```python
{
    "answer": "Response text or structured data",
    "reasoning": "Thinking process (if hide_thinking=False)",
    "tool_calls": [...],      # All tool calls the model made
    "tool_results": [...],    # Executed tool results
    "verbose": {              # If verbose=True
        "tokens": 123,
        "tokens_per_second": 45.6,
        "latency": 0.12
    }
}
```

---

#### `stream_response()`

Streaming synchronous inference.

```python
for event in llm.stream_response(
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
):
    pass
```

The generator supports `send()` for providing decisions on review requests (see [Tool Interceptors](#tool-interceptors)).

---

#### `async_response()` & `async_stream_response()`

Asynchronous versions with identical parameter signatures.

```python
# Async non-streaming
response = await llm.async_response(messages)

# Async streaming
async for event in llm.async_stream_response(messages):
    print(event)
```

The async stream supports `asend()` for review decisions.

---

### Stream Event Types

All events follow the `StreamEvent` format:

```python
{
    "type": str,              # Event type
    "content": Any,           # Event payload
    "source": Optional[str],  # Tool name (if applicable)
    "tool_id": Optional[str], # Tool call ID (if applicable)
    "job": Optional[int],     # Tool job number (if applicable)
    "depth": int              # Nesting depth (0 = top level)
}
```

| Type | Description | Content |
|------|-------------|---------|
| `"answer"` | Response text chunk (or full structured output) | `str` or parsed object |
| `"reasoning"` | Thinking/reasoning token chunk | `str` |
| `"tool_call"` | Tool invocation detected | `{"id": str, "name": str, "arguments": dict}` |
| `"tool_result"` | Tool execution completed | `{"name": str, "result": Any, "id": str}` |
| `"tool_error"` | Tool execution failed or rejected | `{"name": str, "error": str, "id": str}` |
| `"tool_stream"` | Streaming output from a generator tool | Wrapped chunk from the generator |
| `"review_request"` | Human approval requested for a tool call | `{"id": str, "name": str, "arguments": dict}` |
| `"verbose"` | Timing and token metrics | `{"tokens": int, "tokens_per_second": float, "latency": float}` |
| `"final"` | Aggregated final response (when `final=True`) | `FinalResponse` dict |
| `"done"` | Stream end marker | `None` |

---

### Tool Calling

Define tools as Python functions with type hints:

```python
def get_weather(city: str, unit: str = "celsius") -> dict:
    """Get the current weather for a city.

    Args:
        city: The name of the city
        unit: Temperature unit (celsius or fahrenheit)
    """
    return {"temperature": 22, "condition": "sunny"}

messages = [{"role": "user", "content": "What's the weather in Paris?"}]

response = llm.response(messages=messages, tools=[get_weather])
print(response["tool_results"])
```

You can also use OpenAI standard schema format (dict):

```python
search_tool_schema = {
    "type": "function",
    "function": {
        "name": "search_internet",
        "description": "Search the internet for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "max_results": {"type": "integer", "description": "Max results"}
            },
            "required": ["query"]
        }
    }
}

# Mix both formats
response = llm.response(messages=messages, tools=[get_weather, search_tool_schema])
```

**Tool Flow:**
1. **Callable functions** are auto-converted to OpenAI tool schema via introspection
2. Required parameters are detected via `inspect.signature()`
3. Parameters with non-LLM types (complex objects) are automatically hidden from the schema and injected at call time
4. Default values are included in the schema description
5. Callable tools are **executed automatically** (unless `tool_exec=False`)
6. Results are included in `tool_results`
7. **Dict schemas** are passed through and returned in `tool_calls`
8. You can mix both formats in the same `tools` list

#### Disabling Tool Execution

Set `tool_exec=False` to receive tool calls without executing them:

```python
# Instance-level
llm = LLM(model="qwen2.5-coder-7b", tool_exec=False)

# Or per-request override
response = llm.response(messages, tools=[get_weather], tool_exec=False)
# response["tool_calls"] contains the calls, no execution
```

---

### Tool Interceptors

Tool interceptors enable human-in-the-loop approval before tool execution:

```python
from llm import LLM, tool_interceptor

# Create interceptor for specific tools
interceptor = tool_interceptor("get_weather", "delete_file")

# Or pass callables directly
interceptor = tool_interceptor(get_weather, delete_file)
```

When using streaming, the generator yields a `review_request` event and waits for a decision via `send()`:

```python
gen = llm.stream_response(
    messages=messages,
    tools=[get_weather],
    tool_interceptor=interceptor
)

for event in gen:
    if event["type"] == "review_request":
        # Inspect event["content"] for tool name and arguments
        user_decision = input("Approve? (approve/reject): ")
        event = gen.send(user_decision)  # "approve" to proceed
    elif event["type"] == "answer":
        print(event["content"], end="")
```

You can also write a custom interceptor function:

```python
def my_interceptor(tool_name: str, arguments: dict) -> bool:
    """Return True if tool needs review, False to auto-approve."""
    return tool_name == "dangerous_tool"

response = llm.response(messages, tools=tools, tool_interceptor=my_interceptor)
```

Async interceptors are also supported:

```python
async def async_interceptor(tool_name: str, arguments: dict) -> bool:
    return tool_name in {"delete_file", "send_email"}
```

---

### Generator Tools

Tools can be **generators** (sync or async) to stream intermediate output:

```python
def streaming_search(query: str):
    """Search that streams results as they arrive."""
    for i in range(3):
        yield {"type": "answer", "content": f"Result {i+1} for '{query}'"}
    # The return value becomes the tool result
    return f"Found 3 results for '{query}'"
```

Generator chunks are wrapped in `tool_stream` events with incrementing `depth`:

```python
for event in llm.stream_response(messages, tools=[streaming_search]):
    if event["type"] == "tool_stream":
        print(f"[depth={event['depth']}] {event['content']}")
    elif event["type"] == "tool_result":
        print(f"Final: {event['content']['result']}")
```

Async generators work the same way:

```python
async def async_streaming_tool(query: str):
    """Async generator tool."""
    for i in range(3):
        yield {"type": "answer", "content": f"Result {i+1}"}
    return "Done"
```

Generator tools can also yield `review_request` events for nested human-in-the-loop workflows. The `depth` field tracks nesting level.

---

### Structured Output

Define your output schema as a plain Python class:

```python
from typing import List, Optional, Literal
from llm import LLM

class PersonInfo:
    """Information about a person."""
    name: str = "Unknown"         # Has default → optional in schema
    age: int                      # No default → required
    hobbies: List[str]            # No default → required
    email: Optional[str]          # Optional[T] → implicitly optional
    status: Literal["active", "inactive"]  # Enum constraint

llm = LLM(model="gpt-4")
messages = [
    {"role": "user", "content": "Tell me about Alice, 28, likes coding and reading, active"}
]

response = llm.response(messages, output_format=PersonInfo)
print(response["answer"])
# {"name": "Alice", "age": 28, "hobbies": ["coding", "reading"], "email": null, "status": "active"}
```

**Supported types:**
- ✅ Basic types: `str`, `int`, `float`, `bool`
- ✅ `List[T]` and `Dict[str, T]`
- ✅ `Optional[T]` and `Union[T1, T2]`
- ✅ `Literal[...]` (converted to JSON Schema `enum`)
- ✅ Nested classes with `__annotations__`
- ✅ Class docstrings as schema descriptions
- ✅ Fields with defaults are optional; without defaults are required

**Dict schema passthrough still works:**

```python
schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "person_info",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"],
            "additionalProperties": False
        }
    }
}
response = llm.response(messages, output_format=schema)
```

---

### Multi-Modal (Images)

Images can be provided in four formats. All are automatically converted to base64 for the API.

#### From File Path

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image", "image_path": "/path/to/image.jpg"}
        ]
    }
]

response = llm.response(messages)
```

#### From PIL Image

```python
from PIL import Image

img = Image.open("photo.jpg")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image", "image_pil": img}
        ]
    }
]
```

#### From URL

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What do you see?"},
            {"type": "image", "image_url": "https://example.com/image.jpg"}
        ]
    }
]
```

The URL value can also be a dict for additional options:

```python
{"type": "image", "image_url": {"url": "https://example.com/image.jpg"}}
```

#### From Base64

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze this"},
            {"type": "image", "image_base64": "<base64-encoded-string>"}
        ]
    }
]
```

---

### Thinking Tokens

Models can output reasoning tokens that are parsed and separated. Built-in supported patterns (case-insensitive):

- `<think>...</think>`
- `<thinking>...</thinking>`
- `[THINK]...[/THINK]`

```python
response = llm.response(messages, hide_thinking=False)

print(response["reasoning"])  # "Let me analyze this..."
print(response["answer"])      # "The answer is 42."
```

#### Custom Thinking Tokens

For models with non-standard thinking patterns:

```python
from llm import LLM, CustomThinkingToken

# Custom delimiters
llm = LLM(
    model="custom-model",
    custom_thinking_token=CustomThinkingToken(
        start_token="<<BEGIN_THOUGHT>>",
        end_token="<<END_THOUGHT>>"
    )
)

# Content starts in thinking mode (no start token needed)
llm = LLM(
    model="custom-model",
    custom_thinking_token=CustomThinkingToken(
        from_beginning=True,
        start_token="<<THINK>>",
        end_token="<<END>>"
    )
)
```

**`CustomThinkingToken` parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `from_beginning` | `bool` | `False` | Whether content starts inside thinking mode |
| `start_token` | `str` | `None` | Custom start token pattern |
| `end_token` | `str` | `None` | Custom end token pattern |

Both `start_token` and `end_token` must be provided together.

---

### Verbose Mode

Get token counts, throughput, and latency:

```python
response = llm.response(messages, verbose=True)

print(response["verbose"])
# {"tokens": 150, "tokens_per_second": 42.3, "latency": 0.12}
```

In streaming mode, a `verbose` event is emitted:

```python
for event in llm.stream_response(messages, verbose=True):
    if event["type"] == "verbose":
        info = event["content"]
        print(f"Tokens: {info['tokens']}, Speed: {info['tokens_per_second']:.1f} t/s")
```

---

### Extra Body & Model Parameters

Pass additional parameters to the API request:

```python
# Instance-level (applies to all requests)
llm = LLM(
    model="qwen2.5-coder-7b",
    extra_body={"temperature": 0.7, "top_p": 0.9}
)

# Per-request override
response = llm.response(
    messages,
    extra_body={"temperature": 0.0},
    reasoning_effort="high",
    max_tokens=1024
)
```

---

### LM Studio Integration

```python
llm = LLM(
    model="local-model",
    base_url="http://localhost:1234/v1"
)

# Count tokens
token_count = llm.lm_studio_count_tokens("Hello, world!")
print(f"Tokens: {token_count}")

# Get context length
context_length = llm.lm_studio_get_context_length()
print(f"Max context: {context_length}")

# Auto-unload other models before inference
response = llm.response(
    messages=messages,
    lm_studio_unload_model=True
)
```

---

## 🚨 Error Handling

The SDK defines a structured exception hierarchy:

| Exception | Description |
|-----------|-------------|
| `LLMError` | Base exception for all SDK errors |
| `ConfigurationError` | Invalid configuration (bad params, missing tokens, etc.) |
| `SchemaConversionError` | Schema conversion failure (circular deps, missing annotations) |
| `ToolExecutionError` | Tool execution failure (includes `tool_name` and `original_error`) |
| `ModelRequestError` | API request failure |
| `LLMTimeoutError` | Operation timed out |

```python
from llm import LLM, ModelRequestError, ConfigurationError, LLMTimeoutError

try:
    response = llm.response(messages)
except ModelRequestError as e:
    print(f"API error: {e}")
except ConfigurationError as e:
    print(f"Config error: {e}")
except LLMTimeoutError as e:
    print(f"Timeout: {e}")
```

---

## 💡 Examples

### Complete Chatbot

```python
from llm import LLM

def chat():
    llm = LLM(model="qwen2.5-coder-7b")
    messages = []

    print("Chatbot started. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        messages.append({"role": "user", "content": user_input})

        print("Bot: ", end="", flush=True)
        full_response = ""

        for event in llm.stream_response(messages, final=True):
            if event["type"] == "answer":
                print(event["content"], end="", flush=True)
                full_response += event["content"]
            elif event["type"] == "final":
                full_response = event["content"]["answer"]

        print()
        messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    chat()
```

### Async Batch Processing

```python
import asyncio
from llm import LLM

async def process_batch(prompts: list[str]):
    async with LLM(model="qwen2.5-coder-7b") as llm:
        tasks = [
            llm.async_response([{"role": "user", "content": p}])
            for p in prompts
        ]
        results = await asyncio.gather(*tasks)
        return [r["answer"] for r in results]

prompts = [
    "Translate 'hello' to French",
    "What's 2+2?",
    "Name a color"
]

results = asyncio.run(process_batch(prompts))
for prompt, result in zip(prompts, results):
    print(f"Q: {prompt}\nA: {result}\n")
```

### Agent with Tools and Interceptor

```python
from llm import LLM, tool_interceptor
import datetime

def get_current_time() -> str:
    """Get the current time."""
    return datetime.datetime.now().strftime("%H:%M:%S")

def delete_file(path: str) -> str:
    """Delete a file at the given path.

    Args:
        path: File path to delete
    """
    # os.remove(path)
    return f"Deleted {path}"

llm = LLM(model="qwen2.5-coder-7b")

# Only require approval for dangerous tools
interceptor = tool_interceptor("delete_file")

messages = [
    {"role": "user", "content": "What time is it? Then delete /tmp/test.txt"}
]

# Non-streaming: interceptor auto-rejects if no send() mechanism
response = llm.response(
    messages=messages,
    tools=[get_current_time, delete_file],
    tool_interceptor=interceptor
)

print("Answer:", response["answer"])
for result in response.get("tool_results", []):
    print(f"  {result['name']}: {result['result']}")
```

### Generator Tool with Streaming Output

```python
from llm import LLM

def search_database(query: str):
    """Search database with streaming results."""
    results = []
    for i in range(5):
        result = f"Record {i+1} matching '{query}'"
        results.append(result)
        yield {"type": "answer", "content": result}
    return {"total": len(results), "results": results}

llm = LLM(model="qwen2.5-coder-7b")
messages = [{"role": "user", "content": "Search for 'python'"}]

for event in llm.stream_response(messages, tools=[search_database]):
    if event["type"] == "tool_stream":
        print(f"  [streaming] {event['content']}")
    elif event["type"] == "tool_result":
        print(f"  [final] {event['content']['result']}")
```

---

## 📝 Public API

```python
from llm import (
    # Main class
    LLM,

    # Configuration
    LLMConfig,
    CustomThinkingToken,

    # Type definitions
    ToolInterceptor,     # Callable[[str, Dict], Union[bool, Awaitable[bool]]]
    StreamEvent,         # TypedDict for stream events
    ToolCall,            # TypedDict: {"id", "name", "arguments"}
    ToolResult,          # TypedDict: {"name", "result", "id", "is_error"}
    FinalResponse,       # TypedDict: {"answer", "reasoning", "tool_calls", ...}
    VerboseInfo,         # TypedDict: {"tokens", "tokens_per_second", "latency"}

    # Enums
    EventType,           # ANSWER, REASONING, TOOL_CALL, TOOL_RESULT, etc.

    # Exceptions
    LLMError,
    ConfigurationError,
    SchemaConversionError,
    ToolExecutionError,
    ModelRequestError,
    LLMTimeoutError,

    # Factory
    tool_interceptor,    # Create interceptor from tool names/callables
)
```

### Message Format

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "What's the weather?"}
]
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
