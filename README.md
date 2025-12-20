# LLM SDK

> ⚠️ **This repository is archived.**  
> The SDK has been merged into a new, unified AI SDK with more features, embeddings support, and RAG.  
> Check out the latest version here: [Local AI SDK](https://github.com/flgaertig/Local-AI-SDK)

A universal, production-ready Python sdk for OpenAI-compatible LLM APIs with advanced features including streaming, tool calling, thinking token parsing, and multi-modal support.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ✨ Features

- 🔄 **Streaming & Non-Streaming Support**: Both sync and async implementations
- 🛠️ **Tool Calling**: Automatic function execution with OpenAI function calling
- 🧠 **Thinking Token Parsing**: Extract and handle reasoning tokens (`<think>`, `[THINK]`)
- 🖼️ **Multi-Modal Support**: Image processing with PIL and base64 encoding
- 📦 **Structured Output**: JSON schema validation for responses
- ⚡ **Async/Await**: Full async support for high-performance applications

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
from llm_sdk import LLM

# Initialize the sdk
llm = LLM(
    model="qwen3-vl-4b-thinking",
    base_url="http://localhost:1234/v1",  # Your API endpoint
    api_key="your-api-key"
)

# Simple request
messages = [
    {"role": "user", "content": "Hello, how are you?"}
]

response = llm.response(messages)
print(response["answer"])
```

### Streaming Response

```python
for chunk in llm.stream_response(messages):
    if chunk["type"] == "answer":
        print(chunk["content"], end="", flush=True)
```

### Async Usage

```python
import asyncio

async def main():
    llm = LLM(model="qwen3-vl-4b-thinking")
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
    vllm_mode: bool = False,
    api_key: str = "lm-studio",
    base_url: str = "http://localhost:1234/v1"
)
```

**Parameters:**
- `model`: Model identifier (e.g., `"qwen3-4b"`, `"llama-3.1-8b"`)
- `vllm_mode`: Enable vLLM-specific image processing
- `api_key`: API authentication key
- `base_url`: API endpoint URL

---

### Methods

#### `response()`

Non-streaming synchronous inference.

```python
response = llm.response(
    messages: List[Dict[str, Any]],
    output_format: Dict = None,
    tools: List = None,
    lm_studio_unload_model: bool = False,
    hide_thinking: bool = True
) -> Dict[str, Any]
```

**Parameters:**
- `messages`: List of conversation messages
- `output_format`: JSON schema for structured output
- `tools`: List of callable functions or tool definitions
- `lm_studio_unload_model`: Unload other models in LM Studio
- `hide_thinking`: Hide reasoning tokens from output

**Returns:**
```python
{
    "answer": "Response text or structured data",
    "reasoning": "Thinking process (if hide_thinking=False)",
    "tool_calls": [...],  # Unanswered tool calls
    "tool_results": [...]  # Executed tool results
}
```

---

#### `stream_response()`

Streaming synchronous inference.

```python
for chunk in llm.stream_response(
    messages: List[Dict],
    output_format: Dict = None,
    final: bool = False,
    tools: List = None,
    lm_studio_unload_model: bool = False,
    hide_thinking: bool = True
):
    # Process chunk
    pass
```

**Yields:**
```python
{"type": "reasoning", "content": "..."}   # Thinking tokens
{"type": "answer", "content": "..."}      # Response chunks
{"type": "tool_call", "content": {...}}   # Tool to be called (dict schemas)
{"type": "tool_result", "content": {...}}  # Executed tool result
{"type": "tool_error", "content": {...}}   # Tool execution error
{"type": "final", "content": {...}}       # Final aggregated response
{"type": "done", "content": None}         # Stream end marker
```

---

#### `async_response()` & `async_stream_response()`

Asynchronous versions of the above methods with identical signatures.

```python
# Async non-streaming
response = await llm.async_response(messages)

# Async streaming
async for chunk in llm.async_stream_response(messages):
    print(chunk)
```

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
    # Your implementation
    return {"temperature": 22, "condition": "sunny"}

# You can also use OpenAI standard schema format (dict):
search_tool_schema = {
    "type": "function",
    "function": {
        "name": "search_internet",
        "description": "Search the internet for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return"
                }
            },
            "required": ["query"]
        }
    }
}

# Use both callable functions and dict schemas together
messages = [
    {"role": "user", "content": "What's the weather in Paris?"}
]

response = llm.response(
    messages=messages,
    tools=[get_weather, search_tool_schema]  # Mix both formats
)

print(response)
# Callable tools (get_weather) are executed automatically
# Dict schemas are returned in tool_calls for manual handling
```

**Tool Flow:**
1. **Callable functions** are auto-converted to OpenAI tool schema
2. Required parameters are detected via `inspect.signature()`
3. Callable tools are **executed automatically**
4. Results are included in `tool_results`
5. **Dict schemas** are passed through and returned in `tool_calls`
6. You can mix both formats in the same `tools` list
7. **Async tools** in sync methods yield `tool_error` (use async methods instead)


---

### Structured Output

Define your output schema as a plain Python class:

```python
from typing import List, Optional
from llm_sdk import LLM

class PersonInfo:
    """Information about a person."""
    name: str = "Unknown"  # Has default → optional
    age: int               # Required (no default)
    hobbies: List[str]     # Required (no default)
    email: Optional[str]   # Implicitly optional (None default)

llm = LLM(model="gpt-4")
messages = [
    {"role": "user", "content": "Tell me about Alice, 28, likes coding and reading"}
]

response = llm.response(messages, output_format=PersonInfo)
print(response["answer"])
# Output: {"name": "Alice", "age": 28, "hobbies": ["coding", "reading"], "email": null}
```

**Features:**
- ✅ **Required fields**: No default value defined
- ✅ **Optional fields**: Has default value defined
- ✅ **Nested classes**: Automatically converted
- ✅ **Lists and Dicts**: Full support
- ✅ **Class docstring**: Added to schema description

**Still works - Dict Schema:**

```python
schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "person_info",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }
    }
}
response = llm.response(messages, output_format=schema)
```

---

### Multi-Modal (Images)

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

llm = LLM(model="gpt-4-vision", vllm_mode=True)
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

response = llm.response(messages)
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

---

### Thinking Tokens

Models can output reasoning tokens that are parsed and separated:

```python
# Model outputs: <think>Let me analyze this...</think>The answer is 42.
# Or: [THINK]Reasoning here[/THINK]Answer here.

response = llm.response(messages, hide_thinking=False)

print(response["reasoning"])  # "Let me analyze this..." or "Reasoning here"
print(response["answer"])      # "The answer is 42." or "Answer here."
```

Supported formats:
- `<think>...</think>`
- `[THINK]...[/THINK]`
- Case-insensitive

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

## 💡 Examples

### Complete Chatbot Example

```python
from llm_sdk import LLM

def chat():
    llm = LLM(model="gpt-4")
    messages = []
    
    print("Chatbot started. Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        
        messages.append({"role": "user", "content": user_input})
        
        print("Bot: ", end="", flush=True)
        full_response = ""
        
        for chunk in llm.stream_response(messages, final=True):
            if chunk["type"] == "answer":
                print(chunk["content"], end="", flush=True)
                full_response += chunk["content"]
            elif chunk["type"] == "final":
                full_response = chunk["content"]["answer"]
        
        print()  # New line
        messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    chat()
```

### Async Batch Processing

```python
import asyncio
from llm_sdk import LLM

async def process_batch(prompts: list[str]):
    llm = LLM(model="gpt-4")
    
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

### Agent with Tools

```python
from llm_sdk import LLM
import datetime

def get_current_time() -> str:
    """Get the current time."""
    return datetime.datetime.now().strftime("%H:%M:%S")

def set_reminder(message: str, minutes: int) -> str:
    """Set a reminder.
    
    Args:
        message: The reminder message
        minutes: Minutes until reminder
    """
    return f"Reminder '{message}' set for {minutes} minutes from now"

llm = LLM(model="gpt-4")

messages = [
    {"role": "user", "content": "What time is it? And remind me to call mom in 30 minutes"}
]

response = llm.response(
    messages=messages,
    tools=[get_current_time, set_reminder]
)

print("Answer:", response["answer"])
print("\nExecuted Tools:")
for result in response.get("tool_results", []):
    print(f"  {result['name']}: {result['result']}")
```

---

## 🔧 Advanced Configuration

### Custom Error Handling

```python
try:
    response = llm.response(messages)
except ValueError as e:
    print(f"Validation error: {e}")
except RuntimeError as e:
    print(f"API error: {e}")
```

### Message Format

```python
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "Hello!"
    },
    {
        "role": "assistant",
        "content": "Hi! How can I help?"
    },
    {
        "role": "user",
        "content": "What's the weather?"
    }
]
```

---

## 🧪 Testing

```python
# test_llm_sdk.py
from llm_sdk import LLM

def test_basic_response():
    llm = LLM(model="gpt-4", base_url="http://localhost:1234/v1")
    messages = [{"role": "user", "content": "Say 'test'"}]
    response = llm.response(messages)
    
    assert "answer" in response
    assert isinstance(response["answer"], str)

def test_tool_execution():
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
    
    llm = LLM(model="gpt-4")
    messages = [{"role": "user", "content": "What is 5 + 3?"}]
    response = llm.response(messages, tools=[add])
    
    assert "tool_results" in response or "answer" in response
```

---

## 📝 API Reference

### Type Definitions

```python
Message = Dict[str, Any]  # {"role": str, "content": str | List[Dict]}
Tool = Callable | Dict[str, Any]
Schema = Dict[str, Any]
Response = Dict[str, Any]  # {"answer": Any, "reasoning"?: str, ...}
Chunk = Dict[str, Any]  # {"type": str, "content": Any}
```

### Helper Methods

| Method | Description |
|--------|-------------|
| `_get_json_type(python_type)` | Convert Python type to JSON Schema type |
| `_prepare_tools(tools)` | Convert callables to OpenAI format |
| `_process_images(messages)` | Convert images to base64 |
| `_parse_thinking_content(content, inside_think)` | Extract thinking tokens |
| `_unload_other_models()` | Unload LM Studio models |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
