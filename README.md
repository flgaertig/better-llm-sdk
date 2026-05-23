# 📦 llm-sdk

Small Python SDK for OpenAI-compatible LLM APIs.

One file, clean API, boring on purpose. Use it with local servers, OpenAI-style endpoints, structured output, tool calls, vision inputs, and reasoning streams.

## ✨ Features

- Sync and async clients
- Streaming and non-streaming responses
- OpenAI Chat Completions support
- Optional OpenAI Responses API mode
- Structured output from JSON schema or typed Python classes
- Tool schema generation from Python callables
- Vision input normalization from URL, path, base64, or PIL image
- Thinking/reasoning token parsing
- Lightweight verbose stats for streams

## 🚀 Get Started

Install the only required dependency:

```bash
pip install openai
```

Optional, only for PIL image inputs:

```bash
pip install pillow
```

Drop `llm_sdk.py` into your project or import it from this repo:

```python
from llm_sdk import LLM

llm = LLM(
    model="qwen3.6-27b",
    base_url="http://localhost:1234",
    api_key="lm-studio",
)

response = llm.response([
    {"role": "user", "content": "Write a tiny haiku about fast code."}
])

print(response["answer"])
```

By default, `base_url="http://localhost:1234/v1"` and `api_key="lm-studio"`, so local LM Studio-style servers work with very little setup.

## 📡 Streaming

```python
for event in llm.stream_response([
    {"role": "user", "content": "Explain adapters in one paragraph."}
]):
    if event["type"] == "answer":
        print(event["content"], end="", flush=True)
```

Events are small dictionaries:

```python
{"type": "answer", "content": "..."}
{"type": "reasoning", "content": "..."}
{"type": "tool_call", "content": {"id": "...", "name": "...", "arguments": {...}}}
{"type": "verbose", "content": {"tokens": 42, "tokens_per_second": 91.3, "latency": 0.2, "prompt_tokens": 10, "completion_tokens": 32, "total_tokens": 42}}
{"type": "final", "content": {"answer": "..."}}
{"type": "done"}
```

Use `final=True` if you also want a final aggregated response event.

## ⏱️ Async

```python
import asyncio
from llm_sdk import LLM

async def main():
    async with LLM(model="gpt-5.5", api_key="sk-...", base_url="https://api.openai.com/v1",use_responses_api=True) as llm:
        response = await llm.async_response([
            {"role": "user", "content": "Give me a crisp project name."}
        ])
        print(response["answer"])

asyncio.run(main())
```

## 📐 Structured Output

Pass a JSON schema or a typed class. Classes are converted into OpenAI-compatible JSON schema.

```python
class Verdict:
    sentiment: str
    score: float
    tags: list[str]

result = llm.response(
    [{"role": "user", "content": "Review: fast, small, surprisingly nice."}],
    output_format=Verdict,
)

print(result["answer"])
```

## 🛠️ Tools

Pass Python callables or already-built OpenAI tool definitions. The SDK exposes tool definitions and returns streamed/final tool calls.

It does not execute tools for you. You stay in control.

```python
def search_docs(query: str, limit: int = 5) -> str:
    """Search internal docs."""
    return "..."

response = llm.response(
    [{"role": "user", "content": "Find the auth setup notes."}],
    tools=[search_docs],
)

print(response.get("tool_calls", []))
```

## 👁️ Vision

Image content can be a URL, a local path, base64, or a PIL image.

```python
response = llm.response([
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_path", "image_path": "photo.png"},
        ],
    }
])
```

Supported image forms include:

- `{"type": "image_url", "image_url": "https://..."}`
- `{"type": "image_path", "image_path": "local-file.png"}`
- `{"type": "image_base64", "image_base64": "..."}`
- `{"type": "image_pil", "image_pil": image}`

## 🔌 Responses API

Use `use_responses_api=True` for endpoints that prefer OpenAI's Responses API shape.

```python
llm = LLM(
    model="gpt-5.5",
    api_key="sk-...",
    base_url="https://api.openai.com/v1",
    use_responses_api=True,
)
```

## 🧠 Reasoning Effort

Use `reasoning_effort="high"` to set models's reasoning effort.

```python
response = llm.response(
    [{"role": "user", "content": "..."}],
    reasoning_effort="high"
)
```

## ⚙️ API

- `response(...)` and `stream_response(...)`
- `async_response(...)` and `async_stream_response(...)`
- `list_models(fallback=[...])`
- `reasoning_effort="low|medium|high"` where supported
- `hide_thinking=False` to stream/return reasoning content
- `CustomThinkingToken(...)` for custom `<think>`-style parsing
- `verbose=True` for token-ish stream stats
- `with LLM(...) as llm:` / `async with LLM(...) as llm:` for cleanup

## 💡 Why

Most LLM wrappers either become frameworks or stay too close to raw HTTP. This sits in the middle: enough structure to be pleasant, little enough surface area to understand in one sitting.

## 📜 License

MIT
