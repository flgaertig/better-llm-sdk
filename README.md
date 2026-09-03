# 📦 llm-sdk

Small Python SDK for OpenAI-compatible LLM APIs.

One file, clean API, boring on purpose. Use it with local servers, OpenAI-style endpoints, structured output, tool calls, multimodal inputs, and reasoning streams.

<img width="1280" height="640" alt="main" src="https://github.com/user-attachments/assets/49f08e18-b6ae-4948-ab17-af18e329c6ec" />

## ✨ Features

- Sync and async clients
- Streaming and non-streaming responses
- Configurable retry with per-call override
- OpenAI Chat Completions support
- Optional OpenAI Responses API mode
- Structured output from JSON schema or typed Python classes
- Tool schema generation from Python callables
- Vision input normalization from URL, path, base64, or PIL image
- Audio, video & file inputs with the same path/base64/URL pattern
- Reasoning token parsing
- Lightweight verbose stats for streams

## 🚀 Get Started

Install the package directly from PyPI:

```bash
pip install llm-sdk-py
```

If you also need PIL image support:

```bash
pip install "llm-sdk-py[pillow]"
```

Alternatively, since it's designed to be simple, you can still just drop `llm_sdk.py` directly into your project!

```python
from llm_sdk import LLM

llm = LLM(
    model="qwen3.6-27b",
    base_url="http://localhost:1234",
    api_key="lm-studio",
)

response = llm.response(system="You're a helpful assistant!",input="Write a tiny haiku about fast code.")

print(response["answer"])
```

By default, `base_url="http://localhost:1234/v1"` and `api_key="lm-studio"`, so local LM Studio-style servers just work. `base_url` must be http(s) without credentials; `?api-version=`-style queries are fine.

All inference methods accept either `input="..."` for the common single-user-message case or a Chat Completions-style message list + `system` for a system prompt:

```python
response = llm.response(messages=[
    {"role": "user", "content": "Write a tiny haiku about fast code."},
])
```

## 📡 Streaming

```python
for event in llm.stream_response(input="Explain adapters in one paragraph."):
    if event["type"] == "answer":
        print(event["content"], end="", flush=True)
```

Events are small dictionaries:

```python
{"type": "answer", "content": "..."}
{"type": "reasoning", "content": "..."}
{"type": "refusal", "content": "..."}
{"type": "tool_call", "content": {"id": "...", "name": "...", "arguments": {...}, "callable": Callable}}
{"type": "tool_call_part", "content": {"id": "...", "name": "...", "args_delta": "{\"city\": \"Berlin\"}"}}
{"type": "verbose", "content": {"tokens": 42, "chunks": 42, "tokens_per_second": 91.3, "latency": 0.2, "prompt_tokens": 10, "completion_tokens": 32, "total_tokens": 42, "stop_reason": "stop"}}
{"type": "final", "content": {"answer": "...", "reasoning": "...", "stop_reason": "stop"}}
{"type": "done", "content": None}
```

Use `final=True` for a final aggregated event. Reasoning is included by default; `include_reasoning=False` hides it.

## ⏱️ Async

```python
import asyncio
import os
from llm_sdk import LLM

async def main():
    async with LLM(model="gpt-5.5", api_key=os.environ["OPENAI_API_KEY"], base_url="https://api.openai.com/v1", use_responses_api=True) as llm:
        response = await llm.async_response(input="Give me a crisp project name.")
        print(response["answer"])

asyncio.run(main())
```
## 🔄 Retry

Set `max_retries` globally or override it per call.

```python
# global
llm = LLM(model="qwen3.6-27b", max_retries=5)
# per call
llm.response(input="...", max_retries=0)
```

## 📐 Structured Output

Pass a JSON schema or a typed class (enums, literals, optional fields included). Parsing is always strict: bad JSON raises `StructuredOutputError` (original text in `.raw`).

```python
class Verdict:
    sentiment: str
    score: float
    tags: list[str]

result = llm.response(
    input="Review: fast, small, surprisingly nice.",
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
    input="Find the auth setup notes.",
    tools=[search_docs],
)

print(response.get("tool_calls", []))
```

For multi-turn conversation loops with tool calls, use the built-in helper functions to easily format messages:

```python
from llm_sdk import assistant_message, system_message, tool_result, user_message

# 0. Optional system message
msg0 = system_message("You are a helpful assistant.")

# 1. Format the assistant's response (includes both answer text and tool calls;
#    reasoning is dropped by default — pass include_reasoning=True to keep it)
msg1 = assistant_message(response)

# 2. Format a tool call execution result
msg2 = tool_result(response["tool_calls"][0], "result string or dict")

# 3. Format subsequent user messages
msg3 = user_message("Tell me more about the results.")
```

In Responses mode, `assistant_message(response)` also carries `response_items`, so the next turn replays correctly — just append it.


## 👁️ Vision

Image content can be a URL, a local path, base64, or a PIL image.

```python
response = llm.response([
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image", "image_path": "photo.png"},
        ],
    }
])
```

Supported image forms include:

- `{"type": "image", "image_url": "https://..."}`
- `{"type": "image", "image_path": "local-file.png"}`
- `{"type": "image", "image_base64": "..."}` (MIME type auto-detected from the bytes)
- `{"type": "image", "image_pil": image}`
- plus optional `detail` (`low`/`high`/`auto`; `original` is Responses-only)

Local paths (`str` or `pathlib.Path`) just work. Oversized images shrink automatically instead of failing.

## 🎞️ Audio, Video & Files

Same pattern for other modalities (input only):

- Audio: `{"type": "audio", "audio_path": "clip.wav"}` (also `audio_base64`, `audio_url`) → `input_audio` (wav/mp3/aiff/aac/ogg/flac/m4a, max 25MB; OpenAI itself takes wav/mp3 only)
- Video: `{"type": "video", "video_path": "clip.mp4"}` (also `video_base64`, `video_url`, optional `processing`) → `video_url` (mp4/mov/webm, max 100MB)
- File: `{"type": "file", "file_path": "doc.pdf"}` (also `file_base64`, `file_url`, `file_id`, optional `filename`/`detail`) → `file` part (max 50MB inline; `detail` is Responses-only)

Remote URLs pass through untouched (http(s) only, no credentials). Bad content raises `Image/Audio/Video/FileProcessingError`; bad types/URLs raise `ConfigurationError`.

## 🔌 Responses API

Use `use_responses_api=True` for endpoints that prefer OpenAI's Responses API shape.

```python
import os

llm = LLM(
    model="gpt-5.5",
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.openai.com/v1",
    use_responses_api=True,
)
```

Remote files map to `input_file` with `file_url`, uploads via `file_id`. Limits: no video, no remote audio URLs (base64 `input_audio` only) – both raise `ConfigurationError`.

## 🧠 Reasoning Effort

Use `reasoning_effort="high"` — or `reasoning_budget=2000` for a token budget (mutually exclusive). Unknown values pass through with a warning.

```python
response = llm.response(
    input="...",
    reasoning_effort="high"
)
```

## ⚙️ API

- `response(...)` and `stream_response(...)`
- `async_response(...)` and `async_stream_response(...)`
- `input="..."` or `messages=[...]` for all inference methods
- `LLM.list_models(...)` / `LLM.async_list_models(...)` and standalone `list_models(...)` / `async_list_models(...)`
- `max_retries=3` globally on `LLM(...)` or per call
- `reasoning_effort="high"` where supported (`reasoning_budget=N` as token-budget alternative; mutually exclusive)
- `temperature/top_p/max_tokens/stop/seed/user/tool_choice/store/extra_body` per call (`seed`/`stop` ignored with a warning in Responses mode)
- `extra_body` bypasses validation (escape hatch); `model`/`messages`/`input`/`stream` are rejected. On `api.openai.com` reasoning models, `stop`/`temperature`/`top_p` are dropped and `max_tokens` maps to `max_completion_tokens`
- `schema_strict=False` relaxes the generated schema only (fewer `required`) – parsing stays strict
- `LLMConfig(...)` + `LLM.from_config(config)` for config-object style; `default_stop_sequences=[...]`; `normalize_base_url=False` keeps `base_url` as given
- `configure_debug_logging()` / `configure_quiet_logging()` for explicit log control (beyond `debug=True`)
- `include_reasoning=False` to hide reasoning content (shown by default)
- `max_image_side=8192` caps the longest image side (`None` disables, byte budget still applies)
- `CustomReasoningPattern(...)` for custom `<think>`-style parsing
- `verbose=True` for stream stats (incl. `stop_reason`); without server `usage`, `tokens` falls back to chunk count
- `debug=True` for SDK debug logs (quiet by default)
- `stop_reason`: `"stop" | "length" | "tool_calls" | "content_filter" | "refusal"` (plus `"failed"`/`"cancelled"`/`"incomplete"` in Responses mode)
- message helpers: `system_message(...)`, `user_message(...)`, `assistant_message(...)`, `tool_result(...)`
- `with LLM(...) as llm:` / `async with LLM(...) as llm:` for cleanup
- `from llm_sdk import __version__` for the package version

## 💡 Why

Most LLM wrappers either become frameworks or stay too close to raw HTTP. This sits in the middle: enough structure to be pleasant, little enough surface area to understand in one sitting.

If this saves you time, please ⭐ the repo! Thanks! ♥️

## 📜 License

MIT
