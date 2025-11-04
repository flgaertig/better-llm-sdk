# 🚀 Universal LLM API Wrapper

A versatile Python wrapper built on top of the official `openai` SDK — designed to interface with **any OpenAI-compatible** LLM API, such as **LM Studio**, **Ollama**, or other self-hosted services.

This wrapper simplifies advanced workflows (streaming, multimodal input, function calling, reasoning extraction, etc.) for both **local** and **remote** development.

---

## ✨ Features

- **🔗 OpenAI API Compatibility:** Works with any API exposing the OpenAI schema (`/v1/chat/completions`).
- **⚡ Streaming Support:** Yields structured chunks (`answer`, `reasoning`, `tool_call`, `final`).
- **🧠 Reasoning Extraction**: Separates text inside `<think> … </think>` tags and returns it as a structured "reasoning" field.
- **🧰 Function / Tool Calls:** Aggregates fragmented tool calls into a structured Python list.
- **🖼️ Multimodal Input (`vllm_mode`):** Converts local or in-memory images into Base64 `data:image/png;base64,...` URLs.
- **🧹 LM Studio Model Management:** Optionally unloads other loaded models before inference (`lm_studio_unload_model=True`).

---

## 📦 Installation

```bash
pip install openai
# Optional dependencies:
pip install lmstudio pillow
```

---

## 🧑‍💻 Usage

### 1️⃣ Initialize the Wrapper
```python
from llm_wrapper import LLM

llm = LLM(
    model="openai/gpt-oss-20b",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)
```

### 2️⃣ Basic Chat
```python
messages = [
    {"role": "user", "content": [{"type": "text", "text": "Solve 2+2*3"}]}
]
response = llm.response(messages=messages)
print(response["answer"])
```

**Output:**
```python
{
    "reasoning": "...model reasoning...",
    "answer": "8",
    "tool_calls": []
}
```

---

### 3️⃣ Streaming Responses
```python
for chunk in llm.stream_response(
    messages=messages,
    final=True # adds a final structured summary chunk at the end of the stream
):
    print(chunk["content"])
```

**Chunk examples:**
```python
{"type": "reasoning", "content": "Chain of Thought (if reasoning model)"}
{"type": "answer", "content": "8"}
{"type": "tool_call", "content": {...}}
{"type": "tool_result", "content": {"name": "tool name", "result": "..."}}
{"type": "final", "content": {...}}
{"type":"done", "content": None}
```

---

### 4️⃣ Multimodal Example (Image Input)
```python
llm = LLM(model="qwen3-vl-4b-thinking", vllm_mode=True)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this picture?"},
            {"type": "image", "image_path": "example.png"},
            #{"type": "image", "image_url": "example.com/image.png"}
            #{"type": "image", "image_pil": PIL Image object}
        ]
    }
]

response = llm.response(messages=messages)
print(response["answer"])
```

---

### 5️⃣ Using Tools / Functions
```python

def get_weather(location: str): # converts callable tools in OpenAI-style tool definitions, runs them automatically and returns the result in the response
    """Weather tool, gives weather at location

    param location: string
    return: string with weather information
    """
    return "sunny, 25 degrees Celsius"

sum_tool = { # gives back a tool call
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        }
    }
}

tools=[sum_tool, get_weather]

response = llm.response(messages=messages, tools=tools)
print(response["answer"])
```
---

### 6️⃣ Structured Output
```python
output_format = {
    "type": "json_schema",
    "json_schema": {
        "schema":{
            "type": "object",
            "properties": {
                "name": {
                    "type": "string"
                },
            },
            "required": [
                "name"
            ]
        }
    }
}

response = llm.response(messages=messages, output_format=output_format)
print(response["answer"]) # JSON object if successful
```
Requires an API endpoint that supports the `response_format` parameter (e.g. OpenAI `gpt-4o` or LM Studio with JSON schema support). ⚠️

---

## ⚙️ Parameters Overview

| Parameter | Type | Default | Description |
|------------|------|----------|-------------|
| `model` | `str` | — | The model name (e.g. `"gpt-4"`, `"qwen3-8b"`) |
| `base_url` | `str` | `"http://localhost:1234/v1"` | OpenAI-compatible endpoint |
| `api_key` | `str` | `"lm-studio"` | API key (if required) |
| `vllm_mode` | `bool` | `False` | Enables local image handling for multimodal inputs |
| `lm_studio_unload_model` | `bool` | `False` | Automatically unloads other models in LM Studio |
| `hide_thinking` | `bool` | `True` | If `True`, reasoning (inside `<think>` tags) is hidden from output chunks |

---

## 📄 License

Licensed under the [MIT License](./LICENSE).

