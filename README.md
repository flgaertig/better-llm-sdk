# 🚀 Universal LLM API Wrapper

A versatile Python wrapper built on the official `openai` SDK — designed to interface with **any OpenAI-compatible LLM API**, such as **LM Studio**, **vLLM**, or other self-hosted services.

This wrapper simplifies advanced workflows (streaming, multimodal input, function calling, reasoning extraction, etc.) for both **local** and **remote** development.

---

## ✨ Features

- **🔗 OpenAI API Compatibility:** Works with any API exposing the OpenAI schema (`/v1/chat/completions`).
- **⚡ Streaming Support:** Yields structured chunks (`answer`, `reasoning`, `tool_call`, `final`).
- **🧠 Reasoning Extraction:** Detects and separates model reasoning enclosed in `<think>`…`</think>` tags.
- **🧰 Function / Tool Calls:** Aggregates fragmented tool calls into a structured Python list.
- **🖼️ Multimodal Input (`vllm_mode`):** Converts local or in-memory images into Base64 `data:image/png;base64,...` URLs.
- **🧹 LM Studio Model Management:** Optionally unloads other loaded models before inference (`lm_studio_unload_model=True`).

---

## 📦 Installation

```bash
pip install openai pillow
# optional
pip install lmstudio
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
print(response)
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
for chunk in llm.response(messages=messages, stream=True, final=True):
    print(chunk)
```

**Chunk examples:**
```python
{"type": "reasoning", "content": "Thinking..."}
{"type": "answer", "content": "8"}
{"type": "tool_call", "content": {...}}
{"type": "final", "content": {...}}
```

---

### 4️⃣ Multimodal Example (Image Input)
```python
llm = LLM(model="qwen3-vl-4b-thinking", vllm_mode=True)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What’s in this picture?"},
            {"type": "image", "image_path": "example.png"}
        ]
    }
]

response = llm.response(messages=messages)
print(response)
```

---

### 5️⃣ Using Tools / Functions
```python
tools = [
    {
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
]

response = llm.response(messages=messages, tools=tools)
print(response)
```

---

## ⚙️ Parameters Overview

| Parameter | Type | Default | Description |
|------------|------|----------|-------------|
| `model` | `str` | — | The model name (e.g. `"gpt-4"`, `"local-llm"`) |
| `base_url` | `str` | `"http://localhost:1234/v1"` | OpenAI-compatible endpoint |
| `api_key` | `str` | `"lm-studio"` | API key (if required) |
| `vllm_mode` | `bool` | `False` | Enable local image handling for multimodal inputs |
| `lm_studio_unload_model` | `bool` | `True` | Automatically unload other models in LM Studio |
| `stream` | `bool` | `False` | Return generator chunks instead of a full dict |
| `final` | `bool` | `False` | Add a final summary chunk at end of stream |

---

## 📄 License

Licensed under the [MIT License](./LICENSE).

