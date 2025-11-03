# 🚀 Universal LLM API Wrapper

A versatile Python wrapper built on the official `openai-python` SDK. It is designed to interface with any Large Language Model (LLM) server that exposes an **OpenAI-compatible API** (e.g., local servers like LM Studio, vLLM, etc., or remote services).

This wrapper focuses on simplifying advanced tasks and local development.

### ✨ Key Features

* **OpenAI API Compatibility:** Uses the standard `openai.OpenAI` client with configurable `base_url` and `api_key` in the constructor.
* **Streaming Output:** Implements a streaming generator to process responses in chunks, separating them into **"answer"**, **"reasoning"**, and **"tool_call"** types.
* **Tool/Function Calling:** Accumulates and reconstructs fragmented tool call information from the stream, including the function name and JSON arguments.
* **Multimodal Support (`vllm_mode`):** When enabled, it converts local image sources (paths via `"image_path"` or PIL objects via `"image_pil"`) into **Base64-encoded data URLs**, compatible with multimodal LLMs that follow the OpenAI Vision format.
* **Reasoning/CoT Extraction:** Separates internal thinking from the main response by detecting and processing content between custom **`<think>`** and **`</think>`** tags.
* **LM Studio Unloading (Optional):** Includes specific logic (`lm_studio_unload=True`) using the `lmstudio` package to automatically check and unload other loaded models to prepare for the configured model.

### ⬇️ Dependencies

This project requires the `openai` SDK and `Pillow` for image handling:

```bash
pip install openai pillow
```
If you plan to use the lm_studio_unload_model=True feature, you'll also need:
```bash
pip install lmstudio
```

### 🧑‍💻 Usage Example

#### 1. Initialize the Wrapper
```python
from llm_wrapper import LLM

llm = LLM(
    model="openai/gpt-oss-20b",
    base_url="your base url", #(defaults to LM Studio settings)
    api_key="your api key", #(defaults to LM Studio settings)
)
```

#### 2. Get response
```
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Solve the problem 2+2*3"}
        ]
    }
]
response = llm.response(messages=messages)
print(response)
```

#### Response Format
```
chunk = {"type":"reasoning/answer/tool_call","content":"your content"}
```

#### 3. Image Input
```
llm = LLM(
    model="qwen3-vl-4b-thinking", 
    vllm_mode=True, # Enable local image handling (e.g., image_path)
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Solve the problem in the picture"}
            {"type": "image", "image_path":"your local image path"}
            # {"type": "image", "image_url":"your image url"}
            # {"type": "image", "image_pil":"your PIL Image object"}
        ]
    }
]

response = llm.response(messages=messages)
print(response)
```

#### 4. Streaming
```
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Solve the problem 2+2*3"}
        ]
    }
]
response = llm.response(
    messages=messages,
    stream=True,
    # final=True # includes a final chunk with answer, reasoning and tool calls in one piece
)
for chunk in response:
    print(chunk["content"], end = "",flush=True)
```
#### Chunk Format
```
chunk = {"type":"reasoning/answer/tool_call","content":"your content"}
```

#### 5. LM Studio Model Unload
```
response = llm.response(
    messages=messages,
    lm_studio_model_unload = True, # unloads other models to save memory
)
print(response)
```

#### 6. Using TOOLS
```
tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_sum",
                    "description": "Get sum of two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["a", "b"]
                    },
                }
            }
        ]

response = llm.response(
    messages=messages,
    tools=tools
)
print(response)
```

### 📄 License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.
