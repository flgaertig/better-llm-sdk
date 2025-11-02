🚀 Universal LLM API Wrapper
A robust and flexible Python wrapper designed to interact with any Large Language Model (LLM) server that exposes an OpenAI-compatible API, such as LM Studio, vLLM, OLLAMA, or even the official OpenAI service.

It simplifies advanced tasks like streaming output, multi-modal input (vision models), and function/tool calling.

✨ Features
Universal Compatibility: Connects to any endpoint via base_url (defaults to LM Studio).

Streaming Support: Efficiently processes and yields chunks for real-time output.

Tool Calling: Accumulates and reconstructs fragmented tool call data from the stream.

Multimodal (V-LLM) Ready: Converts local image paths (image_path) and PIL objects (image_pil) into the required Base64 data URL format.

Reasoning Extraction: Parses <think> tags (if emitted by the model) to separate internal reasoning from the final answer.

LM Studio Integration: Optional feature to unload currently loaded models to free up VRAM before making a request to the configured model.

⬇️ Installation
This wrapper requires the following core dependencies:

Bash

pip install openai pillow
If you plan to use the lm_studio_unload=True feature, you'll also need:

Bash

pip install lmstudio
🧑‍💻 Usage Example
The LLM class is initialized with the model name, and the base_url and api_key can be configured for any compatible server.

Python

from llm_wrapper import LLM # Assuming you save your class in llm_wrapper.py

# 1. Initialize for a local server (e.g., LM Studio default)
llm = LLM(
    model="TheBloke/My-Awesome-Model-GGUF", 
    base_url="http://localhost:1234/v1",
    vllm_mode=True # Enable image path/PIL object conversion
)

# 2. Define a multi-content message (text and image path)
# Note: For multimodal models, the message content must be a list.
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What color is the cat in this image?"},
            {"type": "image", "image_path": "/path/to/your/cat_picture.jpg"}
        ]
    }
]

# 3. Stream the response
print("LLM Response (Streaming):\n---")
for chunk in llm.response(messages=messages, stream=True, lm_studio_unload=False):
    # Print the streamed answer content
    if chunk["type"] == "answer":
        print(chunk["content"], end = "", flush=True)

print("\n\n--- End of Stream ---")
🚨 Robustness and Error Handling
The wrapper utilizes the exception handling of the openai client. If a network issue, server error, or invalid model name is encountered, the generator will yield an {"type": "error", "content": "..."} chunk before terminating.

Important Note on LM Studio: The lm_studio_unload=True feature requires the separate lmstudio Python package. If this package is not installed and the feature is requested, the wrapper will print a warning and automatically skip the unloading logic.

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.
