import json
import base64
from openai import OpenAI
import io
from typing import Any
from PIL import Image

class LLM:
    """Universal LLM API Wrapper compatible with OpenAI-style APIs."""
    def __init__(self,model:str, vllm_mode:bool=False, api_key:str="lm-studio", base_url:str="http://localhost:1234/v1"):
        """initialize the wrapper"""
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.vllm_mode = vllm_mode

    def response(self,messages:list[dict[str, Any]]=None,output_format:dict=None,tools:list[dict[str, Any]]=None,lm_studio_unload_model:bool=False):
        """request model inference"""

        if messages is None:
            raise ValueError("messages must be provided")
        
        response = self.stream_response(
            messages=messages,
            output_format=output_format,
            final=True,
            tools=tools,
            lm_studio_unload_model=lm_studio_unload_model
        )

        for r in response:
            if r["type"] == "final":
                return r["content"]
                

    def stream_response(self,messages:list[dict[str, Any]]=None,output_format:dict=None,final:bool=False,tools:list[dict[str, Any]]=None,lm_studio_unload_model:bool=False):
        """request model inference"""

        if messages is None:
            raise ValueError("messages must be provided")

        if self.vllm_mode:
            for msg in messages:
                for i in range(len(msg["content"])):
                    c = msg["content"][i]
                    if c["type"] == "image":
                        if "image_path" in c:
                            img = Image.open(c["image_path"])
                            buffer = io.BytesIO()
                            img.save(buffer, format="PNG")
                            img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                            msg["content"][i] = {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                            }
                        elif "image_pil" in c:
                            img = c["image_pil"]
                            buffer = io.BytesIO()
                            img.save(buffer, format="PNG")
                            img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                            msg["content"][i] = {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                            }
                        elif "image_url" in c:
                            url_data = c["image_url"]
                            if isinstance(url_data, str):
                                url_data = {"url": url_data}
                            msg["content"][i] = {"type": "image_url", "image_url": url_data}
        
        if lm_studio_unload_model:
            import lmstudio as lms
            lms.configure_default_client("localhost:1234")
            all_loaded_models = lms.list_loaded_models()
            for loaded_model in all_loaded_models:
                if loaded_model != self.model:
                    model = lms.llm(loaded_model)
                    model.unload()

        structured_output = output_format is not None

        if structured_output:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    tools=tools if tools is not None else [],
                    response_format=output_format if output_format is not None else None,
                )
            except Exception as e:
                raise RuntimeError(f"Model request failed: {e}")
        else:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    tools=tools if tools is not None else [],
                )
            except Exception as e:
                raise RuntimeError(f"Model request failed: {e}")

        thinking = ""
        answer = ""
        tool_calls_accumulator = {}
        inside_think = False

        for chunk in completion:
            x = chunk.choices[0].delta
            if not x:
                continue

            reasoning = getattr(x, "reasoning", None)
            content = getattr(x, "content", None)
            tool_calls = getattr(x, "tool_calls", None)

            if not (content or tool_calls or reasoning):
                continue

            if content:
                if "<think>" in content:
                    inside_think = True
                    content = content.replace("<think>", "")
                if "</think>" in content:
                    inside_think = False
                    content = content.replace("</think>", "")
                if inside_think:
                    thinking += content
                    yield {"type": "reasoning", "content": content}
                else:
                    answer += content
                    if not structured_output:
                        yield {"type": "answer", "content": content}

            if reasoning:
                thinking += reasoning
                yield {"type": "reasoning", "content": reasoning}
                
            if tool_calls:
                for tool_call in tool_calls:
                    tool_id = tool_call.id or "0"
                    func = tool_call.function
                    if tool_id not in tool_calls_accumulator:
                        tool_calls_accumulator[tool_id] = {"name": func.name or "", "arguments": ""}
                    if func.name:
                        tool_calls_accumulator[tool_id]["name"] = func.name
                    if func.arguments:
                        tool_calls_accumulator[tool_id]["arguments"] += func.arguments

        if structured_output:
            temp_answer = answer
            try:
                data = json.loads(answer)
            except json.JSONDecodeError:
                try:
                    decoded = answer.encode('utf-8').decode('unicode_escape')
                    data = json.loads(decoded)
                except Exception:
                    data = temp_answer
            answer = data
            yield {"type": "answer", "content": answer}

        final_tool_calls = []
        for tool_id, data in tool_calls_accumulator.items():
            try:
                args = json.loads(data["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {"_raw": data["arguments"] or ""}
            final_tool_calls.append({"id": tool_id, "name": data["name"], "arguments": args})

        for tool_call in final_tool_calls:
            yield {"type": "tool_call", "content": tool_call}

        if final:
            yield {"type": "final", "content": {
                "reasoning": thinking,
                "answer": answer,
                "tool_calls": final_tool_calls
                }    
            }
        yield {"type": "done", "content": None}
    
    def lm_studio_count_tokens(self,input_text: str) -> int:
        """count tokens used in lm studio"""
        import lmstudio as lms
        lms.configure_default_client("localhost:1234")
        model = lms.llm(self.model)
        token_count = len(model.tokenize(input_text))
        return token_count
    
    def lm_studio_get_context_length(self) -> int:
        import lmstudio as lms
        lms.configure_default_client("localhost:1234")
        model = lms.llm(self.model)
        return model.get_context_length()

