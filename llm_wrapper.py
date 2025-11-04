import json
import base64
from openai import OpenAI
import io
from PIL import Image
from typing import Callable, Any

class LLM:
    """Universal LLM API Wrapper compatible with OpenAI-style APIs."""
    def __init__(self,model:str, vllm_mode:bool=False, api_key:str="lm-studio", base_url:str="http://localhost:1234/v1"):
        """initialize the wrapper"""
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.vllm_mode = vllm_mode

    def response(self,messages:list[dict[str, Any]]=None,output_format:dict=None,tools:list=None,lm_studio_unload_model:bool=False,hide_thinking:bool=True):
        """request model inference"""

        if messages is None:
            raise ValueError("messages must be provided")
        
        response = self.stream_response(
            messages=messages,
            output_format=output_format,
            final=True,
            tools=tools,
            lm_studio_unload_model=lm_studio_unload_model,
            hide_thinking=hide_thinking,
        )

        for r in response:
            if r["type"] == "final":
                return r["content"]
                
    def stream_response(self,messages:list[dict]=None,output_format:dict=None,final:bool=False,tools:list=None,lm_studio_unload_model:bool=False,hide_thinking:bool=True):
        """request model inference"""
        _tools = list(tools) if tools else []
        callable_tools = {}
        if _tools:
            types = {
                        "str":"string",
                        "int":"integer",
                        "float":"number",
                        "bool":"boolean",
                        "list":"array",
                        "dict":"object"
                    }
            for i in range(len(_tools)):
                if isinstance(_tools[i],Callable):
                    name = _tools[i].__name__.strip()
                    doc = _tools[i].__doc__.strip()
                    param = _tools[i].__annotations__
                    required_params = []
                    parameters = {}
                    for k,v in param.items():
                        if k == "return":
                            continue
                        else:
                            parameters[str(k)] = {"type": types[v.__name__]}
                            required_params.append(str(k))
                    _tools[i] = {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": doc,
                            "parameters": {
                                "type": "object",
                                "properties": parameters,
                                "required": required_params
                            }
                        }
                    }
                    callable_tools[name] = tools[i]
                elif isinstance(_tools[i],dict):
                    continue
                else:
                    raise ValueError("tools must be a list of callables or dicts")

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
            lms.configure_default_client(self.base_url)
            all_loaded_models = lms.list_loaded_models()
            for loaded_model in all_loaded_models:
                if loaded_model.identifier != self.model:
                    loaded_model.unload()

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
                    if not hide_thinking:
                        yield {"type": "reasoning", "content": content}
                else:
                    answer += content
                    if not structured_output:
                        yield {"type": "answer", "content": content}

            if reasoning:
                thinking += reasoning
                if not hide_thinking:
                    yield {"type": "reasoning", "content": reasoning}
                
            if tool_calls:
                for tool_call in tool_calls:
                    tool_id = tool_call.id or "0"
                    funct = tool_call.function
                    if tool_id not in tool_calls_accumulator:
                        tool_calls_accumulator[tool_id] = {"name": funct.name or "", "arguments": ""}
                    if funct.name:
                        tool_calls_accumulator[tool_id]["name"] = funct.name
                    if funct.arguments:
                        tool_calls_accumulator[tool_id]["arguments"] += funct.arguments

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
        
        for tool_call in final_tool_calls[:]: 
            tool_name = tool_call["name"]
            
            if tool_name in callable_tools:
                try:
                    func_to_call = callable_tools[tool_name]
                    result = func_to_call(**tool_call["arguments"])
                    yield {
                        "type": "tool_result", 
                        "content": {
                            "name": tool_name,
                            "result": result
                        }
                    }
                    final_tool_calls.remove(tool_call)
                    
                except Exception as e:
                    print(f"Error executing tool {tool_name}: {e}")
                    final_tool_calls.remove(tool_call)
                    
            else:
                yield {"type": "tool_call", "content": tool_call}

        if final:
            if hide_thinking or thinking.strip() == "":
                if final_tool_calls == []:
                    yield {"type": "final", "content": {
                        "answer": answer,
                        }    
                    }
                else:
                    yield {"type": "final", "content": {
                        "answer": answer,
                        "tool_calls": final_tool_calls
                        }    
                    }
            else:
                if final_tool_calls == []:
                    yield {"type": "final", "content": {
                        "reasoning": thinking,
                        "answer": answer,
                        }    
                    }
                else:
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
        lms.configure_default_client(self.base_url)
        model = lms.llm(self.model)
        token_count = len(model.tokenize(input_text))
        return token_count
    
    def lm_studio_get_context_length(self) -> int:
        import lmstudio as lms
        lms.configure_default_client(self.base_url)
        model = lms.llm(self.model)
        return model.get_context_length()
