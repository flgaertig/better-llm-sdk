import json
import base64
from openai import OpenAI, AsyncOpenAI
import io
from typing import Any, AsyncGenerator
import inspect
import asyncio

class LLM:
    """Universal api wrapper for LLM models with openai compatible api (e.g., LM Studio)"""
    def __init__(self, model: str, vllm_mode: bool = False, api_key: str = "lm-studio",
                 base_url: str = "http://localhost:1234/v1"):
        """initialize the wrapper"""
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.vllm_mode = vllm_mode

    def response(self, messages: list[dict[str, Any]] = None, output_format: dict = None, tools: list = None,
                 lm_studio_unload_model: bool = False, hide_thinking: bool = True):
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

    def stream_response(self, messages: list[dict] = None, output_format: dict = None, final: bool = False, tools: list = None,
                        lm_studio_unload_model: bool = False, hide_thinking: bool = True):
        """request model inference"""
        _tools = list(tools) if tools else []
        callable_tools = {}
        if _tools:
            types = {
                "str": "string",
                "int": "integer",
                "float": "number",
                "bool": "boolean",
                "list": "array",
                "dict": "object"
            }
            for i in range(len(_tools)):
                if callable(_tools[i]):
                    func=_tools[i]
                    name = func.__name__.strip()
                    callable_tools[name] = func
                    doc = (func.__doc__ or "").strip()
                    param = func.__annotations__
                    required_params = []
                    parameters = {}
                    for k, v in param.items():
                        if k == "return":
                            continue
                        else:
                            type_name = getattr(v, "__name__", None)
                            if type_name is None:
                                type_name = str(v).split("[", 1)[0].replace("typing.", "").lower()
                            parameters[str(k)] = {"type": types.get(type_name, "string")}
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
                elif isinstance(_tools[i], dict):
                    continue
                else:
                    raise ValueError("tools must be a list of callables or dicts")

        if messages is None:
            raise ValueError("messages must be provided")

        if self.vllm_mode:
            from PIL import Image
            for msg in messages:
                for i in range(len(msg["content"])):
                    c = msg["content"][i]
                    if c["type"] == "image":
                        if "image_path" in c:
                            with Image.open(c["image_path"]) as img:
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
            for loaded_model in (all_loaded_models or []):
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
                if "<think>" in str(content):
                    inside_think = True
                    content = str(content).replace("<think>", "")
                if "</think>" in str(content):
                    inside_think = False
                    content = str(content).replace("</think>", "")
                if inside_think:
                    thinking += str(content)
                    if not hide_thinking:
                        yield {"type": "reasoning", "content": str(content)}
                else:
                    answer += str(content)
                    if not structured_output:
                        yield {"type": "answer", "content": str(content)}

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
                        args_val = funct.arguments
                        if isinstance(args_val, dict):
                            args_val = json.dumps(args_val)
                        tool_calls_accumulator[tool_id]["arguments"] += args_val or ""

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

        executed_tool_results = []
        for tool_call in final_tool_calls[:]:
            tool_name = tool_call["name"]

            if tool_name in callable_tools:
                try:
                    func_to_call = callable_tools[tool_name]
                    result = func_to_call(**tool_call["arguments"])
                    
                    tool_result_content = {
                        "name": tool_name,
                        "result": result
                    }
                    executed_tool_results.append(tool_result_content)

                    yield {
                        "type": "tool_result",
                        "content": tool_result_content
                    }
                    final_tool_calls.remove(tool_call)

                except Exception as e:
                    print(f"Error executing tool {tool_name}: {e}")
                    final_tool_calls.remove(tool_call)

            else:
                yield {"type": "tool_call", "content": tool_call}

        if final:
            content = {"answer": answer.strip()}
            if not hide_thinking and thinking.strip():
                content["reasoning"] = thinking.strip()
            if final_tool_calls:
                content["tool_calls"] = final_tool_calls
            if executed_tool_results:
                content["tool_results"] = executed_tool_results
                
            yield {"type": "final", "content": content}
        yield {"type": "done", "content": None}

    async def async_response(self, messages: list[dict[str, Any]] = None, output_format: dict = None, tools: list = None,
                             lm_studio_unload_model: bool = False, hide_thinking: bool = True):
        """asynchron request model inference"""
        if messages is None:
            raise ValueError("messages must be provided")

        async for r in self.async_stream_response(
            messages=messages,
            output_format=output_format,
            final=True,
            tools=tools,
            lm_studio_unload_model=lm_studio_unload_model,
            hide_thinking=hide_thinking
        ):
            if r["type"] == "final":
                return r["content"]

    async def async_stream_response(self, messages: list[dict] = None, output_format: dict = None, final: bool = False,
                                    tools: list = None, lm_studio_unload_model: bool = False, hide_thinking: bool = True
                                    ) -> AsyncGenerator[dict, None]:
        """asynchron request model inference (streaming)"""
        _tools = list(tools) if tools else []
        callable_tools = {}
        if _tools:
            types = {
                "str": "string",
                "int": "integer",
                "float": "number",
                "bool": "boolean",
                "list": "array",
                "dict": "object"
            }
            for i in range(len(_tools)):
                if callable(_tools[i]):
                    func = _tools[i]
                    name = func.__name__.strip()
                    callable_tools[name] = func
                    doc = (func.__doc__ or "").strip()
                    param = func.__annotations__
                    required_params = []
                    parameters = {}
                    for k, v in param.items():
                        if k == "return":
                            continue
                        else:
                            type_name = getattr(v, "__name__", None)
                            if type_name is None:
                                type_name = str(v).split("[", 1)[0].replace("typing.", "").lower()
                            parameters[str(k)] = {"type": types.get(type_name, "string")}
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
                elif isinstance(_tools[i], dict):
                    continue
                else:
                    raise ValueError("tools must be a list of callables or dicts")

        if messages is None:
            raise ValueError("messages must be provided")

        if self.vllm_mode:
            from PIL import Image
            for msg in messages:
                for i in range(len(msg["content"])):
                    c = msg["content"][i]
                    if c["type"] == "image":
                        if "image_path" in c:
                            with Image.open(c["image_path"]) as img:
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
            for loaded_model in (all_loaded_models or []):
                if loaded_model.identifier != self.model:
                    loaded_model.unload()

        structured_output = output_format is not None

        try:
            if structured_output:
                create_call = self.async_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    tools=tools if tools is not None else [],
                    response_format=output_format if output_format is not None else None,
                )
            else:
                create_call = self.async_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                    tools=tools if tools is not None else [],
                )

            if asyncio.iscoroutine(create_call):
                completion = await create_call
            else:
                completion = create_call

        except Exception as e:
            raise RuntimeError(f"Async model request failed: {e}")

        thinking = ""
        answer = ""
        tool_calls_accumulator = {}
        inside_think = False

        async for chunk in completion:
            x = chunk.choices[0].delta
            if not x:
                continue

            reasoning = getattr(x, "reasoning", None)
            content = getattr(x, "content", None)
            tool_calls = getattr(x, "tool_calls", None)

            if not (content or tool_calls or reasoning):
                continue

            if content:
                if "<think>" in str(content):
                    inside_think = True
                    content = str(content).replace("<think>", "")
                if "</think>" in str(content):
                    inside_think = False
                    content = str(content).replace("</think>", "")
                if inside_think:
                    thinking += str(content)
                    if not hide_thinking:
                        yield {"type": "reasoning", "content": str(content)}
                else:
                    answer += str(content)
                    if not structured_output:
                        yield {"type": "answer", "content": str(content)}

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
                        args_val = funct.arguments
                        if isinstance(args_val, dict):
                            args_val = json.dumps(args_val)
                        tool_calls_accumulator[tool_id]["arguments"] += args_val or ""

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

        executed_tool_results = []
        for tool_call in final_tool_calls[:]:
            tool_name = tool_call["name"]

            if tool_name in callable_tools:
                try:
                    func_to_call = callable_tools[tool_name]
                    if inspect.iscoroutinefunction(func_to_call):
                        result = await func_to_call(**tool_call["arguments"])
                    else:
                        result = func_to_call(**tool_call["arguments"])

                    tool_result_content = {
                        "name": tool_name,
                        "result": result
                    }
                    executed_tool_results.append(tool_result_content)

                    yield {
                        "type": "tool_result",
                        "content": tool_result_content
                    }
                    final_tool_calls.remove(tool_call)

                except Exception as e:
                    print(f"Error executing async tool {tool_name}: {e}")
                    final_tool_calls.remove(tool_call)

            else:
                yield {"type": "tool_call", "content": tool_call}

        if final:
            content = {"answer": answer.strip()}
            if not hide_thinking and thinking.strip():
                content["reasoning"] = thinking.strip()
            if final_tool_calls:
                content["tool_calls"] = final_tool_calls
            if executed_tool_results:
                content["tool_results"] = executed_tool_results
            
            yield {"type": "final", "content": content}
        yield {"type": "done", "content": None}

    def lm_studio_count_tokens(self, input_text: str) -> int:
        """count tokens used in lm studio"""
        import lmstudio as lms
        lms.configure_default_client(self.base_url)
        try:
            model = lms.llm(self.model)
            tokens = model.tokenize(input_text)
            return len(tokens)
        except Exception as e:
            raise RuntimeError(f"Could not count tokens for {self.model}: {e}")

    def lm_studio_get_context_length(self) -> int:
        import lmstudio as lms
        lms.configure_default_client(self.base_url)
        model = lms.llm(self.model)
        return model.get_context_length()
