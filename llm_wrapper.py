import json
import base64
import re
import uuid
from openai import OpenAI, AsyncOpenAI
import io
from typing import Any, AsyncGenerator, Dict, Optional, List, Callable
import inspect
import asyncio


class LLM:
    """Universal api wrapper for LLM models with openai compatible api (e.g., LM Studio)"""

    # Type mapping for JSON Schema
    TYPE_MAPPING = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
        "nonetype": "null",
    }

    def __init__(self, model: str, vllm_mode: bool = False, api_key: str = "lm-studio",
                 base_url: str = "http://localhost:1234/v1"):
        """
        Initialize the wrapper.
        
        Args:
            model: The model identifier to use
            vllm_mode: Whether to use vLLM-specific image processing
            api_key: API key for authentication
            base_url: Base URL for the API endpoint
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.vllm_mode = vllm_mode

    def _get_json_type(self, python_type) -> str:
        """
        Convert Python type annotation to JSON Schema type.
        
        Args:
            python_type: A Python type annotation
            
        Returns:
            JSON Schema type string
        """
        type_name = getattr(python_type, "__name__", None)
        if type_name is None:
            type_str = str(python_type)
            # Handle Optional, Union types
            if "Optional" in type_str or "Union" in type_str:
                # Extract the first inner type
                inner = type_str.split("[", 1)[1].rsplit("]", 1)[0].split(",")[0].strip()
                type_name = inner.replace("typing.", "").lower()
            else:
                type_name = type_str.split("[", 1)[0].replace("typing.", "").lower()
        
        return self.TYPE_MAPPING.get(type_name.lower() if type_name else "str", "string")

    def _prepare_tools(self, tools: Optional[List]) -> tuple[List[Dict], Dict[str, Callable]]:
        """
        Convert callable functions to OpenAI tool format.
        
        Args:
            tools: List of callables or tool definition dicts
            
        Returns:
            Tuple of (prepared_tools, callable_tools_dict)
        """
        if not tools:
            return [], {}
        
        _tools = list(tools)
        callable_tools = {}
        
        for i in range(len(_tools)):
            if callable(_tools[i]):
                func = _tools[i]
                name = func.__name__.strip()
                callable_tools[name] = func
                doc = (func.__doc__ or "").strip()
                
                # Get parameter annotations and signature
                param_annotations = func.__annotations__
                sig = inspect.signature(func)
                
                required_params = []
                parameters = {}
                
                for param_name, param_obj in sig.parameters.items():
                    if param_name == "return":
                        continue
                    
                    # Get type from annotations if available
                    if param_name in param_annotations:
                        json_type = self._get_json_type(param_annotations[param_name])
                    else:
                        json_type = "string"
                    
                    parameters[param_name] = {"type": json_type}
                    
                    # Check if parameter has no default value → required
                    if param_obj.default == inspect.Parameter.empty:
                        required_params.append(param_name)
                
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
        
        return _tools, callable_tools

    def _process_images(self, messages: List[Dict]) -> None:
        """
        Convert custom image formats to base64 for vLLM mode.
        Modifies messages in-place.
        
        Args:
            messages: List of message dicts to process
        """
        from PIL import Image
        
        for msg in messages:
            if "content" not in msg:
                continue
            if not isinstance(msg["content"], list):
                continue
            
            for i in range(len(msg["content"])):
                c = msg["content"][i]
                if not isinstance(c, dict):
                    continue
                if c.get("type") != "image":
                    continue
                
                if "image_path" in c:
                    try:
                        with Image.open(c["image_path"]) as img:
                            buffer = io.BytesIO()
                            img.save(buffer, format="PNG")
                            buffer.seek(0)
                            img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                        msg["content"][i] = {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                        }
                    except Exception as e:
                        raise ValueError(f"Failed to process image from path: {e}")
                        
                elif "image_pil" in c:
                    try:
                        img = c["image_pil"]
                        buffer = io.BytesIO()
                        img.save(buffer, format="PNG")
                        buffer.seek(0)
                        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                        msg["content"][i] = {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                        }
                    except Exception as e:
                        raise ValueError(f"Failed to process PIL image: {e}")
                        
                elif "image_url" in c:
                    url_data = c["image_url"]
                    if isinstance(url_data, str):
                        url_data = {"url": url_data}
                    msg["content"][i] = {"type": "image_url", "image_url": url_data}

    def _parse_thinking_content(self, content: str, inside_think: bool) -> tuple[str, bool, str, str]:
        """
        Parse thinking tags from content (case-insensitive).
        
        Args:
            content: The content string to parse
            inside_think: Current thinking state
            
        Returns:
            Tuple of (cleaned_content, new_inside_think, thinking_part, answer_part)
        """
        thinking_part = ""
        answer_part = ""
        
        # Case-insensitive tag detection and removal
        content_check = content.lower()
        
        if "<think>" in content_check or "[think]" in content_check:
            inside_think = True
            content = re.sub(r'<think>|\[THINK\]', '', content, flags=re.IGNORECASE)
        
        if "</think>" in content_check or "[/think]" in content_check:
            # Split content at closing tag
            parts = re.split(r'</think>|\[/THINK\]', content, flags=re.IGNORECASE)
            if len(parts) > 1:
                thinking_part = parts[0]
                answer_part = parts[1]
                inside_think = False
                return "", inside_think, thinking_part, answer_part
            else:
                inside_think = False
                content = re.sub(r'</think>|\[/THINK\]', '', content, flags=re.IGNORECASE)
        
        if inside_think:
            thinking_part = content
        else:
            answer_part = content
            
        return content, inside_think, thinking_part, answer_part

    def _unload_other_models(self) -> None:
        """Unload all models except the current one in LM Studio."""
        import lmstudio as lms
        lms.configure_default_client(self.base_url)
        all_loaded_models = lms.list_loaded_models()
        for loaded_model in (all_loaded_models or []):
            if loaded_model.identifier != self.model:
                loaded_model.unload()

    def response(self, messages: List[Dict[str, Any]] = None, output_format: Dict = None, 
                 tools: List = None, lm_studio_unload_model: bool = False, 
                 hide_thinking: bool = True) -> Optional[Dict[str, Any]]:
        """
        Request model inference (non-streaming).
        
        Args:
            messages: List of conversation messages
            output_format: JSON schema for structured output
            tools: List of callable functions or tool definitions
            lm_studio_unload_model: Whether to unload other models in LM Studio
            hide_thinking: Whether to hide reasoning tokens
            
        Returns:
            Dict containing the final response with answer and optional tool_calls,
            or None if no final response received
            
        Raises:
            ValueError: If messages is None
            RuntimeError: If no final response is received
        """
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

        final_content = None
        for r in response:
            if r["type"] == "final":
                final_content = r["content"]
                break
        
        if final_content is None:
            raise RuntimeError("No final response received from model")
        
        return final_content

    def stream_response(self, messages: List[Dict] = None, output_format: Dict = None, 
                        final: bool = False, tools: List = None,
                        lm_studio_unload_model: bool = False, 
                        hide_thinking: bool = True) -> Any:
        """
        Request model inference with streaming.
        
        Args:
            messages: List of conversation messages
            output_format: JSON schema for structured output
            final: Whether to yield a final aggregated response
            tools: List of callable functions or tool definitions
            lm_studio_unload_model: Whether to unload other models in LM Studio
            hide_thinking: Whether to hide reasoning tokens
            
        Yields:
            Dicts with type and content for each chunk/event
        """
        if messages is None:
            raise ValueError("messages must be provided")

        # Prepare tools using helper method
        _tools, callable_tools = self._prepare_tools(tools)

        # Process images if in vLLM mode
        if self.vllm_mode:
            self._process_images(messages)

        # Unload other models if requested
        if lm_studio_unload_model:
            self._unload_other_models()

        structured_output = output_format is not None

        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "tools": _tools if _tools else [],
            }
            if structured_output:
                kwargs["response_format"] = output_format
            
            completion = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            raise RuntimeError(f"Model request failed: {e}")

        thinking = ""
        answer = ""
        tool_calls_accumulator = {}
        inside_think = False

        for chunk in completion:
            if not chunk.choices:
                continue
            x = chunk.choices[0].delta
            if not x:
                continue

            reasoning = getattr(x, "reasoning", None)
            content = getattr(x, "content", None)
            tool_calls = getattr(x, "tool_calls", None)

            if not (content or tool_calls or reasoning):
                continue

            if content:
                _, inside_think, thinking_part, answer_part = self._parse_thinking_content(
                    str(content), inside_think
                )
                
                if thinking_part:
                    thinking += thinking_part
                    if not hide_thinking:
                        yield {"type": "reasoning", "content": thinking_part}
                
                if answer_part:
                    answer += answer_part
                    if not structured_output:
                        yield {"type": "answer", "content": answer_part}

            if reasoning:
                thinking += reasoning
                if not hide_thinking:
                    yield {"type": "reasoning", "content": reasoning}

            if tool_calls:
                for tool_call in tool_calls:
                    tool_id = tool_call.id or str(uuid.uuid4())
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

        # Build final tool calls list
        final_tool_calls = []
        for tool_id, data in tool_calls_accumulator.items():
            try:
                args = json.loads(data["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {"_raw": data["arguments"] or ""}
            final_tool_calls.append({"id": tool_id, "name": data["name"], "arguments": args})

        # Execute callable tools and track remaining
        executed_tool_results = []
        remaining_tool_calls = []
        
        for tool_call in final_tool_calls:
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

                except Exception as e:
                    print(f"Error executing tool {tool_name}: {e}")

            else:
                remaining_tool_calls.append(tool_call)
                yield {"type": "tool_call", "content": tool_call}

        if final:
            # Type-safe answer handling
            answer_value = answer.strip() if isinstance(answer, str) else answer
            content = {"answer": answer_value}
            if not hide_thinking and thinking.strip():
                content["reasoning"] = thinking.strip()
            if remaining_tool_calls:
                content["tool_calls"] = remaining_tool_calls
            if executed_tool_results:
                content["tool_results"] = executed_tool_results

            yield {"type": "final", "content": content}
        yield {"type": "done", "content": None}

    async def async_response(self, messages: List[Dict[str, Any]] = None, output_format: Dict = None, 
                             tools: List = None, lm_studio_unload_model: bool = False, 
                             hide_thinking: bool = True) -> Optional[Dict[str, Any]]:
        """
        Asynchronous request for model inference.
        
        Args:
            messages: List of conversation messages
            output_format: JSON schema for structured output
            tools: List of callable functions or tool definitions
            lm_studio_unload_model: Whether to unload other models in LM Studio
            hide_thinking: Whether to hide reasoning tokens
            
        Returns:
            Dict containing the final response with answer and optional tool_calls,
            or None if no final response received
            
        Raises:
            ValueError: If messages is None
            RuntimeError: If no final response is received
        """
        if messages is None:
            raise ValueError("messages must be provided")

        final_content = None
        async for r in self.async_stream_response(
            messages=messages,
            output_format=output_format,
            final=True,
            tools=tools,
            lm_studio_unload_model=lm_studio_unload_model,
            hide_thinking=hide_thinking
        ):
            if r["type"] == "final":
                final_content = r["content"]
                break
        
        if final_content is None:
            raise RuntimeError("No final response received from model")
        
        return final_content

    async def async_stream_response(self, messages: List[Dict] = None, output_format: Dict = None, 
                                    final: bool = False, tools: List = None, 
                                    lm_studio_unload_model: bool = False, 
                                    hide_thinking: bool = True) -> AsyncGenerator[Dict, None]:
        """
        Asynchronous request for model inference with streaming.
        
        Args:
            messages: List of conversation messages
            output_format: JSON schema for structured output
            final: Whether to yield a final aggregated response
            tools: List of callable functions or tool definitions
            lm_studio_unload_model: Whether to unload other models in LM Studio
            hide_thinking: Whether to hide reasoning tokens
            
        Yields:
            Dicts with type and content for each chunk/event
        """
        if messages is None:
            raise ValueError("messages must be provided")

        # Prepare tools using helper method
        _tools, callable_tools = self._prepare_tools(tools)

        # Process images if in vLLM mode
        if self.vllm_mode:
            self._process_images(messages)

        # Unload other models if requested
        if lm_studio_unload_model:
            self._unload_other_models()

        structured_output = output_format is not None

        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "tools": _tools if _tools else [],
            }
            if structured_output:
                kwargs["response_format"] = output_format

            create_call = self.async_client.chat.completions.create(**kwargs)

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
            if not chunk.choices:
                continue
            x = chunk.choices[0].delta
            if not x:
                continue

            reasoning = getattr(x, "reasoning", None)
            content = getattr(x, "content", None)
            tool_calls = getattr(x, "tool_calls", None)

            if not (content or tool_calls or reasoning):
                continue

            if content:
                _, inside_think, thinking_part, answer_part = self._parse_thinking_content(
                    str(content), inside_think
                )
                
                if thinking_part:
                    thinking += thinking_part
                    if not hide_thinking:
                        yield {"type": "reasoning", "content": thinking_part}
                
                if answer_part:
                    answer += answer_part
                    if not structured_output:
                        yield {"type": "answer", "content": answer_part}

            if reasoning:
                thinking += reasoning
                if not hide_thinking:
                    yield {"type": "reasoning", "content": reasoning}

            if tool_calls:
                for tool_call in tool_calls:
                    tool_id = tool_call.id or str(uuid.uuid4())
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

        # Build final tool calls list
        final_tool_calls = []
        for tool_id, data in tool_calls_accumulator.items():
            try:
                args = json.loads(data["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {"_raw": data["arguments"] or ""}
            final_tool_calls.append({"id": tool_id, "name": data["name"], "arguments": args})

        # Execute callable tools and track remaining
        executed_tool_results = []
        remaining_tool_calls = []
        
        for tool_call in final_tool_calls:
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

                except Exception as e:
                    print(f"Error executing async tool {tool_name}: {e}")

            else:
                remaining_tool_calls.append(tool_call)
                yield {"type": "tool_call", "content": tool_call}

        if final:
            # Type-safe answer handling
            answer_value = answer.strip() if isinstance(answer, str) else answer
            content = {"answer": answer_value}
            if not hide_thinking and thinking.strip():
                content["reasoning"] = thinking.strip()
            if remaining_tool_calls:
                content["tool_calls"] = remaining_tool_calls
            if executed_tool_results:
                content["tool_results"] = executed_tool_results

            yield {"type": "final", "content": content}
        yield {"type": "done", "content": None}

    def lm_studio_count_tokens(self, input_text: str) -> int:
        """
        Count tokens used in LM Studio.
        
        Args:
            input_text: The text to tokenize
            
        Returns:
            Number of tokens
            
        Raises:
            RuntimeError: If tokenization fails
        """
        import lmstudio as lms
        lms.configure_default_client(self.base_url)
        try:
            model = lms.llm(self.model)
            tokens = model.tokenize(input_text)
            return len(tokens)
        except Exception as e:
            raise RuntimeError(f"Could not count tokens for {self.model}: {e}")

    def lm_studio_get_context_length(self) -> int:
        """
        Get the context length of the model in LM Studio.
        
        Returns:
            Context length in tokens
        """
        import lmstudio as lms
        lms.configure_default_client(self.base_url)
        model = lms.llm(self.model)
        return model.get_context_length()
