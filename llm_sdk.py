import json
import base64
import re
import uuid
from openai import OpenAI, AsyncOpenAI
import io
from typing import Any, AsyncGenerator, Dict, Optional, List, Callable, Union, get_type_hints, get_origin, get_args
import inspect
import asyncio


class LLM:
    """Universal api wrapper for LLM models with openai compatible api (e.g., LM Studio)"""



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

    def _python_type_to_json_schema(self, python_type, seen_models: set = None) -> Dict[str, Any]:
        """
        Convert Python type annotation to JSON Schema.
        
        Supports:
        - Basic types: str, int, float, bool
        - list, List[T]
        - dict, Dict[str, T]
        - Optional[T], Union[T, None]
        - Literal[...] (as enum)
        - Nested classes with __annotations__
        """
        if seen_models is None:
            seen_models = set()

        # Handle None type
        if python_type is type(None):
            return {"type": "null"}
        
        # Get origin for generic types (List, Dict, etc.)
        origin = get_origin(python_type)
        args = get_args(python_type)
        
        # Handle List[T]
        if origin is list:
            if args:
                item_schema = self._python_type_to_json_schema(args[0], seen_models)
                return {"type": "array", "items": item_schema}
            return {"type": "array"}
        
        # Handle Dict[K, V]
        if origin is dict:
            if len(args) == 2:
                value_schema = self._python_type_to_json_schema(args[1], seen_models)
                return {
                    "type": "object",
                    "additionalProperties": value_schema
                }
            return {"type": "object"}
        
        # Handle Optional[T] / Union[T, None]
        if origin is Union:
            non_none_types = [t for t in args if t is not type(None)]
            
            # This is Optional[T]
            if len(non_none_types) == 1:
                base_schema = self._python_type_to_json_schema(non_none_types[0], seen_models)
                return {
                    "anyOf": [
                        base_schema,
                        {"type": "null"}
                    ]
                }
            
            # Multiple non-None types
            return {
                "anyOf": [
                    self._python_type_to_json_schema(t, seen_models) for t in args
                ]
            }
        
        # Handle Literal (as enum)
        try:
            from typing import Literal
            if origin is Literal:
                return {"enum": list(args)}
        except ImportError:
            pass
        
        # Handle nested class with __annotations__
        if (hasattr(python_type, "__annotations__") and 
            python_type.__annotations__ and
            isinstance(python_type, type)):
            
            # Recursion Guard
            if python_type in seen_models:
                raise ValueError(
                    f"Circular dependency detected for class {python_type.__name__}. "
                    "Recursive schemas are not currently supported."
                )
            
            # Recursively convert nested class
            nested_schema = self._convert_class_to_schema(python_type, seen_models=seen_models)
            # Return just the schema part, not the full wrapper
            return nested_schema["json_schema"]["schema"]
        
        # Basic types
        type_name = getattr(python_type, "__name__", str(python_type)).lower()
        
        basic_mapping = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
        }
        
        json_type = basic_mapping.get(type_name, "string")
        return {"type": json_type}

    def _convert_class_to_schema(self, schema_class: type, name: Optional[str] = None, seen_models: set = None) -> Dict[str, Any]:
        """Convert plain class with __annotations__ to OpenAI JSON schema."""
        
        if seen_models is None:
            seen_models = set()
            
        # Add current class to seen set to prevent recursion
        seen_models.add(schema_class)
        
        # Check if class has annotations
        if not hasattr(schema_class, "__annotations__") or not schema_class.__annotations__:
            raise ValueError(
                f"Class {schema_class.__name__} has no type annotations. "
                "Ensure class fields are annotated."
            )
        
        hints = get_type_hints(schema_class)
        properties = {}
        required = []
        
        # Get class-level defaults
        class_defaults = {}
        for key, value in schema_class.__dict__.items():
            # Skip private attributes and methods
            if not key.startswith("_") and not callable(value):
                class_defaults[key] = value
        
        # Process each annotated field
        for field_name, field_type in hints.items():
            properties[field_name] = self._python_type_to_json_schema(field_type, seen_models)
            
            # Check if type is Optional (Union[..., NoneType])
            is_optional = False
            origin = get_origin(field_type)
            if origin is Union:
                if type(None) in get_args(field_type):
                    is_optional = True

            # Field is required if it has no default value AND is not Optional
            if field_name not in class_defaults and not is_optional:
                required.append(field_name)
        
        # Get class docstring as schema description
        description = inspect.getdoc(schema_class)
        
        schema = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False
        }
        
        if description:
            schema["description"] = description
        
        # Remove from seen set after processing
        seen_models.remove(schema_class)
        
        return {
            "type": "json_schema",
            "json_schema": {
                "name": name or schema_class.__name__,
                "strict": True,
                "schema": schema
            }
        }

    def _prepare_output_format(self, output_format: Union[Dict, type, None]) -> Optional[Dict]:
        """
        Convert output_format to OpenAI schema format.
        
        Args:
            output_format: 
                - None: No structured output
                - Dict: OpenAI schema (passthrough)
                - type: Plain class with __annotations__ (will be converted)
        
        Returns:
            OpenAI-compatible schema dict or None
        
        Raises:
            ValueError: If output_format type is unsupported
        """
        if output_format is None:
            return None
        
        # Dict: assume it's already in OpenAI format
        if isinstance(output_format, dict):
            return output_format
        
        # Type: convert to schema
        if isinstance(output_format, type):
            return self._convert_class_to_schema(output_format)
        
        raise ValueError(
            f"output_format must be dict, type, or None. Got: {type(output_format)}"
        )

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
                        param_schema = self._python_type_to_json_schema(param_annotations[param_name])
                    else:
                        param_schema = {"type": "string"}
                    
                    parameters[param_name] = param_schema
                    
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
                # Validate tool dict has required structure
                tool_dict = _tools[i]
                if "type" not in tool_dict or "function" not in tool_dict:
                    raise ValueError(
                        f"Tool dict at index {i} must have 'type' and 'function' keys. "
                        "Expected format: {'type': 'function', 'function': {'name': ..., 'parameters': ...}}"
                    )
                func_def = tool_dict.get("function", {})
                if "name" not in func_def:
                    raise ValueError(
                        f"Tool dict at index {i} missing 'name' in function definition."
                    )
            else:
                raise ValueError("tools must be a list of callables or dicts")
        
        return _tools, callable_tools

    def _process_images(self, messages: List[Dict]) -> None:
        """
        Convert custom image formats (image_path, image_pil) to OpenAI-compatible image_url format.
        Modifies messages in-place.
        
        Args:
            messages: List of message dicts to process
        """
        pil_available = False
        try:
            from PIL import Image
            pil_available = True
        except ImportError:
            pass  # PIL not required if not using image_path or image_pil
        
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
                    if not pil_available:
                        raise ImportError(
                            "PIL/Pillow is required for image_path processing. "
                            "Install it with: pip install Pillow"
                        )
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
                    if not pil_available:
                        raise ImportError(
                            "PIL/Pillow is required for image_pil processing. "
                            "Install it with: pip install Pillow"
                        )
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

    def _parse_thinking_content(self, content: str, inside_think: bool) -> tuple[bool, str, str]:
        """
        Parse thinking tags from content (case-insensitive) in a more robust way.
        
        Args:
            content: The content string to parse
            inside_think: Current thinking state
            
        Returns:
            Tuple of (new_inside_think, thinking_part, answer_part)
        """
        thinking_part = ""
        answer_part = ""
        
        remaining_content = content
        
        while remaining_content:
            if inside_think:
                # Look for end tag
                end_match = re.search(r'</think>|\[/THINK\]', remaining_content, flags=re.IGNORECASE)
                if end_match:
                    # Found end tag -> up to tag is thought, rest is processed next loop
                    thought = remaining_content[:end_match.start()]
                    thinking_part += thought
                    
                    inside_think = False
                    remaining_content = remaining_content[end_match.end():]
                else:
                    # No end tag -> all is thought
                    thinking_part += remaining_content
                    remaining_content = ""
            else:
                # Look for start tag
                start_match = re.search(r'<think>|\[THINK\]', remaining_content, flags=re.IGNORECASE)
                if start_match:
                    # Found start tag -> up to tag is answer
                    answer = remaining_content[:start_match.start()]
                    answer_part += answer
                    
                    inside_think = True
                    # Skip the tag itself
                    remaining_content = remaining_content[start_match.end():]
                else:
                    # No start tag -> all is answer
                    answer_part += remaining_content
                    remaining_content = ""
                    
        return inside_think, thinking_part, answer_part

    def _unload_other_models(self) -> None:
        """Unload all models except the current one in LM Studio."""
        try:
            import lmstudio as lms
            # Try to configure, ignore if already configured (singleton)
            try:
                lms.configure_default_client(self.base_url)
            except Exception:
                pass  # Already configured, continue
            all_loaded_models = lms.list_loaded_models()
            for loaded_model in (all_loaded_models or []):
                if loaded_model.identifier != self.model:
                    loaded_model.unload()
        except ImportError:
            pass  # lmstudio not installed, skip
        except Exception:
            pass  # Ignore errors in unloading

    def response(self, messages: List[Dict[str, Any]] = None, output_format: Union[Dict, type, None] = None, 
                 tools: List = None, lm_studio_unload_model: bool = False, 
                 hide_thinking: bool = True) -> Optional[Dict[str, Any]]:
        """
        Request model inference (non-streaming).
        
        Args:
            messages: List of conversation messages
            output_format: JSON schema (dict) or plain class (type) for structured output
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

        # Convert class to schema if needed
        output_format = self._prepare_output_format(output_format)


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

    def stream_response(self, messages: List[Dict] = None, output_format: Union[Dict, type, None] = None, 
                        final: bool = False, tools: List = None,
                        lm_studio_unload_model: bool = False, 
                        hide_thinking: bool = True) -> Any:
        """
        Request model inference with streaming.
        
        Args:
            messages: List of conversation messages
            output_format: JSON schema (dict) or plain class (type) for structured output
            final: Whether to yield a final aggregated response
            tools: List of callable functions or tool definitions
            lm_studio_unload_model: Whether to unload other models in LM Studio
            hide_thinking: Whether to hide reasoning tokens
            
        Yields:
            Dicts with type and content for each chunk/event
        """
        if messages is None:
            raise ValueError("messages must be provided")

        # Convert class to schema if needed
        output_format = self._prepare_output_format(output_format)

        # Prepare tools using helper method
        _tools, callable_tools = self._prepare_tools(tools)

        # Always process custom image formats (image_path, image_pil, image_url)
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
            }
            if _tools:
                kwargs["tools"] = _tools
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
                inside_think, thinking_part, answer_part = self._parse_thinking_content(
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
                for idx, tool_call in enumerate(tool_calls):
                    # Use ID if available, otherwise use index for consistent tracking
                    tool_id = tool_call.id if tool_call.id else f"_idx_{idx}"
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
            try:
                answer = json.loads(answer)
            except json.JSONDecodeError:
                pass  # Keep answer as raw string if JSON parsing fails
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
                    
                    # Check if async function passed to sync method
                    if inspect.iscoroutinefunction(func_to_call):
                        error_content = {
                            "name": tool_name,
                            "error": f"Async tool '{tool_name}' cannot be executed in sync mode. Use async_stream_response instead."
                        }
                        yield {"type": "tool_error", "content": error_content}
                        remaining_tool_calls.append(tool_call)
                        continue
                    
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
                    error_content = {
                        "name": tool_name,
                        "error": str(e)
                    }
                    yield {"type": "tool_error", "content": error_content}

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

    async def async_response(self, messages: List[Dict[str, Any]] = None, output_format: Union[Dict, type, None] = None, 
                             tools: List = None, lm_studio_unload_model: bool = False, 
                             hide_thinking: bool = True) -> Optional[Dict[str, Any]]:
        """
        Asynchronous request for model inference.
        
        Args:
            messages: List of conversation messages
            output_format: JSON schema (dict) or plain class (type) for structured output
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

        # Convert class to schema if needed
        output_format = self._prepare_output_format(output_format)

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

    async def async_stream_response(self, messages: List[Dict] = None, output_format: Union[Dict, type, None] = None, 
                                    final: bool = False, tools: List = None, 
                                    lm_studio_unload_model: bool = False, 
                                    hide_thinking: bool = True) -> AsyncGenerator[Dict, None]:
        """
        Asynchronous request for model inference with streaming.
        
        Args:
            messages: List of conversation messages
            output_format: JSON schema (dict) or plain class (type) for structured output
            final: Whether to yield a final aggregated response
            tools: List of callable functions or tool definitions
            lm_studio_unload_model: Whether to unload other models in LM Studio
            hide_thinking: Whether to hide reasoning tokens
            
        Yields:
            Dicts with type and content for each chunk/event
        """
        if messages is None:
            raise ValueError("messages must be provided")

        # Convert class to schema if needed
        output_format = self._prepare_output_format(output_format)

        # Prepare tools using helper method
        _tools, callable_tools = self._prepare_tools(tools)

        # Always process custom image formats (image_path, image_pil, image_url)
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
            }
            if _tools:
                kwargs["tools"] = _tools
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
                inside_think, thinking_part, answer_part = self._parse_thinking_content(
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
                for idx, tool_call in enumerate(tool_calls):
                    # Use ID if available, otherwise use index for consistent tracking
                    tool_id = tool_call.id if tool_call.id else f"_idx_{idx}"
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
            try:
                answer = json.loads(answer)
            except json.JSONDecodeError:
                pass  # Keep answer as raw string if JSON parsing fails
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
                    error_content = {
                        "name": tool_name,
                        "error": str(e)
                    }
                    yield {"type": "tool_error", "content": error_content}

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
        try:
            import lmstudio as lms
            # LM Studio SDK expects base URL without /v1 suffix
            sdk_url = self.base_url.rstrip('/').removesuffix('/v1')
            
            # Try to configure, ignore if already configured (singleton)
            try:
                lms.configure_default_client(sdk_url)
            except Exception:
                pass  # Already configured, continue
            
            model = lms.llm(self.model)
            tokens = model.tokenize(input_text)
            return len(tokens)
        except ImportError:
            raise RuntimeError("lmstudio package not installed. Install with: pip install lmstudio")
        except Exception as e:
            raise RuntimeError(f"Could not count tokens for {self.model}: {e}")

    def lm_studio_get_context_length(self) -> int:
        """
        Get the context length of the model in LM Studio.
        
        Returns:
            Context length in tokens
        """
        try:
            import lmstudio as lms
            # LM Studio SDK expects base URL without /v1 suffix
            sdk_url = self.base_url.rstrip('/').removesuffix('/v1')
            
            # Try to configure, ignore if already configured (singleton)
            try:
                lms.configure_default_client(sdk_url)
            except Exception:
                pass  # Already configured, continue
            
            model = lms.llm(self.model)
            return model.get_context_length()
        except ImportError:
            raise RuntimeError("lmstudio package not installed. Install with: pip install lmstudio")
        except Exception as e:
            raise RuntimeError(f"Could not get context length for {self.model}: {e}")
