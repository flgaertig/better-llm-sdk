import functools
import unittest
from types import SimpleNamespace
from typing import Optional

from llm_sdk import LLM, EventType


def _chat_chunk(content="", usage=None):
    return SimpleNamespace(
        usage=usage,
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content))],
    )


class LLMSDKTests(unittest.TestCase):
    def tearDown(self):
        llm = getattr(self, "llm", None)
        if llm is not None:
            llm.close()

    def _fake_chat_stream(self, llm, chunks):
        captured = {}

        def create(**kwargs):
            captured.update(kwargs)
            return iter(chunks)

        llm._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        return captured

    def test_input_string_becomes_user_message(self):
        self.llm = LLM(model="test")
        captured = self._fake_chat_stream(self.llm, [_chat_chunk("ok")])

        events = list(self.llm.stream_response(input="hello"))

        self.assertEqual(
            captured["messages"],
            [{"role": "user", "content": "hello"}],
        )
        self.assertEqual(events[0]["content"], "ok")

    def test_messages_list_is_accepted(self):
        self.llm = LLM(model="test")
        messages = [{"role": "system", "content": "brief"}, {"role": "user", "content": "hi"}]
        captured = self._fake_chat_stream(self.llm, [_chat_chunk("ok")])

        list(self.llm.stream_response(messages=messages))

        self.assertEqual(captured["messages"], messages)
        self.assertIsNot(captured["messages"], messages)

    def test_extra_body_deep_merges_instance_defaults(self):
        self.llm = LLM(
            model="test",
            extra_body={
                "provider": {"only_default": True, "temperature": 0.2},
                "top": "default",
            },
        )

        kwargs, _, _ = self.llm._build_request(
            [{"role": "user", "content": "hi"}],
            output_format=None,
            tools=None,
            reasoning_effort=None,
            max_tokens=None,
            extra_body={"provider": {"temperature": 0.7}, "call": "value"},
        )

        self.assertEqual(
            kwargs["extra_body"],
            {
                "provider": {"only_default": True, "temperature": 0.7},
                "top": "default",
                "call": "value",
            },
        )

    def test_callable_tools_unwrap_decorators_and_partial(self):
        self.llm = LLM(model="test")

        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)
            return wrapper

        @decorator
        def search_docs(query: str, limit: int = 5) -> str:
            """Search docs."""
            return "ok"

        captured = self._fake_chat_stream(self.llm, [_chat_chunk("ok")])

        list(self.llm.stream_response(input="find notes", tools=[functools.partial(search_docs)]))

        tool = captured["tools"][0]["function"]
        self.assertEqual(tool["name"], "search_docs")
        self.assertEqual(tool["description"], "Search docs.")
        self.assertEqual(set(tool["parameters"]["properties"]), {"query", "limit"})
        self.assertEqual(tool["parameters"]["required"], ["query"])

    def test_responses_tool_mapper_preserves_required(self):
        converted = LLM._convert_tools_for_responses_api([
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "unit": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                },
            }
        ])

        params = converted[0]["parameters"]
        self.assertEqual(params["required"], ["city"])
        self.assertFalse(params["additionalProperties"])

    def test_responses_api_collects_tool_calls_and_usage(self):
        self.llm = LLM(model="gpt-5-test", use_responses_api=True)

        events = [
            SimpleNamespace(type="response.output_text.delta", delta="done"),
            SimpleNamespace(
                type="response.output_item.added",
                output_index=0,
                item=SimpleNamespace(
                    type="function_call",
                    id="item_1",
                    call_id="call_1",
                    name="weather",
                ),
            ),
            SimpleNamespace(
                type="response.function_call_arguments.done",
                item_id="item_1",
                output_index=0,
                arguments='{"city":"Berlin"}',
                name="weather",
            ),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    usage=SimpleNamespace(input_tokens=3, output_tokens=7, total_tokens=10)
                ),
            ),
        ]

        def create(**kwargs):
            return iter(events)

        self.llm._client = SimpleNamespace(responses=SimpleNamespace(create=create))

        result = list(self.llm.stream_response(input="use a tool", final=True, verbose=True))
        tool_event = next(event for event in result if event["type"] == EventType.TOOL_CALL.value)
        verbose_event = next(event for event in result if event["type"] == EventType.VERBOSE.value)
        final_event = next(event for event in result if event["type"] == EventType.FINAL.value)

        self.assertEqual(
            tool_event["content"],
            {"id": "call_1", "name": "weather", "arguments": {"city": "Berlin"}, "callable": None},
        )
        self.assertEqual(verbose_event["content"]["tokens"], 7)
        self.assertEqual(final_event["content"]["tool_calls"], [tool_event["content"]])

    def test_base_url_normalization(self):
        cases = {
            "http://localhost:1234": "http://localhost:1234/v1",
            "http://localhost:1234/": "http://localhost:1234/v1",
            "http://localhost:1234/v1": "http://localhost:1234/v1",
            "http://localhost:1234/v1/": "http://localhost:1234/v1",
        }
        for base_url, expected in cases.items():
            with self.subTest(base_url=base_url):
                llm = LLM(model="test", base_url=base_url)
                try:
                    self.assertEqual(llm._api_base, expected)
                finally:
                    llm.close()

    def test_tool_call_stream_handler_logic(self):
        from llm_sdk import EventBuilder, ToolCallStreamHandler
        eb = EventBuilder()
        handler = ToolCallStreamHandler(eb)

        # 1. Chunk with tool call index=0 but no id/name (should not emit tool_call_part yet)
        tc_chunk_1 = SimpleNamespace(
            index=0,
            id=None,
            function=SimpleNamespace(name=None, arguments='{"ci')
        )
        events_1 = handler.process_chunk([tc_chunk_1])
        self.assertEqual(events_1, [])

        # 2. Chunk with tool call index=0, adding name/id (should emit tool_call_part with all accumulated delta)
        tc_chunk_2 = SimpleNamespace(
            index=0,
            id="call_123",
            function=SimpleNamespace(name="weather", arguments='ty":"Ber')
        )
        events_2 = handler.process_chunk([tc_chunk_2])
        self.assertEqual(len(events_2), 1)
        self.assertEqual(events_2[0]["type"], EventType.TOOL_CALL_PART.value)
        self.assertEqual(events_2[0]["content"], {
            "id": "call_123",
            "name": "weather",
            "args_delta": '{"city":"Ber'
        })

        # 3. Chunk with tool call index=0, further arguments
        tc_chunk_3 = SimpleNamespace(
            index=0,
            id=None,
            function=SimpleNamespace(name=None, arguments='lin"}')
        )
        events_3 = handler.process_chunk([tc_chunk_3])
        self.assertEqual(len(events_3), 1)
        self.assertEqual(events_3[0]["type"], EventType.TOOL_CALL_PART.value)
        self.assertEqual(events_3[0]["content"], {
            "id": "call_123",
            "name": "weather",
            "args_delta": 'lin"}'
        })

        # 4. End of the tool call part stream (no tool calls chunk or new index)
        # Should yield the completed parsed JSON tool call
        events_4 = handler.process_chunk(None)
        self.assertEqual(len(events_4), 1)
        self.assertEqual(events_4[0]["type"], EventType.TOOL_CALL.value)
        self.assertEqual(events_4[0]["content"], {
            "id": "call_123",
            "name": "weather",
            "arguments": {"city": "Berlin"},
            "callable": None
        })

        # Check get_all_calls
        all_calls = handler.get_all_calls()
        self.assertEqual(all_calls, [{
            "id": "call_123",
            "name": "weather",
            "arguments": {"city": "Berlin"},
            "callable": None
        }])

        # Test parallel tool calls with switching
        handler.clear()

        # Stream tool call 0
        events_a = handler.process_chunk([
            SimpleNamespace(
                index=0,
                id="call_0",
                function=SimpleNamespace(name="get_weather", arguments='{"city":"Berlin"}')
            )
        ])
        # Stream tool call 1
        events_b = handler.process_chunk([
            SimpleNamespace(
                index=1,
                id="call_1",
                function=SimpleNamespace(name="get_time", arguments='{"timezone":"UTC"}')
            )
        ])

        # Tool call 0 should be emitted complete immediately upon switching to index 1
        self.assertEqual(len(events_a), 1)  # Just the part event
        self.assertEqual(events_a[0]["type"], EventType.TOOL_CALL_PART.value)

        self.assertEqual(len(events_b), 2)  # Part event for index 1 + Complete event for index 0!
        self.assertEqual(events_b[0]["type"], EventType.TOOL_CALL.value)
        self.assertEqual(events_b[0]["content"]["id"], "call_0")
        self.assertEqual(events_b[1]["type"], EventType.TOOL_CALL_PART.value)
        self.assertEqual(events_b[1]["content"]["id"], "call_1")

        # Finalize should return the remaining tool call 1
        finalized = handler.finalize()
        self.assertEqual(len(finalized), 1)
        self.assertEqual(finalized[0]["id"], "call_1")

        # get_all_calls should return both
        all_calls = handler.get_all_calls()
        self.assertEqual(len(all_calls), 2)
        self.assertEqual(all_calls[0]["id"], "call_0")
        self.assertEqual(all_calls[1]["id"], "call_1")

    def test_helper_functions(self):
        from llm_sdk import assistant_message, system_message, tool_result, user_message

        # Test user_message
        self.assertEqual(
            user_message("hello"),
            {"role": "user", "content": "hello"}
        )

        # Test system_message
        self.assertEqual(
            system_message("be brief"),
            {"role": "system", "content": "be brief"}
        )

        # Test tool_result
        self.assertEqual(
            tool_result({"id": "call_123"}, "success"),
            {"role": "tool", "tool_call_id": "call_123", "content": "success"}
        )

        # Test tool_result serializes dict content
        self.assertEqual(
            tool_result({"id": "call_123"}, {"result": 42}),
            {"role": "tool", "tool_call_id": "call_123", "content": '{"result": 42}'}
        )

        # Test assistant_message with no tool calls
        self.assertEqual(
            assistant_message({"answer": "hello"}),
            {"role": "assistant", "content": "hello", "tool_calls": None}
        )

        # Test assistant_message serializes dict answers
        self.assertEqual(
            assistant_message({"answer": {"sentiment": "pos"}}),
            {"role": "assistant", "content": '{"sentiment": "pos"}', "tool_calls": None}
        )

        # Test assistant_message with tool calls (should serialize arguments)
        final_resp = {
            "answer": "checking",
            "tool_calls": [
                {
                    "id": "call_abc",
                    "name": "weather",
                    "arguments": {"city": "Berlin"}
                }
            ]
        }
        expected = {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": '{"city": "Berlin"}'
                    }
                }
            ]
        }
        self.assertEqual(assistant_message(final_resp), expected)

    def test_var_args_kwargs_skipped_in_tool_schema(self):
        self.llm = LLM(model="test")

        def search_docs(query: str, *args, **kwargs) -> str:
            """Search docs."""
            return "ok"

        captured = self._fake_chat_stream(self.llm, [_chat_chunk("ok")])

        list(self.llm.stream_response(input="find notes", tools=[search_docs]))

        properties = captured["tools"][0]["function"]["parameters"]["properties"]
        self.assertEqual(set(properties), {"query"})

    def test_invalid_tool_name_raises(self):
        from llm_sdk import ConfigurationError

        self.llm = LLM(model="test")
        with self.assertRaises(ConfigurationError):
            self.llm._tool_preparator.prepare([lambda x: x])

    def test_any_annotation_maps_to_string(self):
        from typing import Any

        from llm_sdk import SchemaConverter, ToolPreparator

        def tool(a: Any, b: str) -> str:
            return b

        defs = ToolPreparator(SchemaConverter()).prepare([tool]).definitions
        self.assertEqual(defs[0]["function"]["parameters"]["properties"]["a"]["type"], "string")

    def test_enum_converted_to_enum_schema(self):
        from enum import Enum

        from llm_sdk import SchemaConverter, ToolPreparator

        class Color(Enum):
            RED = "red"
            GREEN = "green"

        class Verdict:
            sentiment: str
            color: Color

        schema = SchemaConverter().convert_class_to_schema(Verdict)
        self.assertEqual(
            schema["json_schema"]["schema"]["properties"]["color"],
            {"enum": ["red", "green"]},
        )

        def pick(c: Color) -> str:
            return c.value

        defs = ToolPreparator(SchemaConverter()).prepare([pick]).definitions
        self.assertEqual(defs[0]["function"]["parameters"]["required"], ["c"])
        self.assertEqual(
            defs[0]["function"]["parameters"]["properties"]["c"],
            {"enum": ["red", "green"]},
        )

    def test_tool_call_without_id_gets_fallback_id(self):
        from llm_sdk import EventBuilder, ToolCallStreamHandler

        handler = ToolCallStreamHandler(EventBuilder())
        tc = SimpleNamespace(
            index=0,
            id=None,
            function=SimpleNamespace(name="weather", arguments='{"city":"Berlin"}'),
        )
        handler.process_chunk([tc])
        calls = handler.finalize()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["id"], "call_0")
        self.assertEqual(calls[0]["name"], "weather")
        self.assertEqual(calls[0]["arguments"], {"city": "Berlin"})

    def test_stop_reason_from_chat_finish_reason(self):
        self.llm = LLM(model="test")
        final = SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(delta=SimpleNamespace(content="done"), finish_reason="stop")],
        )
        self._fake_chat_stream(self.llm, [final])

        events = list(self.llm.stream_response(input="x", final=True))
        final_event = next(event for event in events if event["type"] == "final")
        self.assertEqual(final_event["content"]["stop_reason"], "stop")

    def test_stop_reason_normalizes_function_call(self):
        self.llm = LLM(model="test")
        final = SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(delta=SimpleNamespace(content=""), finish_reason="function_call")],
        )
        self._fake_chat_stream(self.llm, [final])

        events = list(self.llm.stream_response(input="x", final=True))
        final_event = next(event for event in events if event["type"] == "final")
        self.assertEqual(final_event["content"]["stop_reason"], "tool_calls")

    def test_stop_reason_from_responses_completed(self):
        self.llm = LLM(model="gpt-5-test", use_responses_api=True)

        events = [
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(status="completed", usage=None),
            ),
        ]

        def create(**kwargs):
            return iter(events)

        self.llm._client = SimpleNamespace(responses=SimpleNamespace(create=create))

        result = list(self.llm.stream_response(input="x", final=True))
        final_event = next(event for event in result if event["type"] == "final")
        self.assertEqual(final_event["content"]["stop_reason"], "stop")

    def test_stop_reason_from_responses_incomplete(self):
        self.llm = LLM(model="gpt-5-test", use_responses_api=True)

        events = [
            SimpleNamespace(
                type="response.incomplete",
                response=SimpleNamespace(
                    status="incomplete",
                    usage=None,
                    incomplete_details=SimpleNamespace(reason="max_output_tokens"),
                ),
            ),
        ]

        def create(**kwargs):
            return iter(events)

        self.llm._client = SimpleNamespace(responses=SimpleNamespace(create=create))

        result = list(self.llm.stream_response(input="x", final=True))
        final_event = next(event for event in result if event["type"] == "final")
        self.assertEqual(final_event["content"]["stop_reason"], "length")

    def test_image_base64_mime_sniffing(self):
        import base64 as b64

        from llm_sdk import ImageProcessor

        png = b64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8).decode()
        jpeg = b64.b64encode(b"\xff\xd8\xff\xe0" + b"\x00" * 8).decode()
        webp = b64.b64encode(b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 4).decode()
        gif = b64.b64encode(b"GIF89a" + b"\x00" * 8).decode()

        def mime_of(data):
            return ImageProcessor._from_base64(data)["image_url"]["url"].split(";")[0]

        self.assertEqual(mime_of(png), "data:image/png")
        self.assertEqual(mime_of(jpeg), "data:image/jpeg")
        self.assertEqual(mime_of(webp), "data:image/webp")
        self.assertEqual(mime_of(gif), "data:image/gif")

    def test_new_client_is_independent_and_safe_to_close(self):
        self.llm = LLM(model="test")

        temp = self.llm._new_client(max_retries=0)

        self.assertIsNot(temp, self.llm._client)
        self.assertIsNot(temp._client, self.llm._client._client)
        temp.close()
        self.assertFalse(self.llm._client._client.is_closed)

    def test_close_async_temp_client_keeps_main_transport_open(self):
        from llm_sdk import _close_async_resource

        self.llm = LLM(model="test")

        temp = self.llm._new_client(max_retries=0, async_client=True)

        self.assertIsNot(temp, self.llm._async_client)
        self.assertIsNot(temp._client, self.llm._async_client._client)
        _close_async_resource(temp)
        self.assertFalse(self.llm._async_client._client.is_closed)

    def test_stream_with_max_retries_uses_and_closes_temp_client(self):
        import unittest.mock as mock

        self.llm = LLM(model="test")
        self._fake_chat_stream(self.llm, [_chat_chunk("ok")])

        temp_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: iter([_chat_chunk("ok")]))
            ),
            closed=False,
        )
        temp_client.close = lambda: setattr(temp_client, "closed", True)

        with mock.patch.object(self.llm, "_new_client", return_value=temp_client):
            events = list(self.llm.stream_response(input="hi", max_retries=0))

        self.assertEqual(events[0]["content"], "ok")
        self.assertTrue(temp_client.closed)

    def test_create_error_raises_model_request_error(self):
        from llm_sdk import ModelRequestError

        self.llm = LLM(model="test")

        def create(**kwargs):
            raise RuntimeError("boom")

        self.llm._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        with self.assertRaises(ModelRequestError):
            list(self.llm.stream_response(input="hi"))

    def test_stream_error_raises_model_request_error(self):
        from llm_sdk import ModelRequestError

        self.llm = LLM(model="test")

        def create(**kwargs):
            def gen():
                yield _chat_chunk("ok")
                raise RuntimeError("boom")

            return gen()

        self.llm._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        with self.assertRaises(ModelRequestError):
            list(self.llm.stream_response(input="hi"))

    def test_response_without_final_raises_runtime_error(self):
        import unittest.mock as mock

        self.llm = LLM(model="test")
        fake_stream = iter(
            [{"type": "answer", "content": "x"}, {"type": "done", "content": None}]
        )

        with (
            mock.patch.object(self.llm, "stream_response", return_value=fake_stream),
            self.assertRaisesRegex(RuntimeError, "No final response received"),
        ):
            self.llm.response(input="hi")

    def test_stream_options_fallback_on_eager_error(self):
        import httpx
        from openai import APIStatusError

        self.llm = LLM(model="test")
        captured = {}

        def make_error():
            response = httpx.Response(
                400, request=httpx.Request("POST", "http://localhost/v1/chat/completions")
            )
            return APIStatusError("bad request", response=response, body=None)

        state = {"attempts": 0}

        def create(**kwargs):
            captured.clear()
            captured.update(kwargs)
            state["attempts"] += 1
            if state["attempts"] == 1:
                raise make_error()
            return iter([_chat_chunk("ok")])

        self.llm._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        events = list(self.llm.stream_response(input="hi"))

        self.assertEqual(state["attempts"], 2)
        self.assertNotIn("stream_options", captured)
        self.assertEqual(events[0]["content"], "ok")

    def test_stream_options_fallback_on_lazy_error(self):
        import httpx
        from openai import APIStatusError

        self.llm = LLM(model="test")
        captured = {}

        def make_error():
            response = httpx.Response(
                400, request=httpx.Request("POST", "http://localhost/v1/chat/completions")
            )
            return APIStatusError("bad request", response=response, body=None)

        state = {"attempts": 0}

        def create(**kwargs):
            captured.clear()
            captured.update(kwargs)
            state["attempts"] += 1
            if state["attempts"] == 1:
                def gen():
                    raise make_error()
                    yield  # pragma: no cover
                return gen()
            return iter([_chat_chunk("ok")])

        self.llm._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        events = list(self.llm.stream_response(input="hi"))

        self.assertEqual(state["attempts"], 2)
        self.assertNotIn("stream_options", captured)
        self.assertEqual(events[0]["content"], "ok")

    def test_stream_options_error_without_stream_options_not_retried(self):
        import httpx
        from openai import APIStatusError

        from llm_sdk import ModelRequestError

        self.llm = LLM(model="test")

        def make_error():
            response = httpx.Response(
                500, request=httpx.Request("POST", "http://localhost/v1/chat/completions")
            )
            return APIStatusError("server error", response=response, body=None)

        def create(**kwargs):
            def gen():
                raise make_error()
                yield  # pragma: no cover

            return gen()

        self.llm._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        with self.assertRaises(ModelRequestError):
            list(self.llm.stream_response(input="hi"))

    def test_tool_call_without_index_falls_back_monotonic(self):
        from llm_sdk import EventBuilder, ToolCallStreamHandler

        handler = ToolCallStreamHandler(EventBuilder(), {})

        tc = SimpleNamespace(
            index=None,
            id="call_1",
            function=SimpleNamespace(name="get_weather", arguments='{"city":"Berlin"}'),
        )
        tc2 = SimpleNamespace(
            index=None,
            id="call_2",
            function=SimpleNamespace(name="get_time", arguments='{"tz":"UTC"}'),
        )

        events = handler.process_chunk([tc])
        events += handler.process_chunk([tc2])
        events += handler.process_chunk(None)

        self.assertTrue(events)
        self.assertTrue(any(e["content"]["id"] == "call_1" for e in events if e["type"] == "tool_call"))
        self.assertTrue(any(e["content"]["id"] == "call_2" for e in events if e["type"] == "tool_call"))

    def test_pep604_optional_schema_and_required_fields(self):
        from llm_sdk import SchemaConverter, ToolPreparator

        class Output:
            required: str
            optional_pep604: str | None
            optional_typing: "Optional[int]"

        schema = SchemaConverter().convert_class_to_schema(Output)["json_schema"]["schema"]
        self.assertEqual(schema["required"], ["required"])
        self.assertEqual(
            schema["properties"]["optional_pep604"],
            {"anyOf": [{"type": "string"}, {"type": "null"}]},
        )
        self.assertEqual(
            schema["properties"]["optional_typing"],
            {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        )

        def tool(required: str, optional: str | None, optional_default: int | None = None):
            return required, optional, optional_default

        definition = ToolPreparator(SchemaConverter()).prepare([tool]).definitions[0]
        parameters = definition["function"]["parameters"]
        self.assertEqual(parameters["required"], ["required"])
        self.assertEqual(
            parameters["properties"]["optional"],
            {"anyOf": [{"type": "string"}, {"type": "null"}]},
        )

    def test_tool_calls_without_index_are_keyed_by_id(self):
        from llm_sdk import EventBuilder, ToolCallStreamHandler

        handler = ToolCallStreamHandler(EventBuilder())
        handler.process_chunk([
            SimpleNamespace(
                index=None,
                id="call_weather",
                function=SimpleNamespace(name="weather", arguments='{"city":"'),
            )
        ])
        handler.process_chunk([
            SimpleNamespace(
                index=None,
                id="call_weather",
                function=SimpleNamespace(name=None, arguments='Berlin"}'),
            )
        ])

        calls = handler.finalize()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["id"], "call_weather")
        self.assertEqual(calls[0]["arguments"], {"city": "Berlin"})

    def test_parallel_tool_calls_without_index_do_not_collide(self):
        from llm_sdk import EventBuilder, ToolCallStreamHandler

        handler = ToolCallStreamHandler(EventBuilder())
        chunks = [
            SimpleNamespace(
                index=None,
                id="call_weather",
                function=SimpleNamespace(name="weather", arguments='{"city":"'),
            ),
            SimpleNamespace(
                index=None,
                id="call_time",
                function=SimpleNamespace(name="time", arguments='{"timezone":"'),
            ),
            SimpleNamespace(
                index=None,
                id="call_weather",
                function=SimpleNamespace(name=None, arguments='Berlin"}'),
            ),
            SimpleNamespace(
                index=None,
                id="call_time",
                function=SimpleNamespace(name=None, arguments='UTC"}'),
            ),
        ]
        events = []
        for chunk in chunks:
            events.extend(handler.process_chunk([chunk]))
        events.extend(handler.process_chunk(None))

        complete = {
            event["content"]["id"]: event["content"]
            for event in events
            if event["type"] == EventType.TOOL_CALL.value
        }
        self.assertEqual(complete["call_weather"]["arguments"], {"city": "Berlin"})
        self.assertEqual(complete["call_time"]["arguments"], {"timezone": "UTC"})

    def test_indexless_idless_complete_calls_start_new_fallback_keys(self):
        from llm_sdk import EventBuilder, ToolCallStreamHandler

        handler = ToolCallStreamHandler(EventBuilder())
        handler.process_chunk([
            SimpleNamespace(
                index=None,
                id=None,
                function=SimpleNamespace(name="weather", arguments='{"city":"Berlin"}'),
            )
        ])
        handler.process_chunk([
            SimpleNamespace(
                index=None,
                id=None,
                function=SimpleNamespace(name="time", arguments='{"timezone":"UTC"}'),
            )
        ])

        calls = handler.finalize()
        self.assertEqual([call["name"] for call in calls], ["weather", "time"])
        self.assertEqual(calls[0]["arguments"], {"city": "Berlin"})
        self.assertEqual(calls[1]["arguments"], {"timezone": "UTC"})

    def test_resolve_messages_system_requires_list(self):
        from llm_sdk import _resolve_messages

        with self.assertRaisesRegex(ValueError, "messages must be a list"):
            _resolve_messages(messages={"role": "user", "content": "hi"}, system="sys")

    def test_context_manager_closes_clients(self):
        with LLM(model="test") as llm:
            self.assertFalse(llm._client._client.is_closed)
        self.assertTrue(llm._client._client.is_closed)

    def test_module_list_models_uses_defaults(self):
        from llm_sdk import list_models

        result = list_models(
            api_key="x",
            base_url="http://127.0.0.1:1/v1",
            max_retries=0,
            fallback=["fallback-model"],
        )
        self.assertEqual(result, ["fallback-model"])

    def test_module_list_models_normalizes_base_url(self):
        from llm_sdk import _norm_api_base

        self.assertEqual(_norm_api_base("http://localhost:1234"), "http://localhost:1234/v1")
        self.assertEqual(_norm_api_base("http://localhost:1234/"), "http://localhost:1234/v1")
        self.assertEqual(_norm_api_base("http://localhost:1234/v1"), "http://localhost:1234/v1")
        self.assertEqual(_norm_api_base("http://localhost:1234/v2/"), "http://localhost:1234/v2")


async def _async_chat_stream(chunks):
    for chunk in chunks:
        yield chunk


async def _async_responses_stream(events):
    for event in events:
        yield event


async def _noop_async_close():
    return None


class _AsyncCloseOnlyStream:
    def __init__(self, items):
        self._iterator = _async_responses_stream(items)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._iterator.__anext__()

    async def close(self):
        self.closed = True


class LLMAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        llm = getattr(self, "llm", None)
        if llm is not None:
            await llm.aclose()

    def _fake_async_chat_stream(self, llm, chunks, captured=None):
        async def create(**kwargs):
            if captured is not None:
                captured.clear()
                captured.update(kwargs)
            return _async_chat_stream(chunks)

        llm._async_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            close=_noop_async_close,
        )

    def _fake_async_responses_stream(self, llm, events):
        async def create(**kwargs):
            return _async_responses_stream(events)

        llm._async_client = SimpleNamespace(
            responses=SimpleNamespace(create=create),
            close=_noop_async_close,
        )

    async def test_async_chat_stream(self):
        self.llm = LLM(model="test")
        self._fake_async_chat_stream(self.llm, [_chat_chunk("ok")])

        events = [event async for event in self.llm.async_stream_response(input="hello")]

        self.assertEqual(events[0]["content"], "ok")
        self.assertEqual(events[-1]["type"], "done")

    async def test_async_chat_stream_awaits_async_close_method(self):
        self.llm = LLM(model="test")
        stream = _AsyncCloseOnlyStream([_chat_chunk("ok")])

        async def create(**kwargs):
            return stream

        self.llm._async_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            close=_noop_async_close,
        )

        [event async for event in self.llm.async_stream_response(input="hello")]

        self.assertTrue(stream.closed)

    async def test_async_response_returns_final(self):
        self.llm = LLM(model="test")
        chunks = [
            _chat_chunk("Hel"),
            _chat_chunk("lo!"),
            SimpleNamespace(
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
                choices=[SimpleNamespace(delta=None, finish_reason="stop")],
            ),
        ]
        self._fake_async_chat_stream(self.llm, chunks)

        response = await self.llm.async_response(input="hi")

        self.assertEqual(response["answer"], "Hello!")
        self.assertEqual(response["stop_reason"], "stop")

    async def test_async_responses_stream(self):
        self.llm = LLM(model="gpt-5-test", use_responses_api=True)

        events = [
            SimpleNamespace(
                type="response.output_text.delta",
                delta="Hi",
            ),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(status="completed", usage=None),
            ),
        ]
        self._fake_async_responses_stream(self.llm, events)

        result = [event async for event in self.llm.async_stream_response(input="x", final=True)]

        self.assertTrue(any(e["type"] == "answer" and e["content"] == "Hi" for e in result))
        self.assertTrue(any(e["type"] == "final" for e in result))

    async def test_async_responses_stream_awaits_async_close_method(self):
        self.llm = LLM(model="gpt-5-test", use_responses_api=True)
        stream = _AsyncCloseOnlyStream([
            SimpleNamespace(type="response.output_text.delta", delta="Hi"),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(status="completed", usage=None),
            ),
        ])

        async def create(**kwargs):
            return stream

        self.llm._async_client = SimpleNamespace(
            responses=SimpleNamespace(create=create),
            close=_noop_async_close,
        )

        [event async for event in self.llm.async_stream_response(input="x")]

        self.assertTrue(stream.closed)

    async def test_async_stream_error_raises_model_request_error(self):
        from llm_sdk import ModelRequestError

        self.llm = LLM(model="test")

        async def create(**kwargs):
            async def gen():
                yield _chat_chunk("ok")
                raise RuntimeError("boom")

            return gen()

        self.llm._async_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            close=_noop_async_close,
        )

        with self.assertRaises(ModelRequestError):
            async for _ in self.llm.async_stream_response(input="hi"):
                pass

    async def test_async_list_models_uses_and_closes_temp_client(self):
        import unittest.mock as mock

        self.llm = LLM(model="test")

        async def _fake_models_list():
            return SimpleNamespace(data=[SimpleNamespace(id="a"), SimpleNamespace(id="b")])

        temp_client = SimpleNamespace(
            models=SimpleNamespace(list=_fake_models_list),
            closed=False,
        )

        async def _close():
            temp_client.closed = True

        temp_client.close = _close

        with mock.patch.object(self.llm, "_new_client", return_value=temp_client):
            models = await self.llm.async_list_models(max_retries=0)

        self.assertEqual(models, ["a", "b"])
        self.assertTrue(temp_client.closed)

    async def test_aclose_closes_clients(self):
        llm = LLM(model="test")
        await llm.aclose()
        self.assertTrue(llm._client._client.is_closed)
        self.assertTrue(llm._async_client._client.is_closed)


if __name__ == "__main__":
    unittest.main()
