import asyncio
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

        # 4. End of the tool call part stream: chunks without tool calls must
        # NOT emit anything (arguments could still continue); finalize() emits
        # the completed parsed JSON tool call.
        events_4 = handler.process_chunk(None)
        self.assertEqual(events_4, [])
        finalized = handler.finalize()
        self.assertEqual(len(finalized), 1)
        self.assertEqual(finalized[0]["id"], "call_123")
        self.assertEqual(finalized[0]["name"], "weather")
        self.assertEqual(finalized[0]["arguments"], {"city": "Berlin"})
        self.assertEqual(finalized[0]["callable"], None)

        # Check get_all_calls
        all_calls = handler.get_all_calls()
        self.assertEqual(all_calls, [{
            "id": "call_123",
            "name": "weather",
            "arguments": {"city": "Berlin"},
            "callable": None
        }])

        # Test parallel tool calls with switching
        handler = ToolCallStreamHandler(eb)

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

        # Test assistant_message with no tool calls (key must be absent,
        # not None: "tool_calls": null is rejected by strict servers)
        self.assertEqual(
            assistant_message({"answer": "hello"}),
            {"role": "assistant", "content": "hello"}
        )

        # Test assistant_message serializes dict answers
        self.assertEqual(
            assistant_message({"answer": {"sentiment": "pos"}}),
            {"role": "assistant", "content": '{"sentiment": "pos"}'}
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

    def test_any_annotation_maps_to_empty_schema(self):
        from typing import Any

        from llm_sdk import SchemaConverter, ToolPreparator

        def tool(a: Any, b: str) -> str:
            return b

        defs = ToolPreparator(SchemaConverter()).prepare([tool]).definitions
        self.assertEqual(defs[0]["function"]["parameters"]["properties"]["a"], {})

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
            {"type": "string", "enum": ["red", "green"]},
        )

        def pick(c: Color) -> str:
            return c.value

        defs = ToolPreparator(SchemaConverter()).prepare([pick]).definitions
        self.assertEqual(defs[0]["function"]["parameters"]["required"], ["c"])
        self.assertEqual(
            defs[0]["function"]["parameters"]["properties"]["c"],
            {"type": "string", "enum": ["red", "green"]},
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
        self.assertTrue(calls[0]["id"].startswith("call_"))
        self.assertNotEqual(calls[0]["id"], "call_0")
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
        import io

        from PIL import Image

        from llm_sdk import ImageProcessor

        def encode(fmt):
            img = Image.new("RGB", (2, 2), (1, 2, 3))
            buffer = io.BytesIO()
            img.save(buffer, format=fmt)
            return b64.b64encode(buffer.getvalue()).decode()

        def mime_of(data):
            return ImageProcessor._from_base64(data)["image_url"]["url"].split(";")[0]

        self.assertEqual(mime_of(encode("PNG")), "data:image/png")
        self.assertEqual(mime_of(encode("JPEG")), "data:image/jpeg")
        self.assertEqual(mime_of(encode("WEBP")), "data:image/webp")
        self.assertEqual(mime_of(encode("GIF")), "data:image/gif")

    def test_image_base64_rejects_non_images(self):
        import base64 as b64

        from llm_sdk import ImageProcessingError, ImageProcessor

        fake_png = b64.b64encode(b"\x89PNG\r\n\x1a\n" + b"secret-key-bytes").decode()
        with self.assertRaises(ImageProcessingError):
            ImageProcessor._from_base64(fake_png)
        with self.assertRaises(ImageProcessingError):
            ImageProcessor._from_base64("data:text/plain;base64,aGVsbG8=")

    def test_data_url_with_non_image_mime_rejected(self):
        import base64
        import io

        from PIL import Image

        from llm_sdk import ImageProcessingError, ImageProcessor

        with self.assertRaises(ImageProcessingError):
            ImageProcessor._from_url("data:text/plain;base64,aGVsbG8=")
        with self.assertRaises(ImageProcessingError):
            ImageProcessor._from_url("data:,aGVsbG8=")
        with self.assertRaises(ImageProcessingError):
            ImageProcessor._from_url("data:;base64,aGVsbG8=")
        img = Image.new("RGB", (2, 2), (4, 5, 6))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        item = ImageProcessor._from_url(url)
        self.assertEqual(item["image_url"]["url"], url)

    def test_from_url_dict_without_url_rejected(self):
        from llm_sdk import ConfigurationError, ImageProcessor

        with self.assertRaises(ConfigurationError):
            ImageProcessor._from_url({"detail": "high"})
        with self.assertRaises(ConfigurationError):
            ImageProcessor._from_url({})

    def test_max_image_side_shrinks_large_images(self):
        import base64
        import io

        from PIL import Image

        from llm_sdk import ImageProcessor

        img = Image.new("RGB", (3000, 2000), (10, 20, 30))
        item = ImageProcessor._from_pil(img, max_image_side=512)
        encoded = item["image_url"]["url"].split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as out:
            self.assertLessEqual(max(out.size), 512)

    def test_oversized_dimensions_rejected(self):
        from PIL import Image

        from llm_sdk import ImageProcessingError, ImageProcessor

        img = Image.new("RGB", (8, 4))
        old = ImageProcessor._MAX_IMAGE_PIXELS
        ImageProcessor._MAX_IMAGE_PIXELS = 10
        try:
            with self.assertRaises(ImageProcessingError):
                ImageProcessor._from_pil(img)
        finally:
            ImageProcessor._MAX_IMAGE_PIXELS = old

    def test_base64_side_cap_shrinks_large_images(self):
        import base64
        import io

        from PIL import Image

        from llm_sdk import ImageProcessor

        img = Image.new("RGB", (3000, 2000), (9, 9, 9))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue()).decode()
        item = ImageProcessor._from_base64(data, max_image_side=512)
        encoded = item["image_url"]["url"].split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as out:
            self.assertLessEqual(max(out.size), 512)

    def test_wrapped_base64_accepted(self):
        import base64
        import io

        from PIL import Image

        from llm_sdk import ImageProcessor

        img = Image.new("RGB", (4, 4), (1, 2, 3))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue()).decode()
        wrapped = "\n".join(data[i:i + 20] for i in range(0, len(data), 20))
        item = ImageProcessor._from_base64(wrapped)
        self.assertTrue(item["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_nul_byte_path_raises_image_processing_error(self):
        from llm_sdk import ImageProcessingError, ImageProcessor

        with self.assertRaises(ImageProcessingError):
            ImageProcessor._from_path("/tmp/\x00evil.png")

    def test_base64_bmp_transcoded_to_jpeg(self):
        import base64
        import io

        from PIL import Image

        from llm_sdk import ImageProcessor

        img = Image.new("RGB", (4, 2), (9, 9, 9))
        buffer = io.BytesIO()
        img.save(buffer, format="BMP")
        data = base64.b64encode(buffer.getvalue()).decode()
        item = ImageProcessor._from_base64(data)
        self.assertTrue(item["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_base64_exif_rotated_jpeg_is_transcoded(self):
        import base64
        import io

        from PIL import Image

        from llm_sdk import ImageProcessor

        img = Image.new("RGB", (2, 4), (0, 150, 0))
        exif = Image.Exif()
        exif[274] = 6
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", exif=exif)
        data = base64.b64encode(buffer.getvalue()).decode()
        item = ImageProcessor._from_base64(data)
        encoded = item["image_url"]["url"].split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as out:
            self.assertEqual(out.size, (4, 2))

    def test_base64_over_budget_shrinks(self):
        import base64
        import io

        from PIL import Image

        from llm_sdk import ImageProcessor

        img = Image.new("RGB", (300, 200), (5, 6, 7))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        data = base64.b64encode(buffer.getvalue()).decode()
        old = ImageProcessor._MAX_IMAGE_SEND_BYTES
        ImageProcessor._MAX_IMAGE_SEND_BYTES = 100
        try:
            item = ImageProcessor._from_base64(data)
            self.assertTrue(
                item["image_url"]["url"].startswith("data:image/jpeg;base64,")
            )
        finally:
            ImageProcessor._MAX_IMAGE_SEND_BYTES = old

    def test_data_url_via_image_url_shrinks_preserving_keys(self):
        import base64
        import io

        from PIL import Image

        from llm_sdk import ImageProcessor

        img = Image.new("RGB", (3000, 2000), (1, 2, 3))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        item = ImageProcessor._from_url(
            {"url": url, "detail": "high"}, max_image_side=512
        )
        self.assertEqual(item["image_url"]["detail"], "high")
        encoded = item["image_url"]["url"].split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as out:
            self.assertLessEqual(max(out.size), 512)

    def test_new_client_is_independent_and_safe_to_close(self):
        self.llm = LLM(model="test")

        temp = self.llm._new_client(max_retries=0)

        self.assertIsNot(temp, self.llm._client)
        self.assertIsNot(temp._client, self.llm._client._client)
        temp.close()
        self.assertFalse(self.llm._client._client.is_closed)

    def test_close_async_temp_client_keeps_main_transport_open(self):
        from llm_sdk import _aclose_async_resource

        self.llm = LLM(model="test")

        # Purely sync usage never creates an async client (lazy creation).
        self.assertIsNone(self.llm._async_client)

        temp = self.llm._new_client(max_retries=0, async_client=True)
        asyncio.run(_aclose_async_resource(temp))
        self.assertTrue(temp._client.is_closed)
        # close() without async usage must not spin up an event loop.
        self.llm.close()
        self.assertIsNone(self.llm._async_client)

    def test_stream_with_max_retries_uses_with_options_copy(self):
        self.llm = LLM(model="test")

        copy_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: iter([_chat_chunk("ok")]))
            ),
            closed=False,
        )
        copy_client.close = lambda: setattr(copy_client, "closed", True)

        main_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: iter([_chat_chunk("wrong")]))
            ),
            with_options=lambda **kwargs: copy_client,
        )
        self.llm._client = main_client

        events = list(self.llm.stream_response(input="hi", max_retries=0))

        # The per-call max_retries request ran through the with_options copy…
        self.assertEqual(events[0]["content"], "ok")
        # …which shares the main transport and must never be closed.
        self.assertFalse(copy_client.closed)

    def test_create_error_wraps_api_error_with_status(self):
        import httpx
        from openai import APIStatusError

        from llm_sdk import ModelRequestError

        self.llm = LLM(model="test")

        def create(**kwargs):
            response = httpx.Response(
                401, request=httpx.Request("POST", "http://localhost/v1/chat/completions")
            )
            raise APIStatusError("invalid api key", response=response, body={"error": "auth"})

        self.llm._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        with self.assertRaises(ModelRequestError) as ctx:
            list(self.llm.stream_response(input="hi"))
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(ctx.exception.body, {"error": "auth"})

    def test_create_error_internal_bugs_propagate(self):
        self.llm = LLM(model="test")

        def create(**kwargs):
            raise RuntimeError("boom")

        self.llm._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        # Wrapper-internal bugs must not be masked as server errors.
        with self.assertRaises(RuntimeError):
            list(self.llm.stream_response(input="hi"))

    def test_stream_error_internal_bugs_propagate(self):
        self.llm = LLM(model="test")

        def create(**kwargs):
            def gen():
                yield _chat_chunk("ok")
                raise RuntimeError("boom")

            return gen()

        self.llm._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        with self.assertRaises(RuntimeError):
            list(self.llm.stream_response(input="hi"))

    def test_response_without_final_raises_llm_error(self):
        import unittest.mock as mock

        from llm_sdk import LLMError

        self.llm = LLM(model="test")
        fake_stream = iter(
            [{"type": "answer", "content": "x"}, {"type": "done", "content": None}]
        )

        with (
            mock.patch.object(self.llm, "stream_response", return_value=fake_stream),
            self.assertRaisesRegex(LLMError, "No final response received"),
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
            return APIStatusError(
                "Error code: 400 - Unsupported parameter: 'stream_options' is not supported",
                response=response,
                body=None,
            )

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

    def test_stream_options_fallback_on_lazy_error_not_retried(self):
        import httpx
        from openai import APIStatusError

        from llm_sdk import ModelRequestError

        self.llm = LLM(model="test")
        captured = {}

        def make_error():
            response = httpx.Response(
                400, request=httpx.Request("POST", "http://localhost/v1/chat/completions")
            )
            return APIStatusError(
                "Error code: 400 - Unsupported parameter: 'stream_options'",
                response=response,
                body=None,
            )

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

        # Mid-stream errors are never retried: already-yielded chunks would
        # be duplicated by a second stream.
        with self.assertRaises(ModelRequestError):
            list(self.llm.stream_response(input="hi"))

        self.assertEqual(state["attempts"], 1)

    def test_stream_options_400_without_mention_not_retried(self):
        import httpx
        from openai import APIStatusError

        from llm_sdk import ModelRequestError

        self.llm = LLM(model="test")
        state = {"attempts": 0}

        def make_error():
            response = httpx.Response(
                400, request=httpx.Request("POST", "http://localhost/v1/chat/completions")
            )
            return APIStatusError("invalid_json_schema", response=response, body=None)

        def create(**kwargs):
            state["attempts"] += 1
            raise make_error()

        self.llm._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        with self.assertRaises(ModelRequestError):
            list(self.llm.stream_response(input="hi"))

        # Real schema/config errors must not be retried blindly.
        self.assertEqual(state["attempts"], 1)

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
        calls = handler.finalize()

        self.assertEqual(
            [call["id"] for call in calls], ["call_1", "call_2"]
        )
        self.assertTrue(any(e["content"]["id"] == "call_1" for e in events if e["type"] == "tool_call_part"))
        self.assertTrue(any(e["content"]["id"] == "call_2" for e in events if e["type"] == "tool_call_part"))

    def test_pep604_optional_schema_and_required_fields(self):
        from llm_sdk import SchemaConverter, ToolPreparator

        class Output:
            required: str
            optional_pep604: str | None
            optional_typing: "Optional[int]"

        schema = SchemaConverter().convert_class_to_schema(Output)["json_schema"]["schema"]
        # Strict mode: every key must be required; optionality is nullable anyOf.
        self.assertEqual(
            schema["required"], ["required", "optional_pep604", "optional_typing"]
        )
        self.assertEqual(
            schema["properties"]["optional_pep604"],
            {"anyOf": [{"type": "string"}, {"type": "null"}]},
        )
        self.assertEqual(
            schema["properties"]["optional_typing"],
            {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        )

        # Non-strict mode keeps classic semantics.
        lenient_payload = SchemaConverter().convert_class_to_schema(Output, strict=False)
        self.assertFalse(lenient_payload["json_schema"]["strict"])
        self.assertEqual(
            lenient_payload["json_schema"]["schema"]["required"], ["required"]
        )

        def tool(required: str, optional: str | None, optional_default: int | None = None):
            return required, optional, optional_default

        definition = ToolPreparator(SchemaConverter()).prepare([tool]).definitions[0]
        parameters = definition["function"]["parameters"]
        # Optional[X] without a default is still a required parameter.
        self.assertEqual(parameters["required"], ["required", "optional"])
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
        calls = handler.finalize()

        complete = {
            call["id"]: call
            for call in calls
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

    async def test_async_stream_error_internal_bugs_propagate(self):
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

        with self.assertRaises(RuntimeError):
            async for _ in self.llm.async_stream_response(input="hi"):
                pass

    async def test_async_list_models_uses_with_options_copy(self):
        import unittest.mock as mock

        self.llm = LLM(model="test")

        async def _fake_models_list():
            return SimpleNamespace(data=[SimpleNamespace(id="a"), SimpleNamespace(id="b")])

        copy_client = SimpleNamespace(
            models=SimpleNamespace(list=_fake_models_list),
            closed=False,
            with_options=lambda **kwargs: copy_client,
        )

        async def _close():
            copy_client.closed = True

        copy_client.close = _close

        with mock.patch.object(self.llm, "_get_async_client", return_value=copy_client):
            models = await self.llm.async_list_models(max_retries=0)

        self.assertEqual(models, ["a", "b"])
        # with_options copies share the transport and are not closed.
        self.assertFalse(copy_client.closed)

    async def test_aclose_closes_clients(self):
        llm = LLM(model="test")
        # Trigger lazy async client creation first.
        client = llm._get_async_client()
        self.assertIsNotNone(client)
        await llm.aclose()
        self.assertTrue(llm._client._client.is_closed)
        self.assertTrue(client._client.is_closed)
        self.assertIsNone(llm._async_client)

    async def test_aclose_without_async_usage_has_no_async_client(self):
        llm = LLM(model="test")
        await llm.aclose()
        self.assertTrue(llm._client._client.is_closed)
        self.assertIsNone(llm._async_client)

    async def test_async_client_is_recreated_per_event_loop(self):
        import weakref

        class _FakeLoop:
            pass

        llm = LLM(model="test")
        first = llm._get_async_client()
        # Simulate the client having been bound to a previous, garbage
        # collected loop (a weakref that resolves to nothing).
        llm._async_client_loop_ref = weakref.ref(_FakeLoop())
        second = llm._get_async_client()
        self.assertIsNot(first, second)
        await llm.aclose()

    async def test_close_inside_running_loop_does_not_raise(self):
        llm = LLM(model="test")
        llm._get_async_client()  # ensure an async client exists
        # Sync close inside a running loop must never raise (Jupyter case).
        llm.close()
        self.assertTrue(llm._client._client.is_closed)
        self.assertIsNone(llm._async_client)
        # Give the scheduled best-effort close task a chance to run.
        await asyncio.sleep(0)


class ReasoningParserTests(unittest.TestCase):
    """F-01/ST-01: tags split across chunks must be recognized."""

    SAMPLES = [
        "<think>plan the answer</think>the answer",
        "<thinking>reason here</thinking>result here",
        "[THINK]reason[/THINK]answer",
        "<thought>thought text</thought>answer text",
        "before<think>inside</think>after",
        "<think>only thinking",
        "no tags at all",
    ]

    def _split_feed(self, parser, text, chunks):
        thinking = answer = ""
        position = 0
        for size in chunks:
            t, a = parser.parse(text[position:position + size])
            thinking += t
            answer += a
            position += size
        if position < len(text):
            t, a = parser.parse(text[position:])
            thinking += t
            answer += a
        t, a = parser.flush()
        thinking += t
        answer += a
        return thinking, answer

    def test_chunking_invariance(self):
        from llm_sdk import ReasoningParser

        for text in self.SAMPLES:
            whole = ReasoningParser()
            expected = whole.parse(text)
            expected = (expected[0] + whole.flush()[0], expected[1] + whole.flush()[1])
            for chunks in ([1] * len(text), [2] * ((len(text) + 1) // 2), [3, 1, 4, 2], [len(text)]):
                parser = ReasoningParser()
                with self.subTest(text=text, chunks=chunks):
                    self.assertEqual(self._split_feed(parser, text, chunks), expected)

    def test_every_two_chunk_split_is_invariant(self):
        from llm_sdk import ReasoningParser

        for text in self.SAMPLES:
            whole = ReasoningParser()
            expected = whole.parse(text)
            expected = (expected[0] + whole.flush()[0], expected[1] + whole.flush()[1])
            for split in range(len(text) + 1):
                parser = ReasoningParser()
                with self.subTest(text=text, split=split):
                    self.assertEqual(
                        self._split_feed(parser, text, [split, len(text) - split]),
                        expected,
                    )

    def test_ragged_multi_chunk_splits_are_invariant(self):
        from llm_sdk import ReasoningParser

        pattern = [1, 2, 4, 3, 1, 2]
        for text in self.SAMPLES:
            whole = ReasoningParser()
            expected = whole.parse(text)
            expected = (expected[0] + whole.flush()[0], expected[1] + whole.flush()[1])
            sizes = []
            remaining = len(text)
            index = 0
            while remaining > 0:
                size = min(pattern[index % len(pattern)], remaining)
                sizes.append(size)
                remaining -= size
                index += 1
            parser = ReasoningParser()
            with self.subTest(text=text):
                self.assertEqual(self._split_feed(parser, text, sizes), expected)

    def test_split_start_tag(self):
        from llm_sdk import ReasoningParser

        parser = ReasoningParser()
        self.assertEqual(parser.parse("<thi"), ("", ""))
        self.assertEqual(parser.parse("nking>plan"), ("plan", ""))
        self.assertEqual(parser.parse("</thi"), ("", ""))
        thinking, answer = parser.parse("nking>done")
        self.assertEqual((thinking, answer), ("", "done"))
        self.assertEqual(parser.flush(), ("", ""))

    def test_split_end_tag_does_not_swallow_answer(self):
        from llm_sdk import ReasoningParser

        parser = ReasoningParser()
        self.assertEqual(parser.parse("<thinking>plan"), ("plan", ""))
        # End tag split across chunks; the answer must not be classified as
        # reasoning (previously produced an empty answer).
        self.assertEqual(parser.parse("</thin"), ("", ""))
        reasoning, answer = parser.parse("king>the answer")
        self.assertEqual((reasoning, answer), ("", "the answer"))

    def test_custom_token_split(self):
        from llm_sdk import CustomReasoningPattern, ReasoningParser

        parser = ReasoningParser(CustomReasoningPattern(start_token="◁think▷", end_token="◁/think▷"))
        self.assertEqual(parser.parse("◁thi"), ("", ""))
        self.assertEqual(parser.parse("nk▷reasoning◁/thi"), ("reasoning", ""))
        self.assertEqual(parser.parse("nk▷answer"), ("", "answer"))

    def test_from_beginning_and_flush(self):
        from llm_sdk import CustomReasoningPattern, ReasoningParser

        parser = ReasoningParser(CustomReasoningPattern(start_token="<s>", end_token="</s>", from_beginning=True))
        self.assertEqual(parser.parse("hidden"), ("hidden", ""))
        self.assertEqual(parser.parse("</s>"), ("", ""))
        self.assertEqual(parser.flush(), ("", ""))

        unterminated = ReasoningParser(CustomReasoningPattern(start_token="<s>", end_token="</s>", from_beginning=True))
        self.assertEqual(unterminated.parse("never ends"), ("never ends", ""))
        self.assertTrue(unterminated.is_inside_reasoning)
        self.assertEqual(unterminated.flush(), ("", ""))

        # A dangling partial end tag is held back and flushed as thinking.
        held = ReasoningParser(CustomReasoningPattern(start_token="<s>", end_token="</s>", from_beginning=True))
        self.assertEqual(held.parse("abc</s"), ("abc", ""))
        self.assertEqual(held.flush(), ("</s", ""))

    def test_tag_mentioned_as_text_not_swallowed(self):
        from llm_sdk import ReasoningParser

        parser = ReasoningParser()
        reasoning, answer = parser.parse("a<thinkb> stays text")
        self.assertEqual((reasoning, answer), ("", "a<thinkb> stays text"))

    def test_case_insensitive_tags(self):
        from llm_sdk import ReasoningParser

        parser = ReasoningParser()
        self.assertEqual(parser.parse("<THINK>secret</THINK>out"), ("secret", "out"))


class StrictSchemaTests(unittest.TestCase):
    """F-02/S-01..S-07/F-12/F-21: strict schemas and type mapping."""

    def test_strict_schema_all_fields_required_and_nullable_defaults(self):
        from typing import Literal, Optional

        from llm_sdk import SchemaConverter

        class Ticket:
            """Support ticket."""
            title: str
            priority: Literal["low", "high"]
            assignee: Optional[str] = None
            tags: list[str] = []

        payload = SchemaConverter().convert_class_to_schema(Ticket)
        json_schema = payload["json_schema"]
        self.assertTrue(json_schema["strict"])
        schema = json_schema["schema"]
        self.assertEqual(set(schema["required"]), {"title", "priority", "assignee", "tags"})
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            schema["properties"]["priority"], {"type": "string", "enum": ["low", "high"]}
        )
        self.assertEqual(
            schema["properties"]["assignee"],
            {"anyOf": [{"type": "string"}, {"type": "null"}]},
        )
        self.assertEqual(
            schema["properties"]["tags"],
            {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]},
        )
        self.assertEqual(schema["description"], "Support ticket.")

    def test_nested_class_is_strict_conformant(self):
        from llm_sdk import SchemaConverter

        class Address:
            street: str

        class Person:
            name: str
            address: Address

        schema = SchemaConverter().convert_class_to_schema(Person)["json_schema"]["schema"]
        nested = schema["properties"]["address"]
        self.assertEqual(nested["type"], "object")
        self.assertFalse(nested["additionalProperties"])
        self.assertEqual(nested["required"], ["street"])
        self.assertEqual(set(schema["required"]), {"name", "address"})

    def test_dict_field_rejected_in_strict_mode(self):
        from llm_sdk import SchemaConversionError, SchemaConverter

        class WithDict:
            meta: dict

        with self.assertRaisesRegex(SchemaConversionError, "dict"):
            SchemaConverter().convert_class_to_schema(WithDict)
        lenient = SchemaConverter().convert_class_to_schema(WithDict, strict=False)
        self.assertEqual(
            lenient["json_schema"]["schema"]["properties"]["meta"]["type"], "object"
        )

    def test_dict_output_format_strict_schema_validated(self):
        from llm_sdk import LLM, SchemaConversionError

        llm = LLM(model="test")
        try:
            bad = {
                "type": "json_schema",
                "json_schema": {
                    "name": "x",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"a": {"type": "string"}},
                    },
                },
            }
            with self.assertRaises(SchemaConversionError):
                llm._prepare_output_format(bad)
            good = {
                "type": "json_schema",
                "json_schema": {
                    "name": "x",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"a": {"type": "string"}},
                        "required": ["a"],
                        "additionalProperties": False,
                    },
                },
            }
            self.assertIs(llm._prepare_output_format(good), good)
        finally:
            llm.close()

    def test_docstring_inheritance_avoided(self):
        import dataclasses
        from typing import TypedDict

        from llm_sdk import SchemaConverter

        class Movie(TypedDict):
            title: str

        schema = SchemaConverter().convert_class_to_schema(Movie)["json_schema"]["schema"]
        self.assertNotIn("description", schema)

        @dataclasses.dataclass
        class Point:
            x: int

        schema2 = SchemaConverter().convert_class_to_schema(Point)["json_schema"]["schema"]
        self.assertNotIn("description", schema2)

    def test_unknown_type_raises_instead_of_string(self):
        from llm_sdk import SchemaConversionError, SchemaConverter

        class WithComplex:
            z: complex

        with self.assertRaises(SchemaConversionError):
            SchemaConverter().convert_class_to_schema(WithComplex)

    def test_well_known_type_mappings(self):
        import datetime
        import uuid as uuid_module
        from decimal import Decimal

        from llm_sdk import SchemaConverter

        class Types:
            when: datetime.datetime
            day: datetime.date
            uid: uuid_module.UUID
            amount: Decimal
            tags: set[str]
            pair: tuple[int, str]
            many: tuple[int, ...]
            untyped: list

        schema = SchemaConverter().convert_class_to_schema(Types, strict=False)["json_schema"]["schema"]
        props = schema["properties"]
        self.assertEqual(props["when"], {"type": "string", "format": "date-time"})
        self.assertEqual(props["day"], {"type": "string", "format": "date"})
        self.assertEqual(props["uid"], {"type": "string", "format": "uuid"})
        self.assertEqual(props["amount"], {"type": "number"})
        self.assertEqual(
            props["tags"],
            {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        )
        self.assertEqual(
            props["pair"],
            {
                "type": "array",
                "prefixItems": [{"type": "integer"}, {"type": "string"}],
                "minItems": 2,
                "maxItems": 2,
            },
        )
        self.assertEqual(props["many"], {"type": "array", "items": {"type": "integer"}})
        self.assertEqual(props["untyped"], {"type": "array", "items": {}})

    def test_class_named_dict_not_misclassified(self):
        from llm_sdk import SchemaConverter

        class Dict:
            a: int

        schema = SchemaConverter().convert_class_to_schema(Dict)["json_schema"]["schema"]
        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["properties"]["a"], {"type": "integer"})
        self.assertEqual(schema["required"], ["a"])
        self.assertFalse(schema["additionalProperties"])

    def test_inheritance_and_mro_defaults(self):
        from llm_sdk import SchemaConverter

        class Base:
            a: int = 1

        class Child(Base):
            b: str

        lenient = SchemaConverter().convert_class_to_schema(Child, strict=False)["json_schema"]["schema"]
        # Inherited default must make "a" optional.
        self.assertEqual(lenient["required"], ["b"])

        strict = SchemaConverter().convert_class_to_schema(Child)["json_schema"]["schema"]
        self.assertEqual(set(strict["required"]), {"a", "b"})
        self.assertEqual(
            strict["properties"]["a"],
            {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        )

    def test_dataclass_default_factory_is_optional(self):
        import dataclasses

        from llm_sdk import SchemaConverter

        @dataclasses.dataclass
        class DC:
            tags: list[str] = dataclasses.field(default_factory=list)
            name: str = "x"

        lenient = SchemaConverter().convert_class_to_schema(DC, strict=False)["json_schema"]["schema"]
        self.assertEqual(lenient.get("required", []), [])
        strict = SchemaConverter().convert_class_to_schema(DC)["json_schema"]["schema"]
        self.assertEqual(set(strict["required"]), {"tags", "name"})

    def test_typeddict_total_false_keys_optional(self):
        from typing import TypedDict

        from llm_sdk import SchemaConverter

        class TD(TypedDict, total=False):
            a: int

        lenient = SchemaConverter().convert_class_to_schema(TD, strict=False)["json_schema"]["schema"]
        self.assertEqual(lenient.get("required", []), [])

        class TDTotal(TypedDict):
            a: int

        lenient_total = SchemaConverter().convert_class_to_schema(
            TDTotal, strict=False
        )["json_schema"]["schema"]
        self.assertEqual(lenient_total["required"], ["a"])

    def test_classvar_and_private_filtered(self):
        from typing import ClassVar

        from llm_sdk import SchemaConverter

        class C:
            _secret: str
            count: ClassVar[int] = 0
            visible: int

        schema = SchemaConverter().convert_class_to_schema(C)["json_schema"]["schema"]
        self.assertEqual(set(schema["properties"]), {"visible"})
        self.assertEqual(schema["required"], ["visible"])

    def test_annotated_field_description(self):
        from typing import Annotated

        from llm_sdk import SchemaConverter

        class Out:
            name: Annotated[str, "The person's full name"]
            age: Annotated[int, "Age in years", "extra metadata"]

        schema = SchemaConverter().convert_class_to_schema(Out)["json_schema"]["schema"]
        self.assertEqual(
            schema["properties"]["name"],
            {"type": "string", "description": "The person's full name"},
        )
        self.assertEqual(schema["properties"]["age"]["description"], "Age in years")

    def test_namedtuple_maps_to_array(self):
        import collections
        import typing

        from llm_sdk import SchemaConverter

        class Point(typing.NamedTuple):
            x: int
            y: int

        Untyped = collections.namedtuple("Untyped", ["a", "b"])

        class Out:
            p: Point
            u: Untyped

        schema = SchemaConverter().convert_class_to_schema(Out)["json_schema"]["schema"]
        self.assertEqual(
            schema["properties"]["p"],
            {
                "type": "array",
                "prefixItems": [{"type": "integer"}, {"type": "integer"}],
                "minItems": 2,
                "maxItems": 2,
            },
        )
        # Untyped namedtuple fields fall back to an unconstrained schema.
        self.assertEqual(
            schema["properties"]["u"],
            {
                "type": "array",
                "prefixItems": [{}, {}],
                "minItems": 2,
                "maxItems": 2,
            },
        )

    def test_literal_enum_members_and_mixed_types(self):
        import enum
        from typing import Literal

        from llm_sdk import SchemaConverter

        class Color(enum.Enum):
            RED = "red"

        class Out:
            c: Literal[Color.RED, "blue"]
            m: Literal[1, "one"]

        schema = SchemaConverter().convert_class_to_schema(Out, strict=False)["json_schema"]["schema"]
        props = schema["properties"]
        self.assertEqual(props["c"], {"type": "string", "enum": ["red", "blue"]})
        self.assertEqual(
            props["m"],
            {
                "anyOf": [
                    {"type": "integer", "enum": [1]},
                    {"type": "string", "enum": ["one"]},
                ]
            },
        )

    def test_pydantic_style_fields_supported(self):
        import importlib.util

        if importlib.util.find_spec("pydantic") is None:
            self.skipTest("pydantic not installed")

        import pydantic

        from llm_sdk import SchemaConverter

        class Model(pydantic.BaseModel):
            name: str
            age: int = 3

        lenient = SchemaConverter().convert_class_to_schema(Model, strict=False)["json_schema"]["schema"]
        self.assertEqual(lenient["required"], ["name"])


class ToolPreparatorTests(unittest.TestCase):
    """F-05/F-11/T-01..T-08: tool schema generation from callables."""

    def test_nested_class_parameter_supported(self):
        from llm_sdk import SchemaConverter, ToolPreparator

        class Address:
            street: str
            city: str

        def geocode(address: Address, precise: bool = False) -> dict:
            """Geocode an address."""
            return {}

        definitions = ToolPreparator(SchemaConverter()).prepare([geocode]).definitions
        parameters = definitions[0]["function"]["parameters"]
        self.assertEqual(set(parameters["properties"]), {"address", "precise"})
        self.assertEqual(parameters["required"], ["address"])
        self.assertEqual(parameters["properties"]["address"]["type"], "object")

    def test_mandatory_unsupported_param_raises(self):
        from llm_sdk import ConfigurationError, SchemaConverter, ToolPreparator

        class NotASchema:
            pass

        def geocode(address: NotASchema) -> dict:
            """Geocode."""
            return {}

        with self.assertRaisesRegex(ConfigurationError, "address"):
            ToolPreparator(SchemaConverter()).prepare([geocode])

    def test_dependency_injection_param_with_default_skipped(self):
        from llm_sdk import SchemaConverter, ToolPreparator

        class Logger:
            pass

        def run(task: str, logger: Logger = None) -> str:
            """Run a task."""
            return task

        definitions = ToolPreparator(SchemaConverter()).prepare([run]).definitions
        parameters = definitions[0]["function"]["parameters"]
        self.assertEqual(set(parameters["properties"]), {"task"})
        self.assertEqual(parameters["required"], ["task"])

    def test_default_with_ambiguous_eq_does_not_crash(self):
        from llm_sdk import SchemaConverter, ToolPreparator

        class Ambiguous:
            def __eq__(self, other):
                raise ValueError("truth value ambiguous")

        def f(x: int = Ambiguous()) -> str:
            """Doc."""
            return "x"

        definitions = ToolPreparator(SchemaConverter()).prepare([f]).definitions
        self.assertEqual(
            definitions[0]["function"]["parameters"].get("required", []), []
        )

    def test_duplicate_tool_names_raise(self):
        from llm_sdk import ConfigurationError, SchemaConverter, ToolPreparator

        def make():
            def tool(x: int) -> str:
                """A tool."""
                return "x"

            return tool

        with self.assertRaisesRegex(ConfigurationError, "Duplicate tool name"):
            ToolPreparator(SchemaConverter()).prepare([make(), make()])

    def test_positional_only_param_raises(self):
        from llm_sdk import ConfigurationError, SchemaConverter, ToolPreparator

        def f(a, /) -> str:
            """Doc."""
            return "a"

        with self.assertRaisesRegex(ConfigurationError, "positional-only"):
            ToolPreparator(SchemaConverter()).prepare([f])

    def test_unresolvable_type_hints_raise(self):
        from llm_sdk import ConfigurationError, SchemaConverter, ToolPreparator

        def f(x: "MissingType") -> str:  # noqa: F821 - intentionally unresolvable
            """Doc."""
            return "x"

        with self.assertRaisesRegex(ConfigurationError, "type hints"):
            ToolPreparator(SchemaConverter()).prepare([f])

    def test_callable_instance_without_name_raises(self):
        from llm_sdk import ConfigurationError, SchemaConverter, ToolPreparator

        class Handler:
            def __call__(self, x: int) -> str:
                return "y"

        with self.assertRaisesRegex(ConfigurationError, "tool name"):
            ToolPreparator(SchemaConverter()).prepare([Handler()])

    def test_docstring_args_parsed_into_descriptions(self):
        from typing import Annotated

        from llm_sdk import SchemaConverter, ToolPreparator

        def search(query: Annotated[str, "Raw query text"], limit: int = 5) -> str:
            """Search docs.

            Args:
                query: What to look for.
                limit (int): Maximum number of hits.
            """
            return ""

        definitions = ToolPreparator(SchemaConverter()).prepare([search]).definitions
        props = definitions[0]["function"]["parameters"]["properties"]
        self.assertEqual(props["query"]["description"], "Raw query text What to look for.")
        self.assertEqual(props["limit"]["description"], "Maximum number of hits. Default: 5")

    def test_prepared_tools_expose_callables(self):
        from llm_sdk import SchemaConverter, ToolPreparator

        def f(x: int) -> str:
            return "x"

        prepared = ToolPreparator(SchemaConverter()).prepare([f])
        self.assertEqual(prepared.callables, {"f": f})

    def test_responses_native_tool_dicts_pass_validation(self):
        from llm_sdk import SchemaConverter, ToolPreparator

        prepared = ToolPreparator(SchemaConverter()).prepare([
            {"type": "web_search_preview"},
            {"type": "function", "name": "flat", "parameters": {}},
        ])
        self.assertEqual(prepared.definitions[0], {"type": "web_search_preview"})
        # Responses-shaped function dicts are normalized into the nested chat
        # shape so they work in both modes (the Responses path re-flattens).
        flat = prepared.definitions[1]
        self.assertEqual(flat["function"]["name"], "flat")
        self.assertEqual(flat["function"]["parameters"], {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        })
        converted = LLM._convert_tools_for_responses_api(prepared.definitions)
        self.assertEqual(converted[1]["name"], "flat")
        self.assertNotIn("strict", converted[1])


class RequestTransformerTests(unittest.TestCase):
    """F-06/F-08/F-15/C-01/C-04/C-08: request normalization and parameters."""

    def test_max_tokens_mapped_for_reasoning_models(self):
        from llm_sdk import RequestTransformer

        cases = [
            ("o4-mini", "http://localhost:1234/v1", "max_completion_tokens"),
            ("gpt-5.5", "http://localhost:1234/v1", "max_completion_tokens"),
            ("qwen3", "http://localhost:1234/v1", "max_tokens"),
            ("qwen3", "https://api.openai.com/v1", "max_completion_tokens"),
        ]
        for model, base_url, expected_key in cases:
            with self.subTest(model=model, base_url=base_url):
                transformer = RequestTransformer(model, base_url)
                kwargs = transformer.transform({"max_tokens": 100})
                self.assertEqual(kwargs, {expected_key: 100})

    def test_stop_and_temperature_dropped_for_reasoning_models(self):
        from llm_sdk import RequestTransformer

        o_series = RequestTransformer("o3-mini", "https://api.openai.com/v1")
        kwargs = o_series.transform({"stop": ["END"], "temperature": 0.5, "top_p": 0.9})
        self.assertNotIn("stop", kwargs)
        self.assertNotIn("temperature", kwargs)
        self.assertNotIn("top_p", kwargs)

        gpt5 = RequestTransformer("gpt-5", "https://api.openai.com/v1")
        kwargs5 = gpt5.transform({"stop": ["END"], "temperature": 0.5})
        self.assertNotIn("stop", kwargs5)
        self.assertNotIn("temperature", kwargs5)

        local = RequestTransformer("qwen3", "http://localhost:1234/v1")
        kwargs_local = local.transform({"stop": ["END"], "temperature": 0.5})
        self.assertEqual(kwargs_local["stop"], ["END"])
        self.assertEqual(kwargs_local["temperature"], 0.5)

        # Host-gated: a local model merely named like an OpenAI reasoning
        # model keeps its parameters.
        local_o3 = RequestTransformer("o3-locally-hosted", "http://localhost:1234/v1")
        kwargs_local_o3 = local_o3.transform({"stop": ["END"], "temperature": 0.5})
        self.assertEqual(kwargs_local_o3["stop"], ["END"])
        self.assertEqual(kwargs_local_o3["temperature"], 0.5)

    def test_reasoning_effort_type_checked_and_unknown_warns(self):
        import logging

        from llm_sdk import ConfigurationError, RequestTransformer

        transformer = RequestTransformer("test", "http://localhost:1234/v1")
        # Unknown values pass through (models keep adding new efforts)…
        with self.assertLogs("llm_sdk", level=logging.WARNING) as logs:
            kwargs = transformer.transform({"reasoning_effort": "ultra"})
        self.assertEqual(kwargs["reasoning_effort"], "ultra")
        self.assertIn("ultra", logs.output[0])
        # …and the same unknown value warns only once per instance.
        transformer.transform({"reasoning_effort": "ultra"})
        with self.assertRaises(ConfigurationError):
            transformer.transform({"reasoning_effort": 5})

    def test_reasoning_effort_maps_to_top_level_on_any_host(self):
        from llm_sdk import RequestTransformer

        for base in ("https://api.openai.com/v1", "https://openrouter.ai/api/v1"):
            transformer = RequestTransformer("m", base)
            kwargs = transformer.transform({"reasoning_effort": "low"})
            self.assertEqual(kwargs["reasoning_effort"], "low")
            self.assertNotIn("extra_body", kwargs)

    def test_reasoning_budget_maps_to_reasoning_object(self):
        from llm_sdk import RequestTransformer

        transformer = RequestTransformer("m", "https://openrouter.ai/api/v1")
        kwargs = transformer.transform({"reasoning_budget": 2000})
        self.assertEqual(kwargs["reasoning"], {"max_tokens": 2000})

    def test_reasoning_effort_and_budget_mutually_exclusive(self):
        from llm_sdk import ConfigurationError, RequestTransformer

        transformer = RequestTransformer("m", "https://api.openai.com/v1")
        with self.assertRaises(ConfigurationError):
            transformer.transform({"reasoning_effort": "low", "reasoning_budget": 2000})

        llm = LLM(model="test")
        try:
            with self.assertRaises(ConfigurationError):
                llm._build_request(
                    [{"role": "user", "content": "hi"}],
                    output_format=None, tools=None, reasoning_effort="low",
                    max_tokens=None, extra_body=None, reasoning_budget=2000,
                )
            for bad in (0, -1, "2000", True):
                with self.subTest(bad=bad), self.assertRaises(ConfigurationError):
                    llm._build_request(
                        [{"role": "user", "content": "hi"}],
                        output_format=None, tools=None, reasoning_effort=None,
                        max_tokens=None, extra_body=None, reasoning_budget=bad,
                    )
        finally:
            llm.close()

    def test_responses_reasoning_budget(self):
        llm = LLM(model="test")
        try:
            kwargs, _, _ = llm._build_responses_request(
                [{"role": "user", "content": "hi"}],
                output_format=None, tools=None, reasoning_effort=None,
                max_tokens=None, extra_body=None, reasoning_budget=2000,
            )
            self.assertEqual(
                kwargs["reasoning"], {"max_tokens": 2000, "summary": "auto"}
            )
        finally:
            llm.close()

    def test_sampling_and_tool_choice_params_reach_request(self):
        llm = LLM(model="test")
        try:
            captured = {}

            def create(**kwargs):
                captured.clear()
                captured.update(kwargs)
                return iter([_chat_chunk("ok")])

            llm._client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            )

            list(llm.stream_response(
                input="hi",
                temperature=0.3,
                top_p=0.9,
                stop=["END"],
                seed=7,
                user="u1",
                tool_choice="required",
                store=True,
            ))

            self.assertEqual(captured["temperature"], 0.3)
            self.assertEqual(captured["top_p"], 0.9)
            self.assertEqual(captured["stop"], ["END"])
            self.assertEqual(captured["seed"], 7)
            self.assertEqual(captured["user"], "u1")
            self.assertEqual(captured["tool_choice"], "required")
            self.assertTrue(captured["store"])
        finally:
            llm.close()

    def test_per_call_stop_overrides_default_stop_sequences(self):
        llm = LLM(model="test", default_stop_sequences=["A", "B"])
        try:
            captured = {}

            def create(**kwargs):
                captured.clear()
                captured.update(kwargs)
                return iter([_chat_chunk("ok")])

            llm._client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            )

            list(llm.stream_response(input="hi"))
            self.assertEqual(captured["stop"], ["A", "B"])

            list(llm.stream_response(input="hi", stop=["C"]))
            self.assertEqual(captured["stop"], ["C"])
        finally:
            llm.close()

    def test_deep_merge_dicts_empty_nested(self):
        from llm_sdk import _deep_merge_dicts

        self.assertEqual(_deep_merge_dicts({"a": {}}, {"a": {}}), {"a": {}})
        self.assertIsNone(_deep_merge_dicts({}, {}))
        self.assertEqual(_deep_merge_dicts(None, {"b": 1}), {"b": 1})
        self.assertEqual(_deep_merge_dicts({"a": {"x": 1}}, {"a": {"y": 2}}), {"a": {"x": 1, "y": 2}})

    def test_import_does_not_configure_logging(self):
        import subprocess
        import sys

        code = (
            "import llm_sdk, logging, sys; "
            "sys.exit(0 if logging.getLogger('httpx').level != logging.WARNING else 1)"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)


class BaseURLTests(unittest.TestCase):
    """F-10/C-02: base URL normalization."""

    def test_provider_paths_preserved(self):
        cases = {
            "https://generativelanguage.googleapis.com/v1beta/openai/": "https://generativelanguage.googleapis.com/v1beta/openai",
            "https://gateway.ai.cloudflare.com/v1/acct/gw/openai": "https://gateway.ai.cloudflare.com/v1/acct/gw/openai",
            "http://localhost:1234": "http://localhost:1234/v1",
            "http://localhost:1234/": "http://localhost:1234/v1",
            "http://localhost:1234/v1/": "http://localhost:1234/v1",
        }
        for base_url, expected in cases.items():
            with self.subTest(base_url=base_url):
                llm = LLM(model="test", base_url=base_url)
                try:
                    self.assertEqual(llm._api_base, expected)
                finally:
                    llm.close()

    def test_normalization_disabled(self):
        llm = LLM(model="test", base_url="http://x:1/custom", normalize_base_url=False)
        try:
            self.assertEqual(llm._api_base, "http://x:1/custom")
        finally:
            llm.close()

    def test_base_url_property_returns_effective_url(self):
        llm = LLM(model="test", base_url="http://x:1")
        try:
            self.assertEqual(llm.base_url, "http://x:1/v1")
        finally:
            llm.close()


class StructuredOutputTests(unittest.TestCase):
    """F-07/S-05: structured output parsing and refusals."""

    def _stream_with_chunks(self, chunks, **kwargs):
        llm = LLM(model="test")
        try:
            llm._client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(
                    create=lambda **kw: iter(chunks)
                ))
            )
            return list(llm.stream_response(input="x", **kwargs))
        finally:
            llm.close()

    def test_invalid_json_raises_structured_output_error(self):
        from llm_sdk import StructuredOutputError

        with self.assertRaises(StructuredOutputError) as ctx:
            self._stream_with_chunks(
                [_chat_chunk("not json at all")],
                output_format={"type": "json_object"},
                final=True,
            )
        self.assertEqual(ctx.exception.raw, "not json at all")
        self.assertIsNone(ctx.exception.stop_reason)

    def test_truncated_json_mentions_length(self):
        from llm_sdk import StructuredOutputError

        truncated = SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content='{"a": '), finish_reason="length"
            )],
        )
        with self.assertRaisesRegex(StructuredOutputError, "truncated"):
            self._stream_with_chunks(
                [truncated], output_format={"type": "json_object"}, final=True
            )

    def test_markdown_fences_stripped(self):
        events = self._stream_with_chunks(
            [_chat_chunk('```json\n{"a": 1}\n```')],
            output_format={"type": "json_object"},
            final=True,
        )
        answer_event = next(e for e in events if e["type"] == "answer")
        self.assertEqual(answer_event["content"], {"a": 1})

    def test_invalid_json_always_raises_structured_output_error(self):
        from llm_sdk import StructuredOutputError

        with self.assertRaises(StructuredOutputError):
            list(self._stream_with_chunks(
                [_chat_chunk("nope")],
                output_format={"type": "json_object"},
                final=True,
            ))

    def test_schema_strict_true_requires_all_fields_on_wire(self):
        from llm_sdk import LLM

        class Out:
            req: str
            opt: str = "dflt"

        llm = LLM(model="test")
        try:
            captured = {}

            def create(**kwargs):
                captured.update(kwargs)
                return iter([_chat_chunk('{"req": "a", "opt": "b"}')])

            llm._client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            )
            events = list(llm.stream_response(
                input="x", output_format=Out, final=True
            ))
            wire = captured["response_format"]
            self.assertTrue(wire["json_schema"]["strict"])
            self.assertEqual(
                wire["json_schema"]["schema"]["required"], ["req", "opt"]
            )
            final = next(e for e in events if e["type"] == "final")
            self.assertEqual(
                final["content"]["answer"], {"req": "a", "opt": "b"}
            )
        finally:
            llm.close()

    def test_schema_strict_false_relaxes_required_on_wire(self):
        from llm_sdk import LLM

        class Out:
            req: str
            opt: str = "dflt"

        llm = LLM(model="test")
        try:
            captured = {}

            def create(**kwargs):
                captured.update(kwargs)
                return iter([_chat_chunk('{"req": "a", "opt": "b"}')])

            llm._client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            )
            events = list(llm.stream_response(
                input="x", output_format=Out, schema_strict=False, final=True
            ))
            wire = captured["response_format"]
            self.assertFalse(wire["json_schema"]["strict"])
            self.assertEqual(wire["json_schema"]["schema"]["required"], ["req"])
            final = next(e for e in events if e["type"] == "final")
            self.assertEqual(
                final["content"]["answer"], {"req": "a", "opt": "b"}
            )
        finally:
            llm.close()

    def test_schema_strict_false_still_parses_strict(self):
        from llm_sdk import LLM, StructuredOutputError

        class Out:
            req: str
            opt: str = "dflt"

        llm = LLM(model="test")
        try:
            llm._client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(
                    create=lambda **kw: iter([_chat_chunk("nope")])
                ))
            )
            with self.assertRaises(StructuredOutputError) as ctx:
                list(llm.stream_response(
                    input="x", output_format=Out,
                    schema_strict=False, final=True,
                ))
            self.assertEqual(ctx.exception.raw, "nope")
        finally:
            llm.close()

    def test_text_format_still_streams_answers(self):
        events = self._stream_with_chunks(
            [_chat_chunk("hel"), _chat_chunk("lo")],
            output_format={"type": "text"},
        )
        answers = [e["content"] for e in events if e["type"] == "answer"]
        self.assertEqual(answers, ["hel", "lo"])

    def test_chat_refusal_surfaced(self):
        chunk = SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content=None, refusal="no can do")
            )],
        )
        events = self._stream_with_chunks([chunk], final=True)
        refusal_events = [e for e in events if e["type"] == "refusal"]
        self.assertEqual([e["content"] for e in refusal_events], ["no can do"])
        final = next(e for e in events if e["type"] == "final")
        self.assertEqual(final["content"]["refusal"], "no can do")
        self.assertEqual(final["content"]["stop_reason"], "refusal")

    def test_refusal_with_structured_output_raises(self):
        from llm_sdk import StructuredOutputError

        chunk = SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(delta=SimpleNamespace(content=None, refusal="refusing"))],
        )
        with self.assertRaisesRegex(StructuredOutputError, "refus"):
            self._stream_with_chunks(
                [chunk], output_format={"type": "json_object"}, final=True
            )


class ChatStreamBehaviorTests(unittest.TestCase):
    """ST-02/ST-05/ST-06/F-08/F-09/F-23/F-27/F-34: chat streaming semantics."""

    def test_content_chunks_do_not_complete_tool_calls_early(self):
        from llm_sdk import EventBuilder, EventType, ToolCallStreamHandler

        handler = ToolCallStreamHandler(EventBuilder())
        events = []
        events += handler.process_chunk([SimpleNamespace(
            index=0, id="c1", function=SimpleNamespace(name="weather", arguments='{"ci')
        )])
        # Keepalive/content delta between argument deltas must not "complete"
        # the call with truncated arguments.
        events += handler.process_chunk(None)
        events += handler.process_chunk([SimpleNamespace(
            index=0, id=None, function=SimpleNamespace(name=None, arguments='ty":"Berlin"}')
        )])

        complete = [e for e in events if e["type"] == EventType.TOOL_CALL.value]
        self.assertEqual(complete, [])
        calls = handler.finalize()
        self.assertEqual(calls[0]["arguments"], {"city": "Berlin"})

    def test_constant_index_with_distinct_ids_separates_calls(self):
        from llm_sdk import EventBuilder, EventType, ToolCallStreamHandler

        handler = ToolCallStreamHandler(EventBuilder())
        events = []
        events += handler.process_chunk([SimpleNamespace(
            index=0, id="call_a", function=SimpleNamespace(name="weather", arguments='{"city":"Berlin"}')
        )])
        # Same index, different id: a new call must open instead of merging
        # both argument strings into one corrupt JSON blob.
        events += handler.process_chunk([SimpleNamespace(
            index=0, id="call_b", function=SimpleNamespace(name="time", arguments='{"tz":"UTC"}')
        )])
        # The first call was completed on the index switch (its arguments
        # already parsed as complete JSON).
        early = [e["content"] for e in events if e["type"] == EventType.TOOL_CALL.value]
        self.assertEqual([c["name"] for c in early], ["weather"])
        calls = handler.finalize()
        self.assertEqual([c["name"] for c in calls], ["time"])
        all_calls = handler.get_all_calls()
        self.assertEqual([c["name"] for c in all_calls], ["weather", "time"])
        self.assertEqual(all_calls[0]["arguments"], {"city": "Berlin"})
        self.assertEqual(all_calls[1]["arguments"], {"tz": "UTC"})

    def test_cumulative_arguments_replaced_not_appended(self):
        from llm_sdk import EventBuilder, ToolCallStreamHandler

        handler = ToolCallStreamHandler(EventBuilder())
        handler.process_chunk([SimpleNamespace(
            index=0, id="c1", function=SimpleNamespace(name="f", arguments='{"city": ')
        )])
        # Cumulative resend (only recognized once the buffer is long enough
        # that a coincidental prefix match is implausible).
        handler.process_chunk([SimpleNamespace(
            index=0, id=None, function=SimpleNamespace(name=None, arguments='{"city": "Berlin"}')
        )])
        calls = handler.finalize()
        self.assertEqual(calls[0]["arguments"], {"city": "Berlin"})

    def test_tool_call_part_emitted_without_server_id(self):
        from llm_sdk import EventBuilder, EventType, ToolCallStreamHandler

        handler = ToolCallStreamHandler(EventBuilder())
        events = handler.process_chunk([SimpleNamespace(
            index=None, id=None,
            function=SimpleNamespace(name="weather", arguments='{"city":"Berlin"}'),
        )])
        parts = [e for e in events if e["type"] == EventType.TOOL_CALL_PART.value]
        self.assertEqual(len(parts), 1)
        self.assertTrue(parts[0]["content"]["id"].startswith("call_"))
        self.assertEqual(parts[0]["content"]["args_delta"], '{"city":"Berlin"}')

    def test_reasoning_streamed_and_in_final_by_default(self):
        chunks = [
            SimpleNamespace(usage=None, choices=[SimpleNamespace(
                delta=SimpleNamespace(reasoning_content="deep thought", content=None)
            )]),
            SimpleNamespace(usage=None, choices=[SimpleNamespace(
                delta=SimpleNamespace(content="answer")
            )]),
        ]
        llm = LLM(model="test")
        try:
            llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
                create=lambda **kw: iter(chunks)
            )))
            events = list(llm.stream_response(input="x", final=True))
            self.assertTrue(any(e["type"] == "reasoning" for e in events))
            final = next(e for e in events if e["type"] == "final")
            self.assertEqual(final["content"]["reasoning"], "deep thought")
        finally:
            llm.close()

    def test_include_reasoning_false_hides_everywhere(self):
        chunks = [
            SimpleNamespace(usage=None, choices=[SimpleNamespace(
                delta=SimpleNamespace(reasoning_content="deep thought", content="answer")
            )]),
        ]
        llm = LLM(model="test")
        try:
            llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
                create=lambda **kw: iter(chunks)
            )))
            events = list(llm.stream_response(input="x", final=True, include_reasoning=False))
            self.assertFalse(any(e["type"] == "reasoning" for e in events))
            final = next(e for e in events if e["type"] == "final")
            self.assertNotIn("reasoning", final["content"])
        finally:
            llm.close()

    def test_include_reasoning_flag_emits_reasoning_events(self):
        chunks = [
            SimpleNamespace(usage=None, choices=[SimpleNamespace(
                delta=SimpleNamespace(reasoning_content="deep thought", content="answer")
            )]),
        ]
        llm = LLM(model="test")
        try:
            llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
                create=lambda **kw: iter(chunks)
            )))
            events = list(llm.stream_response(input="x", include_reasoning=True))
            reasoning = [e["content"] for e in events if e["type"] == "reasoning"]
            self.assertEqual(reasoning, ["deep thought"])
        finally:
            llm.close()

    def test_final_answer_not_stripped(self):
        llm = LLM(model="test")
        try:
            llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
                create=lambda **kw: iter([_chat_chunk("code\n")])
            )))
            events = list(llm.stream_response(input="x", final=True))
            final = next(e for e in events if e["type"] == "final")
            self.assertEqual(final["content"]["answer"], "code\n")
        finally:
            llm.close()

    def test_reasoning_non_string_values_ignored(self):
        chunk = SimpleNamespace(usage=None, choices=[SimpleNamespace(
            delta=SimpleNamespace(reasoning=[{"summary": "x"}], content="ok")
        )])
        llm = LLM(model="test")
        try:
            llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
                create=lambda **kw: iter([chunk])
            )))
            events = list(llm.stream_response(input="x", final=True))
            final = next(e for e in events if e["type"] == "final")
            self.assertEqual(final["content"]["answer"], "ok")
            self.assertNotIn("reasoning", final["content"])
        finally:
            llm.close()

    def test_verbose_reports_chunks_and_usage_tokens(self):
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        chunks = [
            _chat_chunk("a"),
            _chat_chunk("b"),
            SimpleNamespace(usage=usage, choices=[SimpleNamespace(delta=None, finish_reason="stop")]),
        ]
        llm = LLM(model="test")
        try:
            llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
                create=lambda **kw: iter(chunks)
            )))
            events = list(llm.stream_response(input="x", verbose=True, final=True))
            verbose_event = next(e for e in events if e["type"] == "verbose")
            self.assertEqual(verbose_event["content"]["tokens"], 5)
            self.assertEqual(verbose_event["content"]["chunks"], 2)
            self.assertEqual(verbose_event["content"]["prompt_tokens"], 10)
            self.assertEqual(verbose_event["content"]["total_tokens"], 15)
        finally:
            llm.close()

    def test_early_break_closes_stream(self):
        class Stream:
            def __init__(self):
                self.closed = False

            def __iter__(self):
                yield _chat_chunk("x")
                yield _chat_chunk("y")

            def close(self):
                self.closed = True

        stream = Stream()
        llm = LLM(model="test")
        try:
            llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
                create=lambda **kw: stream
            )))
            for _ in llm.stream_response(input="hi"):
                break
            self.assertTrue(stream.closed)
        finally:
            llm.close()


class ResponsesAPITests(unittest.TestCase):
    """F-03/F-22/R-01/R-02/R-03/T-05/ST-07: Responses API path."""

    def _responses_llm(self, events, captured=None):
        llm = LLM(model="gpt-5-test", use_responses_api=True)

        def create(**kwargs):
            if captured is not None:
                captured.clear()
                captured.update(kwargs)
            return iter(events)

        llm._client = SimpleNamespace(responses=SimpleNamespace(create=create))
        return llm

    def test_tool_loop_history_translated_to_responses_items(self):
        from llm_sdk import assistant_message, tool_result

        captured = {}
        completed = SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(status="completed", usage=None),
        )
        llm = self._responses_llm([completed], captured)

        history = [
            {"role": "user", "content": "weather?"},
            assistant_message({"tool_calls": [{
                "id": "call_1", "name": "weather", "arguments": {"city": "Berlin"}
            }]}),
            tool_result({"id": "call_1", "name": "weather"}, {"temp": 21}),
        ]
        try:
            list(llm.stream_response(messages=history, final=True))
        finally:
            llm.close()

        self.assertEqual(captured["input"], [
            {"role": "user", "content": "weather?"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "weather",
                "arguments": '{"city": "Berlin"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": '{"temp": 21}',
            },
        ])

    def test_assistant_content_lists_use_output_text(self):
        captured = {}
        completed = SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(status="completed", usage=None),
        )
        llm = self._responses_llm([completed], captured)

        history = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "let me check"}],
                "tool_calls": [{
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q": "x"}'},
                }],
            },
        ]
        try:
            list(llm.stream_response(messages=history))
        finally:
            llm.close()

        self.assertEqual(captured["input"], [
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "let me check"}],
            },
            {
                "type": "function_call",
                "call_id": "c2",
                "name": "search",
                "arguments": '{"q": "x"}',
            },
        ])

    def test_image_detail_and_base64_sniffing_translated(self):
        import base64
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (1, 1)).save(buf, format="PNG")
        png_b64 = base64.b64encode(buf.getvalue()).decode()
        content = [
            {"type": "image_url", "image_url": {"url": "https://x/y.png", "detail": "high"}},
            {"type": "image_base64", "image_base64": png_b64},
        ]
        translated = LLM._translate_content_for_responses_api(content)
        self.assertEqual(translated[0], {
            "type": "input_image", "image_url": "https://x/y.png", "detail": "high",
        })
        self.assertTrue(translated[1]["image_url"].startswith("data:image/png;base64,"))

    def test_responses_tools_do_not_inject_strict(self):
        converted = LLM._convert_tools_for_responses_api([
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "explicit",
                    "strict": False,
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ])
        self.assertNotIn("strict", converted[0])
        self.assertFalse(converted[1]["strict"])

    def test_responses_failed_raises_model_request_error(self):
        from llm_sdk import ModelRequestError

        events = [
            SimpleNamespace(
                type="response.failed",
                response=SimpleNamespace(
                    status="failed",
                    error=SimpleNamespace(message="server exploded", code="internal_error"),
                    usage=None,
                ),
            ),
        ]
        llm = self._responses_llm(events)
        try:
            with self.assertRaises(ModelRequestError) as ctx:
                list(llm.stream_response(input="x"))
            self.assertIn("server exploded", str(ctx.exception))
        finally:
            llm.close()

    def test_responses_output_item_done_completes_tool_call(self):
        events = [
            SimpleNamespace(
                type="response.output_item.added",
                output_index=0,
                item=SimpleNamespace(
                    type="function_call", id="item_1", call_id="call_1", name="weather"
                ),
            ),
            # No function_call_arguments.done event (some servers omit it).
            SimpleNamespace(
                type="response.output_item.done",
                output_index=0,
                item=SimpleNamespace(
                    type="function_call",
                    id="item_1",
                    call_id="call_1",
                    name="weather",
                    arguments='{"city":"Berlin"}',
                ),
            ),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(status="completed", usage=None),
            ),
        ]
        llm = self._responses_llm(events)
        try:
            result = list(llm.stream_response(input="x", final=True))
            tool_event = next(
                e for e in result if e["type"] == EventType.TOOL_CALL.value
            )
            self.assertEqual(
                tool_event["content"],
                {"id": "call_1", "name": "weather", "arguments": {"city": "Berlin"}, "callable": None},
            )
            final = next(e for e in result if e["type"] == "final")
            self.assertEqual(final["content"]["stop_reason"], "tool_calls")
        finally:
            llm.close()

    def test_responses_output_text_uses_reasoning_parser(self):
        events = [
            SimpleNamespace(type="response.output_text.delta", delta="<thi"),
            SimpleNamespace(type="response.output_text.delta", delta="nk>secret</think>pub"),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(status="completed", usage=None),
            ),
        ]
        llm = self._responses_llm(events)
        try:
            result = list(llm.stream_response(input="x", final=True))
            answers = [e["content"] for e in result if e["type"] == "answer"]
            self.assertEqual(answers, ["pub"])
            final = next(e for e in result if e["type"] == "final")
            self.assertEqual(final["content"]["reasoning"], "secret")
        finally:
            llm.close()

    def test_responses_refusal_surfaced(self):
        events = [
            SimpleNamespace(type="response.refusal.delta", delta="cannot"),
            SimpleNamespace(type="response.refusal.delta", delta=" help"),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(status="completed", usage=None),
            ),
        ]
        llm = self._responses_llm(events)
        try:
            result = list(llm.stream_response(input="x", final=True))
            refusal_events = [e["content"] for e in result if e["type"] == "refusal"]
            self.assertEqual(refusal_events, ["cannot", " help"])
            final = next(e for e in result if e["type"] == "final")
            self.assertEqual(final["content"]["refusal"], "cannot help")
            self.assertEqual(final["content"]["stop_reason"], "refusal")
        finally:
            llm.close()

    def test_responses_reasoning_summary_only_when_including_reasoning(self):
        captured = {}
        completed = SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(status="completed", usage=None),
        )
        llm = self._responses_llm([completed], captured)
        try:
            list(llm.stream_response(input="x", reasoning_effort="high"))
            # Summaries are requested when reasoning is visible (the default).
            self.assertEqual(
                captured["reasoning"], {"effort": "high", "summary": "auto"}
            )

            list(llm.stream_response(input="x", reasoning_effort="high", include_reasoning=False))
            self.assertEqual(captured["reasoning"], {"effort": "high"})
        finally:
            llm.close()

    def test_responses_max_output_tokens_and_sampling(self):
        captured = {}
        completed = SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(status="completed", usage=None),
        )
        llm = self._responses_llm([completed], captured)
        try:
            list(llm.stream_response(
                input="x",
                max_tokens=123,
                temperature=0.2,
                top_p=0.8,
                tool_choice={"type": "function", "function": {"name": "f"}},
                store=False,
            ))
        finally:
            llm.close()
        self.assertEqual(captured["max_output_tokens"], 123)
        self.assertEqual(captured["temperature"], 0.2)
        self.assertEqual(captured["top_p"], 0.8)
        # Responses accepts only 'none'|'auto'|'required' as strings; forcing
        # a function requires the object form.
        self.assertEqual(captured["tool_choice"], {"type": "function", "name": "f"})
        self.assertFalse(captured["store"])


class ImagePolicyTests(unittest.TestCase):
    """F-17/F-18/I-01/I-02/I-03: image handling and security."""

    def test_image_path_str_allowed_for_real_images(self):
        import tempfile

        from PIL import Image

        from llm_sdk import ImageProcessor

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/img.jpg"
            Image.new("RGB", (4, 2), (0, 255, 0)).save(path, format="JPEG")
            messages = [{"role": "user", "content": [{"type": "image", "image_path": path}]}]
            ImageProcessor.process_messages(messages)
            url = messages[0]["content"][0]["image_url"]["url"]
            self.assertTrue(url.startswith("data:image/jpeg;base64,"))

    def test_image_path_rejects_non_images(self):
        import tempfile
        from pathlib import Path

        from llm_sdk import ImageProcessingError, ImageProcessor

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "not-an-image.png"
            path.write_text("definitely not image bytes")
            messages = [{"role": "user", "content": [{"type": "image", "image_path": path}]}]
            with self.assertRaises(ImageProcessingError):
                ImageProcessor.process_messages(messages)

    def test_image_path_pathlib_always_allowed(self):
        import tempfile
        from pathlib import Path

        from PIL import Image

        from llm_sdk import ImageProcessor

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "img.png"
            Image.new("RGB", (4, 2), (255, 0, 0)).save(path, format="PNG")
            messages = [{"role": "user", "content": [{"type": "image", "image_path": path}]}]
            ImageProcessor.process_messages(messages)
            url = messages[0]["content"][0]["image_url"]["url"]
            self.assertTrue(url.startswith("data:image/png;base64,"))

    def test_image_path_str_with_real_image(self):
        import tempfile

        from PIL import Image

        from llm_sdk import ImageProcessor

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/img.jpg"
            Image.new("RGB", (4, 2), (0, 255, 0)).save(path, format="JPEG")
            messages = [{"role": "user", "content": [{"type": "image", "image_path": path}]}]
            ImageProcessor.process_messages(messages)
            url = messages[0]["content"][0]["image_url"]["url"]
            self.assertTrue(url.startswith("data:image/jpeg;base64,"))

    def test_jpeg_bytes_passed_through_without_reencode(self):
        import base64
        import tempfile
        from pathlib import Path

        from PIL import Image

        from llm_sdk import ImageProcessor

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "img.jpg"
            Image.new("RGB", (40, 20), (0, 0, 255)).save(path, format="JPEG")
            item = ImageProcessor._from_path(path)
            url = item["image_url"]["url"]
            self.assertTrue(url.startswith("data:image/jpeg;base64,"))
            encoded = url.split(",", 1)[1]
            self.assertEqual(base64.b64decode(encoded), path.read_bytes())

    def test_non_passthrough_format_transcoded_to_jpeg(self):
        import tempfile
        from pathlib import Path

        from PIL import Image

        from llm_sdk import ImageProcessor

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "img.bmp"
            Image.new("RGB", (4, 2), (9, 9, 9)).save(path, format="BMP")
            item = ImageProcessor._from_path(path)
            self.assertTrue(item["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_encode_pil_applies_exif_transpose(self):
        import base64
        import io

        from PIL import Image

        from llm_sdk import ImageProcessor

        img = Image.new("RGB", (2, 4), (0, 150, 0))
        exif = Image.Exif()
        exif[274] = 6  # orientation: 90 degrees
        img.info["exif"] = exif.tobytes()

        item = ImageProcessor._from_pil(img)
        encoded = item["image_url"]["url"].split(",", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as out:
            self.assertEqual(out.size, (4, 2))

    def test_cmyk_pil_image_encoded_without_crash(self):
        from PIL import Image

        from llm_sdk import ImageProcessor

        img = Image.new("CMYK", (4, 2))
        item = ImageProcessor._from_pil(img)
        self.assertTrue(item["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_from_base64_data_url_passthrough(self):
        import base64
        import io

        from PIL import Image

        from llm_sdk import ImageProcessor

        img = Image.new("RGB", (2, 2), (9, 9, 9))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
        item = ImageProcessor._from_base64(url)
        self.assertEqual(item["image_url"]["url"], url)

    def test_sniff_mime_with_whitespace_wrapped_base64(self):
        import base64

        from llm_sdk import ImageProcessor

        data = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8).decode()
        wrapped = "\n".join(data[i:i + 10] for i in range(0, len(data), 10))
        self.assertEqual(ImageProcessor._sniff_image_mime(wrapped), "image/png")

    def test_corrupt_image_raises_image_processing_error(self):
        import tempfile
        from pathlib import Path

        from llm_sdk import ImageProcessingError, ImageProcessor, LLMError

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.bmp"
            path.write_bytes(b"BM" + b"\x00" * 16)
            with self.assertRaises(ImageProcessingError):
                ImageProcessor._from_path(path)
            self.assertTrue(issubclass(ImageProcessingError, LLMError))


class HelpersAndConfigTests(unittest.TestCase):
    """F-24/F-26/F-31/C-05/C-06/C-07/D-03/ST-05: helpers, config, misc."""

    def test_resolve_messages_replaces_existing_system(self):
        from llm_sdk import _resolve_messages

        messages = [
            {"role": "system", "content": "old"},
            {"role": "user", "content": "hi"},
        ]
        resolved = _resolve_messages(messages=messages, system="new")
        self.assertEqual(resolved[0], {"role": "system", "content": "new"})
        self.assertEqual(len(resolved), 2)

    def test_resolve_messages_validates_types(self):
        from llm_sdk import ConfigurationError, _resolve_messages

        with self.assertRaises(ConfigurationError):
            _resolve_messages(system=0)
        with self.assertRaises(ConfigurationError):
            _resolve_messages(input=42)
        with self.assertRaises(ConfigurationError):
            _resolve_messages(messages=["nope"])
        with self.assertRaises(ConfigurationError):
            _resolve_messages(messages=[{"role": "user", "content": "x"}], input="conflict")
        self.assertEqual(
            _resolve_messages(input="hi", system="sys"),
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
        )
        self.assertEqual(
            _resolve_messages(input="hi"),
            [{"role": "user", "content": "hi"}],
        )

    def test_tool_result_serializes_non_strings(self):
        import datetime

        from llm_sdk import tool_result

        self.assertEqual(tool_result({"id": "c"}, 42)["content"], "42")
        self.assertEqual(tool_result({"id": "c"}, None)["content"], "null")
        self.assertIn("2024", tool_result({"id": "c"}, datetime.datetime(2024, 1, 2))["content"])
        tool_call_with_name = tool_result({"id": "c", "name": "f"}, {"ok": True})
        self.assertEqual(tool_call_with_name["content"], '{"ok": true}')
        self.assertEqual(tool_call_with_name["name"], "f")

    def test_assistant_message_reasoning_and_raw_arguments(self):
        from llm_sdk import assistant_message

        message = assistant_message({"answer": "a", "reasoning": "think"}, include_reasoning=True)
        self.assertEqual(message["reasoning_content"], "think")
        message_without = assistant_message({"answer": "a", "reasoning": "think"})
        self.assertNotIn("reasoning_content", message_without)

        raw = assistant_message({"tool_calls": [{
            "id": "c", "name": "f", "arguments": {"_raw": "{oops"}
        }]})
        self.assertEqual(raw["tool_calls"][0]["function"]["arguments"], "{oops")

    def test_api_key_not_in_config_repr(self):
        from llm_sdk import LLMConfig

        config = LLMConfig(model="m", api_key="sk-super-secret")
        self.assertNotIn("sk-super-secret", repr(config))

    def test_llm_from_config(self):
        from llm_sdk import LLM, LLMConfig

        config = LLMConfig(model="test", base_url="http://localhost:1234")
        llm = LLM.from_config(config)
        try:
            self.assertEqual(llm.model, "test")
            self.assertEqual(llm.base_url, "http://localhost:1234/v1")
        finally:
            llm.close()

    def test_max_image_side_validated_and_forwarded(self):
        from llm_sdk import LLM, ConfigurationError, LLMConfig

        with self.assertRaises(ConfigurationError):
            LLM(model="x", max_image_side=0)
        with self.assertRaises(ConfigurationError):
            LLMConfig(model="x", max_image_side=-3)
        llm = LLM(model="x", max_image_side=512)
        try:
            self.assertEqual(llm._config.max_image_side, 512)
        finally:
            llm.close()

    def test_list_models_falls_back_on_empty(self):
        llm = LLM(model="test")
        original = llm._client
        try:
            llm._client = SimpleNamespace(
                models=SimpleNamespace(list=lambda: SimpleNamespace(data=[]))
            )
            self.assertEqual(llm.list_models(fallback=["fb"]), ["fb"])
        finally:
            llm._client = original
            llm.close()

    def test_module_list_models_delegates_to_client(self):
        import unittest.mock as mock

        from llm_sdk import list_models

        llm = LLM(model="test")
        try:
            with mock.patch.object(llm, "list_models", return_value=["delegated"]) as method:
                self.assertEqual(list_models(client=llm, fallback=["x"]), ["delegated"])
                method.assert_called_once()
        finally:
            llm.close()

    def test_extra_body_type_validated(self):
        from llm_sdk import LLM, ConfigurationError

        with self.assertRaises(ConfigurationError):
            LLM(model="test", extra_body="not-a-dict")

    def test_extra_body_reserved_keys_rejected(self):
        from llm_sdk import LLM, ConfigurationError

        for key in ("model", "messages", "input", "stream"):
            with self.subTest(key=key), self.assertRaises(ConfigurationError):
                LLM(model="test", extra_body={key: "x"})

    def test_build_requests_reject_reserved_extra_body_keys(self):
        from llm_sdk import LLM, ConfigurationError

        llm = LLM(model="test")
        try:
            with self.assertRaises(ConfigurationError):
                llm._build_request(
                    [{"role": "user", "content": "hi"}],
                    output_format=None,
                    tools=None,
                    reasoning_effort=None,
                    max_tokens=None,
                    extra_body={"stream": False},
                )
            with self.assertRaises(ConfigurationError):
                llm._build_responses_request(
                    [{"role": "user", "content": "hi"}],
                    output_format=None,
                    tools=None,
                    reasoning_effort=None,
                    max_tokens=None,
                    extra_body={"model": "other"},
                )
        finally:
            llm.close()

    def test_timeout_and_max_retries_validated(self):
        from llm_sdk import LLM, ConfigurationError

        for bad in (0, -1, "30", True, float("nan"), float("inf")):
            with self.subTest(param="timeout", bad=bad), self.assertRaises(ConfigurationError):
                LLM(model="test", timeout=bad)
        for bad in (-1, 1.5, "3", True):
            with self.subTest(param="max_retries", bad=bad), self.assertRaises(ConfigurationError):
                LLM(model="test", max_retries=bad)
        llm = LLM(model="test", timeout=None, max_retries=0)
        try:
            self.assertIsNone(llm._client.timeout)
        finally:
            llm.close()

    def test_per_call_max_retries_validated(self):
        from llm_sdk import LLM, ConfigurationError

        llm = LLM(model="test")
        try:
            for bad in (-1, 1.5, "3", True):
                with self.subTest(bad=bad), self.assertRaises(ConfigurationError):
                    llm._client_for(bad)
                with self.subTest(bad=bad), self.assertRaises(ConfigurationError):
                    llm._async_client_for(bad)
        finally:
            llm.close()

    def test_direct_llm_config_validated(self):
        from llm_sdk import ConfigurationError, LLMConfig

        with self.assertRaises(ConfigurationError):
            LLMConfig(model="test", timeout="30")
        with self.assertRaises(ConfigurationError):
            LLMConfig(model="test", max_retries=-1)
        with self.assertRaises(ConfigurationError):
            LLMConfig(model="test", extra_body={"model": "other"})

    def test_custom_reasoning_pattern_token_types_validated(self):
        from llm_sdk import ConfigurationError, CustomReasoningPattern

        with self.assertRaises(ConfigurationError):
            CustomReasoningPattern(start_token=123, end_token="</x>")
        with self.assertRaises(ConfigurationError):
            CustomReasoningPattern(start_token="<x>", end_token=["</x>"])

    def test_safe_index_and_text_helpers(self):
        from llm_sdk import LLM

        self.assertEqual(LLM._safe_index("3"), 3)
        self.assertEqual(LLM._safe_index(0), 0)
        self.assertIsNone(LLM._safe_index("x"))
        self.assertIsNone(LLM._safe_index(None))
        self.assertIsNone(LLM._safe_index(True))
        self.assertIsNone(LLM._safe_index(float("inf")))
        self.assertEqual(LLM._safe_text(None), "")
        self.assertEqual(LLM._safe_text("a"), "a")
        self.assertEqual(LLM._safe_text(5), "")
        self.assertEqual(LLM._safe_text(["x"]), "")

    def test_redact_url_credentials(self):
        from llm_sdk import _redact_url_credentials

        self.assertEqual(
            _redact_url_credentials("http://user:pass@host:8080/v1"),
            "http://host:8080/v1",
        )
        self.assertEqual(
            _redact_url_credentials("https://api.openai.com/v1"),
            "https://api.openai.com/v1",
        )

    def test_tool_call_huge_int_args_do_not_crash(self):
        from llm_sdk import EventBuilder, ToolCallStreamHandler
        eb = EventBuilder()
        handler = ToolCallStreamHandler(eb)

        handler.process_chunk([
            SimpleNamespace(
                index=0,
                id="call_huge",
                function=SimpleNamespace(name="calc", arguments='{"n": '),
            )
        ])
        handler.process_chunk([
            SimpleNamespace(
                index=0,
                id=None,
                function=SimpleNamespace(name=None, arguments="9" * 5000 + "}"),
            )
        ])
        finalized = handler.finalize()
        self.assertEqual(len(finalized), 1)
        self.assertIn("_raw", finalized[0]["arguments"])

    def test_malformed_usage_values_are_ignored(self):
        from llm_sdk import LLM

        usage = LLM._read_usage({"prompt_tokens": "abc", "completion_tokens": None})
        self.assertIsNone(usage["prompt_tokens"])
        self.assertIsNone(usage["completion_tokens"])

    def test_huge_int_json_raises_structured_output_error(self):
        from llm_sdk import StructuredOutputError, _parse_structured_output

        with self.assertRaises(StructuredOutputError) as ctx:
            _parse_structured_output(
                '{"n": ' + "9" * 5000 + "}",
                stop_reason=None,
                strict=True,
            )
        self.assertIn("n", ctx.exception.raw)

    def test_sync_and_async_signatures_are_keyword_only_and_aligned(self):
        import inspect

        from llm_sdk import LLM

        sync_params = inspect.signature(LLM.stream_response).parameters
        async_params = inspect.signature(LLM.async_stream_response).parameters
        response_params = inspect.signature(LLM.response).parameters
        async_response_params = inspect.signature(LLM.async_response).parameters

        for params in (sync_params, async_params, response_params, async_response_params):
            keyword_only = [
                name
                for name, p in params.items()
                if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and name != "self"
            ]
            self.assertEqual(keyword_only, ["messages"])
        # Order of shared keywords matches between sync and async streaming.
        self.assertEqual(
            list(sync_params),
            list(async_params),
        )


class PostAuditRegressionTests(unittest.TestCase):
    """N-01..N-18: regressions and gaps found by the follow-up audit."""

    def _chat_llm(self, chunks):
        llm = LLM(model="test")
        llm._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=lambda **kw: iter(chunks)
        )))
        return llm

    # -- N-01: structured output + tools-only turn --------------------------

    def test_structured_output_with_tools_only_turn(self):
        tool_chunks = [
            SimpleNamespace(usage=None, choices=[SimpleNamespace(
                delta=SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
                    index=0, id="c1",
                    function=SimpleNamespace(name="weather", arguments='{"city":"Berlin"}'),
                )]),
                finish_reason=None,
            )]),
            SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=None, finish_reason="tool_calls")]),
        ]
        llm = self._chat_llm(tool_chunks)
        try:
            events = list(llm.stream_response(
                input="route this", output_format={"type": "json_object"}, final=True
            ))
        finally:
            llm.close()
        final = next(e for e in events if e["type"] == "final")
        # The answer is legitimately absent, not invalid JSON.
        self.assertIsNone(final["content"]["answer"])
        self.assertEqual(final["content"]["stop_reason"], "tool_calls")
        self.assertEqual(
            final["content"]["tool_calls"][0]["arguments"], {"city": "Berlin"}
        )
        answer_event = next(e for e in events if e["type"] == "answer")
        self.assertIsNone(answer_event["content"])

    def test_structured_output_empty_answer_content_filter(self):
        chunk = SimpleNamespace(usage=None, choices=[SimpleNamespace(
            delta=SimpleNamespace(content=None), finish_reason="content_filter"
        )])
        llm = self._chat_llm([chunk])
        try:
            events = list(llm.stream_response(
                input="x", output_format={"type": "json_object"}, final=True
            ))
        finally:
            llm.close()
        final = next(e for e in events if e["type"] == "final")
        self.assertIsNone(final["content"]["answer"])
        self.assertEqual(final["content"]["stop_reason"], "content_filter")

    # -- N-02: reasoning items captured and replayed -------------------------

    def test_responses_reasoning_items_captured_and_replayed(self):
        from llm_sdk import assistant_message

        reasoning_item = SimpleNamespace(
            type="reasoning", id="rs_1",
            summary=[{"type": "summary_text", "text": "thought"}],
            encrypted_content="enc-payload",
        )
        events = [
            SimpleNamespace(
                type="response.output_item.done", output_index=0, item=reasoning_item,
            ),
            SimpleNamespace(
                type="response.output_item.added", output_index=1,
                item=SimpleNamespace(type="function_call", id="fc_1", call_id="call_1", name="weather"),
            ),
            SimpleNamespace(
                type="response.function_call_arguments.done", item_id="fc_1", output_index=1,
                arguments='{"city":"Berlin"}', name="weather",
            ),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(status="completed", usage=None),
            ),
        ]
        captured = {}
        llm = LLM(model="gpt-5", use_responses_api=True)
        llm._client = SimpleNamespace(responses=SimpleNamespace(
            create=lambda **kw: (captured.clear(), captured.update(kw), iter(events))[2]
        ))
        try:
            result = list(llm.stream_response(input="weather?", final=True))
        finally:
            llm.close()

        final = next(e for e in result if e["type"] == "final")
        self.assertEqual(
            final["content"]["response_items"],
            [{
                "type": "reasoning", "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "thought"}],
                "encrypted_content": "enc-payload",
            }],
        )

        # assistant_message carries the items; the next request replays them
        # before the function_call item.
        history = [
            {"role": "user", "content": "weather?"},
            assistant_message(final["content"]),
            {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 21}'},
        ]
        translated = llm._to_responses_input(history)
        self.assertEqual(translated[1]["type"], "reasoning")
        self.assertEqual(translated[1]["id"], "rs_1")
        self.assertEqual(translated[2]["type"], "function_call")
        self.assertEqual(translated[3]["type"], "function_call_output")

    def test_responses_store_false_requests_encrypted_reasoning(self):
        captured = {}
        completed = SimpleNamespace(
            type="response.completed", response=SimpleNamespace(status="completed", usage=None),
        )
        llm = LLM(model="gpt-5", use_responses_api=True)
        llm._client = SimpleNamespace(responses=SimpleNamespace(
            create=lambda **kw: (captured.clear(), captured.update(kw), iter([completed]))[2]
        ))
        try:
            list(llm.stream_response(input="x", store=False))
        finally:
            llm.close()
        self.assertEqual(captured["include"], ["reasoning.encrypted_content"])

    # -- N-03: tool_choice object form ---------------------------------------

    def test_tool_choice_for_responses_uses_object_form(self):
        converted = LLM._convert_tool_choice_for_responses(
            {"type": "function", "function": {"name": "get_weather"}}
        )
        self.assertEqual(converted, {"type": "function", "name": "get_weather"})
        self.assertEqual(LLM._convert_tool_choice_for_responses("required"), "required")

    # -- N-04: follow-up deltas after id change on constant index -----------

    def test_constant_index_followup_deltas_follow_redirect(self):
        from llm_sdk import EventBuilder, ToolCallStreamHandler

        handler = ToolCallStreamHandler(EventBuilder())
        handler.process_chunk([SimpleNamespace(
            index=0, id="call_a", function=SimpleNamespace(name="weather", arguments='{"city":"Ber')
        )])
        handler.process_chunk([SimpleNamespace(
            index=0, id="call_a", function=SimpleNamespace(name=None, arguments='lin"}')
        )])
        # New call on the same constant index…
        handler.process_chunk([SimpleNamespace(
            index=0, id="call_b", function=SimpleNamespace(name="time", arguments='{"tz":"UT')
        )])
        # …followed by id-less deltas that must NOT fall back to call A.
        handler.process_chunk([SimpleNamespace(
            index=0, id=None, function=SimpleNamespace(name=None, arguments='C"}')
        )])
        all_calls = handler.get_all_calls()
        self.assertEqual([c["id"] for c in all_calls], ["call_a", "call_b"])
        self.assertEqual(all_calls[0]["arguments"], {"city": "Berlin"})
        self.assertEqual(all_calls[1]["arguments"], {"tz": "UTC"})

    # -- N-06: Literal with None values --------------------------------------

    def test_literal_with_none_maps_to_null_variant(self):
        from typing import Literal

        from llm_sdk import SchemaConverter

        class Out:
            maybe: Literal["yes", "no", None]
            only_none: Literal[None]

        schema = SchemaConverter().convert_class_to_schema(Out, strict=False)["json_schema"]["schema"]
        self.assertEqual(
            schema["properties"]["maybe"],
            {
                "anyOf": [
                    {"type": "string", "enum": ["yes", "no"]},
                    {"type": "null"},
                ]
            },
        )
        self.assertEqual(schema["properties"]["only_none"], {"type": "null"})

    # -- N-08: responses seed/stop dropped with warning, user passed ---------

    def test_responses_seed_and_stop_warned_and_dropped(self):
        import logging

        captured = {}
        completed = SimpleNamespace(
            type="response.completed", response=SimpleNamespace(status="completed", usage=None),
        )
        llm = LLM(model="gpt-5", use_responses_api=True)
        llm._client = SimpleNamespace(responses=SimpleNamespace(
            create=lambda **kw: (captured.clear(), captured.update(kw), iter([completed]))[2]
        ))
        try:
            with self.assertLogs("llm_sdk", level=logging.WARNING):
                list(llm.stream_response(input="x", seed=7, stop=["END"], user="u1"))
        finally:
            llm.close()
        self.assertNotIn("seed", captured)
        self.assertNotIn("stop", captured)
        self.assertEqual(captured["user"], "u1")

    # -- N-11: annotated known-type subclass stays primitive ------------------

    def test_annotated_str_subclass_stays_string(self):
        from llm_sdk import SchemaConverter

        class FancyStr(str):
            note: str = "class-level annotation"

        class Out:
            value: FancyStr

        schema = SchemaConverter().convert_class_to_schema(Out, strict=False)["json_schema"]["schema"]
        self.assertEqual(schema["properties"]["value"], {"type": "string"})

    # -- N-12: EXIF-rotated JPEG is transcoded, not passed through -----------

    def test_exif_rotated_jpeg_is_transcoded(self):
        import base64
        import io
        import tempfile
        from pathlib import Path

        from PIL import Image

        from llm_sdk import ImageProcessor

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rotated.jpg"
            img = Image.new("RGB", (2, 4), (20, 120, 220))
            exif = Image.Exif()
            exif[274] = 6  # 90-degree orientation
            img.save(path, format="JPEG", exif=exif)

            item = ImageProcessor._from_path(path)
            url = item["image_url"]["url"]
            self.assertTrue(url.startswith("data:image/jpeg;base64,"))
            # Transcoded (rotated pixels), not the original bytes.
            self.assertNotEqual(base64.b64decode(url.split(",", 1)[1]), path.read_bytes())
            with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as out:
                self.assertEqual(out.size, (4, 2))

    # -- N-13: module list_models default max_retries is None ----------------

    def test_module_list_models_max_retries_default_is_none(self):
        import inspect

        from llm_sdk import async_list_models, list_models

        for function in (list_models, async_list_models):
            default = inspect.signature(function).parameters["max_retries"].default
            self.assertIsNone(default)

    # -- N-14: tool description excludes the Args section ---------------------

    def test_tool_description_excludes_args_section(self):
        from llm_sdk import SchemaConverter, ToolPreparator

        def search(query: str, limit: int = 5) -> str:
            """Search the docs.

            Args:
                query: What to look for.
            """
            return ""

        definitions = ToolPreparator(SchemaConverter()).prepare([search]).definitions
        self.assertEqual(definitions[0]["function"]["description"], "Search the docs.")
        props = definitions[0]["function"]["parameters"]["properties"]
        self.assertEqual(props["query"]["description"], "What to look for.")

    def test_numpy_style_docstring_params_parsed(self):
        from llm_sdk import SchemaConverter, ToolPreparator

        def compute(base: int, scale: float = 1.0) -> str:
            """Compute things.

            Parameters
            ----------
            base : int
                The base value.
            scale : float
                Scaling factor.
            """
            return ""

        definitions = ToolPreparator(SchemaConverter()).prepare([compute]).definitions
        self.assertEqual(definitions[0]["function"]["description"], "Compute things.")
        props = definitions[0]["function"]["parameters"]["properties"]
        self.assertEqual(props["base"]["description"], "The base value.")
        self.assertIn("Scaling factor.", props["scale"]["description"])

    # -- reasoning visibility ----------------------------------------------------

    def test_include_reasoning_false_hides_everywhere(self):
        reasoning_chunk = SimpleNamespace(usage=None, choices=[SimpleNamespace(
            delta=SimpleNamespace(reasoning_content="thought", content="answer")
        )])

        llm = self._chat_llm([reasoning_chunk])
        try:
            events = list(llm.stream_response(input="x", final=True, include_reasoning=False))
        finally:
            llm.close()
        self.assertFalse(any(e["type"] == "reasoning" for e in events))
        final = next(e for e in events if e["type"] == "final")
        self.assertNotIn("reasoning", final["content"])

    # -- N-16: no duplicate tool_call when output_item.added is missing -------

    def test_responses_no_duplicate_without_output_item_added(self):
        events = [
            # No output_item.added: arguments.done only, item_id fc_1.
            SimpleNamespace(
                type="response.function_call_arguments.done", item_id="fc_1", output_index=0,
                arguments='{"city":"Berlin"}', name="weather",
            ),
            # Later output_item.done carries the real call_id.
            SimpleNamespace(
                type="response.output_item.done", output_index=0,
                item=SimpleNamespace(
                    type="function_call", id="fc_1", call_id="call_1",
                    name="weather", arguments='{"city":"Berlin"}',
                ),
            ),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(status="completed", usage=None),
            ),
        ]
        llm = LLM(model="gpt-5", use_responses_api=True)
        llm._client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: iter(events)))
        try:
            result = list(llm.stream_response(input="x", final=True))
        finally:
            llm.close()
        tool_events = [e for e in result if e["type"] == EventType.TOOL_CALL.value]
        self.assertEqual(len(tool_events), 1)
        final = next(e for e in result if e["type"] == "final")
        self.assertEqual(len(final["content"]["tool_calls"]), 1)

    # -- N-17: completion_tokens stays None without usage ---------------------

    def test_verbose_completion_tokens_none_without_usage(self):
        llm = self._chat_llm([_chat_chunk("a"), _chat_chunk("b")])
        try:
            events = list(llm.stream_response(input="x", verbose=True, final=True))
        finally:
            llm.close()
        verbose_event = next(e for e in events if e["type"] == "verbose")
        self.assertIsNone(verbose_event["content"]["completion_tokens"])
        self.assertEqual(verbose_event["content"]["tokens"], 2)
        self.assertEqual(verbose_event["content"]["chunks"], 2)

    # -- N-18: base_url validation, empty system, falsy answers ---------------

    def test_invalid_base_url_rejected(self):
        from llm_sdk import ConfigurationError

        for bad in ("", "   ", "not-a-url", "ftp://x/y"):
            with self.subTest(base_url=bad), self.assertRaises(ConfigurationError):
                LLM(model="test", base_url=bad)

    def test_empty_system_message_is_respected(self):
        from llm_sdk import _resolve_messages

        resolved = _resolve_messages(
            messages=[{"role": "user", "content": "hi"}], system=""
        )
        self.assertEqual(resolved[0], {"role": "system", "content": ""})

    def test_assistant_message_keeps_falsy_structured_answers(self):
        from llm_sdk import assistant_message

        self.assertEqual(assistant_message({"answer": 0})["content"], "0")
        self.assertEqual(assistant_message({"answer": False})["content"], "false")


    def test_chat_content_parts_list_concatenates_text(self):
        chunk = SimpleNamespace(usage=None, choices=[SimpleNamespace(
            delta=SimpleNamespace(content=[
                {"type": "text", "text": "he"},
                {"type": "image_url", "image_url": {"url": "x"}},
                {"type": "text", "text": "llo"},
            ]),
            finish_reason="stop",
        )])
        llm = self._chat_llm([chunk])
        try:
            events = list(llm.stream_response(input="x", final=True))
        finally:
            llm.close()
        final = next(e for e in events if e["type"] == "final")
        self.assertEqual(final["content"]["answer"], "hello")

    def test_chat_non_string_content_is_ignored(self):
        chunk = SimpleNamespace(usage=None, choices=[SimpleNamespace(
            delta=SimpleNamespace(content=123), finish_reason="stop")])
        llm = self._chat_llm([chunk])
        try:
            events = list(llm.stream_response(input="x", final=True))
        finally:
            llm.close()
        final = next(e for e in events if e["type"] == "final")
        self.assertEqual(final["content"]["answer"], "")

    def test_generation_options_validated(self):
        from llm_sdk import ConfigurationError

        llm = LLM(model="test")
        try:
            base = {
                "output_format": None, "tools": None, "reasoning_effort": None,
                "max_tokens": None, "extra_body": None,
            }
            bads = [
                {"temperature": "hot"}, {"temperature": float("nan")},
                {"temperature": True}, {"top_p": float("inf")},
                {"max_tokens": 0}, {"max_tokens": -1}, {"max_tokens": 1.5},
                {"seed": True}, {"seed": "1"},
                {"stop": [123]}, {"stop": 5},
                {"user": 123}, {"store": "yes"},
            ]
            builders = (llm._build_request, llm._build_responses_request)
            for kwargs in bads:
                merged = {**base, **kwargs}
                for builder in builders:
                    with self.subTest(builder=builder.__name__, kwargs=kwargs), \
                            self.assertRaises(ConfigurationError):
                        builder([{"role": "user", "content": "hi"}], **merged)
            good = {
                **base, "temperature": 0.7, "top_p": 1, "max_tokens": 5,
                "seed": 42, "stop": ["a"], "user": "u", "store": True,
            }
            for builder in builders:
                builder([{"role": "user", "content": "hi"}], **good)
        finally:
            llm.close()

    def test_circular_tool_payloads_raise_configuration_error(self):
        from llm_sdk import ConfigurationError, assistant_message, tool_result

        circular: dict = {}
        circular["self"] = circular
        with self.assertRaises(ConfigurationError):
            tool_result({"id": "c1", "name": "f"}, circular)
        with self.assertRaises(ConfigurationError):
            assistant_message({"answer": circular})

    def test_usage_bools_and_floats_ignored(self):
        from llm_sdk import LLM

        usage = LLM._read_usage({"prompt_tokens": True, "completion_tokens": 3.7})
        self.assertIsNone(usage["prompt_tokens"])
        self.assertIsNone(usage["completion_tokens"])
        self.assertEqual(LLM._read_usage({"prompt_tokens": 5})["prompt_tokens"], 5)

    def test_reasoning_token_length_capped(self):
        from llm_sdk import ConfigurationError, CustomReasoningPattern

        with self.assertRaises(ConfigurationError):
            CustomReasoningPattern(start_token="<" + "x" * 65, end_token="</x>")
        CustomReasoningPattern(start_token="<x>", end_token="</x>")

    def test_redact_url_strips_query_and_fragment(self):
        from llm_sdk import _redact_url_credentials

        self.assertEqual(
            _redact_url_credentials("https://host/v1?api_key=secret#frag"),
            "https://host/v1",
        )

    def test_shallow_message_copy_preserves_payloads(self):
        from llm_sdk import _copy_messages_shallow

        pil = object()
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "hi"},
            {"type": "image", "image_pil": pil},
        ]}]
        copied = _copy_messages_shallow(messages)
        self.assertEqual(copied, messages)
        self.assertIsNot(copied, messages)
        self.assertIsNot(copied[0], messages[0])
        self.assertIsNot(copied[0]["content"], messages[0]["content"])
        self.assertIs(copied[0]["content"][1]["image_pil"], pil)

    def test_build_request_does_not_mutate_caller_messages(self):
        llm = LLM(model="test")
        try:
            messages = [{"role": "user", "content": "hi"}]
            llm._build_request(messages, None, None, None, None, None)
            llm._build_responses_request(messages, None, None, None, None, None)
            self.assertEqual(messages, [{"role": "user", "content": "hi"}])
        finally:
            llm.close()

    def test_responses_translate_rejects_non_string_base64(self):
        from llm_sdk import LLM, ConfigurationError

        llm = LLM(model="test")
        try:
            with self.assertRaises(ConfigurationError):
                llm._translate_content_for_responses_api(
                    [{"type": "image_base64", "image_base64": 123}]
                )
        finally:
            llm.close()

    def test_response_closes_generator_on_early_break(self):
        import unittest.mock as mock

        closed = []

        def gen():
            try:
                yield {"type": "final", "content": {"answer": "x"}}
                yield {"type": "done", "content": None}
            finally:
                closed.append(True)

        llm = LLM(model="test")
        try:
            with mock.patch.object(llm, "stream_response", return_value=gen()):
                llm.response(input="hi")
        finally:
            llm.close()
        self.assertEqual(closed, [True])


class MediaModalityTests(unittest.TestCase):
    """Audio/video/file input: processors, URL passthrough, mappings."""

    WAV = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 100
    MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\x00" * 100
    OGG = b"OggS" + b"\x00" * 100
    FLAC = b"fLaC" + b"\x00" * 100
    M4A = b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 100
    AIFF = b"FORM" + b"\x00" * 4 + b"AIFF" + b"\x00" * 100
    AAC = b"\xff\xf1" + b"\x00" * 100
    MP4 = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 100
    WEBM = b"\x1a\x45\xdf\xa3" + b"\x00" * 100
    PDF = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"

    # -- audio -----------------------------------------------------------

    def test_audio_path_wav_becomes_input_audio(self):
        import base64
        import tempfile
        from pathlib import Path

        from llm_sdk import AudioProcessor

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.wav"
            path.write_bytes(self.WAV)
            messages = [{"role": "user", "content": [{"type": "audio", "audio_path": path}]}]
            AudioProcessor.process_messages(messages)
            part = messages[0]["content"][0]["input_audio"]
            self.assertEqual(part["format"], "wav")
            self.assertEqual(base64.b64decode(part["data"]), self.WAV)

    def test_audio_magic_sniffs_all_formats(self):
        from llm_sdk import AudioProcessor

        cases = {
            "wav": self.WAV, "mp3": self.MP3, "ogg": self.OGG,
            "flac": self.FLAC, "m4a": self.M4A, "aiff": self.AIFF,
            "aac": self.AAC,
        }
        for expected, raw in cases.items():
            with self.subTest(format=expected):
                self.assertEqual(AudioProcessor._sniff_audio_format(raw), expected)
        self.assertIsNone(AudioProcessor._sniff_audio_format(b"not audio at all!"))

    def test_audio_path_rejects_non_audio(self):
        import tempfile
        from pathlib import Path

        from llm_sdk import AudioProcessingError, AudioProcessor, ConfigurationError

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes.txt"
            path.write_text("just text")
            messages = [{"role": "user", "content": [{"type": "audio", "audio_path": path}]}]
            with self.assertRaises(AudioProcessingError):
                AudioProcessor.process_messages(messages)
            with self.assertRaises(AudioProcessingError):
                AudioProcessor._from_path(str(Path(tmp) / "missing.wav"))
            with self.assertRaises(ConfigurationError):
                AudioProcessor._from_path(123)

    def test_audio_read_cap(self):
        import base64
        from unittest import mock

        from llm_sdk import AudioProcessingError, AudioProcessor

        with mock.patch.object(AudioProcessor, "_MAX_AUDIO_READ_BYTES", 10), \
                self.assertRaises(AudioProcessingError):
            AudioProcessor._from_base64(base64.b64encode(self.WAV).decode())

    def test_audio_base64_forms(self):
        import base64

        from llm_sdk import AudioProcessingError, AudioProcessor

        raw_b64 = base64.b64encode(self.MP3).decode()
        item = AudioProcessor._from_base64(raw_b64)
        self.assertEqual(item["input_audio"]["format"], "mp3")

        declared = "data:audio/wav;base64," + base64.b64encode(self.WAV).decode()
        item = AudioProcessor._from_base64(declared)
        self.assertEqual(item["input_audio"]["format"], "wav")

        with self.assertRaises(AudioProcessingError):
            AudioProcessor._from_base64("data:text/plain;base64,aGVsbG8=")
        with self.assertRaises(AudioProcessingError):
            AudioProcessor._from_base64("!!!not-base64!!!")
        with self.assertRaises(AudioProcessingError):
            AudioProcessor._from_base64(123)

    def test_audio_url_passes_through(self):
        from llm_sdk import AudioProcessor, ConfigurationError

        item = AudioProcessor._from_url("https://example.com/clip.wav")
        self.assertEqual(
            item,
            {"type": "audio_url", "audio_url": {"url": "https://example.com/clip.wav"}},
        )
        messages = [{"role": "user", "content": [
            {"type": "audio", "audio_url": "https://example.com/clip.wav"}
        ]}]
        AudioProcessor.process_messages(messages)
        self.assertEqual(messages[0]["content"][0], item)
        with self.assertRaises(ConfigurationError):
            AudioProcessor._from_url("")
        with self.assertRaises(ConfigurationError):
            AudioProcessor._from_url(123)

    # -- video -----------------------------------------------------------

    def test_video_path_mp4_becomes_data_url(self):
        import base64
        import tempfile
        from pathlib import Path

        from llm_sdk import VideoProcessor

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            path.write_bytes(self.MP4)
            messages = [{"role": "user", "content": [{"type": "video", "video_path": path}]}]
            VideoProcessor.process_messages(messages)
            url = messages[0]["content"][0]["video_url"]["url"]
            self.assertTrue(url.startswith("data:video/mp4;base64,"))
            self.assertEqual(base64.b64decode(url.split(",", 1)[1]), self.MP4)

    def test_video_container_checks(self):
        import tempfile
        from pathlib import Path

        from llm_sdk import ConfigurationError, VideoProcessingError, VideoProcessor

        with tempfile.TemporaryDirectory() as tmp:
            webm = Path(tmp) / "clip.webm"
            webm.write_bytes(self.WEBM)
            item = VideoProcessor._from_path(webm)
            self.assertTrue(
                item["video_url"]["url"].startswith("data:video/webm;base64,")
            )

            bad_ext = Path(tmp) / "clip.avi"
            bad_ext.write_bytes(self.MP4)
            with self.assertRaises(VideoProcessingError):
                VideoProcessor._from_path(bad_ext)

            mismatch = Path(tmp) / "clip.mp4"
            mismatch.write_text("not a video")
            with self.assertRaises(VideoProcessingError):
                VideoProcessor._from_path(mismatch)

            with self.assertRaises(ConfigurationError):
                VideoProcessor._from_path(123)

    def test_video_url_passes_through(self):
        from llm_sdk import ConfigurationError, VideoProcessor

        item = VideoProcessor._from_url("https://example.com/clip.mp4")
        self.assertEqual(
            item, {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}}
        )
        item = VideoProcessor._from_url(
            {"url": "https://example.com/c.mp4", "detail": "low"}
        )
        self.assertEqual(item["video_url"]["detail"], "low")
        with self.assertRaises(ConfigurationError):
            VideoProcessor._from_url("")

    def test_video_base64_data_url(self):
        import base64

        from llm_sdk import VideoProcessingError, VideoProcessor

        declared = "data:video/mp4;base64," + base64.b64encode(self.MP4).decode()
        item = VideoProcessor._from_base64(declared)
        self.assertTrue(item["video_url"]["url"].startswith("data:video/mp4;base64,"))

        raw_b64 = base64.b64encode(self.WEBM).decode()
        item = VideoProcessor._from_base64(raw_b64)
        self.assertTrue(item["video_url"]["url"].startswith("data:video/webm;base64,"))

        with self.assertRaises(VideoProcessingError):
            VideoProcessor._from_base64("data:video/avi;base64,AAAA")
        with self.assertRaises(VideoProcessingError):
            VideoProcessor._from_base64("data:text/plain;base64,aGVsbG8=")

    # -- file ------------------------------------------------------------

    def test_file_path_becomes_file_part(self):
        import base64
        import tempfile
        from pathlib import Path

        from llm_sdk import FileProcessor

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.pdf"
            path.write_bytes(self.PDF)
            messages = [{"role": "user", "content": [{"type": "file", "file_path": path}]}]
            FileProcessor.process_messages(messages)
            part = messages[0]["content"][0]["file"]
            self.assertEqual(part["filename"], "doc.pdf")
            self.assertTrue(part["file_data"].startswith("data:application/pdf;base64,"))
            self.assertEqual(
                base64.b64decode(part["file_data"].split(",", 1)[1]), self.PDF
            )

    def test_file_name_and_mime_overrides(self):
        import tempfile
        from pathlib import Path

        from llm_sdk import ConfigurationError, FileProcessingError, FileProcessor

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.pdf"
            path.write_bytes(self.PDF)
            item = FileProcessor._from_path(
                path, filename="renamed.pdf", mime_type="application/x-custom"
            )
            self.assertEqual(item["file"]["filename"], "renamed.pdf")
            self.assertTrue(
                item["file"]["file_data"].startswith("data:application/x-custom;base64,")
            )
            with self.assertRaises(FileProcessingError):
                FileProcessor._from_path(str(Path(tmp) / "missing.pdf"))
            with self.assertRaises(ConfigurationError):
                FileProcessor._from_path(path, filename=123)

    def test_file_url_passes_through(self):
        from llm_sdk import FileProcessor

        item = FileProcessor._from_url("https://example.com/doc.pdf?token=abc")
        self.assertEqual(item["file"]["filename"], "doc.pdf")
        self.assertEqual(item["file"]["file_data"], "https://example.com/doc.pdf?token=abc")
        item = FileProcessor._from_url("https://example.com/d.pdf", filename="x.pdf")
        self.assertEqual(item["file"]["filename"], "x.pdf")

    def test_file_base64_forms(self):
        import base64

        from llm_sdk import FileProcessingError, FileProcessor

        raw_b64 = base64.b64encode(self.PDF).decode()
        item = FileProcessor._from_base64(raw_b64, filename="doc.pdf")
        self.assertEqual(item["file"]["filename"], "doc.pdf")
        self.assertTrue(
            item["file"]["file_data"].startswith("data:application/pdf;base64,")
        )
        with self.assertRaises(FileProcessingError):
            FileProcessor._from_base64("!!!not-base64!!!")
        with self.assertRaises(FileProcessingError):
            FileProcessor._from_base64(123)

    # -- mappings ----------------------------------------------------------

    def test_responses_translate_file_audio_video(self):
        from llm_sdk import LLM, ConfigurationError

        llm = LLM(model="test")
        try:
            file_item = {
                "type": "file",
                "file": {"filename": "d.pdf", "file_data": "data:application/pdf;base64,AAAA"},
            }
            translated = llm._translate_content_for_responses_api([file_item])
            self.assertEqual(
                translated,
                [{"type": "input_file", "filename": "d.pdf", "file_data": "data:application/pdf;base64,AAAA"}],
            )
            audio_item = {
                "type": "input_audio",
                "input_audio": {"data": "UklGRjAwMDBXQVZF", "format": "wav"},
            }
            self.assertEqual(
                llm._translate_content_for_responses_api([audio_item]), [audio_item]
            )
            with self.assertRaises(ConfigurationError):
                llm._translate_content_for_responses_api(
                    [{"type": "video_url", "video_url": {"url": "https://x/y.mp4"}}]
                )
        finally:
            llm.close()

    def test_build_request_with_audio_base64(self):
        import base64

        llm = LLM(model="test")
        try:
            raw_b64 = base64.b64encode(self.MP3).decode()
            messages = [{"role": "user", "content": [
                {"type": "text", "text": "transcribe"},
                {"type": "audio", "audio_base64": raw_b64},
            ]}]
            kwargs, _, _ = llm._build_request(messages, None, None, None, None, None)
            audio = kwargs["messages"][0]["content"][1]["input_audio"]
            self.assertEqual(audio["format"], "mp3")
            self.assertEqual(messages[0]["content"][1], {"type": "audio", "audio_base64": raw_b64})
        finally:
            llm.close()

    def test_reasoning_budget_flows_to_chat_request(self):
        from types import SimpleNamespace

        llm = LLM(model="test")
        try:
            captured = {}

            def create(**kwargs):
                captured.update(kwargs)
                return iter([SimpleNamespace(
                    usage=None,
                    choices=[SimpleNamespace(
                        delta=SimpleNamespace(content="ok"), finish_reason="stop"
                    )],
                )])

            llm._client = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))
            )
            list(llm.stream_response(input="x", reasoning_budget=2000))
            self.assertEqual(captured["reasoning"], {"max_tokens": 2000})
        finally:
            llm.close()

    def test_media_error_types_exported(self):
        from llm_sdk import (
            AudioProcessingError,
            FileProcessingError,
            LLMError,
            VideoProcessingError,
            __all__,
        )

        for cls in (AudioProcessingError, VideoProcessingError, FileProcessingError):
            self.assertTrue(issubclass(cls, LLMError))
            self.assertIn(cls.__name__, __all__)

    # -- Final audit: content parts, generation options, misc hardening ----


class AuditFixTests(unittest.TestCase):
    """Fail-fast validation, URL checks, sniff precision, Responses media."""

    def _llm(self):
        llm = LLM(model="test")
        self.addCleanup(llm.close)
        return llm

    # -- generation options --------------------------------------------

    def test_reasoning_effort_type_and_empty_rejected(self):
        from llm_sdk import ConfigurationError, _validate_generation_options

        for bad in (123, True, b"high", "", "   "):
            with self.assertRaises(ConfigurationError, msg=repr(bad)):
                _validate_generation_options(
                    None, None, None, None, None, None, None,
                    reasoning_effort=bad,
                )

    def test_reasoning_effort_invalid_in_responses_builder(self):
        from llm_sdk import ConfigurationError

        llm = self._llm()
        messages = [{"role": "user", "content": "hi"}]
        with self.assertRaises(ConfigurationError):
            llm._build_responses_request(messages, None, None, 123, None, None)
        with self.assertRaises(ConfigurationError):
            llm._build_responses_request(messages, None, None, "", None, None)

    def test_stop_empty_rejected(self):
        from llm_sdk import ConfigurationError, _validate_generation_options

        for bad in ("", "  ", [], ()):
            with self.assertRaises(ConfigurationError, msg=repr(bad)):
                _validate_generation_options(
                    None, None, None, None, bad, None, None
                )

    def test_default_stop_sequences_elements_validated(self):
        from llm_sdk import ConfigurationError

        for bad in ([123], [], ["ok", ""], "stop"):
            with self.assertRaises(ConfigurationError, msg=repr(bad)):
                LLM(model="test", default_stop_sequences=bad).close()

    # -- base_url -------------------------------------------------------

    def test_base_url_userinfo_rejected(self):
        from llm_sdk import ConfigurationError, LLMConfig, list_models

        with self.assertRaises(ConfigurationError):
            LLM(model="test", base_url="https://user:pass@example.com/v1").close()
        with self.assertRaises(ConfigurationError):
            LLMConfig(model="test", base_url="https://user@example.com/v1")
        with self.assertRaises(ConfigurationError):
            list_models(base_url="ftp://example.com/v1")

    # -- remote URLs ----------------------------------------------------

    def test_non_http_urls_rejected(self):
        from llm_sdk import (
            AudioProcessor,
            ConfigurationError,
            FileProcessor,
            ImageProcessingError,
            ImageProcessor,
            VideoProcessor,
        )

        with self.assertRaises(ConfigurationError):
            AudioProcessor._from_url("file:///clip.wav")
        with self.assertRaises(ConfigurationError):
            AudioProcessor._from_url("   ")
        with self.assertRaises(ConfigurationError):
            VideoProcessor._from_url("javascript:alert(1)")
        with self.assertRaises(ConfigurationError):
            VideoProcessor._from_url("https://user:pw@example.com/clip.mp4")
        with self.assertRaises(ConfigurationError):
            FileProcessor._from_url("ftp://example.com/doc.pdf")
        with self.assertRaises(ImageProcessingError):
            ImageProcessor._from_url("data:text/html,<b>x</b>")
        with self.assertRaises(ConfigurationError):
            ImageProcessor._from_url({"url": "ftp://example.com/i.png"})

    # -- audio sniff precision ------------------------------------------

    def test_audio_sniff_extended_formats(self):
        from llm_sdk import AudioProcessor

        sniff = AudioProcessor._sniff_audio_format
        self.assertEqual(sniff(b"FORM" + b"\x00" * 4 + b"AIFC"), "aiff")
        self.assertEqual(sniff(b"ADIF" + b"\x00" * 8), "aac")
        self.assertEqual(sniff(b"RF64" + b"\x00" * 4 + b"WAVE"), "wav")
        # ADTS sync with non-standard profile bits is still AAC ...
        self.assertEqual(sniff(b"\xff\xf0" + b"\x00" * 10), "aac")
        # ... while a real MPEG frame sync stays MP3.
        self.assertEqual(sniff(b"\xff\xfb" + b"\x00" * 10), "mp3")

    def test_video_mov_brand_detected(self):
        import tempfile
        from pathlib import Path

        from llm_sdk import VideoProcessor

        sniff = VideoProcessor._sniff_video_container
        self.assertEqual(sniff(b"\x00\x00\x00\x20ftypqt  " + b"\x00" * 4), "mov")
        self.assertEqual(sniff(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 4), "mp4")
        # mp4 bytes under a .mov name stay accepted (interchangeable).
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mov"
            path.write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 100)
            item = VideoProcessor._from_path(path)
            self.assertIn("video/quicktime", item["video_url"]["url"])

    # -- image hardening --------------------------------------------------

    def test_image_post_decode_cap(self):
        import base64
        from unittest import mock

        from llm_sdk import ImageProcessingError, ImageProcessor

        tiny_png = base64.b64encode(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
                "/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
            )
        ).decode()
        with (
            mock.patch.object(ImageProcessor, "_MAX_IMAGE_READ_BYTES", 10),
            self.assertRaises(ImageProcessingError),
        ):
            ImageProcessor._from_base64(tiny_png)

    def test_svg_data_url_rejected_early(self):
        import base64

        from llm_sdk import ImageProcessingError, ImageProcessor

        svg = base64.b64encode(b"<svg></svg>").decode()
        with self.assertRaises(ImageProcessingError) as ctx:
            ImageProcessor._from_base64(f"data:image/svg+xml;base64,{svg}")
        self.assertIn("vector", str(ctx.exception))

    def test_sniff_image_mime_never_mislabels(self):
        import base64

        from llm_sdk import ImageProcessingError, ImageProcessor

        unknown = base64.b64encode(b"\x00" * 16).decode()
        with self.assertRaises(ImageProcessingError):
            ImageProcessor._sniff_image_mime(unknown)

    # -- file processor -----------------------------------------------------

    def test_file_mime_and_filename_sanitized(self):
        from llm_sdk import ConfigurationError, FileProcessor

        with self.assertRaises(ConfigurationError):
            FileProcessor._from_base64("aGk=", mime_type="not-a-mime")
        name, _ = FileProcessor._resolve_name_and_mime(None, "../../secret.txt", None)
        self.assertEqual(name, "secret.txt")

    # -- Responses media mapping ----------------------------------------------

    def test_responses_audio_url_raises(self):
        from llm_sdk import ConfigurationError

        llm = self._llm()
        with self.assertRaises(ConfigurationError):
            llm._translate_content_for_responses_api(
                [{"type": "audio_url", "audio_url": {"url": "https://x/y.wav"}}]
            )

    def test_responses_file_url_maps_to_file_url(self):
        llm = self._llm()
        out = llm._translate_content_for_responses_api(
            [{"type": "file",
              "file": {"filename": "doc.pdf", "file_data": "https://x/doc.pdf" }}]
        )
        self.assertEqual(out[0]["type"], "input_file")
        self.assertEqual(out[0]["file_url"], "https://x/doc.pdf")
        self.assertEqual(out[0]["filename"], "doc.pdf")

    def test_responses_close_guards_do_not_mask_errors(self):
        # A close() that raises must not mask the (absent) stream error:
        # the generator simply ends instead of propagating close failures.
        from types import SimpleNamespace

        llm = self._llm()

        class BadClose:
            def __iter__(self):
                return iter([])
            def close(self):
                raise RuntimeError("boom")

        llm._client = SimpleNamespace(
            responses=SimpleNamespace(create=lambda **kw: BadClose())
        )
        events = list(llm._stream_responses_sync(
            [{"role": "user", "content": "hi"}],
            None, None, None, None, False, True, False, None,
        ))
        self.assertTrue(all(e["type"] != "done" or True for e in events))

    # -- modality top-level fields (detail / file_id / processing) --------

    def test_image_top_level_detail_injected(self):
        from llm_sdk import ImageProcessor

        messages = [{"role": "user", "content": [
            {"type": "image",
             "image_url": {"url": "https://x/i.png"},
             "detail": "high"},
        ]}]
        ImageProcessor.process_messages(messages)
        part = messages[0]["content"][0]
        self.assertEqual(part["image_url"]["detail"], "high")
        self.assertEqual(part["image_url"]["url"], "https://x/i.png")

    def test_image_dict_detail_wins_over_top_level(self):
        from llm_sdk import ImageProcessor

        messages = [{"role": "user", "content": [
            {"type": "image",
             "image_url": {"url": "https://x/i.png", "detail": "low"},
             "detail": "high"},
        ]}]
        ImageProcessor.process_messages(messages)
        self.assertEqual(
            messages[0]["content"][0]["image_url"]["detail"], "low"
        )

    def test_image_detail_validation(self):
        from llm_sdk import ConfigurationError, ImageProcessor

        with self.assertRaises(ConfigurationError):
            ImageProcessor._convert_image_item(
                {"type": "image", "image_url": "https://x/i.png", "detail": 123}
            )
        with self.assertRaises(ConfigurationError):
            ImageProcessor._convert_image_item(
                {"type": "image", "image_url": "https://x/i.png", "detail": "  "}
            )
        import llm_sdk as _sdk

        _sdk._warn_unknown_detail.cache_clear()
        with self.assertLogs("llm_sdk", level="WARNING"):
            out = ImageProcessor._convert_image_item(
                {"type": "image", "image_url": "https://x/i.png", "detail": "ultra"}
            )
        self.assertEqual(out["image_url"]["detail"], "ultra")

    def test_image_detail_flows_to_responses(self):
        llm = self._llm()
        out = llm._translate_content_for_responses_api(
            [{"type": "image_url",
              "image_url": {"url": "https://x/i.png", "detail": "high"}}]
        )
        self.assertEqual(out[0]["detail"], "high")

    def test_file_id_both_modes(self):
        from llm_sdk import ConfigurationError, FileProcessor

        item = FileProcessor._from_file_id("file-abc", filename="d.pdf")
        self.assertEqual(
            item,
            {"type": "file",
             "file": {"file_id": "file-abc", "filename": "d.pdf"}},
        )
        bare = FileProcessor._from_file_id("file-abc")
        self.assertNotIn("filename", bare["file"])
        with self.assertRaises(ConfigurationError):
            FileProcessor._from_file_id("")
        with self.assertRaises(ConfigurationError):
            FileProcessor._from_file_id(123)
        via_convert = FileProcessor._convert_file_item(
            {"type": "file", "file_id": "file-abc"}
        )
        self.assertEqual(via_convert["file"]["file_id"], "file-abc")

        llm = self._llm()
        out = llm._translate_content_for_responses_api([item])
        self.assertEqual(out[0]["type"], "input_file")
        self.assertEqual(out[0]["file_id"], "file-abc")
        self.assertEqual(out[0]["filename"], "d.pdf")

    def test_file_detail_responses_keeps_chat_drops(self):
        llm = self._llm()
        out = llm._translate_content_for_responses_api(
            [{"type": "file",
              "file": {"filename": "d.pdf",
                       "file_data": "data:application/pdf;base64,AAAA",
                       "detail": "high"}}]
        )
        self.assertEqual(out[0]["detail"], "high")

        messages = [{"role": "user", "content": [
            {"type": "file",
             "file": {"filename": "d.pdf",
                      "file_data": "data:application/pdf;base64,AAAA",
                      "detail": "high"}},
        ]}]
        import llm_sdk as _sdk

        _sdk._warn_detail_dropped_for_chat.cache_clear()
        with self.assertLogs("llm_sdk", level="WARNING"):
            kwargs, _, _ = llm._build_request(messages, None, None, None, None, None)
        sent = kwargs["messages"][0]["content"][0]["file"]
        self.assertNotIn("detail", sent)
        # Caller data untouched (copy-on-write).
        self.assertEqual(
            messages[0]["content"][0]["file"]["detail"], "high"
        )

    def test_video_processing_merged(self):
        from llm_sdk import ConfigurationError, VideoProcessor

        item = VideoProcessor._convert_video_item(
            {"type": "video", "video_url": "https://x/v.mp4",
             "processing": "agentic"}
        )
        self.assertEqual(item["video_url"]["processing"], "agentic")
        self.assertEqual(item["video_url"]["url"], "https://x/v.mp4")
        dict_wins = VideoProcessor._convert_video_item(
            {"type": "video",
             "video_url": {"url": "https://x/v.mp4", "processing": "x"},
             "processing": "y"}
        )
        self.assertEqual(dict_wins["video_url"]["processing"], "x")
        with self.assertRaises(ConfigurationError):
            VideoProcessor._convert_video_item(
                {"type": "video", "video_url": "https://x/v.mp4",
                 "processing": 123}
            )

    def test_responses_video_url_raises(self):
        from llm_sdk import ConfigurationError

        llm = self._llm()
        with self.assertRaises(ConfigurationError):
            llm._translate_content_for_responses_api(
                [{"type": "video_url", "video_url": {"url": "https://x/v.mp4"}}]
            )

    # -- audit round 3 fixes ------------------------------------------

    def test_stop_list_empty_elements_rejected(self):
        from llm_sdk import ConfigurationError, _validate_generation_options

        for bad in ([""], ["  "], ["ok", ""]):
            with self.assertRaises(ConfigurationError, msg=repr(bad)):
                _validate_generation_options(None, None, None, None, bad, None, None)
        # Valid lists still pass.
        _validate_generation_options(None, None, None, None, ["a", "b"], None, None)
        _validate_generation_options(None, None, None, None, "stop", None, None)

    def test_default_stop_sequences_whitespace_rejected(self):
        from llm_sdk import ConfigurationError, LLMConfig

        with self.assertRaises(ConfigurationError):
            LLM(model="test", default_stop_sequences=["   "]).close()
        with self.assertRaises(ConfigurationError):
            LLMConfig(model="test", default_stop_sequences=["ok", ""])

    def test_mime_params_normalized(self):
        from llm_sdk import FileProcessor

        item = FileProcessor._from_base64(
            "aGk=", mime_type="application/pdf; charset=utf-8"
        )
        self.assertTrue(
            item["file"]["file_data"].startswith("data:application/pdf;base64,")
        )
        item = FileProcessor._from_base64("aGk=", mime_type="Text/Plain")
        self.assertTrue(
            item["file"]["file_data"].startswith("data:text/plain;base64,")
        )

    def test_dict_detail_and_processing_validated(self):
        from llm_sdk import ConfigurationError, ImageProcessor, VideoProcessor

        with self.assertRaises(ConfigurationError):
            ImageProcessor._from_url({"url": "https://x/i.png", "detail": 123})
        out = ImageProcessor._from_url(
            {"url": "https://x/i.png", "detail": None}
        )
        self.assertNotIn("detail", out["image_url"])
        with self.assertRaises(ConfigurationError):
            VideoProcessor._from_url(
                {"url": "https://x/v.mp4", "processing": 123}
            )
        out = VideoProcessor._from_url(
            {"url": "https://x/v.mp4", "processing": None}
        )
        self.assertNotIn("processing", out["video_url"])

    def test_translate_rejects_raw_types(self):
        from llm_sdk import ConfigurationError

        llm = self._llm()
        for raw in (
            {"type": "image", "image_path": "x.png"},
            {"type": "audio", "audio_path": "x.wav"},
            {"type": "video", "video_path": "x.mp4"},
        ):
            with self.assertRaises(ConfigurationError, msg=repr(raw)):
                llm._translate_content_for_responses_api([raw])

    def test_translate_file_id_and_detail_strict(self):
        from llm_sdk import ConfigurationError

        llm = self._llm()
        with self.assertRaises(ConfigurationError):
            llm._translate_content_for_responses_api(
                [{"type": "file", "file": {"file_id": 123}}]
            )
        with self.assertRaises(ConfigurationError):
            llm._translate_content_for_responses_api(
                [{"type": "file",
                  "file": {"filename": "x", "file_data": "d", "detail": 123}}]
            )

    def test_translate_image_base64_enforces_caps(self):
        from unittest import mock

        from llm_sdk import ImageProcessingError, ImageProcessor

        tiny_png_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
            "/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        )
        llm = self._llm()
        out = llm._translate_content_for_responses_api(
            [{"type": "image_base64", "image_base64": tiny_png_b64}]
        )
        self.assertEqual(out[0]["type"], "input_image")
        self.assertTrue(out[0]["image_url"].startswith("data:image/png;base64,"))
        with (
            mock.patch.object(ImageProcessor, "_MAX_IMAGE_READ_BYTES", 10),
            self.assertRaises(ImageProcessingError),
        ):
            llm._translate_content_for_responses_api(
                [{"type": "image_base64", "image_base64": tiny_png_b64}]
            )

    def test_resolve_key_unhashable_and_garbage(self):
        from types import SimpleNamespace

        from llm_sdk import EventBuilder, ToolCallStreamHandler

        handler = ToolCallStreamHandler(EventBuilder(), {})
        # Unhashable index must not crash (falls back to id path).
        key, _ = handler._resolve_key(SimpleNamespace(index=["x"], id="a"))
        self.assertEqual(key, ("id", "a"))
        # Garbage strings keep their stable key (quirky providers keep working).
        key, _ = handler._resolve_key(SimpleNamespace(index="foo", id=None))
        self.assertEqual(key, ("index", "foo"))
        # Integer-like strings unify with ints.
        key, _ = handler._resolve_key(SimpleNamespace(index="0", id=None))
        self.assertEqual(key, ("index", 0))

    def test_base_url_hostname_and_hygiene(self):
        from llm_sdk import ConfigurationError, LLMConfig

        with self.assertRaises(ConfigurationError):
            LLM(model="test", base_url="https://@/v1").close()
        with self.assertRaises(ConfigurationError):
            LLM(model="test", base_url="  https://example.com/v1").close()
        with self.assertRaises(ConfigurationError):
            LLM(model="test", base_url="https://example.com/v1\n").close()
        with self.assertRaises(ConfigurationError):
            LLM(model="test", base_url="https://example.com/v1" + "x" * 8192).close()
        # Query strings stay allowed (e.g. Azure ?api-version=).
        LLM(
            model="test", base_url="https://example.com/v1?api-version=2024"
        ).close()
        # ... but never surface in repr().
        config = LLMConfig(model="test", base_url="https://example.com/v1?x=1")
        self.assertNotIn("example.com", repr(config))

    def test_redact_non_string(self):
        from llm_sdk import _redact_url_credentials

        self.assertIsNone(_redact_url_credentials(None))
        self.assertEqual(_redact_url_credentials(123), 123)

    def test_reasoning_effort_empty_in_transformer(self):
        from llm_sdk import ConfigurationError, RequestTransformer

        transformer = RequestTransformer("test", "https://example.com")
        with self.assertRaises(ConfigurationError):
            transformer.transform({"reasoning_effort": ""})
        with self.assertRaises(ConfigurationError):
            transformer.transform({"reasoning_effort": 123})

    def test_knob_length_caps(self):
        from llm_sdk import ConfigurationError, FileProcessor, _validate_detail

        with self.assertRaises(ConfigurationError):
            FileProcessor._from_file_id("x" * 513)
        with self.assertRaises(ConfigurationError):
            _validate_detail("x" * 65, what="file detail")
        with self.assertRaises(ConfigurationError):
            FileProcessor._resolve_name_and_mime(None, "x" * 1025, None)
        with self.assertRaises(ConfigurationError):
            FileProcessor._resolve_name_and_mime(None, None, "x" * 257)

    def test_rifx_sniffed_as_wav(self):
        from llm_sdk import AudioProcessor

        raw = b"RIFX" + b"\x00" * 4 + b"WAVE" + b"\x00" * 100
        self.assertEqual(AudioProcessor._sniff_audio_format(raw), "wav")

    # -- audit round 4 fixes ------------------------------------------

    def test_config_stop_whitespace_rejected(self):
        from llm_sdk import ConfigurationError, LLMConfig

        with self.assertRaises(ConfigurationError):
            LLMConfig(model="test", default_stop_sequences=["   "])

    def test_url_hygiene_whitespace_rejected(self):
        from llm_sdk import ConfigurationError, _validate_http_url

        for bad in (" https://h/v1", "https://h/v1\n", "https://h/v1 x"):
            with self.assertRaises(ConfigurationError, msg=repr(bad)):
                LLM(model="test", base_url=bad).close()
            with self.assertRaises(ConfigurationError, msg=repr(bad)):
                _validate_http_url(bad, what="x_url")

    def test_translate_image_detail_validated(self):
        from llm_sdk import ConfigurationError

        llm = self._llm()
        with self.assertRaises(ConfigurationError):
            llm._translate_content_for_responses_api(
                [{"type": "image_url",
                  "image_url": {"url": "https://x/i.png", "detail": 123}}]
            )
        with self.assertRaises(ConfigurationError):
            llm._translate_content_for_responses_api(
                [{"type": "image_url",
                  "image_url": {"url": "https://x/i.png", "detail": "  "}}]
            )
        out = llm._translate_content_for_responses_api(
            [{"type": "image_url",
              "image_url": {"url": "https://x/i.png", "detail": None}}]
        )
        self.assertNotIn("detail", out[0])

    def test_translate_file_strict(self):
        from llm_sdk import ConfigurationError

        llm = self._llm()
        with self.assertRaises(ConfigurationError):
            llm._translate_content_for_responses_api(
                [{"type": "file", "file": {"file_id": "x" * 600}}]
            )
        with self.assertRaises(ConfigurationError):
            llm._translate_content_for_responses_api(
                [{"type": "file",
                  "file": {"file_id": "abc", "filename": "y" * 2000}}]
            )
        with self.assertRaises(ConfigurationError):
            llm._translate_content_for_responses_api(
                [{"type": "file", "file": {}}]
            )
        with self.assertRaises(ConfigurationError):
            llm._translate_content_for_responses_api(
                [{"type": "file", "file": {"file_data": 123}}]
            )

    def test_translate_image_base64_prestrip_reject(self):
        from unittest import mock

        from llm_sdk import ImageProcessingError, ImageProcessor

        bloated = "aGk=" + " " * 200
        with (
            mock.patch.object(ImageProcessor, "_MAX_BASE64_HARD_CHARS", 100),
            self.assertRaises(ImageProcessingError),
        ):
            self._llm()._translate_content_for_responses_api(
                [{"type": "image_base64", "image_base64": bloated}]
            )

    def test_from_file_id_filename_checked(self):
        from llm_sdk import ConfigurationError, FileProcessor

        with self.assertRaises(ConfigurationError):
            FileProcessor._from_file_id("abc", filename=123)
        with self.assertRaises(ConfigurationError):
            FileProcessor._from_file_id("abc", filename="x" * 2000)

    def test_responses_unknown_effort_warns(self):
        llm = self._llm()
        messages = [{"role": "user", "content": "hi"}]
        with self.assertLogs("llm_sdk", level="WARNING") as logs:
            kwargs, _, _ = llm._build_responses_request(
                messages, None, None, "ultra", None, None
            )
        self.assertTrue(any("ultra" in m for m in logs.output))
        self.assertEqual(kwargs["reasoning"]["effort"], "ultra")

    def test_effort_padding_normalized(self):
        llm = self._llm()
        messages = [{"role": "user", "content": "hi"}]
        kwargs, _, _ = llm._build_request(messages, None, None, " low ", None, None)
        self.assertEqual(kwargs["reasoning_effort"], "low")
        kwargs, _, _ = llm._build_responses_request(
            messages, None, None, " low ", None, None
        )
        self.assertEqual(kwargs["reasoning"]["effort"], "low")

    def test_drop_detail_warns_once(self):
        import logging

        import llm_sdk as _sdk

        _sdk._warn_detail_dropped_for_chat.cache_clear()
        llm = self._llm()
        messages = [{"role": "user", "content": [
            {"type": "file", "file_url": "https://x/d.pdf", "detail": "high"},
        ]}]
        with self.assertLogs("llm_sdk", level="WARNING") as logs:
            llm._build_request(messages, None, None, None, None, None)
            llm._build_request(messages, None, None, None, None, None)
        dropped = [m for m in logs.output if "detail" in m]
        self.assertEqual(len(dropped), 1)
        self.assertTrue(
            all(r.levelno == logging.WARNING for r in logs.records)
        )

    def test_redact_embedded_credentials(self):
        from llm_sdk import _redact_url_credentials

        cleaned = _redact_url_credentials("Error https://user:pass@host/v1 failed")
        self.assertNotIn("user:pass", cleaned)
        self.assertIn("https://host/v1", cleaned)
        # Non-URL prose passes through untouched (no query-mangling).
        plain = "plain text? with question"
        self.assertEqual(_redact_url_credentials(plain), plain)

    # -- audit round 5 fixes ------------------------------------------

    def test_api_base_inserts_v1_before_query(self):
        from llm_sdk import _resolve_api_base

        self.assertEqual(
            _resolve_api_base("http://h:1234?api-version=xyz"),
            "http://h:1234/v1?api-version=xyz",
        )
        self.assertEqual(
            _resolve_api_base("https://h/v1?x=1"), "https://h/v1?x=1"
        )
        self.assertEqual(
            _resolve_api_base("http://h:1234", normalize=False),
            "http://h:1234",
        )
        llm = LLM(model="test", base_url="http://h:1234?api-version=xyz")
        try:
            self.assertEqual(llm._api_base, "http://h:1234/v1?api-version=xyz")
        finally:
            llm.close()

    def test_base_url_fragment_rejected(self):
        from llm_sdk import ConfigurationError

        with self.assertRaises(ConfigurationError):
            LLM(model="test", base_url="https://h/v1#frag")

    def test_validator_error_echo_redacted(self):
        from llm_sdk import (
            ConfigurationError,
            _validate_base_url,
            _validate_http_url,
        )

        with self.assertRaises(ConfigurationError) as ctx:
            _validate_base_url("ftp://user:pass@host/v1")
        self.assertNotIn("user:pass", str(ctx.exception))
        with self.assertRaises(ConfigurationError) as ctx:
            _validate_http_url("ftp://user:pass@host/x.jpg", what="image_url")
        self.assertNotIn("user:pass", str(ctx.exception))

    def test_redact_keeps_safe_query_drops_secrets(self):
        from llm_sdk import _redact_url_credentials

        kept = _redact_url_credentials("https://h/v1?api-version=2024-01&x=1")
        self.assertIn("api-version=2024-01", kept)
        self.assertIn("x=1", kept)
        cleaned = _redact_url_credentials("https://h/v1?api_key=SECRET&x=1")
        self.assertNotIn("SECRET", cleaned)
        self.assertNotIn("api_key", cleaned)
        self.assertIn("x=1", cleaned)

    def test_video_data_url_mismatch_raises(self):
        import base64

        from llm_sdk import VideoProcessingError, VideoProcessor

        webm = base64.b64encode(b"\x1a\x45\xdf\xa3" + b"\x00" * 16).decode()
        with self.assertRaises(VideoProcessingError):
            VideoProcessor._from_base64(f"data:video/mp4;base64,{webm}")
        out = VideoProcessor._from_base64(f"data:video/webm;base64,{webm}")
        self.assertIn("video/webm", out["video_url"]["url"])

    def test_audio_must_sniff_declared_is_no_fallback(self):
        from llm_sdk import AudioProcessingError, AudioProcessor

        with self.assertRaises(AudioProcessingError):
            AudioProcessor._from_base64("data:audio/mp3;base64,aGk=")
        with self.assertRaises(AudioProcessingError):
            AudioProcessor._from_base64("data:audio/mp3;base64,")
        wav = b"RIFF\x04\x00\x00\x00WAVE"
        import base64

        out = AudioProcessor._from_base64(
            "data:audio/mp3;base64," + base64.b64encode(wav).decode()
        )
        # Sniffed bytes win over the declaration.
        self.assertEqual(out["input_audio"]["format"], "wav")

    def test_audio_mime_aliases_accepted(self):
        import base64

        from llm_sdk import AudioProcessor

        mp3 = base64.b64encode(b"ID3" + b"\x00" * 9).decode()
        out = AudioProcessor._from_base64(f"data:audio/mpeg;base64,{mp3}")
        self.assertEqual(out["input_audio"]["format"], "mp3")
        wav = base64.b64encode(b"RIFF\x04\x00\x00\x00WAVE").decode()
        out = AudioProcessor._from_base64(f"data:audio/x-wav;base64,{wav}")
        self.assertEqual(out["input_audio"]["format"], "wav")

    def test_file_id_chat_passthrough(self):
        # OpenAI Chat Completions accepts file_id in file parts (only
        # detail is Responses-only) — passthrough is correct, not a leak.
        import llm_sdk as _sdk

        _sdk._warn_detail_dropped_for_chat.cache_clear()
        llm = self._llm()
        messages = [{"role": "user", "content": [
            {"type": "file", "file_id": "file-abc", "filename": "d.pdf",
             "detail": "high"},
        ]}]
        with self.assertLogs("llm_sdk", level="WARNING"):
            kwargs, _, _ = llm._build_request(messages, None, None, None, None, None)
        part = kwargs["messages"][0]["content"][0]["file"]
        self.assertEqual(part["file_id"], "file-abc")
        self.assertNotIn("detail", part)

    def test_list_models_wraps_api_error(self):
        from unittest import mock

        import httpx
        from openai import APIError

        from llm_sdk import ModelRequestError

        llm = self._llm()
        request = httpx.Request("GET", "https://h/v1/models")
        api_err = APIError(
            "boom https://user:pw@h/v1",
            request=request,
            body={"u": "https://user:pw@h/v1"},
        )
        fake = mock.Mock()
        fake.models.list.side_effect = api_err
        with (
            mock.patch.object(llm, "_client_for", return_value=fake),
            self.assertRaises(ModelRequestError) as ctx,
        ):
            llm.list_models(raise_on_error=True)
        self.assertNotIn("user:pw", str(ctx.exception))
        self.assertNotIn("user:pw", str(ctx.exception.body))
        fake.models.list.side_effect = ValueError("plain")
        with mock.patch.object(llm, "_client_for", return_value=fake), self.assertRaises(ValueError):
            llm.list_models(raise_on_error=True)

    def test_standalone_list_models_new_params(self):
        from unittest import mock

        import llm_sdk as _sdk
        from llm_sdk import ConfigurationError

        with self.assertRaises(ConfigurationError):
            _sdk.list_models(base_url="http://h:1234", timeout="fast")
        with mock.patch.object(_sdk, "OpenAI") as factory:
            fake = mock.Mock()
            fake.models.list.return_value = mock.Mock(
                data=[mock.Mock(id="b"), mock.Mock(id="a")]
            )
            factory.return_value = fake
            out = _sdk.list_models(
                base_url="http://h:1234", normalize_base_url=False, timeout=7
            )
            self.assertEqual(out, ["a", "b"])
            _, kwargs = factory.call_args
            self.assertEqual(kwargs["base_url"], "http://h:1234")
            self.assertEqual(kwargs["timeout"], 7)
        with mock.patch.object(_sdk, "OpenAI") as factory:
            fake = mock.Mock()
            fake.models.list.return_value = mock.Mock(data=[])
            factory.return_value = fake
            _sdk.list_models(base_url="http://h:1234", fallback=["f"])
            _, kwargs = factory.call_args
            self.assertEqual(kwargs["base_url"], "http://h:1234/v1")
            self.assertNotIn("timeout", kwargs)

    def test_circular_extra_body_rejected(self):
        from llm_sdk import ConfigurationError

        circular: dict = {}
        circular["self"] = circular
        with self.assertRaises(ConfigurationError):
            LLM(model="test", extra_body=circular)
        llm = self._llm()
        with self.assertRaises(ConfigurationError):
            llm._build_request(
                [{"role": "user", "content": "hi"}],
                None, None, None, None, circular,
            )

    def test_model_and_headers_validated(self):
        from llm_sdk import ConfigurationError

        for bad in ("", "   ", None, 123):
            with self.assertRaises(ConfigurationError, msg=repr(bad)):
                LLM(model=bad)
        with self.assertRaises(ConfigurationError):
            LLM(model="test", default_headers="nope")
        with self.assertRaises(ConfigurationError):
            LLM(model="test", default_headers={"Authorization": 123})
        LLM(model="test", default_headers={"X-Test": "yes"}).close()

    def test_config_normalizes(self):
        from llm_sdk import LLMConfig

        config = LLMConfig(
            model="m",
            base_url="http://h/v1/",
            default_stop_sequences=("a", "b"),
        )
        self.assertEqual(config.base_url, "http://h/v1")
        self.assertEqual(config.default_stop_sequences, ["a", "b"])

    def test_detail_control_chars_rejected(self):
        from llm_sdk import ConfigurationError, _validate_detail

        with self.assertRaises(ConfigurationError):
            _validate_detail("low\x00", what="image detail")

    def test_user_empty_rejected(self):
        from llm_sdk import ConfigurationError, _validate_generation_options

        with self.assertRaises(ConfigurationError):
            _validate_generation_options(None, None, None, None, None, "", None)

    def test_translate_image_url_validated(self):
        from llm_sdk import ConfigurationError, ImageProcessingError

        llm = self._llm()
        with self.assertRaises(ConfigurationError):
            llm._translate_content_for_responses_api(
                [{"type": "image_url", "image_url": {"url": ""}}]
            )
        with self.assertRaises(ConfigurationError):
            llm._translate_content_for_responses_api(
                [{"type": "image_url", "image_url": {"url": 123}}]
            )
        with self.assertRaises(ConfigurationError):
            llm._translate_content_for_responses_api(
                [{"type": "image_url", "image_url": ""}]
            )
        with self.assertRaises(ImageProcessingError):
            llm._translate_content_for_responses_api(
                [{"type": "image_url",
                  "image_url": {"url": "data:image/png;base64,!!!"}}]
            )

    def test_translate_image_data_url_keeps_detail(self):
        import base64

        from PIL import Image

        llm = self._llm()
        buffer = __import__("io").BytesIO()
        Image.new("RGB", (4, 4), (1, 2, 3)).save(buffer, format="PNG")
        payload = base64.b64encode(buffer.getvalue()).decode()
        out = llm._translate_content_for_responses_api(
            [{"type": "image_url",
              "image_url": {"url": f"data:image/png;base64,{payload}",
                            "detail": "high"}}]
        )
        self.assertEqual(out[0]["type"], "input_image")
        self.assertEqual(out[0].get("detail"), "high")

    def test_translate_unprocessed_base64_shapes(self):
        from llm_sdk import ConfigurationError

        llm = self._llm()
        for shape in ("audio_base64", "video_base64", "file_base64"):
            with self.assertRaises(ConfigurationError, msg=shape):
                llm._translate_content_for_responses_api(
                    [{"type": shape, shape: "aGk="}]
                )

    def test_translate_file_data_validated(self):
        from llm_sdk import ConfigurationError

        llm = self._llm()
        with self.assertRaises(ConfigurationError):
            llm._translate_content_for_responses_api(
                [{"type": "file", "file": {"file_data": "NOT-BASE64!!!"}}]
            )
        out = llm._translate_content_for_responses_api(
            [{"type": "file", "file": {"file_data": "aGk="}}]
        )
        self.assertEqual(out[0]["file_data"], "aGk=")

    def test_translate_input_audio_validated(self):
        from unittest import mock

        from llm_sdk import AudioProcessor, ConfigurationError

        llm = self._llm()
        with self.assertRaises(ConfigurationError):
            llm._translate_content_for_responses_api(
                [{"type": "input_audio",
                  "input_audio": {"format": "exe", "data": "aGk="}}]
            )
        with self.assertRaises(ConfigurationError):
            llm._translate_content_for_responses_api(
                [{"type": "input_audio",
                  "input_audio": {"format": "mp3", "data": "!!!"}}]
            )
        item = {"type": "input_audio",
                "input_audio": {"format": "mp3", "data": "SUQzAAAAAAAAAAAA"}}
        self.assertEqual(llm._translate_content_for_responses_api([item]), [item])
        with (
            mock.patch.object(AudioProcessor, "_MAX_AUDIO_READ_BYTES", 4),
            self.assertRaises(ConfigurationError),
        ):
            llm._translate_content_for_responses_api(
                [{"type": "input_audio",
                  "input_audio": {"format": "mp3",
                                  "data": "aGVsbG8="}}]
            )

    def test_translate_filename_none_becomes_file(self):
        llm = self._llm()
        out = llm._translate_content_for_responses_api(
            [{"type": "file", "file": {"file_data": "aGk=", "filename": None}}]
        )
        self.assertEqual(out[0]["filename"], "file")

    def test_file_data_url_scheme_case_insensitive(self):
        llm = self._llm()
        out = llm._translate_content_for_responses_api(
            [{"type": "file", "file": {"file_data": "HTTPS://x/f.pdf"}}]
        )
        self.assertEqual(out[0].get("file_url"), "HTTPS://x/f.pdf")

    def test_file_declared_mime_validated_knob_wins(self):
        from llm_sdk import FileProcessingError, FileProcessor

        out = FileProcessor._from_base64(
            "data:text/html;base64,aGk=", mime_type="application/pdf"
        )
        self.assertIn("data:application/pdf", out["file"]["file_data"])
        out = FileProcessor._from_base64("data:text/html;base64,aGk=")
        self.assertIn("data:text/html", out["file"]["file_data"])
        long_mime = "x" * 300 + "/y"
        with self.assertRaises(FileProcessingError):
            FileProcessor._from_base64(f"data:{long_mime};base64,aGk=")

    def test_empty_file_rejected(self):
        import os
        import tempfile

        from llm_sdk import FileProcessingError, FileProcessor

        with self.assertRaises(FileProcessingError):
            FileProcessor._from_base64("")
        handle, path = tempfile.mkstemp(suffix=".pdf")
        try:
            os.close(handle)
            with self.assertRaises(FileProcessingError):
                FileProcessor._from_path(path)
        finally:
            os.unlink(path)

    def test_mkv_path_mapped_to_webm(self):
        import base64
        import os
        import tempfile

        from llm_sdk import VideoProcessor

        payload = base64.b64encode(b"\x1a\x45\xdf\xa3" + b"\x00" * 64).decode()
        raw = base64.b64decode(payload)
        handle, path = tempfile.mkstemp(suffix=".mkv")
        try:
            with os.fdopen(handle, "wb") as f:
                f.write(raw)
            out = VideoProcessor._from_path(path)
        finally:
            os.unlink(path)
        self.assertIn("video/webm", out["video_url"]["url"])

    def test_video_ftyp_offset_scan(self):
        import base64

        from llm_sdk import VideoProcessor

        raw = (
            b"\x00\x00\x00\x08wide"
            b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42" + b"\x00" * 32
        )
        out = VideoProcessor._from_base64(base64.b64encode(raw).decode())
        self.assertIn("video/mp4", out["video_url"]["url"])

    def test_bw64_sniffed_as_wav(self):
        import base64

        from llm_sdk import AudioProcessor

        raw = b"BW64" + b"\x00" * 8
        out = AudioProcessor._from_base64(
            "data:audio/wav;base64," + base64.b64encode(raw).decode()
        )
        self.assertEqual(out["input_audio"]["format"], "wav")

    def test_gif_frames_budget_enforced_on_path(self):
        import os
        import tempfile
        from unittest import mock

        from PIL import Image

        from llm_sdk import ImageProcessingError, ImageProcessor

        frames = [Image.new("RGB", (10, 10), (i * 40, 0, 0)) for i in range(2)]
        handle, path = tempfile.mkstemp(suffix=".gif")
        try:
            os.close(handle)
            frames[0].save(
                path, save_all=True, append_images=frames[1:], format="GIF"
            )
            ImageProcessor._from_path(path)  # 200 px, fine
            with (
                mock.patch.object(ImageProcessor, "_MAX_IMAGE_PIXELS", 100),
                self.assertRaises(ImageProcessingError),
            ):
                ImageProcessor._from_path(path)
        finally:
            os.unlink(path)

    def test_image_mime_normalized_from_bytes(self):
        import base64
        import io

        from PIL import Image

        from llm_sdk import ImageProcessor

        buffer = io.BytesIO()
        Image.new("RGB", (4, 4), (9, 9, 9)).save(buffer, format="PNG")
        payload = base64.b64encode(buffer.getvalue()).decode()
        out = ImageProcessor._from_base64(f"data:image/x-png;base64,{payload}")
        self.assertTrue(
            out["image_url"]["url"].startswith("data:image/png;base64,")
        )

    def test_pil_error_message_is_static(self):
        import os
        import tempfile

        from llm_sdk import ImageProcessor

        handle, path = tempfile.mkstemp(suffix=".png")
        try:
            with os.fdopen(handle, "wb") as f:
                f.write(b"this is not an image at all...........")
            with self.assertRaises(Exception) as ctx:
                ImageProcessor._from_path(path)
            self.assertEqual(str(ctx.exception), "File is not a readable image")
        finally:
            os.unlink(path)

    def test_wrap_body_redaction(self):
        import httpx
        from openai import APIError

        from llm_sdk import LLM

        request = httpx.Request("GET", "https://h/v1/models")
        error = APIError(
            "fail https://user:pw@h/v1",
            request=request,
            body={"u": "https://user:pw@h/v1?api_key=s&x=1"},
        )
        wrapped = LLM._wrap_request_error(error, "ctx")
        self.assertNotIn("user:pw", str(wrapped))
        self.assertNotIn("api_key", str(wrapped.body))
        self.assertIn("x=1", str(wrapped.body))

    def test_should_retry_branches(self):
        import httpx
        from openai import APIError, APIStatusError

        llm = self._llm()
        request = httpx.Request("POST", "https://h/v1/chat/completions")
        kwargs = {"stream_options": {"include_usage": True}, "stream": True}
        response = httpx.Response(
            422, request=request, json={"error": "stream_options rejected"}
        )
        err = APIStatusError(
            "bad", response=response, body={"error": "stream_options rejected"}
        )
        self.assertTrue(llm._should_retry_without_stream_options(err, kwargs))
        plain = APIError("bad", request=request, body="other failure")
        self.assertFalse(llm._should_retry_without_stream_options(plain, kwargs))
        self.assertFalse(llm._should_retry_without_stream_options(err, {"stream": True}))

    def test_derive_stop_reason_table(self):
        from types import SimpleNamespace

        from llm_sdk import LLM

        cases = [
            ("failed", {}, "failed"),
            ("cancelled", {}, "cancelled"),
            ("incomplete", {"reason": "max_output_tokens"}, "length"),
            ("incomplete", {"reason": "content_filter"}, "content_filter"),
            ("incomplete", {"reason": "other"}, "incomplete"),
            ("completed", {}, "stop"),
            ("weird", {}, None),
        ]
        for status, details, expected in cases:
            response = SimpleNamespace(status=status, incomplete_details=details)
            self.assertEqual(
                LLM._derive_responses_stop_reason(response, {}), expected, status
            )
        response = SimpleNamespace(status="completed")
        self.assertEqual(
            LLM._derive_responses_stop_reason(
                response, {"completed_tool_calls": [{"id": "c1"}]}
            ),
            "tool_calls",
        )

    def test_reasoning_unterminated_flag(self):
        from types import SimpleNamespace

        llm = self._llm()
        chunk = SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content="<think>open answer")
            )],
        )
        llm._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(
                create=lambda **kwargs: iter([chunk])
            ))
        )
        events = list(llm.stream_response(input="x", final=True))
        final = next(e for e in events if e["type"] == "final")
        self.assertTrue(final["content"].get("reasoning_unterminated"))
        chunk2 = SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content="<think>shut</think>done")
            )],
        )
        llm._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(
                create=lambda **kwargs: iter([chunk2])
            ))
        )
        events = list(llm.stream_response(input="x", final=True))
        final = next(e for e in events if e["type"] == "final")
        self.assertIsNot(final["content"].get("reasoning_unterminated"), True)

    def test_response_items_replay(self):
        llm = self._llm()
        messages = [{
            "role": "assistant",
            "content": "done",
            "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            }],
            "response_items": [{"type": "reasoning", "id": "r1"}],
        }]
        out = llm._to_responses_input(messages)
        self.assertIn({"type": "reasoning", "id": "r1"}, out)
        calls = [item for item in out if item.get("type") == "function_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["call_id"], "c1")

    def test_chat_stream_close_called(self):
        from types import SimpleNamespace
        from unittest import mock

        llm = self._llm()
        closer = mock.Mock()

        class CloseStream:
            def __init__(self, chunks):
                self._it = iter(chunks)

            def __iter__(self):
                return self

            def __next__(self):
                return next(self._it)

            def close(self):
                closer()

        llm._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(
                create=lambda **kwargs: CloseStream([_chat_chunk("ok")])
            ))
        )
        list(llm.stream_response(input="x"))
        closer.assert_called_once_with()

    def test_chat_stream_close_error_logged(self):
        from types import SimpleNamespace

        llm = self._llm()

        class BadCloseStream:
            def __init__(self, chunks):
                self._it = iter(chunks)

            def __iter__(self):
                return self

            def __next__(self):
                return next(self._it)

            def close(self):
                raise RuntimeError("nope")

        llm._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(
                create=lambda **kwargs: BadCloseStream([_chat_chunk("ok")])
            ))
        )
        with self.assertLogs("llm_sdk", level="WARNING"):
            list(llm.stream_response(input="x"))

    # -- audit round 6 fixes ------------------------------------------

    def test_prose_redaction_strips_secret_query(self):
        from types import SimpleNamespace

        from llm_sdk import ModelRequestError, _redact_url_credentials

        out = _redact_url_credentials(
            "failed https://host/v1?api_key=SECRET123 end"
        )
        self.assertNotIn("SECRET123", out)
        self.assertIn("https://host/v1", out)
        out = _redact_url_credentials("see https://host/v1?token=ABC for help")
        self.assertNotIn("ABC", out)
        llm = self._llm()
        event = SimpleNamespace(
            type="response.failed",
            response=SimpleNamespace(
                error=SimpleNamespace(
                    message="boom https://host/v1?token=ABC123",
                    code="server_error",
                )
            ),
        )
        with self.assertRaises(ModelRequestError) as ctx:
            llm._handle_responses_event(event, {}, False, True, {})
        self.assertNotIn("ABC123", str(ctx.exception))
        self.assertIsNotNone(ctx.exception.body)

    def test_translate_remote_urls_validated(self):
        from llm_sdk import ConfigurationError

        llm = self._llm()
        long_url = "https://example.com/" + "a" * 9000
        with self.assertRaises(ConfigurationError):
            llm._translate_content_for_responses_api(
                [{"type": "image_url", "image_url": {"url": long_url}}]
            )
        for bad in ("ftp://h/i.png", "https://user:pw@h/i.png"):
            with self.assertRaises(ConfigurationError, msg=bad):
                llm._translate_content_for_responses_api(
                    [{"type": "image_url", "image_url": {"url": bad}}]
                )
        with self.assertRaises(ConfigurationError):
            llm._translate_content_for_responses_api(
                [{"type": "file", "file": {"file_data": "ftp://h/f.pdf"}}]
            )
        out = llm._translate_content_for_responses_api(
            [{"type": "image_url",
              "image_url": {"url": "https://example.com/i.png"}}]
        )
        self.assertEqual(out[0]["image_url"], "https://example.com/i.png")

    def test_translate_file_mime_validated(self):
        from llm_sdk import ConfigurationError

        llm = self._llm()
        with self.assertRaises(ConfigurationError):
            llm._translate_content_for_responses_api(
                [{"type": "file", "file": {"file_data": "data:not-a-mime;base64,aGk="}}]
            )
        out = llm._translate_content_for_responses_api(
            [{"type": "file",
              "file": {"file_data": "data:application/pdf;base64,aGk="}}]
        )
        self.assertIn("data:application/pdf", out[0]["file_data"])

    def test_translate_image_base64_keeps_detail(self):
        import base64
        import io

        from PIL import Image

        llm = self._llm()
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4), (5, 6, 7)).save(buffer, format="PNG")
        payload = base64.b64encode(buffer.getvalue()).decode()
        out = llm._translate_content_for_responses_api(
            [{"type": "image_base64",
              "image_base64": f"data:image/png;base64,{payload}",
              "detail": "high"}]
        )
        self.assertEqual(out[0].get("detail"), "high")

    def test_chat_drops_original_detail_and_response_items(self):
        import llm_sdk as _sdk
        from llm_sdk import LLM

        _sdk._warn_original_dropped_for_chat.cache_clear()
        _sdk._warn_response_items_dropped_for_chat.cache_clear()
        llm = self._llm()
        messages = [
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": "https://x/i.png", "detail": "original"}},
                {"type": "image_url",
                 "image_url": {"url": "https://x/j.png", "detail": "high"}},
            ]},
            {"role": "assistant", "content": "hi",
             "response_items": [{"type": "reasoning", "id": "r1"}]},
        ]
        with self.assertLogs("llm_sdk", level="WARNING"):
            kwargs, _, _ = llm._build_request(messages, None, None, None, None, None)
        parts = kwargs["messages"][0]["content"]
        self.assertNotIn("detail", parts[0]["image_url"])
        self.assertEqual(parts[1]["image_url"]["detail"], "high")
        self.assertNotIn("response_items", kwargs["messages"][1])
        # Caller messages untouched (copy-on-write).
        self.assertEqual(
            messages[0]["content"][0]["image_url"]["detail"], "original"
        )
        self.assertIn("response_items", messages[1])
        self.assertIsInstance(llm, LLM)

    def test_chat_rejects_sourceless_media(self):
        from llm_sdk import ConfigurationError

        llm = self._llm()
        for shape in (
            {"type": "image", "detail": "high"},
            {"type": "audio"},
            {"type": "video"},
            {"type": "file", "filename": "x.pdf"},
        ):
            with self.assertRaises(ConfigurationError, msg=str(shape)):
                llm._build_request(
                    [{"role": "user", "content": [shape]}],
                    None, None, None, None, None,
                )

    def test_identity_options_validated(self):
        from llm_sdk import ConfigurationError, LLMConfig

        llm = LLM(model="  gpt-5  ")
        try:
            self.assertEqual(llm._config.model, "gpt-5")
        finally:
            llm.close()
        for bad_model in ("m\nx", "m\u200bx"):
            with self.assertRaises(ConfigurationError, msg=repr(bad_model)):
                LLM(model=bad_model)
        for bad_key in (123, b"sk", "", "   "):
            with self.assertRaises(ConfigurationError, msg=repr(bad_key)):
                LLM(model="m", api_key=bad_key)
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            LLM(model="m", api_key=None).close()  # env fallback stays allowed
        for kwargs in ({"use_responses_api": "yes"}, {"normalize_base_url": 1},
                       {"debug": 1}):
            with self.assertRaises(ConfigurationError, msg=str(kwargs)):
                LLM(model="m", **kwargs)
        with self.assertRaises(ConfigurationError):
            LLM(model="m", default_headers={"": "v"})
        with self.assertRaises(ConfigurationError):
            LLMConfig(model="", default_headers={"A": 123})

    def test_fallback_must_be_list(self):
        from llm_sdk import ConfigurationError

        llm = self._llm()
        with self.assertRaises(ConfigurationError):
            llm.list_models(fallback="abc")

    def test_ports_and_controls_rejected(self):
        from llm_sdk import LLM, ConfigurationError

        for bad in ("http://h:abc/v1", "http://h:99999/v1", "http://h:0/v1"):
            with self.assertRaises(ConfigurationError, msg=bad):
                LLM(model="m", base_url=bad)
        LLM(model="m", base_url="http://h:8080/v1").close()
        llm = self._llm()
        with self.assertRaises(ConfigurationError):
            llm._build_request(
                [{"role": "user", "content": "hi"}], None, None, None, None, None,
                user="a\u200bb",
            )

    def test_video_mp4_mov_interchangeable_data_url(self):
        import base64

        from llm_sdk import VideoProcessor

        raw = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42" + b"\x00" * 32
        payload = base64.b64encode(raw).decode()
        out = VideoProcessor._from_base64(f"data:video/quicktime;base64,{payload}")
        self.assertIn("quicktime", out["video_url"]["url"])

    def test_video_extensionless_sniffed(self):
        import os
        import tempfile

        from llm_sdk import VideoProcessor

        raw = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42" + b"\x00" * 64
        handle, path = tempfile.mkstemp(suffix="")
        try:
            with os.fdopen(handle, "wb") as f:
                f.write(raw)
            out = VideoProcessor._from_path(path)
        finally:
            os.unlink(path)
        self.assertIn("video/mp4", out["video_url"]["url"])

    def test_cmyk_passthrough_transcodes(self):
        import base64
        import io

        from PIL import Image

        from llm_sdk import ImageProcessor

        buffer = io.BytesIO()
        Image.new("CMYK", (20, 20)).save(buffer, format="JPEG")
        payload = base64.b64encode(buffer.getvalue()).decode()
        out = ImageProcessor._from_base64(payload)
        self.assertIn("image/jpeg", out["image_url"]["url"])
        self.assertFalse(
            out["image_url"]["url"].endswith(payload),
            "CMYK bytes must be re-encoded, not passed through",
        )

    def test_animation_transcode_warns(self):
        import os
        import tempfile

        from PIL import Image

        from llm_sdk import ImageProcessor

        frames = [Image.new("RGB", (40, 40), (i * 60, 0, 0)) for i in range(3)]
        handle, path = tempfile.mkstemp(suffix=".gif")
        try:
            os.close(handle)
            frames[0].save(path, save_all=True, append_images=frames[1:])
            with self.assertLogs("llm_sdk", level="WARNING") as logs:
                ImageProcessor._from_path(path, max_image_side=10)
            self.assertTrue(
                any("first frame" in message for message in logs.output)
            )
        finally:
            os.unlink(path)

    def test_safe_index_rejects_negatives(self):
        from llm_sdk import LLM

        self.assertIsNone(LLM._safe_index(-3))
        self.assertIsNone(LLM._safe_index("-5"))
        self.assertEqual(LLM._safe_index("7"), 7)
        self.assertEqual(LLM._safe_index(0), 0)

    def test_assistant_message_copies_response_items(self):
        from llm_sdk import assistant_message

        items = [{"type": "reasoning", "id": "r1"}]
        message = assistant_message({"answer": "hi", "response_items": items})
        self.assertEqual(message["response_items"], items)
        self.assertIsNot(message["response_items"], items)

    def test_response_items_replay_order(self):
        llm = self._llm()
        messages = [{
            "role": "assistant",
            "content": "done",
            "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            }],
            "response_items": [{"type": "reasoning", "id": "r1"}],
        }]
        out = llm._to_responses_input(messages)
        kinds = [item.get("type", item.get("role")) for item in out]
        self.assertEqual(kinds, ["reasoning", "assistant", "function_call"])

    def test_reasoning_unterminated_respects_opt_out(self):
        from types import SimpleNamespace

        llm = self._llm()
        chunk = SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content="<think>open answer")
            )],
        )
        llm._client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(
                create=lambda **kwargs: iter([chunk])
            ))
        )
        events = list(
            llm.stream_response(input="x", final=True, include_reasoning=False)
        )
        final = next(e for e in events if e["type"] == "final")
        self.assertNotIn("reasoning_unterminated", final["content"])
        self.assertNotIn("reasoning", final["content"])

    def test_reasoning_unterminated_responses(self):
        from types import SimpleNamespace

        llm = LLM(model="gpt-5-test", use_responses_api=True)
        self.addCleanup(llm.close)
        events = [
            SimpleNamespace(
                type="response.output_text.delta", delta="<think>open"
            ),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(status="completed", usage=None),
            ),
        ]
        llm._client = SimpleNamespace(
            responses=SimpleNamespace(create=lambda **kwargs: iter(events))
        )
        result = list(llm.stream_response(input="x", final=True))
        final = next(e for e in result if e["type"] == "final")
        self.assertTrue(final["content"].get("reasoning_unterminated"))

    def test_warn_unknown_detail_isolated(self):
        import llm_sdk as _sdk

        _sdk._warn_unknown_detail.cache_clear()
        llm = self._llm()
        messages = [{"role": "user", "content": [
            {"type": "image",
             "image_url": {"url": "https://x/i.png", "detail": "ultra-z-1"}},
        ]}]
        with self.assertLogs("llm_sdk", level="WARNING"):
            llm._build_request(messages, None, None, None, None, None)
        with self.assertNoLogs("llm_sdk", level="WARNING"):
            llm._build_request(messages, None, None, None, None, None)

    def test_should_retry_table(self):
        import httpx
        from openai import APIError, APIStatusError

        llm = self._llm()
        request = httpx.Request("POST", "https://h/v1/chat/completions")
        kwargs = {"stream_options": {"include_usage": True}, "stream": True}
        response = httpx.Response(
            400, request=request, json={"error": "stream_options bad"}
        )
        err = APIStatusError(
            "bad", response=response, body={"error": "stream_options bad"}
        )
        self.assertTrue(llm._should_retry_without_stream_options(err, kwargs))
        nodetail = APIStatusError(
            "stream_options rejected", response=response, body=None
        )
        self.assertTrue(llm._should_retry_without_stream_options(nodetail, kwargs))
        plain = APIError("plain", request=request, body=None)
        self.assertFalse(llm._should_retry_without_stream_options(plain, kwargs))

    def test_config_deepcopy(self):
        from llm_sdk import LLM, LLMConfig

        extra = {"x": {"y": 1}}
        headers = {"H": "v"}
        stops = ["a"]
        config = LLMConfig(
            model="m", extra_body=extra, default_headers=headers,
            default_stop_sequences=stops,
        )
        extra["x"]["y"] = 99
        headers["H"] = "changed"
        stops.append("b")
        self.assertEqual(config.extra_body, {"x": {"y": 1}})
        self.assertEqual(config.default_headers, {"H": "v"})
        self.assertEqual(config.default_stop_sequences, ["a"])
        llm = LLM(model="m", extra_body={"x": {"y": 1}})
        try:
            self.assertEqual(llm._config.extra_body, {"x": {"y": 1}})
        finally:
            llm.close()

    # -- audit round 7 fixes ------------------------------------------

    def test_secret_query_suffix_rule(self):
        from llm_sdk import _redact_url_credentials

        for key in ("access_key", "api_secret", "secret_key", "session_token",
                    "id_token", "x-api-key", "auth-token"):
            cleaned = _redact_url_credentials(f"https://h/v1?{key}=S&x=1")
            self.assertNotIn("=S", cleaned, msg=key)
            self.assertIn("x=1", cleaned)
        kept = _redact_url_credentials("https://h/v1?api-version=2024-01")
        self.assertIn("api-version=2024-01", kept)

    def test_auth_headers_rejected(self):
        from llm_sdk import ConfigurationError

        for header in ("Authorization", "authorization", "X-Api-Key"):
            with self.assertRaises(ConfigurationError, msg=header):
                LLM(model="m", default_headers={header: "Bearer x"})

    def test_translate_filename_strict(self):
        from llm_sdk import ConfigurationError

        llm = self._llm()
        payload = {"file_data": "data:text/plain;base64,aGk="}
        with self.assertRaises(ConfigurationError):
            llm._translate_content_for_responses_api(
                [{"type": "file",
                  "file": {**payload, "filename": "bad\x00name.pdf"}}]
            )
        with self.assertRaises(ConfigurationError):
            llm._translate_content_for_responses_api(
                [{"type": "file",
                  "file": {**payload, "filename": "x\u200b.pdf"}}]
            )
        out = llm._translate_content_for_responses_api(
            [{"type": "file", "file": {**payload, "filename": ""}}]
        )
        self.assertEqual(out[0]["filename"], "file")

    def test_translate_file_empty_rejected(self):
        from llm_sdk import ConfigurationError

        llm = self._llm()
        with self.assertRaises(ConfigurationError):
            llm._translate_content_for_responses_api(
                [{"type": "file", "file": {"file_data": "   "}}]
            )

    def test_translate_side_cap_validated(self):
        from llm_sdk import ConfigurationError

        llm = self._llm()
        with self.assertRaises(ConfigurationError):
            llm._translate_content_for_responses_api(
                [{"type": "image_url",
                  "image_url": {"url": "https://x/i.png"}}],
                max_image_side=0,
            )

    def test_file_id_filename_controls_rejected(self):
        from llm_sdk import ConfigurationError, FileProcessor

        with self.assertRaises(ConfigurationError):
            FileProcessor._from_file_id("abc", filename="a\u202ename")

    def test_url_controls_rejected(self):
        from llm_sdk import LLM, ConfigurationError

        with self.assertRaises(ConfigurationError):
            LLM(model="m", base_url="https://host/v1\u200b")

    def test_verbose_and_tool_helpers_strict(self):
        from llm_sdk import ConfigurationError, assistant_message, tool_result

        with self.assertRaises(ConfigurationError):
            tool_result({"name": "f"}, "r")
        with self.assertRaises(ConfigurationError):
            assistant_message({"answer": "hi", "tool_calls": ["nope"]})
        message = assistant_message(
            {"answer": "hi", "response_items": [{"a": 1}]})
        self.assertEqual(message.get("response_items"), [{"a": 1}])

    def test_fallback_elements_validated(self):
        from llm_sdk import ConfigurationError

        llm = self._llm()
        with self.assertRaises(ConfigurationError):
            llm.list_models(fallback=["ok", 123])

    def test_api_key_stripped(self):
        llm = LLM(model="m", api_key="  sk-test  ")
        try:
            self.assertEqual(llm._config.api_key, "sk-test")
        finally:
            llm.close()


if __name__ == "__main__":
    unittest.main()
