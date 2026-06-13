import functools
import unittest
from types import SimpleNamespace

from llm_sdk import EventType, LLM


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
        from llm_sdk import ToolCallStreamHandler, EventBuilder
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
        from llm_sdk import user_message, tool_result, assistant_message

        # Test user_message
        self.assertEqual(
            user_message("hello"),
            {"role": "user", "content": "hello"}
        )

        # Test tool_result
        self.assertEqual(
            tool_result({"id": "call_123"}, "success"),
            {"role": "tool", "tool_call_id": "call_123", "content": "success"}
        )

        # Test assistant_message with no tool calls
        self.assertEqual(
            assistant_message({"answer": "hello"}),
            {"role": "assistant", "content": "hello", "tool_calls": None}
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


if __name__ == "__main__":
    unittest.main()
