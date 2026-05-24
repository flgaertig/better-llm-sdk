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
            {"id": "call_1", "name": "weather", "arguments": {"city": "Berlin"}},
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


if __name__ == "__main__":
    unittest.main()
