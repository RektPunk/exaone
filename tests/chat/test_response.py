import json

import httpx2
import pytest

from exaone.chat.models import (
    ChatCompletionChunkResponse,
    ChatCompletionResponse,
    ChatMessageResponse,
)
from exaone.chat.response import parse_chat_completion, parse_chat_completion_event
from exaone.chat.sse import ServerSentEvent
from exaone.exceptions import ExaoneResponseError


def test_parse_chat_completion_returns_typed_response(completion_payload):
    result = parse_chat_completion(
        httpx2.Response(
            200,
            json=completion_payload,
        ),
    )

    assert isinstance(result, ChatCompletionResponse)
    assert isinstance(result.choices[0].message, ChatMessageResponse)
    assert result.choices[0].message.reasoning_content == "Thinking"
    assert result.usage is not None
    assert result.usage.reasoning_tokens == 3


def test_parse_chat_stream_chunk_and_done_marker():
    event = ServerSentEvent(
        data=json.dumps(
            {
                "id": "chunk-1",
                "created": 1,
                "model": "k-exaone_v2.5",
                "choices": [{"index": 0, "delta": {"content": "Hi"}}],
            },
        ),
    )

    chunk = parse_chat_completion_event(event)
    assert isinstance(chunk, ChatCompletionChunkResponse)
    assert chunk.choices[0].delta.content == "Hi"
    assert parse_chat_completion_event(ServerSentEvent(data="[DONE]")) is None


@pytest.mark.parametrize(
    "response",
    [
        httpx2.Response(200, json={"bad": "shape"}),
        httpx2.Response(200, content=b"not-json"),
    ],
)
def test_parse_invalid_chat_response_raises_sdk_error(response):
    with pytest.raises(ExaoneResponseError):
        parse_chat_completion(response)


def test_parse_invalid_stream_event_raises_sdk_error():
    with pytest.raises(ExaoneResponseError, match="stream event"):
        parse_chat_completion_event(ServerSentEvent(data="not-json"))
