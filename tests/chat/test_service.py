import asyncio
import json

import httpx2
import pytest

from exaone.chat.models import ChatCompletionRequest, ChatMessageRequest
from exaone.chat.service import AsyncChatService, ChatService
from exaone.exceptions import ExaoneAPIError

BASE_URL = "https://api.exaone.ai/v1"


def test_sync_chat_service_sends_request_and_parses_response(completion_payload):
    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        assert body["model"] == "k-exaone_v2.5"
        return httpx2.Response(200, json=completion_payload)

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        service = ChatService(client, api_key="test-key", base_url=BASE_URL)
        result = service.create({"messages": [{"role": "user", "content": "Hi"}]})

    assert result.choices[0].message.content == "Hello"


def test_sync_chat_service_streams_sse_chunks():
    body = (
        'data: {"id":"s1","created":1,"model":"k-exaone_v2.5",'
        '"choices":[{"index":0,"delta":{"content":"Hello"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    with httpx2.Client(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, text=body)),
    ) as client:
        service = ChatService(client, api_key="test-key", base_url=BASE_URL)
        chunks = list(service.stream({"messages": [{"role": "user", "content": "Hi"}]}))

    assert chunks[0].choices[0].delta.content == "Hello"


def test_sync_chat_service_rejects_stream_flag_for_create():
    with httpx2.Client(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200)),
    ) as client:
        service = ChatService(client, api_key="test-key", base_url=BASE_URL)
        request = ChatCompletionRequest(
            messages=[ChatMessageRequest(role="user", content="Hi")],
            stream=True,
        )
        with pytest.raises(ValueError, match=r"stream\(\)"):
            service.create(request)


def test_async_chat_service_create_and_stream(completion_payload):
    async def run():
        def handler(request):
            if json.loads(request.content).get("stream"):
                return httpx2.Response(
                    200,
                    text=(
                        'data: {"id":"s1","created":1,"model":"k-exaone_v2.5",'
                        '"choices":[{"index":0,"delta":{"reasoning_content":"why"}}]}\n\n'
                        "data: [DONE]\n\n"
                    ),
                )
            return httpx2.Response(200, json=completion_payload)

        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(handler),
        ) as client:
            service = AsyncChatService(client, api_key="test-key", base_url=BASE_URL)
            request = ChatCompletionRequest(
                messages=[ChatMessageRequest(role="user", content="Hi")],
            )
            response = await service.create(request)
            chunks = [chunk async for chunk in service.stream(request)]
        return response, chunks

    response, chunks = asyncio.run(run())
    assert response.choices[0].message.content == "Hello"
    assert chunks[0].choices[0].delta.reasoning_content == "why"


def test_chat_service_translates_http_error():
    with httpx2.Client(
        transport=httpx2.MockTransport(
            lambda _: httpx2.Response(429, json={"error": {"message": "slow down"}}),
        ),
    ) as client:
        service = ChatService(client, api_key="test-key", base_url=BASE_URL)
        with pytest.raises(ExaoneAPIError, match="slow down"):
            service.create({"messages": [{"role": "user", "content": "Hi"}]})
