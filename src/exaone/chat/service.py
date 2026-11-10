from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

import httpx2

from exaone.chat.models import (
    ChatCompletionChunkResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from exaone.chat.response import parse_chat_completion, parse_chat_completion_event
from exaone.chat.sse import SSEDecoder
from exaone.exceptions import raise_for_api_error


def _request_payload(
    request: ChatCompletionRequest | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(request, ChatCompletionRequest):
        return request.model_dump(exclude_none=True)
    return ChatCompletionRequest.model_validate(request).model_dump(exclude_none=True)


class ChatService:
    """Synchronous Chat Completions API."""

    def __init__(self, client: httpx2.Client, *, api_key: str, base_url: str) -> None:
        self._client = client
        self._api_key = api_key
        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"

    def create(
        self,
        request: ChatCompletionRequest | Mapping[str, Any],
    ) -> ChatCompletionResponse:
        """Create a non-streaming chat completion."""
        payload = _request_payload(request)
        if payload.get("stream"):
            msg = "Use stream() when stream=True"
            raise ValueError(msg)

        response = self._client.post(
            url=self._endpoint,
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        raise_for_api_error(response)

        return parse_chat_completion(response)

    def stream(
        self,
        request: ChatCompletionRequest | Mapping[str, Any],
    ) -> Iterator[ChatCompletionChunkResponse]:
        """Yield response chunks until the server sends its end marker."""
        payload = _request_payload(request)
        payload["stream"] = True
        with self._client.stream(
            method="POST",
            url=self._endpoint,
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key}"},
        ) as response:
            if not response.is_success:
                response.read()
                raise_for_api_error(response)

            for event in SSEDecoder().decode(response.iter_lines()):
                chunk = parse_chat_completion_event(event)
                if chunk is None:
                    return
                yield chunk


class AsyncChatService:
    """Asynchronous Chat Completions API."""

    def __init__(
        self,
        client: httpx2.AsyncClient,
        *,
        api_key: str,
        base_url: str,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"

    async def create(
        self,
        request: ChatCompletionRequest | Mapping[str, Any],
    ) -> ChatCompletionResponse:
        """Asynchronously create a non-streaming chat completion."""
        payload = _request_payload(request)
        if payload.get("stream"):
            msg = "Use stream() when stream=True"
            raise ValueError(msg)

        response = await self._client.post(
            url=self._endpoint,
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        raise_for_api_error(response)

        return parse_chat_completion(response)

    async def stream(
        self,
        request: ChatCompletionRequest | Mapping[str, Any],
    ) -> AsyncIterator[ChatCompletionChunkResponse]:
        """Asynchronously yield chunks until the server sends its end marker."""
        payload = _request_payload(request)
        payload["stream"] = True
        async with self._client.stream(
            method="POST",
            url=self._endpoint,
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key}"},
        ) as response:
            if not response.is_success:
                await response.aread()
                raise_for_api_error(response)

            async for event in SSEDecoder().decode_async(response.aiter_lines()):
                chunk = parse_chat_completion_event(event)
                if chunk is None:
                    return
                yield chunk
