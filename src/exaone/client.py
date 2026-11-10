import os
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any, Self

import httpx2

from exaone.chat import AsyncChatService, ChatService
from exaone.chat.models import (
    ChatCompletionChunkResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from exaone.tabular import AsyncTabularService, TabularService
from exaone.tabular.models import (
    ClassificationRequest,
    ClassificationResponse,
    RegressionRequest,
    RegressionResponse,
    TabularFile,
)

BASE_URL = "https://api.exaone.ai/v1"


def _resolve_api_key(api_key: str | None) -> str:
    resolved = api_key or os.getenv("EXAONE_API_KEY")
    if not resolved:
        msg = "API key is required. Pass api_key or set the EXAONE_API_KEY environment variable."
        raise ValueError(msg)
    return resolved


class Exaone:
    """Synchronous EXAONE client."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = BASE_URL,
        timeout: float | httpx2.Timeout = 60.0,
        http_client: httpx2.Client | None = None,
    ) -> None:
        """Create a client using an API key or ``EXAONE_API_KEY``."""
        self._api_key = _resolve_api_key(api_key)
        self._base_url = base_url.rstrip("/")
        self._owns_client = http_client is None
        self._client = http_client or httpx2.Client(
            base_url=self._base_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        self.chat = ChatService(
            self._client,
            api_key=self._api_key,
            base_url=self._base_url,
        )
        self.tabular = TabularService(
            client=self._client,
            api_key=self._api_key,
            base_url=self._base_url,
        )

    def close(self) -> None:
        """Close the HTTP client when this instance owns it."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        """Return this client for use in a ``with`` statement."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the owned HTTP client after a ``with`` block."""
        self.close()

    def create_chat_completion(
        self,
        request: ChatCompletionRequest | Mapping[str, Any],
    ) -> ChatCompletionResponse:
        """Create a chat completion."""
        return self.chat.create(request)

    def stream_chat_completion(
        self,
        request: ChatCompletionRequest | Mapping[str, Any],
    ) -> Iterator[ChatCompletionChunkResponse]:
        """Yield chunks from a streamed chat completion."""
        return self.chat.stream(request)

    def classify(
        self,
        train_file: TabularFile,
        test_file: TabularFile,
        request: ClassificationRequest,
    ) -> ClassificationResponse:
        """Predict labels or probabilities for CSV rows."""
        return self.tabular.classify(train_file, test_file, request)

    def regress(
        self,
        train_file: TabularFile,
        test_file: TabularFile,
        request: RegressionRequest,
    ) -> RegressionResponse:
        """Predict numeric values or quantiles for CSV rows."""
        return self.tabular.regress(train_file, test_file, request)


class AsyncExaone:
    """Asynchronous EXAONE client."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = BASE_URL,
        timeout: float | httpx2.Timeout = 60.0,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        """Create an async client using an API key or ``EXAONE_API_KEY``."""
        self._api_key = _resolve_api_key(api_key)
        self._base_url = base_url.rstrip("/")
        self._owns_client = http_client is None
        self._client = http_client or httpx2.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        self.chat = AsyncChatService(
            self._client,
            api_key=self._api_key,
            base_url=self._base_url,
        )
        self.tabular = AsyncTabularService(
            self._client,
            api_key=self._api_key,
            base_url=self._base_url,
        )

    async def close(self) -> None:
        """Close the async HTTP client when this instance owns it."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        """Return this client for use in an ``async with`` statement."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the owned async HTTP client after an ``async with`` block."""
        await self.close()

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest | Mapping[str, Any],
    ) -> ChatCompletionResponse:
        """Asynchronously create a chat completion."""
        return await self.chat.create(request)

    async def stream_chat_completion(
        self,
        request: ChatCompletionRequest | Mapping[str, Any],
    ) -> AsyncIterator[ChatCompletionChunkResponse]:
        """Yield chunks from an asynchronous chat stream."""
        return self.chat.stream(request)

    async def classify(
        self,
        train_file: TabularFile,
        test_file: TabularFile,
        request: ClassificationRequest,
    ) -> ClassificationResponse:
        """Asynchronously predict labels or probabilities for CSV rows."""
        return await self.tabular.classify(train_file, test_file, request)

    async def regress(
        self,
        train_file: TabularFile,
        test_file: TabularFile,
        request: RegressionRequest,
    ) -> RegressionResponse:
        """Asynchronously predict numeric values or quantiles for CSV rows."""
        return await self.tabular.regress(train_file, test_file, request)
