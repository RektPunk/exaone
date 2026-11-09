import os
from typing import Self

import httpx2

from .tabular import AsyncTabularService, TabularService
from .tabular.models import (
    ClassificationRequest,
    ClassificationResponse,
    RegressionRequest,
    RegressionResponse,
    TabularFile,
)

DEFAULT_BASE_URL = "https://api.exaone.ai/v1"


def _resolve_api_key(api_key: str | None) -> str:
    resolved = api_key or os.getenv("EXAONE_API_KEY")
    if not resolved:
        raise ValueError(
            "API key is required. Pass api_key or set the EXAONE_API_KEY environment variable."
        )
    return resolved


class Exaone:
    """Synchronous EXAONE client with ``chat`` and ``tabular`` services."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float | httpx2.Timeout = 60.0,
        http_client: httpx2.Client | None = None,
    ) -> None:
        self._api_key = _resolve_api_key(api_key)
        self._base_url = base_url.rstrip("/")
        self._owns_client = http_client is None
        self._client = http_client or httpx2.Client(
            base_url=self._base_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        self.tabular = TabularService(
            self._client, api_key=self._api_key, base_url=self._base_url
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def classify(
        self,
        train_file: TabularFile,
        test_file: TabularFile,
        request: ClassificationRequest,
    ) -> ClassificationResponse:
        return self.tabular.classify(train_file, test_file, request)

    def regress(
        self,
        train_file: TabularFile,
        test_file: TabularFile,
        request: RegressionRequest,
    ) -> RegressionResponse:
        return self.tabular.regress(train_file, test_file, request)


class AsyncExaone:
    """Asynchronous EXAONE client with ``chat`` and ``tabular`` services."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float | httpx2.Timeout = 60.0,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self._api_key = _resolve_api_key(api_key)
        self._base_url = base_url.rstrip("/")
        self._owns_client = http_client is None
        self._client = http_client or httpx2.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        self.tabular = AsyncTabularService(
            self._client, api_key=self._api_key, base_url=self._base_url
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def classify(
        self,
        train_file: TabularFile,
        test_file: TabularFile,
        request: ClassificationRequest,
    ) -> ClassificationResponse:
        return await self.tabular.classify(train_file, test_file, request)

    async def regress(
        self,
        train_file: TabularFile,
        test_file: TabularFile,
        request: RegressionRequest,
    ) -> RegressionResponse:
        return await self.tabular.regress(train_file, test_file, request)
