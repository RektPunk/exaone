import asyncio

import httpx2
import pytest

from exaone import AsyncExaone, Exaone
from exaone.chat.service import AsyncChatService, ChatService
from exaone.tabular.service import AsyncTabularService, TabularService


def test_client_reads_api_key_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("EXAONE_API_KEY", "test-key")
    client = Exaone()

    assert isinstance(client.chat, ChatService)
    assert isinstance(client.tabular, TabularService)
    client.close()


def test_client_requires_api_key_when_environment_is_empty(monkeypatch) -> None:
    monkeypatch.delenv("EXAONE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="EXAONE_API_KEY"):
        Exaone()


def test_client_closes_only_client_it_owns() -> None:
    owned_client = Exaone("test-key")
    owned_transport = owned_client._client
    owned_client.close()
    assert owned_transport.is_closed

    injected_transport = httpx2.Client()
    injected_client = Exaone("test-key", http_client=injected_transport)
    injected_client.close()
    assert not injected_transport.is_closed
    injected_transport.close()


def test_async_client_services_and_owned_transport_close() -> None:
    async def run() -> None:
        client = AsyncExaone("test-key")
        transport = client._client

        assert isinstance(client.chat, AsyncChatService)
        assert isinstance(client.tabular, AsyncTabularService)
        await client.close()
        assert transport.is_closed

    asyncio.run(run())
