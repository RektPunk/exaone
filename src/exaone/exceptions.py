from collections.abc import Mapping
from typing import Any

import httpx2


class ExaoneError(Exception):
    """Base exception for client, protocol, and API errors."""


class ExaoneAPIError(ExaoneError):
    """An HTTP error returned by the EXAONE API."""

    def __init__(self, status_code: int, message: str, *, body: Any = None) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"EXAONE API returned HTTP {status_code}: {message}")


class ExaoneResponseError(ExaoneError):
    """A successful HTTP response that could not be decoded as expected."""


class ExaoneTabularError(ExaoneError):
    """An error reported in a Tabular API response body."""

    def __init__(self, result_code: int, description: str | None) -> None:
        self.result_code = result_code
        self.description = description
        message = description or "Tabular API returned an unsuccessful result"
        super().__init__(f"EXAONE Tabular result {result_code}: {message}")


def raise_for_api_error(response: httpx2.Response) -> None:
    """Raise a structured error when the API returned a non-success status."""
    if response.is_success:
        return

    try:
        body: Any = response.json()
    except ValueError:
        body = response.text

    message = body
    if isinstance(body, Mapping):
        error = body.get("error", body)
        message = error.get("message", error) if isinstance(error, Mapping) else error

    raise ExaoneAPIError(response.status_code, str(message), body=body)


def raise_for_tabular_error(result_code: int, description: str | None) -> None:
    """Raise when a Tabular API reports a non-200 result in its response body."""
    if result_code != 200:
        raise ExaoneTabularError(result_code, description)
