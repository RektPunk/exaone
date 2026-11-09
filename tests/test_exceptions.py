import httpx2
import pytest

from exaone.exceptions import (
    ExaoneAPIError,
    ExaoneTabularError,
    raise_for_api_error,
    raise_for_tabular_error,
)


def test_raise_for_api_error_translates_http_error():
    response = httpx2.Response(429, json={"error": {"message": "slow down"}})

    with pytest.raises(ExaoneAPIError, match="slow down") as error:
        raise_for_api_error(response)

    assert error.value.status_code == 429
    assert error.value.body["error"]["message"] == "slow down"


def test_raise_for_api_error_allows_success():
    raise_for_api_error(httpx2.Response(200, json={"ok": True}))


def test_raise_for_tabular_error_translates_body_error():
    with pytest.raises(ExaoneTabularError, match="invalid csv") as error:
        raise_for_tabular_error(422, "invalid csv")

    assert error.value.result_code == 422


def test_raise_for_tabular_error_allows_success():
    raise_for_tabular_error(200, None)
