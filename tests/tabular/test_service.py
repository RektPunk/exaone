import asyncio
from io import BytesIO

import httpx2
import pytest

from exaone.exceptions import ExaoneTabularError
from exaone.tabular.models import ClassificationRequest, RegressionRequest
from exaone.tabular.service import AsyncTabularService, TabularService

BASE_URL = "https://api.exaone.ai/v1"


def test_sync_tabular_classification_multipart():
    def handler(request):
        assert request.url.path == "/v1/tabular/classification"
        assert request.headers["authorization"] == "Bearer test-key"
        assert "multipart/form-data" in request.headers["content-type"]
        body = request.content.decode()
        assert 'name="train_file"; filename="train.csv"' in body
        assert 'name="test_file"; filename="test.csv"' in body
        assert 'name="target_columns"' in body and "churn" in body
        return httpx2.Response(
            200,
            json={
                "result_code": 200,
                "description": None,
                "data": {
                    "target_columns": ["churn"],
                    "classes": [["no", "yes"]],
                    "labels": [["yes"]],
                    "probabilities": None,
                },
            },
        )

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        service = TabularService(client, api_key="test-key", base_url=BASE_URL)
        result = service.classify(
            BytesIO(b"churn,age\nno,20\nyes,50\n"),
            BytesIO(b"age\n30\n"),
            ClassificationRequest(target_columns=["churn"]),
        )

    assert result.data is not None
    assert result.data.labels == [["yes"]]


def test_async_tabular_regression_and_result_error():
    async def run():
        count = 0

        def handler(request):
            nonlocal count
            count += 1
            if count == 1:
                return httpx2.Response(
                    200,
                    json={
                        "result_code": 200,
                        "description": None,
                        "data": {
                            "target_columns": ["price"],
                            "predictions": [[42.0]],
                            "quantiles": None,
                            "quantile_levels": None,
                        },
                    },
                )
            return httpx2.Response(
                200,
                json={"result_code": 422, "description": "bad csv", "data": None},
            )

        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(handler)
        ) as client:
            service = AsyncTabularService(client, api_key="test-key", base_url=BASE_URL)
            request = RegressionRequest(target_columns=["price"])
            first = await service.regress(
                BytesIO(b"price\n42\n"), BytesIO(b"\n"), request
            )
            with pytest.raises(ExaoneTabularError, match="bad csv"):
                await service.regress(BytesIO(b"price\n42\n"), BytesIO(b"\n"), request)
        return first

    result = asyncio.run(run())
    assert result.data is not None
    assert result.data.predictions == [[42.0]]
