import httpx2
import pytest

from exaone.exceptions import ExaoneResponseError
from exaone.tabular.models import ClassificationResponse, RegressionResponse
from exaone.tabular.response import (
    parse_classification_response,
    parse_regression_response,
)


def test_parse_classification_response() -> None:
    result = parse_classification_response(
        httpx2.Response(
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
        ),
    )

    assert isinstance(result, ClassificationResponse)
    assert result.data is not None
    assert result.data.labels == [["yes"]]


def test_parse_regression_full_response() -> None:
    result = parse_regression_response(
        httpx2.Response(
            200,
            json={
                "result_code": 200,
                "description": None,
                "data": {
                    "target_columns": ["price"],
                    "predictions": None,
                    "quantiles": [[[1.0, 2.0]]],
                    "quantile_levels": [[0.1, 0.9]],
                },
            },
        ),
    )

    assert isinstance(result, RegressionResponse)
    assert result.data is not None
    assert result.data.quantile_levels == [[0.1, 0.9]]


@pytest.mark.parametrize(
    "parser",
    [parse_classification_response, parse_regression_response],
)
def test_parse_invalid_tabular_response(parser) -> None:
    with pytest.raises(ExaoneResponseError):
        parser(httpx2.Response(200, json={"not": "a response"}))
