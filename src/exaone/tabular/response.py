"""Convert Tabular API payloads into typed response models."""

import httpx2

from ..exceptions import ExaoneResponseError
from .models import ClassificationResponse, RegressionResponse


def parse_classification_response(response: httpx2.Response) -> ClassificationResponse:
    """Validate a classification endpoint response."""
    try:
        return ClassificationResponse.model_validate_json(response.content)
    except (TypeError, ValueError) as exc:
        raise ExaoneResponseError("Invalid tabular classification response") from exc


def parse_regression_response(response: httpx2.Response) -> RegressionResponse:
    """Validate a regression endpoint response."""
    try:
        return RegressionResponse.model_validate_json(response.content)
    except (TypeError, ValueError) as exc:
        raise ExaoneResponseError("Invalid tabular regression response") from exc
