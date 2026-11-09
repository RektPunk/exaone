import httpx2

from exaone.exceptions import ExaoneResponseError
from exaone.tabular.models import ClassificationResponse, RegressionResponse


def parse_classification_response(response: httpx2.Response) -> ClassificationResponse:
    """Validate a classification endpoint response."""
    try:
        return ClassificationResponse.model_validate_json(response.content)
    except (TypeError, ValueError) as exc:
        msg = "Invalid tabular classification response"
        raise ExaoneResponseError(msg) from exc


def parse_regression_response(response: httpx2.Response) -> RegressionResponse:
    """Validate a regression endpoint response."""
    try:
        return RegressionResponse.model_validate_json(response.content)
    except (TypeError, ValueError) as exc:
        msg = "Invalid tabular regression response"
        raise ExaoneResponseError(msg) from exc
