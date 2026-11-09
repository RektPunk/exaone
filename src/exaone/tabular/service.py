import httpx2

from exaone.exceptions import raise_for_api_error, raise_for_tabular_error
from exaone.tabular.models import (
    ClassificationRequest,
    ClassificationResponse,
    RegressionRequest,
    RegressionResponse,
    TabularFile,
)
from exaone.tabular.multipart import open_tabular_files
from exaone.tabular.response import (
    parse_classification_response,
    parse_regression_response,
)


def _form_fields(request: ClassificationRequest | RegressionRequest) -> dict[str, str]:
    return {
        "mode": request.mode,
        "output_type": request.output_type,
        "target_columns": request.target_columns[0],
    }


class TabularService:
    """Synchronous Tabular classification and regression API."""

    def __init__(self, client: httpx2.Client, *, api_key: str, base_url: str) -> None:
        self._client = client
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    def classify(
        self,
        train_file: TabularFile,
        test_file: TabularFile,
        request: ClassificationRequest,
    ) -> ClassificationResponse:
        """Predict class labels or class probabilities for the test CSV rows."""
        with open_tabular_files(train_file, test_file) as files:
            response = self._client.post(
                f"{self._base_url}/tabular/classification",
                data=_form_fields(request),
                files=files,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            raise_for_api_error(response)

        result = parse_classification_response(response)
        raise_for_tabular_error(result.result_code, result.description)
        return result

    def regress(
        self,
        train_file: TabularFile,
        test_file: TabularFile,
        request: RegressionRequest,
    ) -> RegressionResponse:
        """Predict point values or quantile distributions for the test rows."""
        with open_tabular_files(train_file, test_file) as files:
            response = self._client.post(
                f"{self._base_url}/tabular/regression",
                data=_form_fields(request),
                files=files,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            raise_for_api_error(response)

        result = parse_regression_response(response)
        raise_for_tabular_error(result.result_code, result.description)
        return result


class AsyncTabularService:
    """Asynchronous Tabular classification and regression API."""

    def __init__(
        self,
        client: httpx2.AsyncClient,
        *,
        api_key: str,
        base_url: str,
    ) -> None:
        self._client = client
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")

    async def classify(
        self,
        train_file: TabularFile,
        test_file: TabularFile,
        request: ClassificationRequest,
    ) -> ClassificationResponse:
        """Asynchronously classify the test CSV rows."""
        with open_tabular_files(train_file, test_file) as files:
            response = await self._client.post(
                f"{self._base_url}/tabular/classification",
                data=_form_fields(request),
                files=files,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            raise_for_api_error(response)

        result = parse_classification_response(response)
        raise_for_tabular_error(result.result_code, result.description)
        return result

    async def regress(
        self,
        train_file: TabularFile,
        test_file: TabularFile,
        request: RegressionRequest,
    ) -> RegressionResponse:
        """Asynchronously predict regression values for the test rows."""
        with open_tabular_files(train_file, test_file) as files:
            response = await self._client.post(
                f"{self._base_url}/tabular/regression",
                data=_form_fields(request),
                files=files,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
            raise_for_api_error(response)

        result = parse_regression_response(response)
        raise_for_tabular_error(result.result_code, result.description)
        return result
