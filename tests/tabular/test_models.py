import pytest
from pydantic import ValidationError

from exaone.tabular.models import ClassificationRequest, RegressionRequest


def test_tabular_request_defaults():
    assert ClassificationRequest().target_columns == ["target"]
    assert ClassificationRequest().output_type == "class"
    assert RegressionRequest().output_type == "point"


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (ClassificationRequest, {"output_type": "point"}),
        (RegressionRequest, {"output_type": "probability"}),
        (ClassificationRequest, {"target_columns": ["a", "b"]}),
    ],
)
def test_tabular_request_rejects_invalid_options(factory, kwargs):
    with pytest.raises(ValidationError):
        factory(**kwargs)
