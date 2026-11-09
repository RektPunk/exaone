from os import PathLike
from typing import BinaryIO, Literal

from pydantic import BaseModel, ConfigDict, Field

TabularFile = str | PathLike[str] | BinaryIO


class _TabularRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_columns: list[str] = Field(
        default_factory=lambda: ["target"],
        min_length=1,
        max_length=1,
    )
    mode: str = "default"


class ClassificationRequest(_TabularRequest):
    output_type: Literal["class", "probability"] = "class"


class RegressionRequest(_TabularRequest):
    output_type: Literal["point", "full"] = "point"


class _TabularResponseModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class ClassificationData(_TabularResponseModel):
    """Labels, classes, or probabilities returned for classification rows."""

    target_columns: list[str] = Field(default_factory=list)
    classes: list[list[str]] | None = None
    labels: list[list[str]] | None = None
    probabilities: list[list[list[float]]] | None = None


class ClassificationResponse(_TabularResponseModel):
    """Complete response from the Tabular classification endpoint."""

    result_code: int
    description: str | None = None
    data: ClassificationData | None = None


class RegressionData(_TabularResponseModel):
    """Point predictions or quantile distributions for regression rows."""

    target_columns: list[str] = Field(default_factory=list)
    predictions: list[list[float]] | None = None
    quantiles: list[list[list[float]]] | None = None
    quantile_levels: list[list[float]] | None = None


class RegressionResponse(_TabularResponseModel):
    """Complete response from the Tabular regression endpoint."""

    result_code: int
    description: str | None = None
    data: RegressionData | None = None
