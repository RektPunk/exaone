from os import PathLike
from typing import BinaryIO, Literal

from pydantic import BaseModel, ConfigDict, Field

TabularFile = str | PathLike[str] | BinaryIO


class _TabularRequest(BaseModel):
    mode: str = "default"
    target_columns: list[str] = Field(
        default_factory=lambda: ["target"],
        min_length=1,
        max_length=1,
    )
    model_config = ConfigDict(extra="forbid")


class ClassificationRequest(_TabularRequest):
    """Options for classifying rows in a CSV file."""

    output_type: Literal["class", "probability"] = "class"


class RegressionRequest(_TabularRequest):
    """Options for predicting numeric values from a CSV file."""

    output_type: Literal["point", "full"] = "point"


class _TabularResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class ClassificationData(_TabularResponse):
    """Labels, classes, or probabilities returned for classification rows."""

    target_columns: list[str] = Field(default_factory=list)
    classes: list[list[str]] | None = None
    labels: list[list[str]] | None = None
    probabilities: list[list[list[float]]] | None = None


class ClassificationResponse(_TabularResponse):
    """Complete response from the Tabular classification endpoint."""

    result_code: int
    description: str | None = None
    data: ClassificationData | None = None


class RegressionData(_TabularResponse):
    """Point predictions or quantile distributions for regression rows."""

    target_columns: list[str] = Field(default_factory=list)
    predictions: list[list[float]] | None = None
    quantiles: list[list[list[float]]] | None = None
    quantile_levels: list[list[float]] | None = None


class RegressionResponse(_TabularResponse):
    """Complete response from the Tabular regression endpoint."""

    result_code: int
    description: str | None = None
    data: RegressionData | None = None
