"""Loading and validating the frozen model feature contract."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast


class FeatureContractError(ValueError):
    """Raised when model features do not match the saved contract."""


CATEGORICAL_LEVELS: Final[dict[str, tuple[int, ...]]] = {
    "day_of_week": tuple(range(0, 7)),
    "forecast_horizon": tuple(range(1, 7)),
    "hour_of_day": tuple(range(0, 24)),
    "location_id": tuple(range(1, 264)),
    "month": tuple(range(1, 13)),
}


@dataclass(frozen=True)
class FeatureContract:
    """Preserving the ordered features expected by the saved model."""

    feature_names: tuple[str, ...]
    feature_types: tuple[str, ...]
    order_must_match_exactly: bool

    @property
    def feature_count(self) -> int:
        """Returning the number of required model features."""
        return len(self.feature_names)


def load_feature_contract(path: Path) -> FeatureContract:
    """Loading and structurally validating a saved feature contract."""

    payload = cast(
        dict[str, object],
        json.loads(path.read_text(encoding="utf-8")),
    )

    saved_feature_count = int(cast(int, payload["feature_count"]))

    feature_names = tuple(
        str(value)
        for value in cast(
            list[object],
            payload["ordered_feature_names"],
        )
    )

    feature_types = tuple(
        str(value)
        for value in cast(
            list[object],
            payload["xgboost_feature_types"],
        )
    )

    order_must_match_exactly = bool(payload["order_must_match_exactly"])

    if saved_feature_count != len(feature_names):
        raise FeatureContractError("Saved feature_count does not match ordered_feature_names.")

    if len(feature_types) != len(feature_names):
        raise FeatureContractError(
            "Feature names and XGBoost feature types have different lengths."
        )

    if len(set(feature_names)) != len(feature_names):
        raise FeatureContractError("Feature names must be unique.")

    if not order_must_match_exactly:
        raise FeatureContractError("Production inference requires exact feature order.")

    return FeatureContract(
        feature_names=feature_names,
        feature_types=feature_types,
        order_must_match_exactly=order_must_match_exactly,
    )


def validate_feature_columns(
    columns: list[str] | tuple[str, ...],
    contract: FeatureContract,
) -> None:
    """Checking model-input names and order against the contract."""

    actual_columns = tuple(columns)

    if actual_columns != contract.feature_names:
        raise FeatureContractError(
            "Model-input columns do not match the saved feature names and order."
        )


def get_categorical_feature_names(
    contract: FeatureContract,
) -> tuple[str, ...]:
    """Returning features XGBoost treats as categorical."""
    names = contract.feature_names
    categories = contract.feature_types
    categorical_features = tuple(
        name
        for name, cat_type in zip(
            names,
            categories,
            strict=True,
        )
        if cat_type == "c"
    )

    return categorical_features
