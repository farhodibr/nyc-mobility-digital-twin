"""Loading verified XGBoost models for inference."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import xgboost as xgb

from ride_demand.modeling.artifact import (
    load_expected_sha256,
    verify_sha256,
)
from ride_demand.modeling.contract import (
    CATEGORICAL_LEVELS,
    FeatureContract,
    FeatureContractError,
    get_categorical_feature_names,
    validate_feature_columns,
)


class PredictionValidationError(RuntimeError):
    """Raised when model predictions violate the inference contract."""


def load_verified_booster(
    model_path: Path,
    metadata_path: Path,
) -> xgb.Booster:
    """Loading an XGBoost model only after integrity verification."""

    expected_sha256 = load_expected_sha256(metadata_path)

    verify_sha256(model_path, expected_sha256)

    booster = xgb.Booster()

    booster.load_model(model_path)

    return booster


def prepare_feature_frame(
    frame: pd.DataFrame,
    contract: FeatureContract,
) -> pd.DataFrame:
    """Validating and copying a model-input feature frame."""

    frame_columns = frame.columns.tolist()

    validate_feature_columns(frame_columns, contract)

    frame_defensive_copy_df = frame.copy(deep=True)

    categorical_feature_names = get_categorical_feature_names(contract)

    if categorical_feature_names:
        frame_defensive_copy_df = convert_categorical_columns(frame_defensive_copy_df)

    return frame_defensive_copy_df


def build_feature_matrix(
    frame: pd.DataFrame,
    contract: FeatureContract,
) -> xgb.DMatrix:
    """Building a validated XGBoost inference matrix."""

    prepared_frame = prepare_feature_frame(
        frame,
        contract,
    )

    feature_matrix = xgb.DMatrix(
        data=prepared_frame,
        feature_names=list(contract.feature_names),
        feature_types=list(contract.feature_types),
        enable_categorical=True,
    )

    return feature_matrix


def validate_categorical_values(
    frame: pd.DataFrame,
) -> None:
    """Rejecting categorical values absent from the training contract."""

    for feature_name, allowed_levels in CATEGORICAL_LEVELS.items():
        if frame[feature_name].isna().any():
            raise FeatureContractError(
                f"Categorical feature '{feature_name}' contains missing values."
            )
        actual_levels = set(frame[feature_name].dropna().unique())

        unexpected_levels = actual_levels.difference(allowed_levels)
        if unexpected_levels:
            raise FeatureContractError(
                f"Categorical feature '{feature_name}' contains "
                f"unexpected levels: {sorted(unexpected_levels)}."
            )


def convert_categorical_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Converting validated features to fixed pandas categories."""

    validate_categorical_values(frame)

    converted_frame = frame.copy(deep=True)

    for feature_name, allowed_levels in CATEGORICAL_LEVELS.items():
        converted_frame[feature_name] = pd.Categorical(
            converted_frame[feature_name],
            categories=allowed_levels,
        )

    return converted_frame


def predict_demand(
    booster: xgb.Booster,
    frame: pd.DataFrame,
    contract: FeatureContract,
) -> npt.NDArray[np.float32]:
    """Predicting demand and clipping negative values to zero."""

    feature_matrix = build_feature_matrix(
        frame,
        contract,
    )

    raw_predictions = booster.predict(feature_matrix)
    expected_prediction_count = len(frame)

    if raw_predictions.ndim != 1:
        raise PredictionValidationError("Model predictions must be one-dimensional.")

    actual_prediction_count = len(raw_predictions)

    if actual_prediction_count != expected_prediction_count:
        raise PredictionValidationError(
            f"Expected {expected_prediction_count} predictions, "
            f"but received {actual_prediction_count}."
        )

    if not np.isfinite(raw_predictions).all():
        raise PredictionValidationError("Model predictions must contain only finite values.")

    clipped_predictions = np.maximum(
        raw_predictions,
        0.0,
    )

    return np.asarray(
        clipped_predictions,
        dtype=np.float32,
    )
