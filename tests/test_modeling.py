"""Checks for the model artifact and feature contract helpers."""

import hashlib
import json
from pathlib import Path

import pytest

from ride_demand.modeling.artifact import (
    ArtifactIntegrityError,
    calculate_sha256,
    load_expected_sha256,
    verify_sha256,
)
from ride_demand.modeling.contract import (
    FeatureContractError,
    get_categorical_feature_names,
    load_feature_contract,
    validate_feature_columns,
)


def write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def contract_payload() -> dict[str, object]:
    return {
        "feature_count": 3,
        "ordered_feature_names": ["location_id", "hour_of_day", "pickup_lag_168"],
        "xgboost_feature_types": ["c", "c", "float"],
        "order_must_match_exactly": True,
    }


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    model_file = tmp_path / "model.ubj"
    model_file.write_bytes(b"x" * 3_000_000)

    expected = hashlib.sha256(b"x" * 3_000_000).hexdigest()

    assert calculate_sha256(model_file, chunk_size=1024) == expected
    assert verify_sha256(model_file, expected) == expected


def test_changed_file_fails_verification(tmp_path: Path) -> None:
    model_file = tmp_path / "model.ubj"
    model_file.write_bytes(b"original")
    expected = calculate_sha256(model_file)

    model_file.write_bytes(b"changed")

    with pytest.raises(ArtifactIntegrityError):
        verify_sha256(model_file, expected)


def test_metadata_needs_valid_sha256(tmp_path: Path) -> None:
    good = write_json(tmp_path / "good.json", {"model_sha256": "A" * 64})
    bad = write_json(tmp_path / "bad.json", {"model_sha256": "not-a-hash"})

    assert load_expected_sha256(good) == "a" * 64

    with pytest.raises(ArtifactIntegrityError):
        load_expected_sha256(bad)


def test_contract_loads_and_finds_categoricals(tmp_path: Path) -> None:
    contract = load_feature_contract(write_json(tmp_path / "c.json", contract_payload()))

    assert contract.feature_count == 3
    assert get_categorical_feature_names(contract) == ("location_id", "hour_of_day")


def test_contract_rejects_wrong_column_order(tmp_path: Path) -> None:
    contract = load_feature_contract(write_json(tmp_path / "c.json", contract_payload()))

    with pytest.raises(FeatureContractError):
        validate_feature_columns(["hour_of_day", "location_id", "pickup_lag_168"], contract)


def test_contract_rejects_count_mismatch(tmp_path: Path) -> None:
    payload = contract_payload() | {"feature_count": 4}

    with pytest.raises(FeatureContractError):
        load_feature_contract(write_json(tmp_path / "c.json", payload))
