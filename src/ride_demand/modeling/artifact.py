"""Validating saved model artifacts before inference."""

import hashlib
import json
from pathlib import Path
from typing import cast


def calculate_sha256(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Calculating a file checksum without loading it all at once."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


class ArtifactIntegrityError(RuntimeError):
    """Raised when a saved artifact fails validation."""


def verify_sha256(
    path: Path,
    expected_sha256: str,
) -> str:
    """Verifying a file against its expected SHA-256 value."""

    actual_sha256 = calculate_sha256(path)

    if actual_sha256 != expected_sha256:
        raise ArtifactIntegrityError("Artifact checksum does not match metadata.")

    return actual_sha256


def load_expected_sha256(
    metadata_path: Path,
) -> str:
    """Loading the expected model checksum from metadata."""

    payload = cast(
        dict[str, object],
        json.loads(metadata_path.read_text(encoding="utf-8")),
    )

    expected_sha256 = payload.get("model_sha256")

    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256.lower())
    ):
        raise ArtifactIntegrityError("Model metadata does not contain a valid SHA-256.")

    return expected_sha256.lower()
