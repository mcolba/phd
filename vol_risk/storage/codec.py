"""Serialization helpers for SQLite calibration artifacts."""

from __future__ import annotations

import io
import json
import pickle
from typing import TYPE_CHECKING
from zipfile import BadZipFile

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping


def canonical_config(config: Mapping[str, object]) -> str:
    """Return the canonical JSON serialization of a configuration."""
    try:
        payload = json.dumps(
            dict(config),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        msg = "config must contain only JSON-serializable finite values"
        raise ValueError(msg) from error
    return payload


def pickle_bytes(value: object) -> bytes:
    """Serialize a non-numeric calibration value for SQLite BLOB storage."""
    return pickle.dumps(value, protocol=5)


def unpickle_bytes(value: bytes, field_name: str) -> object:
    """Deserialize a value read from a trusted calibration database."""
    try:
        return pickle.loads(value)  # noqa: S301
    except (AttributeError, EOFError, ImportError, IndexError, pickle.UnpicklingError) as error:
        msg = f"Stored {field_name} is not a valid artifact"
        raise ValueError(msg) from error


def encode_numpy_fields(fields: Mapping[str, object]) -> bytes:
    """Encode named numeric values and arrays as a NumPy BLOB."""
    arrays = {}
    for name, value in fields.items():
        if not isinstance(name, str) or not name or name == "file":
            msg = "NumPy field names must be non-empty strings other than 'file'"
            raise ValueError(msg)
        try:
            array = np.asarray(value)
        except (TypeError, ValueError) as error:
            msg = f"NumPy field {name!r} must be numeric"
            raise ValueError(msg) from error
        if array.dtype.kind not in "biufc":
            msg = f"NumPy field {name!r} must be numeric"
            raise ValueError(msg)
        arrays[name] = array

    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def decode_numpy_fields(value: bytes) -> dict[str, np.ndarray]:
    """Decode named numeric values and arrays from a NumPy BLOB."""
    try:
        with np.load(io.BytesIO(value), allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    except (BadZipFile, EOFError, OSError, TypeError, ValueError) as error:
        msg = "Stored NumPy fields are not a valid artifact"
        raise ValueError(msg) from error
