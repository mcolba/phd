from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Protocol, runtime_checkable

import numpy as np

from vol_risk.protocols import ModelParams

Transform = tuple[
    Callable[[ModelParams], tuple[np.ndarray, np.ndarray]],
    Callable[[np.ndarray | float], dict[str, np.ndarray]],
]


@runtime_checkable
class ParamTransform(Protocol):
    """Protocol for model parameter ↔ optimisation parameter transforms."""

    def encode(self, params: ModelParams) -> tuple[Sequence[np.ndarray], Sequence[np.ndarray]]:
        """Split full ModelParams into (free_params, fixed_params)."""
        ...

    def decode(self, free_params: Sequence[np.ndarray], fixed_params: Sequence[np.ndarray]) -> ModelParams:
        """Reconstruct a ModelParams from free and fixed arrays."""
        ...


def make_ravel_param(
    p0: ModelParams,
    transform: ParamTransform | None = None,
    check_unravel: bool = False,
) -> tuple[np.ndarray, Callable[[np.ndarray], ModelParams | dict]]:
    """Flatten and make unravel function.

    The transform is optional if provided it defined the new variable space. If a variable is not outputed
    by the decoder then it is treated as a constant.
    """
    if not isinstance(p0, ModelParams):
        msg = f"Expected dict or ModelParams; got {type(p0).__name__}"
        raise TypeError(msg)

    if transform is None:
        params_cls = type(p0)

        def encode(x: ModelParams) -> tuple[tuple, tuple]:
            return tuple(asdict(x).values()), ()

        def decode(free_params: list, *_: tuple) -> ModelParams:
            return params_cls(*free_params)
    else:
        encode, decode = transform

    free_params_0, const_params = encode(p0)

    idx = 0
    shapes = []
    raveled_slices = []
    raveled_values = []

    for value in free_params_0:
        value = np.atleast_1d(value)
        shapes.append(value.shape)
        raveled_values.append(np.ravel(value))
        size = value.size
        raveled_slices.append(slice(idx, idx + size))
        idx += size

    flat_free_params = np.concatenate(raveled_values)

    def unravel(x: np.ndarray) -> ModelParams | dict:
        """Converts a flattened NumPy array back to the original structure."""
        if x.size != flat_free_params.size:
            msg = f"Input array size {x.size} != expected {flat_free_params.size}"
            raise ValueError(msg)

        free_params = [x[sl].reshape(shape) for sl, shape in zip(raveled_slices, shapes, strict=True)]
        return decode(free_params, const_params)

    if check_unravel:
        p1 = unravel(flat_free_params)
        if not isinstance(p1, ModelParams):
            msg = f"Expected ModelParams, got {type(p1).__name__}"
            raise TypeError(msg)

        d0, d1 = asdict(p0), asdict(p1)

        if d0.keys() != d1.keys() or not all(np.allclose(d0[j], d1[j]) for j in d0):
            msg = "Unraveling did not reconstruct the original parameters correctly."
            raise ValueError(msg)

    return flat_free_params, unravel


def angles_to_simplex(theta: np.ndarray) -> np.ndarray:
    """Maps angles in [0, π/2] to a simplex in R^n.

    References:
        Rebonato, R. *Volatility and Correlation*, 2nd ed. (2004), §3.3.
    """
    n = len(theta)
    alpha = np.empty(n + 1)

    sin_product = 1.0
    for i in range(n):
        alpha[i] = np.cos(theta[i]) * sin_product
        sin_product *= np.sin(theta[i])

    alpha[n] = sin_product

    return alpha**2


def simplex_to_angles(a: np.ndarray) -> np.ndarray:
    """Maps a simplex in R^n to angles in [0, π/2]."""
    a = np.asarray(a)
    a_inv_cumsum = np.cumsum(a[::-1])[::-1]
    return np.pi / 2 - np.arctan(np.sqrt(a[:-1] / a_inv_cumsum[1:]))
