"""Protocols and data structures for option pricing, volatility surfaces, and risk management.
"""

import datetime as dt
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import jax.numpy as jnp
import numpy as np
import pandas as pd
from pandera import Check, Column, DataFrameSchema

"""
TODO: Move implementations out of here, keep only protocols and abstract classes.
TODO: adopt registry pattern.
"""

# ---------------------------------------------------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------------------------------------------------

Array = np.ndarray | jnp.ndarray


@runtime_checkable
class Interpolator(Protocol):
    """Protocol for IVS interpolation."""

    def __call__(self, x: Array, xp: Array, fp: Array, *args, **kwargs) -> Array: ...


Curve = tuple[Array, Interpolator]


option_chain_schema = DataFrameSchema(
    columns={
        "anchor": Column(dt.date, required=True),
        "strike": Column(float, Check.ge(0), required=True),
        "expiry": Column(dt.date, required=True),
        "bit": Column(int, required=True),
        "ask": Column(int, required=True),
        "volume": Column(int, required=True),
        "spot": Column(float, required=True),
        "option_type": Column(str, Check.isin(["call", "put"]), required=True),
    },
    checks=[Check(lambda df: df["expiry"] >= df["anchor"])],
    unique=["strike", "expiry"],
    coerce=True,
    strict=False,  # allows extra columns
)


@dataclass(frozen=True)
class OptionChain:
    """Option chain data."""

    df: pd.DataFrame

    def __post_init__(self):
        object.__setattr__(self, "df", option_chain_schema.validate(self.df))

    @property
    def strikes(self) -> Array:
        arr = self.df["strike"].to_numpy(copy=False)
        arr.flags.writeable = False
        return arr

    @property
    def expiries(self) -> Array:
        arr = self.df["expiry"].to_numpy(copy=False)
        arr.flags.writeable = False
        return arr

    @property
    def mid_prices(self) -> Array:
        arr = (self.df["bid"] - self.df["ask"]).to_numpy(copy=False)
        arr.flags.writeable = False
        return arr

    # ...


@dataclass(frozen=True)
class VolCalibrationContext:
    """Market data used in calibration."""

    rate_curve: tuple[Array, Interpolator]
    forward_curve: tuple[Array, Interpolator]


@dataclass(frozen=True)
class ValuationContext:
    anchor: dt.datetime
    vol_surface: "VolSurface"
    rates: Array


@dataclass(frozen=True)
class CalibrationSettings:
    """The input parameters for the calibration routine."""

    initial_guess: Array
    solver_settings: Array
    # ...


@dataclass(frozen=True)
class CalibrationResult:
    """The output of a successful calibration."""

    params: Array
    fit_error: float | None = None


@dataclass(frozen=True)
class EuropeanOption:
    expiry: Array
    strike: Array
    option_type: Array


# ---------------------------------------------------------------------------------------------------------------------
# Valuation protocols and abstract classes
# ---------------------------------------------------------------------------------------------------------------------


@runtime_checkable
class VolInterpolator(Protocol):
    """Valatility interpolation method.

    Protocol is used since an interpolator can be both a spline or a valuation model.

    # ========== Example 1 (Closure) ========== #

    from scipy.interpolate import RegularGridInterpolator

    def make_spline_interpolator(t_nodes, k_nodes, vol, method='cubic', **kwargs):
        interp = RegularGridInterpolator(points=(t_nodes, k_nodes), values=vol, *kwargs)
        def _vol_interp(t: Array, k: Array) -> Array:
            return interp(np.stack([t, k]))
    return _vol_interp

    # ========== Example 2 (Class) ========== #

    class GausianMixtureIntepolator:
        means: Array
        sigmas: Array
        weights: Array
        def __call__(self, t: Array, k: Array) -> Array:
            ...
    """

    def __call__(self, t: Array, k: Array) -> Array: ...


class VolSurface:
    def __init__(self, interpolator: VolInterpolator):
        self._interpolator = interpolator

    def vol(self, t, k) -> Array:
        return self._interpolator(t, k)


class ValuationModel(ABC):
    """Valuation model.

    Can be used both for interpolation and valuation of vanillas.
    """

    params: Array

    def __call__(self, t, k) -> Array:
        """Thanks to this we can use an option pricer as interpolator"""
        raise NotImplementedError

    def value(self):
        """Value of put and cal options."""
        raise NotImplementedError

    def make_vol_interpolator(self) -> VolInterpolator:
        """Returns an VolInterpolator protocol that can be used to construct a VolSurface."""
        raise NotImplementedError

    def make_surface(self, t: Array, k: Array) -> VolSurface:
        raise NotImplementedError


@runtime_checkable
class Solver(Protocol):
    def __call__(
        self,
        fun: Callable[[Array], float],
        x0: Array,
        kwargs=None,
    ) -> tuple[Array, Array]: ...


@runtime_checkable
class Calibrate(Protocol):
    """Defines a standard interface for a pure calibration function.

    This protocol acts as a contract for any function that performs volatility
    surface calibration. It ensures that all calibration routines in the system
    have a consistent signature, promoting modularity and a functional style.

    The protocol itself contains no mathematical logic. It only defines the
    "what" (the inputs and outputs). The actual implementation (the "how")
    is done in separate, concrete functions that adhere to this signature.

    This approach allows different calibration models (e.g., SVI, Heston) to be
    used interchangeably as long as they provide a function that follows this
    contract.

    Args:
        chain (OptionChain): The market data for options.
        mkt (VolCalibrationContext): The market context (spot, rates, etc.).
        config (CalibrationSettings): The configuration for the calibration run,
            including initial guesses and optimizer settings.

    Returns:
        CalibrationResult: An object containing the resulting volatility
            surface, the calibrated parameters, and the fit error.

    Example:
        A concrete implementation for a model named 'MyModel'.

        ```python
        # In a separate implementation file, e.g., black_scholes.py

        from phd.src.protocols import (
            Calibrate,
            CalibrationSettings,
            CalibrationResult,
            OptionChain,
            VolCalibrationContext,
        )


        def calibrate_xyz(
            chain: OptionChain, mkt: VolCalibrationContext, config: CalibrationSettings
        ) -> CalibrationResult:
            # 1. Unpack data and run the core mathematical optimization
            #    (This is where the real work happens)
            params, error = _calibrate_xyz(chain, mkt, config.initial_guess)

            # 2. Build the surface from the results
            surface = ...  # Build VolSurface from params

            # 3. Package results into the standard object
            return CalibrationResult(surface=surface, calibrated_params=params, fit_error=error)
        ```
    """

    def __call__(
        self,
        chain: OptionChain,
        mkt: VolCalibrationContext,
        solver: Solver,
        settings: CalibrationSettings,
        kwargs,
    ) -> CalibrationResult: ...


# ---------------------------------------------------------------------------------------------------------------------
# Risk factors protocols and abstract classes
# ---------------------------------------------------------------------------------------------------------------------


@runtime_checkable
class RiskFactorEncoder(Protocol):
    """Used to convert a vol surfaces to a reduced set of risk factors.

    For instance, VG, Melz, PCA factor, functional PCA, Heston parameters).
    """

    def encode(self, ivs: VolSurface, **kwargs) -> Array: ...
    def decode(self, rf: Array, **kwargs) -> VolSurface: ...


@runtime_checkable
class ShiftType(Protocol):
    """E.g., relative, absolute, and log-shifts."""

    def __call__(self, x: Array) -> Array: ...


class RiskFactorProcess(ABC):
    """E.g., GARCH, Neural-SDE, etc."""

    params: list[Array]  # needs a calibrated model

    def simulate(self, x: Array, shift: ShiftType) -> Array:
        raise NotImplementedError


class UnivariateProcess(RiskFactorProcess):
    """E.g., Univariate GARCH on each factor. Used in filtered HS."""

    def simulate_std_error(self):
        raise NotImplementedError

    def simulate(self, x: Array, shift: ShiftType) -> Array:
        raise NotImplementedError


class MultivariateProcess(RiskFactorProcess):
    """E.g., Univariate GARCH + Copula. Used in Monte Carlo engines."""



class RiskFactorEngine(ABC):
    def __init__(self, encoder: RiskFactorEncoder, process: UnivariateProcess, shift: ShiftType):
        self.encoder = encoder
        self.shift = shift
        self.process = process

    def simulate(self, base: Array[VolSurface], *, key, steps, paths) -> Array[VolSurface]:
        latent = self.process.simulate(self.encoder.encode(base), key, steps, paths)
        return self.encoder.decode(latent)


class HistoricalSimulator(RiskFactorEngine):
    """Simple or filtered, depending on the process."""



class MonteCarloSimulator(RiskFactorEngine):
    """Require a calibrated multivariate multivariate."""



@dataclass(frozen=True)
class ScenarioContext:
    mxt_t0: VolSurface


@dataclass(frozen=True)
class Product(ABC):
    """Base class for all products."""



@dataclass(frozen=True)
class AmericanOption(Product):
    """Equity American Option."""



@dataclass
class Position:
    product: Any
    quantity: float


@dataclass
class Portfolio:
    positions: list[Position]


@runtime_checkable
class ValuationStrategy(Protocol):
    def evaluate(self, product: Product, scenario: dict) -> float: ...


def build_strategy(name: str, product: Product, config: dict) -> ValuationStrategy:
    if name == "full_reval":
        msg = "Full revaluation strategy is not implemented."
        raise NotImplementedError(msg)
    if name == "nn":
        msg = "Neural network valuation strategy is not implemented."
        raise NotImplementedError(msg)
    if name == "sens":
        msg = "Sensitivity-based valuation strategy is not implemented."
        raise NotImplementedError(msg)


class VarEngine:
    """Converts scenarios → P/L distribution → risk metric."""

