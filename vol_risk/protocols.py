"""Protocols and data structures for option pricing, volatility surfaces, and risk management."""

import datetime as dt
from abc import ABC
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import jax.numpy as jnp
import numpy as np
import pandera as pa
from pandera.pandas import Check, Column, DataFrameSchema

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

@runtime_checkable
class DayCountCalendar(Protocol):
    """Protocol for computing year-fraction between two dates."""

    def year_fraction(self, start: dt.datetime | np.ndarray, end: dt.datetime | np.ndarray) -> float | np.ndarray:
        raise NotImplementedError


@runtime_checkable
class OptionChainLike(Protocol):
    """Read-only option chain interface.

    This matches the current `vol_risk.data.option_chain.OptionChain` API while
    allowing alternative backends (Arrow/JAX/etc.) to implement the same shape.
    """

    def __iter__(self) -> Iterator[tuple[dt.datetime, "OptionChainLike"]]:
        """Iterate by expiry, yielding (expiry, slice)."""
        ...

    def __getitem__(self, column: str) -> Array:
        """Column access (backend-defined semantics)."""
        ...

    @property
    def spot(self) -> float:
        """Spot price S (unique within the chain)."""
        ...

    @property
    def k(self) -> Array:
        """Strike array K."""
        ...

    @property
    def expiry(self) -> Array:
        """Expiry timestamps."""
        ...

    @property
    def tau(self) -> Array:
        """Time-to-maturity in years (Actual/365 fixed by convention)."""
        ...

    @property
    def mid(self) -> Array:
        """Mid prices (typically (bid + ask) / 2)."""
        ...

    @property
    def option_type(self) -> Array:
        """Option type codes (e.g., 'C'/'P')."""
        ...


@runtime_checkable
class OptionSliceLike(OptionChainLike, Protocol):
    """Single-expiry option chain slice."""

    @property
    def slice_expiry(self) -> dt.datetime:
        """The unique expiry of the slice."""
        ...


@dataclass(frozen=True)
class LinearMarketData:
    """Market data used for equity linear models."""

    disc: tuple[Array, Interpolator]
    forward: tuple[Array, Interpolator]


@dataclass(frozen=True)
class OptionMarketData:
    """Option chain data."""

    chain: OptionChainLike
    lm: LinearMarketData


Transform = tuple[
    Callable[[float], float],  # constrain   (flat → readable)
    Callable[[float], float],  # unconstrain (readable → flat)
]


@dataclass(frozen=True, slots=True)
class ModelParams(Mapping[str, Any]):
    """Base class for model parameters."""

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dataclass_fields__)

    def __len__(self) -> int:
        return len(self.__dataclass_fields__)

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None


@dataclass(frozen=True)
class LinearMarketData:
    """Market data used in option calibration."""

    spot: float
    disc: tuple[Array, Interpolator]
    forward: tuple[Array, Interpolator]


@dataclass(frozen=True)
class ValuationContext:
    """Market data used in valuation."""

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
    """European option data."""

    expiry: Array
    strike: Array
    option_type: Array


@dataclass(frozen=True)
class Product(ABC):
    """Base class for all products."""


@dataclass(frozen=True)
class Position:
    product: Any
    quantity: float


@dataclass(frozen=True)
class Portfolio:
    positions: list[Position]


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
        mkt (LinearMarketData): The market context (spot, rates, etc.).
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
            LinearMarketData,
        )


        def calibrate_xyz(chain: OptionChain, mkt: LinearMarketData, config: CalibrationSettings) -> CalibrationResult:
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
        chain: OptionChainLike,
        mkt: LinearMarketData,
        solver: Solver,
        settings: CalibrationSettings,
        kwargs=None,
    ) -> CalibrationResult: ...


# ---------------------------------------------------------------------------------------------------------------------
# Risk factors protocols and abstract classes
# ---------------------------------------------------------------------------------------------------------------------


@runtime_checkable
class VolSurfaceEncoder(Protocol):
    """Used to convert a vol surfaces to a reduced set of risk factors.

    For instance, VG, Melz, PCA factor, functional PCA, Heston parameters).
    """

    def encode(self, ivs: VolSurface, kwargs: None) -> Array: ...
    def decode(self, rf: Array, kwargs: None) -> VolSurface: ...


@runtime_checkable
class ShiftType(Protocol):
    """E.g., relative, absolute, and log-shifts."""

    def __call__(self, x: Array) -> Array: ...


class RiskFactorProcess(ABC):
    """E.g., GARCH, Neural-SDE, etc."""

    params: list[Array]  # needs a calibrated model

    def simulate(self, dim: tuple[int, int]) -> Array:
        raise NotImplementedError

    def sample_std_error(self, x: Array) -> Array:
        """Sample the standard error of the process at x."""
        raise NotImplementedError

    def actualise(self, x: Array) -> Array:
        """Scale by the current volatiltiy."""
        raise NotImplementedError

    def simulate_hist(x: Array) -> Array:
        """Poduce historical scenarios for the process."""
        raise NotImplementedError


@runtime_checkable
class RiskFactorProcessFitter(Protocol):
    """Protocol for fitting risk factor processes.

    Separates fitting logic from process dynamics, following JAX ecosystem style.
    """

    def fit(self, ts: Array, **kwargs) -> RiskFactorProcess:
        """Fit a risk factor process to historical data ts."""
        ...


class MultivariateProcess:
    """E.g., Univariate GARCH + Copula. Used in Monte Carlo engines"""

    marginals: list[RiskFactorProcess]
    corr: None  # "Copula"


class ScenarioGenerator(Protocol):
    """Abstract interface for scenario generators."""

    def simulate(self, **kwargs) -> list[VolSurface]:
        """Simulate scenarios from a base volatility surface.

        Args:
            **kwargs: Generator-specific parameters (e.g., ts for historical or n_scenarios for Monte Carlo).

        Returns:
            A list of VolSurface scenarios.
        """
        ...


@dataclass(frozen=True)
class VolRiskFactor:
    key: str
    encoder: VolSurfaceEncoder
    shift: ShiftType
    process: RiskFactorProcess


class HistScenarioGenerator(ScenarioGenerator):
    """Historical scenario generator using filtered historical simulation."""

    def __init__(
        self,
        risk_factor: VolRiskFactor,
    ):
        # ...existing code...
        self.rf = risk_factor

    def simulate(self, base: VolSurface, ts: Array, **kwargs) -> list[VolSurface]:
        """Simulate scenarios based on historical vol surface time series."""
        latent_hist = self.rf.encoder.encode(ts, **kwargs)
        shocks = self.rf.process.simulate_hist(latent_hist)
        base_latent = self.rf.encoder.encode(base, **kwargs)
        scenarios_latent = base_latent + shocks
        return [self.rf.encoder.decode(lat, **kwargs) for lat in scenarios_latent]


class MonteCarloScenarioGenerator(ScenarioGenerator):
    """Monte Carlo scenario generator for risk factor processes."""

    def __init__(self):
        raise NotImplementedError


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


class ProxyStrategy(Protocol):
    def assign_risk_factor(self, name: str) -> dict: ...


class VarEngineBase(ABC):
    """Abstract base class for VaR engines that stores PnL history.

    Attributes:
        portfolio: The portfolio to analyze.
        strategy: ValuationStrategy for pricing under scenarios.
        scenario_generator: ScenarioGenerator for risk-factor scenarios.
        proxy_assigner: ProxyStrategy for transforming raw risk-factor series.
        _pnl: Stored PnL array for the last evaluation.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        valuation_strategy: ValuationStrategy,
        scenario_generator: ScenarioGenerator,
        proxy_assigner: ProxyStrategy,
    ) -> None:
        """Initialize dependencies and prepare PnL storage."""
        self.portfolio = portfolio
        self.scenario_generator = scenario_generator
        self.proxy_assigner = proxy_assigner
        self.strategy = valuation_strategy
        self._pnl: Array | None = None

    def collect_risk_factors(self, context: ValuationContext) -> Array:
        """Collect and return historical risk-factor data for the portfolio."""
        raise NotImplementedError

    def simulate_scenarios(self, rf_ts: Array, **kwargs: Any) -> list[VolSurface]:
        """Simulate risk-factor scenarios using the injected generator."""
        return self.scenario_generator.simulate(rf_ts, **kwargs)

    def evaluate(self, scenarios: Sequence[VolSurface]) -> Array:
        """Evaluate portfolio PnL under each scenario and store results."""
        pnl = jnp.array(
            [
                sum(self.strategy.evaluate(pos.product, scenario) * pos.quantity for pos in self.portfolio.positions)
                for scenario in scenarios
            ]
        )
        self._pnl = pnl
        return pnl

    def run(self, context: ValuationContext, **generate_kwargs: Any) -> tuple[float, float]:
        """Execute full VaR pipeline and store PnL."""
        raw_ts = self.collect_risk_factors(context)
        proxied_ts = self.proxy_assigner.assign_risk_factor(raw_ts)
        scenarios = self.simulate_scenarios(proxied_ts, **generate_kwargs)
        self.evaluate(scenarios)
