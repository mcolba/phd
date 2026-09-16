# Linear equity market model

## Purpose and Scope

The model extracts an option-implied discount curve and a continuous implied forward curve from European equity
call and put quotes.

| Main public API | Module | Responsibility |
|---|---|---|
| [`run_linear_model_pipeline`][pipeline] | `calibration.linear_market` | Filter the chain and run calibration. |
| [`calib_linear_equity_market`][linear] | `models.linear` | Fit the linear equity model to an option chain. |
| [`LinearEquityMarket`][linear] | `models.linear` | Evaluate discount factors, forwards, and implied equity yields. |
| [`make_raw_interpolator`][linear], [`make_raw_disc_curve`][linear] | `models.linear` | Build rate and yield curves. |
| [`LinearModelStore`][store] | `dao.linear_market` | Save and load calibration artifacts. |

[pipeline]: ../../vol_risk/calibration/linear_market.py
[linear]: ../../vol_risk/models/linear.py
[store]: ../../vol_risk/dao/linear_market.py

## Methodology for European Options

### Theoretical Framework

For a European call $C(K,\tau)$ and put $P(K,\tau)$ with the same underlying, strike $K$, and expiry $\tau$,
put–call parity is

$$
C(K,\tau)-P(K,\tau)=D(\tau)(F(\tau) - K),
$$

where $D(\tau)$ is the discount factor and $F(\tau)$ the equity forward price. Define the zero rate $r(\tau)$
and implied equity yield $q(\tau)$ by

$$
D(\tau)=e^{-r(\tau)\tau}, \quad
F(\tau)=D(\tau)^{-1}S_0e^{-q(\tau)\tau},
$$

The implied equity yield can reflect dividends and equity repo effects; the regression does not identify these
components separately. Put–call parity then becomes

$$
C(K,\tau)-P(K,\tau)=S_0e^{-q(\tau)\tau} - e^{-r(\tau)\tau}K.
$$

At each expiry, fit an ordinary least squares regression:

$$
g_j=C_j^{\mathrm{mid}}-P_j^{\mathrm{mid}}
    =\alpha_\tau-\beta_\tau K_j+\varepsilon_j,
$$

and recover

$$
r(\tau)=-\log(\beta_\tau)/\tau, \quad
q(\tau)=-\log(\alpha_\tau/S_0)/\tau.
$$

The parity-regression approach is also described by van Binsbergen, Diamond, and Grotteria (2022).

### Calibration Procedure

**Input**: `OptionChain`, `LinearModelCalibConfig`.

**Output**: `LinearModelCalibResult`.

1. **Prepare the chain.** Apply liquidity filters and remove expiry slices with fewer than `min_k_per_slice`
   distinct strikes. The default parameters are

    ```python
    @dataclass(frozen=True)
    class LinearModelCalibConfig:
        liquidity_filter: ChainFilter | None = field(
            default_factory=lambda: ChainFilter(
                oi_min=1,
                bid_min=0.01,
                mid_min=0.02,
                rel_bid_ask_max=2.5,
                min_ttm=10,
            ),
        )
        min_k_per_slice: int = 10
    ```

2. **Pair quotes.** For each expiry, keep only strikes with both call and put mids, and compute

   $$
   g_j=C_j^{\mathrm{mid}}-P_j^{\mathrm{mid}}, \\
   g_j^{\min}=C_j^{\mathrm{bid}}-P_j^{\mathrm{ask}}, \\
   g_j^{\max}=C_j^{\mathrm{ask}}-P_j^{\mathrm{bid}}.
   $$

3. **Fit each expiry independently.** If fewer than five complete pairs remain, exclude the expiry. Otherwise
   fit a `LinearRegression(fit_intercept=True)` with response $g_j$ and predictor $-K_j$. Exclude the expiry
   if either coefficient is nonpositive.

4. **Record diagnostics.** The returned `stats` maps each expiry to `coeff=(alpha, beta)`, `n_obs`, `tau`,
   `excluded`, and a Boolean `in_bid_ask` array comparing fitted $g_j$ with $[g_j^{\min},g_j^{\max}]$.
   A fit outside that range for more than half the paired strikes emits a warning.

5. **Build curves.** Use *raw interpolation* (Hagan and West, 2006). At each accepted expiry $\tau_i$, the
   regression gives a discount factor $\beta_i$ and a prepaid forward value $\alpha_i$. The equity forward
   price is $F_i=\alpha_i/\beta_i$. Define

   $$
   R_i:=-\log \beta_i=r_i\tau_i,\quad
   Q_i:=-\log(\alpha_i/S_0)=q_i\tau_i.
   $$

   Use $R(0)=Q(0)=0$ as interpolation anchors, then interpolate $R(\tau)$ and $Q(\tau)$ linearly between
   consecutive expiries. Their slopes—the instantaneous forward rate and instantaneous implied equity
   yield—are constant within each interval. The instantaneous cost of carry, $d\log F(\tau)/d\tau$, is their
   difference and is also constant within each interval.

   The discount factor and equity forward are

   $$
   D(\tau)=e^{-R(\tau)},\quad
   F(\tau)=S_0 e^{R(\tau)-Q(\tau)}.
   $$

   For $0<\tau<\tau_1$, the first zero rate $r_1$ and implied equity yield $q_1$ apply. Beyond the last
   expiry, the implementation holds the last zero rate $r_n$ and implied equity yield $q_n$ constant.

## Validation (coverage not measured)

- The [synthetic recovery test](../../tests/integration/test_linear_market.py) builds European call and put quotes
from an independent Black formula at two expiries. It runs the default pipeline and checks recovery of the
input rate and implied equity yield nodes. 
- The [curve test](../../tests/unit/test_linear_market_curves.py)
constructs reference curves and verifies interpolation and extrapolation of zero rates, implied equity yields,
discount factors, and forwards.

## Scripts and Examples

- [`scripts/calibration/calib_linear_mkt.py`](../../scripts/calibration/calib_linear_mkt.py) loads daily SPX
  OptionMetrics parquet partitions from the path in `scripts/config.yaml`, calibrates each date, and saves
  parameters and statistics to the SQLite `linear_market` table. It uses the default pipeline config,
  excludes `SPXW` weeklies through the loader when a `symbol` column exists, sets `run_id="main"`, and
  overwrites an existing date/run. The stored configuration includes the effective filter settings.
- [`examples/calib_linear_equity.ipynb`](../../examples/calib_linear_equity.ipynb) loads a CBOE end-of-day
  SPX CSV from `data/test`, applies its own quote filters, fits the model directly, and plots regressions,
  zero rates, and implied equity yields; it displays diagnostics in a table. Its filters differ from the
  script defaults.

## References

- van Binsbergen, J. H., W. F. Diamond, and M. Grotteria (2022), “Risk-Free Interest Rates,”
  *Journal of Financial Economics* 143(1), 1–29.
  [doi:10.1016/j.jfineco.2021.06.012](https://doi.org/10.1016/j.jfineco.2021.06.012).
- Hagan, P. S., and G. West (2006), “Interpolation Methods for Curve Construction,”
  *Applied Mathematical Finance* 13(2), 89–129.
