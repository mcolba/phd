"""Generate QuantLib Variance Gamma fixtures for reduced-VGSI tests."""

from __future__ import annotations

import math
from itertools import product
from pathlib import Path

import pandas as pd
import QuantLib as ql

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "tests" / "data" / "vg_prices_quantlib.csv"


def _price_vg_call(
    *,
    valuation_date: ql.Date,
    spot: float,
    strike: float,
    tau: float,
    r: float,
    q: float,
    sigma: float,
    nu: float,
    theta: float,
) -> tuple[float, float]:
    day_count = ql.Actual365Fixed()
    ql.Settings.instance().evaluationDate = valuation_date

    maturity_days = max(1, round(tau * 365.0))
    maturity = valuation_date + ql.Period(maturity_days, ql.Days)
    tau_eff = maturity_days / 365.0

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    r_handle = ql.YieldTermStructureHandle(ql.FlatForward(valuation_date, r, day_count))
    q_handle = ql.YieldTermStructureHandle(ql.FlatForward(valuation_date, q, day_count))

    process = ql.VarianceGammaProcess(
        spot_handle,
        q_handle,
        r_handle,
        sigma,
        nu,
        theta,
    )

    engine = ql.VarianceGammaEngine(process)
    exercise = ql.EuropeanExercise(maturity)

    # QuantLib VG compensator:
    base = 1.0 - theta * nu - 0.5 * sigma * sigma * nu
    if base <= 0.0:
        raise ValueError("Invalid VG parameters: martingale compensator is undefined.")

    omega = math.log(base) / nu

    risk_free_discount = math.exp(-r * tau_eff)
    dividend_discount = math.exp(-q * tau_eff)

    zero_clock_call_intrinsic = max(
        spot * math.exp(omega * tau_eff) * dividend_discount - strike * risk_free_discount,
        0.0,
    )

    zero_clock_put_intrinsic = max(
        strike * risk_free_discount - spot * math.exp(omega * tau_eff) * dividend_discount,
        0.0,
    )

    # Price the payoff whose zero-clock intrinsic value is zero.
    # This avoids multiplying a nonzero intrinsic payoff by the singular
    # gamma density near x=0.
    if zero_clock_call_intrinsic > zero_clock_put_intrinsic and tau < 0.25:
        put = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Put, strike),
            exercise,
        )
        put.setPricingEngine(engine)
        put_price = float(put.NPV())

        call_price = put_price + spot * dividend_discount - strike * risk_free_discount
    else:
        call = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql.Option.Call, strike),
            exercise,
        )
        call.setPricingEngine(engine)
        call_price = float(call.NPV())

    return call_price, tau_eff


def main() -> None:
    valuation_date = ql.Date(1, 1, 2025)
    spot = 100.0
    r = 0.05
    q = 0.02
    maturities = [1 / 12, 1]
    moneyness_grid = [0.70, 1.00, 1.30]
    parameter_sets = [
        {"sigma": 0.12, "nu": 0.05, "theta": -0.05},
        {"sigma": 0.20, "nu": 0.15, "theta": 0.00},
        {"sigma": 0.30, "nu": 0.25, "theta": -0.15},
        {"sigma": 0.18, "nu": 0.40, "theta": 0.10},
        {"sigma": 0.35, "nu": 0.60, "theta": -0.25},
    ]

    rows: list[dict[str, float | str]] = []
    for params, tau, moneyness in product(parameter_sets, maturities, moneyness_grid):
        strike = spot * moneyness
        price, tau_eff = _price_vg_call(
            valuation_date=valuation_date,
            spot=spot,
            strike=strike,
            tau=tau,
            r=r,
            q=q,
            sigma=float(params["sigma"]),
            nu=float(params["nu"]),
            theta=float(params["theta"]),
        )

        lb = spot - strike * math.exp(-r * tau_eff)
        if price < 1e-4 or price < lb + 1e-8:
            continue

        rows.append(
            {
                "spot": spot,
                "strike": strike,
                "tau": tau_eff,
                "r": r,
                "q": q,
                "sigma": float(params["sigma"]),
                "nu": float(params["nu"]),
                "theta": float(params["theta"]),
                "quantlib_call_price": price,
            }
        )

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
