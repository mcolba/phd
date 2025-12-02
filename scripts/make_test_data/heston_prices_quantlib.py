"""Generates Heston option price fixtures for unit tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import QuantLib as ql

output_path = Path(__file__).resolve().parents[2] / "data" / "test" / "heston_prices_quantlib.csv"

valuation_date = ql.Date(25, ql.November, 2025)
ql.Settings.instance().evaluationDate = valuation_date

dc = ql.Actual365Fixed()

spot = 100.0
r = 0.05
q = 0.02

kappa = 1.5
v_inf = 0.05
vol_of_vol = 0.8
rho = -0.5
v0 = 0.04

strikes = np.linspace(80.0, 120.0, 5)
maturity_date = [
    valuation_date + ql.Period("1M"),
    valuation_date + ql.Period("3M"),
    valuation_date + ql.Period("12M"),
    valuation_date + ql.Period("24M"),
]
year_fraction = np.array([dc.yearFraction(valuation_date, t) for t in maturity_date])

spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
r_curve = ql.FlatForward(valuation_date, r, dc)
q_curve = ql.FlatForward(valuation_date, q, dc)
r_handle = ql.YieldTermStructureHandle(r_curve)
q_handle = ql.YieldTermStructureHandle(q_curve)

process = ql.HestonProcess(r_handle, q_handle, spot_handle, v0, kappa, v_inf, vol_of_vol, rho)
model = ql.HestonModel(process)

# Using adaptive Gauss-Lobatto integration and Gatheral's version of complex logarithm
rel_tol = 1e-10
max_evaluations = 1_000_000
engine = ql.AnalyticHestonEngine(model, rel_tol, max_evaluations)

rows = []
for tau, t in zip(year_fraction, maturity_date, strict=True):
    exercise = ql.EuropeanExercise(t)
    for K in strikes:
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, float(K))
        option = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(engine)
        price = float(option.NPV())
        rows.append(
            {
                "valuation_date": valuation_date.ISO(),
                "maturity_date": t.ISO(),
                "S": float(spot),
                "K": float(K),
                "tau": float(tau),
                "r": float(r),
                "q": float(q),
                "price": price,
                "kappa": float(kappa),
                "v_inf": float(v_inf),
                "vol_of_vol": float(vol_of_vol),
                "rho": float(rho),
                "v0": float(v0),
            }
        )

df = pd.DataFrame(rows)
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
