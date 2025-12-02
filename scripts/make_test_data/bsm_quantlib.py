from pathlib import Path

import numpy as np
import pandas as pd
import QuantLib as ql

output_path = Path(__file__).resolve().parents[2] / "data" / "test" / "vanilla_opt.csv"

# Fixed evaluation date for reproducibility
valuation_date = ql.Date(1, 1, 2025)
ql.Settings.instance().evaluationDate = valuation_date
dc = ql.Actual365Fixed()

t_1d = valuation_date + ql.Period(1, ql.Days)
t_1y = valuation_date + ql.Period(365, ql.Days)
t_5y = valuation_date + ql.Period(365 * 5, ql.Days)

spot = 100.0
r = 0.02
r_curve = ql.FlatForward(valuation_date, r, dc)
r_handle = ql.YieldTermStructureHandle(r_curve)

cases = {
    "c_atm": {"type": "C", "K": 100.0, "T": t_1y, "sigma": 0.20, "q": 0.02},
    "c_itm": {"type": "C", "K": 80.0, "T": t_1y, "sigma": 0.20, "q": 0.02},
    "c_otm": {"type": "C", "K": 120.0, "T": t_1y, "sigma": 0.20, "q": 0.02},
    "c_1d": {"type": "C", "K": 100.0, "T": t_1d, "sigma": 0.20, "q": 0.02},
    "c_5y": {"type": "C", "K": 100.0, "T": t_5y, "sigma": 0.20, "q": 0.02},
    "c_high_vol": {"type": "C", "K": 100.0, "T": t_1y, "sigma": 1.00, "q": 0.02},
    "c_no_dvd": {"type": "C", "K": 100.0, "T": t_1y, "sigma": 0.20, "q": 0.0},
    "p_atm": {"type": "P", "K": 100.0, "T": t_1y, "sigma": 0.20, "q": 0.02},
    "p_itm": {"type": "P", "K": 120.0, "T": t_1y, "sigma": 0.20, "q": 0.02},
    "p_otm": {"type": "P", "K": 80.0, "T": t_1y, "sigma": 0.20, "q": 0.02},
    "p_1d": {"type": "P", "K": 100.0, "T": t_1d, "sigma": 0.20, "q": 0.02},
    "p_5y": {"type": "P", "K": 100.0, "T": t_5y, "sigma": 0.20, "q": 0.02},
    "p_high_vol": {"type": "P", "K": 100.0, "T": t_1y, "sigma": 1.00, "q": 0.02},
    "p_no_dvd": {"type": "P", "K": 100.0, "T": t_1y, "sigma": 0.20, "q": 0.0},
}

rows = []
for k, v in cases.items():
    S = spot
    K = v["K"]
    T = v["T"]
    q = v["q"]
    sigma = v["sigma"]
    is_call = v["type"] == "C"

    tau = dc.yearFraction(valuation_date, T)

    # Build Black-Scholes components
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
    vol_ts = ql.BlackConstantVol(valuation_date, ql.NullCalendar(), sigma, dc)
    vol_handle = ql.BlackVolTermStructureHandle(vol_ts)

    q_curve = ql.FlatForward(valuation_date, q, dc)
    q_handle = ql.YieldTermStructureHandle(q_curve)

    process = ql.BlackScholesMertonProcess(spot_handle, q_handle, r_handle, vol_handle)
    engine = ql.AnalyticEuropeanEngine(process)

    payoff = ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, K)
    exercise = ql.EuropeanExercise(T)
    opt = ql.VanillaOption(payoff, exercise)
    opt.setPricingEngine(engine)

    # Discount factor and forward
    D = float(r_curve.discount(T))
    F = S * np.exp((r - q) * tau)

    price = float(opt.NPV())
    delta = float(opt.delta())
    gamma = float(opt.gamma())
    vega = float(opt.vega()) / 100.0  # per 1.0 change in vol

    rows.append(
        {
            "id": k,
            "type": v["type"],
            "S": float(S),
            "F": float(F),
            "K": float(K),
            "tau": float(tau),
            "r": float(r),
            "q": float(q),
            "sigma": float(sigma),
            "DF": float(D),
            "price": float(price),
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
        }
    )


# Write CSV with pandas
df = pd.DataFrame(rows)
df.to_csv(output_path, index=False)
