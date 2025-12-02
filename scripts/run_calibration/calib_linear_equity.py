import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from vol_risk.models.linear import LinearEquityMarket, make_raw_disc_curve
from vol_risk.protocols import Actual365Fixed, OptionChain

logger = logging.getLogger(__name__)

input_data_path = Path(__file__).resolve().parents[1] / "data" / "test" / "CBOE_EOD_2023-08-25.csv"

# Import data
df = (
    pd.read_csv(input_data_path)
    .rename(
        columns={
            "quote_date": "anchor",
            "expiration": "expiry",
            "trade_volume": "volume",
            "bid_1545": "bid",
            "ask_1545": "ask",
        }
    )
    .assign(
        spot=lambda x: 0.5 * (x.underlying_bid_1545 + x.underlying_ask_1545),
        anchor=lambda x: pd.to_datetime(x.anchor, format="%Y-%m-%d", errors="raise"),
        expiry=lambda x: pd.to_datetime(x.expiry, format="%Y-%m-%d", errors="raise"),
        mid=lambda x: 0.5 * (x.bid + x.ask),
    )
)

# Filter data
idx = (df.underlying_symbol == "SPY") & (df["anchor"] == df["expiry"])
spot = (
    df.loc[df.underlying_symbol == "SPY", ["underlying_bid_1545", "underlying_ask_1545"]]
    .mean(axis=1)
    .drop_duplicates()
    .to_numpy()
)

df_spx = df[
    ((df.underlying_symbol == "^SPX") & (df["root"] == "SPX"))
    & (df["open_interest"].notna() & (df["open_interest"] > 10))
    & (df["bid"] > 0)
    & (df["mid"] > 0.10)
    & (df["ask"] > df["bid"])
    & ((df["ask"] - df["bid"]) / df["mid"] < 0.2)
    & (df["expiry"] - df["anchor"] >= pd.Timedelta(days=50))
    # & ((df["strike"] / df["spot"]).between(0.5, 1.05))
]

spx_chain = OptionChain(df_spx, Actual365Fixed)


def put_call_df(opt: OptionChain) -> pd.DataFrame:
    pt = opt.df.pivot_table(index=["strike", "spot"], columns="option_type", values=["mid", "bid", "ask"]).pipe(
        lambda x: x.loc[x.notna().all(axis=1), :]
    )

    g_mid = pt.loc[:, ("mid", "C")] - pt.loc[:, ("mid", "P")]
    g_max = pt.loc[:, ("bid", "C")] - pt.loc[:, ("ask", "P")]
    g_min = pt.loc[:, ("ask", "C")] - pt.loc[:, ("bid", "P")]

    return (
        pd.DataFrame(index=pt.index)
        .assign(
            g_mid=g_mid,
            g_max=g_max,
            g_min=g_min,
        )
        .sort_values(by="strike")
        .reset_index()
    )


def calib_linear_equity_market(opt: OptionChain) -> LinearEquityMarket:
    out = []
    for t, sl in opt:
        tau = sl.tau[0]

        pc_df = put_call_df(sl)

        if pc_df.shape[0] < 10:
            msg = f"Fitted line for maturity {t} has less than 10 rows, skipping."
            logger.warning(msg)
            continue

        # Calculate P - C and perform linear regression against strike (K)
        K = pc_df["strike"].to_numpy().reshape(-1, 1)
        y = pc_df["g_mid"].to_numpy()

        # Fit linear regression
        reg = LinearRegression().fit(-K, y)
        alpha = reg.intercept_
        beta = reg.coef_[0]

        # check if fitted line is within bid-ask bounds
        fitted = reg.predict(-K)
        in_bid_ask = np.all((fitted > pc_df["g_max"]) & (fitted > pc_df["g_max"]))

        if not in_bid_ask:
            msg = f"Fitted line for maturity {t} is not within the put-call bid-ask bounds"
            logger.warning(msg)

        out.append(
            {
                "tau": tau,
                "alpha": alpha,
                "beta": beta,
                "in_bid_ask": in_bid_ask,
            }
        )

    reg_df = pd.DataFrame(out)

    stats = {"all_in_bid_ask": all(reg_df["in_bid_ask"])}

    spot = opt.spot
    tau = reg_df["tau"]
    r = -np.log(reg_df["beta"]) / tau
    q = -np.log(reg_df["alpha"] / spot) / tau

    model = LinearEquityMarket(
        spot=float(spot),
        disc_curve=make_raw_disc_curve(tau=tau, r=r),
        cont_dvd_curve=make_raw_disc_curve(tau=tau, r=q),
    )

    return model, stats


out = []
for t, sl in spx_chain:
    _df = put_call_df(sl)
    tau = (t - spx_chain._df.anchor[0]).days / 365.0
    spot = spx_chain._df.spot.iloc[0]

    if _df.shape[0] < 10:
        print(f"less than 10 raws for maturity {t}, skipping")
        continue

    # Calculate P - C and perform linear regression against strike (K)
    K = _df["strike"].values.reshape(-1, 1)
    y = (_df["g_mid"]).values

    # Fit linear regression: diff = alpha + beta * K
    reg = LinearRegression().fit(-K, y)
    # reg = HuberRegressor(epsilon=100).fit(-K, y)
    alpha = reg.intercept_
    beta = reg.coef_[0]
    fitted = reg.predict(-K)

    # Plot results

    # plt.figure(figsize=(8, 5))
    # plt.scatter(_df["strike"], y, label="C - P", color="blue")
    # plt.plot(_df["strike"], fitted, label=f"Fit: alpha={alpha:.2f}, beta={beta:.4f}", color="red")
    # fwd = np.mean((_df["g_mid"]) / beta + _df["strike"])
    # plt.axvline(fwd, color="green", linestyle="--", label=f"Forward: {fwd:.2f}")
    # # Add band between g_min and g_max
    # plt.fill_between(_df["strike"], _df["g_min"], _df["g_max"], color="gray", alpha=0.2, label="Arb-free band")
    # plt.xlabel("Strike (K)")
    # plt.ylabel("C - P")
    # plt.title(f"Call-Put Difference vs Strike with Linear Regression (tau={tau:.3f} years)")
    # plt.legend()
    # plt.show()

    out.append(
        {
            "date": t,
            "N": _df.shape[0],
            "tau": tau,
            "spot": spot,
            "fwd": np.mean((_df["g_mid"]) / beta + _df["strike"]),
            "alpha": alpha,
            "disc": beta,
            "r": -np.log(beta) / tau,
            "q": -np.log(alpha / spot) / tau,
        }
    )

lm = pd.DataFrame(out)


model, _ = calib_linear_equity_market(spx_chain)

zero_curve = model.zero_rate(np.linspace(0.0001, lm["tau"].max(), 100))
dvd_curve = model.zero_dvd_yield(np.linspace(0.0001, lm["tau"].max(), 100))

# Plot r on primary axis and fwd on secondary axis

fig, ax = plt.subplots(figsize=(10, 6))
ax2 = ax.twinx()

# Plot calibrated rates on primary axis
lm.plot(x="tau", y="r", ax=ax, marker="o", label="r (calibrated)", color="blue")
lm.plot(x="tau", y="q", ax=ax, marker="o", label="q (calibrated)", color="orange")

# Plot interpolated curves on primary axis
tau_grid = np.linspace(0.0001, lm["tau"].max(), 100)
ax.plot(tau_grid, zero_curve, label="r (interpolated)", linestyle="--", color="blue", alpha=0.7)
ax.plot(tau_grid, dvd_curve, label="q (interpolated)", linestyle="--", color="orange", alpha=0.7)

# Plot forward on secondary axis
lm.plot(x="tau", y="fwd", ax=ax2, marker="o", label="fwd (calibrated)", color="green")

ax.set_xlabel("Time to Maturity (tau)")
ax.set_ylabel("Risk-Free Rate (r) / Dividend Yield (q)")
ax2.set_ylabel("Forward Price (fwd)")
ax.set_title("Calibrated Rate and Forward Curve")
ax.legend(loc="upper left")
ax2.legend(loc="upper right")
ax.grid(True)
plt.tight_layout()
plt.show()

print("Done")
