"""Calibration script for Linear Equity Market model using index option data."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vol_risk.models.linear import calib_linear_equity_market
from vol_risk.protocols import Actual365Fixed, OptionChain

logger = logging.getLogger(__name__)
plt.style.use("ggplot")

input_data_path = Path(__file__).resolve().parents[2] / "data" / "test" / "CBOE_EOD_2023-08-25.csv"

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
]

spx_chain = OptionChain(df_spx, Actual365Fixed)

n = np.unique(spx_chain.expiry).size
n_col = 3
n_row = n // n_col + int(n % n_col > 0)
fig, axes = plt.subplots(n_row, n_col, sharex=True, sharey=True, figsize=(10, 15))

model, stats = calib_linear_equity_market(spx_chain, axes=axes.flatten())
plt.tight_layout()
plt.show()
plt.close(fig)

# plot model output
tau_granular = np.linspace(0.0001, spx_chain.tau.max() + 0.25, 100)
zero_curve_granular = model.zero_rate(tau_granular)
dvd_curve_granular = model.zero_dvd_yield(tau_granular)
fwd_price_granular = model.fwd(tau_granular)

fig, ax = plt.subplots(figsize=(10, 6))
ax2 = ax.twinx()

ax.plot(tau_granular, zero_curve_granular, label="r", color="blue", alpha=0.7)
ax.plot(tau_granular, dvd_curve_granular, label="q", color="orange", alpha=0.7)
ax2.plot(tau_granular, fwd_price_granular, label="fwd price", color="green")

# Plot calibrated points
mkt_tau = np.array([x["tau"] for x in stats.values()])
reg_params = np.array([list(x["coeff"]) for x in stats.values()])
excluded_idx = np.array([x["excluded"] for x in stats.values()])

spot = spx_chain.spot
r_mkt = -np.log(reg_params[:, 1]) / mkt_tau
q_mkt = -np.log(reg_params[:, 0] / spot) / mkt_tau

ax.scatter(mkt_tau[~excluded_idx], r_mkt[~excluded_idx], color="blue", marker="o")
ax.scatter(mkt_tau[~excluded_idx], q_mkt[~excluded_idx], color="orange", marker="o")

if any(excluded_idx):
    ax.scatter(mkt_tau[excluded_idx], r_mkt[excluded_idx], color="blue", marker="x")
    ax.scatter(mkt_tau[excluded_idx], q_mkt[excluded_idx], color="orange", marker="x")

ax.set_xlabel("Time to Maturity")
ax.set_ylabel("Implied Rates")
ax2.set_ylabel("Forward Price")
ax.legend(loc="upper left")
ax2.legend(loc="upper right")
# ax.grid(True)
plt.tight_layout()
plt.show()

pd.DataFrame(stats).T
