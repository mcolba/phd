from __future__ import annotations

import datetime as dt
import logging
import shelve
from pathlib import Path

import pyarrow.dataset as ds
from tqdm import tqdm

from vol_risk.calibration.data_loaders import make_optionmetrics_chain
from vol_risk.calibration.mixture_pipeline import (
    ChainCutoff,
    ChainFilter,
    MixtureCalibConfig,
    run_mixture_pipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_FILE_PATH = PROJECT_ROOT / "results" / "logging" / "spx_ivs_calib.log"

INPUT_PATH = Path(r"D:\option_metrics\parquet")
OUTPUT_PATH = PROJECT_ROOT / "data" / "derived" / "mixture"

file_handler = logging.FileHandler(LOG_FILE_PATH, mode="w")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)


class TqdmLoggingHandler(logging.Handler):
    """Write log records via tqdm so they do not corrupt the progress bar."""

    def emit(self, record: logging.LogRecord) -> None:
        tqdm.write(self.format(record))
        self.flush()


stream_handler = TqdmLoggingHandler()
stream_handler.setLevel(logging.WARNING)
stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        file_handler,
        stream_handler,
    ],
)
log = logging.getLogger(__name__)

TICKER = "SPX"

epsilon = 1e-7
cutoff_cfg = ChainCutoff("delta", (epsilon, 1.0 - epsilon))
CALIB_CONFIG = MixtureCalibConfig(
    n_components=3,
    lw_type="vega",
    transform_method="totvar_simplex",
    repair_arbitrage=True,
    filters=ChainFilter(
        oi_min=50,
        bid_min=0.01,
        mid_min=0.02,
        rel_bid_ask_max=1.0,
        min_k_per_slice=10,
        min_ttm=10,
        cutoff=cutoff_cfg,
    ),
)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    log.info("Opening parquet dataset at %s", INPUT_PATH)
    dataset = ds.dataset(str(INPUT_PATH), format="parquet", partitioning="hive")

    dates_str = sorted(ds.get_partition_keys(frag.partition_expression)["date"] for frag in dataset.get_fragments())
    log.info("Dates to calibrate: %d", len(dates_str))

    with shelve.open(str(OUTPUT_PATH)) as db:
        for t in tqdm(dates_str, desc="Calibrating", unit="date", dynamic_ncols=True):
            key = f"{TICKER}_{dt.datetime.fromisoformat(t).strftime(r'%Y%m%d')}"
            log.info("Calibrating %s", key)

            try:
                df = dataset.to_table(
                    filter=ds.field("date") == t,
                ).to_pandas()

                if df.empty:
                    log.warning("No data for %s, skipping", key)
                    continue

                chain = make_optionmetrics_chain(df)
                result = run_mixture_pipeline(chain, CALIB_CONFIG)
                db[key] = {
                    "date": dt.datetime.fromisoformat(t).date(),
                    "params": result.params,
                    "stats": result.stats,
                }

            except Exception:
                log.exception("Failed to calibrate %s", key)

    log.info("Done. Results saved to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
