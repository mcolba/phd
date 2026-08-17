from __future__ import annotations

import datetime as dt
import logging
import shelve
from pathlib import Path

import pyarrow.dataset as ds
import yaml
from tqdm import tqdm

from vol_risk.calibration.config.mixture_config import MixtureCalibIndexConfig
from vol_risk.calibration.logging_utils import calibration_log_context, configure_calibration_logging
from vol_risk.calibration.mixture_pipeline import (
    run_mixture_pipeline,
)
from vol_risk.market_data.opt_chain_loaders import make_optionmetrics_chain

PROJECT_ROOT = Path(__file__).resolve().parents[2]

with (Path(__file__).parents[1] / "config.yaml").open("r") as stream:
    scripts_config = yaml.safe_load(stream)["config"]

INPUT_PATH = Path(scripts_config["opt_data_dir"])
OUTPUT_PATH = PROJECT_ROOT / "data" / "derived" / "mixture"

log = logging.getLogger(__name__)

# ================================================== Configuration =================================================== #

TICKER = "SPX"
RUN_ID = "main"
OVEWRITE_EXISTING = True
LOG_FILE_NAME = f"{Path(__file__).stem}_{TICKER}_{RUN_ID}.log"
FILE_LOG_LEVEL = logging.WARNING
STREAM_LOG_LEVEL = logging.WARNING
CALIB_CONFIG = MixtureCalibIndexConfig
# # ================================================================================================================== #


def main() -> None:
    configure_calibration_logging(
        log_file_name=LOG_FILE_NAME,
        file_level=FILE_LOG_LEVEL,
        stream_level=STREAM_LOG_LEVEL,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    log.info("Opening parquet dataset at %s", INPUT_PATH)
    dataset = ds.dataset(str(INPUT_PATH), format="parquet", partitioning="hive")

    dates_str = sorted({ds.get_partition_keys(frag.partition_expression)["date"] for frag in dataset.get_fragments()})
    log.info("Dates to calibrate: %d", len(dates_str))

    with shelve.open(str(OUTPUT_PATH)) as db:
        for t in tqdm(dates_str, desc="Calibrating", unit="date", dynamic_ncols=True):
            key = f"{TICKER}_{dt.datetime.fromisoformat(t).strftime(r'%Y%m%d')}"
            with calibration_log_context(key):
                if key in db and not OVEWRITE_EXISTING:
                    log.info("Results for %s already exist, skipping", key)
                    continue

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
