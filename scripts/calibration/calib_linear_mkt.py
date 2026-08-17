"""Calibrate daily SPX linear-equity models and persist them in SQLite."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import asdict
from pathlib import Path

import pyarrow.dataset as ds
import yaml
from tqdm import tqdm

from vol_risk.calibration.config.linear_mkt_config import LinearModelCalibConfig
from vol_risk.calibration.linear_market import run_linear_model_pipeline
from vol_risk.calibration.logging_utils import calibration_log_context, configure_calibration_logging
from vol_risk.dao.linear_market import LinearModelStore
from vol_risk.market_data.opt_chain_loaders import make_optionmetrics_chain

SCRIPTS_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

TICKER = "SPX"
RUN_ID = "main"
OVERWRITE_EXISTING = True
FILE_LOG_LEVEL = logging.WARNING
STREAM_LOG_LEVEL = logging.WARNING
LOG_FILE_NAME = f"{Path(__file__).stem}_{TICKER}_{RUN_ID}.log"
CALIB_CONFIG = LinearModelCalibConfig()

log = logging.getLogger(__name__)


def _input_path(config_path: Path) -> Path:
    """Read the OptionMetrics dataset path from the scripts configuration."""
    with config_path.open("r", encoding="utf-8") as stream:
        raw_config = yaml.safe_load(stream)
    try:
        return Path(raw_config["config"]["opt_data_dir"])
    except (KeyError, TypeError) as error:
        msg = f"Configuration {config_path} must define config.opt_data_dir"
        raise ValueError(msg) from error


def _partition_dates(dataset: ds.Dataset) -> list[str]:
    """Return sorted date partition values from a Hive dataset."""
    dates = {
        str(keys["date"])
        for fragment in dataset.get_fragments()
        if "date" in (keys := ds.get_partition_keys(fragment.partition_expression))
    }
    if not dates:
        msg = "Option dataset contains no Hive date partitions"
        raise ValueError(msg)
    return sorted(dates)


def main() -> None:
    """Calibrate every available date and persist parameters and statistics."""
    configure_calibration_logging(
        log_file_name=LOG_FILE_NAME,
        file_level=FILE_LOG_LEVEL,
        stream_level=STREAM_LOG_LEVEL,
    )
    input_path = _input_path(SCRIPTS_CONFIG_PATH)
    log.info("Opening parquet dataset at %s", input_path)
    dataset = ds.dataset(str(input_path), format="parquet", partitioning="hive")
    dates = _partition_dates(dataset)
    log.info("Dates to calibrate: %d", len(dates))

    store = LinearModelStore()
    store.initialize()
    config_metadata = asdict(CALIB_CONFIG)

    for partition_date in tqdm(dates, desc="Calibrating", unit="date", dynamic_ncols=True):
        calibration_date = dt.datetime.fromisoformat(partition_date).date()
        artifact_key = f"{TICKER}_{calibration_date:%Y%m%d}_{RUN_ID}"
        with calibration_log_context(artifact_key):
            if store.contains(TICKER, calibration_date, RUN_ID) and not OVERWRITE_EXISTING:
                log.info("Results for %s already exist, skipping", artifact_key)
                continue

            log.info("Calibrating %s", artifact_key)
            try:
                frame = dataset.to_table(filter=ds.field("date") == partition_date).to_pandas()
                if frame.empty:
                    log.warning("No data for %s, skipping", artifact_key)
                    continue

                chain = make_optionmetrics_chain(frame)
                result = run_linear_model_pipeline(chain, CALIB_CONFIG)
                store.save(
                    ticker=TICKER,
                    calibration_date=calibration_date,
                    run_id=RUN_ID,
                    params=result.params,
                    stats=result.stats,
                    config=config_metadata,
                    overwrite=OVERWRITE_EXISTING,
                )
            except Exception:
                log.exception("Failed to calibrate %s", artifact_key)

    log.info("Done. Results saved to %s (table: linear_market)")


if __name__ == "__main__":
    main()
