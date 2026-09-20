"""Calibrate daily SPX mixture volatility surfaces and persist them in SQLite."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import asdict, fields
from pathlib import Path

import pyarrow.dataset as ds
import yaml
from tqdm import tqdm

from vol_risk.calibration.config.mixture_config import MixtureCalibIndexConfig
from vol_risk.calibration.mixture_pipeline import run_mixture_pipeline
from vol_risk.market_data.opt_chain_loaders import make_optionmetrics_chain
from vol_risk.storage.sqlite import DATABASE_PATH, open_initialized_connection
from vol_risk.storage.vol_market import VolMarketStore

SCRIPTS_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

TICKER = "SPX"
CALIBRATION_ID = "idx"
OVERWRITE_EXISTING = True
CALIB_CONFIG = MixtureCalibIndexConfig

log = logging.getLogger(__name__)


def _config_metadata(config: type[MixtureCalibIndexConfig]) -> dict[str, object]:
    """Materialize class-level calibration defaults for JSON metadata."""
    values = {field.name: getattr(config, field.name) for field in fields(config)}
    return asdict(config(**values))


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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    input_path = _input_path(SCRIPTS_CONFIG_PATH)
    log.info("Opening parquet dataset at %s", input_path)
    dataset = ds.dataset(str(input_path), format="parquet", partitioning="hive")
    dates = _partition_dates(dataset)
    log.info("Dates to calibrate: %d", len(dates))

    config_metadata = _config_metadata(CALIB_CONFIG)

    with open_initialized_connection(DATABASE_PATH) as connection:
        store = VolMarketStore(connection)
        for partition_date in tqdm(dates, desc="Calibrating", unit="date", dynamic_ncols=True):
            calibration_date = dt.datetime.fromisoformat(partition_date).date()
            artifact_key = f"{TICKER}_{calibration_date:%Y%m%d}_{CALIBRATION_ID}"
            log.info("Calibrating %s", artifact_key)
            try:
                frame = dataset.to_table(filter=ds.field("date") == partition_date).to_pandas()
                if frame.empty:
                    log.warning("No data for %s, skipping", artifact_key)
                    continue

                chain = make_optionmetrics_chain(frame)
                result = run_mixture_pipeline(chain, CALIB_CONFIG)
                write_operation = store.overwrite if OVERWRITE_EXISTING else store.write
                with connection:
                    write_operation(
                        ticker=TICKER,
                        calibration_date=calibration_date,
                        calibration_id=CALIBRATION_ID,
                        model_id=result.model_id,
                        algorithm_version=result.algorithm_version,
                        params=result.params[1],
                        stats=result.stats[1],
                        config=config_metadata,
                    )
            except Exception:
                log.exception("Failed to calibrate %s", artifact_key)

    log.info("Done. Results saved to %s (table: vol_market)", DATABASE_PATH)


if __name__ == "__main__":
    main()
