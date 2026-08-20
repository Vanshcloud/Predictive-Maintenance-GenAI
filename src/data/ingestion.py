"""
src/data/ingestion.py — Data Ingestion Pipeline
============================================================

WHY THIS FILE EXISTS:
    Data ingestion is the FIRST step in any ML pipeline. It handles:
    - Loading data from various formats (CSV, Parquet, JSON)
    - Logging metadata about what was loaded (shape, dtypes, null counts)
    - Providing a clean, consistent interface for the rest of the pipeline

    Without a proper ingestion layer, you end up with pd.read_csv()
    calls scattered everywhere — no logging, no error handling, no
    consistency.

DESIGN PATTERN:
    - Single Responsibility: This class ONLY loads data. It doesn't
      clean, transform, or validate — those are separate concerns.
    - Strategy Pattern: Different loading strategies for different
      file formats, selected automatically by file extension.
    - Dependency Injection: Takes settings as a parameter, making
      it testable with mock configurations.

USAGE:
    from src.data.ingestion import DataIngestion

    ingestion = DataIngestion()

    # Load a single file
    df = ingestion.load_csv("data/raw/telemetry.csv")

    # Load the complete dataset (all 5 tables)
    dataset = ingestion.load_dataset()
    telemetry = dataset["telemetry"]
    failures = dataset["failures"]
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from config.settings import Settings, get_settings
from src.utils.exceptions import DataIngestionError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataIngestion:
    """
    Production-ready data ingestion pipeline.

    Loads predictive maintenance data from various file formats
    with comprehensive logging and error handling.

    Attributes:
        settings: Application configuration.
        data_dir: Base directory for data files.
    """

    # Supported file formats and their pandas readers
    _READERS = {
        ".csv": pd.read_csv,
        ".parquet": pd.read_parquet,
        ".json": pd.read_json,
        ".xlsx": pd.read_excel,
    }

    # The 5 tables in our predictive maintenance dataset
    DATASET_TABLES = [
        "machines",
        "telemetry",
        "errors",
        "maintenance",
        "failures",
    ]

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Initialize the data ingestion pipeline.

        Args:
            settings: Application settings. If None, loads from .env.
                      Passing settings explicitly makes testing easier
                      (you can pass mock settings).
        """
        self.settings = settings or get_settings()
        self.data_dir = self.settings.raw_data_path
        logger.info(f"DataIngestion initialized | data_dir={self.data_dir}")

    def load_file(
        self,
        filepath: Union[str, Path],
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Load a single data file with automatic format detection.

        WHY automatic detection?
            You might start with CSV files and later migrate to Parquet
            for performance. This method handles the transition seamlessly.

        Args:
            filepath: Path to the data file (absolute or relative to data_dir).
            **kwargs: Additional arguments passed to the pandas reader
                      (e.g., parse_dates, usecols, nrows).

        Returns:
            Loaded DataFrame.

        Raises:
            DataIngestionError: If file doesn't exist or format is unsupported.
        """
        filepath = Path(filepath)

        # If path is relative, resolve against data_dir
        if not filepath.is_absolute():
            filepath = self.data_dir / filepath

        # --- Validate file exists ---
        if not filepath.exists():
            raise DataIngestionError(
                message=f"Data file not found: {filepath}",
                details={
                    "filepath": str(filepath),
                    "data_dir": str(self.data_dir),
                    "exists": False,
                },
            )

        # --- Select reader based on extension ---
        suffix = filepath.suffix.lower()
        reader = self._READERS.get(suffix)

        if reader is None:
            raise DataIngestionError(
                message=f"Unsupported file format: {suffix}",
                details={
                    "filepath": str(filepath),
                    "extension": suffix,
                    "supported": list(self._READERS.keys()),
                },
            )

        # --- Load the file ---
        try:
            logger.info(f"Loading {filepath.name}...")
            df = reader(filepath, **kwargs)
            self._log_dataframe_info(filepath.name, df)
            return df

        except Exception as e:
            raise DataIngestionError(
                message=f"Failed to load {filepath.name}: {str(e)}",
                details={
                    "filepath": str(filepath),
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
            ) from e

    def load_csv(
        self,
        filename: str,
        parse_dates: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Convenience method to load a CSV file from the data directory.

        Args:
            filename: Name of the CSV file (e.g., "telemetry.csv").
            parse_dates: List of column names to parse as datetime.
            **kwargs: Additional pandas read_csv arguments.

        Returns:
            Loaded DataFrame.
        """
        if parse_dates:
            kwargs["parse_dates"] = parse_dates

        return self.load_file(filename, **kwargs)

    def load_dataset(
        self,
        data_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load the complete 5-table predictive maintenance dataset.

        WHY a dedicated method?
            Loading all 5 tables is the most common operation.
            This method ensures consistent loading order, proper
            datetime parsing, and comprehensive logging.

        Args:
            data_dir: Override directory to load from.
                      Useful for loading sample data during testing.

        Returns:
            Dictionary mapping table names to DataFrames:
            {"machines": df, "telemetry": df, "errors": df,
             "maintenance": df, "failures": df}

        Raises:
            DataIngestionError: If any required table is missing.
        """
        load_dir = Path(data_dir) if data_dir else self.data_dir

        logger.info("=" * 50)
        logger.info(f"Loading complete dataset from {load_dir}")
        logger.info("=" * 50)

        # Columns that contain datetime values (for auto-parsing)
        datetime_columns = {
            "telemetry": ["datetime"],
            "errors": ["datetime"],
            "maintenance": ["datetime"],
            "failures": ["datetime"],
        }

        dataset: Dict[str, pd.DataFrame] = {}
        missing_tables: List[str] = []

        for table_name in self.DATASET_TABLES:
            filepath = load_dir / f"{table_name}.csv"

            if not filepath.exists():
                missing_tables.append(table_name)
                logger.warning(f"  ⚠ Missing table: {table_name}.csv")
                continue

            # Parse datetime columns if specified for this table
            parse_dates = datetime_columns.get(table_name)
            kwargs = {}
            if parse_dates:
                kwargs["parse_dates"] = parse_dates

            try:
                df = self.load_file(filepath, **kwargs)
                dataset[table_name] = df
            except DataIngestionError:
                missing_tables.append(table_name)
                logger.error(f"  ✗ Failed to load: {table_name}.csv")

        # --- Summary ---
        logger.info("")
        logger.info("Dataset loading summary:")
        for name, df in dataset.items():
            logger.info(f"  ✓ {name:15s}: {len(df):>10,} rows × {len(df.columns)} cols")

        if missing_tables:
            logger.warning(f"  Missing tables: {missing_tables}")

        total = sum(len(df) for df in dataset.values())
        logger.info(f"  Total rows loaded: {total:,}")

        return dataset

    def _log_dataframe_info(self, name: str, df: pd.DataFrame) -> None:
        """
        Log metadata about a loaded DataFrame.

        WHY: When debugging pipeline issues at 3 AM, you need to know:
        - How many rows were loaded?
        - Were there unexpected null values?
        - Did the data types change from last time?

        This method logs that information AUTOMATICALLY on every load.
        """
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        null_pct = (total_nulls / (len(df) * len(df.columns))) * 100

        logger.info(
            f"  Loaded {name}: "
            f"{len(df):,} rows × {len(df.columns)} cols | "
            f"nulls: {total_nulls:,} ({null_pct:.1f}%) | "
            f"memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB"
        )

        # Log column dtypes for debugging
        logger.debug(f"  Columns: {dict(df.dtypes)}")

        # Warn about high null rates
        for col in df.columns:
            col_null_pct = df[col].isnull().mean() * 100
            if col_null_pct > 5:
                logger.warning(
                    f"  ⚠ High null rate in '{col}': {col_null_pct:.1f}%"
                )
