from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.dataset_adapter import DatasetAdapter


class EventLogLoader:
    """
    Generic event-log loader for APMOA.

    Supported input formats:

        - CSV
        - Excel (.xlsx / .xls)
        - XES
        - XES.GZ

    Dataset-specific column names are handled by
    DatasetAdapter.

    APMOA receives the canonical schema:

        case_id
        activity
        timestamp
        resource

    Additional dataset columns are preserved.
    """

    def __init__(
        self,
        log_path: str | Path,
    ):
        self.log_path = Path(log_path)

        self.log: Optional[pd.DataFrame] = None

        self.adapter: Optional[
            DatasetAdapter
        ] = None

    # =========================================================
    # LOAD
    # =========================================================

    def load(self) -> pd.DataFrame:
        """
        Load and normalize the event log.

        Returns
        -------
        pandas.DataFrame
            Canonical APMOA event log.
        """

        if not self.log_path.exists():
            raise FileNotFoundError(
                f"Event log not found: "
                f"{self.log_path}"
            )

        print(
            f"Loading event log: "
            f"{self.log_path}"
        )

        # -----------------------------------------------------
        # Initialize DatasetAdapter
        # -----------------------------------------------------

        self.adapter = DatasetAdapter()

        # -----------------------------------------------------
        # Load through DatasetAdapter
        # -----------------------------------------------------

        self.log = self.adapter.load(
            source=self.log_path
        )

        if not isinstance(
            self.log,
            pd.DataFrame,
        ):
            raise TypeError(
                "DatasetAdapter must return "
                "a pandas DataFrame."
            )

        # -----------------------------------------------------
        # Make a copy so downstream agents do not
        # accidentally modify the adapter's DataFrame.
        # -----------------------------------------------------

        self.log = self.log.copy()

        # -----------------------------------------------------
        # Normalize timestamp
        # -----------------------------------------------------

        if "timestamp" in self.log.columns:

            self.log["timestamp"] = (
                pd.to_datetime(
                    self.log["timestamp"],
                    errors="coerce",
                )
            )

        print(
            f"Loaded {len(self.log):,} events."
        )

        print(
            "\nCanonical APMOA schema:"
        )

        for column in self.log.columns:

            print(
                f"  - {column}"
            )

        return self.log

    # =========================================================
    # VALIDATION
    # =========================================================

    def validate(self) -> bool:
        """
        Validate the canonical APMOA event-log schema.
        """

        if self.log is None:
            raise RuntimeError(
                "Event log has not been loaded. "
                "Call load() first."
            )

        required_columns = [
            "case_id",
            "activity",
            "timestamp",
            "resource",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in self.log.columns
        ]

        if missing_columns:

            raise ValueError(
                "Missing canonical columns: "
                f"{missing_columns}"
            )

        # -----------------------------------------------------
        # Validate required values
        # -----------------------------------------------------

        for column in [
            "case_id",
            "activity",
            "timestamp",
        ]:

            missing_count = int(
                self.log[column]
                .isna()
                .sum()
            )

            if missing_count > 0:

                raise ValueError(
                    f"Column '{column}' contains "
                    f"{missing_count:,} missing values."
                )

        print(
            "\nEvent log validation: PASSED"
        )

        return True

    # =========================================================
    # SUMMARY
    # =========================================================

    def get_summary(self) -> dict:
        """
        Return basic statistics about the event log.
        """

        if self.log is None:
            raise RuntimeError(
                "Event log has not been loaded."
            )

        summary = {
            "events": int(
                len(self.log)
            ),

            "cases": int(
                self.log[
                    "case_id"
                ].nunique()
            ),

            "activities": int(
                self.log[
                    "activity"
                ].nunique()
            ),

            "resources": (
                int(
                    self.log[
                        "resource"
                    ].nunique()
                )
                if "resource" in self.log.columns
                else 0
            ),

            "start_time": (
                self.log[
                    "timestamp"
                ].min()
            ),

            "end_time": (
                self.log[
                    "timestamp"
                ].max()
            ),
        }

        return summary

    # =========================================================
    # STANDARDIZED LOG
    # =========================================================

    def get_standardized_log(
        self,
    ) -> pd.DataFrame:
        """
        Return the canonical APMOA event log.

        DatasetAdapter has already performed the
        normalization, so this method returns a copy.
        """

        if self.log is None:
            raise RuntimeError(
                "Event log has not been loaded."
            )

        return self.log.copy()

    # =========================================================
    # DATASET INFORMATION
    # =========================================================

    def get_dataset_info(self) -> dict:
        """
        Return dataset metadata.

        This will also be useful later for the dashboard.
        """

        if self.log is None:
            raise RuntimeError(
                "Event log has not been loaded."
            )

        summary = self.get_summary()

        return {
            "source": str(
                self.log_path
            ),

            "file_name": (
                self.log_path.name
            ),

            "file_type": (
                self.log_path.suffix.lower()
            ),

            "events": summary[
                "events"
            ],

            "cases": summary[
                "cases"
            ],

            "activities": summary[
                "activities"
            ],

            "resources": summary[
                "resources"
            ],

            "start_time": str(
                summary[
                    "start_time"
                ]
            ),

            "end_time": str(
                summary[
                    "end_time"
                ]
            ),

            "columns": list(
                self.log.columns
            ),

            "canonical_schema": [
                "case_id",
                "activity",
                "timestamp",
                "resource",
            ],
        }