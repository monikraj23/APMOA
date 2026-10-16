"""
APMOA Dataset Adapter

Provides a dataset-independent interface for event logs.

Canonical schema
----------------
case_id
activity
timestamp
resource

The adapter accepts common alternative column names and converts
them into the canonical APMOA schema.
"""

from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd


class DatasetAdapter:
    """
    Normalize event-log datasets into the APMOA canonical schema.

    Required canonical columns:

        case_id
        activity
        timestamp

    Resource is strongly recommended but is allowed to be missing
    because not every event log contains resource information.
    """

    CANONICAL_COLUMNS = [
        "case_id",
        "activity",
        "timestamp",
        "resource",
    ]

    REQUIRED_COLUMNS = [
        "case_id",
        "activity",
        "timestamp",
    ]

    COLUMN_ALIASES = {

        "case_id": [
            "case_id",
            "case id",
            "case",
            "caseid",
            "case:concept:name",
            "case:case_id",
            "process_instance",
            "process_instance_id",
            "trace_id",
        ],

        "activity": [
            "activity",
            "activity_name",
            "activity name",
            "event",
            "event_name",
            "event name",
            "concept:name",
            "task",
            "task_name",
        ],

        "timestamp": [
            "timestamp",
            "time",
            "event_time",
            "event_timestamp",
            "event timestamp",
            "time:timestamp",
            "datetime",
            "date_time",
        ],

        "resource": [
            "resource",
            "resource_id",
            "resource id",
            "resource_name",
            "resource name",
            "org:resource",
            "user",
            "user_id",
            "user id",
            "employee",
            "employee_id",
            "performer",
        ],
    }

    def __init__(
        self,
        case_column: Optional[str] = None,
        activity_column: Optional[str] = None,
        timestamp_column: Optional[str] = None,
        resource_column: Optional[str] = None,
    ):
        """
        Optionally provide explicit source-column mappings.

        If mappings are not supplied, the adapter automatically
        detects common column names.
        """

        self.explicit_mapping = {
            "case_id": case_column,
            "activity": activity_column,
            "timestamp": timestamp_column,
            "resource": resource_column,
        }

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def load(
        self,
        source,
        file_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load and normalize an event log.

        Supported input types:

        - pandas DataFrame
        - CSV
        - Excel
        - XES
        - compressed XES (.xes.gz)

        Returns
        -------
        pandas.DataFrame
            Dataframe using the canonical APMOA schema.
        """

        if isinstance(source, pd.DataFrame):

            dataframe = source.copy()

        else:

            path = Path(source)

            if not path.exists():

                raise FileNotFoundError(
                    f"Dataset not found: {path}"
                )

            detected_type = (
                file_type.lower()
                if file_type
                else self._detect_file_type(path)
            )

            dataframe = self._load_file(
                path,
                detected_type
            )

        return self.normalize(dataframe)

    def normalize(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Normalize a dataframe into the canonical APMOA schema.
        """

        if dataframe is None:

            raise ValueError(
                "Dataset cannot be None."
            )

        if not isinstance(dataframe, pd.DataFrame):

            raise TypeError(
                "Dataset must be a pandas DataFrame."
            )

        dataframe = dataframe.copy()

        dataframe.columns = [
            str(column).strip()
            for column in dataframe.columns
        ]

        mapping = self._resolve_columns(
            dataframe.columns
        )

        missing = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in mapping
        ]

        if missing:

            raise ValueError(
                "Unable to identify required event-log columns: "
                f"{missing}\n"
                f"Available columns: "
                f"{list(dataframe.columns)}\n\n"
                "Provide explicit column mappings if necessary."
            )

        rename_mapping = {
            source: target
            for target, source in mapping.items()
        }

        dataframe = dataframe.rename(
            columns=rename_mapping
        )

        # ------------------------------------------------------
        # Resource is optional
        # ------------------------------------------------------

        if "resource" not in dataframe.columns:

            dataframe["resource"] = "UNKNOWN"

        # ------------------------------------------------------
        # Keep canonical columns first
        # ------------------------------------------------------

        canonical = [
            "case_id",
            "activity",
            "timestamp",
            "resource",
        ]

        remaining = [
            column
            for column in dataframe.columns
            if column not in canonical
        ]

        dataframe = dataframe[
            canonical + remaining
        ]

        # ------------------------------------------------------
        # Clean values
        # ------------------------------------------------------

        dataframe["case_id"] = (
            dataframe["case_id"]
            .astype(str)
            .str.strip()
        )

        dataframe["activity"] = (
            dataframe["activity"]
            .astype(str)
            .str.strip()
        )

        dataframe["resource"] = (
            dataframe["resource"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.strip()
        )

        # ------------------------------------------------------
        # Timestamp conversion
        # ------------------------------------------------------

        dataframe["timestamp"] = pd.to_datetime(
            dataframe["timestamp"],
            errors="coerce",
        )

        invalid_timestamps = (
            dataframe["timestamp"]
            .isna()
            .sum()
        )

        if invalid_timestamps > 0:

            raise ValueError(
                f"Dataset contains "
                f"{invalid_timestamps} invalid timestamps."
            )

        # ------------------------------------------------------
        # Remove invalid core rows
        # ------------------------------------------------------

        dataframe = dataframe.dropna(
            subset=[
                "case_id",
                "activity",
                "timestamp",
            ]
        )

        # ------------------------------------------------------
        # Sort event log
        # ------------------------------------------------------

        dataframe = dataframe.sort_values(
            by=[
                "case_id",
                "timestamp",
            ],
            kind="stable",
        ).reset_index(drop=True)

        return dataframe

    def validate(
        self,
        dataframe: pd.DataFrame,
    ) -> Dict:
        """
        Validate an already normalized dataset.
        """

        required = set(
            self.REQUIRED_COLUMNS
        )

        available = set(
            dataframe.columns
        )

        missing = sorted(
            required - available
        )

        validation = {
            "valid": len(missing) == 0,
            "missing_columns": missing,
            "columns": list(dataframe.columns),
            "events": len(dataframe),
            "cases": (
                dataframe["case_id"].nunique()
                if "case_id" in dataframe.columns
                else 0
            ),
            "activities": (
                dataframe["activity"].nunique()
                if "activity" in dataframe.columns
                else 0
            ),
            "resources": (
                dataframe["resource"].nunique()
                if "resource" in dataframe.columns
                else 0
            ),
        }

        return validation

    def get_summary(
        self,
        dataframe: pd.DataFrame,
    ) -> Dict:
        """
        Return dataset-independent summary information.
        """

        return {
            "events": int(len(dataframe)),
            "cases": int(
                dataframe["case_id"].nunique()
            ),
            "activities": int(
                dataframe["activity"].nunique()
            ),
            "resources": int(
                dataframe["resource"].nunique()
            ),
            "start_time": str(
                dataframe["timestamp"].min()
            ),
            "end_time": str(
                dataframe["timestamp"].max()
            ),
            "columns": list(
                dataframe.columns
            ),
        }

    # ==========================================================
    # FILE LOADING
    # ==========================================================

    def _load_file(
        self,
        path: Path,
        file_type: str,
    ) -> pd.DataFrame:

        if file_type == "csv":

            return pd.read_csv(path)

        if file_type in {"xlsx", "xls"}:

            return pd.read_excel(path)

        if file_type in {"xes", "xes.gz"}:

            return self._load_xes(path)

        raise ValueError(
            f"Unsupported file type: {file_type}"
        )

    def _load_xes(
        self,
        path: Path,
    ) -> pd.DataFrame:

        try:

            import pm4py

        except ImportError as exc:

            raise ImportError(
                "PM4Py is required to load XES files."
            ) from exc

        log = pm4py.read_xes(
            str(path)
        )

        return log

    # ==========================================================
    # COLUMN DETECTION
    # ==========================================================

    def _resolve_columns(
        self,
        columns,
    ) -> Dict[str, str]:

        normalized_columns = {
            self._normalize_column_name(column): column
            for column in columns
        }

        mapping = {}

        for canonical in self.CANONICAL_COLUMNS:

            explicit = self.explicit_mapping.get(
                canonical
            )

            if explicit:

                if explicit not in columns:

                    raise ValueError(
                        f"Explicit column "
                        f"'{explicit}' for "
                        f"'{canonical}' was not found."
                    )

                mapping[canonical] = explicit
                continue

            aliases = self.COLUMN_ALIASES[
                canonical
            ]

            for alias in aliases:

                normalized_alias = (
                    self._normalize_column_name(alias)
                )

                if normalized_alias in normalized_columns:

                    mapping[canonical] = (
                        normalized_columns[
                            normalized_alias
                        ]
                    )

                    break

        return mapping

    @staticmethod
    def _normalize_column_name(
        column: str,
    ) -> str:

        return (
            str(column)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

    @staticmethod
    def _detect_file_type(
        path: Path,
    ) -> str:

        name = path.name.lower()

        if name.endswith(".xes.gz"):
            return "xes.gz"

        suffix = path.suffix.lower()

        if suffix == ".xes":
            return "xes"

        if suffix == ".csv":
            return "csv"

        if suffix in {".xlsx", ".xls"}:
            return suffix[1:]

        raise ValueError(
            f"Unable to determine file type "
            f"from: {path}"
        )