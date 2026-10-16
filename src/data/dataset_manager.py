from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.data.dataset_adapter import DatasetAdapter


class DatasetManager:
    """
    Dataset management layer for APMOA.

    Responsibilities:
        1. Load datasets
        2. Normalize datasets
        3. Validate datasets
        4. Generate metadata
        5. Prepare datasets for APMOA
    """

    SUPPORTED_EXTENSIONS = {
        ".csv",
        ".xlsx",
        ".xls",
        ".xes",
        ".gz",
    }

    REQUIRED_COLUMNS = [
        "case_id",
        "activity",
        "timestamp",
        "resource",
    ]

    def __init__(
        self,
        storage_dir: str | Path = "data/raw",
    ):
        self.storage_dir = Path(
            storage_dir
        )

        self.storage_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.current_dataset: Optional[
            pd.DataFrame
        ] = None

        self.current_path: Optional[
            Path
        ] = None

        self.current_metadata: Optional[
            Dict[str, Any]
        ] = None

    # =========================================================
    # LOAD DATASET
    # =========================================================

    def load_dataset(
        self,
        path: str | Path,
    ) -> pd.DataFrame:
        """
        Load a dataset through DatasetAdapter.
        """

        dataset_path = Path(path)

        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: "
                f"{dataset_path}"
            )

        print(
            "\n" + "=" * 60
        )

        print(
            "APMOA DATASET MANAGER"
        )

        print(
            "=" * 60
        )

        print(
            f"\nLoading dataset:"
            f"\n  {dataset_path}"
        )

        adapter = DatasetAdapter(
            source=dataset_path
        )

        dataframe = adapter.load()

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                "DatasetAdapter must return "
                "a pandas DataFrame."
            )

        self.current_dataset = (
            dataframe.copy()
        )

        self.current_path = (
            dataset_path
        )

        self._prepare_dataframe()

        self.current_metadata = (
            self._generate_metadata()
        )

        return self.current_dataset

    # =========================================================
    # PREPARE DATAFRAME
    # =========================================================

    def _prepare_dataframe(
        self,
    ) -> None:
        """
        Prepare the canonical DataFrame
        for downstream APMOA agents.
        """

        if self.current_dataset is None:
            raise RuntimeError(
                "No dataset loaded."
            )

        dataframe = (
            self.current_dataset
        )

        # -----------------------------------------------------
        # Timestamp normalization
        # -----------------------------------------------------

        if "timestamp" in dataframe.columns:

            dataframe["timestamp"] = (
                pd.to_datetime(
                    dataframe["timestamp"],
                    errors="coerce",
                )
            )

        # -----------------------------------------------------
        # Sort event log
        # -----------------------------------------------------

        if {
            "case_id",
            "timestamp",
        }.issubset(
            dataframe.columns
        ):

            dataframe = (
                dataframe.sort_values(
                    by=[
                        "case_id",
                        "timestamp",
                    ],
                    kind="stable",
                )
                .reset_index(
                    drop=True
                )
            )

        self.current_dataset = dataframe

    # =========================================================
    # VALIDATE
    # =========================================================

    def validate_dataset(
        self,
    ) -> Dict[str, Any]:
        """
        Validate the currently loaded dataset.
        """

        if self.current_dataset is None:
            raise RuntimeError(
                "No dataset loaded."
            )

        dataframe = (
            self.current_dataset
        )

        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in dataframe.columns
        ]

        validation = {
            "valid": True,
            "missing_columns": missing_columns,
            "columns": list(
                dataframe.columns
            ),
            "events": len(dataframe),
            "cases": 0,
            "activities": 0,
            "resources": 0,
            "errors": [],
            "warnings": [],
        }

        if missing_columns:

            validation["valid"] = False

            validation["errors"].append(
                "Missing required canonical "
                f"columns: {missing_columns}"
            )

            return validation

        # -----------------------------------------------------
        # Basic statistics
        # -----------------------------------------------------

        validation["cases"] = int(
            dataframe[
                "case_id"
            ].nunique()
        )

        validation["activities"] = int(
            dataframe[
                "activity"
            ].nunique()
        )

        validation["resources"] = int(
            dataframe[
                "resource"
            ].nunique()
        )

        # -----------------------------------------------------
        # Empty dataset
        # -----------------------------------------------------

        if dataframe.empty:

            validation["valid"] = False

            validation["errors"].append(
                "Dataset contains no events."
            )

        # -----------------------------------------------------
        # Missing values
        # -----------------------------------------------------

        for column in [
            "case_id",
            "activity",
            "timestamp",
        ]:

            missing_count = int(
                dataframe[
                    column
                ].isna().sum()
            )

            if missing_count > 0:

                validation["valid"] = False

                validation["errors"].append(
                    f"Column '{column}' contains "
                    f"{missing_count:,} missing values."
                )

        # -----------------------------------------------------
        # Resource warning
        # -----------------------------------------------------

        if (
            "resource"
            in dataframe.columns
        ):

            unknown_resources = int(
                (
                    dataframe[
                        "resource"
                    ]
                    .astype(str)
                    .str.upper()
                    .eq("UNKNOWN")
                ).sum()
            )

            if unknown_resources > 0:

                validation["warnings"].append(
                    f"{unknown_resources:,} events "
                    "have UNKNOWN resource."
                )

        return validation

    # =========================================================
    # METADATA
    # =========================================================

    def _generate_metadata(
        self,
    ) -> Dict[str, Any]:
        """
        Generate dataset metadata.
        """

        if self.current_dataset is None:
            raise RuntimeError(
                "No dataset loaded."
            )

        if self.current_path is None:
            raise RuntimeError(
                "Dataset path unavailable."
            )

        dataframe = (
            self.current_dataset
        )

        return {
            "dataset_name": (
                self.current_path.name
            ),

            "dataset_path": str(
                self.current_path
            ),

            "extension": (
                self.current_path.suffix.lower()
            ),

            "events": int(
                len(dataframe)
            ),

            "cases": int(
                dataframe[
                    "case_id"
                ].nunique()
            ),

            "activities": int(
                dataframe[
                    "activity"
                ].nunique()
            ),

            "resources": int(
                dataframe[
                    "resource"
                ].nunique()
            ),

            "start_time": str(
                dataframe[
                    "timestamp"
                ].min()
            ),

            "end_time": str(
                dataframe[
                    "timestamp"
                ].max()
            ),

            "columns": list(
                dataframe.columns
            ),

            "canonical_schema": [
                "case_id",
                "activity",
                "timestamp",
                "resource",
            ],
        }

    # =========================================================
    # PUBLIC METADATA
    # =========================================================

    def get_metadata(
        self,
    ) -> Dict[str, Any]:
        """
        Return metadata for the current dataset.
        """

        if self.current_metadata is None:

            raise RuntimeError(
                "No dataset metadata available. "
                "Load a dataset first."
            )

        return self.current_metadata.copy()

    # =========================================================
    # GET DATASET
    # =========================================================

    def get_dataset(
        self,
    ) -> pd.DataFrame:
        """
        Return the current canonical dataset.
        """

        if self.current_dataset is None:

            raise RuntimeError(
                "No dataset loaded."
            )

        return self.current_dataset.copy()

    # =========================================================
    # DATASET STATUS
    # =========================================================

    def get_status(
        self,
    ) -> Dict[str, Any]:
        """
        Return the current dataset status.
        """

        if self.current_dataset is None:

            return {
                "loaded": False,
                "valid": False,
                "dataset": None,
            }

        validation = (
            self.validate_dataset()
        )

        return {
            "loaded": True,
            "valid": validation[
                "valid"
            ],
            "dataset": self.get_metadata(),
            "validation": validation,
        }