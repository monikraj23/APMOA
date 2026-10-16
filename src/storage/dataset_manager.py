"""
APMOA Dataset Manager

Responsible for:

1. Storing uploaded event-log files
2. Maintaining dataset metadata
3. Validating datasets
4. Loading datasets later by dataset ID
5. Listing available datasets
6. Removing datasets safely

Supported formats:

- XES
- XES.GZ
- CSV
- XLSX
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS_ROOT = (
    PROJECT_ROOT
    / "data"
    / "datasets"
)

UPLOAD_DIR = (
    DATASETS_ROOT
    / "uploaded"
)

METADATA_DIR = (
    DATASETS_ROOT
    / "metadata"
)

METADATA_FILE = (
    METADATA_DIR
    / "datasets.json"
)


# ---------------------------------------------------------------------
# SUPPORTED FORMATS
# ---------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".xes",
    ".gz",
    ".csv",
    ".xlsx",
}


# ---------------------------------------------------------------------
# DATASET MANAGER
# ---------------------------------------------------------------------

class DatasetManager:
    """
    Manage uploaded event-log datasets.

    Example
    -------
    manager = DatasetManager()

    dataset = manager.register_file(
        file_path="/path/to/log.xes.gz",
        name="BPI 2020 Travel Permit",
    )

    dataset_id = dataset["id"]

    path = manager.get_dataset_path(
        dataset_id
    )
    """

    def __init__(
        self,
        upload_dir: Optional[Path] = None,
        metadata_file: Optional[Path] = None,
    ):
        self.upload_dir = Path(
            upload_dir
            if upload_dir is not None
            else UPLOAD_DIR
        )

        self.metadata_file = Path(
            metadata_file
            if metadata_file is not None
            else METADATA_FILE
        )

        self._initialize_storage()

    # -----------------------------------------------------------------
    # INITIALIZATION
    # -----------------------------------------------------------------

    def _initialize_storage(self):
        """Create required directories and metadata file."""

        self.upload_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.metadata_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.metadata_file.exists():

            self._write_metadata(
                {
                    "datasets": []
                }
            )

    # -----------------------------------------------------------------
    # METADATA
    # -----------------------------------------------------------------

    def _read_metadata(self) -> Dict[str, Any]:
        """Read the dataset metadata registry."""

        if not self.metadata_file.exists():

            return {
                "datasets": []
            }

        try:

            with open(
                self.metadata_file,
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

        except (
            json.JSONDecodeError,
            OSError,
        ):

            return {
                "datasets": []
            }

        if not isinstance(
            data,
            dict,
        ):

            return {
                "datasets": []
            }

        if not isinstance(
            data.get("datasets"),
            list,
        ):

            data["datasets"] = []

        return data

    def _write_metadata(
        self,
        data: Dict[str, Any],
    ):
        """Write metadata atomically."""

        self.metadata_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_file = (
            self.metadata_file.with_suffix(
                ".tmp"
            )
        )

        with open(
            temporary_file,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
            )

        temporary_file.replace(
            self.metadata_file
        )

    # -----------------------------------------------------------------
    # ID GENERATION
    # -----------------------------------------------------------------

    @staticmethod
    def _slugify(
        value: str,
    ) -> str:
        """Convert a dataset name into a safe identifier."""

        value = str(
            value
        ).strip().lower()

        value = re.sub(
            r"[^a-z0-9]+",
            "_",
            value,
        )

        value = re.sub(
            r"_+",
            "_",
            value,
        )

        return value.strip("_")

    def _generate_dataset_id(
        self,
        name: str,
    ) -> str:
        """
        Generate a stable unique dataset ID.

        Example:

        BPI 2020 Travel Permit
        →
        bpi_2020_travel_permit
        """

        base_id = (
            self._slugify(name)
            or "dataset"
        )

        metadata = self._read_metadata()

        existing_ids = {
            item.get("id")
            for item in metadata["datasets"]
            if isinstance(item, dict)
        }

        dataset_id = base_id
        counter = 2

        while dataset_id in existing_ids:

            dataset_id = (
                f"{base_id}_{counter}"
            )

            counter += 1

        return dataset_id

    # -----------------------------------------------------------------
    # FORMAT DETECTION
    # -----------------------------------------------------------------

    @staticmethod
    def detect_format(
        file_name: str,
    ) -> str:
        """Detect supported event-log format."""

        name = str(
            file_name
        ).lower()

        if name.endswith(
            ".xes.gz"
        ):

            return "xes.gz"

        if name.endswith(
            ".xes"
        ):

            return "xes"

        if name.endswith(
            ".csv"
        ):

            return "csv"

        if name.endswith(
            ".xlsx"
        ):

            return "xlsx"

        raise ValueError(
            "Unsupported dataset format. "
            "Supported formats: XES, XES.GZ, CSV, XLSX."
        )

    # -----------------------------------------------------------------
    # FILE VALIDATION
    # -----------------------------------------------------------------

    def _validate_extension(
        self,
        file_name: str,
    ):
        """Validate the uploaded file extension."""

        self.detect_format(
            file_name
        )

    # -----------------------------------------------------------------
    # SAVE UPLOADED FILE
    # -----------------------------------------------------------------

    def save_uploaded_file(
        self,
        uploaded_file,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Save a Streamlit UploadedFile.

        Parameters
        ----------
        uploaded_file:
            Streamlit UploadedFile object.

        name:
            Optional human-readable dataset name.
            If omitted, filename is used.

        Returns
        -------
        dict
            Dataset metadata.
        """

        file_name = (
            uploaded_file.name
        )

        self._validate_extension(
            file_name
        )

        dataset_name = (
            name.strip()
            if name
            else Path(
                file_name
            ).stem
        )

        if file_name.lower().endswith(
            ".xes.gz"
        ):

            dataset_name = (
                name.strip()
                if name
                else file_name[:-7]
            )

        dataset_id = (
            self._generate_dataset_id(
                dataset_name
            )
        )

        dataset_dir = (
            self.upload_dir
            / dataset_id
        )

        dataset_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        destination = (
            dataset_dir
            / file_name
        )

        try:

            with open(
                destination,
                "wb",
            ) as file:

                file.write(
                    uploaded_file.getbuffer()
                )

        except Exception:

            shutil.rmtree(
                dataset_dir,
                ignore_errors=True,
            )

            raise

        return self.register_saved_file(
            dataset_id=dataset_id,
            name=dataset_name,
            file_path=destination,
        )

    # -----------------------------------------------------------------
    # REGISTER EXISTING FILE
    # -----------------------------------------------------------------

    def register_file(
        self,
        file_path,
        name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Copy an existing dataset into APMOA storage.

        Useful for importing existing datasets such as:

        BPI 2020
        BPI 2012
        """

        source = Path(
            file_path
        )

        if not source.exists():

            raise FileNotFoundError(
                f"Dataset file not found: {source}"
            )

        self._validate_extension(
            source.name
        )

        dataset_name = (
            name.strip()
            if name
            else source.stem
        )

        if source.name.lower().endswith(
            ".xes.gz"
        ):

            dataset_name = (
                name.strip()
                if name
                else source.name[:-7]
            )

        dataset_id = (
            self._generate_dataset_id(
                dataset_name
            )
        )

        dataset_dir = (
            self.upload_dir
            / dataset_id
        )

        dataset_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        destination = (
            dataset_dir
            / source.name
        )

        try:

            shutil.copy2(
                source,
                destination,
            )

        except Exception:

            shutil.rmtree(
                dataset_dir,
                ignore_errors=True,
            )

            raise

        return self.register_saved_file(
            dataset_id=dataset_id,
            name=dataset_name,
            file_path=destination,
        )

    # -----------------------------------------------------------------
    # REGISTER SAVED FILE
    # -----------------------------------------------------------------

    def register_saved_file(
        self,
        dataset_id: str,
        name: str,
        file_path: Path,
    ) -> Dict[str, Any]:
        """
        Register an already stored dataset file.
        """

        file_path = Path(
            file_path
        )

        metadata = self._read_metadata()

        # -------------------------------------------------------------
        # Validate using the existing APMOA DatasetAdapter
        # -------------------------------------------------------------

        validation = self._validate_dataset(
            file_path
        )

        now = datetime.now().isoformat()

        dataset_metadata = {
            "id": dataset_id,
            "name": name,
            "filename": file_path.name,
            "path": str(
                file_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "format": self.detect_format(
                file_path.name
            ),
            "events": validation.get(
                "events"
            ),
            "cases": validation.get(
                "cases"
            ),
            "activities": validation.get(
                "activities"
            ),
            "resources": validation.get(
                "resources"
            ),
            "valid": validation.get(
                "valid",
                False,
            ),
            "validation_error": validation.get(
                "error"
            ),
            "uploaded_at": now,
            "updated_at": now,
            "status": (
                "validated"
                if validation.get("valid")
                else "invalid"
            ),
        }

        metadata["datasets"].append(
            dataset_metadata
        )

        self._write_metadata(
            metadata
        )

        return dataset_metadata

    # -----------------------------------------------------------------
    # DATASET VALIDATION
    # -----------------------------------------------------------------

    def _validate_dataset(
        self,
        file_path: Path,
    ) -> Dict[str, Any]:
        """
        Validate a dataset using the existing APMOA DatasetAdapter.

        This keeps dataset-management validation aligned with the
        rest of the APMOA pipeline.
        """

        try:

            from src.data.dataset_adapter import (
                DatasetAdapter
            )

            adapter = DatasetAdapter()

            log = adapter.load(
                source=file_path
            )

            events = len(
                log
            )

            cases = (
                log["case_id"]
                .nunique()
            )

            activities = (
                log["activity"]
                .nunique()
            )

            resources = (
                log["resource"]
                .nunique()
                if "resource" in log.columns
                else 0
            )

            valid = True

            validation_result = None

            if hasattr(
                adapter,
                "validate"
            ):

                try:

                    validation_result = (
                        adapter.validate(
                            log
                        )
                    )

                except TypeError:

                    validation_result = (
                        adapter.validate()
                    )

                if isinstance(
                    validation_result,
                    dict,
                ):

                    valid = bool(
                        validation_result.get(
                            "valid",
                            True,
                        )
                    )

            return {
                "valid": valid,
                "events": events,
                "cases": cases,
                "activities": activities,
                "resources": resources,
                "validation": validation_result,
                "error": None,
            }

        except Exception as exc:

            return {
                "valid": False,
                "events": None,
                "cases": None,
                "activities": None,
                "resources": None,
                "validation": None,
                "error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }

    # -----------------------------------------------------------------
    # LIST DATASETS
    # -----------------------------------------------------------------

    def list_datasets(
        self,
        valid_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Return all registered datasets.
        """

        metadata = self._read_metadata()

        datasets = (
            metadata["datasets"]
        )

        if valid_only:

            datasets = [
                item
                for item in datasets
                if item.get("valid") is True
            ]

        return datasets

    # -----------------------------------------------------------------
    # GET DATASET
    # -----------------------------------------------------------------

    def get_dataset(
        self,
        dataset_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return metadata for a dataset ID."""

        datasets = self.list_datasets()

        for dataset in datasets:

            if dataset.get("id") == dataset_id:

                return dataset

        return None

    # -----------------------------------------------------------------
    # GET DATASET PATH
    # -----------------------------------------------------------------

    def get_dataset_path(
        self,
        dataset_id: str,
    ) -> Path:
        """
        Return the actual stored file path for a dataset.
        """

        dataset = self.get_dataset(
            dataset_id
        )

        if dataset is None:

            raise KeyError(
                f"Dataset not found: {dataset_id}"
            )

        path = (
            PROJECT_ROOT
            / dataset["path"]
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Stored dataset file not found: {path}"
            )

        return path

    # -----------------------------------------------------------------
    # DELETE DATASET
    # -----------------------------------------------------------------

    def delete_dataset(
        self,
        dataset_id: str,
    ) -> bool:
        """
        Delete a dataset and its metadata.
        """

        metadata = self._read_metadata()

        datasets = metadata[
            "datasets"
        ]

        target = None

        for dataset in datasets:

            if dataset.get("id") == dataset_id:

                target = dataset
                break

        if target is None:

            return False

        dataset_dir = (
            self.upload_dir
            / dataset_id
        )

        if dataset_dir.exists():

            shutil.rmtree(
                dataset_dir
            )

        metadata["datasets"] = [
            dataset
            for dataset in datasets
            if dataset.get("id")
            != dataset_id
        ]

        self._write_metadata(
            metadata
        )

        return True

    # -----------------------------------------------------------------
    # DATASET COUNT
    # -----------------------------------------------------------------

    def count(
        self,
        valid_only: bool = False,
    ) -> int:
        """Return the number of registered datasets."""

        return len(
            self.list_datasets(
                valid_only=valid_only
            )
        )


# ---------------------------------------------------------------------
# SIMPLE CLI TEST
# ---------------------------------------------------------------------

if __name__ == "__main__":

    manager = DatasetManager()

    print()
    print(
        "=" * 70
    )

    print(
        "APMOA DATASET MANAGER"
    )

    print(
        "=" * 70
    )

    print()

    print(
        f"Storage directory:"
    )

    print(
        manager.upload_dir
    )

    print()

    datasets = (
        manager.list_datasets()
    )

    print(
        f"Registered datasets: "
        f"{len(datasets)}"
    )

    for dataset in datasets:

        print()

        print(
            f"ID       : {dataset.get('id')}"
        )

        print(
            f"Name     : {dataset.get('name')}"
        )

        print(
            f"Format   : {dataset.get('format')}"
        )

        print(
            f"Events   : {dataset.get('events')}"
        )

        print(
            f"Cases    : {dataset.get('cases')}"
        )

        print(
            f"Activities: {dataset.get('activities')}"
        )

        print(
            f"Resources: {dataset.get('resources')}"
        )

        print(
            f"Status   : {dataset.get('status')}"
        )

    print()