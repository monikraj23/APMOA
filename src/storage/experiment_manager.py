"""
APMOA Experiment Manager

Persistent storage manager for APMOA experiment executions.

Development backend:
    Local filesystem + JSON

Storage structure:
    results/
        experiments/
            EXP-YYYYMMDD-HHMMSS/
                metadata.json
                final_result.json
                evaluation_results.json
                reasoning.json
                adaptation.json
                process_intelligence.json
                optimization.json
                execution.log
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperimentManager:
    """
    Manage persistent APMOA experiment results.

    The manager intentionally exposes a storage-oriented interface so that
    the backend can later be replaced by PostgreSQL/object storage without
    changing the agents or orchestrator.
    """

    def __init__(
        self,
        experiments_dir: str = "results/experiments",
    ):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _timestamp() -> str:
        """Return a filesystem-safe timestamp."""
        return datetime.now().strftime("%Y%m%d-%H%M%S")

    def _experiment_path(self, experiment_id: str) -> Path:
        """Return the directory for an experiment."""
        return self.experiments_dir / experiment_id

    @staticmethod
    def _write_json(
        path: Path,
        data: Any,
    ) -> None:
        """Write JSON safely."""
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        """Read a JSON file."""
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    # ------------------------------------------------------------------
    # EXPERIMENT CREATION
    # ------------------------------------------------------------------

    def create_experiment(
        self,
        dataset_name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new experiment directory.

        Returns:
            Experiment ID.
        """

        base_timestamp = self._timestamp()
        experiment_id = f"EXP-{base_timestamp}"

        # Avoid collisions when multiple experiments start within
        # the same second.
        counter = 1

        while self._experiment_path(experiment_id).exists():
            experiment_id = (
                f"EXP-{base_timestamp}-{counter:02d}"
            )
            counter += 1

        experiment_path = self._experiment_path(
            experiment_id
        )

        experiment_path.mkdir(
            parents=True,
            exist_ok=False,
        )

        experiment_metadata = {
            "experiment_id": experiment_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "dataset_name": dataset_name,
            "dataset_path": dataset_path,
            "description": description,
            "status": "created",
        }

        if metadata:
            experiment_metadata.update(metadata)

        self._write_json(
            experiment_path / "metadata.json",
            experiment_metadata,
        )

        return experiment_id

    # ------------------------------------------------------------------
    # SAVE INDIVIDUAL RESULT COMPONENTS
    # ------------------------------------------------------------------

    def save_component(
        self,
        experiment_id: str,
        component_name: str,
        data: Any,
    ) -> Path:
        """
        Save one experiment component.

        Example:
            save_component(
                experiment_id,
                "reasoning",
                reasoning_result,
            )
        """

        experiment_path = self._experiment_path(
            experiment_id
        )

        if not experiment_path.exists():
            raise FileNotFoundError(
                f"Experiment not found: {experiment_id}"
            )

        allowed_components = {
            "final_result",
            "evaluation_results",
            "reasoning",
            "adaptation",
            "process_intelligence",
            "optimization",
        }

        if component_name not in allowed_components:
            raise ValueError(
                f"Unsupported experiment component: "
                f"{component_name}. "
                f"Allowed components: "
                f"{sorted(allowed_components)}"
            )

        output_path = (
            experiment_path
            / f"{component_name}.json"
        )

        self._write_json(
            output_path,
            data,
        )

        self._update_metadata(
            experiment_id,
            status="running",
        )

        return output_path

    # ------------------------------------------------------------------
    # SAVE EXECUTION LOG
    # ------------------------------------------------------------------

    def save_execution_log(
        self,
        experiment_id: str,
        log_content: str,
    ) -> Path:
        """Save the complete orchestrator execution log."""

        experiment_path = self._experiment_path(
            experiment_id
        )

        if not experiment_path.exists():
            raise FileNotFoundError(
                f"Experiment not found: {experiment_id}"
            )

        log_path = (
            experiment_path / "execution.log"
        )

        log_path.write_text(
            log_content,
            encoding="utf-8",
        )

        return log_path

    # ------------------------------------------------------------------
    # SAVE COMPLETE EXPERIMENT
    # ------------------------------------------------------------------

    def save_experiment(
        self,
        experiment_id: str,
        *,
        final_result: Optional[Any] = None,
        evaluation_results: Optional[Any] = None,
        reasoning: Optional[Any] = None,
        adaptation: Optional[Any] = None,
        process_intelligence: Optional[Any] = None,
        optimization: Optional[Any] = None,
        execution_log: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save all available experiment results.

        Only supplied components are written.
        """

        experiment_path = self._experiment_path(
            experiment_id
        )

        if not experiment_path.exists():
            raise FileNotFoundError(
                f"Experiment not found: {experiment_id}"
            )

        components = {
            "final_result": final_result,
            "evaluation_results": evaluation_results,
            "reasoning": reasoning,
            "adaptation": adaptation,
            "process_intelligence": process_intelligence,
            "optimization": optimization,
        }

        for component_name, data in components.items():
            if data is not None:
                self.save_component(
                    experiment_id,
                    component_name,
                    data,
                )

        if execution_log is not None:
            self.save_execution_log(
                experiment_id,
                execution_log,
            )

        if metadata:
            self._update_metadata(
                experiment_id,
                **metadata,
            )

        self._update_metadata(
            experiment_id,
            status="completed",
        )

        return experiment_path

    # ------------------------------------------------------------------
    # METADATA
    # ------------------------------------------------------------------

    def _update_metadata(
        self,
        experiment_id: str,
        **updates: Any,
    ) -> None:
        """Update experiment metadata."""

        experiment_path = self._experiment_path(
            experiment_id
        )

        metadata_path = (
            experiment_path / "metadata.json"
        )

        if metadata_path.exists():
            metadata = self._read_json(
                metadata_path
            )
        else:
            metadata = {
                "experiment_id": experiment_id
            }

        metadata.update(updates)
        metadata["updated_at"] = datetime.now().isoformat()

        self._write_json(
            metadata_path,
            metadata,
        )

    def get_metadata(
        self,
        experiment_id: str,
    ) -> Dict[str, Any]:
        """Return experiment metadata."""

        experiment_path = self._experiment_path(
            experiment_id
        )

        metadata_path = (
            experiment_path / "metadata.json"
        )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Experiment metadata not found: "
                f"{experiment_id}"
            )

        return self._read_json(
            metadata_path
        )

    # ------------------------------------------------------------------
    # LOAD EXPERIMENT
    # ------------------------------------------------------------------

    def get_experiment(
        self,
        experiment_id: str,
    ) -> Dict[str, Any]:
        """
        Load all persisted components for an experiment.
        """

        experiment_path = self._experiment_path(
            experiment_id
        )

        if not experiment_path.exists():
            raise FileNotFoundError(
                f"Experiment not found: {experiment_id}"
            )

        result: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "path": str(experiment_path),
        }

        json_components = [
            "metadata",
            "final_result",
            "evaluation_results",
            "reasoning",
            "adaptation",
            "process_intelligence",
            "optimization",
        ]

        for component in json_components:
            path = experiment_path / f"{component}.json"

            if path.exists():
                result[component] = self._read_json(
                    path
                )

        execution_log = (
            experiment_path / "execution.log"
        )

        if execution_log.exists():
            result["execution_log"] = (
                execution_log.read_text(
                    encoding="utf-8"
                )
            )

        return result

    # ------------------------------------------------------------------
    # LIST EXPERIMENTS
    # ------------------------------------------------------------------

    def list_experiments(
        self,
    ) -> List[Dict[str, Any]]:
        """
        List all persisted experiments.

        Returns newest experiments first.
        """

        experiments: List[Dict[str, Any]] = []

        for path in self.experiments_dir.iterdir():

            if not path.is_dir():
                continue

            if not path.name.startswith("EXP-"):
                continue

            metadata_path = (
                path / "metadata.json"
            )

            if metadata_path.exists():
                try:
                    metadata = self._read_json(
                        metadata_path
                    )
                except Exception:
                    metadata = {
                        "experiment_id": path.name,
                        "status": "corrupted_metadata",
                    }
            else:
                metadata = {
                    "experiment_id": path.name,
                    "status": "unknown",
                }

            experiments.append(metadata)

        experiments.sort(
            key=lambda item: item.get(
                "created_at",
                "",
            ),
            reverse=True,
        )

        return experiments

    # ------------------------------------------------------------------
    # PATH
    # ------------------------------------------------------------------

    def get_experiment_path(
        self,
        experiment_id: str,
    ) -> Path:
        """Return the experiment directory path."""

        path = self._experiment_path(
            experiment_id
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Experiment not found: {experiment_id}"
            )

        return path

    # ------------------------------------------------------------------
    # DELETE
    # ------------------------------------------------------------------

    def delete_experiment(
        self,
        experiment_id: str,
    ) -> bool:
        """Delete an experiment directory."""

        path = self._experiment_path(
            experiment_id
        )

        if not path.exists():
            return False

        if not path.is_dir():
            raise ValueError(
                f"Experiment path is not a directory: "
                f"{path}"
            )

        shutil.rmtree(path)

        return True

    # ------------------------------------------------------------------
    # COUNT
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return number of persisted experiments."""

        return len(
            self.list_experiments()
        )


__all__ = [
    "ExperimentManager",
]