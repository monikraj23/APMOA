import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List


class ExperimentManager:
    """
    Lightweight experiment persistence for APMOA.
    Stores experiment results in JSON and CSV formats.
    """

    def __init__(self, experiments_dir: str = "results/experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.experiments = {}
        self._load_all()

    def _load_all(self):
        """Loads all existing experiment summaries into memory."""
        for summary_file in self.experiments_dir.glob("*/experiment_summary.json"):
            try:
                with open(summary_file, "r") as f:
                    data = json.load(f)
                    self.experiments[data["experiment_id"]] = data
            except Exception as e:
                print(f"Failed to load experiment {summary_file}: {e}")

    def create_experiment(self, config: Dict[str, Any]) -> str:
        """Initialize a new experiment run."""
        exp_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        exp_data = {
            "experiment_id": exp_id,
            "timestamp": timestamp,
            "configuration": config.get("configuration", {}),
            "algorithm_configuration": config.get("algorithm_configuration", {}),
            "selection_weights": config.get("selection_weights", {}),
            "quality_thresholds": config.get("quality_thresholds", {}),
            "rag_configuration": config.get("rag_configuration", {}),
            "llm_configuration": config.get("llm_configuration", {}),
            "datasets": {},
            "status": "Running"
        }

        self.experiments[exp_id] = exp_data
        self._save_experiment(exp_id)
        return exp_id

    def update_experiment_dataset(self, exp_id: str, dataset_name: str, results: Dict[str, Any]):
        """Update results for a specific dataset in the experiment."""
        if exp_id not in self.experiments:
            raise ValueError(f"Experiment {exp_id} not found.")

        self.experiments[exp_id]["datasets"][dataset_name] = results
        self._save_experiment(exp_id)

    def finish_experiment(self, exp_id: str, status: str = "Completed"):
        """Mark an experiment as completed."""
        if exp_id not in self.experiments:
            raise ValueError(f"Experiment {exp_id} not found.")

        self.experiments[exp_id]["status"] = status
        self.experiments[exp_id]["end_time"] = datetime.now().isoformat()
        self._save_experiment(exp_id)

    def _save_experiment(self, exp_id: str):
        """Persist experiment data to disk."""
        exp_dir = self.experiments_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        summary_path = exp_dir / "experiment_summary.json"
        with open(summary_path, "w") as f:
            json.dump(self.experiments[exp_id], f, indent=4)

        # Optional: Dump flat CSVs if needed here or rely on the JSON

    def get_experiments(self) -> List[Dict]:
        """Return list of all experiments."""
        return list(self.experiments.values())

    def get_experiment(self, exp_id: str) -> Dict:
        """Return a specific experiment."""
        return self.experiments.get(exp_id, {})
