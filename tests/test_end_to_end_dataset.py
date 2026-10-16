"""
APMOA End-to-End Dataset Independence Test

Purpose
-------
Validate that APMOA can process an event log whose original
column names are different from the canonical APMOA schema.

Test flow
---------

    Alternative Dataset
           |
           v
    DatasetAdapter
           |
           v
    Canonical Schema
    case_id
    activity
    timestamp
    resource
           |
           v
    Process Discovery
           |
           v
    Model Evaluation
           |
           v
    Model Selection
           |
           v
    Final Selected Model

This test does NOT use the BPI Travel Permit dataset.

It uses a small synthetic event log so that the test remains
fast and suitable for repeated development runs.
"""

import os
import tempfile

import pandas as pd

from src.data.dataset_adapter import DatasetAdapter
from src.agents.process_discovery_agent import ProcessDiscoveryAgent
from src.agents.evaluation_agent import ModelEvaluationAgent
from src.agents.model_selection_agent import ModelSelectionAgent


# ==============================================================
# CREATE SYNTHETIC ALTERNATIVE DATASET
# ==============================================================

def create_test_dataset():
    """
    Create a small event log using non-canonical column names.

    The purpose is to verify that APMOA does not depend on the
    original dataset column names.
    """

    data = {
        "Case ID": [
            "C001", "C001", "C001", "C001",
            "C002", "C002", "C002", "C002",
            "C003", "C003", "C003", "C003",
            "C004", "C004", "C004", "C004",
            "C005", "C005", "C005", "C005",
            "C006", "C006", "C006",
            "C007", "C007", "C007", "C007",
            "C008", "C008", "C008",
            "C009", "C009", "C009", "C009",
            "C010", "C010", "C010",
        ],

        "Activity Name": [
            # C001
            "Register", "Review", "Approve", "Complete",

            # C002
            "Register", "Review", "Approve", "Complete",

            # C003
            "Register", "Review", "Reject", "Complete",

            # C004
            "Register", "Review", "Approve", "Complete",

            # C005
            "Register", "Review", "Reject", "Complete",

            # C006
            "Register", "Review", "Approve",

            # C007
            "Register", "Review", "Approve", "Complete",

            # C008
            "Register", "Review", "Complete",

            # C009
            "Register", "Review", "Approve", "Complete",

            # C010
            "Register", "Review", "Reject",
        ],

        "Event Timestamp": [
            # C001
            "2025-01-01 09:00:00",
            "2025-01-01 10:00:00",
            "2025-01-01 11:00:00",
            "2025-01-01 12:00:00",

            # C002
            "2025-01-02 09:00:00",
            "2025-01-02 10:00:00",
            "2025-01-02 11:00:00",
            "2025-01-02 12:00:00",

            # C003
            "2025-01-03 09:00:00",
            "2025-01-03 10:00:00",
            "2025-01-03 11:00:00",
            "2025-01-03 12:00:00",

            # C004
            "2025-01-04 09:00:00",
            "2025-01-04 10:00:00",
            "2025-01-04 11:00:00",
            "2025-01-04 12:00:00",

            # C005
            "2025-01-05 09:00:00",
            "2025-01-05 10:00:00",
            "2025-01-05 11:00:00",
            "2025-01-05 12:00:00",

            # C006
            "2025-01-06 09:00:00",
            "2025-01-06 10:00:00",
            "2025-01-06 11:00:00",

            # C007
            "2025-01-07 09:00:00",
            "2025-01-07 10:00:00",
            "2025-01-07 11:00:00",
            "2025-01-07 12:00:00",

            # C008
            "2025-01-08 09:00:00",
            "2025-01-08 10:00:00",
            "2025-01-08 11:00:00",

            # C009
            "2025-01-09 09:00:00",
            "2025-01-09 10:00:00",
            "2025-01-09 11:00:00",
            "2025-01-09 12:00:00",

            # C010
            "2025-01-10 09:00:00",
            "2025-01-10 10:00:00",
            "2025-01-10 11:00:00",
        ],

        "User": [
            # C001
            "User1", "Reviewer1", "Manager1", "System",

            # C002
            "User2", "Reviewer1", "Manager1", "System",

            # C003
            "User3", "Reviewer2", "Manager2", "System",

            # C004
            "User1", "Reviewer1", "Manager1", "System",

            # C005
            "User4", "Reviewer2", "Manager2", "System",

            # C006
            "User5", "Reviewer1", "Manager1",

            # C007
            "User6", "Reviewer2", "Manager1", "System",

            # C008
            "User7", "Reviewer1", "System",

            # C009
            "User8", "Reviewer2", "Manager2", "System",

            # C010
            "User9", "Reviewer1", "Manager2",
        ],
    }

    return pd.DataFrame(data)


# ==============================================================
# PRINT DATASET INFORMATION
# ==============================================================

def print_dataset_info(dataframe, title):
    """Print basic information about an event log."""

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(f"Events     : {len(dataframe):,}")
    print(
        f"Cases      : "
        f"{dataframe['case_id'].nunique():,}"
    )
    print(
        f"Activities : "
        f"{dataframe['activity'].nunique():,}"
    )
    print(
        f"Resources  : "
        f"{dataframe['resource'].nunique():,}"
    )

    print("\nCanonical columns:")
    for column in [
        "case_id",
        "activity",
        "timestamp",
        "resource",
    ]:
        print(f"  ✓ {column}")


# ==============================================================
# MAIN TEST
# ==============================================================

def main():

    print("\n" + "=" * 70)
    print("APMOA END-TO-END DATASET INDEPENDENCE TEST")
    print("=" * 70)

    # ==========================================================
    # TEST 1 - CREATE ALTERNATIVE DATASET
    # ==========================================================

    print("\n")
    print("=" * 70)
    print("TEST 1 - CREATE ALTERNATIVE EVENT LOG")
    print("=" * 70)

    original_data = create_test_dataset()

    print("\nOriginal dataset columns:")

    for column in original_data.columns:
        print(f"  • {column}")

    assert "Case ID" in original_data.columns
    assert "Activity Name" in original_data.columns
    assert "Event Timestamp" in original_data.columns
    assert "User" in original_data.columns

    print("\n✓ Alternative column names created")

    # ==========================================================
    # TEST 2 - SAVE AS CSV
    # ==========================================================

    print("\n")
    print("=" * 70)
    print("TEST 2 - LOAD DATASET THROUGH DATASET ADAPTER")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:

        csv_path = os.path.join(
            temp_dir,
            "alternative_event_log.csv",
        )

        original_data.to_csv(
            csv_path,
            index=False,
        )

        print(f"\nTemporary dataset:")
        print(csv_path)

        adapter = DatasetAdapter()

        normalized_data = adapter.load(
            source=csv_path,
            file_type="csv",
        )

        print("\nDataset loaded successfully.")

        # ======================================================
        # TEST 3 - VERIFY CANONICAL SCHEMA
        # ======================================================

        print("\n")
        print("=" * 70)
        print("TEST 3 - CANONICAL SCHEMA VALIDATION")
        print("=" * 70)

        required_columns = [
            "case_id",
            "activity",
            "timestamp",
            "resource",
        ]

        for column in required_columns:

            assert column in normalized_data.columns, (
                f"Required canonical column missing: {column}"
            )

            print(f"✓ {column}")

        # Timestamp must be datetime-compatible.
        normalized_data["timestamp"] = pd.to_datetime(
            normalized_data["timestamp"],
            errors="raise",
        )

        print("\n✓ Alternative schema converted to canonical APMOA schema")

        print_dataset_info(
            normalized_data,
            "NORMALIZED DATASET",
        )

        # ======================================================
        # TEST 4 - VALIDATE DATASET
        # ======================================================

        print("\n")
        print("=" * 70)
        print("TEST 4 - DATASET VALIDATION")
        print("=" * 70)

        validation = adapter.validate(
            normalized_data
        )

        print("\nValidation result:")

        for key, value in validation.items():
            print(f"{key:<20}: {value}")

        assert validation["valid"] is True

        assert validation["events"] == len(
            normalized_data
        )

        assert validation["cases"] == (
            normalized_data["case_id"].nunique()
        )

        assert validation["activities"] == (
            normalized_data["activity"].nunique()
        )

        print("\n✓ Dataset validation passed")

        # ======================================================
        # TEST 5 - PROCESS DISCOVERY
        # ======================================================

        print("\n")
        print("=" * 70)
        print("TEST 5 - PROCESS DISCOVERY")
        print("=" * 70)

        discovery_output = os.path.join(
            temp_dir,
            "discovery_results",
        )

        discovery_agent = ProcessDiscoveryAgent(
            normalized_data,
            output_dir=discovery_output,
        )

        discovered_models = (
            discovery_agent.discover_all()
        )

        assert isinstance(
            discovered_models,
            dict,
        )

        print("\nDiscovered models:")

        for model_name in discovered_models:
            print(f"  ✓ {model_name}")

        successful_models = [
            name
            for name, model in discovered_models.items()
            if isinstance(model, dict)
            and model.get("status", "SUCCESS") == "SUCCESS"
        ]

        print(
            f"\nSuccessful models: "
            f"{len(successful_models)}"
        )

        assert len(successful_models) >= 1, (
            "No process discovery algorithm "
            "successfully generated a model."
        )

        print("\n✓ Process discovery accepted the alternative dataset")

        # ======================================================
        # TEST 6 - MODEL EVALUATION
        # ======================================================

        print("\n")
        print("=" * 70)
        print("TEST 6 - MODEL EVALUATION")
        print("=" * 70)

        evaluation_agent = ModelEvaluationAgent(
            normalized_data
        )

        evaluations = evaluation_agent.evaluate(
            discovered_models
        )

        print("\nEvaluation completed.")

        # The current Evaluation Agent returns a DataFrame.
        assert isinstance(
            evaluations,
            pd.DataFrame,
        )

        print("\nEvaluation results:")

        display_columns = [
            column
            for column in [
                "model",
                "fitness",
                "precision",
                "generalization",
                "simplicity",
                "status",
            ]
            if column in evaluations.columns
        ]

        print(
            evaluations[
                display_columns
            ].to_string(
                index=False
            )
        )

        successful_evaluations = evaluations[
            evaluations["status"] == "SUCCESS"
        ]

        assert len(successful_evaluations) >= 1, (
            "No discovered model was successfully evaluated."
        )

        print(
            f"\n✓ "
            f"{len(successful_evaluations)} "
            f"model(s) evaluated successfully"
        )

        # ======================================================
        # TEST 7 - MODEL SELECTION
        # ======================================================

        print("\n")
        print("=" * 70)
        print("TEST 7 - MODEL SELECTION")
        print("=" * 70)

        selection_agent = ModelSelectionAgent(
            evaluations
        )

        selection_result = (
            selection_agent.select()
        )

        print("\nSelection result:")

        print(
            f"Type: "
            f"{type(selection_result).__name__}"
        )

        print(
            f"Value:\n"
            f"{selection_result}"
        )

        # ------------------------------------------------------
        # Resolve selected model name.
        # ------------------------------------------------------

        selected_model = None

        if isinstance(
            selection_result,
            str,
        ):

            selected_model = (
                selection_result.strip()
            )

        elif isinstance(
            selection_result,
            dict,
        ):

            for key in [
                "selected_model",
                "model",
                "name",
                "model_name",
            ]:

                value = selection_result.get(
                    key
                )

                if isinstance(
                    value,
                    str,
                ) and value.strip():

                    selected_model = (
                        value.strip()
                    )

                    break

        assert selected_model, (
            "Model selection did not return "
            "a valid selected model."
        )

        print(
            f"\n✓ Selected model: "
            f"{selected_model}"
        )

        # ======================================================
        # TEST 8 - FINAL INDEPENDENCE CHECK
        # ======================================================

        print("\n")
        print("=" * 70)
        print("TEST 8 - DATASET INDEPENDENCE CHECK")
        print("=" * 70)

        print("\nOriginal columns:")
        print(
            list(
                original_data.columns
            )
        )

        print("\nCanonical columns:")
        print(
            list(
                normalized_data.columns
            )
        )

        # Ensure the downstream pipeline received canonical
        # columns rather than the original column names.

        assert "Case ID" not in normalized_data.columns
        assert "Activity Name" not in normalized_data.columns
        assert "Event Timestamp" not in normalized_data.columns
        assert "User" not in normalized_data.columns

        assert all(
            column in normalized_data.columns
            for column in required_columns
        )

        print(
            "\n✓ Original dataset-specific column names "
            "were abstracted by DatasetAdapter"
        )

        print(
            "✓ Discovery operated on canonical schema"
        )

        print(
            "✓ Evaluation operated on canonical schema"
        )

        print(
            "✓ Model selection completed"
        )

        # ======================================================
        # FINAL RESULT
        # ======================================================

        print("\n")
        print("=" * 70)
        print("END-TO-END DATASET INDEPENDENCE TEST PASSED")
        print("=" * 70)

        print("\nAPMOA successfully demonstrated:")

        print(
            "1. Alternative event-log column names"
        )

        print(
            "2. CSV-based dataset loading"
        )

        print(
            "3. Automatic schema normalization"
        )

        print(
            "4. Canonical APMOA event-log interface"
        )

        print(
            "5. Process discovery on a new dataset"
        )

        print(
            "6. Process model evaluation"
        )

        print(
            "7. Model selection"
        )

        print(
            "8. Dataset-independent downstream processing"
        )

        print("\nFinal selected model:")
        print(f"  {selected_model}")

        print("\n" + "=" * 70)
        print("DATASET INDEPENDENCE VERIFIED")
        print("=" * 70)


if __name__ == "__main__":
    main()