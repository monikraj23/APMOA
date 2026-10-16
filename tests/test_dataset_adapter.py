"""
Tests for the APMOA Dataset Adapter.
"""

import pandas as pd

from src.data.dataset_adapter import DatasetAdapter


def main():

    print("\n" + "=" * 70)
    print("APMOA DATASET ADAPTER TEST")
    print("=" * 70)

    # ==========================================================
    # TEST 1
    # Standard APMOA schema
    # ==========================================================

    print("\nTEST 1 - Canonical schema")
    print("-" * 70)

    data_1 = pd.DataFrame(
        {
            "case_id": ["C1", "C1", "C2", "C2"],
            "activity": [
                "Submit",
                "Approve",
                "Submit",
                "Reject",
            ],
            "timestamp": [
                "2025-01-01 09:00:00",
                "2025-01-01 10:00:00",
                "2025-01-02 09:00:00",
                "2025-01-02 11:00:00",
            ],
            "resource": [
                "User1",
                "Manager1",
                "User2",
                "Manager2",
            ],
        }
    )

    adapter = DatasetAdapter()

    normalized_1 = adapter.normalize(
        data_1
    )

    print(
        normalized_1[
            [
                "case_id",
                "activity",
                "timestamp",
                "resource",
            ]
        ]
    )

    assert all(
        column in normalized_1.columns
        for column in [
            "case_id",
            "activity",
            "timestamp",
            "resource",
        ]
    )

    print("✓ Canonical schema accepted")

    # ==========================================================
    # TEST 2
    # Alternative column names
    # ==========================================================

    print("\nTEST 2 - Alternative column names")
    print("-" * 70)

    data_2 = pd.DataFrame(
        {
            "Case ID": ["A1", "A1", "A2", "A2"],
            "Activity Name": [
                "Create",
                "Review",
                "Create",
                "Complete",
            ],
            "Event Timestamp": [
                "2025-02-01 09:00:00",
                "2025-02-01 10:00:00",
                "2025-02-02 09:00:00",
                "2025-02-02 12:00:00",
            ],
            "User": [
                "Alice",
                "Bob",
                "Charlie",
                "David",
            ],
        }
    )

    normalized_2 = adapter.normalize(
        data_2
    )

    print(
        normalized_2[
            [
                "case_id",
                "activity",
                "timestamp",
                "resource",
            ]
        ]
    )

    assert list(
        normalized_2[
            [
                "case_id",
                "activity",
                "timestamp",
                "resource",
            ]
        ].columns
    ) == [
        "case_id",
        "activity",
        "timestamp",
        "resource",
    ]

    print("✓ Alternative column names detected")

    # ==========================================================
    # TEST 3
    # Resource missing
    # ==========================================================

    print("\nTEST 3 - Dataset without resource")
    print("-" * 70)

    data_3 = pd.DataFrame(
        {
            "case": ["B1", "B1", "B2"],
            "event": [
                "Start",
                "Process",
                "End",
            ],
            "time": [
                "2025-03-01 08:00:00",
                "2025-03-01 09:00:00",
                "2025-03-02 10:00:00",
            ],
        }
    )

    normalized_3 = adapter.normalize(
        data_3
    )

    print(
        normalized_3[
            [
                "case_id",
                "activity",
                "timestamp",
                "resource",
            ]
        ]
    )

    assert "resource" in normalized_3.columns

    assert (
        normalized_3["resource"]
        == "UNKNOWN"
    ).all()

    print("✓ Missing resource handled")

    # ==========================================================
    # TEST 4
    # Validation
    # ==========================================================

    print("\nTEST 4 - Dataset validation")
    print("-" * 70)

    validation = adapter.validate(
        normalized_1
    )

    print(validation)

    assert validation["valid"] is True
    assert validation["events"] == 4
    assert validation["cases"] == 2
    assert validation["activities"] == 3
    assert validation["resources"] == 4

    print("✓ Dataset validation passed")

    # ==========================================================
    # TEST 5
    # Summary
    # ==========================================================

    print("\nTEST 5 - Dataset summary")
    print("-" * 70)

    summary = adapter.get_summary(
        normalized_1
    )

    for key, value in summary.items():

        print(
            f"{key:<15}: {value}"
        )

    assert summary["events"] == 4
    assert summary["cases"] == 2

    print("✓ Dataset summary generated")

    # ==========================================================
    # FINAL RESULT
    # ==========================================================

    print("\n" + "=" * 70)
    print("DATASET ADAPTER TEST PASSED")
    print("=" * 70)

    print("\nAPMOA now supports a canonical event-log interface:")
    print()
    print("Case ID")
    print("Activity")
    print("Timestamp")
    print("Resource")
    print()
    print("The downstream agents can operate on the normalized schema")
    print("without knowing the original dataset column names.")


if __name__ == "__main__":
    main()