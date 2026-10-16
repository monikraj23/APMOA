import pm4py
import pandas as pd
from pathlib import Path


# Path to the raw event log
LOG_PATH = Path("data/raw/Travel/PermitLog.xes.gz")


def main():
    print("=" * 60)
    print("APMOA - EVENT LOG INSPECTION")
    print("=" * 60)

    print(f"\nLoading event log:")
    print(LOG_PATH)

    if not LOG_PATH.exists():
        print("\nERROR: Event log not found.")
        print("Please check the file path.")
        return

    # Load XES event log
    log = pm4py.read_xes(str(LOG_PATH))

    print("\nLog loaded successfully.")

    # ---------------------------------------------------------
    # BASIC INFORMATION
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("BASIC INFORMATION")
    print("-" * 60)

    print(f"Data type             : {type(log).__name__}")
    print(f"Number of events      : {len(log):,}")

    # ---------------------------------------------------------
    # COLUMNS
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("EVENT LOG COLUMNS")
    print("-" * 60)

    for column in log.columns:
        print(f"- {column}")

    # ---------------------------------------------------------
    # IDENTIFY IMPORTANT COLUMNS
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("IMPORTANT EVENT ATTRIBUTES")
    print("-" * 60)

    case_column = "case:concept:name"
    activity_column = "concept:name"
    timestamp_column = "time:timestamp"

    if case_column in log.columns:
        number_of_cases = log[case_column].nunique()
        print(f"Case ID column        : {case_column}")
        print(f"Number of cases       : {number_of_cases:,}")
    else:
        print(f"Case ID column        : NOT FOUND")

    if activity_column in log.columns:
        number_of_activities = log[activity_column].nunique()
        print(f"Activity column       : {activity_column}")
        print(f"Number of activities  : {number_of_activities:,}")
    else:
        print("Activity column       : NOT FOUND")

    if timestamp_column in log.columns:
        print(f"Timestamp column      : {timestamp_column}")

        print(f"Start timestamp       : {log[timestamp_column].min()}")
        print(f"End timestamp         : {log[timestamp_column].max()}")
    else:
        print("Timestamp column      : NOT FOUND")

    # ---------------------------------------------------------
    # DATA TYPES
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("DATA TYPES")
    print("-" * 60)

    print(log.dtypes)

    # ---------------------------------------------------------
    # MISSING VALUES
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("MISSING VALUES")
    print("-" * 60)

    missing_values = log.isnull().sum()

    for column, count in missing_values.items():
        if count > 0:
            print(f"{column}: {count:,}")

    if missing_values.sum() == 0:
        print("No missing values found.")

    # ---------------------------------------------------------
    # FIRST 5 EVENTS
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("FIRST 5 EVENTS")
    print("-" * 60)

    print(log.head(5).to_string())

    # ---------------------------------------------------------
    # FIRST CASE
    # ---------------------------------------------------------

    if case_column in log.columns:

        first_case_id = log[case_column].iloc[0]

        first_case = log[
            log[case_column] == first_case_id
        ]

        print("\n" + "-" * 60)
        print("FIRST CASE")
        print("-" * 60)

        print(f"Case ID: {first_case_id}")
        print(f"Number of events: {len(first_case)}")

        print("\nEvents:")

        columns_to_show = [
            column
            for column in [
                case_column,
                activity_column,
                timestamp_column
            ]
            if column in log.columns
        ]

        print(
            first_case[columns_to_show]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
