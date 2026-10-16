from loader import EventLogLoader


LOG_PATH = "data/raw/Travel/PermitLog.xes.gz"


def main():

    print("=" * 60)
    print("APMOA - GENERIC EVENT LOG LOADER")
    print("=" * 60)

    # Create loader
    loader = EventLogLoader(LOG_PATH)

    # Load dataset
    loader.load()

    # Validate dataset
    loader.validate()

    # Get summary
    summary = loader.get_summary()

    print("\n" + "-" * 60)
    print("EVENT LOG SUMMARY")
    print("-" * 60)

    for key, value in summary.items():
        print(f"{key:15}: {value}")

    # Standardize
    standardized_log = loader.get_standardized_log()

    print("\n" + "-" * 60)
    print("STANDARDIZED COLUMNS")
    print("-" * 60)

    print(list(standardized_log.columns))

    print("\n" + "-" * 60)
    print("FIRST 5 EVENTS")
    print("-" * 60)

    print(
        standardized_log[
            ["case_id", "activity", "timestamp"]
        ].head(5).to_string(index=False)
    )


if __name__ == "__main__":
    main()