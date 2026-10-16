from loader import EventLogLoader
from analyzer import ProcessAnalyzer


LOG_PATH = "data/raw/Travel/PermitLog.xes.gz"


def main():

    print("=" * 60)
    print("APMOA - PROCESS ANALYZER")
    print("=" * 60)

    # ---------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------

    loader = EventLogLoader(LOG_PATH)

    loader.load()
    loader.validate()

    standardized_log = loader.get_standardized_log()

    # ---------------------------------------------------------
    # ANALYZE
    # ---------------------------------------------------------

    analyzer = ProcessAnalyzer(standardized_log)

    # ---------------------------------------------------------
    # BASIC STATISTICS
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("BASIC PROCESS STATISTICS")
    print("-" * 60)

    statistics = analyzer.get_basic_statistics()

    for key, value in statistics.items():
        print(f"{key:25}: {value:,}")

    # ---------------------------------------------------------
    # OVERALL SUMMARY
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("PROCESS SUMMARY")
    print("-" * 60)

    summary = analyzer.get_summary()

    for key, value in summary.items():
        print(f"{key:25}: {value}")

    # ---------------------------------------------------------
    # TOP ACTIVITIES
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("TOP 15 ACTIVITIES")
    print("-" * 60)

    activities = analyzer.get_activity_frequency()

    print(
        activities.head(15).to_string(index=False)
    )

    # ---------------------------------------------------------
    # PROCESS VARIANTS
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("TOP 10 PROCESS VARIANTS")
    print("-" * 60)

    variants = analyzer.get_variants()

    for index, row in variants.head(10).iterrows():

        print(
            f"\nVariant {index + 1}"
        )

        print(
            f"Cases      : {row['case_count']:,}"
        )

        print(
            f"Percentage : {row['percentage']:.2f}%"
        )

        print(
            "Sequence   :"
        )

        for activity in row["variant"]:
            print(f"  → {activity}")

    # ---------------------------------------------------------
    # REPEATED ACTIVITIES
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("REPEATED ACTIVITIES")
    print("-" * 60)

    repeated = analyzer.get_repeated_activities()

    print(
        repeated.head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()