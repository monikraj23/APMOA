from src.mining.loader import EventLogLoader

from src.agents.bottleneck_analysis_agent import (
    BottleneckAnalysisAgent,
)


LOG_PATH = (
    "data/raw/Travel/PermitLog.xes.gz"
)


def main():

    print("=" * 70)
    print(
        "APMOA - BOTTLENECK ANALYSIS AGENT TEST"
    )
    print("=" * 70)

    # ---------------------------------------------------------
    # LOAD EVENT LOG
    # ---------------------------------------------------------

    print(
        f"\nLoading event log: {LOG_PATH}"
    )

    loader = EventLogLoader(
        LOG_PATH
    )

    log = loader.load()

    print(
        f"Loaded {len(log):,} events."
    )

    # ---------------------------------------------------------
    # CREATE AGENT
    # ---------------------------------------------------------

    agent = BottleneckAnalysisAgent(
        log
    )

    # ---------------------------------------------------------
    # RUN ANALYSIS
    # ---------------------------------------------------------

    results = agent.analyze()

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    assert results is not None

    assert isinstance(
        results,
        dict,
    )

    required_sections = [
        "activity_analysis",
        "transition_analysis",
        "case_analysis",
        "rework_analysis",
        "bottlenecks",
        "deviations",
        "summary",
    ]

    for section in required_sections:

        assert section in results, (
            f"Missing result section: "
            f"{section}"
        )

    assert isinstance(
        results["activity_analysis"],
        list,
    )

    assert isinstance(
        results["transition_analysis"],
        list,
    )

    assert isinstance(
        results["bottlenecks"],
        list,
    )

    assert isinstance(
        results["deviations"],
        list,
    )

    assert isinstance(
        results["rework_analysis"],
        list,
    )

    assert isinstance(
        results["case_analysis"],
        dict,
    )

    assert isinstance(
        results["summary"],
        dict,
    )

    # ---------------------------------------------------------
    # BASIC CONTENT VALIDATION
    # ---------------------------------------------------------

    assert len(
        results["activity_analysis"]
    ) > 0

    assert (
        results["summary"]["events"]
        > 0
    )

    assert (
        results["summary"]["cases"]
        > 0
    )

    assert (
        results["summary"]["activities"]
        > 0
    )

    # ---------------------------------------------------------
    # ACTIVITY STRUCTURE
    # ---------------------------------------------------------

    first_activity = results[
        "activity_analysis"
    ][0]

    activity_fields = [
        "activity",
        "frequency",
        "percentage",
        "case_count",
        "average_duration_hours",
        "median_duration_hours",
        "maximum_duration_hours",
        "repetition_count",
    ]

    for field in activity_fields:

        assert field in first_activity, (
            f"Missing activity field: "
            f"{field}"
        )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "ANALYSIS SUMMARY"
    )

    print(
        "-" * 70
    )

    summary = results[
        "summary"
    ]

    for key, value in summary.items():

        print(
            f"{key:<35}: {value}"
        )

    # ---------------------------------------------------------
    # TOP BOTTLENECKS
    # ---------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "TOP BOTTLENECKS"
    )

    print(
        "-" * 70
    )

    for bottleneck in results[
        "bottlenecks"
    ][:10]:

        print(
            f"{bottleneck['activity']:<45}"
            f"score="
            f"{bottleneck['bottleneck_score']:.2f}"
        )

    # ---------------------------------------------------------
    # TOP DEVIATIONS
    # ---------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "TOP DEVIATIONS"
    )

    print(
        "-" * 70
    )

    for deviation in results[
        "deviations"
    ][:5]:

        print(
            f"Cases={deviation['case_count']:<6}"
            f" "
            f"Deviation="
            f"{deviation['deviation_ratio']:.2f}"
        )

    # ---------------------------------------------------------
    # FINAL TEST RESULT
    # ---------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "BOTTLENECK ANALYSIS AGENT TEST PASSED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()