"""
APMOA - Data Analysis Agent Test

Tests the DataAnalysisAgent using the Travel event log.

Run:
    python -m src.agents.test_data_analysis_agent
"""

from src.mining.loader import EventLogLoader
from src.agents.data_analysis_agent import DataAnalysisAgent


# ============================================================
# CONFIGURATION
# ============================================================

LOG_PATH = (
    "data/raw/Travel/PermitLog.xes.gz"
)


# ============================================================
# HELPER
# ============================================================

def print_section(title):

    print(
        "\n" + "-" * 70
    )

    print(title)

    print(
        "-" * 70
    )


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "APMOA - DATA ANALYSIS AGENT TEST"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # LOAD EVENT LOG
    # ========================================================

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

    # ========================================================
    # CREATE AGENT
    # ========================================================

    print_section(
        "INITIALIZING DATA ANALYSIS AGENT"
    )

    agent = DataAnalysisAgent(
        log
    )

    print(
        "Data Analysis Agent initialized."
    )

    # ========================================================
    # RUN ANALYSIS
    # ========================================================

    print_section(
        "RUNNING DATA ANALYSIS"
    )

    results = agent.analyze()

    # ========================================================
    # VALIDATE RESULT
    # ========================================================

    assert results is not None, (
        "Data Analysis Agent returned None."
    )

    assert isinstance(
        results,
        dict
    ), (
        "Data Analysis Agent must return a dictionary."
    )

    required_sections = [
        "basic_statistics",
        "variant_statistics",
        "activity_statistics",
        "case_duration",
        "repetition_statistics",
    ]

    for section in required_sections:

        assert section in results, (
            f"Missing analysis section: {section}"
        )

    # ========================================================
    # BASIC STATISTICS
    # ========================================================

    print_section(
        "BASIC STATISTICS"
    )

    basic_statistics = results[
        "basic_statistics"
    ]

    assert isinstance(
        basic_statistics,
        dict
    )

    for key, value in basic_statistics.items():

        print(
            f"{key:<40}: {value}"
        )

    # ========================================================
    # VARIANT STATISTICS
    # ========================================================

    print_section(
        "VARIANT STATISTICS"
    )

    variant_statistics = results[
        "variant_statistics"
    ]

    assert isinstance(
        variant_statistics,
        dict
    )

    for key, value in variant_statistics.items():

        print(
            f"{key:<40}: {value}"
        )

    # ========================================================
    # ACTIVITY STATISTICS
    # ========================================================

    print_section(
        "TOP 10 ACTIVITIES"
    )

    activities = results[
        "activity_statistics"
    ]

    assert isinstance(
        activities,
        list
    )

    print(
        f"{'Activity':<50}"
        f"{'Frequency':>12}"
        f"{'Percentage':>12}"
    )

    print(
        "-" * 74
    )

    for activity in activities[:10]:

        assert (
            "activity"
            in activity
        )

        assert (
            "frequency"
            in activity
        )

        assert (
            "percentage"
            in activity
        )

        print(
            f"{str(activity['activity']):<50}"
            f"{activity['frequency']:>12}"
            f"{activity['percentage']:>11.2f}%"
        )

    # ========================================================
    # CASE DURATION
    # ========================================================

    print_section(
        "CASE DURATION"
    )

    case_duration = results[
        "case_duration"
    ]

    assert isinstance(
        case_duration,
        dict
    )

    for key, value in case_duration.items():

        print(
            f"{key:<40}: {value}"
        )

    # ========================================================
    # REPETITION STATISTICS
    # ========================================================

    print_section(
        "LOOP / REPETITION STATISTICS"
    )

    repetition_statistics = results[
        "repetition_statistics"
    ]

    assert isinstance(
        repetition_statistics,
        dict
    )

    for key, value in repetition_statistics.items():

        print(
            f"{key:<40}: {value}"
        )

    # ========================================================
    # DATA QUALITY CHECKS
    # ========================================================

    print_section(
        "DATA ANALYSIS VALIDATION"
    )

    # --------------------------------------------------------
    # Event count
    # --------------------------------------------------------

    assert len(log) > 0, (
        "Event log contains no events."
    )

    print(
        "✓ Event log contains events"
    )

    # --------------------------------------------------------
    # Activity statistics
    # --------------------------------------------------------

    assert len(activities) > 0, (
        "No activity statistics were produced."
    )

    print(
        "✓ Activity statistics generated"
    )

    # --------------------------------------------------------
    # Variant statistics
    # --------------------------------------------------------

    assert len(variant_statistics) > 0, (
        "No variant statistics were produced."
    )

    print(
        "✓ Variant statistics generated"
    )

    # --------------------------------------------------------
    # Duration statistics
    # --------------------------------------------------------

    assert len(case_duration) > 0, (
        "No case-duration statistics were produced."
    )

    print(
        "✓ Case duration statistics generated"
    )

    # --------------------------------------------------------
    # Repetition statistics
    # --------------------------------------------------------

    assert len(repetition_statistics) > 0, (
        "No repetition statistics were produced."
    )

    print(
        "✓ Repetition statistics generated"
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "DATA ANALYSIS AGENT TEST PASSED"
    )

    print(
        "=" * 70
    )

    print(
        "\nAll required process diagnostics "
        "were successfully generated."
    )

    print(
        "\nThe diagnostics are ready to be consumed by:"
    )

    print(
        "  → Process Discovery Agent"
    )

    print(
        "  → Model Evaluation Agent"
    )

    print(
        "  → Reasoning Agent"
    )

    print(
        "  → Adaptation Agent"
    )

    print(
        "\n" + "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()