"""
APMOA - Optimization / Recommendation Agent Test

Standalone validation for OptimizationAgent.

This test deliberately uses the same structural format
produced by the BottleneckAnalysisAgent.

The test verifies:

    1. Agent initialization
    2. Bottleneck recommendations
    3. Transition recommendations
    4. Rework recommendations
    5. Deviation recommendations
    6. Recommendation ranking
    7. Required recommendation fields
    8. Summary generation
    9. Empty-input safety
    10. No fabricated recommendations
"""

from __future__ import annotations

from src.agents.optimization_agent import (
    OptimizationAgent,
)


# =============================================================
# TEST DATA
# =============================================================

def build_test_analysis():

    return {
        "activity_analysis": [
            {
                "activity":
                    "Send Reminder",

                "frequency":
                    100,

                "percentage":
                    1.0,

                "case_count":
                    80,

                "average_duration_hours":
                    0.0,

                "median_duration_hours":
                    0.0,

                "maximum_duration_hours":
                    0.0,

                "repetition_count":
                    5,
            }
        ],

        "transition_analysis": [
            {
                "from_activity":
                    "Request Payment",

                "to_activity":
                    "Payment Handled",

                "occurrences":
                    100,

                "average_waiting_hours":
                    20.0,

                "median_waiting_hours":
                    10.0,

                "p75_waiting_hours":
                    25.0,

                "maximum_waiting_hours":
                    100.0,
            },

            {
                "from_activity":
                    "Payment Handled",

                "to_activity":
                    "Send Reminder",

                "occurrences":
                    50,

                "average_waiting_hours":
                    5.0,

                "median_waiting_hours":
                    3.0,

                "p75_waiting_hours":
                    8.0,

                "maximum_waiting_hours":
                    20.0,
            },
        ],

        "case_analysis": {
            "number_of_cases":
                100,

            "average_case_duration_hours":
                50.0,

            "median_case_duration_hours":
                40.0,

            "p75_case_duration_hours":
                70.0,

            "p95_case_duration_hours":
                120.0,

            "maximum_case_duration_hours":
                200.0,
        },

        "rework_analysis": [
            {
                "activity":
                    "Permit SUBMITTED by EMPLOYEE",

                "cases_containing_activity":
                    50,

                "repetition_count":
                    20,
            },

            {
                "activity":
                    "Permit APPROVED by ADMINISTRATION",

                "cases_containing_activity":
                    30,

                "repetition_count":
                    10,
            },
        ],

        "bottlenecks": [
            {
                "activity":
                    "Send Reminder",

                "bottleneck_score":
                    1.00,

                "indicators":
                    4,

                "average_waiting_hours":
                    20.0,

                "average_duration_hours":
                    0.0,

                "frequency":
                    100,

                "repetition_count":
                    5,

                "reasons": [
                    "high waiting time",
                    "high activity frequency",
                    "repeated activity / rework",
                ],
            },

            {
                "activity":
                    "Payment Handled",

                "bottleneck_score":
                    0.75,

                "indicators":
                    3,

                "average_waiting_hours":
                    10.0,

                "average_duration_hours":
                    5.0,

                "frequency":
                    80,

                "repetition_count":
                    0,

                "reasons": [
                    "high waiting time",
                    "high activity frequency",
                ],
            },
        ],

        "deviations": [
            {
                "variant":
                    "A -> B -> C",

                "case_count":
                    20,

                "percentage":
                    20.0,

                "sequence_length":
                    3,

                "difference_from_dominant":
                    3,

                "deviation_ratio":
                    1.00,

                "dominant_variant_percentage":
                    50.0,
            },

            {
                "variant":
                    "A -> X -> C",

                "case_count":
                    10,

                "percentage":
                    10.0,

                "sequence_length":
                    3,

                "difference_from_dominant":
                    2,

                "deviation_ratio":
                    0.80,

                "dominant_variant_percentage":
                    50.0,
            },
        ],
    }


# =============================================================
# TEST 1 - INITIALIZATION
# =============================================================

def test_initialization():

    analysis = build_test_analysis()

    agent = OptimizationAgent(
        bottleneck_analysis=analysis
    )

    assert agent is not None

    print(
        "✓ Initialization test passed"
    )


# =============================================================
# TEST 2 - BASIC OUTPUT
# =============================================================

def test_basic_output():

    analysis = build_test_analysis()

    agent = OptimizationAgent(
        bottleneck_analysis=analysis
    )

    result = agent.optimize()

    assert isinstance(
        result,
        dict,
    )

    assert (
        result["status"]
        == "SUCCESS"
    )

    assert (
        "recommendations"
        in result
    )

    assert (
        "summary"
        in result
    )

    print(
        "✓ Basic output test passed"
    )


# =============================================================
# TEST 3 - BOTTLENECK RECOMMENDATION
# =============================================================

def test_bottleneck_recommendation():

    analysis = build_test_analysis()

    agent = OptimizationAgent(
        bottleneck_analysis=analysis
    )

    result = agent.optimize()

    recommendations = (
        result[
            "recommendations"
        ]
    )

    bottleneck_recommendations = [
        item
        for item in recommendations
        if item.get(
            "category"
        )
        == "Bottleneck Reduction"
    ]

    assert (
        len(
            bottleneck_recommendations
        )
        > 0
    )

    assert any(
        item.get(
            "affected_activity"
        )
        == "Send Reminder"
        for item
        in bottleneck_recommendations
    )

    print(
        "✓ Bottleneck recommendation test passed"
    )


# =============================================================
# TEST 4 - TRANSITION RECOMMENDATION
# =============================================================

def test_transition_recommendation():

    analysis = build_test_analysis()

    agent = OptimizationAgent(
        bottleneck_analysis=analysis
    )

    result = agent.optimize()

    recommendations = (
        result[
            "recommendations"
        ]
    )

    transition_recommendations = [
        item
        for item in recommendations
        if item.get(
            "category"
        )
        == "Waiting-Time Reduction"
    ]

    assert (
        len(
            transition_recommendations
        )
        > 0
    )

    assert any(
        "Request Payment"
        in item.get(
            "affected_activity",
            "",
        )
        for item
        in transition_recommendations
    )

    print(
        "✓ Transition recommendation test passed"
    )


# =============================================================
# TEST 5 - REWORK RECOMMENDATION
# =============================================================

def test_rework_recommendation():

    analysis = build_test_analysis()

    agent = OptimizationAgent(
        bottleneck_analysis=analysis
    )

    result = agent.optimize()

    recommendations = (
        result[
            "recommendations"
        ]
    )

    rework_recommendations = [
        item
        for item in recommendations
        if item.get(
            "category"
        )
        == "Rework Reduction"
    ]

    assert (
        len(
            rework_recommendations
        )
        > 0
    )

    assert any(
        item.get(
            "affected_activity"
        )
        == "Permit SUBMITTED by EMPLOYEE"
        for item
        in rework_recommendations
    )

    print(
        "✓ Rework recommendation test passed"
    )


# =============================================================
# TEST 6 - DEVIATION RECOMMENDATION
# =============================================================

def test_deviation_recommendation():

    analysis = build_test_analysis()

    agent = OptimizationAgent(
        bottleneck_analysis=analysis
    )

    result = agent.optimize()

    recommendations = (
        result[
            "recommendations"
        ]
    )

    deviation_recommendations = [
        item
        for item in recommendations
        if item.get(
            "category"
        )
        == "Deviation Investigation"
    ]

    assert (
        len(
            deviation_recommendations
        )
        > 0
    )

    print(
        "✓ Deviation recommendation test passed"
    )


# =============================================================
# TEST 7 - REQUIRED FIELDS
# =============================================================

def test_required_fields():

    analysis = build_test_analysis()

    agent = OptimizationAgent(
        bottleneck_analysis=analysis
    )

    result = agent.optimize()

    required_fields = [
        "recommendation_id",
        "category",
        "priority",
        "affected_activity",
        "problem",
        "evidence",
        "action",
        "expected_benefit",
        "confidence",
        "rank",
    ]

    recommendations = (
        result[
            "recommendations"
        ]
    )

    assert (
        len(recommendations)
        > 0
    )

    for recommendation in (
        recommendations
    ):

        for field in (
            required_fields
        ):

            assert field in recommendation, (
                f"Missing recommendation field: "
                f"{field}"
            )

    print(
        "✓ Required-fields test passed"
    )


# =============================================================
# TEST 8 - RANKING
# =============================================================

def test_ranking():

    analysis = build_test_analysis()

    agent = OptimizationAgent(
        bottleneck_analysis=analysis
    )

    result = agent.optimize()

    recommendations = (
        result[
            "recommendations"
        ]
    )

    assert (
        len(recommendations)
        > 0
    )

    ranks = [
        item["rank"]
        for item
        in recommendations
    ]

    assert ranks == list(
        range(
            1,
            len(
                recommendations
            ) + 1,
        )
    )

    # ---------------------------------------------------------
    # High priority recommendations should come first.
    # ---------------------------------------------------------

    priorities = [
        item["priority"]
        for item
        in recommendations
    ]

    priority_values = {
        "HIGH": 3,
        "MEDIUM": 2,
        "LOW": 1,
    }

    numeric_priorities = [
        priority_values.get(
            value,
            0,
        )
        for value
        in priorities
    ]

    assert numeric_priorities == sorted(
        numeric_priorities,
        reverse=True,
    )

    print(
        "✓ Ranking test passed"
    )


# =============================================================
# TEST 9 - SUMMARY
# =============================================================

def test_summary():

    analysis = build_test_analysis()

    agent = OptimizationAgent(
        bottleneck_analysis=analysis
    )

    result = agent.optimize()

    summary = result[
        "summary"
    ]

    assert isinstance(
        summary,
        dict,
    )

    assert (
        summary[
            "total_recommendations"
        ]
        > 0
    )

    assert (
        summary[
            "total_recommendations"
        ]
        ==
        len(
            result[
                "recommendations"
            ]
        )
    )

    assert (
        summary[
            "high_priority"
        ]
        + summary[
            "medium_priority"
        ]
        + summary[
            "low_priority"
        ]
        ==
        summary[
            "total_recommendations"
        ]
    )

    print(
        "✓ Summary test passed"
    )


# =============================================================
# TEST 10 - EMPTY INPUT
# =============================================================

def test_empty_input():

    agent = OptimizationAgent()

    result = agent.optimize()

    assert isinstance(
        result,
        dict,
    )

    assert (
        result["status"]
        == "SUCCESS"
    )

    assert isinstance(
        result[
            "recommendations"
        ],
        list,
    )

    assert (
        len(
            result[
                "recommendations"
            ]
        )
        == 0
    )

    assert (
        result[
            "summary"
        ][
            "total_recommendations"
        ]
        == 0
    )

    print(
        "✓ Empty-input safety test passed"
    )


# =============================================================
# TEST 11 - NO FABRICATION
# =============================================================

def test_no_fabrication():

    analysis = {
        "bottlenecks": [],
        "transition_analysis": [],
        "rework_analysis": [],
        "deviations": [],
    }

    agent = OptimizationAgent(
        bottleneck_analysis=analysis
    )

    result = agent.optimize()

    assert (
        len(
            result[
                "recommendations"
            ]
        )
        == 0
    )

    print(
        "✓ No-fabrication test passed"
    )


# =============================================================
# MAIN
# =============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "APMOA - OPTIMIZATION AGENT TEST"
    )

    print(
        "=" * 70
    )

    print(
        "\nRunning optimization-agent tests..."
    )

    test_initialization()

    test_basic_output()

    test_bottleneck_recommendation()

    test_transition_recommendation()

    test_rework_recommendation()

    test_deviation_recommendation()

    test_required_fields()

    test_ranking()

    test_summary()

    test_empty_input()

    test_no_fabrication()

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ALL OPTIMIZATION AGENT TESTS PASSED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()