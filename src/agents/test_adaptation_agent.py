"""
APMOA - Adaptation Agent Test

Tests whether the Adaptation Agent correctly interprets
evaluation + reasoning evidence and selects an appropriate
adaptation strategy.

The current discovery algorithms available to APMOA are:

    1. Inductive Miner
    2. Alpha Miner
    3. Alpha+ Miner
    4. Heuristics Miner
    5. Split Miner

ILP Miner is intentionally excluded.

Run:
    python -m src.agents.test_adaptation
"""

from src.agents.adaptation_agent import AdaptationAgent


# ============================================================
# TEST UTILITIES
# ============================================================

def print_result(result):

    """Pretty-print adaptation result."""

    print("\n" + "=" * 60)
    print("ADAPTATION RESULT")
    print("=" * 60)

    print(
        f"\nAction Type : "
        f"{result.get('action_type')}"
    )

    print(
        f"Strategy    : "
        f"{result.get('strategy')}"
    )

    print(
        f"Reason      : "
        f"{result.get('reason')}"
    )

    print(
        f"Parameters  : "
        f"{result.get('parameters')}"
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_result(result):

    assert result is not None

    assert "action_type" in result

    assert "strategy" in result

    assert "reason" in result

    assert "parameters" in result

    assert isinstance(
        result["action_type"],
        str,
    )

    assert isinstance(
        result["strategy"],
        str,
    )

    assert isinstance(
        result["reason"],
        str,
    )

    assert isinstance(
        result["parameters"],
        dict,
    )


# ============================================================
# TEST 1
# PRECISION PROBLEM
# ============================================================

def test_precision_problem():

    print("\n" + "-" * 60)
    print("TEST 1 - PRECISION PROBLEM")
    print("-" * 60)

    evaluation = {

        "fitness":
            1.0000,

        "precision":
            0.0980,

        "generalization":
            0.8617,

        "simplicity":
            0.5947,
    }

    reasoning = {

        "observation": (
            "The selected model achieves perfect fitness "
            "but very low precision."
        ),

        "diagnosis": (
            "The model is likely too permissive."
        ),

        "evidence": [

            "Fitness is 1.0000.",

            "Precision is only 0.0980.",

            "The event log contains 1478 variants.",

            "47.42% of cases contain repeated activities.",
        ],

        "recommended_action": (
            "Investigate a more precision-oriented "
            "discovery strategy."
        ),

        "reasoning": (
            "Improving precision should reduce "
            "unobserved behavior allowed by the model."
        ),

        "confidence":
            0.92,
    }

    agent = AdaptationAgent()

    result = agent.adapt(

        evaluation=evaluation,

        reasoning=reasoning,

        iteration=1,
    )

    validate_result(result)

    print_result(result)

    print(
        "\nTEST 1 PASSED"
    )


# ============================================================
# TEST 2
# FITNESS PROBLEM
# ============================================================

def test_fitness_problem():

    print("\n" + "-" * 60)
    print("TEST 2 - FITNESS PROBLEM")
    print("-" * 60)

    evaluation = {

        "fitness":
            0.62,

        "precision":
            0.85,

        "generalization":
            0.70,

        "simplicity":
            0.60,
    }

    reasoning = {

        "observation": (
            "The current model has insufficient fitness "
            "despite having good precision."
        ),

        "diagnosis": (
            "The model does not adequately reproduce "
            "the observed behavior."
        ),

        "evidence": [

            "Fitness is only 0.62.",

            "Precision is 0.85.",

            "The model fails to represent a significant "
            "portion of observed traces.",
        ],

        "recommended_action": (
            "Use a discovery strategy that improves "
            "fitness and behavioral coverage."
        ),

        "reasoning": (
            "The immediate priority should be improving "
            "the model's ability to reproduce observed traces."
        ),

        "confidence":
            0.88,
    }

    agent = AdaptationAgent()

    result = agent.adapt(

        evaluation=evaluation,

        reasoning=reasoning,

        iteration=1,
    )

    validate_result(result)

    print_result(result)

    print(
        "\nTEST 2 PASSED"
    )


# ============================================================
# TEST 3
# BALANCED MODEL
# ============================================================

def test_balanced_model():

    print("\n" + "-" * 60)
    print("TEST 3 - BALANCED MODEL")
    print("-" * 60)

    evaluation = {

        "fitness":
            0.92,

        "precision":
            0.87,

        "generalization":
            0.85,

        "simplicity":
            0.80,
    }

    reasoning = {

        "observation": (
            "The current model provides a strong balance "
            "across fitness, precision, generalization, "
            "and simplicity."
        ),

        "diagnosis": (
            "No major model-quality weakness is evident."
        ),

        "evidence": [

            "Fitness is 0.92.",

            "Precision is 0.87.",

            "Generalization is 0.85.",

            "Simplicity is 0.80.",
        ],

        "recommended_action": (
            "Retain the current model unless additional "
            "evidence indicates a process problem."
        ),

        "reasoning": (
            "The model is already well balanced, so "
            "unnecessary rediscovery could degrade quality."
        ),

        "confidence":
            0.90,
    }

    agent = AdaptationAgent()

    result = agent.adapt(

        evaluation=evaluation,

        reasoning=reasoning,

        iteration=1,
    )

    validate_result(result)

    print_result(result)

    print(
        "\nTEST 3 PASSED"
    )


# ============================================================
# TEST 4
# HIGH COMPLEXITY
# ============================================================

def test_high_complexity():

    print("\n" + "-" * 60)
    print("TEST 4 - HIGH MODEL COMPLEXITY")
    print("-" * 60)

    evaluation = {

        "fitness":
            0.94,

        "precision":
            0.72,

        "generalization":
            0.82,

        "simplicity":
            0.25,
    }

    reasoning = {

        "observation": (
            "The model has good fitness and reasonable "
            "precision but poor simplicity."
        ),

        "diagnosis": (
            "The discovered model is unnecessarily complex."
        ),

        "evidence": [

            "Fitness is 0.94.",

            "Precision is 0.72.",

            "Generalization is 0.82.",

            "Simplicity is only 0.25.",
        ],

        "recommended_action": (
            "Apply a simplification-oriented discovery "
            "or refinement strategy."
        ),

        "reasoning": (
            "The model should be simplified while "
            "preserving acceptable fitness and precision."
        ),

        "confidence":
            0.86,
    }

    agent = AdaptationAgent()

    result = agent.adapt(

        evaluation=evaluation,

        reasoning=reasoning,

        iteration=1,
    )

    validate_result(result)

    print_result(result)

    print(
        "\nTEST 4 PASSED"
    )


# ============================================================
# TEST 5
# ITERATION HANDLING
# ============================================================

def test_iteration_handling():

    print("\n" + "-" * 60)
    print("TEST 5 - ITERATION HANDLING")
    print("-" * 60)

    evaluation = {

        "fitness":
            1.0,

        "precision":
            0.10,

        "generalization":
            0.86,

        "simplicity":
            0.59,
    }

    reasoning = {

        "observation": (
            "The model has perfect fitness but extremely "
            "low precision."
        ),

        "diagnosis": (
            "The model is overly permissive."
        ),

        "evidence": [

            "Fitness = 1.0.",

            "Precision = 0.10.",
        ],

        "recommended_action": (
            "Use a precision-oriented rediscovery strategy."
        ),

        "reasoning": (
            "Precision must be improved without unnecessarily "
            "sacrificing fitness."
        ),

        "confidence":
            0.95,
    }

    agent = AdaptationAgent()

    result = agent.adapt(

        evaluation=evaluation,

        reasoning=reasoning,

        iteration=3,
    )

    validate_result(result)

    print_result(result)

    print(
        "\nTEST 5 PASSED"
    )


# ============================================================
# TEST 6
# MULTI-ALGORITHM DISCOVERY
# ============================================================

def test_multi_algorithm_strategy():

    print("\n" + "-" * 60)
    print("TEST 6 - MULTI-ALGORITHM STRATEGY")
    print("-" * 60)

    evaluation = {

        "fitness":
            1.0000,

        "precision":
            0.0980,

        "generalization":
            0.8617,

        "simplicity":
            0.5947,

        # ----------------------------------------------------
        # All discovered models
        # ----------------------------------------------------

        "model_count":
            5,

        "available_models": [

            "Inductive Miner",

            "Alpha Miner",

            "Alpha+ Miner",

            "Heuristics Miner",

            "Split Miner",
        ],
    }

    reasoning = {

        "observation": (
            "The current Inductive Miner model has perfect "
            "fitness but very low precision."
        ),

        "diagnosis": (
            "The model is overly permissive and requires "
            "precision-oriented rediscovery."
        ),

        "evidence": [

            "Inductive Miner fitness = 1.0000.",

            "Inductive Miner precision = 0.0980.",

            "Five discovery algorithms are available.",

            "Heuristics Miner is available as a "
            "precision-oriented alternative.",

            "Split Miner is available as an alternative "
            "process discovery strategy.",
        ],

        "recommended_action": (
            "Compare alternative discovery algorithms "
            "with emphasis on precision."
        ),

        "reasoning": (
            "Because multiple discovery algorithms are "
            "available, the autonomous system should "
            "investigate alternative candidates rather "
            "than repeatedly rediscovering the same model."
        ),

        "confidence":
            0.94,
    }

    agent = AdaptationAgent()

    result = agent.adapt(

        evaluation=evaluation,

        reasoning=reasoning,

        iteration=2,
    )

    validate_result(result)

    print_result(result)

    print(
        "\nTEST 6 PASSED"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "APMOA - ADAPTATION AGENT TEST SUITE"
    )

    print("=" * 70)

    print(
        "\nTesting Adaptation Agent with multiple "
        "process-model scenarios..."
    )

    # --------------------------------------------------------
    # Available algorithms
    # --------------------------------------------------------

    print(
        "\nAvailable discovery algorithms:"
    )

    algorithms = [

        "Inductive Miner",

        "Alpha Miner",

        "Alpha+ Miner",

        "Heuristics Miner",

        "Split Miner",
    ]

    for index, algorithm in enumerate(
        algorithms,
        start=1,
    ):

        print(
            f"  {index}. {algorithm}"
        )

    # --------------------------------------------------------
    # Test suite
    # --------------------------------------------------------

    tests = [

        (
            "Precision problem",
            test_precision_problem,
        ),

        (
            "Fitness problem",
            test_fitness_problem,
        ),

        (
            "Balanced model",
            test_balanced_model,
        ),

        (
            "High complexity",
            test_high_complexity,
        ),

        (
            "Iteration handling",
            test_iteration_handling,
        ),

        (
            "Multi-algorithm strategy",
            test_multi_algorithm_strategy,
        ),
    ]

    passed = 0

    failed = 0

    # --------------------------------------------------------
    # Execute
    # --------------------------------------------------------

    for (
        name,
        test_function,
    ) in tests:

        try:

            test_function()

            passed += 1

        except Exception as exc:

            failed += 1

            print(
                "\nTEST FAILED"
            )

            print(
                f"Test    : {name}"
            )

            print(
                f"Error   : "
                f"{type(exc).__name__}: {exc}"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n\n" + "=" * 70)

    print(
        "ADAPTATION AGENT TEST SUMMARY"
    )

    print("=" * 70)

    print(
        f"\nTests passed : {passed}"
    )

    print(
        f"Tests failed : {failed}"
    )

    print(
        f"Total tests  : {len(tests)}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    if failed == 0:

        print(
            "\nALL ADAPTATION AGENT TESTS PASSED"
        )

    else:

        print(
            "\nSOME ADAPTATION AGENT TESTS FAILED"
        )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()