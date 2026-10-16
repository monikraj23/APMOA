"""
APMOA - Reasoning Agent Test

Tests the Reasoning Agent using evaluation results
from all configured discovery algorithms.

Run:

    python -m src.agents.test_reasoning_agent
"""

from src.agents.reasoning_agent import ReasoningAgent


def main():

    print(
        "=" * 70
    )

    print(
        "APMOA - REASONING AGENT TEST"
    )

    print(
        "=" * 70
    )

    # =========================================================
    # PROCESS DIAGNOSTICS
    # =========================================================

    diagnostics = {

        "events": 86581,

        "cases": 7065,

        "activities": 51,

        "average_events_per_case": 12.25,

        "median_events_per_case": 11.0,

        "min_events_per_case": 3,

        "max_events_per_case": 90,

        "number_of_variants": 1478,

        "singleton_variants": 1093,

        "rare_variants_<=5_cases": 1353,

        "dominant_variant_cases": 956,

        "dominant_variant_percentage": 13.53,

        "cases_with_repeated_activities": 3350,

        "percentage_cases_with_repetition": 47.42,

        "maximum_activity_repetition": 17,

        "average_duration_hours": 2097.36,

        "median_duration_hours": 1721.52,
    }

    # =========================================================
    # MODEL EVALUATIONS
    #
    # These are representative results.
    #
    # Replace them with the actual evaluation output once
    # the complete discovery + evaluation pipeline is running.
    # =========================================================

    model_evaluations = [

        # -----------------------------------------------------
        # 1. Inductive Miner
        # -----------------------------------------------------

        {
            "model":
                "Inductive Miner",

            "fitness":
                1.0000,

            "precision":
                0.0980,

            "generalization":
                0.8617,

            "simplicity":
                0.5947,

            "selection_score":
                0.6207,

            "rank":
                1,
        },

        # -----------------------------------------------------
        # 2. Alpha Miner
        # -----------------------------------------------------

        {
            "model":
                "Alpha Miner",

            "fitness":
                0.5709,

            "precision":
                0.0964,

            "generalization":
                0.8595,

            "simplicity":
                0.2612,

            "selection_score":
                0.4243,

            "rank":
                2,
        },

        # -----------------------------------------------------
        # 3. Alpha+ Miner
        #
        # These values are placeholders for testing the
        # reasoning agent structure.
        #
        # Replace with actual evaluation values.
        # -----------------------------------------------------

        {
            "model":
                "Alpha+ Miner",

            "fitness":
                0.6500,

            "precision":
                0.1200,

            "generalization":
                0.8400,

            "simplicity":
                0.3000,

            "selection_score":
                0.4480,

            "rank":
                3,
        },

        # -----------------------------------------------------
        # 4. Heuristics Miner
        #
        # These values are placeholders for testing.
        # -----------------------------------------------------

        {
            "model":
                "Heuristics Miner",

            "fitness":
                0.8200,

            "precision":
                0.4200,

            "generalization":
                0.8000,

            "simplicity":
                0.5000,

            "selection_score":
                0.5980,

            "rank":
                2,
        },

        # -----------------------------------------------------
        # 5. Split Miner
        #
        # These values are placeholders for testing.
        # -----------------------------------------------------

        {
            "model":
                "Split Miner",

            "fitness":
                0.8800,

            "precision":
                0.5200,

            "generalization":
                0.8200,

            "simplicity":
                0.6500,

            "selection_score":
                0.6900,

            "rank":
                1,
        },
    ]

    # =========================================================
    # CREATE AGENT
    # =========================================================

    print(
        "\nInitializing Reasoning Agent..."
    )

    agent = ReasoningAgent()

    print(
        "Reasoning Agent initialized."
    )

    # =========================================================
    # DISPLAY TEST INPUT
    # =========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "MODELS PROVIDED TO REASONING AGENT"
    )

    print(
        "-" * 70
    )

    for evaluation in model_evaluations:

        print(
            f"{evaluation['model']:<25}"
            f"fitness={evaluation['fitness']:.4f}  "
            f"precision={evaluation['precision']:.4f}  "
            f"generalization={evaluation['generalization']:.4f}  "
            f"simplicity={evaluation['simplicity']:.4f}"
        )

    # =========================================================
    # RUN REASONING
    # =========================================================

    print(
        "\nRunning Reasoning Agent..."
    )

    result = agent.reason(

        diagnostics=diagnostics,

        model_evaluations=model_evaluations,

        selected_model="Inductive Miner",
    )

    # =========================================================
    # VALIDATION
    # =========================================================

    print(
        "\n" + "-" * 70
    )

    print(
        "VALIDATING REASONING RESULT"
    )

    print(
        "-" * 70
    )

    assert result is not None

    # ---------------------------------------------------------
    # Required fields
    # ---------------------------------------------------------

    required_fields = [

        "observation",

        "diagnosis",

        "evidence",

        "recommended_action",

        "reasoning",

        "confidence",
    ]

    for field in required_fields:

        assert field in result, (
            f"Missing field: {field}"
        )

        print(
            f"PASS - {field}"
        )

    # ---------------------------------------------------------
    # Evidence
    # ---------------------------------------------------------

    assert isinstance(
        result["evidence"],
        list,
    )

    print(
        "PASS - evidence is a list"
    )

    # ---------------------------------------------------------
    # Confidence
    # ---------------------------------------------------------

    assert isinstance(
        result["confidence"],
        float,
    )

    assert (
        0.0
        <= result["confidence"]
        <= 1.0
    )

    print(
        "PASS - confidence is between 0.0 and 1.0"
    )

    # ---------------------------------------------------------
    # Text fields
    # ---------------------------------------------------------

    text_fields = [

        "observation",

        "diagnosis",

        "recommended_action",

        "reasoning",
    ]

    for field in text_fields:

        assert isinstance(
            result[field],
            str,
        )

        assert len(
            result[field].strip()
        ) > 0

        print(
            f"PASS - {field} contains text"
        )

    # =========================================================
    # FINAL OUTPUT
    # =========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "REASONING AGENT TEST PASSED"
    )

    print(
        "=" * 70
    )


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    main()