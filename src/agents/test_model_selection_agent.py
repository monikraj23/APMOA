"""
APMOA - Test Model Selection Agent

Tests the ModelSelectionAgent using evaluation results
from all configured process discovery algorithms.

Run:
    python -m src.agents.test_model_selection_agent
"""

import pandas as pd

from src.agents.model_selection_agent import (
    ModelSelectionAgent
)


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("APMOA - MODEL SELECTION AGENT TEST")
    print("=" * 70)

    # ========================================================
    # EVALUATION RESULTS
    # ========================================================
    #
    # These represent the output expected from the
    # ModelEvaluationAgent after evaluating all discovered
    # process models.
    #
    # Five discovery algorithms are currently supported:
    #
    #   1. Inductive Miner
    #   2. Alpha Miner
    #   3. Alpha+ Miner
    #   4. Heuristics Miner
    #   5. Split Miner
    #
    # ========================================================

    evaluation_results = pd.DataFrame(
        [
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
            },

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
            },

            {
                "model":
                    "Alpha+ Miner",

                # ------------------------------------------------
                # Placeholder values.
                #
                # Replace these with the actual evaluation
                # results obtained from ModelEvaluationAgent.
                # ------------------------------------------------

                "fitness":
                    0.8500,

                "precision":
                    0.1200,

                "generalization":
                    0.8000,

                "simplicity":
                    0.4500,
            },

            {
                "model":
                    "Heuristics Miner",

                # ------------------------------------------------
                # Placeholder values.
                #
                # Replace these with actual evaluation results.
                # ------------------------------------------------

                "fitness":
                    0.9000,

                "precision":
                    0.3000,

                "generalization":
                    0.8200,

                "simplicity":
                    0.5000,
            },

            {
                "model":
                    "Split Miner",

                # ------------------------------------------------
                # Placeholder values.
                #
                # Replace these with actual evaluation results.
                # ------------------------------------------------

                "fitness":
                    0.9200,

                "precision":
                    0.3500,

                "generalization":
                    0.8300,

                "simplicity":
                    0.6500,
            },
        ]
    )

    # ========================================================
    # DISPLAY INPUT
    # ========================================================

    print("\n" + "-" * 70)
    print("INPUT EVALUATION RESULTS")
    print("-" * 70)

    print(
        evaluation_results.to_string(
            index=False
        )
    )

    # ========================================================
    # VALIDATE INPUT
    # ========================================================

    expected_models = {
        "Inductive Miner",
        "Alpha Miner",
        "Alpha+ Miner",
        "Heuristics Miner",
        "Split Miner",
    }

    actual_models = set(
        evaluation_results["model"]
    )

    assert actual_models == expected_models, (
        "Evaluation results do not contain "
        "the expected five discovery algorithms."
    )

    assert len(
        evaluation_results
    ) == 5, (
        "Expected evaluation results for "
        "exactly five models."
    )

    # ========================================================
    # CREATE AGENT
    # ========================================================

    print("\n" + "-" * 70)
    print("INITIALIZING MODEL SELECTION AGENT")
    print("-" * 70)

    agent = ModelSelectionAgent(
        evaluation_results
    )

    print(
        "Model Selection Agent initialized."
    )

    # ========================================================
    # SELECT MODEL
    # ========================================================

    print("\n" + "-" * 70)
    print("RUNNING MODEL SELECTION")
    print("-" * 70)

    result = agent.select()

    # ========================================================
    # VALIDATE RESULT
    # ========================================================

    assert result is not None

    assert isinstance(
        result,
        dict
    )

    assert (
        "selected_model"
        in result
    )

    assert (
        "selection_score"
        in result
    )

    assert (
        "ranking"
        in result
    )

    assert (
        "explanation"
        in result
    )

    # ========================================================
    # VALIDATE SELECTED MODEL
    # ========================================================

    selected_model = result[
        "selected_model"
    ]

    assert selected_model in expected_models, (
        f"Invalid selected model: "
        f"{selected_model}"
    )

    # ========================================================
    # VALIDATE SCORE
    # ========================================================

    selection_score = float(
        result[
            "selection_score"
        ]
    )

    assert 0.0 <= selection_score <= 1.0, (
        "Selection score must be between "
        "0.0 and 1.0."
    )

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print("\n" + "=" * 70)
    print("MODEL SELECTION RESULT")
    print("=" * 70)

    print(
        f"\nSelected model : "
        f"{selected_model}"
    )

    print(
        f"Selection score: "
        f"{selection_score:.4f}"
    )

    # ========================================================
    # DISPLAY RANKING
    # ========================================================

    print("\n" + "-" * 70)
    print("MODEL RANKING")
    print("-" * 70)

    ranking = result[
        "ranking"
    ]

    print(
        ranking
    )

    # ========================================================
    # DISPLAY EXPLANATION
    # ========================================================

    print("\n" + "-" * 70)
    print("SELECTION EXPLANATION")
    print("-" * 70)

    explanation = result[
        "explanation"
    ]

    if isinstance(
        explanation,
        list
    ):

        for item in explanation:

            print(
                f"  • {item}"
            )

    else:

        print(
            explanation
        )

    # ========================================================
    # TEST PASSED
    # ========================================================

    print("\n" + "=" * 70)
    print("MODEL SELECTION AGENT TEST PASSED")
    print("=" * 70)

    print(
        "\nSuccessfully evaluated "
        "five candidate discovery algorithms."
    )

    print(
        f"Final selected model: "
        f"{selected_model}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()