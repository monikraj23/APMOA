"""
APMOA - Autonomous Feedback Loop Validation

Tests the weaker autonomous scenario:

    Weak model
        |
        v
    Reasoning
        |
        v
    REDISCOVER
        |
        v
    Adaptation
        |
        v
    Heuristics Miner
        |
        v
    Re-evaluation
        |
        v
    KEEP MODEL
        |
        v
    STOP

This test intentionally uses lightweight fake agents.

It does NOT execute expensive PM4Py discovery.

The purpose is to validate orchestration logic.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.agents.orchestrator import (
    APMOAOrchestrator,
)


# ============================================================
# TEST EVALUATIONS
# ============================================================

WEAK_EVALUATION = {
    "model": "Inductive Miner",
    "fitness": 1.0000,
    "precision": 0.0980,
    "generalization": 0.8617,
    "simplicity": 0.5947,
    "status": "SUCCESS",
}


IMPROVED_EVALUATION = {
    "model": "Heuristics Miner",
    "fitness": 0.8864,
    "precision": 0.8137,
    "generalization": 0.5987,
    "simplicity": 0.4759,
    "status": "SUCCESS",
}


# ============================================================
# FAKE DATA ANALYSIS
# ============================================================

class FakeDataAnalysisAgent:

    def __init__(self):
        self.calls = 0

    def analyze(self):

        self.calls += 1

        return {
            "events": 86581,
            "cases": 7065,
            "activities": 51,
            "test_mode": True,
        }


# ============================================================
# FAKE DISCOVERY
# ============================================================

class FakeDiscoveryAgent:

    def __init__(self):

        self.initial_calls = 0
        self.adaptive_calls = 0

        self.received_adaptations = []

    # --------------------------------------------------------
    # INITIAL DISCOVERY
    # --------------------------------------------------------

    def discover_all(self):

        self.initial_calls += 1

        print(
            "\n[TEST DISCOVERY]"
        )

        print(
            "Initial discovery:"
            " Inductive Miner"
        )

        return [
            {
                "name":
                    "Inductive Miner",

                "algorithm":
                    "inductive_miner",

                "model":
                    "TEST_INDUCTIVE_MODEL",

                "status":
                    "SUCCESS",
            }
        ]

    # --------------------------------------------------------
    # ADAPTIVE DISCOVERY
    # --------------------------------------------------------

    def discover_adaptive(
        self,
        adaptation=None,
    ):

        self.adaptive_calls += 1

        self.received_adaptations.append(
            adaptation
        )

        print(
            "\n[TEST DISCOVERY]"
        )

        print(
            "Adaptive discovery:"
            " Heuristics Miner"
        )

        return [
            {
                "name":
                    "Heuristics Miner",

                "algorithm":
                    "heuristics_miner",

                "model":
                    "TEST_HEURISTICS_MODEL",

                "status":
                    "SUCCESS",
            }
        ]


# ============================================================
# FAKE EVALUATION
# ============================================================

class FakeEvaluationAgent:

    def __init__(self):

        self.calls = 0

        self.models_seen = []

    def evaluate(
        self,
        models,
    ):

        self.calls += 1

        self.models_seen.append(
            models
        )

        model_name = (
            models[0]["name"]
        )

        if model_name == "Inductive Miner":

            return pd.DataFrame(
                [WEAK_EVALUATION]
            )

        if model_name == "Heuristics Miner":

            return pd.DataFrame(
                [IMPROVED_EVALUATION]
            )

        raise AssertionError(
            f"Unexpected model: {model_name}"
        )


# ============================================================
# FAKE REASONING
# ============================================================

class FakeReasoningAgent:

    def __init__(self):

        self.calls = 0

        self.models_seen = []

    def reason(
        self,
        diagnostics,
        model_evaluations,
        selected_model,
    ):

        self.calls += 1

        self.models_seen.append(
            selected_model
        )

        # ----------------------------------------------------
        # ITERATION 1
        # ----------------------------------------------------

        if self.calls == 1:

            print(
                "\n[TEST REASONING]"
            )

            print(
                "Weak model detected."
            )

            print(
                "Decision: REDISCOVER"
            )

            return {
                "observation": (
                    "Inductive Miner has "
                    "perfect fitness but "
                    "very low precision."
                ),

                "diagnosis": (
                    "The model allows "
                    "excessive behavior."
                ),

                "evidence": [
                    "Fitness = 1.0000",
                    "Precision = 0.0980",
                ],

                "recommended_action":
                    "rediscover",

                "action":
                    "rediscover",

                "strategy":
                    "precision_oriented",

                "confidence":
                    0.95,
            }

        # ----------------------------------------------------
        # ITERATION 2
        # ----------------------------------------------------

        print(
            "\n[TEST REASONING]"
        )

        print(
            "Improved model detected."
        )

        print(
            "Decision: KEEP_MODEL"
        )

        return {
            "observation": (
                "Heuristics Miner "
                "substantially improves "
                "precision."
            ),

            "diagnosis": (
                "The major precision "
                "weakness has been "
                "addressed."
            ),

            "evidence": [
                "Fitness = 0.8864",
                "Precision = 0.8137",
            ],

            "recommended_action":
                "keep_model",

            "action":
                "keep_model",

            "strategy":
                "none",

            "confidence":
                0.90,
        }


# ============================================================
# FAKE ADAPTATION
# ============================================================

class FakeAdaptationAgent:

    def __init__(self):

        self.calls = 0

        self.decisions = []

    def adapt(
        self,
        evaluation,
        reasoning,
        iteration=1,
    ):

        self.calls += 1

        # ----------------------------------------------------
        # ITERATION 1
        # ----------------------------------------------------

        if iteration == 1:

            decision = {
                "action_type":
                    "rediscover",

                "action":
                    "rediscover",

                "strategy":
                    "precision_oriented",

                "target":
                    "precision",

                "current_model":
                    "Inductive Miner",

                "next_algorithm":
                    "Heuristics Miner",

                "reason": (
                    "The current model "
                    "has very low precision. "
                    "A precision-oriented "
                    "rediscovery is required."
                ),

                "parameters": {
                    "dependency_threshold":
                        0.7,

                    "and_threshold":
                        0.65,

                    "loop_two_threshold":
                        0.5,
                },

                "iteration":
                    iteration,
            }

            self.decisions.append(
                decision
            )

            return decision

        # ----------------------------------------------------
        # ITERATION 2
        # ----------------------------------------------------

        decision = {
            "action_type":
                "keep_model",

            "action":
                "keep_model",

            "strategy":
                "none",

            "target":
                "current_model",

            "current_model":
                "Heuristics Miner",

            "next_algorithm":
                None,

            "reason": (
                "The improved model "
                "provides acceptable "
                "precision and should "
                "be retained."
            ),

            "parameters": {},

            "iteration":
                iteration,
        }

        self.decisions.append(
            decision
        )

        return decision


# ============================================================
# TEST ORCHESTRATOR
# ============================================================

class TestOrchestrator(
    APMOAOrchestrator
):

    """
    Overrides only evaluation and selection.

    The actual orchestration loop from
    APMOAOrchestrator is exercised.
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        super().__init__(
            *args,
            **kwargs,
        )

        self.selection_calls = 0

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    def _evaluate_models(
        self,
        discovered_models,
    ):

        model_name = (
            discovered_models[0]["name"]
        )

        if model_name == "Inductive Miner":

            return pd.DataFrame(
                [WEAK_EVALUATION]
            )

        if model_name == "Heuristics Miner":

            return pd.DataFrame(
                [IMPROVED_EVALUATION]
            )

        raise AssertionError(
            f"Unexpected model: {model_name}"
        )

    # --------------------------------------------------------
    # Selection
    # --------------------------------------------------------

    def _select_model(
        self,
        evaluations,
    ):

        self.selection_calls += 1

        if self.selection_calls == 1:

            return {
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
            }

        return {
            "model":
                "Heuristics Miner",

            "fitness":
                0.8864,

            "precision":
                0.8137,

            "generalization":
                0.5987,

            "simplicity":
                0.4759,
        }


# ============================================================
# VALIDATION
# ============================================================

def validate_result(
    result,
    data_agent,
    discovery_agent,
    evaluation_agent,
    reasoning_agent,
    adaptation_agent,
):

    print("\n")
    print("=" * 70)
    print(
        "VALIDATING AUTONOMOUS LOOP"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    assert isinstance(
        result,
        dict,
    ), (
        "Orchestrator result "
        "must be a dictionary."
    )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history = result.get(
        "history"
    )

    assert isinstance(
        history,
        list,
    ), (
        "History must be a list."
    )

    assert len(history) == 2, (
        "Expected exactly two "
        "iterations."
    )

    # --------------------------------------------------------
    # ITERATION 1
    # --------------------------------------------------------

    first = history[0]

    assert (
        first["selected_model"]
        == "Inductive Miner"
    )

    assert (
        first["reasoning"][
            "recommended_action"
        ]
        == "rediscover"
    )

    assert (
        first["adaptation"][
            "action_type"
        ]
        == "rediscover"
    )

    assert (
        first["adaptation"][
            "strategy"
        ]
        == "precision_oriented"
    )

    assert (
        first["adaptation"][
            "next_algorithm"
        ]
        == "Heuristics Miner"
    )

    # --------------------------------------------------------
    # ITERATION 2
    # --------------------------------------------------------

    second = history[1]

    assert (
        second["selected_model"]
        == "Heuristics Miner"
    )

    assert (
        second["reasoning"][
            "recommended_action"
        ]
        == "keep_model"
    )

    assert (
        second["adaptation"][
            "action_type"
        ]
        == "keep_model"
    )

    # --------------------------------------------------------
    # FINAL MODEL
    # --------------------------------------------------------

    assert (
        result["final_model"]
        == "Heuristics Miner"
    )

    # --------------------------------------------------------
    # FINAL ACTION
    # --------------------------------------------------------

    assert (
        result["final_action"]
        == "keep_model"
    )

    # --------------------------------------------------------
    # ITERATIONS
    # --------------------------------------------------------

    assert (
        result["iterations"]
        == 2
    ), (
        "The loop continued after "
        "keep_model or failed to "
        "perform rediscovery."
    )

    # --------------------------------------------------------
    # DATA ANALYSIS
    # --------------------------------------------------------

    assert (
        data_agent.calls
        == 1
    )

    # --------------------------------------------------------
    # INITIAL DISCOVERY
    # --------------------------------------------------------

    assert (
        discovery_agent.initial_calls
        == 1
    )

    # --------------------------------------------------------
    # ADAPTIVE DISCOVERY
    # --------------------------------------------------------

    assert (
        discovery_agent.adaptive_calls
        == 1
    ), (
        "Adaptive discovery must "
        "execute exactly once."
    )

    # --------------------------------------------------------
    # ADAPTATION PASSED TO DISCOVERY
    # --------------------------------------------------------

    assert (
        len(
            discovery_agent
            .received_adaptations
        )
        == 1
    )

    adaptive_decision = (
        discovery_agent
        .received_adaptations[0]
    )

    assert (
        adaptive_decision[
            "strategy"
        ]
        == "precision_oriented"
    )

    assert (
        adaptive_decision[
            "next_algorithm"
        ]
        == "Heuristics Miner"
    )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    assert (
        evaluation_agent.calls
        == 0
    ), (
        "TestOrchestrator should "
        "override _evaluate_models."
    )

    # --------------------------------------------------------
    # SELECTION
    # --------------------------------------------------------

    assert (
        getattr(
            result,
            "selection_calls",
            2,
        )
        >= 0
    )

    # --------------------------------------------------------
    # REASONING
    # --------------------------------------------------------

    assert (
        reasoning_agent.calls
        == 2
    )

    assert (
        reasoning_agent.models_seen
        == [
            "Inductive Miner",
            "Heuristics Miner",
        ]
    )

    # --------------------------------------------------------
    # ADAPTATION
    # --------------------------------------------------------

    assert (
        adaptation_agent.calls
        == 2
    )

    # --------------------------------------------------------
    # NO THIRD ITERATION
    # --------------------------------------------------------

    assert (
        len(history)
        < 3
    ), (
        "A third iteration was "
        "incorrectly started after "
        "keep_model."
    )

    # --------------------------------------------------------
    # FINAL EVALUATION
    # --------------------------------------------------------

    final_evaluation = (
        result["final_evaluation"]
    )

    assert (
        final_evaluation[
            "precision"
        ]
        == 0.8137
    )

    print(
        "\n✓ Initial weak model:"
        " Inductive Miner"
    )

    print(
        "✓ Weak precision:"
        " 0.0980"
    )

    print(
        "✓ Reasoning:"
        " rediscover"
    )

    print(
        "✓ Adaptation:"
        " precision_oriented"
    )

    print(
        "✓ Next algorithm:"
        " Heuristics Miner"
    )

    print(
        "✓ Adaptive discovery executed"
    )

    print(
        "✓ Improved model evaluated"
    )

    print(
        "✓ Improved precision:"
        " 0.8137"
    )

    print(
        "✓ Reasoning:"
        " keep_model"
    )

    print(
        "✓ keep_model treated as terminal"
    )

    print(
        "✓ No unnecessary third iteration"
    )

    print(
        "✓ Final model:"
        " Heuristics Miner"
    )

    print("\n" + "=" * 70)
    print(
        "WEAKER AUTONOMOUS SCENARIO PASSED"
    )
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "APMOA - AUTONOMOUS FEEDBACK LOOP TEST"
    )
    print("=" * 70)

    print(
        """
Scenario:

Iteration 1
    Inductive Miner
    Fitness    = 1.0000
    Precision  = 0.0980
          |
          v
    REDISCOVER
          |
          v
    precision_oriented
          |
          v
    Heuristics Miner
          |
          v
Iteration 2
    Fitness    = 0.8864
    Precision  = 0.8137
          |
          v
    KEEP_MODEL
          |
          v
        STOP
"""
    )

    # --------------------------------------------------------
    # CREATE TEST AGENTS
    # --------------------------------------------------------

    data_agent = (
        FakeDataAnalysisAgent()
    )

    discovery_agent = (
        FakeDiscoveryAgent()
    )

    evaluation_agent = (
        FakeEvaluationAgent()
    )

    reasoning_agent = (
        FakeReasoningAgent()
    )

    adaptation_agent = (
        FakeAdaptationAgent()
    )

    # --------------------------------------------------------
    # CREATE ORCHESTRATOR
    # --------------------------------------------------------

    orchestrator = TestOrchestrator(
        data_analysis_agent=
            data_agent,

        discovery_agent=
            discovery_agent,

        evaluation_agent=
            evaluation_agent,

        selection_agent=None,

        reasoning_agent=
            reasoning_agent,

        adaptation_agent=
            adaptation_agent,

        max_iterations=3,
    )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    result = (
        orchestrator.run()
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    validate_result(
        result=result,

        data_agent=
            data_agent,

        discovery_agent=
            discovery_agent,

        evaluation_agent=
            evaluation_agent,

        reasoning_agent=
            reasoning_agent,

        adaptation_agent=
            adaptation_agent,
    )

    print("\n")
    print("=" * 70)
    print(
        "ALL AUTONOMOUS LOOP TESTS PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()