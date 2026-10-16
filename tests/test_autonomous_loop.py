"""
APMOA Autonomous Rediscovery Loop Test

Validates a multi-iteration adaptive process discovery loop:

    Inductive Miner
          ↓
    Precision violation
          ↓
    REDISCOVER
          ↓
    Heuristics Miner
          ↓
    Generalization violation
          ↓
    REDISCOVER
          ↓
    Split Miner
          ↓
    Quality accepted
          ↓
    KEEP MODEL
"""

from src.agents.adaptation_agent import AdaptationAgent


def print_metrics(model, metrics):

    print(f"Current Model : {model}")
    print(f"Fitness       : {metrics['fitness']:.4f}")
    print(f"Precision     : {metrics['precision']:.4f}")
    print(f"Generalization: {metrics['generalization']:.4f}")
    print(f"Simplicity    : {metrics['simplicity']:.4f}")


def print_decision(decision):

    print("\nADAPTATION DECISION")
    print("-" * 70)

    print(f"Action      : {decision.get('action_type')}")
    print(f"Strategy    : {decision.get('strategy')}")
    print(f"Next Model  : {decision.get('next_algorithm')}")
    print(f"Policy      : {decision.get('policy_triggered')}")

    print("\nQuality Check:")
    print(decision.get("quality_check"))


def main():

    print("\n" + "=" * 70)
    print("APMOA AUTONOMOUS REDISCOVERY LOOP TEST")
    print("=" * 70)

    agent = AdaptationAgent()

    # ==========================================================
    # ITERATION 1
    # ==========================================================

    model_1 = "Inductive Miner"

    evaluation_1 = {
        "model": model_1,
        "fitness": 1.0000,
        "precision": 0.0980,
        "generalization": 0.8617,
        "simplicity": 0.5947,
    }

    reasoning_1 = {
        "observation": (
            "Inductive Miner has high fitness but very low precision."
        ),
        "diagnosis": (
            "The model may allow excessive behavior compared "
            "with the observed process."
        ),
        "recommended_action": "keep_model",
        "confidence": 0.85,
    }

    print("\n" + "=" * 70)
    print("ITERATION 1")
    print("=" * 70)

    print()
    print_metrics(model_1, evaluation_1)

    print("\nLLM Recommendation:")
    print(reasoning_1["recommended_action"])

    decision_1 = agent.adapt(
        evaluation=evaluation_1,
        reasoning=reasoning_1,
        iteration=1,
    )

    print_decision(decision_1)

    assert decision_1["action_type"] == "rediscover"

    assert decision_1["next_algorithm"] == "Heuristics Miner"

    print("\n✓ ITERATION 1: REDISCOVERY TRIGGERED")
    print("✓ Precision violation detected")
    print("✓ Heuristics Miner selected")

    # ==========================================================
    # ITERATION 2
    # ==========================================================

    model_2 = "Heuristics Miner"

    evaluation_2 = {
        "model": model_2,
        "fitness": 0.8864,
        "precision": 0.8137,
        "generalization": 0.6005,
        "simplicity": 0.4759,
    }

    reasoning_2 = {
        "observation": (
            "Heuristics Miner provides high precision and acceptable "
            "fitness but generalization is below the configured threshold."
        ),
        "diagnosis": (
            "The model may not generalize sufficiently to unseen "
            "process behavior."
        ),
        "recommended_action": "keep_model",
        "confidence": 0.82,
    }

    print("\n" + "=" * 70)
    print("ITERATION 2")
    print("=" * 70)

    print()
    print_metrics(model_2, evaluation_2)

    print("\nLLM Recommendation:")
    print(reasoning_2["recommended_action"])

    decision_2 = agent.adapt(
        evaluation=evaluation_2,
        reasoning=reasoning_2,
        iteration=2,
    )

    print_decision(decision_2)

    assert decision_2["action_type"] == "rediscover"

    assert decision_2["next_algorithm"] == "Split Miner"

    print("\n✓ ITERATION 2: REDISCOVERY TRIGGERED")
    print("✓ Generalization violation detected")
    print("✓ Split Miner selected")

    # ==========================================================
    # ITERATION 3
    # ==========================================================

    model_3 = "Split Miner"

    evaluation_3 = {
        "model": model_3,
        "fitness": 0.9742,
        "precision": 0.6367,
        "generalization": 0.8564,
        "simplicity": 0.5642,
    }

    reasoning_3 = {
        "observation": (
            "Split Miner provides strong fitness, acceptable precision, "
            "good generalization, and acceptable simplicity."
        ),
        "diagnosis": (
            "The current model satisfies the configured quality policy."
        ),
        "recommended_action": "keep_model",
        "confidence": 0.90,
    }

    print("\n" + "=" * 70)
    print("ITERATION 3")
    print("=" * 70)

    print()
    print_metrics(model_3, evaluation_3)

    print("\nLLM Recommendation:")
    print(reasoning_3["recommended_action"])

    decision_3 = agent.adapt(
        evaluation=evaluation_3,
        reasoning=reasoning_3,
        iteration=3,
    )

    print_decision(decision_3)

    assert decision_3["action_type"] == "keep_model"

    print("\n✓ ITERATION 3: MODEL QUALITY ACCEPTED")
    print("✓ KEEP MODEL")
    print("✓ AUTONOMOUS LOOP TERMINATED")

    # ==========================================================
    # FINAL SUMMARY
    # ==========================================================

    print("\n" + "=" * 70)
    print("AUTONOMOUS REDISCOVERY TEST PASSED")
    print("=" * 70)

    print("\nAutonomous Execution Path:")
    print()
    print("Iteration 1")
    print("Inductive Miner")
    print("      ↓")
    print("Precision violation")
    print("      ↓")
    print("REDISCOVER")
    print("      ↓")
    print("Heuristics Miner")
    print()
    print("Iteration 2")
    print("Heuristics Miner")
    print("      ↓")
    print("Generalization violation")
    print("      ↓")
    print("REDISCOVER")
    print("      ↓")
    print("Split Miner")
    print()
    print("Iteration 3")
    print("Split Miner")
    print("      ↓")
    print("All quality thresholds satisfied")
    print("      ↓")
    print("KEEP MODEL")
    print("      ↓")
    print("STOP")

    print("\nFinal Model:", model_3)

    print("\nAPMOA demonstrated:")
    print("1. Multi-iteration process discovery")
    print("2. Quality-based adaptation")
    print("3. Precision-oriented rediscovery")
    print("4. Generalization-oriented adaptation")
    print("5. Adaptive algorithm selection")
    print("6. Model reevaluation")
    print("7. Autonomous termination")


if __name__ == "__main__":
    main()