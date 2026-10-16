"""
APMOA RAG Ablation Experiment

Purpose
-------
Compare the Reasoning Agent with and without RAG while keeping
the process-mining evidence identical.

Experiment
----------

Configuration A:
    Process Evidence
          ↓
       RAG
          ↓
       Groq LLM
          ↓
      Reasoning

Configuration B:
    Process Evidence
          ↓
       Groq LLM
          ↓
      Reasoning

The experiment measures:

    - recommended action
    - confidence
    - reasoning
    - diagnosis
    - RAG usage
    - retrieved sources
    - execution time

IMPORTANT
---------
This experiment does NOT rerun process discovery or model
evaluation. It uses the same validated metrics from the
successful APMOA Travel Permit execution.

This isolates the contribution of RAG to the reasoning stage.
"""

import json
import os
import time
from datetime import datetime

from src.agents.reasoning_agent import ReasoningAgent
from src.rag.knowledge_base import APMOAKnowledgeBase


# ==============================================================
# CONFIGURATION
# ==============================================================

KNOWLEDGE_DIR = "data/knowledge"

OUTPUT_DIR = "results/experiments"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "rag_ablation_results.json",
)


# ==============================================================
# FIXED PROCESS-MINING EVIDENCE
# ==============================================================

# These values come from the successful APMOA Travel Permit
# execution.

MODEL_EVALUATIONS = [
    {
        "model": "Inductive Miner",
        "fitness": 1.0000,
        "precision": 0.0980,
        "generalization": 0.8617,
        "simplicity": 0.5947,
        "selection_score": 0.620678,
        "rank": 3,
        "status": "SUCCESS",
    },
    {
        "model": "Alpha Miner",
        "fitness": 0.4199,
        "precision": 0.0000,
        "generalization": 0.8553,
        "simplicity": 0.4190,
        "selection_score": 0.380832,
        "rank": 5,
        "status": "SUCCESS",
    },
    {
        "model": "Alpha+ Miner",
        "fitness": 0.5862,
        "precision": 0.0964,
        "generalization": 0.8595,
        "simplicity": 0.2646,
        "selection_score": 0.429574,
        "rank": 4,
        "status": "SUCCESS",
    },
    {
        "model": "Heuristics Miner",
        "fitness": 0.8864,
        "precision": 0.8137,
        "generalization": 0.5986,
        "simplicity": 0.4759,
        "selection_score": 0.724941,
        "rank": 2,
        "status": "SUCCESS",
    },
    {
        "model": "Split Miner",
        "fitness": 0.9742,
        "precision": 0.6387,
        "generalization": 0.8564,
        "simplicity": 0.5642,
        "selection_score": 0.767992,
        "rank": 1,
        "status": "SUCCESS",
    },
]


# ==============================================================
# PROCESS DIAGNOSTICS
# ==============================================================

PROCESS_DIAGNOSTICS = {
    "dataset": "BPI Challenge 2020 - Travel Permit",
    "events": 86581,
    "cases": 7065,
    "activities": 51,
    "resources": 2,

    "process_intelligence": {
        "potential_bottlenecks": 23,
        "potential_deviations": 50,
        "activities_with_rework": 41,
        "transitions_analyzed": 553,
    },

    "selection_weights": {
        "fitness": 0.30,
        "precision": 0.30,
        "generalization": 0.20,
        "simplicity": 0.20,
    },

    "quality_thresholds": {
        "fitness_min": 0.80,
        "precision_min": 0.50,
        "generalization_min": 0.70,
        "simplicity_min": 0.40,
    },
}


# ==============================================================
# SELECTED MODEL
# ==============================================================

SELECTED_MODEL = "Split Miner"


# ==============================================================
# UTILITY
# ==============================================================

def ensure_output_directory():
    """Create experiment output directory."""

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )


# ==============================================================
# PRINT RESULT
# ==============================================================

def print_reasoning_result(
    configuration,
    result,
    elapsed,
):
    """Print a readable reasoning result."""

    print("\n" + "=" * 70)
    print(configuration)
    print("=" * 70)

    print(
        f"\nExecution time : "
        f"{elapsed:.2f} seconds"
    )

    print(
        f"RAG used       : "
        f"{result.get('rag_used')}"
    )

    print(
        f"Recommended action : "
        f"{result.get('recommended_action')}"
    )

    print(
        f"Confidence     : "
        f"{result.get('confidence')}"
    )

    print("\nObservation:")
    print(
        result.get(
            "observation",
            "",
        )
    )

    print("\nDiagnosis:")
    print(
        result.get(
            "diagnosis",
            "",
        )
    )

    print("\nEvidence:")

    for item in result.get(
        "evidence",
        [],
    ):
        print(
            f"  • {item}"
        )

    print("\nReasoning:")
    print(
        result.get(
            "reasoning",
            "",
        )
    )

    sources = result.get(
        "rag_sources",
        [],
    )

    if sources:

        print("\nRetrieved RAG sources:")

        for source in sources:

            print(
                f"  • {source}"
            )


# ==============================================================
# RUN RAG ENABLED
# ==============================================================

def run_with_rag(
    knowledge_base,
):
    """
    Run the Reasoning Agent with RAG enabled.
    """

    print("\n")
    print("=" * 70)
    print("EXPERIMENT A - RAG ENABLED")
    print("=" * 70)

    reasoning_agent = ReasoningAgent(
        knowledge_base=knowledge_base,
        rag_top_k=3,
    )

    start_time = time.perf_counter()

    result = reasoning_agent.reason(
        diagnostics=PROCESS_DIAGNOSTICS,
        model_evaluations=MODEL_EVALUATIONS,
        selected_model=SELECTED_MODEL,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print_reasoning_result(
        "RAG ENABLED",
        result,
        elapsed,
    )

    return {
        "configuration": "RAG_ENABLED",
        "rag_enabled": True,
        "elapsed_seconds": round(
            elapsed,
            4,
        ),
        "result": result,
    }


# ==============================================================
# RUN WITHOUT RAG
# ==============================================================

def run_without_rag():
    """
    Run the Reasoning Agent without a knowledge base.

    The process-mining evidence is identical to Experiment A.
    """

    print("\n")
    print("=" * 70)
    print("EXPERIMENT B - RAG DISABLED")
    print("=" * 70)

    reasoning_agent = ReasoningAgent(
        knowledge_base=None,
        rag_top_k=0,
    )

    start_time = time.perf_counter()

    result = reasoning_agent.reason(
        diagnostics=PROCESS_DIAGNOSTICS,
        model_evaluations=MODEL_EVALUATIONS,
        selected_model=SELECTED_MODEL,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print_reasoning_result(
        "RAG DISABLED",
        result,
        elapsed,
    )

    return {
        "configuration": "RAG_DISABLED",
        "rag_enabled": False,
        "elapsed_seconds": round(
            elapsed,
            4,
        ),
        "result": result,
    }


# ==============================================================
# COMPARE RESULTS
# ==============================================================

def compare_results(
    rag_result,
    no_rag_result,
):
    """
    Compare the two reasoning configurations.

    No configuration is declared superior automatically.
    The purpose is to expose measurable differences.
    """

    rag = rag_result["result"]

    no_rag = no_rag_result["result"]

    print("\n")
    print("=" * 70)
    print("RAG ABLATION COMPARISON")
    print("=" * 70)

    print(
        f"\n{'Metric':<30}"
        f"{'RAG':<25}"
        f"{'No RAG':<25}"
    )

    print("-" * 80)

    print(
        f"{'Recommended Action':<30}"
        f"{str(rag.get('recommended_action')):<25}"
        f"{str(no_rag.get('recommended_action')):<25}"
    )

    print(
        f"{'Confidence':<30}"
        f"{str(rag.get('confidence')):<25}"
        f"{str(no_rag.get('confidence')):<25}"
    )

    print(
        f"{'RAG Used':<30}"
        f"{str(rag.get('rag_used')):<25}"
        f"{str(no_rag.get('rag_used')):<25}"
    )

    print(
        f"{'Execution Time (seconds)':<30}"
        f"{str(rag_result['elapsed_seconds']):<25}"
        f"{str(no_rag_result['elapsed_seconds']):<25}"
    )

    rag_sources = rag.get(
        "rag_sources",
        [],
    )

    no_rag_sources = no_rag.get(
        "rag_sources",
        [],
    )

    print(
        f"{'Retrieved Sources':<30}"
        f"{len(rag_sources):<25}"
        f"{len(no_rag_sources):<25}"
    )

    # ----------------------------------------------------------
    # Action consistency
    # ----------------------------------------------------------

    action_same = (
        rag.get("recommended_action")
        ==
        no_rag.get("recommended_action")
    )

    print("\nAction consistency:")

    if action_same:
        print(
            "✓ Both configurations produced "
            "the same recommended action."
        )
    else:
        print(
            "• The configurations produced "
            "different recommended actions."
        )

    # ----------------------------------------------------------
    # Confidence difference
    # ----------------------------------------------------------

    rag_confidence = rag.get(
        "confidence"
    )

    no_rag_confidence = no_rag.get(
        "confidence"
    )

    confidence_difference = None

    if (
        isinstance(
            rag_confidence,
            (int, float),
        )
        and
        isinstance(
            no_rag_confidence,
            (int, float),
        )
    ):

        confidence_difference = round(
            rag_confidence
            -
            no_rag_confidence,
            4,
        )

    print(
        "\nConfidence difference "
        "(RAG - No RAG):"
    )

    print(
        confidence_difference
    )

    return {
        "action_same": action_same,
        "confidence_difference": (
            confidence_difference
        ),
        "rag_sources": len(
            rag_sources
        ),
        "no_rag_sources": len(
            no_rag_sources
        ),
    }


# ==============================================================
# SAVE RESULTS
# ==============================================================

def save_results(
    rag_result,
    no_rag_result,
    comparison,
):
    """Save the complete ablation experiment."""

    ensure_output_directory()

    results = {
        "experiment": "APMOA RAG Ablation",
        "timestamp": datetime.now().isoformat(),

        "description": (
            "Comparison of Reasoning Agent behavior "
            "with and without RAG using identical "
            "process-mining evidence."
        ),

        "selected_model": SELECTED_MODEL,

        "process_diagnostics": PROCESS_DIAGNOSTICS,

        "model_evaluations": MODEL_EVALUATIONS,

        "rag_enabled": rag_result,

        "rag_disabled": no_rag_result,

        "comparison": comparison,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    print("\n")
    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(
        f"\n{OUTPUT_FILE}"
    )

    return results


# ==============================================================
# MAIN
# ==============================================================

def main():

    print("\n" + "=" * 70)
    print("APMOA - RAG ABLATION EXPERIMENT")
    print("=" * 70)

    print(
        "\nPurpose:"
    )

    print(
        "Compare Reasoning Agent behavior "
        "with and without RAG."
    )

    print(
        "\nThe process-mining evidence is identical "
        "for both configurations."
    )

    # ----------------------------------------------------------
    # Check Groq API key
    # ----------------------------------------------------------

    if not os.getenv(
        "GROQ_API_KEY"
    ):

        print("\nERROR:")
        print(
            "GROQ_API_KEY is not set."
        )

        print(
            "\nMake sure your .env configuration "
            "is loaded before running this experiment."
        )

        return

    # ----------------------------------------------------------
    # Initialize RAG knowledge base
    # ----------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("BUILDING RAG KNOWLEDGE BASE")
    print("=" * 70)

    knowledge_base = APMOAKnowledgeBase(
        knowledge_dir=KNOWLEDGE_DIR
    )

    knowledge_base.build()

    print(
        "\n✓ Knowledge base built successfully."
    )

    # ----------------------------------------------------------
    # Experiment A
    # ----------------------------------------------------------

    rag_result = run_with_rag(
        knowledge_base
    )

    # ----------------------------------------------------------
    # Experiment B
    # ----------------------------------------------------------

    no_rag_result = run_without_rag()

    # ----------------------------------------------------------
    # Comparison
    # ----------------------------------------------------------

    comparison = compare_results(
        rag_result,
        no_rag_result,
    )

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    save_results(
        rag_result,
        no_rag_result,
        comparison,
    )

    # ----------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RAG ABLATION EXPERIMENT COMPLETED")
    print("=" * 70)

    print("\nConfiguration A:")
    print("  RAG + Groq")

    print("\nConfiguration B:")
    print("  No RAG + Groq")

    print(
        "\nBoth configurations used:"
    )

    print(
        "  • Same process diagnostics"
    )

    print(
        "  • Same model evaluations"
    )

    print(
        "  • Same selected model"
    )

    print(
        "  • Same Reasoning Agent"
    )

    print(
        "\nThe only intended experimental difference "
        "is the availability of RAG knowledge."
    )

    print(
        "\n✓ RAG ABLATION TEST COMPLETED"
    )


if __name__ == "__main__":
    main()