"""
APMOA - Evaluation Agent Test

Tests the EvaluationAgent against all process discovery
algorithms currently configured in APMOA.

Expected discovery algorithms:

    1. Inductive Miner
    2. Alpha Miner
    3. Alpha+ Miner
    4. Heuristics Miner
    5. Split Miner

Run:

    python -m src.agents.test_evaluation_agent
"""

import pm4py

from src.agents.process_discovery_agent import (
    ProcessDiscoveryAgent
)

from src.agents.evaluation_agent import (
    EvaluationAgent
)


# ============================================================
# CONFIGURATION
# ============================================================

LOG_PATH = (
    "data/raw/Travel/PermitLog.xes.gz"
)

EXPECTED_MODEL_COUNT = 5

EXPECTED_MODELS = {
    "Inductive Miner",
    "Alpha Miner",
    "Alpha+ Miner",
    "Heuristics Miner",
    "Split Miner",
}


# ============================================================
# LOAD EVENT LOG
# ============================================================

def load_event_log():

    print("\n" + "=" * 70)
    print("LOADING EVENT LOG")
    print("=" * 70)

    print(
        f"Path: {LOG_PATH}"
    )

    log = pm4py.read_xes(
        LOG_PATH
    )

    print(
        f"\nLoaded {len(log):,} events."
    )

    try:

        print(
            f"Traces: {len(log):,}"
        )

    except Exception:
        pass

    return log


# ============================================================
# TEST DISCOVERY
# ============================================================

def run_discovery(log):

    print("\n" + "=" * 70)
    print("PROCESS DISCOVERY")
    print("=" * 70)

    discovery_agent = ProcessDiscoveryAgent(
        log
    )

    models = (
        discovery_agent.discover_all()
    )

    print(
        f"\nModels discovered: "
        f"{len(models)}"
    )

    # --------------------------------------------------------
    # Validate model count
    # --------------------------------------------------------

    assert len(models) == EXPECTED_MODEL_COUNT, (
        f"Expected {EXPECTED_MODEL_COUNT} models, "
        f"but discovered {len(models)}."
    )

    return models


# ============================================================
# GET MODEL NAMES
# ============================================================

def get_model_names(models):

    names = set()

    for model in models:

        if isinstance(
            model,
            dict
        ):

            name = (
                model.get("name")
                or model.get("model")
            )

            if name:
                names.add(name)

        else:

            # ------------------------------------------------
            # Fallback if models are returned in another form.
            # ------------------------------------------------

            try:

                name = model.name

                if name:
                    names.add(name)

            except AttributeError:

                pass

    return names


# ============================================================
# RUN EVALUATION
# ============================================================

def run_evaluation(
    log,
    models,
):

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    evaluation_agent = EvaluationAgent(
        log
    )

    results = (
        evaluation_agent.evaluate(
            models
        )
    )

    assert results is not None, (
        "EvaluationAgent returned None."
    )

    return results


# ============================================================
# VALIDATE RESULTS
# ============================================================

def validate_results(
    results,
):

    print("\n" + "=" * 70)
    print("VALIDATING EVALUATION RESULTS")
    print("=" * 70)

    # --------------------------------------------------------
    # Convert DataFrame-like result if necessary
    # --------------------------------------------------------

    if hasattr(
        results,
        "to_dict"
    ):

        records = results.to_dict(
            orient="records"
        )

    elif isinstance(
        results,
        list
    ):

        records = results

    else:

        raise AssertionError(
            "Evaluation results must be a "
            "list or DataFrame-like object."
        )

    # --------------------------------------------------------
    # Validate number of results
    # --------------------------------------------------------

    assert len(records) == EXPECTED_MODEL_COUNT, (
        f"Expected {EXPECTED_MODEL_COUNT} evaluation results, "
        f"but received {len(records)}."
    )

    # --------------------------------------------------------
    # Validate required metrics
    # --------------------------------------------------------

    required_fields = {
        "model",
        "fitness",
        "precision",
        "generalization",
        "simplicity",
    }

    evaluated_models = set()

    for result in records:

        print("\n" + "-" * 60)

        model_name = result.get(
            "model"
        )

        print(
            f"Model: {model_name}"
        )

        # ----------------------------------------------------
        # Validate model name
        # ----------------------------------------------------

        assert model_name in EXPECTED_MODELS, (
            f"Unexpected model evaluated: "
            f"{model_name}"
        )

        evaluated_models.add(
            model_name
        )

        # ----------------------------------------------------
        # Validate fields
        # ----------------------------------------------------

        missing = (
            required_fields
            - set(result.keys())
        )

        assert not missing, (
            f"{model_name} is missing fields: "
            f"{missing}"
        )

        # ----------------------------------------------------
        # Validate metrics
        # ----------------------------------------------------

        for metric in [
            "fitness",
            "precision",
            "generalization",
            "simplicity",
        ]:

            value = float(
                result[metric]
            )

            assert 0.0 <= value <= 1.0, (
                f"{model_name} "
                f"{metric}={value} "
                f"is outside [0, 1]."
            )

            print(
                f"{metric:<18}: "
                f"{value:.4f}"
            )

    # --------------------------------------------------------
    # Validate all five algorithms were evaluated
    # --------------------------------------------------------

    assert (
        evaluated_models
        == EXPECTED_MODELS
    ), (
        "\nNot all expected models were evaluated.\n"
        f"Expected: {EXPECTED_MODELS}\n"
        f"Actual:   {evaluated_models}"
    )

    return records


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("APMOA - EVALUATION AGENT TEST")
    print("=" * 70)

    print(
        f"\nPM4Py version: "
        f"{pm4py.__version__}"
    )

    # --------------------------------------------------------
    # Step 1 - Load event log
    # --------------------------------------------------------

    try:

        log = load_event_log()

    except Exception as exc:

        print(
            "\nERROR: Failed to load event log."
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise

    # --------------------------------------------------------
    # Step 2 - Discover all models
    # --------------------------------------------------------

    try:

        models = run_discovery(
            log
        )

    except Exception as exc:

        print(
            "\nERROR: Process discovery failed."
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise

    # --------------------------------------------------------
    # Step 3 - Evaluate all models
    # --------------------------------------------------------

    try:

        results = run_evaluation(
            log,
            models
        )

    except Exception as exc:

        print(
            "\nERROR: Model evaluation failed."
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise

    # --------------------------------------------------------
    # Step 4 - Validate
    # --------------------------------------------------------

    try:

        records = validate_results(
            results
        )

    except Exception as exc:

        print(
            "\nERROR: Evaluation validation failed."
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)

    print(
        f"\n{'Model':<25}"
        f"{'Fitness':<12}"
        f"{'Precision':<12}"
        f"{'Generalization':<17}"
        f"{'Simplicity':<12}"
    )

    print("-" * 80)

    for result in records:

        print(
            f"{result['model']:<25}"
            f"{float(result['fitness']):<12.4f}"
            f"{float(result['precision']):<12.4f}"
            f"{float(result['generalization']):<17.4f}"
            f"{float(result['simplicity']):<12.4f}"
        )

    # ========================================================
    # SUCCESS
    # ========================================================

    print("\n" + "=" * 70)
    print("EVALUATION AGENT TEST PASSED")
    print("=" * 70)

    print(
        "\nSuccessfully discovered and evaluated "
        "all five process discovery algorithms:"
    )

    for model_name in sorted(
        EXPECTED_MODELS
    ):

        print(
            f"  ✓ {model_name}"
        )

    print(
        "\nTotal models evaluated: "
        f"{len(records)}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()