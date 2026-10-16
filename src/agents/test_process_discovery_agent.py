"""
APMOA - Process Discovery Agent Test

Tests the ProcessDiscoveryAgent against the Travel event log
and verifies that all configured discovery algorithms
successfully produce models.

Run:
    python -m src.agents.test_process_discovery_agent
"""

import time
import traceback

from src.mining.loader import EventLogLoader
from src.agents.process_discovery_agent import (
    ProcessDiscoveryAgent,
)


# ============================================================
# CONFIGURATION
# ============================================================

LOG_PATH = "data/raw/Travel/PermitLog.xes.gz"


# These are the five algorithms currently supported
# by the APMOA discovery stage.
EXPECTED_MODELS = [
    "Inductive Miner",
    "Alpha Miner",
    "Alpha+ Miner",
    "Heuristics Miner",
    "Split Miner",
]


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("\n" + "=" * 70)
    print("APMOA - PROCESS DISCOVERY AGENT TEST")
    print("=" * 70)

    print(
        "\nExpected discovery algorithms:"
    )

    for index, name in enumerate(
        EXPECTED_MODELS,
        start=1,
    ):
        print(
            f"  {index}. {name}"
        )

    # --------------------------------------------------------
    # LOAD EVENT LOG
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING EVENT LOG")
    print("=" * 70)

    print(
        f"Path: {LOG_PATH}"
    )

    start = time.time()

    try:

        loader = EventLogLoader(
            LOG_PATH
        )

        log = loader.load()

    except Exception as exc:

        print(
            "\nERROR: Failed to load event log."
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        traceback.print_exc()

        raise

    elapsed = time.time() - start

    print(
        f"\nEvent log loaded successfully "
        f"in {elapsed:.2f}s."
    )

    print(
        f"Events: {len(log):,}"
    )

    # --------------------------------------------------------
    # CREATE DISCOVERY AGENT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INITIALIZING DISCOVERY AGENT")
    print("=" * 70)

    start = time.time()

    try:

        agent = ProcessDiscoveryAgent(
            log
        )

    except Exception as exc:

        print(
            "\nERROR: Failed to initialize "
            "ProcessDiscoveryAgent."
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        traceback.print_exc()

        raise

    elapsed = time.time() - start

    print(
        f"Discovery Agent initialized "
        f"in {elapsed:.2f}s."
    )

    # --------------------------------------------------------
    # DISCOVER ALL MODELS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RUNNING ALL DISCOVERY ALGORITHMS")
    print("=" * 70)

    print(
        "\nThis may take some time depending "
        "on the event log and algorithm."
    )

    start = time.time()

    try:

        models = agent.discover_all()

    except Exception as exc:

        print(
            "\nERROR: Discovery failed."
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        traceback.print_exc()

        raise

    total_elapsed = time.time() - start

    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DISCOVERY RESULTS")
    print("=" * 70)

    assert models is not None, (
        "discover_all() returned None."
    )

    assert isinstance(
        models,
        dict,
    ), (
        "discover_all() must return a dictionary."
    )

    print(
        f"\nModels returned: {len(models)}"
    )

    # --------------------------------------------------------
    # DISPLAY RETURNED MODEL NAMES
    # --------------------------------------------------------

    print("\nReturned models:")

    for name in models.keys():

        print(
            f"  ✓ {name}"
        )

    # --------------------------------------------------------
    # CHECK EXPECTED MODELS
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("MODEL VALIDATION")
    print("-" * 70)

    missing_models = []

    for expected_model in EXPECTED_MODELS:

        if expected_model in models:

            print(
                f"✓ {expected_model:<25} FOUND"
            )

        else:

            print(
                f"✗ {expected_model:<25} MISSING"
            )

            missing_models.append(
                expected_model
            )

    # --------------------------------------------------------
    # CHECK FOR EXTRA MODELS
    # --------------------------------------------------------

    expected_set = set(
        EXPECTED_MODELS
    )

    returned_set = set(
        models.keys()
    )

    extra_models = (
        returned_set
        - expected_set
    )

    if extra_models:

        print("\nAdditional models returned:")

        for name in sorted(
            extra_models
        ):

            print(
                f"  + {name}"
            )

    # --------------------------------------------------------
    # MODEL OBJECT VALIDATION
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("MODEL OBJECT VALIDATION")
    print("-" * 70)

    invalid_models = []

    for name in EXPECTED_MODELS:

        if name not in models:
            continue

        model = models[name]

        if model is None:

            print(
                f"✗ {name:<25} "
                f"returned None"
            )

            invalid_models.append(
                name
            )

            continue

        print(
            f"✓ {name:<25} "
            f"model object created"
        )

        # ----------------------------------------------------
        # Petri-net models
        # ----------------------------------------------------

        if isinstance(
            model,
            dict,
        ):

            print(
                f"    Returned object type: dict"
            )

            continue

        # ----------------------------------------------------
        # Generic object validation
        # ----------------------------------------------------

        print(
            f"    Type: {type(model).__name__}"
        )

        # ----------------------------------------------------
        # Check Petri net structure where available
        # ----------------------------------------------------

        if hasattr(
            model,
            "places",
        ):

            print(
                f"    Places      : "
                f"{len(model.places)}"
            )

        if hasattr(
            model,
            "transitions",
        ):

            print(
                f"    Transitions : "
                f"{len(model.transitions)}"
            )

        if hasattr(
            model,
            "arcs",
        ):

            print(
                f"    Arcs        : "
                f"{len(model.arcs)}"
            )

        # ----------------------------------------------------
        # Check tuple-based Petri net result
        # ----------------------------------------------------

        if isinstance(
            model,
            tuple,
        ):

            if len(model) == 3:

                net = model[0]

                initial_marking = model[1]

                final_marking = model[2]

                print(
                    f"    Tuple length: 3"
                )

                if hasattr(
                    net,
                    "places",
                ):

                    print(
                        f"    Places      : "
                        f"{len(net.places)}"
                    )

                if hasattr(
                    net,
                    "transitions",
                ):

                    print(
                        f"    Transitions : "
                        f"{len(net.transitions)}"
                    )

                if hasattr(
                    net,
                    "arcs",
                ):

                    print(
                        f"    Arcs        : "
                        f"{len(net.arcs)}"
                    )

                print(
                    f"    Initial marking: "
                    f"{type(initial_marking).__name__}"
                )

                print(
                    f"    Final marking  : "
                    f"{type(final_marking).__name__}"
                )

    # --------------------------------------------------------
    # ASSERTIONS
    # --------------------------------------------------------

    assert not missing_models, (
        "Missing discovery algorithms: "
        f"{missing_models}"
    )

    assert not invalid_models, (
        "Invalid model objects: "
        f"{invalid_models}"
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n\n" + "=" * 70)
    print("PROCESS DISCOVERY AGENT TEST SUMMARY")
    print("=" * 70)

    print(
        f"\nExpected models       : "
        f"{len(EXPECTED_MODELS)}"
    )

    print(
        f"Models returned       : "
        f"{len(models)}"
    )

    print(
        f"Missing models        : "
        f"{len(missing_models)}"
    )

    print(
        f"Invalid models        : "
        f"{len(invalid_models)}"
    )

    print(
        f"Total discovery time  : "
        f"{total_elapsed:.2f}s"
    )

    print("\n" + "-" * 70)

    for name in EXPECTED_MODELS:

        if name in models:

            print(
                f"✓ {name:<25} SUCCESS"
            )

        else:

            print(
                f"✗ {name:<25} FAILED"
            )

    print("=" * 70)

    # --------------------------------------------------------
    # FINAL ASSERTION
    # --------------------------------------------------------

    assert (
        len(missing_models) == 0
    ), (
        "Not all expected models were discovered."
    )

    print(
        "\nALL DISCOVERY AGENT TESTS PASSED"
    )

    print("=" * 70)

    print(
        "\nThe Process Discovery Agent successfully "
        "supports all five configured algorithms:"
    )

    for name in EXPECTED_MODELS:

        print(
            f"  ✓ {name}"
        )

    print("\n" + "=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()