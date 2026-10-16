"""
APMOA - Test All Process Discovery Algorithms

Tests all supported/configured discovery algorithms against
the Travel event log.

Algorithms tested:
    1. Inductive Miner
    2. Alpha Miner
    3. Alpha+ Miner
    4. Heuristics Miner
    5. Split Miner

Run:
    python -m src.agents.test_all_discovery
"""

import os
import time
import traceback

import pm4py


# ============================================================
# CONFIGURATION
# ============================================================

LOG_PATH = "data/raw/Travel/PermitLog.xes.gz"

OUTPUT_DIR = "results/travel/discovery_tests"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# LOAD EVENT LOG
# ============================================================

def load_event_log():

    print("\n" + "=" * 70)
    print("LOADING EVENT LOG")
    print("=" * 70)

    print(f"Path: {LOG_PATH}")

    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(
            f"Event log not found: {LOG_PATH}"
        )

    start = time.time()

    log = pm4py.read_xes(
        LOG_PATH
    )

    elapsed = time.time() - start

    print(
        f"Event log loaded successfully "
        f"in {elapsed:.2f} seconds."
    )

    # --------------------------------------------------------
    # PM4Py EventLog length represents traces, not events.
    # --------------------------------------------------------

    try:

        number_of_traces = len(log)

        number_of_events = sum(
            len(trace)
            for trace in log
        )

        print(
            f"Events: {number_of_events:,}"
        )

        print(
            f"Traces: {number_of_traces:,}"
        )

    except Exception:

        print(
            "Could not calculate event/trace counts."
        )

    return log


# ============================================================
# RESULT HELPER
# ============================================================

def create_petri_result(
    name,
    net,
    initial_marking,
    final_marking,
    elapsed,
):

    return {

        "name": name,

        "type": "petri_net",

        "net": net,

        "initial_marking":
            initial_marking,

        "final_marking":
            final_marking,

        "places":
            len(net.places),

        "transitions":
            len(net.transitions),

        "arcs":
            len(net.arcs),

        "time":
            elapsed,

        "status":
            "SUCCESS",
    }


# ============================================================
# INDUCTIVE MINER
# ============================================================

def test_inductive(log):

    print("\n" + "-" * 60)
    print("TESTING: INDUCTIVE MINER")
    print("-" * 60)

    start = time.time()

    (
        net,
        initial_marking,
        final_marking,
    ) = pm4py.discover_petri_net_inductive(
        log
    )

    elapsed = time.time() - start

    print(
        "SUCCESS - Inductive Miner"
    )

    print(
        f"Places      : {len(net.places)}"
    )

    print(
        f"Transitions : {len(net.transitions)}"
    )

    print(
        f"Arcs        : {len(net.arcs)}"
    )

    print(
        f"Time        : {elapsed:.2f}s"
    )

    return create_petri_result(
        "Inductive Miner",
        net,
        initial_marking,
        final_marking,
        elapsed,
    )


# ============================================================
# ALPHA MINER
# ============================================================

def test_alpha(log):

    print("\n" + "-" * 60)
    print("TESTING: ALPHA MINER")
    print("-" * 60)

    start = time.time()

    (
        net,
        initial_marking,
        final_marking,
    ) = pm4py.discover_petri_net_alpha(
        log
    )

    elapsed = time.time() - start

    print(
        "SUCCESS - Alpha Miner"
    )

    print(
        f"Places      : {len(net.places)}"
    )

    print(
        f"Transitions : {len(net.transitions)}"
    )

    print(
        f"Arcs        : {len(net.arcs)}"
    )

    print(
        f"Time        : {elapsed:.2f}s"
    )

    return create_petri_result(
        "Alpha Miner",
        net,
        initial_marking,
        final_marking,
        elapsed,
    )


# ============================================================
# ALPHA+ MINER
# ============================================================

def test_alpha_plus(log):

    print("\n" + "-" * 60)
    print("TESTING: ALPHA+ MINER")
    print("-" * 60)

    start = time.time()

    (
        net,
        initial_marking,
        final_marking,
    ) = pm4py.discover_petri_net_alpha_plus(
        log
    )

    elapsed = time.time() - start

    print(
        "SUCCESS - Alpha+ Miner"
    )

    print(
        f"Places      : {len(net.places)}"
    )

    print(
        f"Transitions : {len(net.transitions)}"
    )

    print(
        f"Arcs        : {len(net.arcs)}"
    )

    print(
        f"Time        : {elapsed:.2f}s"
    )

    return create_petri_result(
        "Alpha+ Miner",
        net,
        initial_marking,
        final_marking,
        elapsed,
    )


# ============================================================
# HEURISTICS MINER
# ============================================================

def test_heuristics(log):

    print("\n" + "-" * 60)
    print("TESTING: HEURISTICS MINER")
    print("-" * 60)

    start = time.time()

    (
        net,
        initial_marking,
        final_marking,
    ) = pm4py.discover_petri_net_heuristics(
        log
    )

    elapsed = time.time() - start

    print(
        "SUCCESS - Heuristics Miner"
    )

    print(
        f"Places      : {len(net.places)}"
    )

    print(
        f"Transitions : {len(net.transitions)}"
    )

    print(
        f"Arcs        : {len(net.arcs)}"
    )

    print(
        f"Time        : {elapsed:.2f}s"
    )

    return create_petri_result(
        "Heuristics Miner",
        net,
        initial_marking,
        final_marking,
        elapsed,
    )


# ============================================================
# SPLIT MINER
# ============================================================

def test_split_miner(log):

    print("\n" + "-" * 60)
    print("TESTING: SPLIT MINER")
    print("-" * 60)

    start = time.time()

    bpmn = pm4py.discover_bpmn_split_miner(
        log
    )

    elapsed = time.time() - start

    print(
        "SUCCESS - Split Miner"
    )

    print(
        f"Time        : {elapsed:.2f}s"
    )

    # --------------------------------------------------------
    # Count BPMN nodes/flows if available
    # --------------------------------------------------------

    nodes = None
    flows = None

    try:

        nodes = len(
            list(
                bpmn.get_nodes()
            )
        )

    except Exception:
        pass

    try:

        flows = len(
            list(
                bpmn.get_flows()
            )
        )

    except Exception:
        pass

    if nodes is not None:

        print(
            f"Nodes       : {nodes}"
        )

    if flows is not None:

        print(
            f"Flows       : {flows}"
        )

    return {

        "name":
            "Split Miner",

        "type":
            "bpmn",

        "bpmn":
            bpmn,

        "nodes":
            nodes,

        "flows":
            flows,

        "time":
            elapsed,

        "status":
            "SUCCESS",
    }


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "APMOA - ALL DISCOVERY ALGORITHM TEST"
    )

    print("=" * 70)

    print(
        f"\nPM4Py version: "
        f"{pm4py.__version__}"
    )

    # --------------------------------------------------------
    # Load event log
    # --------------------------------------------------------

    try:

        log = load_event_log()

    except Exception as exc:

        print(
            "\nERROR loading event log:"
        )

        print(exc)

        traceback.print_exc()

        return

    # --------------------------------------------------------
    # Discovery algorithms
    #
    # IMPORTANT:
    # ILP Miner has intentionally been removed because it
    # becomes computationally expensive for this event log.
    # --------------------------------------------------------

    tests = [

        (
            "Inductive Miner",
            test_inductive,
        ),

        (
            "Alpha Miner",
            test_alpha,
        ),

        (
            "Alpha+ Miner",
            test_alpha_plus,
        ),

        (
            "Heuristics Miner",
            test_heuristics,
        ),

        (
            "Split Miner",
            test_split_miner,
        ),
    ]

    results = []

    # --------------------------------------------------------
    # Execute algorithms
    # --------------------------------------------------------

    for name, test_function in tests:

        try:

            result = test_function(
                log
            )

            results.append(
                result
            )

        except Exception as exc:

            print("\nFAILED")

            print(
                f"Algorithm: {name}"
            )

            print(
                f"Error: "
                f"{type(exc).__name__}: {exc}"
            )

            traceback.print_exc()

            results.append(
                {

                    "name":
                        name,

                    "status":
                        "FAILED",

                    "error":
                        str(exc),

                    "time":
                        0.0,
                }
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n\n" + "=" * 70)

    print(
        "DISCOVERY ALGORITHM TEST SUMMARY"
    )

    print("=" * 70)

    print()

    print(
        f"{'Algorithm':<25}"
        f"{'Status':<12}"
        f"{'Time (s)':<12}"
    )

    print(
        "-" * 70
    )

    for result in results:

        name = result.get(
            "name",
            "Unknown",
        )

        status = result.get(
            "status",
            "UNKNOWN",
        )

        elapsed = result.get(
            "time",
            0.0,
        )

        print(
            f"{name:<25}"
            f"{status:<12}"
            f"{elapsed:<12.2f}"
        )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    successful = sum(

        1

        for result in results

        if result.get(
            "status"
        ) == "SUCCESS"

    )

    failed = (
        len(results)
        - successful
    )

    print("\n" + "=" * 70)

    print(
        f"Successful algorithms : "
        f"{successful}"
    )

    print(
        f"Failed algorithms     : "
        f"{failed}"
    )

    print(
        f"Total algorithms tested : "
        f"{len(results)}"
    )

    print("=" * 70)

    if failed == 0:

        print(
            "\nALL DISCOVERY ALGORITHMS PASSED"
        )

    else:

        print(
            "\nSOME DISCOVERY ALGORITHMS FAILED"
        )

        print(
            "\nFailed algorithms:"
        )

        for result in results:

            if result.get(
                "status"
            ) != "SUCCESS":

                print(
                    f"  - {result.get('name')}: "
                    f"{result.get('error')}"
                )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()