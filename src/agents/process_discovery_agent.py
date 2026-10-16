"""
APMOA - Process Discovery Agent

Discovers multiple candidate process models from an event log.

Supported discovery algorithms:

    1. Inductive Miner
    2. Alpha Miner
    3. Alpha+ Miner
    4. Heuristics Miner
    5. Split Miner

ILP Miner is intentionally excluded because it is computationally
expensive for the current event log and can remain stuck for a
very long time.

The discovery agent returns a dictionary whose keys are the
algorithm names. This structure is consumed by the Evaluation
Agent and Model Selection Agent.

Canonical APMOA event-log schema:

    case_id
    activity
    timestamp
    resource

The agent also creates PM4Py-compatible aliases internally:

    case:concept:name
    concept:name
    time:timestamp
    org:resource

Run standalone test:

    python -m src.agents.process_discovery_agent

Or use:

    python -m src.agents.test_process_discovery_agent
"""

import os
import time
import traceback
from typing import Any, Dict

import pandas as pd
import pm4py


class ProcessDiscoveryAgent:
    """
    Autonomous Process Discovery Agent.

    The agent discovers multiple candidate process models from
    the supplied event log so that downstream evaluation and
    model-selection agents can compare them.

    The input can be:

    1. A canonical pandas DataFrame produced by DatasetAdapter.
    2. A PM4Py EventLog.
    3. Another PM4Py-compatible event-log object.

    Canonical DataFrame columns:

        case_id
        activity
        timestamp
        resource

    Each algorithm is isolated so that failure of one discovery
    algorithm does not prevent the remaining algorithms from
    running.
    """

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        log,
        output_dir: str = "results/travel",
    ):
        """
        Initialize the Process Discovery Agent.

        Parameters
        ----------
        log:
            Canonical pandas DataFrame, PM4Py EventLog,
            or compatible event-log object.

        output_dir:
            Directory used for storing discovered models.
        """

        # -----------------------------------------------------
        # Prepare event log for PM4Py
        # -----------------------------------------------------

        self.log = self._prepare_event_log(log)

        self.output_dir = output_dir

        os.makedirs(
            self.output_dir,
            exist_ok=True,
        )

        # -----------------------------------------------------
        # Supported discovery algorithms
        # -----------------------------------------------------

        self.algorithms = [
            "Inductive Miner",
            "Alpha Miner",
            "Alpha+ Miner",
            "Heuristics Miner",
            "Split Miner",
        ]

    # =========================================================
    # EVENT LOG NORMALIZATION
    # =========================================================

    @staticmethod
    def _prepare_event_log(log):
        """
        Prepare an event log for PM4Py discovery.

        Supports both:

        1. Canonical APMOA schema:
           case_id
           activity
           timestamp
           resource

        2. Native PM4Py/XES schema:
           case:concept:name
           concept:name
           time:timestamp
           org:resource
        """

        # =====================================================
        # CASE 1 - pandas DataFrame
        # =====================================================

        if isinstance(log, pd.DataFrame):

            prepared_log = log.copy()

            print("\nPreparing event log for PM4Py...")
            print(
                f"Input columns: {list(prepared_log.columns)}"
            )

            # =================================================
            # FORMAT 1 - CANONICAL APMOA SCHEMA
            # =================================================

            canonical_columns = {
                "case_id",
                "activity",
                "timestamp",
            }

            if canonical_columns.issubset(
                prepared_log.columns
            ):

                print(
                    "Detected canonical APMOA "
                    "event-log schema."
                )

                # ---------------------------------------------
                # Case ID
                # ---------------------------------------------

                prepared_log["case:concept:name"] = (
                    prepared_log["case_id"]
                    .astype(str)
                )

                # ---------------------------------------------
                # Activity
                # ---------------------------------------------

                prepared_log["concept:name"] = (
                    prepared_log["activity"]
                    .astype(str)
                )

                # ---------------------------------------------
                # Timestamp
                # ---------------------------------------------

                prepared_log["time:timestamp"] = (
                    pd.to_datetime(
                        prepared_log["timestamp"],
                        errors="coerce",
                    )
                )

                # ---------------------------------------------
                # Resource
                # ---------------------------------------------

                if "resource" in prepared_log.columns:

                    prepared_log["org:resource"] = (
                        prepared_log["resource"]
                        .fillna("UNKNOWN")
                        .astype(str)
                    )

                else:

                    prepared_log["org:resource"] = (
                        "UNKNOWN"
                    )

            # =================================================
            # FORMAT 2 - NATIVE PM4PY / XES SCHEMA
            # =================================================

            elif {
                "case:concept:name",
                "concept:name",
                "time:timestamp",
            }.issubset(prepared_log.columns):

                print(
                    "Detected native PM4Py/XES "
                    "event-log schema."
                )

                # ---------------------------------------------
                # Case ID
                # ---------------------------------------------

                prepared_log["case:concept:name"] = (
                    prepared_log["case:concept:name"]
                    .astype(str)
                )

                # ---------------------------------------------
                # Activity
                # ---------------------------------------------

                prepared_log["concept:name"] = (
                    prepared_log["concept:name"]
                    .astype(str)
                )

                # ---------------------------------------------
                # Timestamp
                # ---------------------------------------------

                prepared_log["time:timestamp"] = (
                    pd.to_datetime(
                        prepared_log["time:timestamp"],
                        errors="coerce",
                    )
                )

                # ---------------------------------------------
                # Resource
                # ---------------------------------------------

                if "org:resource" not in (
                    prepared_log.columns
                ):

                    prepared_log["org:resource"] = (
                        "UNKNOWN"
                    )

            # =================================================
            # UNKNOWN DATASET FORMAT
            # =================================================

            else:

                raise ValueError(
                    "\nUnable to identify event-log schema.\n\n"

                    "Expected either:\n\n"

                    "Canonical APMOA schema:\n"
                    "  case_id\n"
                    "  activity\n"
                    "  timestamp\n"
                    "  resource (optional)\n\n"

                    "OR native PM4Py/XES schema:\n"
                    "  case:concept:name\n"
                    "  concept:name\n"
                    "  time:timestamp\n"
                    "  org:resource (optional)\n\n"

                    f"Available columns:\n"
                    f"{list(prepared_log.columns)}"
                )

            # =================================================
            # TIMESTAMP VALIDATION
            # =================================================

            invalid_timestamps = (
                prepared_log["time:timestamp"]
                .isna()
                .sum()
            )

            if invalid_timestamps > 0:

                raise ValueError(
                    f"Event log contains "
                    f"{invalid_timestamps} invalid "
                    f"timestamp value(s)."
                )

            # =================================================
            # STABLE EVENT ORDERING
            # =================================================

            prepared_log = (
                prepared_log
                .sort_values(
                    [
                        "case:concept:name",
                        "time:timestamp",
                    ],
                    kind="stable",
                )
                .reset_index(drop=True)
            )

            # =================================================
            # DIAGNOSTICS
            # =================================================

            print("\nEvent log prepared for PM4Py.")

            print(
                f"  Events     : "
                f"{len(prepared_log):,}"
            )

            print(
                f"  Cases      : "
                f"{prepared_log['case:concept:name'].nunique():,}"
            )

            print(
                f"  Activities : "
                f"{prepared_log['concept:name'].nunique():,}"
            )

            print(
                f"  Resources  : "
                f"{prepared_log['org:resource'].nunique():,}"
            )

            print("\nPM4Py compatibility columns:")

            print(
                "  ✓ case:concept:name"
            )

            print(
                "  ✓ concept:name"
            )

            print(
                "  ✓ time:timestamp"
            )

            print(
                "  ✓ org:resource"
            )

            return prepared_log

        # =====================================================
        # CASE 2 - PM4Py EventLog
        # =====================================================

        try:

            from pm4py.objects.log.obj import EventLog

            if isinstance(log, EventLog):

                print(
                    "\nDetected PM4Py EventLog object."
                )

                return log

        except Exception:
            pass

        # =====================================================
        # UNSUPPORTED INPUT
        # =====================================================

        raise TypeError(
            "Unsupported event-log type: "
            f"{type(log)}"
        )

    # =========================================================
    # UTILITY
    # =========================================================

    def _print_header(
        self,
        title: str,
    ):
        """
        Print a formatted section header.
        """

        print(
            "\n" + "=" * 60
        )

        print(
            title
        )

        print(
            "=" * 60
        )

    # =========================================================
    # INDUCTIVE MINER
    # =========================================================

    def _discover_inductive(
        self,
    ) -> Dict[str, Any]:

        print(
            "\nRunning Inductive Miner..."
        )

        start = time.time()

        net, initial_marking, final_marking = (
            pm4py.discover_petri_net_inductive(
                self.log,
                case_id_key="case:concept:name",
                activity_key="concept:name",
                timestamp_key="time:timestamp",
            )
        )

        elapsed = time.time() - start

        output_path = os.path.join(
            self.output_dir,
            "inductive_miner_model.pnml",
        )

        # -----------------------------------------------------
        # Save Petri net
        # -----------------------------------------------------

        try:

            pm4py.write_pnml(
                net,
                initial_marking,
                final_marking,
                output_path,
            )

        except Exception as exc:

            print(
                "Warning: could not save Inductive Miner "
                f"model: {exc}"
            )

        # -----------------------------------------------------
        # Statistics
        # -----------------------------------------------------

        print(
            "Inductive Miner completed."
        )

        print(
            f"  Places      : {len(net.places)}"
        )

        print(
            f"  Transitions : {len(net.transitions)}"
        )

        print(
            f"  Arcs        : {len(net.arcs)}"
        )

        print(
            f"  Time        : {elapsed:.2f}s"
        )

        return {

            "name":
                "Inductive Miner",

            "algorithm":
                "inductive",

            "type":
                "petri_net",

            "net":
                net,

            "initial_marking":
                initial_marking,

            "final_marking":
                final_marking,

            "time":
                elapsed,

            "status":
                "SUCCESS",

            "output_path":
                output_path,
        }

    # =========================================================
    # ALPHA MINER
    # =========================================================

    def _discover_alpha(
        self,
    ) -> Dict[str, Any]:

        print(
            "\nRunning Alpha Miner..."
        )

        start = time.time()

        net, initial_marking, final_marking = (
            pm4py.discover_petri_net_alpha(
                self.log,
                case_id_key="case:concept:name",
                activity_key="concept:name",
                timestamp_key="time:timestamp",
            )
        )

        elapsed = time.time() - start

        output_path = os.path.join(
            self.output_dir,
            "alpha_miner_model.pnml",
        )

        # -----------------------------------------------------
        # Save Petri net
        # -----------------------------------------------------

        try:

            pm4py.write_pnml(
                net,
                initial_marking,
                final_marking,
                output_path,
            )

        except Exception as exc:

            print(
                "Warning: could not save Alpha Miner "
                f"model: {exc}"
            )

        # -----------------------------------------------------
        # Statistics
        # -----------------------------------------------------

        print(
            "Alpha Miner completed."
        )

        print(
            f"  Places      : {len(net.places)}"
        )

        print(
            f"  Transitions : {len(net.transitions)}"
        )

        print(
            f"  Arcs        : {len(net.arcs)}"
        )

        print(
            f"  Time        : {elapsed:.2f}s"
        )

        return {

            "name":
                "Alpha Miner",

            "algorithm":
                "alpha",

            "type":
                "petri_net",

            "net":
                net,

            "initial_marking":
                initial_marking,

            "final_marking":
                final_marking,

            "time":
                elapsed,

            "status":
                "SUCCESS",

            "output_path":
                output_path,
        }

    # =========================================================
    # ALPHA+ MINER
    # =========================================================

    def _discover_alpha_plus(
        self,
    ) -> Dict[str, Any]:

        print(
            "\nRunning Alpha+ Miner..."
        )

        start = time.time()

        net, initial_marking, final_marking = (
            pm4py.discover_petri_net_alpha_plus(
                self.log,
                case_id_key="case:concept:name",
                activity_key="concept:name",
                timestamp_key="time:timestamp",
            )
        )

        elapsed = time.time() - start

        output_path = os.path.join(
            self.output_dir,
            "alpha_plus_miner_model.pnml",
        )

        # -----------------------------------------------------
        # Save Petri net
        # -----------------------------------------------------

        try:

            pm4py.write_pnml(
                net,
                initial_marking,
                final_marking,
                output_path,
            )

        except Exception as exc:

            print(
                "Warning: could not save Alpha+ Miner "
                f"model: {exc}"
            )

        # -----------------------------------------------------
        # Statistics
        # -----------------------------------------------------

        print(
            "Alpha+ Miner completed."
        )

        print(
            f"  Places      : {len(net.places)}"
        )

        print(
            f"  Transitions : {len(net.transitions)}"
        )

        print(
            f"  Arcs        : {len(net.arcs)}"
        )

        print(
            f"  Time        : {elapsed:.2f}s"
        )

        return {

            "name":
                "Alpha+ Miner",

            "algorithm":
                "alpha_plus",

            "type":
                "petri_net",

            "net":
                net,

            "initial_marking":
                initial_marking,

            "final_marking":
                final_marking,

            "time":
                elapsed,

            "status":
                "SUCCESS",

            "output_path":
                output_path,
        }

    # =========================================================
    # HEURISTICS MINER
    # =========================================================

    def _discover_heuristics(
        self,
    ) -> Dict[str, Any]:

        print(
            "\nRunning Heuristics Miner..."
        )

        start = time.time()

        net, initial_marking, final_marking = (
            pm4py.discover_petri_net_heuristics(
                self.log,
                case_id_key="case:concept:name",
                activity_key="concept:name",
                timestamp_key="time:timestamp",
            )
        )

        elapsed = time.time() - start

        output_path = os.path.join(
            self.output_dir,
            "heuristics_miner_model.pnml",
        )

        # -----------------------------------------------------
        # Save Petri net
        # -----------------------------------------------------

        try:

            pm4py.write_pnml(
                net,
                initial_marking,
                final_marking,
                output_path,
            )

        except Exception as exc:

            print(
                "Warning: could not save Heuristics Miner "
                f"model: {exc}"
            )

        # -----------------------------------------------------
        # Statistics
        # -----------------------------------------------------

        print(
            "Heuristics Miner completed."
        )

        print(
            f"  Places      : {len(net.places)}"
        )

        print(
            f"  Transitions : {len(net.transitions)}"
        )

        print(
            f"  Arcs        : {len(net.arcs)}"
        )

        print(
            f"  Time        : {elapsed:.2f}s"
        )

        return {

            "name":
                "Heuristics Miner",

            "algorithm":
                "heuristics",

            "type":
                "petri_net",

            "net":
                net,

            "initial_marking":
                initial_marking,

            "final_marking":
                final_marking,

            "time":
                elapsed,

            "status":
                "SUCCESS",

            "output_path":
                output_path,
        }

    # =========================================================
    # SPLIT MINER
    # =========================================================

    def _discover_split_miner(
        self,
    ) -> Dict[str, Any]:

        print(
            "\nRunning Split Miner..."
        )

        start = time.time()

        bpmn = pm4py.discover_bpmn_split_miner(
            self.log,
            case_id_key="case:concept:name",
            activity_key="concept:name",
            timestamp_key="time:timestamp",
        )

        elapsed = time.time() - start

        output_path = os.path.join(
            self.output_dir,
            "split_miner_model.bpmn",
        )

        # -----------------------------------------------------
        # Save BPMN model
        # -----------------------------------------------------

        try:

            pm4py.write_bpmn(
                bpmn,
                output_path,
            )

        except Exception as exc:

            print(
                "Warning: could not save Split Miner "
                f"model: {exc}"
            )

        # -----------------------------------------------------
        # Statistics
        # -----------------------------------------------------

        print(
            "Split Miner completed."
        )

        print(
            f"  Time        : {elapsed:.2f}s"
        )

        return {

            "name":
                "Split Miner",

            "algorithm":
                "split_miner",

            "type":
                "bpmn",

            "bpmn":
                bpmn,

            "time":
                elapsed,

            "status":
                "SUCCESS",

            "output_path":
                output_path,
        }

    # =========================================================
    # SAFE DISCOVERY WRAPPER
    # =========================================================

    def _run_algorithm(
        self,
        name: str,
        function,
    ) -> Dict[str, Any]:
        """
        Execute a discovery algorithm safely.

        If one algorithm fails, the remaining algorithms
        continue running.
        """

        try:

            result = function()

            return result

        except Exception as exc:

            print(
                "\nFAILED"
            )

            print(
                f"Algorithm : {name}"
            )

            print(
                "Error     : "
                f"{type(exc).__name__}: {exc}"
            )

            traceback.print_exc()

            return {

                "name":
                    name,

                "algorithm":
                    name.lower().replace(
                        " ",
                        "_",
                    ),

                "status":
                    "FAILED",

                "error":
                    str(exc),

                "error_type":
                    type(exc).__name__,

                "time":
                    0.0,
            }

    # =========================================================
    # DISCOVER ALL
    # =========================================================

    def discover_all(
        self,
        strategies=None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run all supported discovery algorithms.

        Parameters
        ----------
        strategies : optional list

            Optional list of algorithms to execute.

            Example:

                [
                    "Inductive Miner",
                    "Heuristics Miner"
                ]

            If None, all five supported algorithms are run.

        Returns
        -------
        dict

            Dictionary keyed by algorithm name.

        Notes
        -----
        Failed algorithms are not included in the returned
        dictionary. Their failures are printed to the console.

        ILP Miner is intentionally not part of this portfolio.
        """

        self._print_header(
            "APMOA - PROCESS DISCOVERY AGENT"
        )

        # =====================================================
        # DETERMINE PORTFOLIO
        # =====================================================

        if strategies is None:

            strategies = list(
                self.algorithms
            )

        # -----------------------------------------------------
        # Normalize user-provided strategy names
        # -----------------------------------------------------

        normalized_strategies = []

        for strategy in strategies:

            if not isinstance(
                strategy,
                str,
            ):
                continue

            strategy_clean = (
                strategy.strip()
                .lower()
            )

            normalized_strategies.append(
                strategy_clean
            )

        # -----------------------------------------------------
        # Supported mapping
        # -----------------------------------------------------

        strategy_mapping = {

            "inductive":
                "Inductive Miner",

            "inductive_miner":
                "Inductive Miner",

            "alpha":
                "Alpha Miner",

            "alpha_miner":
                "Alpha Miner",

            "alpha_plus":
                "Alpha+ Miner",

            "alpha+":
                "Alpha+ Miner",

            "alpha_plus_miner":
                "Alpha+ Miner",

            "heuristics":
                "Heuristics Miner",

            "heuristics_miner":
                "Heuristics Miner",

            "split":
                "Split Miner",

            "split_miner":
                "Split Miner",
        }

        # -----------------------------------------------------
        # Convert canonical names if supplied
        # -----------------------------------------------------

        canonical_strategies = []

        for strategy in normalized_strategies:

            if strategy in strategy_mapping:

                canonical_strategies.append(
                    strategy_mapping[strategy]
                )

                continue

            # Already canonical name
            for algorithm in self.algorithms:

                if (
                    strategy
                    == algorithm.lower()
                ):

                    canonical_strategies.append(
                        algorithm
                    )

                    break

        # -----------------------------------------------------
        # Remove duplicates while preserving order
        # -----------------------------------------------------

        unique_strategies = []

        for strategy in canonical_strategies:

            if strategy not in unique_strategies:

                unique_strategies.append(
                    strategy
                )

        # -----------------------------------------------------
        # Fallback
        # -----------------------------------------------------

        if not unique_strategies:

            unique_strategies = list(
                self.algorithms
            )

        # =====================================================
        # DISPLAY PORTFOLIO
        # =====================================================

        print(
            "\nDiscovery portfolio:"
        )

        for index, strategy in enumerate(
            unique_strategies,
            start=1,
        ):

            print(
                f"  {index}. {strategy}"
            )

        print(
            "\nTotal algorithms requested: "
            f"{len(unique_strategies)}"
        )

        # =====================================================
        # DISCOVERY FUNCTION MAPPING
        # =====================================================

        discovery_functions = {

            "Inductive Miner":
                self._discover_inductive,

            "Alpha Miner":
                self._discover_alpha,

            "Alpha+ Miner":
                self._discover_alpha_plus,

            "Heuristics Miner":
                self._discover_heuristics,

            "Split Miner":
                self._discover_split_miner,
        }

        # =====================================================
        # RESULT CONTAINER
        # =====================================================

        discovered_models = {}

        failed_models = []

        # =====================================================
        # RUN EACH ALGORITHM
        # =====================================================

        for strategy in unique_strategies:

            print(
                "\n" + "-" * 60
            )

            print(
                f"Running {strategy}..."
            )

            function = discovery_functions.get(
                strategy
            )

            # -------------------------------------------------
            # Unknown algorithm
            # -------------------------------------------------

            if function is None:

                print(
                    f"Unsupported discovery algorithm: "
                    f"{strategy}"
                )

                failed_models.append(
                    {

                        "name":
                            strategy,

                        "algorithm":
                            strategy.lower().replace(
                                " ",
                                "_",
                            ),

                        "status":
                            "FAILED",

                        "error":
                            "Unsupported discovery algorithm.",

                        "error_type":
                            "ValueError",

                        "time":
                            0.0,
                    }
                )

                continue

            # -------------------------------------------------
            # Execute safely
            # -------------------------------------------------

            result = self._run_algorithm(
                strategy,
                function,
            )

            # -------------------------------------------------
            # Successful result
            # -------------------------------------------------

            if (
                result.get("status")
                == "SUCCESS"
            ):

                discovered_models[
                    strategy
                ] = result

                print(
                    f"{strategy} completed successfully."
                )

            # -------------------------------------------------
            # Failed result
            # -------------------------------------------------

            else:

                failed_models.append(
                    result
                )

        # =====================================================
        # SUMMARY
        # =====================================================

        self._print_header(
            "PROCESS DISCOVERY SUMMARY"
        )

        print(
            f"\n{'Algorithm':<25}"
            f"{'Status':<12}"
            f"{'Time (s)':<12}"
        )

        print(
            "-" * 60
        )

        # -----------------------------------------------------
        # Successful models
        # -----------------------------------------------------

        for algorithm in unique_strategies:

            if algorithm in discovered_models:

                result = discovered_models[
                    algorithm
                ]

                print(
                    f"{algorithm:<25}"
                    f"{result.get('status', 'UNKNOWN'):<12}"
                    f"{result.get('time', 0.0):<12.2f}"
                )

        # -----------------------------------------------------
        # Failed models
        # -----------------------------------------------------

        for result in failed_models:

            print(
                f"{result.get('name', 'Unknown'):<25}"
                f"{result.get('status', 'FAILED'):<12}"
                f"{result.get('time', 0.0):<12.2f}"
            )

        # =====================================================
        # COUNTS
        # =====================================================

        successful_count = len(
            discovered_models
        )

        failed_count = len(
            failed_models
        )

        total_count = (
            successful_count
            + failed_count
        )

        print(
            "\n" + "-" * 60
        )

        print(
            f"Successful models : "
            f"{successful_count}"
        )

        print(
            f"Failed models     : "
            f"{failed_count}"
        )

        print(
            f"Total algorithms  : "
            f"{total_count}"
        )

        # =====================================================
        # AVAILABLE MODELS
        # =====================================================

        print(
            "\nModels available for evaluation: "
            f"{successful_count}"
        )

        for model_name in discovered_models:

            print(
                f"  ✓ {model_name}"
            )

        if failed_models:

            print(
                "\nFailed discovery algorithms:"
            )

            for result in failed_models:

                print(
                    f"  ✗ "
                    f"{result.get('name', 'Unknown')}"
                )

        # =====================================================
        # FINAL RESULT
        # =====================================================

        print(
            "\n" + "=" * 60
        )

        return discovered_models


# =============================================================
# STANDALONE TEST
# =============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "APMOA - PROCESS DISCOVERY AGENT TEST"
    )

    print(
        "=" * 70
    )

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    log_path = (
        "data/raw/Travel/PermitLog.xes.gz"
    )

    print(
        f"\nLoading event log: {log_path}"
    )

    if not os.path.exists(
        log_path
    ):

        raise FileNotFoundError(
            f"Event log not found: {log_path}"
        )

    # ---------------------------------------------------------
    # Load event log
    # ---------------------------------------------------------

    start = time.time()

    log = pm4py.read_xes(
        log_path
    )

    elapsed = time.time() - start

    print(
        f"Loaded event log in "
        f"{elapsed:.2f}s"
    )

    try:

        print(
            f"Events : {len(log):,}"
        )

        print(
            f"Traces : {len(log):,}"
        )

    except Exception:

        pass

    # ---------------------------------------------------------
    # Create agent
    # ---------------------------------------------------------

    agent = ProcessDiscoveryAgent(
        log
    )

    # ---------------------------------------------------------
    # Discover all models
    # ---------------------------------------------------------

    models = agent.discover_all()

    # =========================================================
    # VALIDATION
    # =========================================================

    expected_models = [
        "Inductive Miner",
        "Alpha Miner",
        "Alpha+ Miner",
        "Heuristics Miner",
        "Split Miner",
    ]

    assert isinstance(
        models,
        dict,
    ), (
        "discover_all() must return "
        "a dictionary."
    )

    # ---------------------------------------------------------
    # Check expected algorithms
    # ---------------------------------------------------------

    missing_models = [

        name

        for name in expected_models

        if name not in models
    ]

    assert not missing_models, (
        "Missing discovery algorithms: "
        f"{missing_models}"
    )

    # ---------------------------------------------------------
    # Check model objects
    # ---------------------------------------------------------

    invalid_models = []

    for name in expected_models:

        model = models.get(
            name
        )

        if not isinstance(
            model,
            dict,
        ):

            invalid_models.append(
                name
            )

            continue

        if model.get(
            "status"
        ) != "SUCCESS":

            invalid_models.append(
                name
            )

    assert not invalid_models, (
        "Invalid model results: "
        f"{invalid_models}"
    )

    # =========================================================
    # DISPLAY RESULTS
    # =========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "DISCOVERY RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nModels successfully discovered: "
        f"{len(models)}"
    )

    for name in expected_models:

        result = models[
            name
        ]

        print(
            f"\n✓ {name}"
        )

        print(
            f"    Algorithm : "
            f"{result.get('algorithm')}"
        )

        print(
            f"    Type      : "
            f"{result.get('type')}"
        )

        print(
            f"    Time      : "
            f"{result.get('time', 0.0):.2f}s"
        )

        print(
            f"    Status    : "
            f"{result.get('status')}"
        )

        # -----------------------------------------------------
        # Petri net information
        # -----------------------------------------------------

        if (
            result.get("type")
            == "petri_net"
        ):

            net = result.get(
                "net"
            )

            if net is not None:

                print(
                    f"    Places    : "
                    f"{len(net.places)}"
                )

                print(
                    f"    Transitions: "
                    f"{len(net.transitions)}"
                )

                print(
                    f"    Arcs      : "
                    f"{len(net.arcs)}"
                )

        # -----------------------------------------------------
        # BPMN information
        # -----------------------------------------------------

        elif (
            result.get("type")
            == "bpmn"
        ):

            bpmn = result.get(
                "bpmn"
            )

            if bpmn is not None:

                print(
                    f"    BPMN Type  : "
                    f"{type(bpmn).__name__}"
                )

    # =========================================================
    # FINAL SUMMARY
    # =========================================================

    print(
        "\n\n" + "=" * 70
    )

    print(
        "PROCESS DISCOVERY AGENT TEST SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"\nExpected models       : "
        f"{len(expected_models)}"
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
        "\n" + "-" * 70
    )

    for name in expected_models:

        print(
            f"✓ {name:<25} SUCCESS"
        )

    print(
        "=" * 70
    )

    print(
        "\nALL DISCOVERY AGENT TESTS PASSED"
    )

    print(
        "=" * 70
    )

    print(
        "\nThe Process Discovery Agent successfully "
        "supports all five configured algorithms:"
    )

    for name in expected_models:

        print(
            f"  ✓ {name}"
        )

    print(
        "\nThese five models are ready "
        "for the Evaluation Agent."
    )

    print(
        "=" * 70
    )


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":

    main()
