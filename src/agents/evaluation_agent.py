"""
APMOA - Process Model Evaluation Agent

Evaluates all process-discovery candidates produced by the
Process Discovery Agent.

Supported discovery algorithms:

    1. Inductive Miner
    2. Alpha Miner
    3. Alpha+ Miner
    4. Heuristics Miner
    5. Split Miner

All models are normalized to Petri nets before evaluation.

Quality dimensions:

    - Fitness
    - Precision
    - Generalization
    - Simplicity

The evaluator is compatible with the APMOA orchestrator:

    evaluator = ModelEvaluationAgent(log)
    results = evaluator.evaluate(models)

The returned evaluation results are a pandas DataFrame so that
ModelSelectionAgent can directly consume them.

The evaluator accepts:

    1. Canonical APMOA DataFrame
    2. Native PM4Py/XES DataFrame
    3. PM4Py EventLog
"""

from __future__ import annotations

import math
import time
import traceback
from typing import Any, Dict, List

import pandas as pd
import pm4py


class ModelEvaluationAgent:
    """
    APMOA Model Evaluation Agent.

    Parameters
    ----------
    log:
        Canonical APMOA pandas DataFrame,
        native PM4Py/XES DataFrame, or PM4Py EventLog.

    activity_key:
        Internal PM4Py activity attribute.

    timestamp_key:
        Internal PM4Py timestamp attribute.

    case_id_key:
        Internal PM4Py case identifier attribute.
    """

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        log=None,
        activity_key: str = "concept:name",
        timestamp_key: str = "time:timestamp",
        case_id_key: str = "case:concept:name",
    ):
        self.log = log

        self.activity_key = activity_key
        self.timestamp_key = timestamp_key
        self.case_id_key = case_id_key

    # =========================================================
    # EVENT LOG NORMALIZATION
    # =========================================================

    def _prepare_event_log(self, log):
        """
        Normalize the supplied event log for PM4Py evaluation.

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

        Returns
        -------
        pandas.DataFrame or pm4py.objects.log.obj.EventLog
            PM4Py-compatible event log.
        """

        if log is None:
            raise ValueError(
                "Cannot prepare event log: log is None."
            )

        # =====================================================
        # CASE 1 - PANDAS DATAFRAME
        # =====================================================

        if isinstance(log, pd.DataFrame):

            prepared_log = log.copy()

            print(
                "\nPreparing event log for PM4Py evaluation..."
            )

            print(
                f"Input columns: {list(prepared_log.columns)}"
            )

            # -------------------------------------------------
            # CANONICAL APMOA SCHEMA
            # -------------------------------------------------

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

                # Case ID
                prepared_log[
                    "case:concept:name"
                ] = (
                    prepared_log["case_id"]
                    .astype(str)
                )

                # Activity
                prepared_log[
                    "concept:name"
                ] = (
                    prepared_log["activity"]
                    .astype(str)
                )

                # Timestamp
                prepared_log[
                    "time:timestamp"
                ] = pd.to_datetime(
                    prepared_log["timestamp"],
                    errors="coerce",
                )

                # Resource
                if "resource" in prepared_log.columns:

                    prepared_log[
                        "org:resource"
                    ] = (
                        prepared_log["resource"]
                        .fillna("UNKNOWN")
                        .astype(str)
                    )

                else:

                    prepared_log[
                        "org:resource"
                    ] = "UNKNOWN"

            # -------------------------------------------------
            # NATIVE PM4PY / XES SCHEMA
            # -------------------------------------------------

            elif {
                "case:concept:name",
                "concept:name",
                "time:timestamp",
            }.issubset(
                prepared_log.columns
            ):

                print(
                    "Detected native PM4Py/XES "
                    "event-log schema."
                )

                prepared_log[
                    "case:concept:name"
                ] = (
                    prepared_log[
                        "case:concept:name"
                    ].astype(str)
                )

                prepared_log[
                    "concept:name"
                ] = (
                    prepared_log[
                        "concept:name"
                    ].astype(str)
                )

                prepared_log[
                    "time:timestamp"
                ] = pd.to_datetime(
                    prepared_log[
                        "time:timestamp"
                    ],
                    errors="coerce",
                )

                if "org:resource" not in (
                    prepared_log.columns
                ):

                    prepared_log[
                        "org:resource"
                    ] = "UNKNOWN"

                else:

                    prepared_log[
                        "org:resource"
                    ] = (
                        prepared_log[
                            "org:resource"
                        ]
                        .fillna("UNKNOWN")
                        .astype(str)
                    )

            # -------------------------------------------------
            # UNKNOWN SCHEMA
            # -------------------------------------------------

            else:

                raise ValueError(
                    "\nUnable to identify event-log schema "
                    "for evaluation.\n\n"

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

            # -------------------------------------------------
            # TIMESTAMP VALIDATION
            # -------------------------------------------------

            invalid_timestamps = (
                prepared_log[
                    "time:timestamp"
                ]
                .isna()
                .sum()
            )

            if invalid_timestamps > 0:

                raise ValueError(
                    f"Event log contains "
                    f"{invalid_timestamps} invalid "
                    f"timestamp value(s)."
                )

            # -------------------------------------------------
            # REQUIRED VALUE VALIDATION
            # -------------------------------------------------

            missing_case_ids = (
                prepared_log[
                    "case:concept:name"
                ]
                .isna()
                .sum()
            )

            missing_activities = (
                prepared_log[
                    "concept:name"
                ]
                .isna()
                .sum()
            )

            if missing_case_ids > 0:

                raise ValueError(
                    f"Event log contains "
                    f"{missing_case_ids} missing case ID "
                    f"value(s)."
                )

            if missing_activities > 0:

                raise ValueError(
                    f"Event log contains "
                    f"{missing_activities} missing activity "
                    f"value(s)."
                )

            # -------------------------------------------------
            # STABLE EVENT ORDERING
            # -------------------------------------------------

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

            # -------------------------------------------------
            # DIAGNOSTICS
            # -------------------------------------------------

            print(
                "\nEvaluation event log prepared."
            )

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

            print(
                "\nPM4Py evaluation columns:"
            )

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
        # CASE 2 - PM4PY EVENTLOG
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
            "Unsupported event-log type for evaluation: "
            f"{type(log)}"
        )

    # =========================================================
    # SAFE FLOAT
    # =========================================================

    @staticmethod
    def _safe_float(
        value,
        default: float = 0.0,
    ) -> float:
        """
        Safely convert a value to float.
        """

        try:

            value = float(value)

            if math.isnan(value):
                return default

            if math.isinf(value):
                return default

            return value

        except (
            TypeError,
            ValueError,
        ):

            return default

    # =========================================================
    # MODEL NAME
    # =========================================================

    @staticmethod
    def _get_model_name(
        model_data,
        fallback: str = "Unknown Model",
    ) -> str:
        """
        Resolve the model name from a discovery result.
        """

        if not isinstance(
            model_data,
            dict,
        ):
            return fallback

        for key in (
            "name",
            "model_name",
            "algorithm",
        ):

            value = model_data.get(
                key
            )

            if value is not None:

                value = str(
                    value
                ).strip()

                if value:
                    return value

        return fallback

    # =========================================================
    # MODEL TYPE
    # =========================================================

    def _get_model_type(
        self,
        model_data: Dict[str, Any],
    ) -> str:
        """
        Determine whether a discovered model is a:

            PETRI_NET
            BPMN
        """

        if not isinstance(
            model_data,
            dict,
        ):

            raise TypeError(
                "Model candidate must be a dictionary."
            )

        # -----------------------------------------------------
        # BPMN
        # -----------------------------------------------------

        if model_data.get(
            "bpmn"
        ) is not None:

            return "BPMN"

        # -----------------------------------------------------
        # PETRI NET
        # -----------------------------------------------------

        if model_data.get(
            "net"
        ) is not None:

            return "PETRI_NET"

        # -----------------------------------------------------
        # GENERIC MODEL KEY
        # -----------------------------------------------------

        model = model_data.get(
            "model"
        )

        if model is not None:

            class_name = (
                type(model)
                .__name__
                .lower()
            )

            if (
                "bpmn"
                in class_name
            ):

                return "BPMN"

            if (
                "petrinet"
                in class_name
                or "petri_net"
                in class_name
            ):

                return "PETRI_NET"

        raise ValueError(
            "Unable to determine model type. "
            "Expected 'net' or 'bpmn'."
        )

    # =========================================================
    # CONVERT TO PETRI NET
    # =========================================================

    def _convert_to_petri_net(
        self,
        model_data: Dict[str, Any],
    ):
        """
        Normalize every supported model into:

            net
            initial_marking
            final_marking

        Split Miner produces BPMN, so BPMN is converted to a
        Petri net before conformance evaluation.
        """

        model_type = (
            self._get_model_type(
                model_data
            )
        )

        # =====================================================
        # PETRI NET
        # =====================================================

        if model_type == "PETRI_NET":

            net = model_data.get(
                "net"
            )

            initial_marking = (
                model_data.get(
                    "initial_marking"
                )
            )

            final_marking = (
                model_data.get(
                    "final_marking"
                )
            )

            if (
                net is None
                or initial_marking is None
                or final_marking is None
            ):

                raise ValueError(
                    "Petri-net model is missing "
                    "net/initial_marking/final_marking."
                )

            return (
                net,
                initial_marking,
                final_marking,
            )

        # =====================================================
        # BPMN
        # =====================================================

        if model_type == "BPMN":

            bpmn = model_data.get(
                "bpmn"
            )

            if bpmn is None:

                bpmn = model_data.get(
                    "model"
                )

            if bpmn is None:

                raise ValueError(
                    "BPMN model object is missing."
                )

            print(
                "  Converting BPMN model to Petri net..."
            )

            (
                net,
                initial_marking,
                final_marking,
            ) = pm4py.convert_to_petri_net(
                bpmn
            )

            print(
                "  BPMN conversion completed."
            )

            return (
                net,
                initial_marking,
                final_marking,
            )

        raise ValueError(
            f"Unsupported model type: {model_type}"
        )

    # =========================================================
    # FITNESS
    # =========================================================

    def _calculate_fitness(
        self,
        log,
        net,
        initial_marking,
        final_marking,
    ) -> Dict[str, float]:
        """
        Calculate token-based replay fitness.
        """

        print(
            "  Calculating fitness..."
        )

        result = (
            pm4py.fitness_token_based_replay(
                log,
                net,
                initial_marking,
                final_marking,
                activity_key=self.activity_key,
                timestamp_key=self.timestamp_key,
            )
        )

        if isinstance(
            result,
            dict,
        ):

            log_fitness = result.get(
                "log_fitness"
            )

            if log_fitness is None:

                log_fitness = result.get(
                    "average_trace_fitness"
                )

            fitness = (
                self._safe_float(
                    log_fitness
                )
            )

            return {
                "score": fitness,

                "log_fitness": fitness,

                "average_trace_fitness":
                    self._safe_float(
                        result.get(
                            "average_trace_fitness"
                        )
                    ),

                "perc_fit_traces":
                    self._safe_float(
                        result.get(
                            "perc_fit_traces"
                        )
                    ),
            }

        fitness = (
            self._safe_float(
                result
            )
        )

        return {
            "score": fitness,

            "log_fitness": fitness,

            "average_trace_fitness": fitness,

            "perc_fit_traces": 0.0,
        }

    # =========================================================
    # PRECISION
    # =========================================================

    def _calculate_precision(
        self,
        log,
        net,
        initial_marking,
        final_marking,
    ) -> float:
        """
        Calculate token-based replay precision.
        """

        print(
            "  Calculating precision..."
        )

        precision = (
            pm4py.precision_token_based_replay(
                log,
                net,
                initial_marking,
                final_marking,
                activity_key=self.activity_key,
                timestamp_key=self.timestamp_key,
            )
        )

        return self._safe_float(
            precision
        )

    # =========================================================
    # GENERALIZATION
    # =========================================================

    def _calculate_generalization(
        self,
        log,
        net,
        initial_marking,
        final_marking,
    ) -> float:
        """
        Calculate token-based generalization.
        """

        print(
            "  Calculating generalization..."
        )

        generalization = (
            pm4py.generalization_tbr(
                log,
                net,
                initial_marking,
                final_marking,
                activity_key=self.activity_key,
                timestamp_key=self.timestamp_key,
                case_id_key=self.case_id_key,
            )
        )

        return self._safe_float(
            generalization
        )

    # =========================================================
    # SIMPLICITY
    # =========================================================

    def _calculate_simplicity(
        self,
        net,
    ) -> float:
        """
        Calculate structural simplicity.
        """

        print(
            "  Calculating simplicity..."
        )

        from pm4py.algo.evaluation.simplicity import (
            algorithm as simplicity_evaluator,
        )

        simplicity = (
            simplicity_evaluator.apply(
                net
            )
        )

        return self._safe_float(
            simplicity
        )

    # =========================================================
    # STRUCTURAL METRICS
    # =========================================================

    @staticmethod
    def _structural_metrics(
        net,
    ) -> Dict[str, int]:
        """
        Return basic Petri-net structural metrics.
        """

        return {
            "places":
                len(
                    net.places
                ),

            "transitions":
                len(
                    net.transitions
                ),

            "arcs":
                len(
                    net.arcs
                ),
        }

    # =========================================================
    # EVALUATE SINGLE MODEL
    # =========================================================

    def _evaluate_single_model(
        self,
        log,
        model_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Evaluate one discovered process model.
        """

        model_name = (
            self._get_model_name(
                model_data
            )
        )

        print(
            f"\nEvaluating model: {model_name}"
        )

        start_time = time.time()

        # -----------------------------------------------------
        # MODEL TYPE
        # -----------------------------------------------------

        model_type = (
            self._get_model_type(
                model_data
            )
        )

        print(
            f"  Model type: {model_type}"
        )

        # -----------------------------------------------------
        # NORMALIZE MODEL
        # -----------------------------------------------------

        (
            net,
            initial_marking,
            final_marking,
        ) = self._convert_to_petri_net(
            model_data
        )

        # -----------------------------------------------------
        # STRUCTURE
        # -----------------------------------------------------

        structure = (
            self._structural_metrics(
                net
            )
        )

        # -----------------------------------------------------
        # FITNESS
        # -----------------------------------------------------

        fitness_result = (
            self._calculate_fitness(
                log,
                net,
                initial_marking,
                final_marking,
            )
        )

        fitness = (
            fitness_result[
                "score"
            ]
        )

        # -----------------------------------------------------
        # PRECISION
        # -----------------------------------------------------

        precision = (
            self._calculate_precision(
                log,
                net,
                initial_marking,
                final_marking,
            )
        )

        # -----------------------------------------------------
        # GENERALIZATION
        # -----------------------------------------------------

        generalization = (
            self._calculate_generalization(
                log,
                net,
                initial_marking,
                final_marking,
            )
        )

        # -----------------------------------------------------
        # SIMPLICITY
        # -----------------------------------------------------

        simplicity = (
            self._calculate_simplicity(
                net
            )
        )

        elapsed = (
            time.time()
            - start_time
        )

        # -----------------------------------------------------
        # RESULT
        # -----------------------------------------------------

        result = {
            "model":
                model_name,

            "model_type":
                model_type,

            "fitness":
                fitness,

            "precision":
                precision,

            "generalization":
                generalization,

            "simplicity":
                simplicity,

            "places":
                structure[
                    "places"
                ],

            "transitions":
                structure[
                    "transitions"
                ],

            "arcs":
                structure[
                    "arcs"
                ],

            "evaluation_time":
                elapsed,

            "status":
                "SUCCESS",
        }

        print(
            f"  Fitness        : "
            f"{fitness:.4f}"
        )

        print(
            f"  Precision      : "
            f"{precision:.4f}"
        )

        print(
            f"  Generalization : "
            f"{generalization:.4f}"
        )

        print(
            f"  Simplicity     : "
            f"{simplicity:.4f}"
        )

        print(
            f"  Places         : "
            f"{structure['places']}"
        )

        print(
            f"  Transitions    : "
            f"{structure['transitions']}"
        )

        print(
            f"  Arcs           : "
            f"{structure['arcs']}"
        )

        print(
            f"  Time           : "
            f"{elapsed:.2f}s"
        )

        return result

    # =========================================================
    # NORMALIZE MODELS
    # =========================================================

    def _normalize_models(
        self,
        models,
    ) -> List[Dict[str, Any]]:
        """
        Normalize discovery output into a list of dictionaries.

        Supported:

            list
            tuple
            dictionary
        """

        if isinstance(
            models,
            dict,
        ):

            model_list = []

            # -------------------------------------------------
            # Case:
            #
            # {
            #     "Inductive Miner": {...}
            # }
            # -------------------------------------------------

            for name, model_data in models.items():

                if not isinstance(
                    model_data,
                    dict,
                ):

                    raise TypeError(
                        f"Model '{name}' must be "
                        f"a dictionary."
                    )

                candidate = dict(
                    model_data
                )

                candidate.setdefault(
                    "name",
                    name,
                )

                model_list.append(
                    candidate
                )

            return model_list

        if isinstance(
            models,
            (
                list,
                tuple,
            ),
        ):

            model_list = []

            for model_data in models:

                if not isinstance(
                    model_data,
                    dict,
                ):

                    raise TypeError(
                        "Each discovered model "
                        "must be a dictionary."
                    )

                model_list.append(
                    dict(
                        model_data
                    )
                )

            return model_list

        raise TypeError(
            "models must be a dictionary, "
            "list, or tuple."
        )

    # =========================================================
    # EVALUATE ALL MODELS
    # =========================================================

    def evaluate(
        self,
        models,
        log=None,
    ) -> pd.DataFrame:
        """
        Evaluate all discovered models.

        Current APMOA interface:

            evaluator = ModelEvaluationAgent(log)
            results = evaluator.evaluate(models)

        Backward-compatible interface:

            evaluator.evaluate(
                models,
                log=log
            )

        Parameters
        ----------
        models:
            Models returned by ProcessDiscoveryAgent.

        log:
            Optional event log. If omitted, the log supplied to
            the constructor is used.

        Returns
        -------
        pandas.DataFrame
            One row per discovery algorithm.
        """

        # =====================================================
        # RESOLVE LOG
        # =====================================================

        evaluation_log = (
            log
            if log is not None
            else self.log
        )

        if evaluation_log is None:

            raise ValueError(
                "No event log supplied. "
                "Initialize ModelEvaluationAgent(log) "
                "or pass log to evaluate()."
            )

        # =====================================================
        # NORMALIZE LOG
        # =====================================================

        evaluation_log = (
            self._prepare_event_log(
                evaluation_log
            )
        )

        # =====================================================
        # HEADER
        # =====================================================

        print(
            "\n" + "=" * 60
        )

        print(
            "APMOA - PROCESS MODEL EVALUATION AGENT"
        )

        print(
            "=" * 60
        )

        print(
            "\nEvaluating discovered process models..."
        )

        # =====================================================
        # NORMALIZE MODELS
        # =====================================================

        model_list = (
            self._normalize_models(
                models
            )
        )

        if not model_list:

            raise ValueError(
                "No discovered models were supplied "
                "for evaluation."
            )

        # =====================================================
        # EVALUATE MODELS
        # =====================================================

        results = []

        for model_data in model_list:

            model_name = (
                self._get_model_name(
                    model_data
                )
            )

            try:

                result = (
                    self._evaluate_single_model(
                        evaluation_log,
                        model_data,
                    )
                )

                results.append(
                    result
                )

            except Exception as exc:

                print(
                    f"\nFAILED evaluating model: "
                    f"{model_name}"
                )

                print(
                    f"Error: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                traceback.print_exc()

                results.append(
                    {
                        "model":
                            model_name,

                        "model_type":
                            model_data.get(
                                "model_type",
                                "UNKNOWN",
                            ),

                        "fitness":
                            0.0,

                        "precision":
                            0.0,

                        "generalization":
                            0.0,

                        "simplicity":
                            0.0,

                        "places":
                            0,

                        "transitions":
                            0,

                        "arcs":
                            0,

                        "evaluation_time":
                            0.0,

                        "status":
                            "FAILED",

                        "error":
                            str(exc),

                        "error_type":
                            type(exc).__name__,
                    }
                )

        # =====================================================
        # DATAFRAME
        # =====================================================

        evaluation_dataframe = (
            pd.DataFrame(
                results
            )
        )

        # =====================================================
        # ENSURE REQUIRED COLUMNS
        # =====================================================

        required_columns = [
            "model",
            "model_type",
            "fitness",
            "precision",
            "generalization",
            "simplicity",
            "places",
            "transitions",
            "arcs",
            "evaluation_time",
            "status",
        ]

        for column in required_columns:

            if column not in (
                evaluation_dataframe.columns
            ):

                evaluation_dataframe[
                    column
                ] = 0.0

        # =====================================================
        # NUMERIC CLEANUP
        # =====================================================

        numeric_columns = [
            "fitness",
            "precision",
            "generalization",
            "simplicity",
            "places",
            "transitions",
            "arcs",
            "evaluation_time",
        ]

        for column in numeric_columns:

            evaluation_dataframe[
                column
            ] = pd.to_numeric(
                evaluation_dataframe[
                    column
                ],
                errors="coerce",
            ).fillna(
                0.0
            )

        # =====================================================
        # SUMMARY
        # =====================================================

        self._print_summary(
            evaluation_dataframe
        )

        # =====================================================
        # RETURN
        # =====================================================

        return evaluation_dataframe

    # =========================================================
    # SUMMARY
    # =========================================================

    def _print_summary(
        self,
        evaluations: pd.DataFrame,
    ):
        """
        Print evaluation summary.
        """

        print(
            "\n" + "-" * 90
        )

        print(
            "MODEL EVALUATION RESULTS"
        )

        print(
            "-" * 90
        )

        print(
            f"{'Model':<24}"
            f"{'Fitness':>10}"
            f"{'Precision':>12}"
            f"{'General.':>12}"
            f"{'Simplicity':>12}"
            f"{'Status':>12}"
        )

        print(
            "-" * 90
        )

        for _, row in evaluations.iterrows():

            model = str(
                row.get(
                    "model",
                    "Unknown",
                )
            )

            fitness = (
                self._safe_float(
                    row.get(
                        "fitness",
                        0.0,
                    )
                )
            )

            precision = (
                self._safe_float(
                    row.get(
                        "precision",
                        0.0,
                    )
                )
            )

            generalization = (
                self._safe_float(
                    row.get(
                        "generalization",
                        0.0,
                    )
                )
            )

            simplicity = (
                self._safe_float(
                    row.get(
                        "simplicity",
                        0.0,
                    )
                )
            )

            status = str(
                row.get(
                    "status",
                    "UNKNOWN",
                )
            )

            print(
                f"{model:<24}"
                f"{fitness:>10.4f}"
                f"{precision:>12.4f}"
                f"{generalization:>12.4f}"
                f"{simplicity:>12.4f}"
                f"{status:>12}"
            )

        print(
            "-" * 90
        )

        successful = int(
            (
                evaluations[
                    "status"
                ]
                == "SUCCESS"
            ).sum()
        )

        failed = (
            len(evaluations)
            - successful
        )

        print(
            f"Successful models : "
            f"{successful}"
        )

        print(
            f"Failed models     : "
            f"{failed}"
        )

        print(
            f"Total models      : "
            f"{len(evaluations)}"
        )

        print(
            "-" * 90
        )

        if failed == 0:

            print(
                "All discovered models "
                "were evaluated successfully."
            )

        else:

            print(
                "Some discovered models "
                "could not be evaluated."
            )

        print(
            "-" * 90
        )


# =============================================================
# BACKWARD COMPATIBILITY
# =============================================================

EvaluationAgent = ModelEvaluationAgent

ProcessModelEvaluationAgent = (
    ModelEvaluationAgent
)


# =============================================================
# STANDALONE TEST
# =============================================================

def main():

    print(
        "\n" + "=" * 60
    )

    print(
        "APMOA - EVALUATION AGENT"
    )

    print(
        "=" * 60
    )

    print(
        "\nModelEvaluationAgent initialized."
    )

    print(
        "\nSupported discovery algorithms:"
    )

    print(
        "  1. Inductive Miner"
    )

    print(
        "  2. Alpha Miner"
    )

    print(
        "  3. Alpha+ Miner"
    )

    print(
        "  4. Heuristics Miner"
    )

    print(
        "  5. Split Miner"
    )

    print(
        "\nExpected interface:"
    )

    print(
        "  agent = ModelEvaluationAgent(log)"
    )

    print(
        "  results = agent.evaluate(models)"
    )

    print(
        "\nReturned result type:"
    )

    print(
        "  pandas.DataFrame"
    )

    print(
        "\n" + "=" * 60
    )


if __name__ == "__main__":
    main()