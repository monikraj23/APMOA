"""
APMOA - MULTI-DATASET EXPERIMENT TEST

Purpose
-------
Run the APMOA pipeline across multiple event-log datasets and
collect structured experimental results.

Pipeline
--------
Dataset
    ↓
Dataset Loading
    ↓
Data Validation
    ↓
Data Analysis
    ↓
Process Discovery
    ↓
Model Evaluation
    ↓
Model Selection
    ↓
RAG Knowledge Base
    ↓
Groq Reasoning
    ↓
Autonomous Adaptation
    ↓
Bottleneck / Deviation Analysis
    ↓
Optimization
    ↓
Experiment Results

Important
---------
This file is an experiment/test layer.

Existing APMOA agents are NOT modified here.

Each dataset is processed independently. If one dataset fails,
the experiment continues with the remaining datasets.

Results are saved under:

    results/
        experiments/
            multi_dataset/
                dataset_results.csv
                model_results.csv
                adaptation_results.csv
                optimization_results.csv
                experiment_summary.json
"""

from pathlib import Path
import sys
import json
import time
import traceback
from datetime import datetime

import pandas as pd


# =====================================================================
# PROJECT ROOT
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =====================================================================
# IMPORTS
# =====================================================================

from src.mining.loader import EventLogLoader

from src.agents.data_analysis_agent import (
    DataAnalysisAgent
)

from src.agents.process_discovery_agent import (
    ProcessDiscoveryAgent
)

from src.agents.evaluation_agent import (
    ModelEvaluationAgent
)

from src.agents.bottleneck_analysis_agent import (
    BottleneckAnalysisAgent
)

from src.agents.optimization_agent import (
    OptimizationAgent
)

from src.agents.reasoning_agent import (
    ReasoningAgent
)

from src.agents.adaptation_agent import (
    AdaptationAgent
)

from src.rag.knowledge_base import (
    APMOAKnowledgeBase
)

from src.agents.model_selection_agent import (
    ModelSelectionAgent
)


# =====================================================================
# CONFIGURATION
# =====================================================================

KNOWLEDGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "knowledge"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "multi_dataset"
)

DISCOVERY_RESULTS_DIR = (
    RESULTS_DIR
    / "discovery"
)


# =====================================================================
# DATASET REGISTRY
# =====================================================================

DATASETS = [
    {
        "name": "BPI 2020 Travel Permit",
        "path": (
            PROJECT_ROOT
            / "data"
            / "raw"
            / "travel"
            / "PermitLog.xes.gz"
        ),
        "expected_events": 86581,
        "expected_cases": 7065,
        "expected_activities": 51,
        "expected_resources": 2,
    },

    {
        "name": "BPI 2012",
        "path": (
            PROJECT_ROOT
            / "data"
            / "raw"
            / "bpi2012"
            / "BPI_Challenge_2012.xes.gz"
        ),
        "expected_events": 262200,
        "expected_cases": 13087,
        "expected_activities": 24,
        "expected_resources": 69,
    },

    # ---------------------------------------------------------------
    # ADD DATASETS HERE
    # ---------------------------------------------------------------
    #
    # Example:
    #
    # {
    #     "name": "Another Dataset",
    #     "path": (
    #         PROJECT_ROOT
    #         / "data"
    #         / "raw"
    #         / "another_dataset"
    #         / "event_log.xes.gz"
    #     ),
    #     "expected_events": None,
    #     "expected_cases": None,
    #     "expected_activities": None,
    #     "expected_resources": None,
    # },
]


# =====================================================================
# EXPECTED DISCOVERY ALGORITHMS
# =====================================================================

EXPECTED_ALGORITHMS = [
    "Inductive Miner",
    "Alpha Miner",
    "Alpha+ Miner",
    "Heuristics Miner",
    "Split Miner",
]


# =====================================================================
# SELECTION WEIGHTS
# =====================================================================

SELECTION_WEIGHTS = {
    "fitness": 0.30,
    "precision": 0.30,
    "generalization": 0.20,
    "simplicity": 0.20,
}


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def print_header(title: str):
    """Print a major section header."""

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_subheader(title: str):
    """Print a subsection header."""

    print()
    print("-" * 80)
    print(title)
    print("-" * 80)


def check(condition: bool, message: str):
    """Raise an assertion error when a check fails."""

    if not condition:
        raise AssertionError(message)

    print(
        f"PASS: {message}"
    )


def safe_value(value):
    """
    Convert values into JSON/CSV-friendly representations.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): safe_value(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            safe_value(item)
            for item in value
        ]

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return str(value)


def make_json_safe(data):
    """Recursively make a result structure JSON serializable."""

    if isinstance(data, dict):

        return {
            str(key): make_json_safe(value)
            for key, value in data.items()
        }

    if isinstance(data, (list, tuple)):

        return [
            make_json_safe(value)
            for value in data
        ]

    if isinstance(data, pd.DataFrame):

        return make_json_safe(
            data.to_dict(
                orient="records"
            )
        )

    if isinstance(data, pd.Series):

        return make_json_safe(
            data.to_dict()
        )

    return safe_value(data)


def extract_selected_model(
    selection_result,
    evaluation_result,
):
    """
    Extract the selected model and selected evaluation from
    ModelSelectionAgent output.

    Also provides a weighted fallback.
    """

    selected_model = None
    selected_evaluation = None

    if isinstance(
        selection_result,
        dict
    ):

        selected_model = (
            selection_result.get(
                "selected_model"
            )
        )

        selected_evaluation = (
            selection_result.get(
                "selected_evaluation"
            )
        )

        if selected_model is None:

            selected_model = (
                selection_result.get(
                    "model"
                )
            )

        if selected_evaluation is None:

            selected_evaluation = (
                selection_result.get(
                    "evaluation"
                )
            )

    elif isinstance(
        selection_result,
        str
    ):

        selected_model = (
            selection_result
        )

    # ---------------------------------------------------------------
    # FALLBACK
    # ---------------------------------------------------------------

    if selected_model is None:

        print(
            "ModelSelectionAgent did not return "
            "a selected model."
        )

        print(
            "Using weighted selection fallback."
        )

        if isinstance(
            evaluation_result,
            pd.DataFrame
        ):

            df = evaluation_result.copy()

            required_metrics = [
                "fitness",
                "precision",
                "generalization",
                "simplicity",
            ]

            if all(
                metric in df.columns
                for metric in required_metrics
            ):

                df["_selection_score"] = (
                    SELECTION_WEIGHTS["fitness"]
                    * df["fitness"]
                    +
                    SELECTION_WEIGHTS["precision"]
                    * df["precision"]
                    +
                    SELECTION_WEIGHTS["generalization"]
                    * df["generalization"]
                    +
                    SELECTION_WEIGHTS["simplicity"]
                    * df["simplicity"]
                )

                best_row = (
                    df.sort_values(
                        "_selection_score",
                        ascending=False,
                    )
                    .iloc[0]
                )

                selected_model = (
                    best_row["model"]
                )

                selected_evaluation = (
                    best_row.to_dict()
                )

        elif isinstance(
            evaluation_result,
            dict
        ):

            candidates = []

            for (
                model_name,
                metrics,
            ) in evaluation_result.items():

                if not isinstance(
                    metrics,
                    dict
                ):
                    continue

                required_metrics = [
                    "fitness",
                    "precision",
                    "generalization",
                    "simplicity",
                ]

                if not all(
                    metric in metrics
                    for metric in required_metrics
                ):
                    continue

                score = (
                    SELECTION_WEIGHTS["fitness"]
                    * metrics["fitness"]
                    +
                    SELECTION_WEIGHTS["precision"]
                    * metrics["precision"]
                    +
                    SELECTION_WEIGHTS["generalization"]
                    * metrics["generalization"]
                    +
                    SELECTION_WEIGHTS["simplicity"]
                    * metrics["simplicity"]
                )

                candidates.append(
                    (
                        score,
                        model_name,
                        metrics,
                    )
                )

            if candidates:

                candidates.sort(
                    reverse=True
                )

                (
                    _,
                    selected_model,
                    selected_evaluation,
                ) = candidates[0]

    return (
        selected_model,
        selected_evaluation,
    )


def extract_model_results(
    dataset_name,
    evaluation_result,
):
    """
    Convert evaluation output into a flat list of records.
    """

    records = []

    if isinstance(
        evaluation_result,
        pd.DataFrame
    ):

        for _, row in evaluation_result.iterrows():

            record = {
                "dataset": dataset_name,
            }

            for column in row.index:

                record[column] = safe_value(
                    row[column]
                )

            records.append(record)

    elif isinstance(
        evaluation_result,
        dict
    ):

        for model_name, metrics in (
            evaluation_result.items()
        ):

            if not isinstance(
                metrics,
                dict
            ):
                continue

            record = {
                "dataset": dataset_name,
                "model": model_name,
            }

            for key, value in metrics.items():

                record[key] = safe_value(
                    value
                )

            records.append(record)

    return records


def extract_reasoning_fields(
    reasoning_result,
):
    """
    Extract commonly used reasoning fields without assuming
    that the ReasoningAgent always returns exactly the same
    dictionary structure.
    """

    if not isinstance(
        reasoning_result,
        dict
    ):

        return {
            "reasoning_action": None,
            "reasoning_strategy": None,
            "reasoning_confidence": None,
            "reasoning_reason": None,
        }

    return {
        "reasoning_action": (
            reasoning_result.get(
                "action",
                reasoning_result.get(
                    "recommended_action"
                ),
            )
        ),

        "reasoning_strategy": (
            reasoning_result.get(
                "strategy"
            )
        ),

        "reasoning_confidence": (
            reasoning_result.get(
                "confidence"
            )
        ),

        "reasoning_reason": (
            reasoning_result.get(
                "reason"
            )
        ),
    }


def extract_adaptation_fields(
    adaptation_result,
):
    """
    Extract adaptation information.
    """

    if not isinstance(
        adaptation_result,
        dict
    ):

        return {
            "adaptation_action": None,
            "adaptation_strategy": None,
            "adaptation_target": None,
            "adaptation_current_model": None,
            "adaptation_next_algorithm": None,
            "adaptation_reason": None,
            "adaptation_policy_triggered": None,
        }

    return {
        "adaptation_action": (
            adaptation_result.get(
                "action"
            )
        ),

        "adaptation_strategy": (
            adaptation_result.get(
                "strategy"
            )
        ),

        "adaptation_target": (
            adaptation_result.get(
                "target"
            )
        ),

        "adaptation_current_model": (
            adaptation_result.get(
                "current_model"
            )
        ),

        "adaptation_next_algorithm": (
            adaptation_result.get(
                "next_algorithm"
            )
        ),

        "adaptation_reason": (
            adaptation_result.get(
                "reason"
            )
        ),

        "adaptation_policy_triggered": (
            adaptation_result.get(
                "policy_triggered"
            )
        ),
    }


def summarize_process_intelligence(
    bottleneck_result,
):
    """
    Extract summary values from bottleneck analysis.
    """

    result = {
        "bottlenecks": None,
        "deviations": None,
        "rework_activities": None,
        "transitions": None,
    }

    if not isinstance(
        bottleneck_result,
        dict
    ):
        return result

    possible_keys = {
        "bottlenecks": [
            "bottlenecks",
            "potential_bottlenecks",
        ],

        "deviations": [
            "deviations",
            "potential_deviations",
        ],

        "rework_activities": [
            "rework_activities",
            "activities_with_rework",
        ],

        "transitions": [
            "transitions",
            "transitions_analyzed",
        ],
    }

    for output_key, keys in possible_keys.items():

        for key in keys:

            if key in bottleneck_result:

                value = bottleneck_result[key]

                if isinstance(
                    value,
                    (list, tuple, dict),
                ):

                    result[output_key] = len(value)

                else:

                    result[output_key] = safe_value(
                        value
                    )

                break

    return result


def summarize_optimization(
    optimization_result,
):
    """
    Extract optimization recommendation counts.
    """

    result = {
        "recommendations": None,
        "high_priority": None,
        "medium_priority": None,
        "low_priority": None,
    }

    if not isinstance(
        optimization_result,
        dict
    ):
        return result

    recommendations = (
        optimization_result.get(
            "recommendations"
        )
    )

    if isinstance(
        recommendations,
        (list, tuple)
    ):

        result["recommendations"] = (
            len(recommendations)
        )

        priorities = {
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0,
        }

        for recommendation in recommendations:

            if not isinstance(
                recommendation,
                dict
            ):
                continue

            priority = str(
                recommendation.get(
                    "priority",
                    "",
                )
            ).upper()

            if priority in priorities:

                priorities[priority] += 1

        result["high_priority"] = (
            priorities["HIGH"]
        )

        result["medium_priority"] = (
            priorities["MEDIUM"]
        )

        result["low_priority"] = (
            priorities["LOW"]
        )

    else:

        for output_key, possible_keys in {
            "recommendations": [
                "recommendations_generated",
                "recommendation_count",
                "total_recommendations",
            ],

            "high_priority": [
                "high_priority",
                "high",
            ],

            "medium_priority": [
                "medium_priority",
                "medium",
            ],

            "low_priority": [
                "low_priority",
                "low",
            ],

        }.items():

            for key in possible_keys:

                if key in optimization_result:

                    result[output_key] = (
                        safe_value(
                            optimization_result[key]
                        )
                    )

                    break

    return result


def save_dataframe(
    dataframe,
    path,
):
    """
    Save a DataFrame safely.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        path,
        index=False,
    )

    print(
        f"Saved: {path}"
    )


# =====================================================================
# SINGLE DATASET PIPELINE
# =====================================================================

def run_dataset(
    dataset_config,
):
    """
    Run the complete APMOA pipeline for one dataset.

    Returns a dictionary containing all experiment results.
    """

    dataset_name = dataset_config["name"]

    log_path = Path(
        dataset_config["path"]
    )

    print_header(
        f"DATASET: {dataset_name}"
    )

    print(
        f"Dataset path: {log_path}"
    )

    started_at = datetime.now()

    total_start = time.perf_counter()

    result = {
        "dataset": dataset_name,
        "path": str(log_path),
        "status": "FAILED",
        "error": None,
        "events": None,
        "cases": None,
        "activities": None,
        "resources": None,
        "selected_model": None,
        "selection_score": None,
        "reasoning_action": None,
        "reasoning_strategy": None,
        "reasoning_confidence": None,
        "adaptation_action": None,
        "adaptation_strategy": None,
        "adaptation_target": None,
        "adaptation_current_model": None,
        "adaptation_next_algorithm": None,
        "adaptation_policy_triggered": None,
        "bottlenecks": None,
        "deviations": None,
        "rework_activities": None,
        "transitions": None,
        "recommendations": None,
        "high_priority": None,
        "medium_priority": None,
        "low_priority": None,
        "rag_enabled": False,
        "rag_index_built": False,
        "rag_documents": None,
        "execution_time_seconds": None,
        "started_at": started_at.isoformat(),
        "completed_at": None,
    }

    model_records = []
    adaptation_record = {}
    optimization_record = {}

    try:

        # ============================================================
        # STEP 1 - DATASET CHECK
        # ============================================================

        print_subheader(
            "STEP 1 - DATASET CHECK"
        )

        check(
            log_path.exists(),
            f"Dataset exists: {log_path}"
        )

        # ============================================================
        # STEP 2 - DATASET LOADING
        # ============================================================

        print_subheader(
            "STEP 2 - DATASET LOADING"
        )

        loader = EventLogLoader(
            str(log_path)
        )

        log = loader.load()

        check(
            isinstance(
                log,
                pd.DataFrame
            ),
            "Event log loaded as pandas DataFrame"
        )

        required_columns = [
            "case_id",
            "activity",
            "timestamp",
            "resource",
        ]

        check(
            all(
                column in log.columns
                for column in required_columns
            ),
            "Canonical APMOA schema is available"
        )

        events = len(log)

        cases = log[
            "case_id"
        ].nunique()

        activities = log[
            "activity"
        ].nunique()

        resources = log[
            "resource"
        ].nunique()

        result["events"] = events
        result["cases"] = cases
        result["activities"] = activities
        result["resources"] = resources

        print()
        print(
            f"Events      : {events:,}"
        )

        print(
            f"Cases       : {cases:,}"
        )

        print(
            f"Activities  : {activities:,}"
        )

        print(
            f"Resources   : {resources:,}"
        )

        # ------------------------------------------------------------
        # Optional exact validation
        # ------------------------------------------------------------

        expected_events = (
            dataset_config.get(
                "expected_events"
            )
        )

        expected_cases = (
            dataset_config.get(
                "expected_cases"
            )
        )

        expected_activities = (
            dataset_config.get(
                "expected_activities"
            )
        )

        expected_resources = (
            dataset_config.get(
                "expected_resources"
            )
        )

        if expected_events is not None:

            check(
                events == expected_events,
                f"Event count = {expected_events:,}"
            )

        if expected_cases is not None:

            check(
                cases == expected_cases,
                f"Case count = {expected_cases:,}"
            )

        if expected_activities is not None:

            check(
                activities == expected_activities,
                (
                    "Activity count = "
                    f"{expected_activities}"
                )
            )

        if expected_resources is not None:

            check(
                resources == expected_resources,
                (
                    "Resource count = "
                    f"{expected_resources}"
                )
            )

        # ============================================================
        # STEP 3 - DATA ANALYSIS
        # ============================================================

        print_subheader(
            "STEP 3 - DATA ANALYSIS"
        )

        try:

            data_analysis_agent = (
                DataAnalysisAgent(log)
            )

        except TypeError:

            data_analysis_agent = (
                DataAnalysisAgent()
            )

        analysis_result = None

        if hasattr(
            data_analysis_agent,
            "analyze"
        ):

            try:

                analysis_result = (
                    data_analysis_agent.analyze(
                        log
                    )
                )

            except TypeError:

                analysis_result = (
                    data_analysis_agent.analyze()
                )

        elif hasattr(
            data_analysis_agent,
            "run"
        ):

            try:

                analysis_result = (
                    data_analysis_agent.run(
                        log
                    )
                )

            except TypeError:

                analysis_result = (
                    data_analysis_agent.run()
                )

        else:

            raise AttributeError(
                "DataAnalysisAgent does not expose "
                "analyze() or run()."
            )

        check(
            analysis_result is not None,
            "Data Analysis Agent returned a result"
        )

        # ============================================================
        # STEP 4 - PROCESS DISCOVERY
        # ============================================================

        print_subheader(
            "STEP 4 - PROCESS DISCOVERY"
        )

        dataset_discovery_dir = (
            DISCOVERY_RESULTS_DIR
            / dataset_name.replace(
                " ",
                "_"
            )
        )

        discovery_agent = (
            ProcessDiscoveryAgent(
                log=log,
                output_dir=str(
                    dataset_discovery_dir
                ),
            )
        )

        discovery_result = (
            discovery_agent.discover_all()
        )

        check(
            discovery_result is not None,
            "Process Discovery Agent returned a result"
        )

        if isinstance(
            discovery_result,
            dict
        ):

            models = discovery_result

        elif isinstance(
            discovery_result,
            (list, tuple)
        ):

            models = discovery_result

        else:

            raise TypeError(
                "Unexpected discovery result type: "
                f"{type(discovery_result).__name__}"
            )

        if isinstance(
            models,
            dict
        ):

            model_count = len(
                models
            )

            for algorithm in EXPECTED_ALGORITHMS:

                check(
                    algorithm in models,
                    f"{algorithm} discovered"
                )

        else:

            model_count = len(
                models
            )

        check(
            model_count > 0,
            "At least one process model discovered"
        )

        # ============================================================
        # STEP 5 - MODEL EVALUATION
        # ============================================================

        print_subheader(
            "STEP 5 - MODEL EVALUATION"
        )

        evaluation_agent = (
            ModelEvaluationAgent(
                log=log
            )
        )

        evaluation_result = (
            evaluation_agent.evaluate(
                models
            )
        )

        check(
            evaluation_result is not None,
            "Evaluation Agent returned a result"
        )

        model_records = extract_model_results(
            dataset_name,
            evaluation_result,
        )

        # ============================================================
        # STEP 6 - MODEL SELECTION
        # ============================================================

        print_subheader(
            "STEP 6 - MODEL SELECTION"
        )

        selection_agent = (
            ModelSelectionAgent(
                evaluation_results=evaluation_result
            )
        )

        if hasattr(
            selection_agent,
            "select"
        ):

            selection_result = (
                selection_agent.select()
            )

        elif hasattr(
            selection_agent,
            "run"
        ):

            selection_result = (
                selection_agent.run()
            )

        else:

            raise AttributeError(
                "ModelSelectionAgent does not expose "
                "select() or run()."
            )

        (
            selected_model,
            selected_evaluation,
        ) = extract_selected_model(
            selection_result,
            evaluation_result,
        )

        check(
            selected_model is not None,
            "A selected model is available"
        )

        selection_score = None

        if isinstance(
            selection_result,
            dict
        ):

            selection_score = (
                selection_result.get(
                    "selection_score"
                )
            )

        if selection_score is None:

            if isinstance(
                selected_evaluation,
                dict
            ):

                selection_score = (
                    selected_evaluation.get(
                        "_selection_score"
                    )
                )

        result["selected_model"] = (
            safe_value(
                selected_model
            )
        )

        result["selection_score"] = (
            safe_value(
                selection_score
            )
        )

        # ============================================================
        # STEP 7 - RAG KNOWLEDGE BASE
        # ============================================================

        print_subheader(
            "STEP 7 - RAG KNOWLEDGE BASE"
        )

        check(
            KNOWLEDGE_DIR.exists(),
            (
                "Knowledge directory exists: "
                f"{KNOWLEDGE_DIR}"
            )
        )

        knowledge_base = (
            APMOAKnowledgeBase(
                knowledge_dir=str(
                    KNOWLEDGE_DIR
                )
            )
        )

        check(
            knowledge_base is not None,
            "RAG knowledge base initialized"
        )

        result["rag_enabled"] = True

        print()
        print(
            "Building RAG knowledge index..."
        )

        build_result = (
            knowledge_base.build()
        )

        result["rag_index_built"] = True

        print(
            "RAG knowledge index built successfully."
        )

        retriever = getattr(
            knowledge_base,
            "retriever",
            None
        )

        check(
            retriever is not None,
            "RAG retriever initialized"
        )

        rag_document_count = None

        for attribute_name in [
            "documents",
            "chunks",
            "texts",
            "metadata",
        ]:

            value = getattr(
                retriever,
                attribute_name,
                None
            )

            if isinstance(
                value,
                (list, tuple)
            ):

                rag_document_count = len(
                    value
                )

                break

        if rag_document_count is None:

            for attribute_name in [
                "documents",
                "chunks",
                "texts",
            ]:

                value = getattr(
                    knowledge_base,
                    attribute_name,
                    None
                )

                if isinstance(
                    value,
                    (list, tuple)
                ):

                    rag_document_count = len(
                        value
                    )

                    break

        result["rag_documents"] = (
            rag_document_count
        )

        if rag_document_count is not None:

            print(
                f"RAG indexed items : "
                f"{rag_document_count}"
            )

            check(
                rag_document_count > 0,
                "RAG knowledge index contains documents"
            )

        else:

            print(
                "RAG index built; internal document count "
                "is not exposed by the current implementation."
            )

        # ============================================================
        # STEP 8 - RAG + GROQ REASONING
        # ============================================================

        print_subheader(
            "STEP 8 - RAG + GROQ REASONING"
        )

        reasoning_agent = (
            ReasoningAgent(
                knowledge_base=knowledge_base,
                rag_top_k=3,
            )
        )

        reasoning_result = None

        if hasattr(
            reasoning_agent,
            "reason"
        ):

            reasoning_result = (
                reasoning_agent.reason(
                    model_evaluations=evaluation_result,
                    selected_model=selected_model,
                    diagnostics=analysis_result,
                )
            )

        elif hasattr(
            reasoning_agent,
            "run"
        ):

            reasoning_result = (
                reasoning_agent.run(
                    model_evaluations=evaluation_result,
                    selected_model=selected_model,
                    diagnostics=analysis_result,
                )
            )

        else:

            raise AttributeError(
                "ReasoningAgent does not expose "
                "reason() or run()."
            )

        check(
            reasoning_result is not None,
            "Reasoning Agent returned a result"
        )

        reasoning_fields = (
            extract_reasoning_fields(
                reasoning_result
            )
        )

        result.update(
            reasoning_fields
        )

        # ============================================================
        # STEP 9 - AUTONOMOUS ADAPTATION
        # ============================================================

        print_subheader(
            "STEP 9 - AUTONOMOUS ADAPTATION"
        )

        adaptation_agent = (
            AdaptationAgent()
        )

        adaptation_result = None

        try:

            adaptation_result = (
                adaptation_agent.adapt(
                    evaluation=selected_evaluation,
                    reasoning=reasoning_result,
                    iteration=1,
                )
            )

        except TypeError:

            adaptation_result = (
                adaptation_agent.adapt(
                    selected_evaluation,
                    reasoning_result,
                    1,
                )
            )

        check(
            adaptation_result is not None,
            "Adaptation Agent returned a result"
        )

        adaptation_fields = (
            extract_adaptation_fields(
                adaptation_result
            )
        )

        result.update(
            adaptation_fields
        )

        adaptation_record = {
            "dataset": dataset_name,
            **adaptation_fields,
        }

        # ============================================================
        # STEP 10 - PROCESS INTELLIGENCE
        # ============================================================

        print_subheader(
            "STEP 10 - PROCESS INTELLIGENCE"
        )

        bottleneck_agent = (
            BottleneckAnalysisAgent(
                log
            )
        )

        bottleneck_result = None

        if hasattr(
            bottleneck_agent,
            "analyze"
        ):

            try:

                bottleneck_result = (
                    bottleneck_agent.analyze()
                )

            except TypeError:

                bottleneck_result = (
                    bottleneck_agent.analyze(
                        log
                    )
                )

        elif hasattr(
            bottleneck_agent,
            "run"
        ):

            try:

                bottleneck_result = (
                    bottleneck_agent.run()
                )

            except TypeError:

                bottleneck_result = (
                    bottleneck_agent.run(
                        log
                    )
                )

        else:

            raise AttributeError(
                "BottleneckAnalysisAgent does not expose "
                "analyze() or run()."
            )

        check(
            bottleneck_result is not None,
            "Bottleneck Analysis returned a result"
        )

        process_summary = (
            summarize_process_intelligence(
                bottleneck_result
            )
        )

        result.update(
            process_summary
        )

        # ============================================================
        # STEP 11 - OPTIMIZATION
        # ============================================================

        print_subheader(
            "STEP 11 - OPTIMIZATION"
        )

        optimization_agent = (
            OptimizationAgent()
        )

        optimization_result = None

        if hasattr(
            optimization_agent,
            "optimize"
        ):

            try:

                optimization_result = (
                    optimization_agent.optimize(
                        bottleneck_result
                    )
                )

            except TypeError:

                try:

                    optimization_result = (
                        optimization_agent.optimize(
                            bottlenecks=bottleneck_result
                        )
                    )

                except TypeError:

                    optimization_result = (
                        optimization_agent.optimize(
                            bottleneck_result,
                            log,
                        )
                    )

        elif hasattr(
            optimization_agent,
            "run"
        ):

            try:

                optimization_result = (
                    optimization_agent.run(
                        bottleneck_result
                    )
                )

            except TypeError:

                optimization_result = (
                    optimization_agent.run()
                )

        elif hasattr(
            optimization_agent,
            "generate_recommendations"
        ):

            try:

                optimization_result = (
                    optimization_agent.generate_recommendations(
                        bottleneck_result
                    )
                )

            except TypeError:

                optimization_result = (
                    optimization_agent.generate_recommendations()
                )

        else:

            raise AttributeError(
                "OptimizationAgent does not expose "
                "a supported execution method."
            )

        check(
            optimization_result is not None,
            "Optimization Agent returned a result"
        )

        optimization_summary = (
            summarize_optimization(
                optimization_result
            )
        )

        result.update(
            optimization_summary
        )

        optimization_record = {
            "dataset": dataset_name,
            **optimization_summary,
        }

        # ============================================================
        # COMPLETE
        # ============================================================

        total_time = (
            time.perf_counter()
            - total_start
        )

        result["execution_time_seconds"] = (
            round(
                total_time,
                3,
            )
        )

        result["status"] = "PASS"

        result["completed_at"] = (
            datetime.now().isoformat()
        )

        print_header(
            f"DATASET COMPLETED: {dataset_name}"
        )

        print(
            "Status          : PASS"
        )

        print(
            f"Events          : {events:,}"
        )

        print(
            f"Cases           : {cases:,}"
        )

        print(
            f"Selected model  : {selected_model}"
        )

        print(
            f"RAG index       : "
            f"{result['rag_index_built']}"
        )

        print(
            f"RAG documents   : "
            f"{result['rag_documents']}"
        )

        print(
            f"Reasoning       : "
            f"{result['reasoning_action']}"
        )

        print(
            f"Adaptation      : "
            f"{result['adaptation_action']}"
        )

        print(
            f"Execution time  : "
            f"{result['execution_time_seconds']} seconds"
        )

    except Exception as exc:

        total_time = (
            time.perf_counter()
            - total_start
        )

        result["status"] = "FAILED"

        result["error"] = (
            f"{type(exc).__name__}: {exc}"
        )

        result["execution_time_seconds"] = (
            round(
                total_time,
                3,
            )
        )

        result["completed_at"] = (
            datetime.now().isoformat()
        )

        print()
        print(
            f"DATASET FAILED: {dataset_name}"
        )

        print(
            f"Error: {exc}"
        )

        print()
        traceback.print_exc()

    return (
        result,
        model_records,
        adaptation_record,
        optimization_record,
    )


# =====================================================================
# MAIN MULTI-DATASET EXPERIMENT
# =====================================================================

def main():

    print_header(
        "APMOA - MULTI-DATASET EXPERIMENT"
    )

    print(
        "Project root:"
    )

    print(
        PROJECT_ROOT
    )

    print()

    print(
        f"Datasets configured: "
        f"{len(DATASETS)}"
    )

    print(
        "Results directory:"
    )

    print(
        RESULTS_DIR
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DISCOVERY_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Validate registry
    # ---------------------------------------------------------------

    print_subheader(
        "DATASET REGISTRY"
    )

    check(
        len(DATASETS) > 0,
        "At least one dataset is configured"
    )

    for dataset in DATASETS:

        print(
            f"  - {dataset['name']}"
        )

        print(
            f"    {dataset['path']}"
        )

    # ---------------------------------------------------------------
    # Run datasets
    # ---------------------------------------------------------------

    dataset_results = []
    model_results = []
    adaptation_results = []
    optimization_results = []

    experiment_start = (
        time.perf_counter()
    )

    for index, dataset_config in enumerate(
        DATASETS,
        start=1,
    ):

        print()
        print(
            "#" * 80
        )

        print(
            f"DATASET {index}/{len(DATASETS)}"
        )

        print(
            "#" * 80
        )

        (
            dataset_result,
            dataset_model_results,
            dataset_adaptation_result,
            dataset_optimization_result,
        ) = run_dataset(
            dataset_config
        )

        dataset_results.append(
            dataset_result
        )

        model_results.extend(
            dataset_model_results
        )

        if dataset_adaptation_result:

            adaptation_results.append(
                dataset_adaptation_result
            )

        if dataset_optimization_result:

            optimization_results.append(
                dataset_optimization_result
            )

        if dataset_result["status"] == "FAILED":

            print()
            print(
                "WARNING:"
            )

            print(
                f"{dataset_config['name']} failed."
            )

            print(
                "Continuing with the next dataset."
            )

    # ---------------------------------------------------------------
    # Experiment duration
    # ---------------------------------------------------------------

    experiment_time = (
        time.perf_counter()
        - experiment_start
    )

    # ---------------------------------------------------------------
    # Save dataset results
    # ---------------------------------------------------------------

    print_header(
        "SAVING EXPERIMENT RESULTS"
    )

    dataset_df = pd.DataFrame(
        dataset_results
    )

    model_df = pd.DataFrame(
        model_results
    )

    adaptation_df = pd.DataFrame(
        adaptation_results
    )

    optimization_df = pd.DataFrame(
        optimization_results
    )

    save_dataframe(
        dataset_df,
        RESULTS_DIR
        / "dataset_results.csv",
    )

    save_dataframe(
        model_df,
        RESULTS_DIR
        / "model_results.csv",
    )

    save_dataframe(
        adaptation_df,
        RESULTS_DIR
        / "adaptation_results.csv",
    )

    save_dataframe(
        optimization_df,
        RESULTS_DIR
        / "optimization_results.csv",
    )

    # ---------------------------------------------------------------
    # JSON summary
    # ---------------------------------------------------------------

    passed = sum(
        1
        for item in dataset_results
        if item["status"] == "PASS"
    )

    failed = sum(
        1
        for item in dataset_results
        if item["status"] == "FAILED"
    )

    summary = {
        "experiment": (
            "APMOA Multi-Dataset Experiment"
        ),

        "timestamp": (
            datetime.now().isoformat()
        ),

        "datasets_configured": len(
            DATASETS
        ),

        "datasets_passed": passed,

        "datasets_failed": failed,

        "total_execution_time_seconds": round(
            experiment_time,
            3,
        ),

        "datasets": dataset_results,

        "model_results": model_results,

        "adaptation_results": adaptation_results,

        "optimization_results": optimization_results,
    }

    summary_path = (
        RESULTS_DIR
        / "experiment_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            make_json_safe(summary),
            file,
            indent=2,
        )

    print(
        f"Saved: {summary_path}"
    )

    # ---------------------------------------------------------------
    # FINAL SUMMARY
    # ---------------------------------------------------------------

    print_header(
        "MULTI-DATASET EXPERIMENT SUMMARY"
    )

    print()

    print(
        f"{'Dataset':35}"
        f"{'Status':10}"
        f"{'Events':>12}"
        f"{'Cases':>10}"
        f"{'Selected Model':25}"
    )

    print(
        "-" * 92
    )

    for item in dataset_results:

        print(
            f"{item['dataset'][:35]:35}"
            f"{item['status']:<10}"
            f"{str(item['events'] or '-'):>12}"
            f"{str(item['cases'] or '-'):>10}"
            f"{str(item['selected_model'] or '-'):25}"
        )

    print()

    print(
        f"Datasets configured : "
        f"{len(DATASETS)}"
    )

    print(
        f"Datasets passed     : "
        f"{passed}"
    )

    print(
        f"Datasets failed     : "
        f"{failed}"
    )

    print(
        f"Total execution time: "
        f"{experiment_time:.2f}s"
    )

    print()

    if failed == 0:

        print(
            "=" * 80
        )

        print(
            "MULTI-DATASET EXPERIMENT PASSED"
        )

        print(
            "=" * 80
        )

    else:

        print(
            "=" * 80
        )

        print(
            "MULTI-DATASET EXPERIMENT COMPLETED "
            "WITH FAILURES"
        )

        print(
            "=" * 80
        )

    print()

    print(
        "Results:"
    )

    print(
        f"  {RESULTS_DIR / 'dataset_results.csv'}"
    )

    print(
        f"  {RESULTS_DIR / 'model_results.csv'}"
    )

    print(
        f"  {RESULTS_DIR / 'adaptation_results.csv'}"
    )

    print(
        f"  {RESULTS_DIR / 'optimization_results.csv'}"
    )

    print(
        f"  {RESULTS_DIR / 'experiment_summary.json'}"
    )

    print()

    if failed > 0:
        return 1

    return 0


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    try:

        exit_code = main()

        sys.exit(
            exit_code
        )

    except KeyboardInterrupt:

        print()

        print(
            "MULTI-DATASET EXPERIMENT INTERRUPTED."
        )

        sys.exit(1)

    except Exception as exc:

        print()

        print(
            "=" * 80
        )

        print(
            "MULTI-DATASET EXPERIMENT FAILED"
        )

        print(
            "=" * 80
        )

        print()

        print(
            f"Error: {exc}"
        )

        print()

        traceback.print_exc()

        sys.exit(1)