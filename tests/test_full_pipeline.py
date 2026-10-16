"""
APMOA - Full Pipeline Integration Test

Tests the complete APMOA pipeline:

Dataset
    ↓
Data Analysis
    ↓
Process Discovery
    ↓
Model Evaluation
    ↓
Model Selection
    ↓
RAG Knowledge Retrieval
    ↓
Groq Reasoning
    ↓
Adaptation
    ↓
Process Intelligence
    ↓
Optimization

Dataset:
    BPI Challenge 2020 - Travel Permit Process
"""

from pathlib import Path
import sys
import traceback

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


# =====================================================================
# CONFIGURATION
# =====================================================================

LOG_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Travel"
    / "PermitLog.xes.gz"
)

KNOWLEDGE_DIR = (
    PROJECT_ROOT
    / "data"
    / "knowledge"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "full_pipeline_test"
)


# =====================================================================
# EXPECTED DATASET STATISTICS
# =====================================================================

EXPECTED_EVENTS = 86581
EXPECTED_CASES = 7065
EXPECTED_ACTIVITIES = 51
EXPECTED_RESOURCES = 2


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def print_header(title: str):
    """Print a major section header."""

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_subheader(title: str):
    """Print a subsection header."""

    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


def check(condition: bool, message: str):
    """Raise an error when an integration check fails."""

    if not condition:
        raise AssertionError(message)

    print(
        f"PASS: {message}"
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print_header(
        "APMOA - FULL PIPELINE INTEGRATION TEST"
    )

    print()
    print("Project root:")
    print(PROJECT_ROOT)

    print()
    print("Dataset:")
    print(LOG_PATH)


    # ================================================================
    # STEP 1 - DATASET LOADING
    # ================================================================

    print_header(
        "STEP 1 - DATASET LOADING"
    )

    check(
        LOG_PATH.exists(),
        f"Dataset exists: {LOG_PATH}"
    )


    # ================================================================
    # STEP 2 - EVENT LOG LOADER
    # ================================================================

    print_header(
        "STEP 2 - EVENT LOG LOADER"
    )

    loader = EventLogLoader(
        str(LOG_PATH)
    )

    log = loader.load()

    check(
        isinstance(
            log,
            pd.DataFrame
        ),
        "Event log loaded as pandas DataFrame"
    )

    print()

    print(
        f"Events      : {len(log):,}"
    )

    print(
        f"Cases       : "
        f"{log['case_id'].nunique():,}"
    )

    print(
        f"Activities  : "
        f"{log['activity'].nunique():,}"
    )

    print(
        f"Resources   : "
        f"{log['resource'].nunique():,}"
    )


    # ================================================================
    # DATASET VALIDATION
    # ================================================================

    print_subheader(
        "DATASET VALIDATION"
    )

    check(
        len(log) == EXPECTED_EVENTS,
        f"Event count = {EXPECTED_EVENTS:,}"
    )

    check(
        log["case_id"].nunique()
        == EXPECTED_CASES,
        f"Case count = {EXPECTED_CASES:,}"
    )

    check(
        log["activity"].nunique()
        == EXPECTED_ACTIVITIES,
        f"Activity count = {EXPECTED_ACTIVITIES}"
    )

    check(
        log["resource"].nunique()
        == EXPECTED_RESOURCES,
        f"Resource count = {EXPECTED_RESOURCES}"
    )


    # ================================================================
    # STEP 3 - DATA ANALYSIS AGENT
    # ================================================================

    print_header(
        "STEP 3 - DATA ANALYSIS AGENT"
    )

    try:

        data_analysis_agent = (
            DataAnalysisAgent(log)
        )

    except TypeError:

        data_analysis_agent = (
            DataAnalysisAgent()
        )

    print(
        "Running Data Analysis Agent..."
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

    print(
        "Data Analysis Agent completed."
    )


    # ================================================================
    # STEP 4 - PROCESS DISCOVERY
    # ================================================================

    print_header(
        "STEP 4 - PROCESS DISCOVERY"
    )

    discovery_agent = (
        ProcessDiscoveryAgent(
            log=log,
            output_dir=str(
                RESULTS_DIR / "discovery"
            )
        )
    )

    print(
        "Running Process Discovery Agent..."
    )

    # Current ProcessDiscoveryAgent API:
    #
    #     discover_all()
    #
    # It returns the model dictionary directly.

    discovery_result = (
        discovery_agent.discover_all()
    )

    check(
        discovery_result is not None,
        "Process Discovery Agent returned a result"
    )


    # ================================================================
    # DISCOVERY RESULTS
    # ================================================================

    print_subheader(
        "DISCOVERY RESULTS"
    )

    print(
        "Discovery result type:"
    )

    print(
        type(discovery_result).__name__
    )


    models = None


    if isinstance(
        discovery_result,
        dict
    ):

        print(
            "Discovery result keys:"
        )

        print(
            list(
                discovery_result.keys()
            )
        )

        # IMPORTANT:
        # discover_all() already returns:
        #
        # {
        #   "Inductive Miner": model,
        #   "Alpha Miner": model,
        #   ...
        # }
        #
        # Therefore do NOT search for a nested
        # "models" or "results" key.

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


    check(
        models is not None,
        "Discovered models are available"
    )


    if isinstance(
        models,
        dict
    ):

        model_count = len(
            models
        )

    else:

        try:

            model_count = len(
                models
            )

        except TypeError:

            model_count = 0


    print()

    print(
        f"Models discovered: "
        f"{model_count}"
    )


    check(
        model_count > 0,
        "At least one process model discovered"
    )


    expected_algorithms = [
        "Inductive Miner",
        "Alpha Miner",
        "Alpha+ Miner",
        "Heuristics Miner",
        "Split Miner",
    ]


    if isinstance(
        models,
        dict
    ):

        for algorithm in expected_algorithms:

            check(
                algorithm in models,
                f"{algorithm} discovered"
            )


    # ================================================================
    # STEP 5 - MODEL EVALUATION
    # ================================================================

    print_header(
        "STEP 5 - MODEL EVALUATION"
    )

    evaluation_agent = (
        ModelEvaluationAgent(
            log=log
        )
    )

    print(
        "Evaluating discovered models..."
    )


    try:

        evaluation_result = (
            evaluation_agent.evaluate(
                models
            )
        )

    except TypeError:

        evaluation_result = (
            evaluation_agent.evaluate(
                discovery_result
            )
        )


    check(
        evaluation_result is not None,
        "Evaluation Agent returned a result"
    )


    # ================================================================
    # EVALUATION RESULTS
    # ================================================================

    print_subheader(
        "EVALUATION RESULTS"
    )


    if isinstance(
        evaluation_result,
        pd.DataFrame
    ):

        print(
            evaluation_result
        )

    elif isinstance(
        evaluation_result,
        dict
    ):

        print(
            "Evaluation result keys:"
        )

        print(
            list(
                evaluation_result.keys()
            )
        )

    else:

        print(
            "Evaluation result type:"
        )

        print(
            type(evaluation_result).__name__
        )


    # ================================================================
    # STEP 6 - MODEL SELECTION
    # ================================================================

    print_header(
        "STEP 6 - MODEL SELECTION"
    )

    selected_model = None
    selected_evaluation = None


    # ================================================================
    # MODEL SELECTION AGENT
    # ================================================================

    try:

        from src.agents.model_selection_agent import (
            ModelSelectionAgent
        )

        print(
            "ModelSelectionAgent found."
        )

        # IMPORTANT:
        #
        # The actual constructor requires:
        #
        #     evaluation_results
        #
        # Therefore we pass the evaluation result
        # during initialization.

        selection_agent = (
            ModelSelectionAgent(
                evaluation_results=evaluation_result
            )
        )

        print(
            "ModelSelectionAgent initialized."
        )


        # ------------------------------------------------------------
        # RUN MODEL SELECTION
        # ------------------------------------------------------------

        selection_result = None


        if hasattr(
            selection_agent,
            "select"
        ):

            try:

                selection_result = (
                    selection_agent.select()
                )

            except TypeError:

                selection_result = (
                    selection_agent.select(
                        evaluation_result
                    )
                )


        elif hasattr(
            selection_agent,
            "run"
        ):

            try:

                selection_result = (
                    selection_agent.run()
                )

            except TypeError:

                selection_result = (
                    selection_agent.run(
                        evaluation_result
                    )
                )


        else:

            raise AttributeError(
                "ModelSelectionAgent does not expose "
                "select() or run()."
            )


        print()

        print(
            "Selection result type:"
        )

        print(
            type(selection_result).__name__
        )


        # ------------------------------------------------------------
        # EXTRACT SELECTION RESULT
        # ------------------------------------------------------------

        if isinstance(
            selection_result,
            dict
        ):

            print(
                "Selection result keys:"
            )

            print(
                list(
                    selection_result.keys()
                )
            )


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


            # Support alternative field names.

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

        else:

            selected_model = (
                selection_result
            )


    except (
        ImportError,
        ModuleNotFoundError
    ):

        print(
            "Standalone ModelSelectionAgent "
            "not found."
        )


    # ================================================================
    # FALLBACK SELECTION
    #
    # This is used only if a standalone selection agent
    # is unavailable.
    #
    # These are the APMOA selection weights:
    #
    # Fitness       = 0.30
    # Precision     = 0.30
    # Generalization = 0.20
    # Simplicity    = 0.20
    # ================================================================

    if selected_model is None:

        print(
            "Using weighted evaluation fallback "
            "for integration testing."
        )


        if isinstance(
            evaluation_result,
            pd.DataFrame
        ):

            df = (
                evaluation_result.copy()
            )


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
                    0.30 * df["fitness"]
                    +
                    0.30 * df["precision"]
                    +
                    0.20 * df["generalization"]
                    +
                    0.20 * df["simplicity"]
                )


                best_row = (
                    df.sort_values(
                        "_selection_score",
                        ascending=False
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


            for key, value in (
                evaluation_result.items()
            ):

                if not isinstance(
                    value,
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
                    metric in value
                    for metric in required_metrics
                ):

                    continue


                score = (
                    0.30
                    * value["fitness"]
                    +
                    0.30
                    * value["precision"]
                    +
                    0.20
                    * value["generalization"]
                    +
                    0.20
                    * value["simplicity"]
                )


                candidates.append(
                    (
                        score,
                        key,
                        value
                    )
                )


            if candidates:

                candidates.sort(
                    reverse=True
                )


                (
                    _,
                    selected_model,
                    selected_evaluation
                ) = candidates[0]


    # ================================================================
    # SELECTION VALIDATION
    # ================================================================

    print()

    print(
        f"Selected model: "
        f"{selected_model}"
    )


    check(
        selected_model is not None,
        "A selected model is available"
    )


    # ================================================================
    # STEP 7 - RAG KNOWLEDGE BASE
    # ================================================================

    print_header(
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


    print(
        "RAG knowledge base initialized."
    )


    check(
        knowledge_base is not None,
        "RAG knowledge base initialized"
    )


    # ================================================================
    # STEP 8 - RAG + GROQ REASONING
    # ================================================================

    print_header(
        "STEP 8 - RAG + GROQ REASONING"
    )


    reasoning_agent = (
        ReasoningAgent(
            knowledge_base=knowledge_base,
            rag_top_k=3
        )
    )


    print(
        "Running Reasoning Agent..."
    )


    reasoning_result = None


    # ---------------------------------------------------------------
    # reason()
    # ---------------------------------------------------------------
    #
    # ReasoningAgent.reason() expects:
    #   1. model_evaluations -> evaluation results for ALL models
    #   2. selected_model    -> currently selected model
    #   3. diagnostics       -> diagnostics produced by Data Analysis
    #
    # IMPORTANT:
    # Pass the complete evaluation result rather than only the
    # selected model's evaluation because the reasoning agent compares
    # the available models before making its recommendation.

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

    # ---------------------------------------------------------------
    # run()
    # ---------------------------------------------------------------

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


    print(
        "Reasoning Agent completed."
    )

    # Display the reasoning output for integration validation.
    print_subheader(
        "REASONING RESULT"
    )

    if isinstance(
        reasoning_result,
        dict
    ):

        print(
            "Reasoning result keys:"
        )

        print(
            list(
                reasoning_result.keys()
            )
        )

        for key in (
            "action",
            "recommended_action",
            "strategy",
            "confidence",
            "reason",
            "sources",
        ):

            if key in reasoning_result:

                print(
                    f"{key}: "
                    f"{reasoning_result[key]}"
                )

    else:

        print(
            f"Reasoning result type: "
            f"{type(reasoning_result).__name__}"
        )



    # ================================================================
    # STEP 9 - ADAPTATION
    # ================================================================

    print_header(
        "STEP 9 - ADAPTATION"
    )


    adaptation_agent = (
        AdaptationAgent()
    )


    print(
        "Running Adaptation Agent..."
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
                1
            )
        )


    check(
        adaptation_result is not None,
        "Adaptation Agent returned a result"
    )


    print(
        "Adaptation Agent completed."
    )


    # ================================================================
    # STEP 10 - PROCESS INTELLIGENCE
    # ================================================================

    print_header(
        "STEP 10 - PROCESS INTELLIGENCE"
    )


    bottleneck_agent = (
        BottleneckAnalysisAgent(
            log
        )
    )


    print(
        "Running Bottleneck Analysis Agent..."
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


    print(
        "Bottleneck analysis completed."
    )


    # ================================================================
    # STEP 11 - OPTIMIZATION
    # ================================================================

    print_header(
        "STEP 11 - OPTIMIZATION"
    )


    optimization_agent = (
        OptimizationAgent()
    )


    print(
        "Running Optimization Agent..."
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
                        log
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
                optimization_agent
                .generate_recommendations(
                    bottleneck_result
                )
            )

        except TypeError:

            optimization_result = (
                optimization_agent
                .generate_recommendations()
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


    print(
        "Optimization analysis completed."
    )


    # ================================================================
    # FINAL PIPELINE VALIDATION
    # ================================================================

    print_header(
        "FINAL PIPELINE VALIDATION"
    )


    check(
        isinstance(
            log,
            pd.DataFrame
        ),
        "Dataset layer passed"
    )


    check(
        model_count > 0,
        "Discovery layer passed"
    )


    check(
        evaluation_result is not None,
        "Evaluation layer passed"
    )


    check(
        selected_model is not None,
        "Selection layer passed"
    )


    check(
        reasoning_result is not None,
        "Reasoning layer passed"
    )


    check(
        adaptation_result is not None,
        "Adaptation layer passed"
    )


    check(
        bottleneck_result is not None,
        "Process intelligence layer passed"
    )


    check(
        optimization_result is not None,
        "Optimization layer passed"
    )


    # ================================================================
    # SUCCESS
    # ================================================================

    print()

    print(
        "=" * 70
    )

    print(
        "FULL APMOA PIPELINE TEST PASSED"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Pipeline successfully executed:"
    )

    print(
        "  Dataset Loading             [PASS]"
    )

    print(
        "  Data Analysis               [PASS]"
    )

    print(
        "  Process Discovery           [PASS]"
    )

    print(
        "  Model Evaluation            [PASS]"
    )

    print(
        "  Model Selection             [PASS]"
    )

    print(
        "  RAG Knowledge Retrieval     [PASS]"
    )

    print(
        "  Groq Reasoning              [PASS]"
    )

    print(
        "  Autonomous Adaptation       [PASS]"
    )

    print(
        "  Bottleneck Analysis         [PASS]"
    )

    print(
        "  Optimization                [PASS]"
    )

    print()

    print(
        f"Final selected model: "
        f"{selected_model}"
    )

    print()

    print(
        "=" * 70
    )


# =====================================================================
# ERROR HANDLING
# =====================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()

        print(
            "TEST INTERRUPTED BY USER."
        )

        sys.exit(1)

    except Exception as exc:

        print()

        print(
            "=" * 70
        )

        print(
            "FULL APMOA PIPELINE TEST FAILED"
        )

        print(
            "=" * 70
        )

        print()

        print(
            f"Error: {exc}"
        )

        print()

        print(
            "Traceback:"
        )

        traceback.print_exc()

        print()

        print(
            "=" * 70
        )

        sys.exit(1)