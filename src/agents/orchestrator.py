"""
APMOA - Autonomous Process Mining Orchestrator

End-to-end workflow:

    Event Log
        |
        v
    Data Analysis
        |
        v
    Multi-Algorithm Process Discovery
        |
        v
    Model Evaluation
        |
        v
    Model Selection
        |
        v
    Groq LLM Reasoning
        |
        v
    Adaptation
        |
        +----------------------+
        |                      |
       STOP                REDISCOVER
                               |
                               v
                          Next Iteration

This orchestrator is intentionally responsible for coordination only.
The specialized agents perform the actual process-mining, evaluation,
reasoning, and adaptation tasks.
"""

from __future__ import annotations
from dotenv import load_dotenv

load_dotenv()

import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.storage.experiment_manager import ExperimentManager


class APMOAOrchestrator:
    """
    Main autonomous controller for APMOA.

    Compatible with the current agent interfaces:

        DataAnalysisAgent(log).analyze()
        ProcessDiscoveryAgent(log=log).discover_all()
        ProcessDiscoveryAgent.discover_adaptive(adaptation=...)
        EvaluationAgent(log).evaluate(models)
        ModelSelectionAgent(evaluations).select()
        ReasoningAgent().reason(...)
        AdaptationAgent().adapt(...)

    The orchestrator is deliberately defensive because the project is
    being developed incrementally and agent APIs may evolve.
    """

    def __init__(
        self,
        data_analysis_agent,
        discovery_agent,
        evaluation_agent,
        selection_agent=None,
        reasoning_agent=None,
        adaptation_agent=None,
        bottleneck_agent=None,
        optimization_agent=None,
        log=None,
        max_iterations: int = 3,
        experiment_manager=None,
        dataset_name: Optional[str] = None,
        dataset_path: Optional[str] = None,
    ):
        self.data_analysis_agent = data_analysis_agent
        self.discovery_agent = discovery_agent
        self.evaluation_agent = evaluation_agent
        self.selection_agent = selection_agent
        self.reasoning_agent = reasoning_agent
        self.adaptation_agent = adaptation_agent
        self.bottleneck_agent = bottleneck_agent
        self.optimization_agent = optimization_agent
        self.log = log

        # Persistent experiment storage.
        self.experiment_manager = (
            experiment_manager
            if experiment_manager is not None
            else ExperimentManager()
        )
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.experiment_id: Optional[str] = None

        self.max_iterations = max(1, int(max_iterations))

        self.history: List[Dict[str, Any]] = []
        self.current_adaptation: Optional[Dict[str, Any]] = None

    # =========================================================
    # MAIN AUTONOMOUS LOOP
    # =========================================================

    def run(self) -> Dict[str, Any]:

        print("\n" + "=" * 70)
        print("APMOA - AUTONOMOUS PROCESS MINING")
        print("=" * 70)

        # =====================================================
        # PERSISTENT EXPERIMENT INITIALIZATION
        # =====================================================

        self.experiment_id = self.experiment_manager.create_experiment(
            dataset_name=self.dataset_name,
            dataset_path=self.dataset_path,
            description="APMOA autonomous process-mining execution",
            metadata={
                "max_iterations": self.max_iterations,
                "workflow": (
                    "data_analysis -> process_discovery -> "
                    "evaluation -> selection -> reasoning -> "
                    "adaptation -> process_intelligence -> optimization"
                ),
            },
        )

        print(f"\nExperiment ID: {self.experiment_id}")
        print(
            "Experiment storage:",
            self.experiment_manager.get_experiment_path(
                self.experiment_id
            ),
        )

        # =====================================================
        # PHASE 1 - DATA ANALYSIS
        # =====================================================

        print("\n" + "#" * 70)
        print("# PHASE 1 - DATA ANALYSIS")
        print("#" * 70)

        try:
            diagnostics = self.data_analysis_agent.analyze()
        except Exception as exc:
            print("\nData Analysis Agent failed.")
            print(f"Error: {exc}")
            traceback.print_exc()
            raise

        print("\nData analysis completed.")

        # =====================================================
        # AUTONOMOUS ITERATIONS
        # =====================================================

        for iteration in range(1, self.max_iterations + 1):

            print("\n" + "#" * 70)
            print(f"# AUTONOMOUS ITERATION {iteration}")
            print("#" * 70)

            # =================================================
            # PHASE 2 - PROCESS DISCOVERY
            # =================================================

            print("\n" + "-" * 60)
            print("PHASE 2 - PROCESS DISCOVERY")
            print("-" * 60)

            try:
                discovered_models = self._run_discovery(iteration)
            except Exception as exc:
                print("\nProcess Discovery Agent failed.")
                print(f"Error: {exc}")
                traceback.print_exc()
                raise

            if not discovered_models:
                raise RuntimeError(
                    "Process Discovery Agent returned no models."
                )

            print(
                f"\nModels discovered: "
                f"{self._count_models(discovered_models)}"
            )

            # =================================================
            # PHASE 3 - MODEL EVALUATION
            # =================================================

            print("\n" + "-" * 60)
            print("PHASE 3 - MODEL EVALUATION")
            print("-" * 60)

            try:
                evaluations = self._evaluate_models(
                    discovered_models
                )
            except Exception as exc:
                print("\nEvaluation Agent failed.")
                print(f"Error: {exc}")
                traceback.print_exc()
                raise

            # =================================================
            # PHASE 4 - MODEL SELECTION
            # =================================================

            print("\n" + "-" * 60)
            print("PHASE 4 - MODEL SELECTION")
            print("-" * 60)

            try:
                selected_model = self._select_model(
                    evaluations
                )
            except Exception as exc:
                print("\nModel Selection Agent failed.")
                print(f"Error: {exc}")
                traceback.print_exc()
                raise

            print("\nDEBUG - MODEL SELECTION OUTPUT")
            print("-" * 60)
            print(
                "Type :",
                type(selected_model).__name__,
            )
            print(
                "Value:",
                self._safe_repr(selected_model),
            )

            selected_name = self._get_selected_name(
                selected_model
            )

            print(
                "\nSelected model resolved as:",
                selected_name,
            )

            selected_evaluation = (
                self._get_selected_evaluation(
                    evaluations=evaluations,
                    selected_model=selected_model,
                    selected_name=selected_name,
                )
            )

            print("\nSelected model evaluation:")
            print(
                self._format_evaluation(
                    selected_evaluation
                )
            )

            # =================================================
            # PHASE 5 - GROQ REASONING
            # =================================================

            print("\n" + "-" * 60)
            print("PHASE 5 - GROQ REASONING")
            print("-" * 60)

            try:
                reasoning = self.reasoning_agent.reason(
                    diagnostics=diagnostics,
                    model_evaluations=evaluations,
                    selected_model=selected_name,
                )
            except Exception as exc:
                print("\nReasoning Agent failed.")
                print(f"Error: {exc}")
                traceback.print_exc()
                raise

            print("\nReasoning completed.")

            # =================================================
            # DISPLAY REASONING
            # =================================================

            self._print_reasoning(reasoning)

            # =================================================
            # PHASE 6 - ADAPTATION
            # =================================================

            print("\n" + "-" * 60)
            print("PHASE 6 - ADAPTATION")
            print("-" * 60)

            try:
                adaptation = self.adaptation_agent.adapt(
                    evaluation=selected_evaluation,
                    reasoning=reasoning,
                    iteration=iteration,
                )
            except Exception as exc:
                print("\nAdaptation Agent failed.")
                print(f"Error: {exc}")
                traceback.print_exc()
                raise

            print("\nAdaptation decision:")
            print(
                self._format_adaptation(
                    adaptation
                )
            )

            # =================================================
            # STORE ITERATION
            # =================================================

            iteration_record = {
                "iteration": iteration,
                "selected_model": selected_name,
                "evaluation": selected_evaluation,
                "reasoning": reasoning,
                "adaptation": adaptation,
            }

            self.history.append(
                iteration_record
            )

            # =================================================
            # STOP CONDITION
            # =================================================

            action_type = self._get_action_type(
                adaptation
            )

            if action_type in {"stop", "keep_model"}:

                print("\n" + "=" * 70)

                if action_type == "stop":
                    print("QUALITY TARGET ACHIEVED")
                else:
                    print("CURRENT MODEL RETAINED")

                print("AUTONOMOUS LOOP STOPPED")
                print("=" * 70)

                break

            # =================================================
            # MAX ITERATIONS
            # =================================================

            if iteration >= self.max_iterations:

                print("\n" + "=" * 70)
                print("MAXIMUM ITERATIONS REACHED")
                print("AUTONOMOUS LOOP STOPPED")
                print("=" * 70)

                break

            # =================================================
            # ADAPTATION -> NEXT ITERATION
            # =================================================

            print("\n" + "=" * 60)
            print("ADAPTATION REQUESTED")
            print("=" * 60)

            if isinstance(adaptation, dict):

                print(
                    "Action   :",
                    adaptation.get(
                        "action_type",
                        adaptation.get(
                            "action",
                            "unknown",
                        ),
                    ),
                )

                print(
                    "Strategy :",
                    adaptation.get(
                        "strategy",
                        "not specified",
                    ),
                )

                print(
                    "Parameters:",
                    adaptation.get(
                        "parameters",
                        {},
                    ),
                )

                self.current_adaptation = adaptation

            self._configure_discovery(
                adaptation
            )

            print(
                "\nPreparing next autonomous discovery iteration..."
            )

        # =====================================================
        # PROCESS INTELLIGENCE + OPTIMIZATION
        # =====================================================

        process_intelligence = self._run_process_intelligence()

        self._save_experiment_component(
            "process_intelligence",
            process_intelligence,
        )

        # =====================================================
        # FINAL RESULT
        # =====================================================

        final_adaptation = (
            self.history[-1]["adaptation"]
            if self.history
            else None
        )

        final_action = self._get_action_type(
            final_adaptation
        )

        final_result = {
            "experiment_id": self.experiment_id,
            "dataset_name": self.dataset_name,
            "dataset_path": self.dataset_path,
            "diagnostics": diagnostics,
            "history": self.history,
            "iterations": len(self.history),
            "final_model": (
                self.history[-1]["selected_model"]
                if self.history
                else None
            ),
            "final_evaluation": (
                self.history[-1]["evaluation"]
                if self.history
                else None
            ),
            "final_reasoning": (
                self.history[-1]["reasoning"]
                if self.history
                else None
            ),
            "final_adaptation": final_adaptation,
            "final_action": final_action,
            "loop_terminated": True,
            "process_intelligence": process_intelligence,
            "bottleneck_analysis": process_intelligence.get(
                "bottleneck_analysis"
            ),
            "optimization": process_intelligence.get(
                "optimization"
            ),
        }

        self._print_final_summary(
            final_result
        )

        # =====================================================
        # PERSIST COMPLETE EXPERIMENT
        # =====================================================

        self._persist_completed_experiment(
            final_result=final_result,
            evaluations=evaluations if "evaluations" in locals() else None,
            reasoning=(
                self.history[-1]["reasoning"]
                if self.history
                else None
            ),
            adaptation=final_adaptation,
            process_intelligence=process_intelligence,
            optimization=process_intelligence.get(
                "optimization"
            ),
        )

        return final_result

    # =========================================================
    # EXPERIMENT STORAGE
    # =========================================================

    def _make_json_safe(self, value: Any) -> Any:
        """Convert common runtime objects into JSON-safe structures."""

        if value is None:
            return None

        if isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, Path):
            return str(value)

        try:
            import pandas as pd

            if isinstance(value, pd.DataFrame):
                return value.to_dict(orient="records")

            if isinstance(value, pd.Series):
                return value.to_dict()
        except Exception:
            pass

        if isinstance(value, dict):
            return {
                str(key): self._make_json_safe(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [
                self._make_json_safe(item)
                for item in value
            ]

        try:
            if hasattr(value, "item") and callable(value.item):
                return self._make_json_safe(value.item())
        except Exception:
            pass

        return str(value)

    def _save_experiment_component(
        self,
        component_name: str,
        data: Any,
    ) -> None:
        """Persist one component without interrupting the workflow."""

        if not self.experiment_id:
            return

        try:
            self.experiment_manager.save_component(
                self.experiment_id,
                component_name,
                self._make_json_safe(data),
            )
            print(
                f"Experiment component saved: {component_name}.json"
            )
        except Exception as exc:
            print(
                f"Warning: failed to save experiment component "
                f"{component_name}: {exc}"
            )

    def _persist_completed_experiment(
        self,
        *,
        final_result: Dict[str, Any],
        evaluations: Any,
        reasoning: Any,
        adaptation: Any,
        process_intelligence: Any,
        optimization: Any,
    ) -> None:
        """Persist the completed APMOA execution."""

        if not self.experiment_id:
            return

        self._save_experiment_component(
            "evaluation_results",
            evaluations,
        )
        self._save_experiment_component(
            "reasoning",
            reasoning,
        )
        self._save_experiment_component(
            "adaptation",
            adaptation,
        )
        self._save_experiment_component(
            "process_intelligence",
            process_intelligence,
        )
        self._save_experiment_component(
            "optimization",
            optimization,
        )
        self._save_experiment_component(
            "final_result",
            final_result,
        )

        final_evaluation = (
            final_result.get("final_evaluation")
            if isinstance(final_result, dict)
            else None
        )

        try:
            self.experiment_manager._update_metadata(
                self.experiment_id,
                status="completed",
                final_model=(
                    final_result.get("final_model")
                    if isinstance(final_result, dict)
                    else None
                ),
                final_action=(
                    final_result.get("final_action")
                    if isinstance(final_result, dict)
                    else None
                ),
                iterations=(
                    final_result.get("iterations", 0)
                    if isinstance(final_result, dict)
                    else 0
                ),
                selection_score=(
                    final_evaluation.get("selection_score")
                    if isinstance(final_evaluation, dict)
                    else None
                ),
                fitness=(
                    final_evaluation.get("fitness")
                    if isinstance(final_evaluation, dict)
                    else None
                ),
                precision=(
                    final_evaluation.get("precision")
                    if isinstance(final_evaluation, dict)
                    else None
                ),
                generalization=(
                    final_evaluation.get("generalization")
                    if isinstance(final_evaluation, dict)
                    else None
                ),
                simplicity=(
                    final_evaluation.get("simplicity")
                    if isinstance(final_evaluation, dict)
                    else None
                ),
            )

            print(
                f"Experiment metadata finalized: {self.experiment_id}"
            )

        except Exception as exc:
            print(
                "Warning: failed to finalize experiment metadata:",
                exc,
            )

    # =========================================================
    # PROCESS INTELLIGENCE
    # =========================================================

    def _run_process_intelligence(self) -> Dict[str, Any]:
        """
        Run Bottleneck/Deviation/Rework analysis followed by Optimization.

        The agents are optional so the autonomous-loop test can continue to
        exercise only the discovery/evaluation/reasoning/adaptation loop.
        In the production orchestrator, both agents are supplied and this
        stage becomes part of the end-to-end workflow.
        """

        result: Dict[str, Any] = {
            "status": "SKIPPED",
            "bottleneck_analysis": None,
            "optimization": None,
        }

        if (
            self.bottleneck_agent is None
            and self.optimization_agent is None
        ):
            return result

        print("\n" + "-" * 60)
        print("PROCESS INTELLIGENCE")
        print("-" * 60)

        # ---------------------------------------------------------
        # Bottleneck / deviation / rework analysis
        # ---------------------------------------------------------

        bottleneck_result = None

        if self.bottleneck_agent is not None:

            try:

                print(
                    "\nRunning bottleneck/deviation/rework analysis..."
                )

                bottleneck_result = self._call_analysis_agent(
                    self.bottleneck_agent
                )

                result["bottleneck_analysis"] = bottleneck_result

                print(
                    "Bottleneck analysis completed."
                )

            except Exception as exc:

                print(
                    "\nBottleneck Analysis Agent failed."
                )
                print(
                    f"Error: {exc}"
                )

                traceback.print_exc()

                result["bottleneck_analysis"] = {
                    "status": "FAILED",
                    "error": str(exc),
                }

                if self.optimization_agent is None:
                    raise

        # ---------------------------------------------------------
        # Optimization recommendations
        # ---------------------------------------------------------

        if self.optimization_agent is not None:

            try:

                print(
                    "\nRunning optimization analysis..."
                )

                optimization_result = (
                    self._call_optimization_agent(
                        self.optimization_agent,
                        bottleneck_result,
                    )
                )

                result["optimization"] = (
                    optimization_result
                )

                print(
                    "Optimization analysis completed."
                )

            except Exception as exc:

                print(
                    "\nOptimization Agent failed."
                )
                print(
                    f"Error: {exc}"
                )

                traceback.print_exc()

                result["optimization"] = {
                    "status": "FAILED",
                    "error": str(exc),
                }

                raise

        result["status"] = "SUCCESS"

        self._print_process_intelligence_summary(
            result
        )

        return result

    # =========================================================
    # BOTTLENECK AGENT ADAPTER
    # =========================================================

    @staticmethod
    def _call_analysis_agent(agent):
        """
        Call a bottleneck-analysis agent across supported interfaces.
        """

        for method_name in (
            "analyze",
            "run",
            "execute",
        ):

            method = getattr(
                agent,
                method_name,
                None,
            )

            if callable(method):
                return method()

        raise AttributeError(
            "BottleneckAnalysisAgent must provide "
            "analyze(), run(), or execute()."
        )

    # =========================================================
    # OPTIMIZATION AGENT ADAPTER
    # =========================================================

    def _call_optimization_agent(
        self,
        agent,
        bottleneck_result,
    ):
        """
        Call the optimization agent defensively.

        The project has evolved through several small agent interfaces, so
        this adapter supports the common method names without changing the
        optimization agent itself.
        """

        import inspect

        candidate_methods = (
            "optimize",
            "generate_recommendations",
            "recommend",
            "analyze",
            "run",
            "execute",
        )

        for method_name in candidate_methods:

            method = getattr(
                agent,
                method_name,
                None,
            )

            if not callable(method):
                continue

            try:

                signature = inspect.signature(
                    method
                )

                params = [
                    p
                    for p in signature.parameters.values()
                    if p.name != "self"
                    and p.kind
                    in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                ]

                # -------------------------------------------------
                # Method requires no arguments
                # -------------------------------------------------

                if not params:
                    return method()

                # -------------------------------------------------
                # Method requires one or more arguments
                # -------------------------------------------------

                first = params[0].name.lower()

                if any(
                    token in first
                    for token in (
                        "bottleneck",
                        "analysis",
                        "result",
                        "input",
                        "data",
                        "insight",
                    )
                ):

                    return method(
                        **{
                            params[0].name:
                                bottleneck_result
                        }
                    )

                return method(
                    bottleneck_result
                )

            except TypeError:

                # Try the next supported interface.
                continue

        raise AttributeError(
            "OptimizationAgent must provide an "
            "optimization/recommendation method such as "
            "optimize() or generate_recommendations()."
        )

    # =========================================================
    # PROCESS INTELLIGENCE SUMMARY
    # =========================================================

    @staticmethod
    def _print_process_intelligence_summary(
        result
    ):
        """
        Print a compact summary suitable for terminal evidence capture.
        """

        analysis = result.get(
            "bottleneck_analysis"
        )

        optimization = result.get(
            "optimization"
        )

        print("\n" + "-" * 60)
        print("PROCESS INTELLIGENCE SUMMARY")
        print("-" * 60)

        if isinstance(
            analysis,
            dict,
        ):

            for key in (
                "events",
                "cases",
                "activities",
                "transitions",
                "potential_bottlenecks",
                "potential_deviations",
                "activities_with_rework",
                "longest_case_duration_hours",
            ):

                if key in analysis:
                    print(
                        f"{key:<34}: "
                        f"{analysis[key]}"
                    )

        if isinstance(
            optimization,
            dict,
        ):

            for key in (
                "recommendations_generated",
                "high_priority",
                "medium_priority",
                "low_priority",
            ):

                if key in optimization:
                    print(
                        f"{key:<34}: "
                        f"{optimization[key]}"
                    )

            recommendations = optimization.get(
                "recommendations"
            )

            if isinstance(
                recommendations,
                list,
            ):

                print(
                    "\nTop optimization recommendations:"
                )

                for item in recommendations[:5]:

                    if isinstance(
                        item,
                        dict,
                    ):

                        code = item.get(
                            "id",
                            item.get(
                                "code",
                                "",
                            ),
                        )

                        title = item.get(
                            "title",
                            item.get(
                                "category",
                                "",
                            ),
                        )

                        activity = item.get(
                            "activity",
                            "",
                        )

                        print(
                            f"  {code:<10} "
                            f"{title}"
                            + (
                                f" | {activity}"
                                if activity
                                else ""
                            )
                        )

                    else:

                        print(
                            f"  - {item}"
                        )

    # =========================================================
    # PROCESS DISCOVERY
    # =========================================================

    def _run_discovery(
        self,
        iteration: int,
    ):

        # -----------------------------------------------------
        # FIRST ITERATION
        # -----------------------------------------------------

        if iteration == 1:

            print(
                "\nRunning multi-algorithm "
                "process discovery..."
            )

            return (
                self.discovery_agent
                .discover_all()
            )

        # -----------------------------------------------------
        # SUBSEQUENT ITERATIONS
        # -----------------------------------------------------

        adaptation = (
            self.current_adaptation
            or {}
        )

        strategy = (
            adaptation.get(
                "strategy"
            )
            if isinstance(
                adaptation,
                dict,
            )
            else None
        )

        print(
            "\nPrevious adaptation strategy:",
            strategy,
        )

        # -----------------------------------------------------
        # Preferred adaptive API
        # -----------------------------------------------------

        if hasattr(
            self.discovery_agent,
            "discover_adaptive",
        ):

            print(
                "\nRunning adaptive process discovery..."
            )

            return (
                self.discovery_agent
                .discover_adaptive(
                    adaptation=adaptation
                )
            )

        # -----------------------------------------------------
        # Strategy-based API
        # -----------------------------------------------------

        if hasattr(
            self.discovery_agent,
            "discover_with_strategy",
        ):

            print(
                "\nRunning strategy-based discovery..."
            )

            return (
                self.discovery_agent
                .discover_with_strategy(
                    strategy=strategy
                )
            )

        # -----------------------------------------------------
        # Precision-oriented API
        # -----------------------------------------------------

        if (
            strategy == "precision_oriented"
            and hasattr(
                self.discovery_agent,
                "discover_precision_oriented",
            )
        ):

            print(
                "\nRunning precision-oriented discovery..."
            )

            return (
                self.discovery_agent
                .discover_precision_oriented()
            )

        # -----------------------------------------------------
        # Configure agent if possible
        # -----------------------------------------------------

        self._configure_discovery(
            adaptation
        )

        # -----------------------------------------------------
        # Fallback
        # -----------------------------------------------------

        print(
            "\nAdaptive discovery API not available."
        )

        print(
            "Running multi-algorithm discovery as fallback."
        )

        return (
            self.discovery_agent
            .discover_all()
        )

    # =========================================================
    # CONFIGURE DISCOVERY
    # =========================================================

    def _configure_discovery(
        self,
        adaptation,
    ):

        if not isinstance(
            adaptation,
            dict,
        ):
            return

        if hasattr(
            self.discovery_agent,
            "set_adaptation",
        ):

            self.discovery_agent.set_adaptation(
                adaptation
            )

            return

        if hasattr(
            self.discovery_agent,
            "configure",
        ):

            self.discovery_agent.configure(
                adaptation
            )

            return

        # Direct attribute fallback.

        try:

            self.discovery_agent.current_adaptation = (
                adaptation
            )

        except Exception:
            pass

    # =========================================================
    # MODEL EVALUATION
    # =========================================================

    def _evaluate_models(
        self,
        discovered_models,
    ):

        print(
            "\nEvaluating all discovered models..."
        )

        if hasattr(
            self.evaluation_agent,
            "evaluate_models",
        ):

            return (
                self.evaluation_agent
                .evaluate_models(
                    discovered_models
                )
            )

        if hasattr(
            self.evaluation_agent,
            "evaluate",
        ):

            return (
                self.evaluation_agent
                .evaluate(
                    discovered_models
                )
            )

        raise AttributeError(
            "EvaluationAgent must provide "
            "evaluate() or evaluate_models()."
        )

    # =========================================================
    # MODEL SELECTION
    # =========================================================

    def _select_model(
        self,
        evaluations,
    ):

        from src.agents.model_selection_agent import (
            ModelSelectionAgent
        )

        # IMPORTANT:
        #
        # The current ModelSelectionAgent expects
        # evaluations during initialization.
        #
        # Do NOT pass weights=... here.

        self.selection_agent = (
            ModelSelectionAgent(
                evaluations
            )
        )

        return (
            self.selection_agent
            .select()
        )

    # =========================================================
    # SELECTED MODEL NAME
    # =========================================================

    @staticmethod
    def _get_selected_name(
        selected_model,
    ) -> str:

        if selected_model is None:
            raise ValueError(
                "ModelSelectionAgent returned None."
            )

        if isinstance(
            selected_model,
            str,
        ):

            value = selected_model.strip()

            if value:
                return value

        if isinstance(
            selected_model,
            dict,
        ):

            candidate_keys = [
                "model",
                "name",
                "model_name",
                "selected_model",
                "selected_model_name",
                "best_model",
                "selected",
            ]

            for key in candidate_keys:

                if key not in selected_model:
                    continue

                value = selected_model[key]

                if isinstance(
                    value,
                    str,
                ):

                    value = value.strip()

                    if value:
                        return value

                if isinstance(
                    value,
                    dict,
                ):

                    try:

                        return (
                            APMOAOrchestrator
                            ._get_selected_name(
                                value
                            )
                        )

                    except ValueError:
                        pass

        # Object attributes.

        attributes = [
            "model",
            "name",
            "model_name",
            "selected_model",
            "selected_model_name",
            "best_model",
        ]

        for attribute in attributes:

            if not hasattr(
                selected_model,
                attribute,
            ):
                continue

            value = getattr(
                selected_model,
                attribute,
            )

            if isinstance(
                value,
                str,
            ):

                value = value.strip()

                if value:
                    return value

            if isinstance(
                value,
                dict,
            ):

                try:

                    return (
                        APMOAOrchestrator
                        ._get_selected_name(
                            value
                        )
                    )

                except ValueError:
                    pass

        # Last-resort textual matching.

        text = str(
            selected_model
        )

        known_names = [
            "Inductive Miner",
            "Alpha Miner",
            "Alpha Plus Miner",
            "Heuristics Miner",
            "ILP Miner",
            "Genetic Miner",
            "Split Miner",
            "DFG",
        ]

        for name in known_names:

            if name.lower() in text.lower():
                return name

        raise ValueError(
            "\nCould not determine selected model name.\n"
            f"Type : {type(selected_model).__name__}\n"
            f"Value: {selected_model}\n"
        )

    # =========================================================
    # SELECTED MODEL EVALUATION
    # =========================================================

    @staticmethod
    def _get_selected_evaluation(
        evaluations,
        selected_model,
        selected_name,
    ) -> Dict[str, Any]:

        """
        Resolve the evaluation row for the selected model.

        ModelSelectionAgent may return a dictionary while EvaluationAgent
        returns a pandas DataFrame. This method normalizes both forms and
        performs exact, case-insensitive, and whitespace-normalized matching.
        """

        import pandas as pd

        target = str(
            selected_name or ""
        ).strip()

        if not target:
            raise ValueError(
                "Selected model name is empty."
            )

        # -----------------------------------------------------
        # Evaluation as DataFrame
        # -----------------------------------------------------

        if isinstance(
            evaluations,
            pd.DataFrame,
        ):

            if evaluations.empty:
                raise ValueError(
                    "Evaluation DataFrame is empty."
                )

            model_column = None

            for column in evaluations.columns:

                normalized = (
                    str(column)
                    .strip()
                    .lower()
                )

                if normalized in {
                    "model",
                    "model_name",
                    "algorithm",
                    "algorithm_name",
                    "name",
                }:

                    model_column = column
                    break

            if model_column is None:

                raise ValueError(
                    "Could not find a model column "
                    "in the evaluation DataFrame. "
                    f"Columns: {list(evaluations.columns)}"
                )

            values = (
                evaluations[model_column]
                .astype(str)
                .str.strip()
            )

            matches = evaluations[
                values == target
            ]

            if matches.empty:

                matches = evaluations[
                    values.str.lower()
                    == target.lower()
                ]

            if matches.empty:

                def normalize(value):

                    return " ".join(
                        str(value)
                        .strip()
                        .lower()
                        .replace("_", " ")
                        .replace("-", " ")
                        .split()
                    )

                matches = evaluations[
                    values.apply(normalize)
                    == normalize(target)
                ]

            if matches.empty:

                available = values.tolist()

                raise ValueError(
                    "\nCould not find evaluation "
                    "for selected model."
                    f"\nSelected model : {target}"
                    "\nEvaluations type: DataFrame"
                    f"\nAvailable models: {available}"
                )

            result = (
                matches
                .iloc[0]
                .to_dict()
            )

            result["model"] = target

            return result

        # -----------------------------------------------------
        # Evaluation as list of dictionaries
        # -----------------------------------------------------

        if isinstance(
            evaluations,
            list,
        ):

            for evaluation in evaluations:

                if not isinstance(
                    evaluation,
                    dict,
                ):
                    continue

                model_name = (
                    evaluation.get("model")
                    or evaluation.get("name")
                    or evaluation.get("model_name")
                )

                if (
                    model_name is not None
                    and str(model_name)
                    .strip()
                    .lower()
                    == target.lower()
                ):

                    result = dict(
                        evaluation
                    )

                    result["model"] = target

                    return result

        # -----------------------------------------------------
        # Evaluation as dictionary
        # -----------------------------------------------------

        if isinstance(
            evaluations,
            dict,
        ):

            for key, value in evaluations.items():

                if (
                    isinstance(
                        key,
                        str,
                    )
                    and key.strip().lower()
                    == target.lower()
                ):

                    if isinstance(
                        value,
                        dict,
                    ):

                        result = dict(
                            value
                        )

                        result["model"] = target

                        return result

            for key in (
                "evaluations",
                "results",
                "model_evaluations",
                "models",
                "evaluation_results",
            ):

                if key in evaluations:

                    try:

                        return (
                            APMOAOrchestrator
                            ._get_selected_evaluation(
                                evaluations[key],
                                selected_model,
                                target,
                            )
                        )

                    except (
                        ValueError,
                        TypeError,
                    ):
                        pass

            model_name = (
                evaluations.get("model")
                or evaluations.get("name")
                or evaluations.get("model_name")
            )

            if (
                model_name is not None
                and str(model_name)
                .strip()
                .lower()
                == target.lower()
            ):

                return evaluations

            if (
                "fitness" in evaluations
                and "precision" in evaluations
            ):

                result = dict(
                    evaluations
                )

                result.setdefault(
                    "model",
                    target,
                )

                return result

        # -----------------------------------------------------
        # Selected model itself may contain metrics
        # -----------------------------------------------------

        if isinstance(
            selected_model,
            dict,
        ):

            if (
                "fitness" in selected_model
                and "precision" in selected_model
            ):

                return dict(
                    selected_model
                )

            for key in (
                "model",
                "selected_model",
                "best_model",
            ):

                nested = selected_model.get(
                    key
                )

                if (
                    isinstance(
                        nested,
                        dict,
                    )
                    and "fitness" in nested
                    and "precision" in nested
                ):

                    return dict(
                        nested
                    )

            nested = selected_model.get(
                "selected_evaluation"
            )

            if isinstance(
                nested,
                dict,
            ):

                result = dict(
                    nested
                )

                result.setdefault(
                    "model",
                    target,
                )

                return result

        raise ValueError(
            "\nCould not find evaluation "
            "for selected model."
            f"\nSelected model : {target}"
            f"\nEvaluations type: "
            f"{type(evaluations).__name__}"
            f"\nEvaluations     : "
            f"{APMOAOrchestrator._safe_repr(evaluations)}\n"
        )

    # =========================================================
    # ACTION TYPE
    # =========================================================

    @staticmethod
    def _get_action_type(
        adaptation,
    ) -> str:

        if not isinstance(
            adaptation,
            dict,
        ):
            return "unknown"

        action = (
            adaptation.get("action_type")
            or adaptation.get("action")
            or adaptation.get("decision")
        )

        if action is None:
            return "unknown"

        return str(
            action
        ).strip().lower()

    # =========================================================
    # COUNT MODELS
    # =========================================================

    @staticmethod
    def _count_models(
        discovered_models,
    ) -> int:

        if isinstance(
            discovered_models,
            list,
        ):

            return len(
                discovered_models
            )

        if isinstance(
            discovered_models,
            dict,
        ):

            # If dictionary is itself one model object.

            if (
                "net" in discovered_models
                or "petri_net" in discovered_models
                or "model" in discovered_models
            ):

                return 1

            # Common containers.

            for key in [
                "models",
                "discovered_models",
                "results",
            ]:

                value = discovered_models.get(
                    key
                )

                if isinstance(
                    value,
                    (list, dict),
                ):

                    return len(
                        value
                    )

            return len(
                discovered_models
            )

        return 1

    # =========================================================
    # PRINT REASONING
    # =========================================================

    @staticmethod
    def _print_reasoning(
        reasoning,
    ):

        if not isinstance(
            reasoning,
            dict,
        ):

            print(
                "\nReasoning result:"
            )

            print(
                reasoning
            )

            return

        print("\n" + "-" * 60)
        print("REASONING RESULT")
        print("-" * 60)

        for key in [
            "observation",
            "diagnosis",
            "evidence",
            "recommended_action",
            "reasoning",
            "confidence",
        ]:

            if key not in reasoning:
                continue

            value = reasoning[key]

            print(
                f"\n{key}:"
            )

            if isinstance(
                value,
                list,
            ):

                for item in value:

                    print(
                        f"  • {item}"
                    )

            else:

                print(
                    f"  {value}"
                )

    # =========================================================
    # FORMAT EVALUATION
    # =========================================================

    @staticmethod
    def _format_evaluation(
        evaluation,
    ) -> str:

        if not isinstance(
            evaluation,
            dict,
        ):

            return str(
                evaluation
            )

        metrics = [
            "fitness",
            "precision",
            "generalization",
            "simplicity",
            "selection_score",
        ]

        lines = []

        for metric in metrics:

            if metric not in evaluation:
                continue

            value = evaluation[
                metric
            ]

            try:

                if isinstance(
                    value,
                    (float, int),
                ):

                    lines.append(
                        f"  {metric:<18}: "
                        f"{float(value):.4f}"
                    )

                else:

                    lines.append(
                        f"  {metric:<18}: "
                        f"{value}"
                    )

            except Exception:

                lines.append(
                    f"  {metric:<18}: "
                    f"{value}"
                )

        return (
            "\n".join(lines)
            if lines
            else str(evaluation)
        )

    # =========================================================
    # FORMAT ADAPTATION
    # =========================================================

    @staticmethod
    def _format_adaptation(
        adaptation,
    ) -> str:

        if not isinstance(
            adaptation,
            dict,
        ):

            return str(
                adaptation
            )

        lines = []

        action = (
            adaptation.get("action_type")
            or adaptation.get("action")
            or adaptation.get("decision")
        )

        strategy = adaptation.get(
            "strategy"
        )

        reason = adaptation.get(
            "reason"
        )

        parameters = adaptation.get(
            "parameters"
        )

        if action is not None:

            lines.append(
                f"  Action   : {action}"
            )

        if strategy is not None:

            lines.append(
                f"  Strategy : {strategy}"
            )

        if reason is not None:

            lines.append(
                f"  Reason   : {reason}"
            )

        if parameters is not None:

            lines.append(
                f"  Parameters: {parameters}"
            )

        return (
            "\n".join(lines)
            if lines
            else str(adaptation)
        )

    # =========================================================
    # FINAL SUMMARY
    # =========================================================

    def _print_final_summary(
        self,
        result,
    ):

        print("\n" + "=" * 70)

        print(
            "APMOA AUTONOMOUS WORKFLOW COMPLETED"
        )

        print("=" * 70)

        print(
            "\nIterations:",
            result.get("iterations"),
        )

        print(
            "Final model:",
            result.get("final_model"),
        )

        final_evaluation = result.get(
            "final_evaluation"
        )

        if final_evaluation:

            print(
                "\nFinal evaluation:"
            )

            print(
                self._format_evaluation(
                    final_evaluation
                )
            )

        final_adaptation = result.get(
            "final_adaptation"
        )

        if final_adaptation:

            print(
                "\nFinal adaptation:"
            )

            print(
                self._format_adaptation(
                    final_adaptation
                )
            )

        if self.history:

            print(
                "\nAutonomous history:"
            )

            for record in self.history:

                print(
                    f"  Iteration "
                    f"{record['iteration']}: "
                    f"{record['selected_model']}"
                )

        print(
            "\n" + "=" * 70
        )

    # =========================================================
    # SAFE REPRESENTATION
    # =========================================================

    @staticmethod
    def _safe_repr(
        value,
    ) -> str:

        try:

            return json.dumps(
                value,
                indent=2,
                default=str,
            )

        except Exception:

            return str(
                value
            )


# =============================================================
# MAIN
# =============================================================

def main(
    log_path=None
):

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

    from src.agents.reasoning_agent import (
        ReasoningAgent
    )

    from src.agents.adaptation_agent import (
        AdaptationAgent
    )

    from src.agents.bottleneck_analysis_agent import (
        BottleneckAnalysisAgent
    )

    from src.agents.optimization_agent import (
        OptimizationAgent
    )

    # ---------------------------------------------------------
    # CONFIGURATION
    # ---------------------------------------------------------

    if log_path is None:

        log_path = (
            "data/raw/Travel/"
            "PermitLog.xes.gz"
        )

    LOG_PATH = str(
        log_path
    )

    MAX_ITERATIONS = 3

    print("\n" + "=" * 70)

    print(
        "APMOA - AUTONOMOUS PROCESS MINING SYSTEM"
    )

    print("=" * 70)

    # ---------------------------------------------------------
    # LOAD EVENT LOG
    # ---------------------------------------------------------

    print(
        f"\nLoading event log: {LOG_PATH}"
    )

    loader = EventLogLoader(
        LOG_PATH
    )

    log = loader.load()

    print(
        f"Loaded {len(log):,} events."
    )

    # ---------------------------------------------------------
    # INITIALIZE AGENTS
    # ---------------------------------------------------------

    print(
        "\nInitializing APMOA agents..."
    )

    data_analysis_agent = (
        DataAnalysisAgent(
            log
        )
    )

    discovery_agent = (
        ProcessDiscoveryAgent(
            log=log
        )
    )

    evaluation_agent = (
        ModelEvaluationAgent(
            log
        )
    )

    reasoning_agent = (
        ReasoningAgent()
    )

    adaptation_agent = (
        AdaptationAgent()
    )

    bottleneck_agent = (
        BottleneckAnalysisAgent(
            log
        )
    )

    optimization_agent = (
        OptimizationAgent()
    )

    print(
        "All agents initialized successfully."
    )

    # ---------------------------------------------------------
    # CREATE ORCHESTRATOR
    # ---------------------------------------------------------

    experiment_manager = ExperimentManager()

    orchestrator = (
        APMOAOrchestrator(
            data_analysis_agent=data_analysis_agent,
            discovery_agent=discovery_agent,
            evaluation_agent=evaluation_agent,
            selection_agent=None,
            reasoning_agent=reasoning_agent,
            adaptation_agent=adaptation_agent,
            bottleneck_agent=bottleneck_agent,
            optimization_agent=optimization_agent,
            log=log,
            max_iterations=MAX_ITERATIONS,
            experiment_manager=experiment_manager,
            dataset_name=Path(LOG_PATH).stem,
            dataset_path=LOG_PATH,
        )
    )

    # ---------------------------------------------------------
    # RUN
    # ---------------------------------------------------------

    result = (
        orchestrator.run()
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "APMOA EXECUTION FINISHED"
    )

    print(
        "Final selected model:",
        result.get("final_model"),
    )

    print(
        "Total iterations:",
        result.get("iterations"),
    )

    print(
        "Experiment ID:",
        result.get("experiment_id"),
    )

    print(
        "=" * 70
    )

    return result


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run the APMOA autonomous "
            "process-mining pipeline."
        )
    )

    parser.add_argument(
        "--log-path",
        default=None,
        help=(
            "Path to the event log "
            "(XES/XES.GZ/CSV/XLSX)."
        ),
    )

    args = parser.parse_args()

    main(
        log_path=args.log_path
    )