"""
APMOA - Adaptation Agent

Converts the Reasoning Agent recommendation and measured
process-model quality into a deterministic adaptation decision
for the autonomous process-mining loop.

Core responsibility:

    Evaluation
        |
        v
    Quality Policy
        |
        +-------------------------+
        |                         |
    Quality OK              Quality NOT OK
        |                         |
        v                         v
   KEEP_MODEL                REDISCOVER
                                  |
                                  v
                         Select next strategy
                                  |
                                  v
                         Select next algorithm
                                  |
                                  v
                         Next iteration

Important design principle:

    Process-model metrics are authoritative for quality checks.
    The LLM provides contextual reasoning and recommendations.
    RAG provides supporting process-mining knowledge.

The Adaptation Agent is deterministic and does not call the LLM.

Supported discovery algorithms:
    - Inductive Miner
    - Alpha Miner
    - Alpha+ Miner
    - Heuristics Miner
    - Split Miner

Supported adaptation strategies:
    - precision_oriented
    - fitness_oriented
    - simplicity_oriented
    - balanced
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


class AdaptationAgent:
    """
    Translate model-quality evidence and reasoning into
    a deterministic adaptation decision.
    """

    # =========================================================
    # SUPPORTED ALGORITHMS
    # =========================================================

    SUPPORTED_ALGORITHMS = (
        "Inductive Miner",
        "Alpha Miner",
        "Alpha+ Miner",
        "Heuristics Miner",
        "Split Miner",
    )

    # =========================================================
    # TERMINAL ACTIONS
    # =========================================================

    TERMINAL_ACTIONS = {
        "keep_model",
        "keep",
        "stop",
        "accept",
        "accepted",
        "no_change",
        "no_change_needed",
    }

    # =========================================================
    # REDISCOVERY ACTIONS
    # =========================================================

    REDISCOVERY_ACTIONS = {
        "rediscover",
        "try_precision_oriented",
        "try_fitness_oriented",
        "try_balanced",
        "try_simplicity_oriented",
        "improve_model",
        "adapt",
    }

    # =========================================================
    # DEFAULT QUALITY POLICY
    # =========================================================
    #
    # These are CONFIGURATION values for the prototype.
    # They are NOT claimed to be universal process-mining
    # thresholds.
    #
    # They can later be tuned experimentally.
    #
    # =========================================================

    DEFAULT_QUALITY_POLICY = {
        "fitness_min": 0.80,
        "precision_min": 0.50,
        "generalization_min": 0.70,
        "simplicity_min": 0.40,
    }

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        quality_policy: Optional[
            Dict[str, float]
        ] = None,
    ) -> None:

        self.quality_policy = (
            dict(
                self.DEFAULT_QUALITY_POLICY
            )
        )

        if quality_policy:

            self.quality_policy.update(
                quality_policy
            )

        self.last_decision: Optional[
            Dict[str, Any]
        ] = None

    # =========================================================
    # NORMALIZATION HELPERS
    # =========================================================

    @staticmethod
    def _normalize(
        value: Any,
    ) -> str:

        if value is None:

            return ""

        return (
            str(value)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

    # =========================================================
    # MODEL NAME
    # =========================================================

    @staticmethod
    def _model_name(
        evaluation: Any,
    ) -> Optional[str]:

        """
        Extract selected model name from
        an evaluation record.
        """

        if isinstance(
            evaluation,
            dict,
        ):

            for key in (
                "model",
                "name",
                "model_name",
                "algorithm",
                "selected_model",
            ):

                value = evaluation.get(
                    key
                )

                if (
                    isinstance(
                        value,
                        str,
                    )
                    and value.strip()
                ):

                    return value.strip()

        # pandas Series / similar
        # mapping-like objects

        try:

            for key in (
                "model",
                "name",
                "model_name",
                "algorithm",
                "selected_model",
            ):

                value = evaluation.get(
                    key
                )

                if (
                    isinstance(
                        value,
                        str,
                    )
                    and value.strip()
                ):

                    return value.strip()

        except Exception:

            pass

        return None

    # =========================================================
    # METRIC
    # =========================================================

    @staticmethod
    def _metric(
        evaluation: Any,
        key: str,
    ) -> Optional[float]:

        try:

            value = evaluation.get(
                key
            )

            if value is None:

                return None

            return float(
                value
            )

        except (
            AttributeError,
            TypeError,
            ValueError,
        ):

            return None

    # =========================================================
    # REASONING ACTION
    # =========================================================

    @staticmethod
    def _reasoning_action(
        reasoning: Any,
    ) -> str:

        if not isinstance(
            reasoning,
            dict,
        ):

            return ""

        for key in (
            "recommended_action",
            "action_type",
            "action",
            "decision",
        ):

            value = reasoning.get(
                key
            )

            if value is not None:

                return (
                    AdaptationAgent
                    ._normalize(value)
                )

        return ""

    # =========================================================
    # REASONING STRATEGY
    # =========================================================

    @staticmethod
    def _reasoning_strategy(
        reasoning: Any,
    ) -> str:

        if not isinstance(
            reasoning,
            dict,
        ):

            return ""

        for key in (
            "strategy",
            "recommended_strategy",
            "adaptation_strategy",
        ):

            value = reasoning.get(
                key
            )

            if value:

                return (
                    AdaptationAgent
                    ._normalize(value)
                )

        return ""

    # =========================================================
    # ATTEMPTED ALGORITHMS
    # =========================================================

    @staticmethod
    def _attempted_algorithms(
        reasoning: Any,
    ) -> list[str]:

        if not isinstance(
            reasoning,
            dict,
        ):

            return []

        value = (
            reasoning.get(
                "attempted_algorithms"
            )
            or reasoning.get(
                "attempted"
            )
            or []
        )

        if isinstance(
            value,
            str,
        ):

            return (
                [value.strip()]
                if value.strip()
                else []
            )

        if isinstance(
            value,
            Iterable,
        ):

            result = []

            for item in value:

                if (
                    isinstance(
                        item,
                        str,
                    )
                    and item.strip()
                ):

                    result.append(
                        item.strip()
                    )

            return result

        return []

    # =========================================================
    # QUALITY POLICY
    # =========================================================

    def _quality_check(
        self,
        evaluation: Any,
    ) -> Dict[str, Any]:

        """
        Compare measured model metrics against
        the configured quality policy.

        Returns:

            {
                "quality_ok": bool,
                "violations": [...],
                "metrics": {...},
                "policy": {...}
            }
        """

        fitness = self._metric(
            evaluation,
            "fitness",
        )

        precision = self._metric(
            evaluation,
            "precision",
        )

        generalization = self._metric(
            evaluation,
            "generalization",
        )

        simplicity = self._metric(
            evaluation,
            "simplicity",
        )

        metrics = {

            "fitness": fitness,

            "precision": precision,

            "generalization":
                generalization,

            "simplicity":
                simplicity,
        }

        violations = []

        fitness_min = self.quality_policy.get(
            "fitness_min"
        )

        precision_min = self.quality_policy.get(
            "precision_min"
        )

        generalization_min = (
            self.quality_policy.get(
                "generalization_min"
            )
        )

        simplicity_min = (
            self.quality_policy.get(
                "simplicity_min"
            )
        )

        if (
            fitness is not None
            and fitness_min is not None
            and fitness < fitness_min
        ):

            violations.append(
                {
                    "metric": "fitness",
                    "value": fitness,
                    "threshold": fitness_min,
                    "direction": "below",
                }
            )

        if (
            precision is not None
            and precision_min is not None
            and precision < precision_min
        ):

            violations.append(
                {
                    "metric": "precision",
                    "value": precision,
                    "threshold": precision_min,
                    "direction": "below",
                }
            )

        if (
            generalization is not None
            and generalization_min is not None
            and generalization < generalization_min
        ):

            violations.append(
                {
                    "metric": "generalization",
                    "value": generalization,
                    "threshold": generalization_min,
                    "direction": "below",
                }
            )

        if (
            simplicity is not None
            and simplicity_min is not None
            and simplicity < simplicity_min
        ):

            violations.append(
                {
                    "metric": "simplicity",
                    "value": simplicity,
                    "threshold": simplicity_min,
                    "direction": "below",
                }
            )

        return {

            "quality_ok":
                len(violations) == 0,

            "violations":
                violations,

            "metrics":
                metrics,

            "policy":
                dict(
                    self.quality_policy
                ),
        }

    # =========================================================
    # DETERMINE STRATEGY FROM QUALITY
    # =========================================================

    @staticmethod
    def _strategy_from_quality(
        quality_check: Dict[str, Any],
    ) -> str:

        """
        Select the adaptation strategy from
        the measured quality violations.

        Priority:

            precision
            fitness
            generalization
            simplicity
            balanced
        """

        violations = (
            quality_check.get(
                "violations",
                [],
            )
        )

        if not violations:

            return "balanced"

        metrics_with_violations = {
            item.get("metric")
            for item in violations
            if isinstance(
                item,
                dict,
            )
        }

        # Precision is prioritized because
        # excessive behavioral allowance is
        # a major process-model quality concern.

        if "precision" in metrics_with_violations:

            return "precision_oriented"

        if "fitness" in metrics_with_violations:

            return "fitness_oriented"

        if "generalization" in metrics_with_violations:

            return "balanced"

        if "simplicity" in metrics_with_violations:

            return "simplicity_oriented"

        return "balanced"

    # =========================================================
    # STRATEGY / ALGORITHM SELECTION
    # =========================================================

    def _select_algorithm(
        self,
        evaluation: Any,
        reasoning: Any,
        strategy: str,
        attempted: list[str],
    ) -> tuple[
        str,
        str,
        str,
        Dict[str, float],
    ]:

        """
        Select the next algorithm for a genuine
        rediscovery request.
        """

        current = self._model_name(
            evaluation
        )

        precision = self._metric(
            evaluation,
            "precision",
        )

        fitness = self._metric(
            evaluation,
            "fitness",
        )

        generalization = self._metric(
            evaluation,
            "generalization",
        )

        simplicity = self._metric(
            evaluation,
            "simplicity",
        )

        # -----------------------------------------------------
        # Precision-oriented
        # -----------------------------------------------------

        if strategy in {
            "precision_oriented",
            "precision",
        }:

            target = "precision"

            candidates = [

                "Heuristics Miner",

                "Split Miner",

                "Inductive Miner",

                "Alpha+ Miner",

                "Alpha Miner",
            ]

            parameters = {

                "dependency_threshold":
                    0.7,

                "and_threshold":
                    0.65,

                "loop_two_threshold":
                    0.5,
            }

        # -----------------------------------------------------
        # Fitness-oriented
        # -----------------------------------------------------

        elif strategy in {
            "fitness_oriented",
            "fitness",
        }:

            target = "fitness"

            candidates = [

                "Inductive Miner",

                "Split Miner",

                "Heuristics Miner",

                "Alpha+ Miner",

                "Alpha Miner",
            ]

            parameters = {

                "dependency_threshold":
                    0.5,

                "and_threshold":
                    0.65,

                "loop_two_threshold":
                    0.5,
            }

        # -----------------------------------------------------
        # Simplicity-oriented
        # -----------------------------------------------------

        elif strategy in {
            "simplicity_oriented",
            "simplicity",
        }:

            target = "simplicity"

            candidates = [

                "Alpha Miner",

                "Alpha+ Miner",

                "Inductive Miner",

                "Split Miner",

                "Heuristics Miner",
            ]

            parameters = {

                "dependency_threshold":
                    0.5,

                "and_threshold":
                    0.65,

                "loop_two_threshold":
                    0.5,
            }

        # -----------------------------------------------------
        # Balanced
        # -----------------------------------------------------

        else:

            target = "balanced"

            candidates = [

                "Split Miner",

                "Heuristics Miner",

                "Inductive Miner",

                "Alpha+ Miner",

                "Alpha Miner",
            ]

            parameters = {

                "dependency_threshold":
                    0.6,

                "and_threshold":
                    0.65,

                "loop_two_threshold":
                    0.5,
            }

        # -----------------------------------------------------
        # Explicit algorithm from reasoning
        # -----------------------------------------------------

        explicit = None

        if isinstance(
            reasoning,
            dict,
        ):

            for key in (
                "next_algorithm",
                "recommended_algorithm",
                "target_algorithm",
            ):

                value = reasoning.get(
                    key
                )

                if not isinstance(
                    value,
                    str,
                ):

                    continue

                for candidate in (
                    self.SUPPORTED_ALGORITHMS
                ):

                    if (
                        self._normalize(
                            candidate
                        )
                        ==
                        self._normalize(
                            value
                        )
                    ):

                        explicit = candidate

                        break

                if explicit:

                    break

        # -----------------------------------------------------
        # Attempted algorithms
        # -----------------------------------------------------

        attempted_normalized = {

            self._normalize(
                item
            )

            for item in attempted
        }

        # -----------------------------------------------------
        # Select candidate
        # -----------------------------------------------------

        if (
            explicit
            and
            self._normalize(
                explicit
            )
            not in attempted_normalized
            and
            self._normalize(
                explicit
            )
            != self._normalize(
                current
            )
        ):

            next_algorithm = explicit

        else:

            next_algorithm = next(
                (
                    candidate

                    for candidate
                    in candidates

                    if (
                        self._normalize(
                            candidate
                        )
                        !=
                        self._normalize(
                            current
                        )
                    )
                    and
                    self._normalize(
                        candidate
                    )
                    not in attempted_normalized
                ),
                None,
            )

        # -----------------------------------------------------
        # All alternatives exhausted
        # -----------------------------------------------------

        if next_algorithm is None:

            next_algorithm = next(
                (
                    candidate

                    for candidate
                    in candidates

                    if (
                        self._normalize(
                            candidate
                        )
                        !=
                        self._normalize(
                            current
                        )
                    )
                ),
                candidates[0],
            )

        # -----------------------------------------------------
        # Explanation
        # -----------------------------------------------------

        reason = (

            f"The current model "
            f"({current or 'unknown'}) requires "
            f"a {target} improvement. "

            f"The Adaptation Agent selected "
            f"{next_algorithm} as the next "
            f"discovery candidate."
        )

        if (
            precision is not None
            and target == "precision"
        ):

            reason += (
                f" Current precision: "
                f"{precision:.4f}."
            )

        if (
            fitness is not None
            and target == "fitness"
        ):

            reason += (
                f" Current fitness: "
                f"{fitness:.4f}."
            )

        if (
            simplicity is not None
            and target == "simplicity"
        ):

            reason += (
                f" Current simplicity: "
                f"{simplicity:.4f}."
            )

        if (
            generalization is not None
            and target == "balanced"
        ):

            reason += (
                f" Current generalization: "
                f"{generalization:.4f}."
            )

        return (
            target,
            next_algorithm,
            reason,
            parameters,
        )

    # =========================================================
    # MAIN ADAPTATION
    # =========================================================

    def adapt(
        self,
        evaluation: Any,
        reasoning: Any,
        iteration: int = 1,
    ) -> Dict[str, Any]:

        """
        Convert evaluation + reasoning into
        a deterministic adaptation decision.

        Decision hierarchy:

        1. Inspect measured quality.
        2. If quality violates the configured policy,
           request rediscovery.
        3. Use LLM recommendation to refine the
           rediscovery strategy when appropriate.
        4. If quality is acceptable and LLM recommends
           keep_model, retain the model.
        5. If neither provides an actionable decision,
           safely retain the model.
        """

        print(
            "\n" + "=" * 60
        )

        print(
            "APMOA - ADAPTATION AGENT"
        )

        print(
            "=" * 60
        )

        print(
            f"\nCurrent iteration: "
            f"{iteration}"
        )

        current_model = (
            self._model_name(
                evaluation
            )
        )

        raw_action = (
            self._reasoning_action(
                reasoning
            )
        )

        strategy_from_reasoning = (
            self._reasoning_strategy(
                reasoning
            )
        )

        attempted = (
            self._attempted_algorithms(
                reasoning
            )
        )

        # =====================================================
        # QUALITY CHECK
        # =====================================================

        quality_check = (
            self._quality_check(
                evaluation
            )
        )

        print(
            "\n------------------------------------------------------------"
        )

        print(
            "QUALITY POLICY CHECK"
        )

        print(
            "------------------------------------------------------------"
        )

        print(
            f"Fitness minimum       : "
            f"{self.quality_policy['fitness_min']}"
        )

        print(
            f"Precision minimum     : "
            f"{self.quality_policy['precision_min']}"
        )

        print(
            f"Generalization minimum: "
            f"{self.quality_policy['generalization_min']}"
        )

        print(
            f"Simplicity minimum    : "
            f"{self.quality_policy['simplicity_min']}"
        )

        print(
            "\nMeasured metrics:"
        )

        for key, value in (
            quality_check[
                "metrics"
            ].items()
        ):

            print(
                f"  {key:<18}: "
                f"{value}"
            )

        violations = (
            quality_check[
                "violations"
            ]
        )

        if violations:

            print(
                "\nQuality violations:"
            )

            for violation in violations:

                print(
                    f"  - "
                    f"{violation['metric']}: "
                    f"{violation['value']:.4f} "
                    f"< "
                    f"{violation['threshold']:.4f}"
                )

        else:

            print(
                "\n✓ All configured quality "
                "requirements satisfied."
            )

        # =====================================================
        # DETERMINE QUALITY-BASED STRATEGY
        # =====================================================

        quality_strategy = (
            self._strategy_from_quality(
                quality_check
            )
        )

        # =====================================================
        # CASE 1 - QUALITY VIOLATION
        # =====================================================

        if violations:

            # -------------------------------------------------
            # Determine strategy
            # -------------------------------------------------

            if raw_action in {
                "try_precision_oriented",
            }:

                strategy = (
                    "precision_oriented"
                )

            elif raw_action in {
                "try_fitness_oriented",
            }:

                strategy = (
                    "fitness_oriented"
                )

            elif raw_action in {
                "try_simplicity_oriented",
            }:

                strategy = (
                    "simplicity_oriented"
                )

            elif raw_action in {
                "try_balanced",
            }:

                strategy = "balanced"

            elif strategy_from_reasoning:

                strategy = (
                    strategy_from_reasoning
                )

            else:

                strategy = (
                    quality_strategy
                )

            # -------------------------------------------------
            # Ensure strategy is valid
            # -------------------------------------------------

            if strategy in {
                "balanced_optimization",
                "balanced",
            }:

                strategy = "balanced"

            # -------------------------------------------------
            # Select next algorithm
            # -------------------------------------------------

            (
                target,
                next_algorithm,
                reason,
                parameters,
            ) = self._select_algorithm(

                evaluation=evaluation,

                reasoning=reasoning,

                strategy=strategy,

                attempted=attempted,
            )

            # -------------------------------------------------
            # Add policy evidence
            # -------------------------------------------------

            reason += (

                " The configured quality policy "
                "requires rediscovery because "
                "one or more quality thresholds "
                "were not satisfied."
            )

            decision = {

                "action_type":
                    "rediscover",

                "action":
                    "rediscover",

                "strategy":
                    strategy,

                "target":
                    target,

                "current_model":
                    current_model,

                "next_algorithm":
                    next_algorithm,

                "reason":
                    reason,

                "parameters":
                    parameters,

                "attempted":
                    attempted,

                "iteration":
                    iteration,

                "quality_check":
                    quality_check,

                "policy_triggered":
                    True,

                "reasoning_action":
                    raw_action,

            }

            self.last_decision = (
                decision
            )

            self._print_decision(
                decision
            )

            return decision

        # =====================================================
        # CASE 2 - QUALITY SATISFIED
        # =====================================================

        if raw_action in (
            self.TERMINAL_ACTIONS
        ):

            decision = {

                "action_type":
                    "keep_model",

                "action":
                    "keep_model",

                "strategy":
                    "none",

                "target":
                    "current_model",

                "current_model":
                    current_model,

                "next_algorithm":
                    None,

                "reason": (

                    "All configured quality "
                    "requirements are satisfied. "

                    f"Reasoning Agent recommended "
                    f"keep_model. "

                    f"The current "
                    f"{current_model or 'selected'} "
                    f"model will be retained."
                ),

                "parameters":
                    {},

                "attempted":
                    attempted,

                "iteration":
                    iteration,

                "quality_check":
                    quality_check,

                "policy_triggered":
                    False,

                "reasoning_action":
                    raw_action,
            }

            self.last_decision = (
                decision
            )

            self._print_decision(
                decision
            )

            return decision

        # =====================================================
        # CASE 3 - LLM REQUESTS REDISCOVERY
        #         EVEN THOUGH POLICY IS SATISFIED
        # =====================================================

        if (
            raw_action
            in self.REDISCOVERY_ACTIONS
        ):

            if raw_action == (
                "try_precision_oriented"
            ):

                strategy = (
                    "precision_oriented"
                )

            elif raw_action == (
                "try_fitness_oriented"
            ):

                strategy = (
                    "fitness_oriented"
                )

            elif raw_action == (
                "try_simplicity_oriented"
            ):

                strategy = (
                    "simplicity_oriented"
                )

            elif raw_action == (
                "try_balanced"
            ):

                strategy = "balanced"

            elif strategy_from_reasoning:

                strategy = (
                    strategy_from_reasoning
                )

            else:

                strategy = (
                    quality_strategy
                )

            if strategy == (
                "balanced_optimization"
            ):

                strategy = "balanced"

            (
                target,
                next_algorithm,
                reason,
                parameters,
            ) = self._select_algorithm(

                evaluation=evaluation,

                reasoning=reasoning,

                strategy=strategy,

                attempted=attempted,
            )

            reason += (

                " The quality policy is "
                "currently satisfied, but the "
                "Reasoning Agent explicitly "
                "requested rediscovery."
            )

            decision = {

                "action_type":
                    "rediscover",

                "action":
                    "rediscover",

                "strategy":
                    strategy,

                "target":
                    target,

                "current_model":
                    current_model,

                "next_algorithm":
                    next_algorithm,

                "reason":
                    reason,

                "parameters":
                    parameters,

                "attempted":
                    attempted,

                "iteration":
                    iteration,

                "quality_check":
                    quality_check,

                "policy_triggered":
                    False,

                "reasoning_action":
                    raw_action,
            }

            self.last_decision = (
                decision
            )

            self._print_decision(
                decision
            )

            return decision

        # =====================================================
        # CASE 4 - SAFE DEFAULT
        # =====================================================

        decision = {

            "action_type":
                "keep_model",

            "action":
                "keep_model",

            "strategy":
                "none",

            "target":
                "current_model",

            "current_model":
                current_model,

            "next_algorithm":
                None,

            "reason": (

                "No actionable adaptation "
                "recommendation was supplied "
                "and the configured quality "
                "requirements are satisfied. "

                "The current model is retained "
                "to avoid unnecessary rediscovery."
            ),

            "parameters":
                {},

            "attempted":
                attempted,

            "iteration":
                iteration,

            "quality_check":
                quality_check,

            "policy_triggered":
                False,

            "reasoning_action":
                raw_action,
        }

        self.last_decision = (
            decision
        )

        self._print_decision(
            decision
        )

        return decision

    # =========================================================
    # OUTPUT
    # =========================================================

    @staticmethod
    def _print_decision(
        decision: Dict[str, Any],
    ) -> None:

        print(
            "\n" + "-" * 60
        )

        print(
            "ADAPTATION DECISION"
        )

        print(
            "-" * 60
        )

        print(
            f"Action          : "
            f"{decision['action_type']}"
        )

        print(
            f"Strategy        : "
            f"{decision['strategy']}"
        )

        print(
            f"Target          : "
            f"{decision['target']}"
        )

        print(
            f"Current model   : "
            f"{decision['current_model']}"
        )

        print(
            f"Next algorithm  : "
            f"{decision['next_algorithm']}"
        )

        print(
            f"Reason          : "
            f"{decision['reason']}"
        )

        print(
            f"Parameters      : "
            f"{decision['parameters']}"
        )

        print(
            f"Attempted       : "
            f"{decision['attempted']}"
        )

        print(
            f"Policy triggered: "
            f"{decision.get('policy_triggered')}"
        )

        print(
            "\n" + "=" * 60
        )

        print(
            "ADAPTATION AGENT COMPLETED"
        )

        print(
            "=" * 60
        )


# =============================================================
# STANDALONE TESTS
# =============================================================

def main() -> None:

    """
    Standalone tests for:

        1. KEEP_MODEL
        2. REDISCOVERY
        3. Automatic precision-based rediscovery
        4. Automatic fitness-based rediscovery
    """

    print(
        "=" * 70
    )

    print(
        "APMOA - ADAPTATION AGENT TEST"
    )

    print(
        "=" * 70
    )

    # =========================================================
    # TEST 1 - KEEP MODEL
    # =========================================================

    print(
        "\nTEST 1 - KEEP MODEL"
    )

    agent = AdaptationAgent()

    keep_result = agent.adapt(

        evaluation={

            "model":
                "Split Miner",

            "fitness":
                0.9742,

            "precision":
                0.6367,

            "generalization":
                0.8564,

            "simplicity":
                0.5642,
        },

        reasoning={

            "recommended_action":
                "keep_model",

            "confidence":
                0.88,
        },

        iteration=1,
    )

    assert (
        keep_result[
            "action_type"
        ]
        ==
        "keep_model"
    )

    assert (
        keep_result[
            "current_model"
        ]
        ==
        "Split Miner"
    )

    assert (
        keep_result[
            "next_algorithm"
        ]
        is None
    )

    print(
        "\n✓ KEEP_MODEL TEST PASSED"
    )

    # =========================================================
    # TEST 2 - EXPLICIT REDISCOVERY
    # =========================================================

    print(
        "\nTEST 2 - EXPLICIT REDISCOVERY"
    )

    rediscover_result = agent.adapt(

        evaluation={

            "model":
                "Inductive Miner",

            "fitness":
                1.0,

            "precision":
                0.098,

            "generalization":
                0.8617,

            "simplicity":
                0.5947,
        },

        reasoning={

            "recommended_action":
                "try_precision_oriented",
        },

        iteration=1,
    )

    assert (
        rediscover_result[
            "action_type"
        ]
        ==
        "rediscover"
    )

    assert (
        rediscover_result[
            "strategy"
        ]
        ==
        "precision_oriented"
    )

    assert (
        rediscover_result[
            "current_model"
        ]
        ==
        "Inductive Miner"
    )

    assert (
        rediscover_result[
            "next_algorithm"
        ]
        ==
        "Heuristics Miner"
    )

    print(
        "✓ EXPLICIT REDISCOVERY TEST PASSED"
    )

    # =========================================================
    # TEST 3 - AUTOMATIC PRECISION VIOLATION
    # =========================================================

    print(
        "\nTEST 3 - AUTOMATIC PRECISION-BASED REDISCOVERY"
    )

    automatic_precision_result = (
        agent.adapt(

            evaluation={

                "model":
                    "Inductive Miner",

                "fitness":
                    1.0,

                "precision":
                    0.098,

                "generalization":
                    0.8617,

                "simplicity":
                    0.5947,
            },

            reasoning={
                "recommended_action":
                    "keep_model"
            },

            iteration=1,
        )
    )

    assert (
        automatic_precision_result[
            "action_type"
        ]
        ==
        "rediscover"
    )

    assert (
        automatic_precision_result[
            "strategy"
        ]
        ==
        "precision_oriented"
    )

    assert (
        automatic_precision_result[
            "next_algorithm"
        ]
        ==
        "Heuristics Miner"
    )

    assert (
        automatic_precision_result[
            "policy_triggered"
        ]
        is True
    )

    print(
        "✓ AUTOMATIC PRECISION REDISCOVERY TEST PASSED"
    )

    # =========================================================
    # TEST 4 - AUTOMATIC FITNESS VIOLATION
    # =========================================================

    print(
        "\nTEST 4 - AUTOMATIC FITNESS-BASED REDISCOVERY"
    )

    automatic_fitness_result = (
        agent.adapt(

            evaluation={

                "model":
                    "Alpha Miner",

                "fitness":
                    0.4199,

                "precision":
                    0.0,

                "generalization":
                    0.8553,

                "simplicity":
                    0.4190,
            },

            reasoning={
                "recommended_action":
                    "keep_model"
            },

            iteration=1,
        )
    )

    assert (
        automatic_fitness_result[
            "action_type"
        ]
        ==
        "rediscover"
    )

    # Precision is violated first, therefore
    # precision-oriented adaptation is expected.

    assert (
        automatic_fitness_result[
            "strategy"
        ]
        ==
        "precision_oriented"
    )

    assert (
        automatic_fitness_result[
            "action_type"
        ]
        ==
        "rediscover"
    )

    print(
        "✓ AUTOMATIC QUALITY-POLICY TEST PASSED"
    )

    # =========================================================
    # COMPLETE
    # =========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "ALL ADAPTATION AGENT TESTS PASSED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()