"""
APMOA - Reasoning Agent

Autonomous Process Mining using a Self-Adaptive
Multi-Agent Framework.

The Reasoning Agent:
    1. Analyzes process-log diagnostics.
    2. Compares ALL discovered process models.
    3. Identifies model-quality problems.
    4. Uses retrieved process knowledge when available.
    5. Determines the most appropriate next strategy.
    6. Provides evidence and confidence.
    7. Produces a structured result for the Adaptation Agent.

The Reasoning Agent does NOT modify or rediscover models.
The Adaptation Agent is responsible for executing adaptation.
"""

import json
import re
from typing import Any, Dict, List, Optional

from src.llm.groq_provider import GroqProvider

try:
    from src.rag.knowledge_base import APMOAKnowledgeBase
except Exception:
    APMOAKnowledgeBase = None


class ReasoningAgent:
    """
    APMOA LLM Reasoning Agent.

    The agent receives compact process-mining evidence,
    optionally retrieves relevant knowledge using RAG,
    and produces a structured reasoning decision.

    RAG is optional so the Reasoning Agent remains compatible
    with the existing APMOA pipeline.
    """

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        llm_provider=None,
        knowledge_base=None,
        rag_top_k: int = 3,
        knowledge_dir: str = "data/knowledge",
        auto_build_rag: bool = True,
    ):
        """
        Initialize the Reasoning Agent.

        Parameters
        ----------
        llm_provider:
            Optional LLM provider.
            If None, GroqProvider is used.

        knowledge_base:
            Optional APMOA knowledge base used for RAG.
            If None, RAG is disabled.

        rag_top_k:
            Number of knowledge documents/chunks to retrieve.
        """

        if llm_provider is None:
            try:
                llm_provider = GroqProvider()
            except Exception as e:
                print(f"Warning: {e} Using deterministic fallback.")
                llm_provider = None

        self.llm = llm_provider

        # -----------------------------------------------------
        # Persistent RAG knowledge base
        # -----------------------------------------------------
        self.knowledge_base = knowledge_base
        self.rag_top_k = max(1, int(rag_top_k))
        self.knowledge_dir = knowledge_dir
        self.rag_enabled = False
        self.rag_build_error = None

        # If the orchestrator does not inject a KB, automatically
        # connect to the persistent project knowledge store.
        if self.knowledge_base is None and auto_build_rag:
            if APMOAKnowledgeBase is None:
                self.rag_build_error = (
                    "APMOAKnowledgeBase could not be imported."
                )
            else:
                try:
                    print("\nInitializing persistent APMOA RAG knowledge base...")
                    self.knowledge_base = APMOAKnowledgeBase(
                        knowledge_dir=self.knowledge_dir
                    )
                    self.knowledge_base.build()
                    print("Persistent RAG knowledge base initialized successfully.")
                except Exception as exc:
                    self.rag_build_error = f"{type(exc).__name__}: {exc}"
                    self.knowledge_base = None
                    print("Warning: RAG knowledge base initialization failed.")
                    print(f"Error: {self.rag_build_error}")
                    print("Continuing with process-mining evidence only.")

        if self.knowledge_base is not None:
            self.rag_enabled = True
            print("RAG knowledge base connected to Reasoning Agent.")
        else:
            print(
                "RAG knowledge base not configured. "
                "Reasoning will use process-mining evidence only."
            )

    # =========================================================
    # SYSTEM PROMPT
    # =========================================================

    def _system_prompt(self) -> str:

        return """
You are the Reasoning Agent in APMOA:

Autonomous Process Mining using a Self-Adaptive
Multi-Agent Framework.

Your role is to reason over process-mining evidence
and determine what the autonomous system should do next.

You receive:

1. Process-log diagnostics.
2. Evaluation results from ALL discovered models.
3. The currently selected model.
4. Optional retrieved process knowledge from the APMOA
   knowledge base.

Available discovery algorithms may include:

- Inductive Miner
- Alpha Miner
- Alpha+ Miner
- Heuristics Miner
- Split Miner

Your responsibilities:

1. Analyze the process-log characteristics.
2. Compare ALL supplied discovery models.
3. Analyze:
   - fitness
   - precision
   - generalization
   - simplicity
   - selection score
4. Identify the most important model-quality issue.
5. Use retrieved knowledge as contextual guidance when it
   is supplied.
6. Determine whether the current model should be kept
   or whether rediscovery is appropriate.
7. Recommend a specific actionable strategy.
8. Explain the reasoning using supplied evidence and
   retrieved knowledge.
9. Provide a confidence score.

Important principles:

- Use the supplied process-mining evidence as the primary
  source for conclusions.
- Never invent metrics.
- Never invent models.
- Never assume Inductive Miner is always best.
- Always compare ALL supplied models.
- Consider fitness and precision together.
- Consider generalization and simplicity.
- High fitness with very low precision may indicate
  an overly permissive model.
- Low fitness may indicate poor representation
  of observed behavior.
- High precision with poor fitness is also problematic.
- A model with a better overall trade-off may be preferable.
- Retrieved knowledge provides contextual guidance but must
  not override actual process-mining measurements.
- Do not treat retrieved knowledge as evidence that a
  particular model actually achieved a metric.
- Repeated activities and process complexity may influence
  the choice of discovery strategy.
- The recommendation must be actionable by the
  Adaptation Agent.

Possible actions:

- keep_model
- rediscover

Possible strategies:

- keep_current
- precision_oriented
- fitness_oriented
- balanced
- heuristics
- split_miner
- inductive_noise_filtering
- alpha_plus
- alternative_model

Do NOT execute adaptation yourself.

Return ONLY valid JSON.

The JSON MUST contain exactly these fields:

{
    "observation": "...",
    "diagnosis": "...",
    "evidence": [
        "...",
        "..."
    ],
    "recommended_action": "...",
    "reasoning": "...",
    "confidence": 0.0
}

The confidence value must be between 0.0 and 1.0.
"""

    # =========================================================
    # SAFE FLOAT
    # =========================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: Optional[float] = None,
    ) -> Optional[float]:

        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return default

    # =========================================================
    # COMPACT DIAGNOSTICS
    # =========================================================

    def _compact_diagnostics(
        self,
        diagnostics: Any,
    ) -> Dict[str, Any]:

        if not isinstance(
            diagnostics,
            dict,
        ):
            return {
                "diagnostics": str(
                    diagnostics
                )
            }

        allowed_fields = [
            "events",
            "cases",
            "activities",
            "average_events_per_case",
            "median_events_per_case",
            "min_events_per_case",
            "max_events_per_case",
            "number_of_variants",
            "singleton_variants",
            "rare_variants_<=5_cases",
            "dominant_variant_cases",
            "dominant_variant_percentage",
            "cases_with_repeated_activities",
            "percentage_cases_with_repetition",
            "maximum_activity_repetition",
            "average_duration_hours",
            "median_duration_hours",
        ]

        compact = {}

        for field in allowed_fields:

            if field in diagnostics:

                compact[field] = diagnostics[
                    field
                ]

        return compact

    # =========================================================
    # COMPACT MODEL EVALUATIONS
    # =========================================================

    def _compact_model_evaluations(
        self,
        model_evaluations: Any,
    ) -> List[Dict[str, Any]]:
        """
        Convert evaluation results into a compact representation.

        Only model name and numerical evaluation metrics are sent
        to the LLM.

        This prevents Petri nets, pandas objects, diagnostics,
        and other large structures from entering the prompt.
        """

        compact = []

        if model_evaluations is None:
            return compact

        # -----------------------------------------------------
        # DataFrame support
        # -----------------------------------------------------

        if hasattr(
            model_evaluations,
            "to_dict",
        ):

            try:

                model_evaluations = (
                    model_evaluations.to_dict(
                        orient="records"
                    )
                )

            except Exception:

                model_evaluations = []

        # -----------------------------------------------------
        # Dictionary support
        # -----------------------------------------------------

        if isinstance(
            model_evaluations,
            dict,
        ):

            # Example:
            #
            # {
            #     "Inductive Miner": {...},
            #     "Alpha Miner": {...}
            # }

            if all(
                isinstance(
                    value,
                    dict,
                )
                for value in model_evaluations.values()
            ):

                converted = []

                for model_name, values in (
                    model_evaluations.items()
                ):

                    row = dict(values)

                    row.setdefault(
                        "model",
                        model_name,
                    )

                    converted.append(
                        row
                    )

                model_evaluations = converted

            else:

                model_evaluations = [
                    model_evaluations
                ]

        # -----------------------------------------------------
        # List support
        # -----------------------------------------------------

        if not isinstance(
            model_evaluations,
            list,
        ):
            return compact

        # -----------------------------------------------------
        # Extract useful metrics
        # -----------------------------------------------------

        metric_fields = [
            "fitness",
            "precision",
            "generalization",
            "simplicity",
            "selection_score",
            "score",
            "rank",
        ]

        for evaluation in model_evaluations:

            if not isinstance(
                evaluation,
                dict,
            ):
                continue

            model_name = (
                evaluation.get("model")
                or evaluation.get("name")
                or evaluation.get("algorithm")
                or "Unknown Model"
            )

            row = {
                "model": str(
                    model_name
                )
            }

            for field in metric_fields:

                if field not in evaluation:
                    continue

                value = evaluation[
                    field
                ]

                numeric_value = (
                    self._safe_float(
                        value
                    )
                )

                if numeric_value is not None:

                    row[field] = round(
                        numeric_value,
                        6,
                    )

                else:

                    row[field] = str(
                        value
                    )

            compact.append(
                row
            )

        return compact

    # =========================================================
    # SELECTED MODEL SUMMARY
    # =========================================================

    def _selected_model_summary(
        self,
        selected_model: Any,
        model_evaluations: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:

        summary = {}

        if selected_model is None:
            return summary

        summary["name"] = str(
            selected_model
        )

        selected_name = (
            str(
                selected_model
            )
            .strip()
            .lower()
        )

        for evaluation in model_evaluations:

            model_name = (
                str(
                    evaluation.get(
                        "model",
                        "",
                    )
                )
                .strip()
                .lower()
            )

            if model_name == selected_name:

                summary.update(
                    evaluation
                )

                break

        return summary

    # =========================================================
    # MODEL COMPARISON SUMMARY
    # =========================================================

    def _build_comparison_summary(
        self,
        evaluations: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:
        """
        Create a small deterministic summary of the available
        models.

        This helps the LLM reason consistently over all models.
        """

        if not evaluations:

            return {
                "model_count": 0,
                "best_fitness_model": None,
                "best_precision_model": None,
                "best_generalization_model": None,
                "best_simplicity_model": None,
                "best_selection_score_model": None,
            }

        def best_by(metric):

            candidates = [
                item
                for item in evaluations
                if self._safe_float(
                    item.get(metric)
                )
                is not None
            ]

            if not candidates:
                return None

            best = max(
                candidates,
                key=lambda item:
                self._safe_float(
                    item.get(metric),
                    -1.0,
                ),
            )

            return {
                "model": best.get(
                    "model"
                ),
                metric: best.get(
                    metric
                ),
            }

        return {
            "model_count": len(
                evaluations
            ),

            "available_models": [
                item.get(
                    "model"
                )
                for item in evaluations
            ],

            "best_fitness_model":
                best_by("fitness"),

            "best_precision_model":
                best_by("precision"),

            "best_generalization_model":
                best_by("generalization"),

            "best_simplicity_model":
                best_by("simplicity"),

            "best_selection_score_model":
                best_by("selection_score"),
        }

    # =========================================================
    # RAG QUERY
    # =========================================================

    def _build_rag_query(
        self,
        diagnostics: Any,
        model_evaluations: Any,
        selected_model: Any = None,
    ) -> str:
        """
        Build a semantic retrieval query from the actual
        process-mining situation.

        The query contains process-mining concepts rather than
        the complete raw event-log data.
        """

        compact_diagnostics = (
            self._compact_diagnostics(
                diagnostics
            )
        )

        compact_evaluations = (
            self._compact_model_evaluations(
                model_evaluations
            )
        )

        query_parts = [
            "process mining model evaluation",
            "fitness precision generalization simplicity",
            "adaptive process discovery",
        ]

        if selected_model:
            query_parts.append(
                f"selected model: {selected_model}"
            )

        # -----------------------------------------------------
        # Add relevant characteristics
        # -----------------------------------------------------

        if compact_diagnostics:

            if compact_diagnostics.get(
                "number_of_variants"
            ) is not None:

                query_parts.append(
                    "process variants"
                )

            if compact_diagnostics.get(
                "cases_with_repeated_activities"
            ) is not None:

                query_parts.append(
                    "repeated activities and process complexity"
                )

            if compact_diagnostics.get(
                "percentage_cases_with_repetition"
            ) is not None:

                query_parts.append(
                    "process repetition"
                )

        # -----------------------------------------------------
        # Add model-quality signals
        # -----------------------------------------------------

        for evaluation in compact_evaluations:

            fitness = self._safe_float(
                evaluation.get("fitness")
            )

            precision = self._safe_float(
                evaluation.get("precision")
            )

            if fitness is not None and fitness >= 0.9:
                query_parts.append(
                    "high fitness"
                )

            if precision is not None and precision < 0.5:
                query_parts.append(
                    "low precision possible overgeneralization"
                )

        return " ".join(query_parts)

    # =========================================================
    # RETRIEVE KNOWLEDGE
    # =========================================================

    def _retrieve_knowledge(
        self,
        diagnostics: Any,
        model_evaluations: Any,
        selected_model: Any = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant knowledge from the RAG knowledge base.

        RAG failures do not terminate the reasoning process.
        """

        if self.knowledge_base is None:
            if self.rag_build_error:
                print("RAG retrieval skipped because the knowledge base is unavailable.")
                print(f"RAG initialization error: {self.rag_build_error}")
            return []

        query = self._build_rag_query(
            diagnostics=diagnostics,
            model_evaluations=model_evaluations,
            selected_model=selected_model,
        )

        print(
            "\n" + "-" * 60
        )

        print(
            "RAG KNOWLEDGE RETRIEVAL"
        )

        print(
            "-" * 60
        )

        print(
            f"Query: {query}"
        )

        try:

            results = self.knowledge_base.retrieve(
                query=query,
                top_k=self.rag_top_k,
            )

        except Exception as exc:

            print(
                "RAG retrieval failed."
            )

            print(
                f"Error: {type(exc).__name__}: {exc}"
            )

            print(
                "Continuing without retrieved knowledge."
            )

            return []

        print(
            f"Knowledge chunks retrieved: "
            f"{len(results)}"
        )

        for result in results:

            print(
                f"  - {result.get('source')} "
                f"(score={result.get('score', 0.0):.4f})"
            )

        print(
            "-" * 60
        )

        return results

    # =========================================================
    # FORMAT RAG CONTEXT
    # =========================================================

    @staticmethod
    def _format_rag_context(
        retrieved_knowledge: List[Dict[str, Any]],
    ) -> str:
        """
        Convert retrieved knowledge into compact prompt context.
        """

        if not retrieved_knowledge:
            return (
                "No external knowledge was retrieved."
            )

        context_parts = []

        for index, item in enumerate(
            retrieved_knowledge,
            start=1,
        ):

            source = item.get(
                "source",
                "Unknown source",
            )

            score = item.get(
                "score",
                0.0,
            )

            text = item.get(
                "text",
                "",
            )

            context_parts.append(
                f"""
Knowledge Item {index}

Source:
{source}

Relevance Score:
{float(score):.4f}

Content:
{text}
""".strip()
            )

        return "\n\n".join(
            context_parts
        )

    # =========================================================
    # BUILD PROMPT
    # =========================================================

    def _build_prompt(
        self,
        diagnostics,
        model_evaluations,
        selected_model=None,
        retrieved_knowledge=None,
    ) -> str:

        compact_diagnostics = (
            self._compact_diagnostics(
                diagnostics
            )
        )

        compact_evaluations = (
            self._compact_model_evaluations(
                model_evaluations
            )
        )

        selected_summary = (
            self._selected_model_summary(
                selected_model,
                compact_evaluations,
            )
        )

        comparison_summary = (
            self._build_comparison_summary(
                compact_evaluations
            )
        )

        # -----------------------------------------------------
        # RAG context
        # -----------------------------------------------------

        rag_context = (
            self._format_rag_context(
                retrieved_knowledge
                if retrieved_knowledge
                else []
            )
        )

        prompt_data = {
            "process_log_diagnostics":
                compact_diagnostics,

            "model_evaluations":
                compact_evaluations,

            "model_comparison":
                comparison_summary,

            "currently_selected_model":
                selected_summary,

            "retrieved_knowledge":
                rag_context,
        }

        prompt = f"""
Analyze the following process-mining evidence.

============================================================
PROCESS-MINING EVIDENCE
============================================================

{json.dumps(
    prompt_data,
    indent=2,
    default=str,
)}

============================================================
RETRIEVED DOMAIN KNOWLEDGE
============================================================

The retrieved knowledge above comes from the APMOA
knowledge base.

Use it to understand process-mining concepts and
adaptation principles.

Important:

- Retrieved knowledge is contextual guidance.
- Actual model metrics and process-log diagnostics
  are the primary evidence.
- Do not treat retrieved knowledge as actual experimental
  results.
- Do not invent metrics based on retrieved knowledge.
- Do not override actual process-mining evidence with
  generic knowledge.
- If retrieved knowledge is not relevant, do not use it
  in the final reasoning.

============================================================
TASK
============================================================

Determine:

1. What is happening in the current model?
2. What is the most important quality problem?
3. How do ALL available models compare?
4. What evidence supports the diagnosis?
5. Should the current model be kept or rediscovered?
6. If rediscovery is required, what strategy should be used?
7. Why is that strategy appropriate?
8. How does the retrieved knowledge support the reasoning,
   if relevant?

============================================================
DECISION GUIDELINES
============================================================

- Compare every supplied model.
- Do not focus only on the selected model.
- Do not invent missing metrics.
- Do not invent discovery algorithms.
- Fitness measures how well observed behavior is represented.
- Precision indicates how much unobserved behavior is allowed.
- Generalization and simplicity must also be considered.
- High fitness + very low precision indicates possible
  overgeneralization.
- Low fitness indicates that observed behavior may not be
  adequately represented.
- A balanced model may be preferable to a model with the
  highest individual metric.
- If Heuristics Miner is substantially more precision-oriented,
  it may be recommended.
- If Split Miner provides a better trade-off, it may be
  recommended.
- If Inductive Miner remains the strongest candidate but is
  overgeneralized, consider noise filtering.
- Do not recommend an unavailable algorithm.
- Use retrieved knowledge only when relevant.
- Do not claim that the knowledge base proves an experimental
  result.

Return ONLY valid JSON using exactly the required fields.
"""

        return prompt

    # =========================================================
    # RESPONSE CLEANING
    # =========================================================

    def _clean_response(
        self,
        response: str,
    ) -> str:

        response = response.strip()

        response = re.sub(
            r"^```json\s*",
            "",
            response,
            flags=re.IGNORECASE,
        )

        response = re.sub(
            r"^```\s*",
            "",
            response,
        )

        response = re.sub(
            r"\s*```$",
            "",
            response,
        )

        return response.strip()

    # =========================================================
    # PARSE RESPONSE
    # =========================================================

    def _deterministic_fallback(
        self,
        model_evaluations: Any,
        selected_model: Any = None,
        parse_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Produce a safe reasoning result when the LLM response cannot
        be parsed.

        This fallback uses only supplied process-mining metrics and
        does not pretend that the failed LLM response was valid.
        """

        evaluations = self._compact_model_evaluations(
            model_evaluations
        )

        selected_summary = self._selected_model_summary(
            selected_model,
            evaluations,
        )

        fitness = self._safe_float(
            selected_summary.get("fitness")
        )
        precision = self._safe_float(
            selected_summary.get("precision")
        )
        generalization = self._safe_float(
            selected_summary.get("generalization")
        )
        simplicity = self._safe_float(
            selected_summary.get("simplicity")
        )

        evidence = []

        if fitness is not None:
            evidence.append(
                f"Selected model fitness: {fitness:.3f}"
            )

        if precision is not None:
            evidence.append(
                f"Selected model precision: {precision:.3f}"
            )

        if generalization is not None:
            evidence.append(
                f"Selected model generalization: {generalization:.3f}"
            )

        if simplicity is not None:
            evidence.append(
                f"Selected model simplicity: {simplicity:.3f}"
            )

        quality_violations = []

        if fitness is not None and fitness < 0.80:
            quality_violations.append("fitness is below 0.80")

        if precision is not None and precision < 0.50:
            quality_violations.append("precision is below 0.50")

        if generalization is not None and generalization < 0.70:
            quality_violations.append(
                "generalization is below 0.70"
            )

        if simplicity is not None and simplicity < 0.40:
            quality_violations.append("simplicity is below 0.40")

        if quality_violations:
            recommended_action = "rediscover"

            if precision is not None and precision < 0.50:
                strategy = "precision_oriented"
            elif fitness is not None and fitness < 0.80:
                strategy = "fitness_oriented"
            elif generalization is not None and generalization < 0.70:
                strategy = "balanced"
            elif simplicity is not None and simplicity < 0.40:
                strategy = "simplicity_oriented"
            else:
                strategy = "balanced"

            diagnosis = (
                "The selected model does not satisfy all configured "
                "prototype quality requirements: "
                + "; ".join(quality_violations)
                + "."
            )

            reasoning = (
                "The LLM response could not be parsed, so a deterministic "
                "safety fallback was used based only on the supplied "
                "model metrics. Rediscovery is recommended because at "
                "least one configured quality requirement is violated."
            )
        else:
            recommended_action = "keep_model"
            strategy = "keep_current"

            diagnosis = (
                "The selected model satisfies all configured prototype "
                "quality requirements."
            )

            reasoning = (
                "The LLM response could not be parsed, so a deterministic "
                "safety fallback was used based only on the supplied "
                "model metrics. The current model is retained because "
                "no configured quality requirement is violated."
            )

        if parse_error:
            evidence.append(
                "LLM JSON parsing failed; deterministic fallback was used."
            )

        return {
            "observation": (
                f"Current selected model: "
                f"{selected_model if selected_model is not None else 'Unknown'}."
            ),
            "diagnosis": diagnosis,
            "evidence": evidence,
            "recommended_action": recommended_action,
            "reasoning": reasoning,
            "confidence": 0.70,
            "reasoning_source": "deterministic_fallback",
            "llm_parse_error": parse_error or "",
            "fallback_strategy": strategy,
        }

    def _parse_response(
        self,
        response: str,
    ) -> Dict[str, Any]:
        """
        Parse and validate the LLM response.
        """

        response = self._clean_response(
            response
        )

        try:
            result = json.loads(
                response
            )

        except json.JSONDecodeError as exc:

            start = response.find(
                "{"
            )

            end = response.rfind(
                "}"
            )

            if (
                start != -1
                and end != -1
                and end > start
            ):

                candidate = response[
                    start:end + 1
                ]

                try:
                    result = json.loads(
                        candidate
                    )

                except json.JSONDecodeError:
                    raise ValueError(
                        "LLM returned invalid JSON.\n\n"
                        f"Raw response:\n{response}"
                    ) from exc
            else:
                raise ValueError(
                    "LLM returned invalid JSON.\n\n"
                    f"Raw response:\n{response}"
                ) from exc

        if not isinstance(
            result,
            dict,
        ):
            raise ValueError(
                "LLM response must be a JSON object."
            )

        required_fields = [
            "observation",
            "diagnosis",
            "evidence",
            "recommended_action",
            "reasoning",
            "confidence",
        ]

        missing_fields = [
            field
            for field in required_fields
            if field not in result
        ]

        if missing_fields:
            raise ValueError(
                "LLM response is missing required fields: "
                f"{missing_fields}"
            )

        if not isinstance(
            result["evidence"],
            list,
        ):
            raise ValueError(
                "Evidence must be a list."
            )

        result["evidence"] = [
            str(item)
            for item in result["evidence"]
        ]

        confidence = self._safe_float(
            result["confidence"]
        )

        if confidence is None:
            raise ValueError(
                "Confidence must be numeric."
            )

        if not (
            0.0 <= confidence <= 1.0
        ):
            raise ValueError(
                "Confidence must be between 0.0 and 1.0."
            )

        result["confidence"] = confidence

        for field in [
            "observation",
            "diagnosis",
            "recommended_action",
            "reasoning",
        ]:
            result[field] = str(
                result[field]
            )

        return result

    # =========================================================
    # PRINT RESULT
    # =========================================================

    def _print_result(
        self,
        result: Dict[str, Any],
    ):

        print(
            "\n" + "-" * 60
        )

        print(
            "OBSERVATION"
        )

        print(
            "-" * 60
        )

        print(
            result["observation"]
        )

        print(
            "\n" + "-" * 60
        )

        print(
            "DIAGNOSIS"
        )

        print(
            "-" * 60
        )

        print(
            result["diagnosis"]
        )

        print(
            "\n" + "-" * 60
        )

        print(
            "EVIDENCE"
        )

        print(
            "-" * 60
        )

        for evidence in result[
            "evidence"
        ]:

            print(
                f"  • {evidence}"
            )

        print(
            "\n" + "-" * 60
        )

        print(
            "RECOMMENDED ACTION"
        )

        print(
            "-" * 60
        )

        print(
            result[
                "recommended_action"
            ]
        )

        print(
            "\n" + "-" * 60
        )

        print(
            "REASONING"
        )

        print(
            "-" * 60
        )

        print(
            result["reasoning"]
        )

        print(
            "\n" + "-" * 60
        )

        print(
            "CONFIDENCE"
        )

        print(
            "-" * 60
        )

        print(
            f"{result['confidence']:.2f}"
        )

    # =========================================================
    # REASON
    # =========================================================

    def reason(
        self,
        diagnostics,
        model_evaluations,
        selected_model=None,
    ) -> Dict[str, Any]:
        """
        Execute the reasoning process.

        Parameters
        ----------
        diagnostics:
            Process-log diagnostics.

        model_evaluations:
            Evaluation results for ALL discovered models.

        selected_model:
            Name of the currently selected model.

        Returns
        -------
        dict
            Structured reasoning result.
        """

        print(
            "\n" + "=" * 60
        )

        print(
            "APMOA - REASONING AGENT"
        )

        print(
            "=" * 60
        )

        print(
            "\nAnalyzing process-mining evidence..."
        )

        # -----------------------------------------------------
        # Compact input
        # -----------------------------------------------------

        compact_evaluations = (
            self._compact_model_evaluations(
                model_evaluations
            )
        )

        print(
            f"Models received: "
            f"{len(compact_evaluations)}"
        )

        if compact_evaluations:

            print(
                "Models being compared:"
            )

            for model in compact_evaluations:

                print(
                    f"  - {model.get('model')}"
                )

        # -----------------------------------------------------
        # Retrieve RAG knowledge
        # -----------------------------------------------------

        retrieved_knowledge = (
            self._retrieve_knowledge(
                diagnostics=diagnostics,
                model_evaluations=model_evaluations,
                selected_model=selected_model,
            )
        )

        # -----------------------------------------------------
        # Build prompt
        # -----------------------------------------------------

        prompt = self._build_prompt(
            diagnostics=diagnostics,
            model_evaluations=model_evaluations,
            selected_model=selected_model,
            retrieved_knowledge=retrieved_knowledge,
        )

        print(
            "Prepared reasoning prompt."
        )

        print(
            f"Prompt size: "
            f"{len(prompt):,} characters"
        )

        # -----------------------------------------------------
        # Call LLM
        # -----------------------------------------------------

        try:

            response = self.llm.generate(
                prompt=prompt,
                system_prompt=self._system_prompt(),
            )

        except Exception as exc:

            print(
                "\nReasoning Agent failed."
            )

            print(
                f"Error: {exc}"
            )

            raise

        # -----------------------------------------------------
        # Parse response
        # -----------------------------------------------------

        try:
            result = self._parse_response(
                response
            )

            result["reasoning_source"] = "llm"

        except ValueError as exc:

            print(
                "\nReasoning Agent received invalid or truncated JSON."
            )

            print(
                f"Parser error: {exc}"
            )

            print(
                "Using deterministic reasoning fallback "
                "from supplied process-mining metrics."
            )

            result = self._deterministic_fallback(
                model_evaluations=model_evaluations,
                selected_model=selected_model,
                parse_error=str(exc),
            )

        # -----------------------------------------------------
        # Add RAG metadata
        # -----------------------------------------------------
        # This metadata is added after parsing so that the
        # required LLM JSON structure remains unchanged.

        result["rag_used"] = bool(
            retrieved_knowledge
        )
        result["rag_enabled"] = bool(self.rag_enabled)
        result["rag_top_k"] = self.rag_top_k
        result["rag_retrieved_count"] = len(retrieved_knowledge)

        result["rag_sources"] = [
            item.get(
                "source",
                "Unknown source",
            )
            for item in retrieved_knowledge
        ]
        result["rag_initialization_error"] = self.rag_build_error or ""

        print(
            "Reasoning Agent completed."
        )

        # -----------------------------------------------------
        # Display result
        # -----------------------------------------------------

        self._print_result(
            result
        )

        print(
            "\n" + "=" * 60
        )

        print(
            "REASONING AGENT COMPLETED"
        )

        print(
            "=" * 60
        )

        return result

def _test_invalid_json_fallback() -> None:
    """Test safe handling of a truncated LLM JSON response."""
    agent = ReasoningAgent(llm_provider=object())

    evaluations = [
        {
            "model": "Split Miner",
            "fitness": 0.9742,
            "precision": 0.6387,
            "generalization": 0.8564,
            "simplicity": 0.5642,
            "selection_score": 0.767992,
        }
    ]

    truncated_response = '{"observation":"truncated"'
    try:
        agent._parse_response(truncated_response)
        raise AssertionError("Truncated JSON unexpectedly parsed.")
    except ValueError:
        pass

    fallback = agent._deterministic_fallback(
        evaluations,
        "Split Miner",
        "LLM returned invalid JSON.",
    )
    assert fallback["recommended_action"] == "keep_model"
    assert fallback["fallback_strategy"] == "keep_current"
    assert fallback["reasoning_source"] == "deterministic_fallback"
    print("\n✓ TRUNCATED JSON FALLBACK TEST PASSED")


if __name__ == "__main__":
    _test_invalid_json_fallback()
