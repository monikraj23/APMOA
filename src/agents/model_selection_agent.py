"""
APMOA - Model Selection Agent

Selects the best process discovery model based on
multiple quality dimensions:

    - Fitness
    - Precision
    - Generalization
    - Simplicity

The agent is algorithm-agnostic and can evaluate
any number of discovered models.

Currently supported discovery algorithms:

    1. Inductive Miner
    2. Alpha Miner
    3. Alpha+ Miner
    4. Heuristics Miner
    5. Split Miner

The selection mechanism is designed so that additional
algorithms can be added later without modifying this agent.
"""

from __future__ import annotations

import pandas as pd


class ModelSelectionAgent:
    """
    Autonomous model selection agent.

    The agent receives model evaluation results and
    calculates a weighted selection score for every
    candidate model.

    Higher score = better candidate.
    """

    # =========================================================
    # DEFAULT WEIGHTS
    # =========================================================

    DEFAULT_WEIGHTS = {

        "fitness": 0.30,

        "precision": 0.30,

        "generalization": 0.20,

        "simplicity": 0.20,
    }

    # =========================================================
    # CONSTRUCTOR
    # =========================================================

    def __init__(
        self,
        evaluation_results: pd.DataFrame,
        weights: dict[str, float] | None = None,
    ):
        """
        Parameters
        ----------
        evaluation_results:
            DataFrame containing model evaluation results.

        Expected columns:

            model
            fitness
            precision
            generalization
            simplicity

        Optional:

            selection_score

        weights:
            Optional custom metric weights.
        """

        if evaluation_results is None:

            raise ValueError(
                "evaluation_results cannot be None."
            )

        if not isinstance(
            evaluation_results,
            pd.DataFrame,
        ):

            raise TypeError(
                "evaluation_results must be a pandas DataFrame."
            )

        if evaluation_results.empty:

            raise ValueError(
                "evaluation_results is empty."
            )

        self.evaluation_results = (
            evaluation_results.copy()
        )

        # -----------------------------------------------------
        # Validate required columns
        # -----------------------------------------------------

        required_columns = {

            "model",

            "fitness",

            "precision",

            "generalization",

            "simplicity",
        }

        missing_columns = (
            required_columns
            - set(
                self.evaluation_results.columns
            )
        )

        if missing_columns:

            raise ValueError(
                "Missing required evaluation columns: "
                + ", ".join(
                    sorted(missing_columns)
                )
            )

        # -----------------------------------------------------
        # Configure weights
        # -----------------------------------------------------

        if weights is None:

            self.weights = (
                self.DEFAULT_WEIGHTS.copy()
            )

        else:

            self.weights = (
                weights.copy()
            )

        self._validate_weights()

    # =========================================================
    # VALIDATE WEIGHTS
    # =========================================================

    def _validate_weights(self):
        """
        Validate selection weights.
        """

        required_metrics = {

            "fitness",

            "precision",

            "generalization",

            "simplicity",
        }

        if set(self.weights.keys()) != required_metrics:

            raise ValueError(
                "Weights must contain exactly: "
                "fitness, precision, generalization, simplicity."
            )

        for metric, weight in self.weights.items():

            if not isinstance(
                weight,
                (int, float),
            ):

                raise TypeError(
                    f"Weight for '{metric}' "
                    "must be numeric."
                )

            if weight < 0:

                raise ValueError(
                    f"Weight for '{metric}' "
                    "cannot be negative."
                )

        total_weight = sum(
            self.weights.values()
        )

        if total_weight <= 0:

            raise ValueError(
                "At least one selection weight "
                "must be greater than zero."
            )

        # -----------------------------------------------------
        # Normalize weights
        # -----------------------------------------------------

        if abs(
            total_weight - 1.0
        ) > 1e-9:

            self.weights = {

                metric:
                    weight / total_weight

                for metric, weight
                in self.weights.items()
            }

    # =========================================================
    # CLEAN EVALUATION RESULTS
    # =========================================================

    def _prepare_results(self):
        """
        Clean and validate evaluation results.
        """

        df = (
            self.evaluation_results.copy()
        )

        numeric_columns = [

            "fitness",

            "precision",

            "generalization",

            "simplicity",
        ]

        # -----------------------------------------------------
        # Convert metrics to numeric
        # -----------------------------------------------------

        for column in numeric_columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

        # -----------------------------------------------------
        # Remove invalid rows
        # -----------------------------------------------------

        invalid_rows = (
            df[numeric_columns]
            .isna()
            .any(axis=1)
        )

        if invalid_rows.any():

            invalid_models = (
                df.loc[
                    invalid_rows,
                    "model",
                ]
                .astype(str)
                .tolist()
            )

            print(
                "\nWARNING:"
            )

            print(
                "Ignoring models with invalid "
                "evaluation metrics:"
            )

            for model in invalid_models:

                print(
                    f"  - {model}"
                )

            df = df.loc[
                ~invalid_rows
            ].copy()

        if df.empty:

            raise ValueError(
                "No valid model evaluation results remain."
            )

        # -----------------------------------------------------
        # Clip metric values
        # -----------------------------------------------------

        for column in numeric_columns:

            df[column] = (
                df[column]
                .clip(
                    lower=0.0,
                    upper=1.0,
                )
            )

        return df

    # =========================================================
    # CALCULATE SELECTION SCORE
    # =========================================================

    def _calculate_selection_score(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate weighted selection score.
        """

        df = df.copy()

        df[
            "selection_score"
        ] = (

            df["fitness"]
            * self.weights["fitness"]

            +

            df["precision"]
            * self.weights["precision"]

            +

            df["generalization"]
            * self.weights["generalization"]

            +

            df["simplicity"]
            * self.weights["simplicity"]
        )

        return df

    # =========================================================
    # RANK MODELS
    # =========================================================

    def _rank_models(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Rank models from best to worst.
        """

        ranked = (
            df.sort_values(
                by=[
                    "selection_score",
                    "precision",
                    "fitness",
                    "generalization",
                    "simplicity",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    False,
                    False,
                ],
            )
            .reset_index(
                drop=True
            )
        )

        ranked[
            "rank"
        ] = (
            ranked.index + 1
        )

        return ranked

    # =========================================================
    # GENERATE EXPLANATION
    # =========================================================

    def _generate_explanation(
        self,
        selected_row: pd.Series,
    ) -> list[str]:
        """
        Generate a human-readable explanation
        for the selected model.
        """

        model = selected_row[
            "model"
        ]

        score = float(
            selected_row[
                "selection_score"
            ]
        )

        fitness = float(
            selected_row[
                "fitness"
            ]
        )

        precision = float(
            selected_row[
                "precision"
            ]
        )

        generalization = float(
            selected_row[
                "generalization"
            ]
        )

        simplicity = float(
            selected_row[
                "simplicity"
            ]
        )

        explanation = []

        explanation.append(
            f"{model} was selected as the best candidate model."
        )

        explanation.append(
            f"Overall selection score: {score:.4f}."
        )

        explanation.append(
            f"Fitness: {fitness:.4f}"
        )

        explanation.append(
            f"Precision: {precision:.4f}"
        )

        explanation.append(
            f"Generalization: {generalization:.4f}"
        )

        explanation.append(
            f"Simplicity: {simplicity:.4f}"
        )

        explanation.append(
            "The score is calculated using weighted "
            "fitness, precision, generalization, and simplicity."
        )

        # -----------------------------------------------------
        # Diagnostic explanation
        # -----------------------------------------------------

        if precision < 0.20:

            explanation.append(
                "Warning: the selected model has very low "
                "precision and may allow excessive unobserved behavior."
            )

        elif precision < 0.40:

            explanation.append(
                "The selected model has relatively low precision "
                "and may benefit from further refinement."
            )

        if fitness < 0.70:

            explanation.append(
                "Warning: fitness is relatively low, "
                "indicating that some observed behavior may "
                "not be adequately represented."
            )

        if simplicity < 0.30:

            explanation.append(
                "The selected model has low simplicity and "
                "may be structurally complex."
            )

        return explanation

    # =========================================================
    # SELECT
    # =========================================================

    def select(self):
        """
        Select the best model.

        Returns
        -------
        dict
            Contains:

                selected_model
                selection_score
                ranking
                explanation
        """

        print("\n" + "=" * 60)

        print(
            "APMOA - MODEL SELECTION AGENT"
        )

        print("=" * 60)

        # -----------------------------------------------------
        # Display weights
        # -----------------------------------------------------

        print(
            "\nSelection weights:"
        )

        print(
            f"  fitness         : "
            f"{self.weights['fitness']:.2f}"
        )

        print(
            f"  precision       : "
            f"{self.weights['precision']:.2f}"
        )

        print(
            f"  generalization  : "
            f"{self.weights['generalization']:.2f}"
        )

        print(
            f"  simplicity      : "
            f"{self.weights['simplicity']:.2f}"
        )

        # -----------------------------------------------------
        # Prepare
        # -----------------------------------------------------

        df = (
            self._prepare_results()
        )

        # -----------------------------------------------------
        # Calculate scores
        # -----------------------------------------------------

        df = (
            self._calculate_selection_score(
                df
            )
        )

        # -----------------------------------------------------
        # Rank
        # -----------------------------------------------------

        ranked = (
            self._rank_models(
                df
            )
        )

        # -----------------------------------------------------
        # Display ranking
        # -----------------------------------------------------

        print(
            "\n" + "-" * 60
        )

        print(
            "MODEL RANKING"
        )

        print(
            "-" * 60
        )

        display_columns = [

            "rank",

            "model",

            "fitness",

            "precision",

            "generalization",

            "simplicity",

            "selection_score",
        ]

        print(
            ranked[
                display_columns
            ].to_string(
                index=False
            )
        )

        # -----------------------------------------------------
        # Select best model
        # -----------------------------------------------------

        selected = ranked.iloc[0]

        selected_model = str(
            selected["model"]
        )

        selected_score = float(
            selected[
                "selection_score"
            ]
        )

        explanation = (
            self._generate_explanation(
                selected
            )
        )

        # -----------------------------------------------------
        # Selected model
        # -----------------------------------------------------

        print(
            "\n" + "-" * 60
        )

        print(
            "SELECTED MODEL"
        )

        print(
            "-" * 60
        )

        print(
            f"Model : {selected_model}"
        )

        print(
            f"Score : {selected_score:.4f}"
        )

        print(
            "\nReason:"
        )

        for reason in explanation:

            print(
                f"  {reason}"
            )

        print(
            "\n" + "=" * 60
        )

        print(
            "MODEL SELECTION AGENT COMPLETED"
        )

        print(
            "=" * 60
        )

        # -----------------------------------------------------
        # Return result
        # -----------------------------------------------------

        return {

            "selected_model":
                selected_model,

            "selection_score":
                selected_score,

            "ranking":
                ranked,

            "explanation":
                explanation,

            "selected_evaluation":
                selected.to_dict(),

            "weights":
                self.weights.copy(),
        }