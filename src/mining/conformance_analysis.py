import pandas as pd
import pm4py


class ConformanceAnalyzer:
    """
    Analyzes the conformance between an event log
    and a discovered Petri-net process model.

    The analysis focuses on:
        - Trace fitness
        - Deviating cases
        - Poorly fitting variants
        - Repeated/rework behavior
        - Activity-level frequency
    """

    def __init__(
        self,
        log: pd.DataFrame,
        petri_net,
        initial_marking,
        final_marking,
        case_column: str = "case_id",
        activity_column: str = "activity",
        timestamp_column: str = "timestamp",
    ):

        self.log = log.copy()

        self.petri_net = petri_net
        self.initial_marking = initial_marking
        self.final_marking = final_marking

        self.case_column = case_column
        self.activity_column = activity_column
        self.timestamp_column = timestamp_column

        self._prepare()

        self.replay_result = None

    # =========================================================
    # PREPARE LOG
    # =========================================================

    def _prepare(self):
        """Prepare the event log."""

        self.log[self.timestamp_column] = pd.to_datetime(
            self.log[self.timestamp_column]
        )

        self.log = self.log.sort_values(
            by=[
                self.case_column,
                self.timestamp_column,
            ]
        ).reset_index(drop=True)

    # =========================================================
    # TOKEN-BASED REPLAY
    # =========================================================

    def perform_replay(self):
        """
        Perform token-based replay.

        The replay result is stored so that other
        diagnostic methods can reuse it.
        """

        print("\nRunning token-based replay...")

        self.replay_result = pm4py.conformance_diagnostics_token_based_replay(
            self.log,
            self.petri_net,
            self.initial_marking,
            self.final_marking,
            case_id_key=self.case_column,
            activity_key=self.activity_column,
            timestamp_key=self.timestamp_column,
        )

        print("Token-based replay completed.")

        return self.replay_result

    # =========================================================
    # TRACE FITNESS
    # =========================================================

    def trace_fitness(self):
        """
        Calculate fitness information for each case.
        """

        if self.replay_result is None:
            self.perform_replay()

        results = []

        for index, replay in enumerate(self.replay_result):

            trace_is_fit = replay.get(
                "trace_is_fit",
                False
            )

            trace_fitness = replay.get(
                "trace_fitness",
                None
            )

            results.append(
                {
                    "case_index": index,
                    "trace_is_fit": trace_is_fit,
                    "trace_fitness": trace_fitness,
                }
            )

        return pd.DataFrame(results)

    # =========================================================
    # DEVIATING CASES
    # =========================================================

    def deviating_cases(self):
        """
        Identify cases that do not perfectly fit
        the discovered model.
        """

        fitness = self.trace_fitness()

        deviations = fitness[
            fitness["trace_is_fit"] == False
        ]

        return deviations

    # =========================================================
    # FITNESS SUMMARY
    # =========================================================

    def fitness_summary(self):
        """
        Generate summary statistics for trace fitness.
        """

        fitness = self.trace_fitness()

        total_cases = len(fitness)

        fitting_cases = (
            fitness["trace_is_fit"]
            .sum()
        )

        deviating_case_count = (
            total_cases - fitting_cases
        )

        fitting_percentage = (
            fitting_cases
            / total_cases
            * 100
            if total_cases > 0
            else 0
        )

        return {
            "total_cases": total_cases,
            "fitting_cases": fitting_cases,
            "deviating_cases": deviating_case_count,
            "fitting_percentage": fitting_percentage,
        }

    # =========================================================
    # VARIANT FITNESS
    # =========================================================

    def variant_fitness(self):
        """
        Analyze fitness at process-variant level.
        """

        fitness = self.trace_fitness()

        case_ids = (
            self.log[
                self.case_column
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        fitness["case_id"] = case_ids

        sequences = (
            self.log
            .groupby(self.case_column)[
                self.activity_column
            ]
            .apply(tuple)
            .reset_index()
        )

        sequences.columns = [
            "case_id",
            "variant",
        ]

        result = fitness.merge(
            sequences,
            on="case_id",
            how="left",
        )

        variant_summary = (
            result
            .groupby("variant")
            .agg(
                cases=(
                    "case_id",
                    "count",
                ),
                fitting_cases=(
                    "trace_is_fit",
                    "sum",
                ),
                average_fitness=(
                    "trace_fitness",
                    "mean",
                ),
            )
            .reset_index()
        )

        variant_summary[
            "fitting_percentage"
        ] = (
            variant_summary["fitting_cases"]
            / variant_summary["cases"]
            * 100
        )

        return variant_summary.sort_values(
            by="average_fitness"
        )

    # =========================================================
    # REPEATED ACTIVITY ANALYSIS
    # =========================================================

    def repeated_activity_analysis(self):
        """
        Identify repeated activities across cases.
        """

        repeated = (
            self.log
            .groupby(
                [
                    self.case_column,
                    self.activity_column,
                ]
            )
            .size()
            .reset_index(
                name="count"
            )
        )

        repeated = repeated[
            repeated["count"] > 1
        ]

        return repeated.sort_values(
            by="count",
            ascending=False,
        )

    # =========================================================
    # ACTIVITY FREQUENCY
    # =========================================================

    def activity_frequency(self):
        """
        Calculate activity frequency.
        """

        frequency = (
            self.log[
                self.activity_column
            ]
            .value_counts()
            .reset_index()
        )

        frequency.columns = [
            "activity",
            "frequency",
        ]

        frequency["percentage"] = (
            frequency["frequency"]
            / len(self.log)
            * 100
        )

        return frequency

    # =========================================================
    # REPORT
    # =========================================================

    def generate_report(self):
        """
        Generate the complete conformance analysis.
        """

        print("\n")
        print("=" * 60)
        print("APMOA - CONFORMANCE ANALYSIS")
        print("=" * 60)

        # -----------------------------------------------------
        # FITNESS
        # -----------------------------------------------------

        summary = self.fitness_summary()

        print("\n" + "-" * 60)
        print("TRACE FITNESS")
        print("-" * 60)

        print(
            f"Total cases          : "
            f"{summary['total_cases']}"
        )

        print(
            f"Fitting cases        : "
            f"{summary['fitting_cases']}"
        )

        print(
            f"Deviating cases      : "
            f"{summary['deviating_cases']}"
        )

        print(
            f"Fitting percentage   : "
            f"{summary['fitting_percentage']:.2f}%"
        )

        # -----------------------------------------------------
        # VARIANTS
        # -----------------------------------------------------

        variants = self.variant_fitness()

        print("\n" + "-" * 60)
        print("LOWEST-FIT PROCESS VARIANTS")
        print("-" * 60)

        for index, row in variants.head(10).iterrows():

            print(
                f"\nVariant rank: {index + 1}"
            )

            print(
                f"Cases             : "
                f"{row['cases']}"
            )

            print(
                f"Average fitness   : "
                f"{row['average_fitness']:.4f}"
            )

            print(
                f"Fitting percentage: "
                f"{row['fitting_percentage']:.2f}%"
            )

            print("Sequence:")

            for activity in row["variant"]:

                print(
                    f"  → {activity}"
                )

        # -----------------------------------------------------
        # REPEATED ACTIVITIES
        # -----------------------------------------------------

        repeated = (
            self.repeated_activity_analysis()
        )

        print("\n" + "-" * 60)
        print("TOP REPEATED ACTIVITIES")
        print("-" * 60)

        if repeated.empty:

            print("No repeated activities found.")

        else:

            print(
                repeated.head(20).to_string(
                    index=False
                )
            )

        # -----------------------------------------------------
        # ACTIVITY FREQUENCY
        # -----------------------------------------------------

        activities = self.activity_frequency()

        print("\n" + "-" * 60)
        print("ACTIVITY FREQUENCY")
        print("-" * 60)

        print(
            activities.head(20).to_string(
                index=False
            )
        )

        # -----------------------------------------------------
        # COMPLETION
        # -----------------------------------------------------

        print("\n" + "=" * 60)
        print("CONFORMANCE ANALYSIS COMPLETED")
        print("=" * 60)