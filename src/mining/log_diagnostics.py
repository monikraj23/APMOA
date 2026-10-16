import pandas as pd


class LogDiagnostics:
    """
    Performs structural diagnostics on an event log.

    The purpose of this component is to understand
    the characteristics of the process before the
    autonomous adaptation stage.
    """

    def __init__(
        self,
        log: pd.DataFrame,
        case_column: str = "case_id",
        activity_column: str = "activity",
        timestamp_column: str = "timestamp",
    ):
        self.log = log.copy()

        self.case_column = case_column
        self.activity_column = activity_column
        self.timestamp_column = timestamp_column

        self._prepare()

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
    # BASIC STATISTICS
    # =========================================================

    def basic_statistics(self):
        """Calculate basic event log statistics."""

        number_of_events = len(self.log)

        number_of_cases = self.log[
            self.case_column
        ].nunique()

        number_of_activities = self.log[
            self.activity_column
        ].nunique()

        events_per_case = (
            self.log.groupby(
                self.case_column
            ).size()
        )

        average_events_per_case = (
            events_per_case.mean()
        )

        median_events_per_case = (
            events_per_case.median()
        )

        max_events_per_case = (
            events_per_case.max()
        )

        min_events_per_case = (
            events_per_case.min()
        )

        return {
            "events": number_of_events,
            "cases": number_of_cases,
            "activities": number_of_activities,
            "average_events_per_case": average_events_per_case,
            "median_events_per_case": median_events_per_case,
            "min_events_per_case": min_events_per_case,
            "max_events_per_case": max_events_per_case,
        }

    # =========================================================
    # ACTIVITY FREQUENCY
    # =========================================================

    def activity_frequency(self):
        """Calculate activity frequencies."""

        frequency = (
            self.log[
                self.activity_column
            ]
            .value_counts()
            .rename_axis("activity")
            .reset_index(name="frequency")
        )

        frequency["percentage"] = (
            frequency["frequency"]
            / len(self.log)
            * 100
        )

        return frequency

    # =========================================================
    # VARIANT ANALYSIS
    # =========================================================

    def variant_analysis(self):
        """
        Analyze process variants.

        A variant is a unique sequence of activities
        executed by a case.
        """

        case_sequences = (
            self.log
            .groupby(self.case_column)[
                self.activity_column
            ]
            .apply(tuple)
        )

        variant_counts = (
            case_sequences
            .value_counts()
            .reset_index()
        )

        variant_counts.columns = [
            "variant",
            "cases",
        ]

        variant_counts["percentage"] = (
            variant_counts["cases"]
            / len(case_sequences)
            * 100
        )

        return variant_counts

    # =========================================================
    # VARIANT STATISTICS
    # =========================================================

    def variant_statistics(self):
        """Calculate variant-level statistics."""

        variants = self.variant_analysis()

        number_of_variants = len(variants)

        cases = self.log[
            self.case_column
        ].nunique()

        singleton_variants = (
            variants["cases"] == 1
        ).sum()

        rare_variants = (
            variants["cases"] <= 5
        ).sum()

        dominant_variant_cases = (
            variants["cases"].max()
        )

        dominant_variant_percentage = (
            dominant_variant_cases
            / cases
            * 100
        )

        return {
            "number_of_variants": number_of_variants,
            "singleton_variants": singleton_variants,
            "rare_variants_<=5_cases": rare_variants,
            "dominant_variant_cases": dominant_variant_cases,
            "dominant_variant_percentage": (
                dominant_variant_percentage
            ),
        }

    # =========================================================
    # REPEATED ACTIVITIES
    # =========================================================

    def repeated_activities(self):
        """
        Identify cases where the same activity occurs
        multiple times.
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
            .reset_index(name="count")
        )

        repeated = repeated[
            repeated["count"] > 1
        ]

        return repeated.sort_values(
            by="count",
            ascending=False,
        )

    # =========================================================
    # LOOP ANALYSIS
    # =========================================================

    def loop_statistics(self):
        """Calculate statistics related to repeated activities."""

        repeated = self.repeated_activities()

        cases_with_repetition = (
            repeated[self.case_column]
            .nunique()
        )

        total_cases = (
            self.log[
                self.case_column
            ].nunique()
        )

        percentage_cases_with_repetition = (
            cases_with_repetition
            / total_cases
            * 100
        )

        maximum_repetition = (
            repeated["count"].max()
            if not repeated.empty
            else 0
        )

        return {
            "cases_with_repeated_activities": (
                cases_with_repetition
            ),
            "percentage_cases_with_repetition": (
                percentage_cases_with_repetition
            ),
            "maximum_activity_repetition": (
                maximum_repetition
            ),
        }

    # =========================================================
    # CASE DURATION
    # =========================================================

    def case_duration_statistics(self):
        """Calculate process case duration statistics."""

        case_times = (
            self.log
            .groupby(self.case_column)[
                self.timestamp_column
            ]
            .agg(
                [
                    "min",
                    "max",
                ]
            )
        )

        case_times["duration"] = (
            case_times["max"]
            - case_times["min"]
        )

        duration_hours = (
            case_times["duration"]
            .dt.total_seconds()
            / 3600
        )

        return {
            "average_duration_hours": (
                duration_hours.mean()
            ),
            "median_duration_hours": (
                duration_hours.median()
            ),
            "minimum_duration_hours": (
                duration_hours.min()
            ),
            "maximum_duration_hours": (
                duration_hours.max()
            ),
        }

    # =========================================================
    # COMPLETE DIAGNOSTIC REPORT
    # =========================================================

    def generate_report(self):
        """
        Generate a complete diagnostic report.
        """

        print("\n" + "=" * 60)
        print("APMOA - EVENT LOG DIAGNOSTICS")
        print("=" * 60)

        # -----------------------------------------------------
        # BASIC STATISTICS
        # -----------------------------------------------------

        basic = self.basic_statistics()

        print("\n" + "-" * 60)
        print("BASIC STATISTICS")
        print("-" * 60)

        for key, value in basic.items():

            if isinstance(value, float):
                print(
                    f"{key:35}: {value:.2f}"
                )
            else:
                print(
                    f"{key:35}: {value}"
                )

        # -----------------------------------------------------
        # VARIANT STATISTICS
        # -----------------------------------------------------

        variants = self.variant_statistics()

        print("\n" + "-" * 60)
        print("VARIANT STATISTICS")
        print("-" * 60)

        for key, value in variants.items():

            if isinstance(value, float):
                print(
                    f"{key:35}: {value:.2f}"
                )
            else:
                print(
                    f"{key:35}: {value}"
                )

        # -----------------------------------------------------
        # LOOP STATISTICS
        # -----------------------------------------------------

        loops = self.loop_statistics()

        print("\n" + "-" * 60)
        print("LOOP / REPETITION STATISTICS")
        print("-" * 60)

        for key, value in loops.items():

            if isinstance(value, float):
                print(
                    f"{key:35}: {value:.2f}"
                )
            else:
                print(
                    f"{key:35}: {value}"
                )

        # -----------------------------------------------------
        # CASE DURATION
        # -----------------------------------------------------

        duration = self.case_duration_statistics()

        print("\n" + "-" * 60)
        print("CASE DURATION")
        print("-" * 60)

        for key, value in duration.items():

            print(
                f"{key:35}: {value:.2f}"
            )

        # -----------------------------------------------------
        # TOP ACTIVITIES
        # -----------------------------------------------------

        activities = self.activity_frequency()

        print("\n" + "-" * 60)
        print("TOP 10 ACTIVITIES")
        print("-" * 60)

        print(
            activities.head(10).to_string(
                index=False
            )
        )

        # -----------------------------------------------------
        # TOP VARIANTS
        # -----------------------------------------------------

        variants_table = self.variant_analysis()

        print("\n" + "-" * 60)
        print("TOP 10 PROCESS VARIANTS")
        print("-" * 60)

        for index, row in variants_table.head(10).iterrows():

            print(
                f"\nVariant {index + 1}"
            )

            print(
                f"Cases      : {row['cases']}"
            )

            print(
                f"Percentage : "
                f"{row['percentage']:.2f}%"
            )

            print(
                "Sequence   :"
            )

            for activity in row["variant"]:

                print(
                    f"  → {activity}"
                )

        # -----------------------------------------------------
        # COMPLETION
        # -----------------------------------------------------

        print("\n" + "=" * 60)
        print("DIAGNOSTIC ANALYSIS COMPLETED")
        print("=" * 60)