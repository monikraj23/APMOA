import pandas as pd


class DataAnalysisAgent:
    """
    APMOA - Data Analysis Agent

    Responsible for:
        1. Basic event-log statistics
        2. Variant analysis
        3. Activity-frequency analysis
        4. Case-duration analysis
        5. Repetition / loop analysis

    The agent accepts both:

    Standardized format:
        case_id
        activity
        timestamp

    PM4Py/XES format:
        case:concept:name
        concept:name
        time:timestamp
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

        # Convert PM4Py/XES columns if necessary
        self._standardize_columns()

        self._validate()
        self._prepare()

    # =========================================================
    # COLUMN STANDARDIZATION
    # =========================================================

    def _standardize_columns(self):
        """
        Convert common PM4Py/XES column names into the
        standardized APMOA column names.
        """

        rename_map = {}

        # -----------------------------------------------------
        # CASE ID
        # -----------------------------------------------------

        if self.case_column not in self.log.columns:

            possible_case_columns = [
                "case:concept:name",
                "concept:name:case",
                "case",
                "case_id",
            ]

            for column in possible_case_columns:

                if column in self.log.columns:
                    rename_map[column] = self.case_column
                    break

        # -----------------------------------------------------
        # ACTIVITY
        # -----------------------------------------------------

        if self.activity_column not in self.log.columns:

            possible_activity_columns = [
                "concept:name",
                "activity",
                "event",
                "activity_name",
            ]

            for column in possible_activity_columns:

                if column in self.log.columns:
                    rename_map[column] = self.activity_column
                    break

        # -----------------------------------------------------
        # TIMESTAMP
        # -----------------------------------------------------

        if self.timestamp_column not in self.log.columns:

            possible_timestamp_columns = [
                "time:timestamp",
                "timestamp",
                "event_timestamp",
                "time",
            ]

            for column in possible_timestamp_columns:

                if column in self.log.columns:
                    rename_map[column] = self.timestamp_column
                    break

        # Apply renaming
        if rename_map:
            self.log = self.log.rename(
                columns=rename_map
            )

    # =========================================================
    # VALIDATION
    # =========================================================

    def _validate(self):
        """
        Validate required event-log columns.
        """

        required_columns = [
            self.case_column,
            self.activity_column,
            self.timestamp_column,
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in self.log.columns
        ]

        if missing_columns:

            print("\nAvailable columns:")
            print(list(self.log.columns))

            raise ValueError(
                f"\nMissing required columns: "
                f"{missing_columns}\n"
                f"\nThe Data Analysis Agent expects:\n"
                f"  - {self.case_column}\n"
                f"  - {self.activity_column}\n"
                f"  - {self.timestamp_column}"
            )

    # =========================================================
    # PREPARATION
    # =========================================================

    def _prepare(self):
        """
        Prepare event log for analysis.
        """

        self.log[self.timestamp_column] = pd.to_datetime(
            self.log[self.timestamp_column],
            errors="coerce",
        )

        # Remove invalid rows
        self.log = self.log.dropna(
            subset=[
                self.case_column,
                self.activity_column,
                self.timestamp_column,
            ]
        )

        # Sort event log
        self.log = self.log.sort_values(
            by=[
                self.case_column,
                self.timestamp_column,
            ]
        ).reset_index(drop=True)

    # =========================================================
    # BASIC STATISTICS
    # =========================================================

    def calculate_basic_statistics(self):
        """
        Calculate basic event-log statistics.
        """

        events = len(self.log)

        cases = self.log[
            self.case_column
        ].nunique()

        activities = self.log[
            self.activity_column
        ].nunique()

        events_per_case = (
            self.log
            .groupby(self.case_column)
            .size()
        )

        return {
            "events": int(events),

            "cases": int(cases),

            "activities": int(activities),

            "average_events_per_case": round(
                float(events_per_case.mean()),
                2,
            ),

            "median_events_per_case": round(
                float(events_per_case.median()),
                2,
            ),

            "min_events_per_case": int(
                events_per_case.min()
            ),

            "max_events_per_case": int(
                events_per_case.max()
            ),
        }

    # =========================================================
    # VARIANT STATISTICS
    # =========================================================

    def calculate_variant_statistics(self):
        """
        Calculate process variant statistics.

        A variant represents the ordered sequence of
        activities executed by a case.
        """

        case_sequences = (
            self.log
            .groupby(self.case_column)[
                self.activity_column
            ]
            .apply(tuple)
        )

        variant_counts = case_sequences.value_counts()

        total_cases = len(case_sequences)

        number_of_variants = len(
            variant_counts
        )

        singleton_variants = int(
            (variant_counts == 1).sum()
        )

        rare_variants = int(
            (variant_counts <= 5).sum()
        )

        if len(variant_counts) > 0:

            dominant_variant_cases = int(
                variant_counts.iloc[0]
            )

        else:

            dominant_variant_cases = 0

        if total_cases > 0:

            dominant_variant_percentage = (
                dominant_variant_cases
                / total_cases
                * 100
            )

        else:

            dominant_variant_percentage = 0

        return {

            "number_of_variants":
                int(number_of_variants),

            "singleton_variants":
                singleton_variants,

            "rare_variants_<=5_cases":
                rare_variants,

            "dominant_variant_cases":
                dominant_variant_cases,

            "dominant_variant_percentage":
                round(
                    dominant_variant_percentage,
                    2,
                ),
        }

    # =========================================================
    # ACTIVITY STATISTICS
    # =========================================================

    def calculate_activity_statistics(self):
        """
        Calculate activity frequency statistics.
        """

        activity_counts = (
            self.log[
                self.activity_column
            ]
            .value_counts()
        )

        total_events = len(self.log)

        activities = []

        for activity, frequency in (
            activity_counts.items()
        ):

            if total_events > 0:

                percentage = (
                    frequency
                    / total_events
                    * 100
                )

            else:

                percentage = 0

            activities.append(
                {
                    "activity": activity,

                    "frequency": int(
                        frequency
                    ),

                    "percentage": round(
                        percentage,
                        4,
                    ),
                }
            )

        return activities

    # =========================================================
    # CASE DURATION
    # =========================================================

    def calculate_case_duration(self):
        """
        Calculate case-duration statistics
        in hours.
        """

        case_times = (
            self.log
            .groupby(self.case_column)[
                self.timestamp_column
            ]
            .agg(["min", "max"])
        )

        durations = (
            case_times["max"]
            - case_times["min"]
        ).dt.total_seconds() / 3600

        if len(durations) == 0:

            return {
                "average_duration_hours": 0,
                "median_duration_hours": 0,
                "minimum_duration_hours": 0,
                "maximum_duration_hours": 0,
            }

        return {

            "average_duration_hours":
                round(
                    float(durations.mean()),
                    2,
                ),

            "median_duration_hours":
                round(
                    float(durations.median()),
                    2,
                ),

            "minimum_duration_hours":
                round(
                    float(durations.min()),
                    2,
                ),

            "maximum_duration_hours":
                round(
                    float(durations.max()),
                    2,
                ),
        }

    # =========================================================
    # REPETITION / LOOP ANALYSIS
    # =========================================================

    def calculate_repetition_statistics(self):
        """
        Identify repeated activities inside cases.
        """

        activity_counts = (
            self.log
            .groupby(
                [
                    self.case_column,
                    self.activity_column,
                ]
            )
            .size()
        )

        repeated_entries = (
            activity_counts[
                activity_counts > 1
            ]
        )

        repeated_cases = (
            repeated_entries
            .index
            .get_level_values(0)
            .nunique()
        )

        total_cases = (
            self.log[
                self.case_column
            ].nunique()
        )

        if total_cases > 0:

            percentage_repeated = (
                repeated_cases
                / total_cases
                * 100
            )

        else:

            percentage_repeated = 0

        if len(activity_counts) > 0:

            maximum_repetition = int(
                activity_counts.max()
            )

        else:

            maximum_repetition = 0

        return {

            "cases_with_repeated_activities":
                int(repeated_cases),

            "percentage_cases_with_repetition":
                round(
                    percentage_repeated,
                    2,
                ),

            "maximum_activity_repetition":
                maximum_repetition,
        }

    # =========================================================
    # COMPLETE ANALYSIS
    # =========================================================

    def analyze(self):
        """
        Execute the complete data-analysis pipeline.

        Returns structured output for downstream agents.
        """

        print("\n" + "=" * 60)
        print("APMOA - DATA ANALYSIS AGENT")
        print("=" * 60)

        print("\nAnalyzing event log...")

        basic_statistics = (
            self.calculate_basic_statistics()
        )

        variant_statistics = (
            self.calculate_variant_statistics()
        )

        activity_statistics = (
            self.calculate_activity_statistics()
        )

        case_duration = (
            self.calculate_case_duration()
        )

        repetition_statistics = (
            self.calculate_repetition_statistics()
        )

        print(
            "Data Analysis Agent completed."
        )

        return {

            "basic_statistics":
                basic_statistics,

            "variant_statistics":
                variant_statistics,

            "activity_statistics":
                activity_statistics,

            "case_duration":
                case_duration,

            "repetition_statistics":
                repetition_statistics,
        }