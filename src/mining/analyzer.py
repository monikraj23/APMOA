import pandas as pd


class ProcessAnalyzer:
    """
    Performs basic process analysis on a standardized event log.

    Expected columns:
        case_id
        activity
        timestamp
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

        self._validate()
        self._prepare()

    def _validate(self):
        """Validate the required columns."""

        required_columns = [
            self.case_column,
            self.activity_column,
            self.timestamp_column,
        ]

        missing = [
            column
            for column in required_columns
            if column not in self.log.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

    def _prepare(self):
        """Prepare timestamps and sort events."""

        self.log[self.timestamp_column] = pd.to_datetime(
            self.log[self.timestamp_column]
        )

        self.log = self.log.sort_values(
            by=[
                self.case_column,
                self.timestamp_column
            ]
        ).reset_index(drop=True)

    def get_basic_statistics(self) -> dict:
        """Return basic process statistics."""

        return {
            "cases": self.log[self.case_column].nunique(),
            "events": len(self.log),
            "activities": self.log[
                self.activity_column
            ].nunique(),
        }

    def get_activity_frequency(self) -> pd.DataFrame:
        """Return activity frequency statistics."""

        frequency = (
            self.log[self.activity_column]
            .value_counts()
            .reset_index()
        )

        frequency.columns = [
            "activity",
            "frequency"
        ]

        frequency["percentage"] = (
            frequency["frequency"]
            / len(self.log)
            * 100
        )

        return frequency

    def get_case_lengths(self) -> pd.Series:
        """Return number of events for every case."""

        return (
            self.log
            .groupby(self.case_column)
            .size()
            .sort_values(ascending=False)
        )

    def get_case_duration(self) -> pd.Series:
        """Return duration of every case."""

        case_duration = (
            self.log
            .groupby(self.case_column)[
                self.timestamp_column
            ]
            .agg(["min", "max"])
        )

        duration = (
            case_duration["max"]
            - case_duration["min"]
        )

        return duration.sort_values(
            ascending=False
        )

    def get_variants(self) -> pd.DataFrame:
        """
        Identify process variants.

        A variant is a unique sequence of activities
        followed by a case.
        """

        variants = (
            self.log
            .groupby(self.case_column)[
                self.activity_column
            ]
            .apply(tuple)
        )

        variant_counts = (
            variants
            .value_counts()
            .reset_index()
        )

        variant_counts.columns = [
            "variant",
            "case_count"
        ]

        variant_counts["percentage"] = (
            variant_counts["case_count"]
            / len(variants)
            * 100
        )

        return variant_counts

    def get_repeated_activities(self) -> pd.DataFrame:
        """
        Identify activities that occur multiple times
        within the same case.
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
            .reset_index(name="count")
        )

        repeated = activity_counts[
            activity_counts["count"] > 1
        ]

        return repeated.sort_values(
            "count",
            ascending=False
        )

    def get_summary(self) -> dict:
        """Generate an overall process summary."""

        basic = self.get_basic_statistics()

        case_lengths = self.get_case_lengths()
        case_duration = self.get_case_duration()
        variants = self.get_variants()

        return {
            **basic,
            "average_events_per_case": (
                case_lengths.mean()
            ),
            "average_case_duration": (
                case_duration.mean()
            ),
            "number_of_variants": len(variants),
        }