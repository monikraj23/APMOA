"""
APMOA - Bottleneck & Deviation Analysis Agent

Analyzes an event log to identify:

    - Activity frequency
    - Activity duration
    - Waiting time between activities
    - Repeated activities / rework
    - Long-running cases
    - Potential bottlenecks
    - Potential process deviations

Input
-----
PM4Py EventLog or pandas DataFrame.

Output
------
Dictionary containing:

    {
        "activity_analysis": [...],
        "transition_analysis": [...],
        "case_analysis": {...},
        "rework_analysis": [...],
        "bottlenecks": [...],
        "deviations": [...],
        "summary": {...}
    }

Important
---------
This agent reports "potential" bottlenecks and deviations.
It does not claim causal relationships.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from typing import Any, Dict, List

import pandas as pd


class BottleneckAnalysisAgent:
    """
    APMOA Bottleneck & Deviation Analysis Agent.

    The agent is intentionally independent from:

        - Process Discovery
        - Model Evaluation
        - Model Selection
        - Reasoning
        - Adaptation

    This allows it to be tested independently before being
    integrated into the autonomous orchestration loop.
    """

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        log=None,
        activity_key: str = "concept:name",
        timestamp_key: str = "time:timestamp",
        case_id_key: str = "case:concept:name",
        bottleneck_threshold: float = 0.75,
        deviation_threshold: float = 0.75,
    ):
        self.log = log

        self.activity_key = activity_key
        self.timestamp_key = timestamp_key
        self.case_id_key = case_id_key

        self.bottleneck_threshold = (
            bottleneck_threshold
        )

        self.deviation_threshold = (
            deviation_threshold
        )

    # =========================================================
    # SAFE FLOAT
    # =========================================================

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:

        try:

            value = float(value)

            if math.isnan(value):
                return default

            if math.isinf(value):
                return default

            return value

        except (
            TypeError,
            ValueError,
        ):

            return default

    # =========================================================
    # PERCENTILE
    # =========================================================

    @staticmethod
    def _percentile(
        values: List[float],
        percentile: float,
    ) -> float:

        if not values:
            return 0.0

        values = sorted(values)

        if len(values) == 1:
            return values[0]

        position = (
            percentile / 100.0
        ) * (len(values) - 1)

        lower = int(
            math.floor(position)
        )

        upper = int(
            math.ceil(position)
        )

        if lower == upper:
            return values[lower]

        weight = position - lower

        return (
            values[lower]
            + weight
            * (
                values[upper]
                - values[lower]
            )
        )

    # =========================================================
    # EVENT LOG -> DATAFRAME
    # =========================================================

    def _to_dataframe(self) -> pd.DataFrame:
        """
        Convert a PM4Py EventLog or pandas DataFrame into the
        internal PM4Py-compatible schema used by this agent.

        Supported input schemas:

        1. Canonical APMOA schema:
           - case_id
           - activity
           - timestamp
           - resource (optional)

        2. Native PM4Py/XES schema:
           - case:concept:name
           - concept:name
           - time:timestamp
           - org:resource (optional)

        This normalization keeps the bottleneck agent compatible with
        the dataset-independent EventLogLoader and orchestrator.
        """

        if self.log is None:
            raise ValueError("Event log is required.")

        # ---------------------------------------------------------
        # Convert input to DataFrame
        # ---------------------------------------------------------
        if isinstance(self.log, pd.DataFrame):
            dataframe = self.log.copy()
        else:
            try:
                dataframe = pd.DataFrame(list(self.log))
            except Exception as exc:
                raise TypeError(
                    "Unable to convert event log to pandas DataFrame."
                ) from exc

        if dataframe.empty:
            raise ValueError("Event log is empty.")

        # ---------------------------------------------------------
        # Normalize canonical APMOA schema -> PM4Py schema
        # ---------------------------------------------------------
        canonical_columns = {
            "case_id",
            "activity",
            "timestamp",
        }

        if canonical_columns.issubset(dataframe.columns):
            rename_map = {
                "case_id": self.case_id_key,
                "activity": self.activity_key,
                "timestamp": self.timestamp_key,
            }

            if "resource" in dataframe.columns:
                rename_map["resource"] = "org:resource"

            dataframe = dataframe.rename(columns=rename_map)

        # ---------------------------------------------------------
        # Normalize common alternative schema if it reaches this
        # agent directly without passing through DatasetAdapter.
        # ---------------------------------------------------------
        else:
            alternative_mapping = {
                "case": self.case_id_key,
                "case_id": self.case_id_key,
                "Case ID": self.case_id_key,
                "activity": self.activity_key,
                "Activity": self.activity_key,
                "event": self.activity_key,
                "timestamp": self.timestamp_key,
                "Timestamp": self.timestamp_key,
                "time": self.timestamp_key,
                "resource": "org:resource",
                "Resource": "org:resource",
            }

            for source, target in alternative_mapping.items():
                if source in dataframe.columns and target not in dataframe.columns:
                    dataframe = dataframe.rename(columns={source: target})

        # ---------------------------------------------------------
        # Required PM4Py columns
        # ---------------------------------------------------------
        required_columns = [
            self.activity_key,
            self.timestamp_key,
            self.case_id_key,
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                "Event log is missing required columns: "
                f"{missing_columns}. Expected either the canonical "
                "APMOA schema (case_id, activity, timestamp) or "
                "the PM4Py/XES schema "
                "(case:concept:name, concept:name, time:timestamp)."
            )

        dataframe = dataframe.copy()

        # ---------------------------------------------------------
        # Ensure string identifiers / activities
        # ---------------------------------------------------------
        dataframe[self.case_id_key] = dataframe[self.case_id_key].astype(str)
        dataframe[self.activity_key] = dataframe[self.activity_key].astype(str)

        # ---------------------------------------------------------
        # Timestamp normalization
        # ---------------------------------------------------------
        dataframe[self.timestamp_key] = pd.to_datetime(
            dataframe[self.timestamp_key],
            errors="coerce",
        )

        # ---------------------------------------------------------
        # Resource is optional for bottleneck/deviation analysis.
        # Add it for schema consistency when it is absent.
        # ---------------------------------------------------------
        if "org:resource" not in dataframe.columns:
            dataframe["org:resource"] = "UNKNOWN"
        else:
            dataframe["org:resource"] = (
                dataframe["org:resource"]
                .fillna("UNKNOWN")
                .astype(str)
            )

        # ---------------------------------------------------------
        # Remove invalid events
        # ---------------------------------------------------------
        dataframe = dataframe.dropna(
            subset=[
                self.activity_key,
                self.timestamp_key,
                self.case_id_key,
            ]
        )

        if dataframe.empty:
            raise ValueError(
                "Event log contains no valid events after timestamp/ID "
                "normalization."
            )

        # ---------------------------------------------------------
        # Sort chronologically within each case
        # ---------------------------------------------------------
        dataframe = (
            dataframe
            .sort_values(
                by=[self.case_id_key, self.timestamp_key],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

        print("\nEvent log normalized for bottleneck analysis:")
        print(f"  Events    : {len(dataframe):,}")
        print(f"  Cases     : {dataframe[self.case_id_key].nunique():,}")
        print(f"  Activities: {dataframe[self.activity_key].nunique():,}")
        print(f"  Resources : {dataframe['org:resource'].nunique():,}")

        return dataframe

    # =========================================================
    # VALIDATION
    # =========================================================

    def _validate_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> None:

        if dataframe.empty:

            raise ValueError(
                "Event log is empty."
            )

        if (
            dataframe[
                self.case_id_key
            ].nunique()
            == 0
        ):

            raise ValueError(
                "No process cases found."
            )

        if (
            dataframe[
                self.activity_key
            ].nunique()
            == 0
        ):

            raise ValueError(
                "No activities found."
            )

    # =========================================================
    # ACTIVITY ANALYSIS
    # =========================================================

    def _analyze_activities(
        self,
        dataframe: pd.DataFrame,
    ) -> List[Dict[str, Any]]:

        activity_groups = dataframe.groupby(
            self.activity_key,
            sort=False,
        )

        total_events = len(dataframe)

        results = []

        for (
            activity,
            group,
        ) in activity_groups:

            frequency = len(group)

            case_count = (
                group[
                    self.case_id_key
                ].nunique()
            )

            # -------------------------------------------------
            # Activity timing
            # -------------------------------------------------
            #
            # The event log does not necessarily contain
            # lifecycle start/complete records.
            #
            # Therefore, activity duration is not fabricated.
            #
            # We use consecutive occurrences of the same
            # activity within a case as an approximate
            # activity timing signal.
            # -------------------------------------------------

            duration_values = []

            for (
                _,
                case_group,
            ) in group.groupby(
                self.case_id_key,
                sort=False,
            ):

                timestamps = (
                    case_group[
                        self.timestamp_key
                    ]
                    .sort_values()
                    .tolist()
                )

                if len(timestamps) > 1:

                    for index in range(
                        len(timestamps) - 1
                    ):

                        delta = (
                            timestamps[
                                index + 1
                            ]
                            - timestamps[index]
                        ).total_seconds()

                        delta_hours = (
                            delta / 3600.0
                        )

                        if delta_hours >= 0:

                            duration_values.append(
                                delta_hours
                            )

            average_duration = (
                statistics.mean(
                    duration_values
                )
                if duration_values
                else 0.0
            )

            median_duration = (
                statistics.median(
                    duration_values
                )
                if duration_values
                else 0.0
            )

            maximum_duration = (
                max(duration_values)
                if duration_values
                else 0.0
            )

            # -------------------------------------------------
            # Repetition
            # -------------------------------------------------

            repetition_count = 0

            for (
                _,
                case_group,
            ) in group.groupby(
                self.case_id_key,
                sort=False,
            ):

                count = len(case_group)

                if count > 1:

                    repetition_count += (
                        count - 1
                    )

            results.append(
                {
                    "activity":
                        str(activity),

                    "frequency":
                        int(frequency),

                    "percentage":
                        round(
                            (
                                frequency
                                / total_events
                                * 100.0
                            ),
                            2,
                        ),

                    "case_count":
                        int(case_count),

                    "average_duration_hours":
                        round(
                            average_duration,
                            2,
                        ),

                    "median_duration_hours":
                        round(
                            median_duration,
                            2,
                        ),

                    "maximum_duration_hours":
                        round(
                            maximum_duration,
                            2,
                        ),

                    "repetition_count":
                        int(
                            repetition_count
                        ),
                }
            )

        # -----------------------------------------------------
        # Sort by frequency
        # -----------------------------------------------------

        results.sort(
            key=lambda item: (
                item["frequency"],
                item[
                    "average_duration_hours"
                ],
            ),
            reverse=True,
        )

        return results

    # =========================================================
    # TRANSITION ANALYSIS
    # =========================================================

    def _analyze_transitions(
        self,
        dataframe: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """
        Analyze transitions between consecutive events
        within each case.

        IMPORTANT FIX
        -------------
        This implementation deliberately uses DataFrame
        column access rather than itertuples().

        PM4Py event-log columns contain names such as:

            concept:name
            time:timestamp
            case:concept:name

        These are not safe to access as Python tuple
        attributes.

        The previous implementation therefore could silently
        fail to identify transitions.

        This implementation directly accesses the DataFrame
        columns and guarantees transition extraction.
        """

        transitions = defaultdict(list)

        # -----------------------------------------------------
        # Process each case independently
        # -----------------------------------------------------

        for (
            case_id,
            case_group,
        ) in dataframe.groupby(
            self.case_id_key,
            sort=False,
        ):

            case_group = (
                case_group
                .sort_values(
                    self.timestamp_key
                )
                .reset_index(
                    drop=True
                )
            )

            # -------------------------------------------------
            # Iterate through consecutive rows by position.
            #
            # This avoids problems with special PM4Py column
            # names.
            # -------------------------------------------------

            for index in range(
                len(case_group) - 1
            ):

                current_activity = (
                    case_group.iloc[
                        index
                    ][
                        self.activity_key
                    ]
                )

                next_activity = (
                    case_group.iloc[
                        index + 1
                    ][
                        self.activity_key
                    ]
                )

                current_timestamp = (
                    case_group.iloc[
                        index
                    ][
                        self.timestamp_key
                    ]
                )

                next_timestamp = (
                    case_group.iloc[
                        index + 1
                    ][
                        self.timestamp_key
                    ]
                )

                # -------------------------------------------------
                # Validate values
                # -------------------------------------------------

                if (
                    pd.isna(
                        current_activity
                    )
                    or pd.isna(
                        next_activity
                    )
                    or pd.isna(
                        current_timestamp
                    )
                    or pd.isna(
                        next_timestamp
                    )
                ):

                    continue

                # -------------------------------------------------
                # Calculate waiting time
                # -------------------------------------------------

                try:

                    waiting_seconds = (
                        next_timestamp
                        - current_timestamp
                    ).total_seconds()

                except Exception:

                    continue

                if waiting_seconds < 0:
                    continue

                waiting_hours = (
                    waiting_seconds
                    / 3600.0
                )

                transition_key = (
                    str(
                        current_activity
                    ),
                    str(
                        next_activity
                    ),
                )

                transitions[
                    transition_key
                ].append(
                    waiting_hours
                )

        # -----------------------------------------------------
        # Build transition results
        # -----------------------------------------------------

        results = []

        for (
            transition,
            values,
        ) in transitions.items():

            source_activity = (
                transition[0]
            )

            target_activity = (
                transition[1]
            )

            if not values:
                continue

            results.append(
                {
                    "from_activity":
                        source_activity,

                    "to_activity":
                        target_activity,

                    "occurrences":
                        int(
                            len(values)
                        ),

                    "average_waiting_hours":
                        round(
                            statistics.mean(
                                values
                            ),
                            2,
                        ),

                    "median_waiting_hours":
                        round(
                            statistics.median(
                                values
                            ),
                            2,
                        ),

                    "p75_waiting_hours":
                        round(
                            self._percentile(
                                values,
                                75,
                            ),
                            2,
                        ),

                    "maximum_waiting_hours":
                        round(
                            max(values),
                            2,
                        ),
                }
            )

        # -----------------------------------------------------
        # Highest waiting time first
        # -----------------------------------------------------

        results.sort(
            key=lambda item: (
                item[
                    "average_waiting_hours"
                ],
                item[
                    "occurrences"
                ],
            ),
            reverse=True,
        )

        return results

    # =========================================================
    # CASE ANALYSIS
    # =========================================================

    def _analyze_cases(
        self,
        dataframe: pd.DataFrame,
    ) -> Dict[str, Any]:

        case_durations = []

        case_event_counts = []

        case_details = []

        for (
            case_id,
            group,
        ) in dataframe.groupby(
            self.case_id_key,
            sort=False,
        ):

            group = (
                group
                .sort_values(
                    self.timestamp_key
                )
                .reset_index(
                    drop=True
                )
            )

            timestamps = (
                group[
                    self.timestamp_key
                ].tolist()
            )

            if not timestamps:
                continue

            if len(timestamps) > 1:

                duration_seconds = (
                    timestamps[-1]
                    - timestamps[0]
                ).total_seconds()

                duration_hours = (
                    duration_seconds
                    / 3600.0
                )

            else:

                duration_hours = 0.0

            event_count = len(group)

            case_durations.append(
                duration_hours
            )

            case_event_counts.append(
                event_count
            )

            case_details.append(
                {
                    "case_id":
                        str(case_id),

                    "duration_hours":
                        round(
                            duration_hours,
                            2,
                        ),

                    "event_count":
                        int(
                            event_count
                        ),

                    "unique_activities":
                        int(
                            group[
                                self.activity_key
                            ].nunique()
                        ),
                }
            )

        # -----------------------------------------------------
        # Longest cases
        # -----------------------------------------------------

        case_details.sort(
            key=lambda item:
                item[
                    "duration_hours"
                ],
            reverse=True,
        )

        return {
            "number_of_cases":
                int(
                    len(
                        case_durations
                    )
                ),

            "average_case_duration_hours":
                round(
                    (
                        statistics.mean(
                            case_durations
                        )
                        if case_durations
                        else 0.0
                    ),
                    2,
                ),

            "median_case_duration_hours":
                round(
                    (
                        statistics.median(
                            case_durations
                        )
                        if case_durations
                        else 0.0
                    ),
                    2,
                ),

            "p75_case_duration_hours":
                round(
                    self._percentile(
                        case_durations,
                        75,
                    ),
                    2,
                ),

            "p95_case_duration_hours":
                round(
                    self._percentile(
                        case_durations,
                        95,
                    ),
                    2,
                ),

            "maximum_case_duration_hours":
                round(
                    (
                        max(
                            case_durations
                        )
                        if case_durations
                        else 0.0
                    ),
                    2,
                ),

            "average_events_per_case":
                round(
                    (
                        statistics.mean(
                            case_event_counts
                        )
                        if case_event_counts
                        else 0.0
                    ),
                    2,
                ),

            "longest_cases":
                case_details[:10],
        }

    # =========================================================
    # REWORK ANALYSIS
    # =========================================================

    def _analyze_rework(
        self,
        dataframe: pd.DataFrame,
    ) -> List[Dict[str, Any]]:

        activity_case_counts = (
            defaultdict(int)
        )

        repetition_counts = (
            defaultdict(int)
        )

        for (
            case_id,
            group,
        ) in dataframe.groupby(
            self.case_id_key,
            sort=False,
        ):

            counts = Counter(
                group[
                    self.activity_key
                ].astype(str)
            )

            for (
                activity,
                count,
            ) in counts.items():

                activity_case_counts[
                    activity
                ] += 1

                if count > 1:

                    repetition_counts[
                        activity
                    ] += (
                        count - 1
                    )

        results = []

        for activity in (
            activity_case_counts
        ):

            repetitions = (
                repetition_counts.get(
                    activity,
                    0,
                )
            )

            if repetitions <= 0:
                continue

            results.append(
                {
                    "activity":
                        activity,

                    "cases_containing_activity":
                        int(
                            activity_case_counts[
                                activity
                            ]
                        ),

                    "repetition_count":
                        int(
                            repetitions
                        ),
                }
            )

        results.sort(
            key=lambda item:
                item[
                    "repetition_count"
                ],
            reverse=True,
        )

        return results

    # =========================================================
    # BOTTLENECK DETECTION
    # =========================================================

    def _identify_bottlenecks(
        self,
        activity_analysis,
        transition_analysis,
    ) -> List[Dict[str, Any]]:
        """
        Identify potential bottleneck activities.

        Indicators:

            1. High waiting time
            2. High frequency
            3. High duration
            4. Rework / repetition

        An activity must satisfy at least two indicators
        before being classified as a potential bottleneck.
        """

        bottlenecks = []

        # -----------------------------------------------------
        # Waiting time by target activity
        # -----------------------------------------------------

        activity_waiting = defaultdict(
            list
        )

        for transition in (
            transition_analysis
        ):

            target_activity = (
                transition[
                    "to_activity"
                ]
            )

            activity_waiting[
                target_activity
            ].append(
                transition[
                    "average_waiting_hours"
                ]
            )

        waiting_values = []

        for values in (
            activity_waiting.values()
        ):

            if values:

                waiting_values.append(
                    statistics.mean(
                        values
                    )
                )

        waiting_p75 = (
            self._percentile(
                waiting_values,
                75,
            )
        )

        # -----------------------------------------------------
        # Frequency threshold
        # -----------------------------------------------------

        frequency_values = [
            item[
                "frequency"
            ]
            for item in activity_analysis
        ]

        frequency_p75 = (
            self._percentile(
                frequency_values,
                75,
            )
        )

        # -----------------------------------------------------
        # Duration threshold
        # -----------------------------------------------------

        duration_values = [
            item[
                "average_duration_hours"
            ]
            for item in activity_analysis
            if item[
                "average_duration_hours"
            ] > 0
        ]

        duration_p75 = (
            self._percentile(
                duration_values,
                75,
            )
        )

        # -----------------------------------------------------
        # Evaluate every activity
        # -----------------------------------------------------

        for activity in (
            activity_analysis
        ):

            activity_name = (
                activity[
                    "activity"
                ]
            )

            waiting_values_for_activity = (
                activity_waiting.get(
                    activity_name,
                    [],
                )
            )

            average_waiting = (
                statistics.mean(
                    waiting_values_for_activity
                )
                if waiting_values_for_activity
                else 0.0
            )

            high_waiting = (
                average_waiting
                >= waiting_p75
                and average_waiting
                > 0
            )

            high_frequency = (
                activity[
                    "frequency"
                ]
                >= frequency_p75
            )

            high_duration = (
                activity[
                    "average_duration_hours"
                ]
                >= duration_p75
                and activity[
                    "average_duration_hours"
                ]
                > 0
            )

            repeated = (
                activity[
                    "repetition_count"
                ]
                > 0
            )

            indicators = sum(
                [
                    high_waiting,
                    high_frequency,
                    high_duration,
                    repeated,
                ]
            )

            # -------------------------------------------------
            # Require at least two indicators
            # -------------------------------------------------

            if indicators < 2:
                continue

            reasons = []

            if high_waiting:

                reasons.append(
                    "high waiting time"
                )

            if high_frequency:

                reasons.append(
                    "high activity frequency"
                )

            if high_duration:

                reasons.append(
                    "high average duration"
                )

            if repeated:

                reasons.append(
                    "repeated activity / rework"
                )

            bottleneck_score = (
                indicators / 4.0
            )

            bottlenecks.append(
                {
                    "activity":
                        activity_name,

                    "bottleneck_score":
                        round(
                            bottleneck_score,
                            2,
                        ),

                    "indicators":
                        int(
                            indicators
                        ),

                    "average_waiting_hours":
                        round(
                            average_waiting,
                            2,
                        ),

                    "average_duration_hours":
                        activity[
                            "average_duration_hours"
                        ],

                    "frequency":
                        activity[
                            "frequency"
                        ],

                    "repetition_count":
                        activity[
                            "repetition_count"
                        ],

                    "reasons":
                        reasons,
                }
            )

        # -----------------------------------------------------
        # Rank bottlenecks
        # -----------------------------------------------------

        bottlenecks.sort(
            key=lambda item: (
                item[
                    "bottleneck_score"
                ],
                item[
                    "average_waiting_hours"
                ],
                item[
                    "frequency"
                ],
            ),
            reverse=True,
        )

        return bottlenecks

    # =========================================================
    # VARIANT ANALYSIS
    # =========================================================

    def _extract_variants(
        self,
        dataframe: pd.DataFrame,
    ) -> Dict[tuple, int]:

        variants = defaultdict(int)

        for (
            case_id,
            group,
        ) in dataframe.groupby(
            self.case_id_key,
            sort=False,
        ):

            group = (
                group
                .sort_values(
                    self.timestamp_key
                )
            )

            activities = tuple(
                group[
                    self.activity_key
                ]
                .astype(str)
                .tolist()
            )

            if activities:

                variants[
                    activities
                ] += 1

        return variants

    # =========================================================
    # DEVIATION ANALYSIS
    # =========================================================

    def _identify_deviations(
        self,
        dataframe: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """
        Identify infrequent variants that differ substantially
        from the dominant process variant.

        This is a heuristic deviation detector.

        It should therefore be interpreted as:

            "potential deviation"

        rather than a conformance/alignment diagnosis.
        """

        variants = (
            self._extract_variants(
                dataframe
            )
        )

        if not variants:
            return []

        dominant_variant, dominant_count = (
            max(
                variants.items(),
                key=lambda item:
                    item[1],
            )
        )

        total_cases = sum(
            variants.values()
        )

        if total_cases <= 0:
            return []

        dominant_percentage = (
            dominant_count
            / total_cases
            * 100.0
        )

        deviations = []

        for (
            variant,
            count,
        ) in variants.items():

            if variant == dominant_variant:
                continue

            percentage = (
                count
                / total_cases
                * 100.0
            )

            max_length = max(
                len(variant),
                len(dominant_variant),
            )

            if max_length == 0:
                continue

            differences = 0

            for index in range(
                max_length
            ):

                dominant_activity = (
                    dominant_variant[
                        index
                    ]
                    if index
                    < len(
                        dominant_variant
                    )
                    else None
                )

                variant_activity = (
                    variant[
                        index
                    ]
                    if index
                    < len(variant)
                    else None
                )

                if (
                    dominant_activity
                    != variant_activity
                ):

                    differences += 1

            deviation_ratio = (
                differences
                / max_length
            )

            if (
                deviation_ratio
                < self.deviation_threshold
            ):
                continue

            deviations.append(
                {
                    "variant":
                        " -> ".join(
                            variant
                        ),

                    "case_count":
                        int(
                            count
                        ),

                    "percentage":
                        round(
                            percentage,
                            2,
                        ),

                    "sequence_length":
                        int(
                            len(variant)
                        ),

                    "difference_from_dominant":
                        int(
                            differences
                        ),

                    "deviation_ratio":
                        round(
                            deviation_ratio,
                            2,
                        ),

                    "dominant_variant_percentage":
                        round(
                            dominant_percentage,
                            2,
                        ),
                }
            )

        # -----------------------------------------------------
        # Most significant deviations first
        # -----------------------------------------------------

        deviations.sort(
            key=lambda item: (
                item[
                    "deviation_ratio"
                ],
                item[
                    "case_count"
                ],
            ),
            reverse=True,
        )

        return deviations[:50]

    # =========================================================
    # SUMMARY
    # =========================================================

    def _build_summary(
        self,
        dataframe: pd.DataFrame,
        activity_analysis,
        transition_analysis,
        bottlenecks,
        deviations,
        case_analysis,
        rework,
    ) -> Dict[str, Any]:

        return {
            "events":
                int(
                    len(dataframe)
                ),

            "cases":
                int(
                    dataframe[
                        self.case_id_key
                    ].nunique()
                ),

            "activities":
                int(
                    dataframe[
                        self.activity_key
                    ].nunique()
                ),

            "transitions":
                int(
                    len(
                        transition_analysis
                    )
                ),

            "potential_bottlenecks":
                int(
                    len(
                        bottlenecks
                    )
                ),

            "potential_deviations":
                int(
                    len(
                        deviations
                    )
                ),

            "activities_with_rework":
                int(
                    len(
                        rework
                    )
                ),

            "longest_case_duration_hours":
                case_analysis[
                    "maximum_case_duration_hours"
                ],
        }

    # =========================================================
    # PUBLIC API
    # =========================================================

    def analyze(
        self,
    ) -> Dict[str, Any]:
        """
        Execute the complete bottleneck and deviation
        analysis pipeline.
        """

        # -----------------------------------------------------
        # Convert log
        # -----------------------------------------------------

        dataframe = (
            self._to_dataframe()
        )

        # -----------------------------------------------------
        # Validate
        # -----------------------------------------------------

        self._validate_dataframe(
            dataframe
        )

        print(
            "\n"
            + "=" * 60
        )

        print(
            "APMOA - BOTTLENECK & DEVIATION ANALYSIS"
        )

        print(
            "=" * 60
        )

        print(
            f"\nEvents    : "
            f"{len(dataframe):,}"
        )

        print(
            f"Cases     : "
            f"{dataframe[self.case_id_key].nunique():,}"
        )

        print(
            f"Activities: "
            f"{dataframe[self.activity_key].nunique():,}"
        )

        # -----------------------------------------------------
        # Activity analysis
        # -----------------------------------------------------

        print(
            "\nAnalyzing activities..."
        )

        activity_analysis = (
            self._analyze_activities(
                dataframe
            )
        )

        # -----------------------------------------------------
        # Transition analysis
        # -----------------------------------------------------

        print(
            "Analyzing transitions..."
        )

        transition_analysis = (
            self._analyze_transitions(
                dataframe
            )
        )

        # -----------------------------------------------------
        # Case analysis
        # -----------------------------------------------------

        print(
            "Analyzing cases..."
        )

        case_analysis = (
            self._analyze_cases(
                dataframe
            )
        )

        # -----------------------------------------------------
        # Rework
        # -----------------------------------------------------

        print(
            "Analyzing rework..."
        )

        rework = (
            self._analyze_rework(
                dataframe
            )
        )

        # -----------------------------------------------------
        # Bottlenecks
        # -----------------------------------------------------

        print(
            "Identifying bottlenecks..."
        )

        bottlenecks = (
            self._identify_bottlenecks(
                activity_analysis,
                transition_analysis,
            )
        )

        # -----------------------------------------------------
        # Deviations
        # -----------------------------------------------------

        print(
            "Identifying deviations..."
        )

        deviations = (
            self._identify_deviations(
                dataframe
            )
        )

        # -----------------------------------------------------
        # Summary
        # -----------------------------------------------------

        summary = (
            self._build_summary(
                dataframe,
                activity_analysis,
                transition_analysis,
                bottlenecks,
                deviations,
                case_analysis,
                rework,
            )
        )

        # -----------------------------------------------------
        # Print summary
        # -----------------------------------------------------

        print(
            "\n"
            + "-" * 60
        )

        print(
            "BOTTLENECK ANALYSIS SUMMARY"
        )

        print(
            "-" * 60
        )

        print(
            f"Potential bottlenecks : "
            f"{len(bottlenecks)}"
        )

        print(
            f"Potential deviations  : "
            f"{len(deviations)}"
        )

        print(
            f"Activities with rework : "
            f"{len(rework)}"
        )

        print(
            f"Transitions analyzed   : "
            f"{len(transition_analysis)}"
        )

        print(
            "\n"
            + "=" * 60
        )

        print(
            "BOTTLENECK & DEVIATION ANALYSIS COMPLETED"
        )

        print(
            "=" * 60
        )

        return {
            "activity_analysis":
                activity_analysis,

            "transition_analysis":
                transition_analysis,

            "case_analysis":
                case_analysis,

            "rework_analysis":
                rework,

            "bottlenecks":
                bottlenecks,

            "deviations":
                deviations,

            "summary":
                summary,
        }


# =============================================================
# STANDALONE TEST
# =============================================================

def main():

    from src.mining.loader import (
        EventLogLoader,
    )

    log_path = (
        "data/raw/Travel/PermitLog.xes.gz"
    )

    print(
        "=" * 60
    )

    print(
        "APMOA - BOTTLENECK ANALYSIS TEST"
    )

    print(
        "=" * 60
    )

    print(
        f"\nLoading event log: "
        f"{log_path}"
    )

    loader = EventLogLoader(
        log_path
    )

    log = loader.load()

    print(
        f"Loaded {len(log):,} events."
    )

    agent = BottleneckAnalysisAgent(
        log
    )

    results = agent.analyze()

    # ---------------------------------------------------------
    # Basic validation
    # ---------------------------------------------------------

    assert isinstance(
        results,
        dict,
    )

    required_sections = [
        "activity_analysis",
        "transition_analysis",
        "case_analysis",
        "rework_analysis",
        "bottlenecks",
        "deviations",
        "summary",
    ]

    for section in required_sections:

        assert section in results, (
            f"Missing result section: "
            f"{section}"
        )

    # ---------------------------------------------------------
    # Transition validation
    # ---------------------------------------------------------

    transition_count = len(
        results[
            "transition_analysis"
        ]
    )

    assert transition_count > 0, (
        "Transition analysis returned "
        "zero transitions. "
        "The event log should contain "
        "at least one consecutive event "
        "transition."
    )

    # ---------------------------------------------------------
    # Activity validation
    # ---------------------------------------------------------

    assert (
        len(
            results[
                "activity_analysis"
            ]
        )
        > 0
    )

    # ---------------------------------------------------------
    # Case validation
    # ---------------------------------------------------------

    assert (
        results[
            "case_analysis"
        ][
            "number_of_cases"
        ]
        > 0
    )

    # ---------------------------------------------------------
    # Summary validation
    # ---------------------------------------------------------

    summary = results[
        "summary"
    ]

    assert (
        summary[
            "events"
        ]
        > 0
    )

    assert (
        summary[
            "cases"
        ]
        > 0
    )

    assert (
        summary[
            "activities"
        ]
        > 0
    )

    assert (
        summary[
            "transitions"
        ]
        > 0
    )

    # ---------------------------------------------------------
    # Print analysis summary
    # ---------------------------------------------------------

    print(
        "\n"
        + "-" * 70
    )

    print(
        "ANALYSIS SUMMARY"
    )

    print(
        "-" * 70
    )

    print(
        f"{'Events':<35}: "
        f"{summary['events']:,}"
    )

    print(
        f"{'Cases':<35}: "
        f"{summary['cases']:,}"
    )

    print(
        f"{'Activities':<35}: "
        f"{summary['activities']:,}"
    )

    print(
        f"{'Transitions':<35}: "
        f"{summary['transitions']:,}"
    )

    print(
        f"{'Potential bottlenecks':<35}: "
        f"{summary['potential_bottlenecks']:,}"
    )

    print(
        f"{'Potential deviations':<35}: "
        f"{summary['potential_deviations']:,}"
    )

    print(
        f"{'Activities with rework':<35}: "
        f"{summary['activities_with_rework']:,}"
    )

    print(
        f"{'Longest case duration (hours)':<35}: "
        f"{summary['longest_case_duration_hours']:.2f}"
    )

    # ---------------------------------------------------------
    # Top transitions
    # ---------------------------------------------------------

    print(
        "\n"
        + "-" * 70
    )

    print(
        "TOP WAITING-TIME TRANSITIONS"
    )

    print(
        "-" * 70
    )

    for transition in results[
        "transition_analysis"
    ][:10]:

        print(
            f"{transition['from_activity']}"
            f" -> "
            f"{transition['to_activity']}"
        )

        print(
            f"  Occurrences       : "
            f"{transition['occurrences']:,}"
        )

        print(
            f"  Average waiting   : "
            f"{transition['average_waiting_hours']:.2f} hours"
        )

        print(
            f"  Median waiting    : "
            f"{transition['median_waiting_hours']:.2f} hours"
        )

        print(
            f"  P75 waiting       : "
            f"{transition['p75_waiting_hours']:.2f} hours"
        )

        print(
            f"  Maximum waiting   : "
            f"{transition['maximum_waiting_hours']:.2f} hours"
        )

    # ---------------------------------------------------------
    # Top bottlenecks
    # ---------------------------------------------------------

    print(
        "\n"
        + "-" * 70
    )

    print(
        "TOP BOTTLENECKS"
    )

    print(
        "-" * 70
    )

    for bottleneck in results[
        "bottlenecks"
    ][:10]:

        print(
            f"{bottleneck['activity']:<50}"
            f"score="
            f"{bottleneck['bottleneck_score']:.2f}"
        )

        print(
            f"  Reasons: "
            f"{', '.join(bottleneck['reasons'])}"
        )

    # ---------------------------------------------------------
    # Top rework
    # ---------------------------------------------------------

    print(
        "\n"
        + "-" * 70
    )

    print(
        "TOP REWORK ACTIVITIES"
    )

    print(
        "-" * 70
    )

    for item in results[
        "rework_analysis"
    ][:10]:

        print(
            f"{item['activity']:<50}"
            f"repetitions="
            f"{item['repetition_count']:,}"
        )

    # ---------------------------------------------------------
    # Top deviations
    # ---------------------------------------------------------

    print(
        "\n"
        + "-" * 70
    )

    print(
        "TOP POTENTIAL DEVIATIONS"
    )

    print(
        "-" * 70
    )

    for deviation in results[
        "deviations"
    ][:5]:

        print(
            f"Cases={deviation['case_count']:<6}"
            f" "
            f"Deviation="
            f"{deviation['deviation_ratio']:.2f}"
        )

        print(
            f"  Variant: "
            f"{deviation['variant']}"
        )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "BOTTLENECK ANALYSIS AGENT TEST PASSED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()