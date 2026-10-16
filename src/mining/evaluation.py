import pandas as pd
import pm4py


class ProcessEvaluator:
    """
    Evaluates a discovered Petri net against an event log.

    Metrics:
        - Fitness
        - Precision
        - Generalization
        - Simplicity
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

        self._validate()
        self._prepare()

    def _validate(self):
        """Validate the event log."""

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
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

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

    def calculate_fitness(self):
        """Calculate model fitness."""

        print("Calculating fitness...")

        fitness = pm4py.fitness_token_based_replay(
            self.log,
            self.petri_net,
            self.initial_marking,
            self.final_marking,
            case_id_key=self.case_column,
            activity_key=self.activity_column,
            timestamp_key=self.timestamp_column,
        )

        return fitness

    def calculate_precision(self):
        """Calculate model precision."""

        print("Calculating precision...")

        precision = pm4py.precision_token_based_replay(
            self.log,
            self.petri_net,
            self.initial_marking,
            self.final_marking,
            case_id_key=self.case_column,
            activity_key=self.activity_column,
            timestamp_key=self.timestamp_column,
        )

        return precision

    def calculate_generalization(self):
        """Calculate model generalization."""

        print("Calculating generalization...")

        generalization = pm4py.generalization_tbr(
            self.log,
            self.petri_net,
            self.initial_marking,
            self.final_marking,
            case_id_key=self.case_column,
            activity_key=self.activity_column,
            timestamp_key=self.timestamp_column,
        )

        return generalization

    def calculate_simplicity(self):
        """Calculate model simplicity."""

        print("Calculating simplicity...")

        simplicity = pm4py.simplicity_petri_net(
            self.petri_net,
            self.initial_marking,
            self.final_marking,
        )

        return simplicity

    def evaluate(self):
        """Calculate all evaluation metrics."""

        print("\nStarting process model evaluation...")

        fitness = self.calculate_fitness()
        precision = self.calculate_precision()
        generalization = self.calculate_generalization()
        simplicity = self.calculate_simplicity()

        return {
            "fitness": fitness,
            "precision": precision,
            "generalization": generalization,
            "simplicity": simplicity,
        }