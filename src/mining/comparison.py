import pandas as pd


class ModelComparator:
    """
    Compares multiple discovered process models
    using their evaluation metrics.
    """

    def __init__(self):
        self.results = []

    def add_result(
        self,
        model_name,
        fitness,
        precision,
        generalization,
        simplicity,
    ):
        """
        Add evaluation results for one model.
        """

        self.results.append(
            {
                "model": model_name,
                "fitness": fitness,
                "precision": precision,
                "generalization": generalization,
                "simplicity": simplicity,
            }
        )

    def get_results(self):
        """
        Return comparison results as a DataFrame.
        """

        return pd.DataFrame(self.results)

    def print_comparison(self):
        """
        Print a formatted comparison table.
        """

        results = self.get_results()

        print("\n")
        print("=" * 80)
        print("APMOA - MODEL COMPARISON")
        print("=" * 80)

        if results.empty:
            print("No models available for comparison.")
            return

        print(
            results.to_string(
                index=False,
                formatters={
                    "fitness": "{:.4f}".format,
                    "precision": "{:.4f}".format,
                    "generalization": "{:.4f}".format,
                    "simplicity": "{:.4f}".format,
                },
            )
        )

    def best_by_precision(self):
        """
        Return the model with the highest precision.
        """

        results = self.get_results()

        if results.empty:
            return None

        return results.loc[
            results["precision"].idxmax()
        ]

    def best_by_fitness(self):
        """
        Return the model with the highest fitness.
        """

        results = self.get_results()

        if results.empty:
            return None

        return results.loc[
            results["fitness"].idxmax()
        ]