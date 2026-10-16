from loader import EventLogLoader
from discovery import ProcessDiscovery
from evaluation import ProcessEvaluator


LOG_PATH = "data/raw/Travel/PermitLog.xes.gz"


def main():

    print("=" * 60)
    print("APMOA - PROCESS MODEL EVALUATION")
    print("=" * 60)

    # Load event log
    loader = EventLogLoader(LOG_PATH)

    loader.load()
    loader.validate()

    standardized_log = loader.get_standardized_log()

    # Process discovery
    print("\n" + "-" * 60)
    print("PROCESS DISCOVERY")
    print("-" * 60)

    discovery = ProcessDiscovery(
        standardized_log
    )

    (
        petri_net,
        initial_marking,
        final_marking,
    ) = discovery.discover_inductive_miner()

    # Evaluation
    evaluator = ProcessEvaluator(
        standardized_log,
        petri_net,
        initial_marking,
        final_marking,
    )

    results = evaluator.evaluate()

    # Results
    print("\n" + "-" * 60)
    print("PROCESS MODEL EVALUATION RESULTS")
    print("-" * 60)

    for metric, value in results.items():

        if isinstance(value, dict):

            print(f"\n{metric.upper()}")

            for key, result in value.items():
                print(f"  {key}: {result}")

        else:

            print(
                f"{metric:20}: {value:.4f}"
            )

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()