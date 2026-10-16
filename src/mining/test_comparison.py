from loader import EventLogLoader
from discovery import ProcessDiscovery
from evaluation import ProcessEvaluator
from comparison import ModelComparator


LOG_PATH = "data/raw/Travel/PermitLog.xes.gz"


def evaluate_model(
    model_name,
    log,
    petri_net,
    initial_marking,
    final_marking,
    comparator,
):
    """
    Evaluate one process model and add the
    results to the comparator.
    """

    print("\n" + "-" * 60)
    print(f"EVALUATING: {model_name}")
    print("-" * 60)

    evaluator = ProcessEvaluator(
        log,
        petri_net,
        initial_marking,
        final_marking,
    )

    results = evaluator.evaluate()

    comparator.add_result(
        model_name=model_name,
        fitness=results["fitness"]["log_fitness"],
        precision=results["precision"],
        generalization=results["generalization"],
        simplicity=results["simplicity"],
    )


def main():

    print("=" * 60)
    print("APMOA - DISCOVERY STRATEGY COMPARISON")
    print("=" * 60)

    # ---------------------------------------------------------
    # LOAD EVENT LOG
    # ---------------------------------------------------------

    loader = EventLogLoader(LOG_PATH)

    loader.load()
    loader.validate()

    log = loader.get_standardized_log()

    # ---------------------------------------------------------
    # DISCOVERY
    # ---------------------------------------------------------

    discovery = ProcessDiscovery(log)

    comparator = ModelComparator()

    # ---------------------------------------------------------
    # MODEL 1 - INDUCTIVE MINER
    # ---------------------------------------------------------

    (
        petri_net,
        initial_marking,
        final_marking,
    ) = discovery.discover_inductive_miner()

    evaluate_model(
        "Inductive Miner",
        log,
        petri_net,
        initial_marking,
        final_marking,
        comparator,
    )

    # ---------------------------------------------------------
    # MODEL 2 - ALPHA MINER
    # ---------------------------------------------------------

    (
        petri_net,
        initial_marking,
        final_marking,
    ) = discovery.discover_alpha_miner()

    evaluate_model(
        "Alpha Miner",
        log,
        petri_net,
        initial_marking,
        final_marking,
        comparator,
    )

    # ---------------------------------------------------------
    # COMPARISON
    # ---------------------------------------------------------

    comparator.print_comparison()

    # ---------------------------------------------------------
    # BEST MODELS
    # ---------------------------------------------------------

    best_precision = comparator.best_by_precision()
    best_fitness = comparator.best_by_fitness()

    print("\n" + "-" * 60)
    print("BEST MODEL BY PRECISION")
    print("-" * 60)

    if best_precision is not None:
        print(
            f"Model     : {best_precision['model']}"
        )
        print(
            f"Precision : {best_precision['precision']:.4f}"
        )

    print("\n" + "-" * 60)
    print("BEST MODEL BY FITNESS")
    print("-" * 60)

    if best_fitness is not None:
        print(
            f"Model  : {best_fitness['model']}"
        )
        print(
            f"Fitness : {best_fitness['fitness']:.4f}"
        )

    print("\nComparison completed successfully.")


if __name__ == "__main__":
    main()