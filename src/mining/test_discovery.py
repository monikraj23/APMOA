from loader import EventLogLoader
from discovery import ProcessDiscovery


LOG_PATH = "data/raw/Travel/PermitLog.xes.gz"

OUTPUT_PATH = (
    "results/travel/"
    "inductive_miner_model.pnml"
)


def main():

    print("=" * 60)
    print("APMOA - PROCESS DISCOVERY")
    print("=" * 60)

    # ---------------------------------------------------------
    # LOAD EVENT LOG
    # ---------------------------------------------------------

    loader = EventLogLoader(LOG_PATH)

    loader.load()
    loader.validate()

    standardized_log = (
        loader.get_standardized_log()
    )

    # ---------------------------------------------------------
    # DISCOVERY
    # ---------------------------------------------------------

    discovery = ProcessDiscovery(
        standardized_log
    )

    (
        petri_net,
        initial_marking,
        final_marking,
    ) = discovery.discover_inductive_miner()

    # ---------------------------------------------------------
    # MODEL INFORMATION
    # ---------------------------------------------------------

    print("\n" + "-" * 60)
    print("DISCOVERED MODEL")
    print("-" * 60)

    print(
        f"Places       : {len(petri_net.places)}"
    )

    print(
        f"Transitions  : {len(petri_net.transitions)}"
    )

    print(
        f"Arcs         : {len(petri_net.arcs)}"
    )

    # ---------------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------------

    discovery.save_petri_net(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()