from loader import EventLogLoader
from discovery import ProcessDiscovery
from conformance_analysis import ConformanceAnalyzer


LOG_PATH = "data/raw/Travel/PermitLog.xes.gz"


def main():

    print("=" * 60)
    print("APMOA - CONFORMANCE ANALYSIS TEST")
    print("=" * 60)

    # =========================================================
    # LOAD EVENT LOG
    # =========================================================

    loader = EventLogLoader(LOG_PATH)

    loader.load()
    loader.validate()

    log = loader.get_standardized_log()

    # =========================================================
    # PROCESS DISCOVERY
    # =========================================================

    discovery = ProcessDiscovery(log)

    (
        petri_net,
        initial_marking,
        final_marking,
    ) = discovery.discover_inductive_miner()

    # =========================================================
    # CONFORMANCE ANALYSIS
    # =========================================================

    analyzer = ConformanceAnalyzer(
        log=log,
        petri_net=petri_net,
        initial_marking=initial_marking,
        final_marking=final_marking,
    )

    analyzer.generate_report()


if __name__ == "__main__":
    main()