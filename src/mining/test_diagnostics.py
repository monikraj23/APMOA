from loader import EventLogLoader
from log_diagnostics import LogDiagnostics


LOG_PATH = "data/raw/Travel/PermitLog.xes.gz"


def main():

    print("=" * 60)
    print("APMOA - LOG DIAGNOSTICS TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # LOAD EVENT LOG
    # ---------------------------------------------------------

    loader = EventLogLoader(LOG_PATH)

    loader.load()
    loader.validate()

    log = loader.get_standardized_log()

    # ---------------------------------------------------------
    # DIAGNOSTICS
    # ---------------------------------------------------------

    diagnostics = LogDiagnostics(log)

    diagnostics.generate_report()


if __name__ == "__main__":
    main()