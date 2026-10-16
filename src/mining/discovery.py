import os
import pandas as pd
import pm4py


class ProcessDiscovery:
    """
    Process discovery component for APMOA.

    Supports multiple process discovery strategies:

        1. Inductive Miner
        2. Alpha Miner
        3. Directly-Follows Graph (DFG)
    """

    def __init__(
        self,
        log: pd.DataFrame,
        case_column: str = "case_id",
        activity_column: str = "activity",
        timestamp_column: str = "timestamp",
    ):
        """
        Initialize the ProcessDiscovery component.

        Parameters
        ----------
        log : pd.DataFrame
            Standardized event log.

        case_column : str
            Column containing the case identifier.

        activity_column : str
            Column containing activity names.

        timestamp_column : str
            Column containing timestamps.
        """

        self.log = log.copy()

        self.case_column = case_column
        self.activity_column = activity_column
        self.timestamp_column = timestamp_column

        self._prepare()

    # =========================================================
    # PREPARE EVENT LOG
    # =========================================================

    def _prepare(self):
        """
        Prepare the event log for process discovery.
        """

        self.log[self.timestamp_column] = pd.to_datetime(
            self.log[self.timestamp_column]
        )

        self.log = self.log.sort_values(
            by=[
                self.case_column,
                self.timestamp_column,
            ]
        ).reset_index(drop=True)

    # =========================================================
    # INDUCTIVE MINER
    # =========================================================

    def discover_inductive_miner(self):
        """
        Discover a Petri net using Inductive Miner.

        Returns
        -------
        petri_net
        initial_marking
        final_marking
        """

        print("\nRunning Inductive Miner...")

        (
            petri_net,
            initial_marking,
            final_marking,
        ) = pm4py.discover_petri_net_inductive(
            self.log,
            case_id_key=self.case_column,
            activity_key=self.activity_column,
            timestamp_key=self.timestamp_column,
        )

        print("Inductive Miner completed.")

        return (
            petri_net,
            initial_marking,
            final_marking,
        )

    # =========================================================
    # ALPHA MINER
    # =========================================================

    def discover_alpha_miner(self):
        """
        Discover a Petri net using Alpha Miner.

        Returns
        -------
        petri_net
        initial_marking
        final_marking
        """

        print("\nRunning Alpha Miner...")

        (
            petri_net,
            initial_marking,
            final_marking,
        ) = pm4py.discover_petri_net_alpha_plus(
            self.log,
            case_id_key=self.case_column,
            activity_key=self.activity_column,
            timestamp_key=self.timestamp_column,
        )

        print("Alpha Miner completed.")

        return (
            petri_net,
            initial_marking,
            final_marking,
        )

    # =========================================================
    # DIRECTLY-FOLLOWS GRAPH
    # =========================================================

    def discover_dfg(self):
        """
        Discover a Directly-Follows Graph (DFG).

        Note:
        DFG is a graph representation and not a Petri net.
        Therefore, it is not directly evaluated using the
        Petri-net evaluation functions.

        Returns
        -------
        dfg
        start_activities
        end_activities
        """

        print("\nRunning Directly-Follows Graph discovery...")

        (
            dfg,
            start_activities,
            end_activities,
        ) = pm4py.discover_dfg(
            self.log,
            case_id_key=self.case_column,
            activity_key=self.activity_column,
            timestamp_key=self.timestamp_column,
        )

        print("DFG discovery completed.")

        return (
            dfg,
            start_activities,
            end_activities,
        )

    # =========================================================
    # SAVE PETRI NET
    # =========================================================

    def save_petri_net(
        self,
        petri_net,
        initial_marking,
        final_marking,
        output_path,
    ):
        """
        Save a discovered Petri net as a PNML file.

        Parameters
        ----------
        petri_net :
            Discovered Petri net.

        initial_marking :
            Initial marking of the Petri net.

        final_marking :
            Final marking of the Petri net.

        output_path : str
            Destination PNML file.
        """

        directory = os.path.dirname(output_path)

        if directory:
            os.makedirs(
                directory,
                exist_ok=True
            )

        pm4py.write_pnml(
            petri_net,
            initial_marking,
            final_marking,
            output_path,
        )

        print(
            f"Petri net saved to: {output_path}"
        )