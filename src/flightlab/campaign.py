from collections.abc import Iterable
from dataclasses import dataclass

from flightlab.experiment import ExperimentCase, ExperimentRun, execute_experiments
from flightlab.persistence import SQLiteExperimentStore


@dataclass(frozen=True, slots=True)
class ExperimentCampaignResult:
    """Immutable result of one completed ordered experiment campaign."""

    runs: tuple[ExperimentRun, ...]

    def __post_init__(self):
        if type(self.runs) is not tuple:
            raise TypeError("runs must be a tuple of ExperimentRun objects")
        for index, run in enumerate(self.runs):
            if not isinstance(run, ExperimentRun):
                raise TypeError(f"runs[{index}] must be an ExperimentRun")


def run_experiment_campaign(
    cases: Iterable[ExperimentCase],
    store: SQLiteExperimentStore | None = None,
) -> ExperimentCampaignResult:
    """Execute explicit cases sequentially and optionally persist them atomically."""
    if store is not None and not isinstance(store, SQLiteExperimentStore):
        raise TypeError("store must be a SQLiteExperimentStore or None")

    runs = execute_experiments(cases)
    if store is not None:
        store.save_many(runs)
    return ExperimentCampaignResult(runs=runs)
