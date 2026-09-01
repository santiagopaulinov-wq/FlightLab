from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from flightlab.experiment import ExperimentCase, ExperimentRun, execute_experiments

if TYPE_CHECKING:
    from flightlab.persistence import SQLiteExperimentStore


@dataclass(frozen=True, slots=True)
class ExperimentCampaignResult:
    """Immutable result of one completed ordered experiment campaign."""

    campaign_id: str
    created_at: datetime
    runs: tuple[ExperimentRun, ...]

    def __post_init__(self):
        if not isinstance(self.campaign_id, str) or not self.campaign_id.strip():
            raise ValueError("campaign_id must be a non-empty string")
        if not isinstance(self.created_at, datetime) or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be a timezone-aware datetime")
        object.__setattr__(self, "campaign_id", str(self.campaign_id))
        object.__setattr__(self, "created_at", self.created_at.astimezone(UTC))
        if type(self.runs) is not tuple:
            raise TypeError("runs must be a tuple of ExperimentRun objects")
        for index, run in enumerate(self.runs):
            if not isinstance(run, ExperimentRun):
                raise TypeError(f"runs[{index}] must be an ExperimentRun")


def run_experiment_campaign(
    cases: Iterable[ExperimentCase],
    store: "SQLiteExperimentStore | None" = None,
    *,
    campaign_id: str | None = None,
    created_at: datetime | None = None,
) -> ExperimentCampaignResult:
    """Execute explicit cases sequentially and optionally persist them atomically."""
    from flightlab.persistence import SQLiteExperimentStore

    if store is not None and not isinstance(store, SQLiteExperimentStore):
        raise TypeError("store must be a SQLiteExperimentStore or None")
    if campaign_id is None:
        campaign_id = str(uuid4())
    if created_at is None:
        created_at = datetime.now(UTC)

    runs = execute_experiments(cases)
    result = ExperimentCampaignResult(
        campaign_id=campaign_id,
        created_at=created_at,
        runs=runs,
    )
    if store is not None:
        store.save_campaign(result)
    return result
