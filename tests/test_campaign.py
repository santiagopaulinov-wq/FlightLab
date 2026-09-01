from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from flightlab.campaign import ExperimentCampaignResult, run_experiment_campaign
from flightlab.experiment import ExperimentCase, SISOSimulationResult
from flightlab.persistence import DuplicateRunIDError, SQLiteExperimentStore

_CREATED_AT = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def _case(run_id, calls, *, failure=None):
    def simulation():
        calls.append(run_id)
        if failure is not None:
            raise failure
        return SISOSimulationResult(
            time=[0.0, 1.0],
            output=[0.0, 1.0],
            reference=[1.0, 1.0],
        )

    return ExperimentCase(
        simulation=simulation,
        initial_state=[0.0],
        method="exact",
        system={"name": "test"},
        controller={"case": run_id},
        reference={"type": "step"},
        run_id=run_id,
        created_at=_CREATED_AT,
    )


def test_in_memory_campaign_preserves_order_and_executes_exactly_once():
    calls = []
    cases = (_case("third", calls), _case("first", calls), _case("second", calls))

    result = run_experiment_campaign(case for case in cases)

    assert isinstance(result, ExperimentCampaignResult)
    assert tuple(run.run_id for run in result.runs) == ("third", "first", "second")
    assert calls == ["third", "first", "second"]


def test_campaign_result_and_its_ordered_run_collection_are_immutable():
    result = run_experiment_campaign((_case("only", []),))

    assert isinstance(result.runs, tuple)
    with pytest.raises(FrozenInstanceError):
        result.runs = ()
    with pytest.raises(TypeError):
        result.runs[0] = result.runs[0]

    with pytest.raises(TypeError, match="runs must be a tuple"):
        ExperimentCampaignResult(runs=[])


def test_empty_campaign_returns_an_empty_result_and_persists_nothing(tmp_path):
    with SQLiteExperimentStore(tmp_path / "experiments.sqlite3") as store:
        result = run_experiment_campaign([], store=store)

        assert result == ExperimentCampaignResult(runs=())
        assert store.list_runs() == ()


def test_successful_campaign_persists_completed_runs_atomically(tmp_path):
    calls = []
    cases = (_case("run-b", calls), _case("run-a", calls))

    with SQLiteExperimentStore(tmp_path / "experiments.sqlite3") as store:
        result = run_experiment_campaign(cases, store=store)

        assert tuple(store.get(run.run_id) for run in result.runs) == tuple(
            run.reproducibility_record() for run in result.runs
        )
    assert calls == ["run-b", "run-a"]


def test_execution_failure_prevents_all_campaign_persistence(tmp_path):
    calls = []
    failure = RuntimeError("simulation failed")
    cases = (
        _case("completed", calls),
        _case("failed", calls, failure=failure),
        _case("not-run", calls),
    )

    with SQLiteExperimentStore(tmp_path / "experiments.sqlite3") as store:
        with pytest.raises(RuntimeError, match="simulation failed") as raised:
            run_experiment_campaign(cases, store=store)

        assert raised.value is failure
        assert calls == ["completed", "failed"]
        assert store.list_runs() == ()


def test_atomic_persistence_failure_propagates_and_rolls_back_campaign(tmp_path):
    calls = []
    cases = (_case("duplicate", calls), _case("duplicate", calls))

    with SQLiteExperimentStore(tmp_path / "experiments.sqlite3") as store:
        with pytest.raises(DuplicateRunIDError, match="duplicate.*already exists"):
            run_experiment_campaign(cases, store=store)

        assert calls == ["duplicate", "duplicate"]
        assert store.list_runs() == ()


def test_invalid_store_is_rejected_before_any_experiment_executes():
    calls = []

    with pytest.raises(TypeError, match="SQLiteExperimentStore or None"):
        run_experiment_campaign((_case("not-run", calls),), store=object())

    assert calls == []
