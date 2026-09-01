from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from flightlab.campaign import ExperimentCampaignResult, run_experiment_campaign
from flightlab.experiment import ExperimentCase, SISOSimulationResult
from flightlab.persistence import (
    DuplicateCampaignIDError,
    DuplicateRunIDError,
    ExperimentCampaignManifest,
    SQLiteExperimentStore,
)

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

    result = run_experiment_campaign(
        (case for case in cases),
        campaign_id="ordered-campaign",
        created_at=_CREATED_AT,
    )

    assert isinstance(result, ExperimentCampaignResult)
    assert result.campaign_id == "ordered-campaign"
    assert result.created_at == _CREATED_AT
    assert tuple(run.run_id for run in result.runs) == ("third", "first", "second")
    assert calls == ["third", "first", "second"]


def test_campaign_result_and_its_ordered_run_collection_are_immutable():
    result = run_experiment_campaign(
        (_case("only", []),),
        campaign_id="immutable",
        created_at=_CREATED_AT,
    )

    assert isinstance(result.runs, tuple)
    with pytest.raises(FrozenInstanceError):
        result.runs = ()
    with pytest.raises(TypeError):
        result.runs[0] = result.runs[0]

    with pytest.raises(TypeError, match="runs must be a tuple"):
        ExperimentCampaignResult(
            campaign_id="invalid",
            created_at=_CREATED_AT,
            runs=[],
        )


def test_empty_campaign_returns_an_empty_result_and_persists_nothing(tmp_path):
    with SQLiteExperimentStore(tmp_path / "experiments.sqlite3") as store:
        result = run_experiment_campaign(
            [], store=store, campaign_id="empty", created_at=_CREATED_AT
        )

        assert result == ExperimentCampaignResult(
            campaign_id="empty", created_at=_CREATED_AT, runs=()
        )
        assert store.list_runs() == ()
        assert store.get_campaign("empty") == ExperimentCampaignManifest(
            campaign_id="empty",
            created_at="2026-09-01T12:00:00+00:00",
            run_ids=(),
        )


def test_successful_campaign_persists_completed_runs_atomically(tmp_path):
    calls = []
    cases = (_case("run-b", calls), _case("run-a", calls))

    with SQLiteExperimentStore(tmp_path / "experiments.sqlite3") as store:
        result = run_experiment_campaign(
            cases,
            store=store,
            campaign_id="persisted",
            created_at=_CREATED_AT,
        )

        assert tuple(store.get(run.run_id) for run in result.runs) == tuple(
            run.reproducibility_record() for run in result.runs
        )
        assert store.get_campaign("persisted") == ExperimentCampaignManifest(
            campaign_id="persisted",
            created_at="2026-09-01T12:00:00+00:00",
            run_ids=("run-b", "run-a"),
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
            run_experiment_campaign(
                cases,
                store=store,
                campaign_id="failed-execution",
                created_at=_CREATED_AT,
            )

        assert raised.value is failure
        assert calls == ["completed", "failed"]
        assert store.list_runs() == ()
        assert store.get_campaign("failed-execution") is None


def test_atomic_persistence_failure_propagates_and_rolls_back_campaign(tmp_path):
    calls = []
    cases = (_case("duplicate", calls), _case("duplicate", calls))

    with SQLiteExperimentStore(tmp_path / "experiments.sqlite3") as store:
        with pytest.raises(DuplicateRunIDError, match="duplicate.*already exists"):
            run_experiment_campaign(
                cases,
                store=store,
                campaign_id="failed-persistence",
                created_at=_CREATED_AT,
            )

        assert calls == ["duplicate", "duplicate"]
        assert store.list_runs() == ()
        assert store.get_campaign("failed-persistence") is None


def test_invalid_store_is_rejected_before_any_experiment_executes():
    calls = []

    with pytest.raises(TypeError, match="SQLiteExperimentStore or None"):
        run_experiment_campaign((_case("not-run", calls),), store=object())

    assert calls == []


def test_campaign_manifest_retrieval_is_detached_and_deterministic(tmp_path):
    calls = []
    path = tmp_path / "experiments.sqlite3"
    with SQLiteExperimentStore(path) as store:
        run_experiment_campaign(
            (_case("z", calls), _case("a", calls)),
            store=store,
            campaign_id="detached",
            created_at=_CREATED_AT,
        )
        first = store.get_campaign("detached")
        second = store.get_campaign("detached")

    assert first == second
    assert first is not second
    assert first.run_ids == ("z", "a")
    with pytest.raises(FrozenInstanceError):
        first.campaign_id = "changed"

    with SQLiteExperimentStore(path) as reopened:
        assert reopened.get_campaign("detached") == first


def test_duplicate_campaign_id_rolls_back_new_runs_and_store_recovers(tmp_path):
    calls = []
    with SQLiteExperimentStore(tmp_path / "experiments.sqlite3") as store:
        run_experiment_campaign(
            (_case("original", calls),),
            store=store,
            campaign_id="same-campaign",
            created_at=_CREATED_AT,
        )
        with pytest.raises(
            DuplicateCampaignIDError, match="same-campaign.*already exists"
        ):
            run_experiment_campaign(
                (_case("rolled-back", calls),),
                store=store,
                campaign_id="same-campaign",
                created_at=_CREATED_AT,
            )

        assert store.get("rolled-back") is None
        assert store.get_campaign("same-campaign").run_ids == ("original",)
        recovered = run_experiment_campaign(
            (_case("later", calls),),
            store=store,
            campaign_id="later-campaign",
            created_at=_CREATED_AT,
        )
        assert store.get_campaign("later-campaign").run_ids == (
            recovered.runs[0].run_id,
        )


def test_existing_run_conflict_preserves_existing_records_and_manifest(tmp_path):
    calls = []
    with SQLiteExperimentStore(tmp_path / "experiments.sqlite3") as store:
        original = run_experiment_campaign(
            (_case("existing", calls),),
            store=store,
            campaign_id="original-campaign",
            created_at=_CREATED_AT,
        )
        with pytest.raises(DuplicateRunIDError, match="existing.*already exists"):
            run_experiment_campaign(
                (_case("new", calls), _case("existing", calls)),
                store=store,
                campaign_id="conflicting-campaign",
                created_at=_CREATED_AT,
            )

        assert store.get("new") is None
        assert store.get("existing") == original.runs[0].reproducibility_record()
        assert store.get_campaign("original-campaign").run_ids == ("existing",)
        assert store.get_campaign("conflicting-campaign") is None


def test_membership_failure_rolls_back_manifest_and_new_runs(tmp_path):
    calls = []
    with SQLiteExperimentStore(tmp_path / "experiments.sqlite3") as store:
        store._connection.execute(
            """
            CREATE TRIGGER reject_campaign_membership
            BEFORE INSERT ON experiment_campaign_runs
            BEGIN
                SELECT RAISE(ABORT, 'selected membership failure');
            END
            """
        )
        with pytest.raises(ValueError, match="experiment campaign schema"):
            run_experiment_campaign(
                (_case("first", calls), _case("second", calls)),
                store=store,
                campaign_id="membership-failure",
                created_at=_CREATED_AT,
            )

        assert store.list_runs() == ()
        assert store.get_campaign("membership-failure") is None
        store._connection.execute("DROP TRIGGER reject_campaign_membership")
        recovered = run_experiment_campaign(
            (_case("later", calls),),
            store=store,
            campaign_id="recovered",
            created_at=_CREATED_AT,
        )
        assert store.get_campaign("recovered").run_ids == (
            recovered.runs[0].run_id,
        )
