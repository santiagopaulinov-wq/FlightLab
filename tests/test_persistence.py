import sqlite3
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from types import MappingProxyType

import numpy as np
import pytest

from flightlab.experiment import experiment_run
from flightlab.persistence import (
    DuplicateRunIDError,
    ExperimentRunSummary,
    SQLiteExperimentStore,
)
from flightlab.response import response_metrics

_CREATED_AT = datetime(2026, 8, 31, 12, 30, 45, 123456, tzinfo=UTC)
_SCHEMA_COLUMNS = (
    ("run_id", "TEXT"),
    ("created_at", "TEXT"),
    ("method", "TEXT"),
    ("start_time", "REAL"),
    ("end_time", "REAL"),
    ("duration", "REAL"),
    ("sample_count", "INTEGER"),
    ("initial_state_json", "TEXT"),
    ("system_json", "TEXT"),
    ("controller_json", "TEXT"),
    ("reference_json", "TEXT"),
    ("user_metadata_json", "TEXT"),
    ("final_output", "REAL"),
    ("final_reference", "REAL"),
    ("steady_state_error", "REAL"),
    ("peak_output", "REAL"),
    ("maximum_absolute_tracking_error", "REAL"),
    ("rms_tracking_error", "REAL"),
    ("iae", "REAL"),
    ("ise", "REAL"),
    ("overshoot_percent", "REAL"),
    ("settling_time", "REAL"),
    ("settling_tolerance", "REAL"),
)


def _run(
    *,
    run_id="run-001",
    created_at=_CREATED_AT,
    method="exact",
    time=(2.0, 2.5, 4.0),
    y=(0.0, 0.75, 1.0),
    reference_trajectory=(1.0, 1.0, 1.0),
    initial_state=(0.25, -0.5),
    system=None,
    controller=None,
    reference=None,
    user_metadata=None,
):
    if system is None:
        system = {"z": 3, "a": "café"}
    if controller is None:
        controller = {"type": "state_feedback", "gains": (2.0, 3.0)}
    if reference is None:
        reference = {"value": 1.0, "type": "step"}
    if user_metadata is None:
        user_metadata = {"seed": 7, "notes": None}

    metrics = response_metrics(time, y, reference_trajectory)
    return experiment_run(
        time=time,
        initial_state=initial_state,
        metrics=metrics,
        method=method,
        system=system,
        controller=controller,
        reference=reference,
        user_metadata=user_metadata,
        run_id=run_id,
        created_at=created_at,
    )


def test_initialize_creates_the_exact_schema_and_created_at_index(tmp_path):
    path = tmp_path / "experiments.sqlite3"
    store = SQLiteExperimentStore(path)
    try:
        store.initialize()
    finally:
        store.close()

    with sqlite3.connect(path) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
        columns = connection.execute(
            "PRAGMA table_info(experiment_runs)"
        ).fetchall()
        index_definitions = connection.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type = 'index'
              AND tbl_name = 'experiment_runs'
              AND sql IS NOT NULL
            """
        ).fetchall()

    assert tables == [
        ("experiment_campaign_runs",),
        ("experiment_campaigns",),
        ("experiment_runs",),
    ]
    assert tuple((row[1], row[2]) for row in columns) == _SCHEMA_COLUMNS
    assert len(columns) == 23
    assert columns[0][3] == 1
    assert columns[0][5] == 1
    assert tuple(row[1] for row in columns if row[3] == 0) == (
        "overshoot_percent",
        "settling_time",
    )
    assert len(index_definitions) == 1
    normalized_index_sql = " ".join(index_definitions[0][0].split())
    assert "ON experiment_runs (created_at DESC, run_id ASC)" in normalized_index_sql


def test_initialize_is_idempotent_and_preserves_existing_data(tmp_path):
    path = tmp_path / "experiments.sqlite3"
    run = _run()
    store = SQLiteExperimentStore(path)
    try:
        store.initialize()
        store.save(run)
        store.initialize()
        store.initialize()
        assert store.get(run.run_id) == run.reproducibility_record()
    finally:
        store.close()

    with SQLiteExperimentStore(path) as reopened:
        reopened.initialize()
        assert reopened.get(run.run_id) == run.reproducibility_record()


def test_save_and_get_round_trip_the_exact_reproducibility_record(tmp_path):
    run = _run()
    expected = run.reproducibility_record()

    with SQLiteExperimentStore(tmp_path / "experiments.sqlite3") as store:
        store.save(run)
        loaded = store.get(run.run_id)

    assert loaded == expected
    assert loaded["run_id"] == "run-001"
    assert loaded["created_at"] == "2026-08-31T12:30:45.123456+00:00"
    assert loaded["initial_state"] == [0.25, -0.5]
    assert loaded["system"] == {"a": "café", "z": 3}
    assert loaded["controller"] == {
        "gains": [2.0, 3.0],
        "type": "state_feedback",
    }
    assert loaded["reference"] == {"type": "step", "value": 1.0}
    assert loaded["user_metadata"] == {"notes": None, "seed": 7}
    assert loaded["metrics"] == {
        "final_output": 1.0,
        "final_reference": 1.0,
        "steady_state_error": 0.0,
        "peak_output": 1.0,
        "maximum_absolute_tracking_error": 1.0,
        "rms_tracking_error": 0.39528470752104744,
        "iae": 0.5,
        "ise": 0.3125,
        "overshoot_percent": 0.0,
        "settling_time": 4.0,
        "settling_tolerance": 0.02,
    }


def test_round_trip_preserves_optional_none_metrics(tmp_path):
    run = _run(
        run_id="zero-reference",
        time=(0.0, 1.0, 2.0),
        y=(0.0, 0.0, 0.0),
        reference_trajectory=(0.0, 0.0, 0.0),
        initial_state=(0.0,),
        reference={"type": "zero"},
    )

    with SQLiteExperimentStore(tmp_path / "experiments.sqlite3") as store:
        store.save(run)
        loaded = store.get(run.run_id)

    assert loaded == run.reproducibility_record()
    assert loaded["metrics"]["overshoot_percent"] is None
    assert loaded["metrics"]["settling_time"] is None
    assert all(
        value == 0.0
        for name, value in loaded["metrics"].items()
        if name not in {"overshoot_percent", "settling_time", "settling_tolerance"}
    )


def test_structured_fields_use_deterministic_compact_unicode_json(tmp_path):
    path = tmp_path / "experiments.sqlite3"
    run = _run()

    with SQLiteExperimentStore(path) as store:
        store.save(run)

    with sqlite3.connect(path) as connection:
        row = connection.execute(
            """
            SELECT
                initial_state_json,
                system_json,
                controller_json,
                reference_json,
                user_metadata_json
            FROM experiment_runs
            WHERE run_id = ?
            """,
            (run.run_id,),
        ).fetchone()

    assert row == (
        "[0.25,-0.5]",
        '{"a":"café","z":3}',
        '{"gains":[2.0,3.0],"type":"state_feedback"}',
        '{"type":"step","value":1.0}',
        '{"notes":null,"seed":7}',
    )


def test_loaded_records_are_detached_and_saving_does_not_mutate_the_run(tmp_path):
    run = _run()
    original = run.reproducibility_record()

    with SQLiteExperimentStore(tmp_path / "experiments.sqlite3") as store:
        store.save(run)
        loaded = store.get(run.run_id)
        loaded["initial_state"][0] = 100.0
        loaded["system"]["a"] = "changed"
        loaded["controller"]["gains"][0] = 100.0
        loaded["metrics"]["iae"] = 100.0

        assert store.get(run.run_id) == original

    assert run.reproducibility_record() == original


def test_get_returns_none_for_unknown_or_sql_like_ids(tmp_path):
    with SQLiteExperimentStore(tmp_path / "experiments.sqlite3") as store:
        store.save(_run())

        assert store.get("missing") is None
        assert store.get("' OR 1 = 1 --") is None


def test_list_runs_returns_a_frozen_lightweight_summary(tmp_path):
    run = _run()
    with SQLiteExperimentStore(tmp_path / "experiments.sqlite3") as store:
        assert store.list_runs() == ()
        store.save(run)
        summaries = store.list_runs()

    assert summaries == (
        ExperimentRunSummary(
            run_id="run-001",
            created_at="2026-08-31T12:30:45.123456+00:00",
            method="exact",
            duration=2.0,
            sample_count=3,
        ),
    )
    assert not hasattr(summaries[0], "__dict__")
    with pytest.raises(FrozenInstanceError):
        summaries[0].method = "changed"


def test_list_runs_is_newest_first_with_a_run_id_tiebreaker(tmp_path):
    newest = datetime(2026, 9, 1, 8, 0, tzinfo=UTC)
    older = newest - timedelta(days=1)
    runs = (
        _run(run_id="z-tied", created_at=newest),
        _run(run_id="old", created_at=older),
        _run(run_id="a-tied", created_at=newest),
    )

    with SQLiteExperimentStore(tmp_path / "experiments.sqlite3") as store:
        for run in runs:
            store.save(run)

        summaries = store.list_runs()

    assert tuple(summary.run_id for summary in summaries) == (
        "a-tied",
        "z-tied",
        "old",
    )


def test_duplicate_id_is_rejected_without_overwrite_and_store_recovers(tmp_path):
    original = _run()
    replacement = _run(run_id=original.run_id, method="rk4")
    later = _run(run_id="run-002")

    with SQLiteExperimentStore(tmp_path / "experiments.sqlite3") as store:
        store.save(original)
        with pytest.raises(DuplicateRunIDError, match="run-001.*already exists"):
            store.save(replacement)

        assert store.get(original.run_id) == original.reproducibility_record()
        store.save(later)
        assert {summary.run_id for summary in store.list_runs()} == {
            "run-001",
            "run-002",
        }


def test_save_commits_before_return_for_a_second_file_connection(tmp_path):
    path = tmp_path / "experiments.sqlite3"
    run = _run()
    writer = SQLiteExperimentStore(path)
    try:
        writer.initialize()
        writer.save(run)
        with SQLiteExperimentStore(path) as reader:
            assert reader.get(run.run_id) == run.reproducibility_record()
    finally:
        writer.close()


def test_save_many_atomically_persists_runs_in_caller_order(tmp_path):
    path = tmp_path / "experiments.sqlite3"
    runs = (
        _run(run_id="run-003"),
        _run(run_id="run-001"),
        _run(run_id="run-002"),
    )
    originals = tuple(run.reproducibility_record() for run in runs)

    with SQLiteExperimentStore(path) as store:
        store.save_many(run for run in runs)
        assert tuple(store.get(run.run_id) for run in runs) == tuple(
            run.reproducibility_record() for run in runs
        )

    with sqlite3.connect(path) as connection:
        inserted_ids = tuple(
            row[0]
            for row in connection.execute(
                "SELECT run_id FROM experiment_runs ORDER BY rowid"
            )
        )

    assert inserted_ids == tuple(run.run_id for run in runs)
    assert tuple(run.reproducibility_record() for run in runs) == originals


def test_save_many_accepts_empty_and_single_run_collections(tmp_path):
    run = _run()
    with SQLiteExperimentStore(tmp_path / "experiments.sqlite3") as store:
        store.save_many([])
        assert store.list_runs() == ()

        store.save_many((run,))
        assert store.get(run.run_id) == run.reproducibility_record()


def test_save_many_rejects_duplicate_ids_within_batch_and_rolls_back(tmp_path):
    first = _run(run_id="duplicate")
    duplicate = _run(run_id="duplicate", method="rk4")

    with SQLiteExperimentStore(tmp_path / "experiments.sqlite3") as store:
        with pytest.raises(DuplicateRunIDError, match="duplicate.*already exists"):
            store.save_many((first, duplicate))

        assert store.list_runs() == ()


def test_save_many_rejects_existing_id_and_rolls_back_complete_batch(tmp_path):
    existing = _run(run_id="existing")
    earlier = _run(run_id="earlier")
    duplicate = _run(run_id="existing", method="rk4")

    with SQLiteExperimentStore(tmp_path / "experiments.sqlite3") as store:
        store.save(existing)
        with pytest.raises(DuplicateRunIDError, match="existing.*already exists"):
            store.save_many((earlier, duplicate))

        assert store.get("earlier") is None
        assert store.get("existing") == existing.reproducibility_record()


def test_save_many_validates_complete_snapshot_before_insertion(tmp_path):
    valid = _run(run_id="valid")
    malformed = replace(valid, run_id="   ")

    with SQLiteExperimentStore(tmp_path / "experiments.sqlite3") as store:
        with pytest.raises(ValueError, match="run_id"):
            store.save_many(iter((valid, malformed)))

        assert store.list_runs() == ()


def test_save_many_database_failure_rolls_back_and_store_remains_usable(tmp_path):
    first = _run(run_id="first")
    rejected = _run(run_id="rejected")
    later = _run(run_id="later")

    with SQLiteExperimentStore(tmp_path / "experiments.sqlite3") as store:
        store._connection.execute(
            """
            CREATE TRIGGER reject_selected_run
            BEFORE INSERT ON experiment_runs
            WHEN NEW.run_id = 'rejected'
            BEGIN
                SELECT RAISE(ABORT, 'selected database failure');
            END
            """
        )
        with pytest.raises(ValueError, match="violates the experiment_runs schema"):
            store.save_many((first, rejected))

        assert store.list_runs() == ()
        store._connection.execute("DROP TRIGGER reject_selected_run")
        store.save_many((later,))
        assert store.get(later.run_id) == later.reproducibility_record()


@pytest.mark.parametrize("runs", [None, 1], ids=("none", "integer"))
def test_save_many_rejects_non_iterable_inputs(tmp_path, runs):
    with SQLiteExperimentStore(tmp_path / "experiments.sqlite3") as store:
        with pytest.raises(TypeError, match="runs must be an iterable"):
            store.save_many(runs)
        assert store.list_runs() == ()


def test_file_database_reopens_when_its_path_contains_spaces(tmp_path):
    path = tmp_path / "flight lab experiment records.sqlite3"
    run = _run()

    with SQLiteExperimentStore(path) as store:
        store.save(run)

    assert path.is_file()
    with SQLiteExperimentStore(path) as reopened:
        assert reopened.get(run.run_id) == run.reproducibility_record()


def test_context_manager_initializes_automatically_and_close_is_terminal(tmp_path):
    path = tmp_path / "experiments.sqlite3"
    run = _run()

    with SQLiteExperimentStore(path) as store:
        store.save(run)
        assert store.get(run.run_id) == run.reproducibility_record()

    for operation in (
        store.initialize,
        store.list_runs,
        lambda: store.get(run.run_id),
        lambda: store.save(run),
        lambda: store.save_many((run,)),
    ):
        with pytest.raises(RuntimeError, match="closed"):
            operation()
    store.close()


def test_memory_databases_persist_for_one_store_lifetime_and_are_isolated():
    run = _run()
    with SQLiteExperimentStore(":memory:") as first:
        first.save(run)
        first.initialize()
        assert first.get(run.run_id) == run.reproducibility_record()

        with SQLiteExperimentStore(":memory:") as second:
            assert second.get(run.run_id) is None
            assert second.list_runs() == ()

        assert first.get(run.run_id) == run.reproducibility_record()


def test_sql_injection_like_ids_and_metadata_are_stored_only_as_data(tmp_path):
    hostile_id = "run'); DROP TABLE experiment_runs;--"
    hostile_value = "Robert'); DELETE FROM experiment_runs;-- ☃"
    hostile = _run(
        run_id=hostile_id,
        system={"payload": hostile_value},
        user_metadata={"query": "' OR 1 = 1 --"},
    )
    normal = _run(run_id="normal")

    with SQLiteExperimentStore(tmp_path / "experiments.sqlite3") as store:
        store.save(hostile)
        loaded = store.get(hostile_id)
        assert loaded == hostile.reproducibility_record()
        assert loaded["system"]["payload"] == hostile_value

        store.save(normal)
        assert {summary.run_id for summary in store.list_runs()} == {
            hostile_id,
            "normal",
        }


def test_operations_before_initialize_fail_clearly(tmp_path):
    run = _run()
    store = SQLiteExperimentStore(tmp_path / "experiments.sqlite3")
    try:
        for operation in (
            store.list_runs,
            lambda: store.get(run.run_id),
            lambda: store.save(run),
            lambda: store.save_many((run,)),
        ):
            with pytest.raises(RuntimeError, match="must be initialized"):
                operation()
    finally:
        store.close()


@pytest.mark.parametrize(
    "invalid_run",
    [None, {}, object()],
    ids=("none", "dictionary", "opaque-object"),
)
def test_save_rejects_non_experiment_run_inputs(tmp_path, invalid_run):
    with SQLiteExperimentStore(tmp_path / "experiments.sqlite3") as store:
        with pytest.raises(TypeError, match="run must be an ExperimentRun"):
            store.save(invalid_run)
        assert store.list_runs() == ()


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("blank-run-id", "run_id"),
        ("naive-created-at", "created_at"),
        ("nonfinite-metric", r"metrics\.iae"),
        ("nonfinite-initial-state", "initial_state"),
        ("nested-metadata", "system"),
    ],
)
def test_save_rejects_malformed_forged_runs_without_partial_writes(
    tmp_path,
    case,
    message,
):
    valid = _run()
    if case == "blank-run-id":
        malformed = replace(valid, run_id="   ")
    elif case == "naive-created-at":
        malformed = replace(valid, created_at=valid.created_at.replace(tzinfo=None))
    elif case == "nonfinite-metric":
        malformed = replace(valid, metrics=valid.metrics._replace(iae=np.nan))
    elif case == "nonfinite-initial-state":
        malformed = replace(valid, initial_state=np.array([np.inf]))
    else:
        malformed = replace(
            valid,
            system=MappingProxyType({"nested": {"unsupported": True}}),
        )

    with SQLiteExperimentStore(tmp_path / "experiments.sqlite3") as store:
        with pytest.raises(ValueError, match=message):
            store.save(malformed)
        assert store.list_runs() == ()

        store.save(valid)
        assert store.get(valid.run_id) == valid.reproducibility_record()
