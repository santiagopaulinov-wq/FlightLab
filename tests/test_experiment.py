import json
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, timezone
from types import MappingProxyType
from uuid import UUID

import numpy as np
import pytest

from flightlab.experiment import ExperimentRun, experiment_run
from flightlab.response import response_metrics


@pytest.fixture
def metrics():
    return response_metrics(
        [2.0, 2.5, 4.0],
        [0.0, 0.75, 1.0],
        [1.0, 1.0, 1.0],
    )


def _run(valid_metrics, **overrides):
    arguments = {
        "time": [2.0, 2.5, 4.0],
        "initial_state": [0.25, -0.5],
        "metrics": valid_metrics,
        "method": "exact",
        "system": {"name": "demo", "order": 2},
        "controller": {"type": "integral_state_feedback"},
        "reference": {"type": "step", "value": 1.0},
    }
    arguments.update(overrides)
    return experiment_run(**arguments)


def test_experiment_run_basic_construction_and_metric_composition(metrics):
    run = experiment_run(
        [2.0, 2.5, 4.0],
        [0.25, -0.5],
        metrics,
        "exact",
        {"name": "demo", "order": 2},
        {"type": "integral_state_feedback"},
        {"type": "step", "value": 1.0},
    )

    assert isinstance(run, ExperimentRun)
    assert run.method == "exact"
    assert run.metrics is metrics
    assert dict(run.system) == {"name": "demo", "order": 2}
    assert dict(run.controller) == {"type": "integral_state_feedback"}
    assert dict(run.reference) == {"type": "step", "value": 1.0}
    assert dict(run.user_metadata) == {}
    assert not hasattr(run, "__dict__")

    with pytest.raises(FrozenInstanceError):
        run.method = "changed"


def test_experiment_run_generates_a_uuid4_string(metrics):
    run = _run(metrics)
    parsed = UUID(run.run_id)

    assert isinstance(run.run_id, str)
    assert parsed.version == 4
    assert str(parsed) == run.run_id


def test_experiment_run_generates_distinct_run_ids(metrics):
    assert _run(metrics).run_id != _run(metrics).run_id


def test_experiment_run_preserves_an_explicit_run_id(metrics):
    assert _run(metrics, run_id="run-001").run_id == "run-001"


@pytest.mark.parametrize("run_id", ["", "   ", 7])
def test_experiment_run_rejects_invalid_explicit_run_ids(metrics, run_id):
    with pytest.raises(ValueError, match="run_id.*non-empty string"):
        _run(metrics, run_id=run_id)


def test_experiment_run_generates_a_timezone_aware_utc_timestamp(metrics):
    before = datetime.now(UTC)
    run = _run(metrics)
    after = datetime.now(UTC)

    assert before <= run.created_at <= after
    assert run.created_at.tzinfo is not None
    assert run.created_at.utcoffset() == timedelta(0)


def test_experiment_run_normalizes_an_explicit_aware_timestamp_to_utc(metrics):
    source = datetime(
        2026,
        8,
        31,
        6,
        30,
        tzinfo=timezone(timedelta(hours=-6)),
    )

    run = _run(metrics, created_at=source)

    assert run.created_at == datetime(2026, 8, 31, 12, 30, tzinfo=UTC)
    assert run.created_at.tzinfo is UTC


@pytest.mark.parametrize(
    "created_at",
    [
        datetime(2026, 8, 31, 12, 30, tzinfo=UTC).replace(tzinfo=None),
        "2026-08-31T12:30:00+00:00",
    ],
)
def test_experiment_run_rejects_invalid_explicit_timestamps(metrics, created_at):
    with pytest.raises(ValueError, match="created_at.*timezone-aware datetime"):
        _run(metrics, created_at=created_at)


def test_experiment_run_derives_timing_from_the_supplied_time(metrics):
    run = _run(metrics)

    assert run.start_time == pytest.approx(2.0)
    assert run.end_time == pytest.approx(4.0)
    assert run.duration == pytest.approx(2.0)
    assert run.sample_count == 3


def test_experiment_run_copies_and_protects_the_initial_state(metrics):
    initial_state = np.array([0.25, -0.5])
    run = _run(metrics, initial_state=initial_state)
    initial_state[:] = 100.0

    np.testing.assert_array_equal(run.initial_state, [0.25, -0.5])
    assert run.initial_state.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        run.initial_state[0] = 0.0


def test_experiment_run_copies_and_protects_metadata(metrics):
    system = {"states": ("position", "velocity"), "name": "demo"}
    controller = {"type": "none"}
    reference = {"type": "step", "value": 1.0}
    user_metadata = {"seed": 7}
    run = _run(
        metrics,
        system=system,
        controller=controller,
        reference=reference,
        user_metadata=user_metadata,
    )
    system["name"] = "changed"
    controller["type"] = "changed"
    reference["value"] = 2.0
    user_metadata["seed"] = 99

    assert run.system["name"] == "demo"
    assert run.controller["type"] == "none"
    assert run.reference["value"] == pytest.approx(1.0)
    assert run.user_metadata["seed"] == 7
    for metadata in (
        run.system,
        run.controller,
        run.reference,
        run.user_metadata,
    ):
        assert isinstance(metadata, MappingProxyType)
        with pytest.raises(TypeError):
            metadata["new"] = "value"


def test_experiment_run_orders_metadata_deterministically(metrics):
    first = _run(
        metrics,
        system={"z": 3, "a": 1, "m": 2},
        controller={"z": 3, "a": 1},
        reference={"z": 3, "a": 1},
        user_metadata={"z": 3, "a": 1},
        run_id="first",
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
    )
    second = _run(
        metrics,
        system={"m": 2, "a": 1, "z": 3},
        controller={"a": 1, "z": 3},
        reference={"a": 1, "z": 3},
        user_metadata={"a": 1, "z": 3},
        run_id="first",
        created_at=datetime(2026, 8, 31, tzinfo=UTC),
    )

    for left, right in (
        (first.system, second.system),
        (first.controller, second.controller),
        (first.reference, second.reference),
        (first.user_metadata, second.user_metadata),
    ):
        assert tuple(left) == tuple(sorted(left))
        assert tuple(left.items()) == tuple(right.items())
    assert first.reproducibility_record() == second.reproducibility_record()


def test_experiment_run_normalizes_supported_metadata_values(metrics):
    run = _run(
        metrics,
        system={
            "none": None,
            "bool": np.bool_(True),
            "int": np.int64(7),
            "float": np.float32(1.25),
            "str": np.str_("demo"),
            "sequence": (None, np.bool_(False), np.int32(4), np.float64(2.5), "x"),
        },
    )

    assert run.system == {
        "bool": True,
        "float": 1.25,
        "int": 7,
        "none": None,
        "sequence": (None, False, 4, 2.5, "x"),
        "str": "demo",
    }
    assert type(run.system["bool"]) is bool
    assert type(run.system["int"]) is int
    assert type(run.system["float"]) is float
    assert type(run.system["str"]) is str


@pytest.mark.parametrize("field", ["system", "controller", "reference", "user_metadata"])
def test_experiment_run_rejects_non_string_metadata_keys(metrics, field):
    metadata_name = "user" if field == "user_metadata" else field
    with pytest.raises(ValueError, match=f"{metadata_name}.*keys must be strings"):
        _run(metrics, **{field: {1: "invalid"}})


@pytest.mark.parametrize(
    "value",
    [[], {}, {"set"}, object(), (1, (2,)), (1, [2])],
)
def test_experiment_run_rejects_mutable_nested_or_opaque_metadata_values(
    metrics,
    value,
):
    with pytest.raises(ValueError, match="system metadata value"):
        _run(metrics, system={"invalid": value})


@pytest.mark.parametrize(
    "value",
    [np.nan, np.inf, -np.inf, (1.0, np.float64(np.nan))],
)
def test_experiment_run_rejects_nonfinite_metadata_floats(metrics, value):
    with pytest.raises(ValueError, match="system.*finite"):
        _run(metrics, system={"invalid": value})


@pytest.mark.parametrize("value", [None, [], object()])
def test_experiment_run_rejects_invalid_required_metadata_mappings(metrics, value):
    with pytest.raises(TypeError, match="system.*mapping"):
        _run(metrics, system=value)


@pytest.mark.parametrize(
    ("time", "message"),
    [
        ([[2.0, 4.0]], "time must be a 1D array"),
        ([2.0], "time must contain at least two samples"),
        ([2.0, np.nan, 4.0], "time values must be finite"),
        ([2.0, 2.0, 4.0], "time values must be strictly increasing"),
        ([4.0, 2.5, 2.0], "time values must be strictly increasing"),
        ([2.0 + 0.0j, 2.5 + 0.0j, 4.0 + 0.0j], "time values must be real"),
        (["two", "three", "four"], "time must be a real numeric 1D array"),
    ],
)
def test_experiment_run_rejects_invalid_time(metrics, time, message):
    with pytest.raises(ValueError, match=message):
        _run(metrics, time=time)


def test_experiment_run_rejects_time_that_does_not_match_metrics(metrics):
    with pytest.raises(ValueError, match=r"time must .*match metrics\.time"):
        _run(metrics, time=[2.0, 2.5, 5.0])


@pytest.mark.parametrize(
    ("initial_state", "message"),
    [
        ([[0.25, -0.5]], "initial_state must be a 1D array"),
        ([0.25, np.nan], "initial_state values must be finite"),
        ([0.25, np.inf], "initial_state values must be finite"),
        ([0.25 + 0.0j], "initial_state values must be real"),
        (["invalid"], "initial_state must be a real numeric 1D array"),
    ],
)
def test_experiment_run_rejects_invalid_initial_state(
    metrics,
    initial_state,
    message,
):
    with pytest.raises(ValueError, match=message):
        _run(metrics, initial_state=initial_state)


@pytest.mark.parametrize("method", ["", "   ", None])
def test_experiment_run_rejects_invalid_methods(metrics, method):
    with pytest.raises(ValueError, match="method must be a non-empty string"):
        _run(metrics, method=method)


@pytest.mark.parametrize("invalid_metrics", [None, object()])
def test_experiment_run_rejects_invalid_metrics(metrics, invalid_metrics):
    with pytest.raises(TypeError, match="metrics must be a SISOResponseMetrics"):
        _run(metrics, metrics=invalid_metrics)


def test_experiment_run_reproducibility_record_has_exact_contents(metrics):
    created_at = datetime(2026, 8, 31, 12, 30, tzinfo=UTC)
    run = _run(
        metrics,
        system={"stable": True, "order": 2, "name": "demo"},
        controller={"type": "integral_state_feedback", "gains": (2.0, 3.0)},
        reference={"value": 1.0, "type": "step"},
        user_metadata={"seed": np.int64(7), "notes": None},
        run_id="run-001",
        created_at=created_at,
    )

    record = run.reproducibility_record()

    assert tuple(record) == (
        "run_id",
        "created_at",
        "method",
        "start_time",
        "end_time",
        "duration",
        "sample_count",
        "initial_state",
        "system",
        "controller",
        "reference",
        "user_metadata",
        "metrics",
    )
    assert record == {
        "run_id": "run-001",
        "created_at": "2026-08-31T12:30:00+00:00",
        "method": "exact",
        "start_time": 2.0,
        "end_time": 4.0,
        "duration": 2.0,
        "sample_count": 3,
        "initial_state": [0.25, -0.5],
        "system": {"name": "demo", "order": 2, "stable": True},
        "controller": {
            "gains": [2.0, 3.0],
            "type": "integral_state_feedback",
        },
        "reference": {"type": "step", "value": 1.0},
        "user_metadata": {"notes": None, "seed": 7},
        "metrics": {
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
        },
    }
    json.dumps(record, allow_nan=False)


def test_experiment_run_propagates_every_scalar_response_metric(metrics):
    record_metrics = _run(metrics).reproducibility_record()["metrics"]

    assert record_metrics == {
        "final_output": metrics.final_output,
        "final_reference": metrics.final_reference,
        "steady_state_error": metrics.steady_state_error,
        "peak_output": metrics.peak_output,
        "maximum_absolute_tracking_error": metrics.maximum_absolute_tracking_error,
        "rms_tracking_error": metrics.rms_tracking_error,
        "iae": metrics.iae,
        "ise": metrics.ise,
        "overshoot_percent": metrics.overshoot_percent,
        "settling_time": metrics.settling_time,
        "settling_tolerance": metrics.settling_tolerance,
    }


def test_experiment_run_reproducibility_records_are_detached(metrics):
    run = _run(
        metrics,
        controller={"gains": (2.0, 3.0), "type": "state_feedback"},
    )
    record = run.reproducibility_record()
    record["initial_state"][0] = 100.0
    record["system"]["name"] = "changed"
    record["controller"]["gains"][0] = 100.0
    record["metrics"]["iae"] = 100.0

    fresh_record = run.reproducibility_record()
    np.testing.assert_array_equal(run.initial_state, [0.25, -0.5])
    assert fresh_record["initial_state"] == [0.25, -0.5]
    assert fresh_record["system"]["name"] == "demo"
    assert fresh_record["controller"]["gains"] == [2.0, 3.0]
    assert fresh_record["metrics"]["iae"] == pytest.approx(0.5)
