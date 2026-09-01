from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from flightlab.analysis import (
    CampaignComparisonEntry,
    CampaignDeltaEntry,
    campaign_metric_deltas,
    compare_campaign_runs,
)
from flightlab.experiment import experiment_run
from flightlab.persistence import (
    ExperimentCampaignBundle,
    ExperimentCampaignManifest,
    campaign_bundle_record,
)
from flightlab.response import response_metrics


def _run(run_id, gain, *, overshoot=True):
    time = [0.0, 1.0, 2.0]
    output = [0.0, 1.2, 1.0] if overshoot else [0.0, 0.5, 1.0]
    metrics = response_metrics(time, output, [1.0, 1.0, 1.0])
    return experiment_run(
        time=time,
        initial_state=[0.0],
        metrics=metrics,
        method="exact",
        system={"order": 1},
        controller={"gain": gain},
        reference={"type": "step"},
        user_metadata={"label": run_id},
        run_id=run_id,
        created_at=datetime(2026, 9, 1, tzinfo=UTC),
    )


def _bundle_record(runs):
    records = tuple(run.reproducibility_record() for run in runs)
    bundle = ExperimentCampaignBundle(
        manifest=ExperimentCampaignManifest(
            campaign_id="comparison",
            created_at="2026-09-01T00:00:00+00:00",
            run_ids=tuple(run.run_id for run in runs),
        ),
        records=records,
    )
    return campaign_bundle_record(bundle)


def test_compare_campaign_runs_preserves_run_and_metric_order():
    record = _bundle_record(
        (_run("third", 3.0), _run("first", 1.0, overshoot=False))
    )

    comparison = compare_campaign_runs(
        record,
        "controller",
        "gain",
        ("iae", "overshoot_percent", "settling_time"),
    )

    assert comparison == (
        CampaignComparisonEntry(
            run_id="third",
            parameter_value=3.0,
            metric_values=(
                ("iae", 0.7),
                ("overshoot_percent", pytest.approx(20.0)),
                ("settling_time", 2.0),
            ),
        ),
        CampaignComparisonEntry(
            run_id="first",
            parameter_value=1.0,
            metric_values=(
                ("iae", 1.0),
                ("overshoot_percent", 0.0),
                ("settling_time", 2.0),
            ),
        ),
    )


def test_comparison_is_immutable_deterministic_and_source_independent():
    record = _bundle_record((_run("one", 1.0),))
    first = compare_campaign_runs(record, "controller", "gain", ["iae"])
    repeated = compare_campaign_runs(record, "controller", "gain", ["iae"])

    assert first == repeated
    with pytest.raises(TypeError):
        first[0] = first[0]
    with pytest.raises(FrozenInstanceError):
        first[0].run_id = "changed"

    record["records"][0]["controller"]["gain"] = 99.0
    assert first[0].parameter_value == 1.0


def test_empty_campaign_comparison_is_deterministic():
    record = _bundle_record(())

    assert compare_campaign_runs(record, "controller", "gain", ["iae"]) == ()


@pytest.mark.parametrize(
    ("category", "key"),
    [
        ("system", "order"),
        ("controller", "gain"),
        ("reference", "type"),
        ("user_metadata", "label"),
    ],
)
def test_parameter_selection_supports_each_provenance_category(category, key):
    record = _bundle_record((_run("run", 2.0),))

    result = compare_campaign_runs(record, category, key, ["iae"])

    assert result[0].parameter_value == record["records"][0][category][key]


def test_optional_existing_metric_values_are_preserved():
    run = _run("zero", 0.0)
    record = _bundle_record((run,))
    record["records"][0]["metrics"]["settling_time"] = None

    result = compare_campaign_runs(record, "controller", "gain", ["settling_time"])

    assert result[0].metric_values == (("settling_time", None),)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda record: record.pop("manifest"), "exactly manifest and records"),
        (lambda record: record["manifest"]["run_ids"].reverse(), "does not match"),
        (lambda record: record["records"][0].pop("metrics"), "missing keys"),
    ],
)
def test_malformed_bundle_records_are_rejected(mutation, message):
    record = _bundle_record((_run("run", 1.0), _run("other", 2.0)))
    mutation(record)

    with pytest.raises(ValueError, match=message):
        compare_campaign_runs(record, "controller", "gain", ["iae"])


def test_missing_and_nonscalar_parameters_are_rejected():
    missing = _bundle_record((_run("run", 1.0),))
    missing["records"][0]["controller"].pop("gain")
    with pytest.raises(ValueError, match="missing parameter controller.gain"):
        compare_campaign_runs(missing, "controller", "gain", ["iae"])

    nonscalar = _bundle_record((_run("run", 1.0),))
    nonscalar["records"][0]["controller"]["gain"] = [1.0, 2.0]
    with pytest.raises(ValueError, match="must be scalar"):
        compare_campaign_runs(nonscalar, "controller", "gain", ["iae"])


@pytest.mark.parametrize(
    ("metrics", "message"),
    [
        ([], "at least one"),
        (["unknown"], "unknown metric"),
        (["iae", "iae"], "must not contain duplicates"),
    ],
)
def test_invalid_metric_selections_are_rejected(metrics, message):
    with pytest.raises(ValueError, match=message):
        compare_campaign_runs(
            _bundle_record((_run("run", 1.0),)),
            "controller",
            "gain",
            metrics,
        )


def test_missing_and_nonscalar_metrics_are_rejected():
    missing = _bundle_record((_run("run", 1.0),))
    missing["records"][0]["metrics"].pop("iae")
    with pytest.raises(ValueError, match="missing keys.*iae"):
        compare_campaign_runs(missing, "controller", "gain", ["iae"])

    nonscalar = _bundle_record((_run("run", 1.0),))
    nonscalar["records"][0]["metrics"]["iae"] = [1.0]
    with pytest.raises(ValueError, match=r"metrics\.iae.*finite real number"):
        compare_campaign_runs(nonscalar, "controller", "gain", ["iae"])


def _comparison():
    return (
        CampaignComparisonEntry(
            run_id="first",
            parameter_value=1.0,
            metric_values=(("iae", 5.0), ("ise", 10.0)),
        ),
        CampaignComparisonEntry(
            run_id="middle",
            parameter_value=3.0,
            metric_values=(("iae", 2.0), ("ise", 14.0)),
        ),
        CampaignComparisonEntry(
            run_id="last",
            parameter_value=-1.0,
            metric_values=(("iae", 8.0), ("ise", 4.0)),
        ),
    )


@pytest.mark.parametrize("baseline_run_id", ["first", "middle", "last"])
def test_campaign_metric_deltas_supports_baseline_at_any_position(baseline_run_id):
    comparison = _comparison()

    result = campaign_metric_deltas(comparison, baseline_run_id)

    assert tuple(entry.run_id for entry in result) == ("first", "middle", "last")
    baseline_index = ("first", "middle", "last").index(baseline_run_id)
    assert result[baseline_index].parameter_delta == 0.0
    assert result[baseline_index].metric_deltas == (
        ("iae", 0.0),
        ("ise", 0.0),
    )


def test_campaign_metric_deltas_preserves_order_and_computes_signed_differences():
    result = campaign_metric_deltas(_comparison(), "middle")

    assert result == (
        CampaignDeltaEntry(
            run_id="first",
            parameter_delta=-2.0,
            metric_deltas=(("iae", 3.0), ("ise", -4.0)),
        ),
        CampaignDeltaEntry(
            run_id="middle",
            parameter_delta=0.0,
            metric_deltas=(("iae", 0.0), ("ise", 0.0)),
        ),
        CampaignDeltaEntry(
            run_id="last",
            parameter_delta=-4.0,
            metric_deltas=(("iae", 6.0), ("ise", -10.0)),
        ),
    )


def test_optional_metric_delta_is_none_when_either_value_is_none():
    comparison = (
        CampaignComparisonEntry("baseline", 1.0, (("settling_time", 2.0),)),
        CampaignComparisonEntry("missing", 2.0, (("settling_time", None),)),
    )
    assert campaign_metric_deltas(comparison, "baseline")[1].metric_deltas == (
        ("settling_time", None),
    )

    baseline_missing = campaign_metric_deltas(comparison, "missing")
    assert tuple(entry.metric_deltas for entry in baseline_missing) == (
        (("settling_time", None),),
        (("settling_time", None),),
    )


def test_empty_comparison_cannot_contain_the_explicit_baseline():
    with pytest.raises(ValueError, match="baseline run_id 'baseline'.*not in comparison"):
        campaign_metric_deltas((), "baseline")


def test_unknown_baseline_and_duplicate_run_ids_are_rejected():
    with pytest.raises(ValueError, match="baseline run_id 'unknown'.*not in comparison"):
        campaign_metric_deltas(_comparison(), "unknown")

    duplicate = (_comparison()[0], _comparison()[0])
    with pytest.raises(ValueError, match="duplicate run_id 'first'"):
        campaign_metric_deltas(duplicate, "first")


def test_incompatible_metric_layouts_are_rejected():
    comparison = (
        CampaignComparisonEntry("first", 1.0, (("iae", 1.0), ("ise", 2.0))),
        CampaignComparisonEntry("second", 2.0, (("ise", 2.0), ("iae", 1.0))),
    )

    with pytest.raises(ValueError, match="matching metric layouts"):
        campaign_metric_deltas(comparison, "first")


@pytest.mark.parametrize("value", [True, "1.0", None, float("inf"), float("nan")])
def test_nonnumeric_or_nonfinite_parameter_values_are_rejected(value):
    comparison = (CampaignComparisonEntry("run", value, (("iae", 1.0),)),)

    with pytest.raises(ValueError, match="parameter_value must be (numeric|finite)"):
        campaign_metric_deltas(comparison, "run")


@pytest.mark.parametrize("value", [True, "1.0", float("inf"), float("nan")])
def test_nonnumeric_or_nonfinite_metric_values_are_rejected(value):
    comparison = (CampaignComparisonEntry("run", 1.0, (("iae", value),)),)

    with pytest.raises(ValueError, match="metric 'iae' must be (numeric|finite)"):
        campaign_metric_deltas(comparison, "run")


def test_campaign_metric_deltas_is_immutable_deterministic_and_detached():
    comparison = list(_comparison())
    first = campaign_metric_deltas(comparison, "first")
    repeated = campaign_metric_deltas(comparison, "first")

    assert first == repeated
    comparison.reverse()
    assert tuple(entry.run_id for entry in first) == ("first", "middle", "last")
    with pytest.raises(TypeError):
        first[0] = first[0]
    with pytest.raises(FrozenInstanceError):
        first[0].parameter_delta = 10.0
