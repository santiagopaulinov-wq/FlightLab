from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from flightlab.analysis import (
    CampaignComparisonEntry,
    CampaignDeltaEntry,
    CampaignMetricChangeProjection,
    CampaignMetricProjectionEnvelope,
    CampaignParameterChange,
    CampaignProjectionScenario,
    CampaignProjectionScenarioResult,
    CampaignSensitivityEntry,
    CampaignSensitivityMatrix,
    SensitivityMatrixParameter,
    campaign_metric_deltas,
    campaign_projection_envelopes,
    campaign_secant_sensitivities,
    campaign_sensitivity_matrix,
    compare_campaign_runs,
    project_campaign_metric_changes,
    project_campaign_scenarios,
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


def _deltas():
    return (
        CampaignDeltaEntry(
            "positive",
            2.0,
            (("iae", 6.0), ("ise", -4.0)),
        ),
        CampaignDeltaEntry(
            "baseline",
            0.0,
            (("iae", 0.0), ("ise", 0.0)),
        ),
        CampaignDeltaEntry(
            "negative",
            -4.0,
            (("iae", 2.0), ("ise", -8.0)),
        ),
    )


def test_campaign_secant_sensitivities_computes_signed_ordered_ratios():
    assert campaign_secant_sensitivities(_deltas()) == (
        CampaignSensitivityEntry(
            "positive",
            2.0,
            (("iae", 3.0), ("ise", -2.0)),
        ),
        CampaignSensitivityEntry(
            "baseline",
            0.0,
            (("iae", None), ("ise", None)),
        ),
        CampaignSensitivityEntry(
            "negative",
            -4.0,
            (("iae", -0.5), ("ise", 2.0)),
        ),
    )


def test_zero_parameter_delta_never_produces_a_secant_sensitivity():
    deltas = (
        CampaignDeltaEntry("baseline", 0.0, (("iae", 0.0),)),
        CampaignDeltaEntry("repeated", 0.0, (("iae", 5.0),)),
    )

    result = campaign_secant_sensitivities(deltas)

    assert result[0].metric_sensitivities == (("iae", None),)
    assert result[1].metric_sensitivities == (("iae", None),)


def test_optional_metric_delta_remains_none_for_nonzero_parameter_delta():
    deltas = (
        CampaignDeltaEntry("baseline", 0.0, (("settling_time", 0.0),)),
        CampaignDeltaEntry("missing", 2.0, (("settling_time", None),)),
    )

    assert campaign_secant_sensitivities(deltas)[1].metric_sensitivities == (
        ("settling_time", None),
    )


def test_empty_secant_sensitivity_input_returns_an_empty_tuple():
    assert campaign_secant_sensitivities(()) == ()


def test_secant_sensitivities_reject_inconsistent_and_malformed_entries():
    inconsistent = (
        CampaignDeltaEntry("baseline", 0.0, (("iae", 0.0), ("ise", 0.0))),
        CampaignDeltaEntry("run", 1.0, (("ise", 1.0), ("iae", 2.0))),
    )
    with pytest.raises(ValueError, match="matching metric layouts"):
        campaign_secant_sensitivities(inconsistent)

    with pytest.raises(TypeError, match=r"deltas\[0\].*CampaignDeltaEntry"):
        campaign_secant_sensitivities((object(),))

    without_baseline = (CampaignDeltaEntry("run", 1.0, (("iae", 2.0),)),)
    with pytest.raises(ValueError, match="zero-delta baseline"):
        campaign_secant_sensitivities(without_baseline)


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (CampaignDeltaEntry("run", float("inf"), (("iae", 0.0),)), "finite"),
        (CampaignDeltaEntry("run", True, (("iae", 0.0),)), "numeric"),
        (CampaignDeltaEntry("run", 0.0, (("iae", float("nan")),)), "finite"),
        (CampaignDeltaEntry("run", 0.0, (("iae", "bad"),)), "numeric"),
    ],
)
def test_secant_sensitivities_reject_nonnumeric_or_nonfinite_values(entry, message):
    with pytest.raises(ValueError, match=message):
        campaign_secant_sensitivities((entry,))


def test_secant_sensitivities_reject_nonfinite_computed_ratio():
    deltas = (
        CampaignDeltaEntry("baseline", 0.0, (("iae", 0.0),)),
        CampaignDeltaEntry("run", 1e-308, (("iae", 1e308),)),
    )

    with pytest.raises(ValueError, match="sensitivity must be finite"):
        campaign_secant_sensitivities(deltas)


def test_secant_sensitivities_are_immutable_deterministic_and_detached():
    source = list(_deltas())
    first = campaign_secant_sensitivities(source)
    repeated = campaign_secant_sensitivities(source)

    assert first == repeated
    source.reverse()
    assert tuple(entry.run_id for entry in first) == (
        "positive",
        "baseline",
        "negative",
    )
    with pytest.raises(TypeError):
        first[0] = first[0]
    with pytest.raises(FrozenInstanceError):
        first[0].parameter_delta = 10.0


def _sensitivity_result(prefix, first_values, second_values):
    return (
        CampaignSensitivityEntry(
            f"{prefix}-baseline",
            0.0,
            (("iae", None), ("ise", None)),
        ),
        CampaignSensitivityEntry(
            f"{prefix}-first",
            1.0,
            (("iae", first_values[0]), ("ise", first_values[1])),
        ),
        CampaignSensitivityEntry(
            f"{prefix}-second",
            2.0,
            (("iae", second_values[0]), ("ise", second_values[1])),
        ),
    )


def test_sensitivity_matrix_preserves_explicit_row_and_column_order():
    parameters = (
        SensitivityMatrixParameter(
            "damping",
            _sensitivity_result("damping", (1.5, -2.0), (3.0, -4.0)),
            "damping-second",
        ),
        SensitivityMatrixParameter(
            "gain",
            _sensitivity_result("gain", (-0.5, None), (-1.0, 4.0)),
            "gain-first",
        ),
    )

    matrix = campaign_sensitivity_matrix(parameters)

    assert matrix == CampaignSensitivityMatrix(
        parameter_names=("damping", "gain"),
        metric_names=("iae", "ise"),
        representative_run_ids=("damping-second", "gain-first"),
        values=((3.0, -0.5), (-4.0, None)),
    )


def test_single_parameter_matrix_selects_run_at_any_campaign_position():
    sensitivities = _sensitivity_result("gain", (2.0, 3.0), (4.0, 5.0))

    first = campaign_sensitivity_matrix(
        (SensitivityMatrixParameter("gain", sensitivities, "gain-first"),)
    )
    last = campaign_sensitivity_matrix(
        (SensitivityMatrixParameter("gain", sensitivities, "gain-second"),)
    )

    assert first.values == ((2.0,), (3.0,))
    assert last.values == ((4.0,), (5.0,))


def test_empty_parameter_collection_returns_an_explicit_empty_matrix():
    assert campaign_sensitivity_matrix(()) == CampaignSensitivityMatrix(
        parameter_names=(),
        metric_names=(),
        representative_run_ids=(),
        values=(),
    )


def test_matrix_rejects_incompatible_metric_layouts():
    first = _sensitivity_result("first", (1.0, 2.0), (3.0, 4.0))
    second = (
        CampaignSensitivityEntry("second-baseline", 0.0, (("ise", None),)),
        CampaignSensitivityEntry("second-run", 1.0, (("ise", 2.0),)),
    )

    with pytest.raises(ValueError, match="matching metric layouts"):
        campaign_sensitivity_matrix(
            (
                SensitivityMatrixParameter("first", first, "first-first"),
                SensitivityMatrixParameter("second", second, "second-run"),
            )
        )


def test_matrix_rejects_invalid_parameter_names_and_representative_ids():
    sensitivities = _sensitivity_result("gain", (1.0, 2.0), (3.0, 4.0))
    with pytest.raises(ValueError, match="name must be non-empty"):
        campaign_sensitivity_matrix(
            (SensitivityMatrixParameter(" ", sensitivities, "gain-first"),)
        )
    with pytest.raises(ValueError, match="duplicate parameter name 'gain'"):
        campaign_sensitivity_matrix(
            (
                SensitivityMatrixParameter("gain", sensitivities, "gain-first"),
                SensitivityMatrixParameter("gain", sensitivities, "gain-second"),
            )
        )
    with pytest.raises(ValueError, match="is not in"):
        campaign_sensitivity_matrix(
            (SensitivityMatrixParameter("gain", sensitivities, "missing"),)
        )
    with pytest.raises(ValueError, match="nonzero parameter delta"):
        campaign_sensitivity_matrix(
            (SensitivityMatrixParameter("gain", sensitivities, "gain-baseline"),)
        )


def test_matrix_rejects_duplicate_representative_and_source_run_ids():
    first = _sensitivity_result("shared", (1.0, 2.0), (3.0, 4.0))
    with pytest.raises(ValueError, match="duplicate representative run_id"):
        campaign_sensitivity_matrix(
            (
                SensitivityMatrixParameter("one", first, "shared-first"),
                SensitivityMatrixParameter("two", first, "shared-first"),
            )
        )

    duplicate_source = (first[0], first[1], first[1])
    with pytest.raises(ValueError, match="duplicate run_id 'shared-first'"):
        campaign_sensitivity_matrix(
            (SensitivityMatrixParameter("one", duplicate_source, "shared-first"),)
        )


@pytest.mark.parametrize("value", [True, "bad", float("inf"), float("nan")])
def test_matrix_rejects_nonnumeric_or_nonfinite_sensitivities(value):
    sensitivities = (
        CampaignSensitivityEntry("baseline", 0.0, (("iae", None),)),
        CampaignSensitivityEntry("run", 1.0, (("iae", value),)),
    )

    with pytest.raises(ValueError, match="metric 'iae'.*(numeric|finite)"):
        campaign_sensitivity_matrix(
            (SensitivityMatrixParameter("gain", sensitivities, "run"),)
        )


def test_matrix_rejects_malformed_sensitivity_entries():
    with pytest.raises(TypeError, match="CampaignSensitivityEntry"):
        campaign_sensitivity_matrix(
            (SensitivityMatrixParameter("gain", (object(),), "run"),)
        )


def test_sensitivity_matrix_is_immutable_deterministic_and_detached():
    source = [
        SensitivityMatrixParameter(
            "gain",
            _sensitivity_result("gain", (1.0, None), (2.0, 3.0)),
            "gain-first",
        )
    ]
    first = campaign_sensitivity_matrix(source)
    repeated = campaign_sensitivity_matrix(source)

    assert first == repeated
    source.clear()
    assert first.values == ((1.0,), (None,))
    with pytest.raises(FrozenInstanceError):
        first.values = ()
    with pytest.raises(TypeError):
        first.values[0][0] = 9.0


def test_projection_with_one_parameter_and_metric():
    matrix = CampaignSensitivityMatrix(
        ("gain",), ("iae",), ("gain-run",), ((2.5,),)
    )

    assert project_campaign_metric_changes(
        matrix, (CampaignParameterChange("gain", -2.0),)
    ) == CampaignMetricChangeProjection(
        parameter_names=("gain",),
        metric_names=("iae",),
        parameter_changes=(-2.0,),
        predicted_metric_changes=(-5.0,),
    )


def test_projection_preserves_order_and_computes_row_vector_products():
    matrix = CampaignSensitivityMatrix(
        parameter_names=("damping", "gain"),
        metric_names=("iae", "ise"),
        representative_run_ids=("damping-run", "gain-run"),
        values=((2.0, -1.0), (-3.0, 4.0)),
    )

    projection = project_campaign_metric_changes(
        matrix,
        (
            CampaignParameterChange("damping", 3.0),
            CampaignParameterChange("gain", -2.0),
        ),
    )

    assert projection.parameter_names == ("damping", "gain")
    assert projection.metric_names == ("iae", "ise")
    assert projection.parameter_changes == (3.0, -2.0)
    assert projection.predicted_metric_changes == (8.0, -17.0)


def test_zero_change_vector_produces_exact_numeric_zeros():
    matrix = CampaignSensitivityMatrix(
        ("first", "second"),
        ("iae", "ise"),
        ("first-run", "second-run"),
        ((2.0, -1.0), (3.0, 4.0)),
    )

    projection = project_campaign_metric_changes(
        matrix,
        (CampaignParameterChange("first", 0.0), CampaignParameterChange("second", 0)),
    )

    assert projection.predicted_metric_changes == (0.0, 0.0)


def test_projection_propagates_undefined_sensitivity_even_for_zero_change():
    matrix = CampaignSensitivityMatrix(
        ("gain",),
        ("iae", "settling_time"),
        ("gain-run",),
        ((2.0,), (None,)),
    )

    projection = project_campaign_metric_changes(
        matrix, (CampaignParameterChange("gain", 0.0),)
    )

    assert projection.predicted_metric_changes == (0.0, None)


def test_empty_matrix_and_vector_produce_an_empty_projection():
    matrix = CampaignSensitivityMatrix((), (), (), ())

    assert project_campaign_metric_changes(matrix, ()) == (
        CampaignMetricChangeProjection((), (), (), ())
    )


def test_projection_rejects_dimension_and_parameter_order_mismatches():
    matrix = CampaignSensitivityMatrix(
        ("first", "second"),
        ("iae",),
        ("first-run", "second-run"),
        ((1.0, 2.0),),
    )
    with pytest.raises(ValueError, match="column count"):
        project_campaign_metric_changes(
            matrix, (CampaignParameterChange("first", 1.0),)
        )
    with pytest.raises(ValueError, match=r"parameter_changes\[0\].*'first'"):
        project_campaign_metric_changes(
            matrix,
            (
                CampaignParameterChange("second", 2.0),
                CampaignParameterChange("first", 1.0),
            ),
        )


@pytest.mark.parametrize("value", [True, "bad", None, float("inf"), float("nan")])
def test_projection_rejects_nonnumeric_or_nonfinite_changes(value):
    matrix = CampaignSensitivityMatrix(
        ("gain",), ("iae",), ("gain-run",), ((1.0,),)
    )

    with pytest.raises(ValueError, match=r"parameter_changes\[0\].*(numeric|finite)"):
        project_campaign_metric_changes(
            matrix, (CampaignParameterChange("gain", value),)
        )


def test_projection_rejects_malformed_or_nonfinite_matrix_values():
    malformed = CampaignSensitivityMatrix(
        ("gain",), ("iae",), ("gain-run",), ((1.0, 2.0),)
    )
    with pytest.raises(ValueError, match="match parameter columns"):
        project_campaign_metric_changes(
            malformed, (CampaignParameterChange("gain", 1.0),)
        )

    nonfinite = CampaignSensitivityMatrix(
        ("gain",), ("iae",), ("gain-run",), ((float("inf"),),)
    )
    with pytest.raises(ValueError, match=r"matrix.values\[0\]\[0\].*finite"):
        project_campaign_metric_changes(
            nonfinite, (CampaignParameterChange("gain", 1.0),)
        )


def test_projection_rejects_nonfinite_products_and_sums():
    product_overflow = CampaignSensitivityMatrix(
        ("gain",), ("iae",), ("gain-run",), ((1e308,),)
    )
    with pytest.raises(ValueError, match="contribution must be finite"):
        project_campaign_metric_changes(
            product_overflow, (CampaignParameterChange("gain", 1e308),)
        )

    sum_overflow = CampaignSensitivityMatrix(
        ("first", "second"),
        ("iae",),
        ("first-run", "second-run"),
        ((1e308, 1e308),),
    )
    with pytest.raises(ValueError, match="metric change 'iae' must be finite"):
        project_campaign_metric_changes(
            sum_overflow,
            (
                CampaignParameterChange("first", 1.0),
                CampaignParameterChange("second", 1.0),
            ),
        )


def test_projection_is_immutable_deterministic_and_detached():
    matrix = CampaignSensitivityMatrix(
        ("gain",), ("iae",), ("gain-run",), ((2.0,),)
    )
    changes = [CampaignParameterChange("gain", 3.0)]

    first = project_campaign_metric_changes(matrix, changes)
    repeated = project_campaign_metric_changes(matrix, changes)

    assert first == repeated
    changes[0] = CampaignParameterChange("gain", 99.0)
    assert first.parameter_changes == (3.0,)
    assert first.predicted_metric_changes == (6.0,)
    with pytest.raises(FrozenInstanceError):
        first.predicted_metric_changes = ()


def _projection_matrix():
    return CampaignSensitivityMatrix(
        parameter_names=("damping", "gain"),
        metric_names=("iae", "settling_time"),
        representative_run_ids=("damping-run", "gain-run"),
        values=((2.0, -1.0), (None, 4.0)),
    )


def test_one_projection_scenario_retains_its_name_and_projection():
    scenario = CampaignProjectionScenario(
        "candidate",
        (
            CampaignParameterChange("damping", 1.0),
            CampaignParameterChange("gain", -2.0),
        ),
    )

    assert project_campaign_scenarios(_projection_matrix(), (scenario,)) == (
        CampaignProjectionScenarioResult(
            name="candidate",
            projection=CampaignMetricChangeProjection(
                parameter_names=("damping", "gain"),
                metric_names=("iae", "settling_time"),
                parameter_changes=(1.0, -2.0),
                predicted_metric_changes=(4.0, None),
            ),
        ),
    )


def test_multiple_projection_scenarios_preserve_exact_caller_order():
    scenarios = (
        CampaignProjectionScenario(
            "negative",
            (
                CampaignParameterChange("damping", -1.0),
                CampaignParameterChange("gain", -1.0),
            ),
        ),
        CampaignProjectionScenario(
            "zero",
            (
                CampaignParameterChange("damping", 0.0),
                CampaignParameterChange("gain", 0.0),
            ),
        ),
        CampaignProjectionScenario(
            "positive",
            (
                CampaignParameterChange("damping", 2.0),
                CampaignParameterChange("gain", 1.0),
            ),
        ),
    )

    results = project_campaign_scenarios(_projection_matrix(), scenarios)

    assert tuple(result.name for result in results) == (
        "negative",
        "zero",
        "positive",
    )
    assert tuple(
        result.projection.predicted_metric_changes[0] for result in results
    ) == (-1.0, 0.0, 3.0)
    assert all(
        result.projection.predicted_metric_changes[1] is None for result in results
    )


def test_empty_scenario_collection_returns_an_empty_tuple():
    assert project_campaign_scenarios(_projection_matrix(), ()) == ()


def test_scenario_generator_is_snapshotted_and_preserves_order():
    definitions = [
        ("first", 1.0),
        ("second", 2.0),
    ]
    scenarios = (
        CampaignProjectionScenario(
            name,
            (
                CampaignParameterChange("damping", change),
                CampaignParameterChange("gain", 0.0),
            ),
        )
        for name, change in definitions
    )

    results = project_campaign_scenarios(_projection_matrix(), scenarios)

    assert tuple(result.name for result in results) == ("first", "second")
    assert tuple(result.projection.parameter_changes for result in results) == (
        (1.0, 0.0),
        (2.0, 0.0),
    )


def test_scenarios_reject_blank_and_duplicate_names_before_projection():
    changes = (
        CampaignParameterChange("damping", 1.0),
        CampaignParameterChange("gain", 1.0),
    )
    with pytest.raises(ValueError, match="name must be non-empty"):
        project_campaign_scenarios(
            _projection_matrix(), (CampaignProjectionScenario(" ", changes),)
        )
    with pytest.raises(ValueError, match="duplicate scenario name 'same'"):
        project_campaign_scenarios(
            _projection_matrix(),
            (
                CampaignProjectionScenario("same", changes),
                CampaignProjectionScenario("same", changes),
            ),
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ((CampaignParameterChange("damping", 1.0),), "column count"),
        (
            (
                CampaignParameterChange("gain", 1.0),
                CampaignParameterChange("damping", 1.0),
            ),
            r"parameter_changes\[0\].*'damping'",
        ),
        (
            (CampaignParameterChange("damping", 1.0), object()),
            r"parameter_changes\[1\].*CampaignParameterChange",
        ),
    ],
)
def test_malformed_scenario_vectors_propagate_projection_errors(changes, message):
    with pytest.raises((TypeError, ValueError), match=message):
        project_campaign_scenarios(
            _projection_matrix(),
            (CampaignProjectionScenario("invalid", changes),),
        )


def test_scenarios_validate_complete_collection_before_any_projection(monkeypatch):
    calls = []

    def projection(*args):
        calls.append(args)
        raise AssertionError("projection must not run")

    monkeypatch.setattr("flightlab.analysis.project_campaign_metric_changes", projection)
    valid = CampaignProjectionScenario(
        "valid",
        (
            CampaignParameterChange("damping", 1.0),
            CampaignParameterChange("gain", 1.0),
        ),
    )

    with pytest.raises(TypeError, match=r"scenarios\[1\].*CampaignProjectionScenario"):
        project_campaign_scenarios(_projection_matrix(), (valid, object()))
    assert calls == []


def test_scenarios_delegate_each_projection_exactly_once(monkeypatch):
    calls = []
    expected = CampaignMetricChangeProjection(
        ("damping", "gain"), ("iae", "settling_time"), (1.0, 2.0), (0.0, None)
    )

    def projection(matrix, changes):
        calls.append((matrix, changes))
        return expected

    monkeypatch.setattr("flightlab.analysis.project_campaign_metric_changes", projection)
    scenarios = tuple(
        CampaignProjectionScenario(
            name,
            (
                CampaignParameterChange("damping", 1.0),
                CampaignParameterChange("gain", 2.0),
            ),
        )
        for name in ("first", "second")
    )

    results = project_campaign_scenarios(_projection_matrix(), scenarios)

    assert len(calls) == 2
    assert tuple(result.name for result in results) == ("first", "second")
    assert all(result.projection is expected for result in results)


def test_scenario_results_are_immutable_deterministic_and_detached():
    source = [
        CampaignProjectionScenario(
            "scenario",
            (
                CampaignParameterChange("damping", 1.0),
                CampaignParameterChange("gain", 2.0),
            ),
        )
    ]
    first = project_campaign_scenarios(_projection_matrix(), source)
    repeated = project_campaign_scenarios(_projection_matrix(), source)

    assert first == repeated
    source.clear()
    assert first[0].name == "scenario"
    assert first[0].projection.parameter_changes == (1.0, 2.0)
    with pytest.raises(FrozenInstanceError):
        first[0].name = "changed"
    with pytest.raises(TypeError):
        first[0] = first[0]


def _scenario_result(name, iae, ise):
    return CampaignProjectionScenarioResult(
        name=name,
        projection=CampaignMetricChangeProjection(
            parameter_names=("gain",),
            metric_names=("iae", "ise"),
            parameter_changes=(1.0,),
            predicted_metric_changes=(iae, ise),
        ),
    )


def test_one_scenario_is_both_minimum_and_maximum():
    assert campaign_projection_envelopes((_scenario_result("only", 2.0, -1.0),)) == (
        CampaignMetricProjectionEnvelope("iae", 2.0, "only", 2.0, "only"),
        CampaignMetricProjectionEnvelope("ise", -1.0, "only", -1.0, "only"),
    )


def test_projection_envelopes_find_signed_extrema_in_metric_order():
    scenarios = (
        _scenario_result("middle", 1.0, -2.0),
        _scenario_result("low-iae", -3.0, 4.0),
        _scenario_result("high-iae", 5.0, -6.0),
    )

    assert campaign_projection_envelopes(scenarios) == (
        CampaignMetricProjectionEnvelope(
            "iae", -3.0, "low-iae", 5.0, "high-iae"
        ),
        CampaignMetricProjectionEnvelope(
            "ise", -6.0, "high-iae", 4.0, "low-iae"
        ),
    )


def test_projection_envelope_ties_select_first_scenario_in_input_order():
    scenarios = (
        _scenario_result("first", -1.0, 3.0),
        _scenario_result("second", -1.0, 3.0),
    )

    envelopes = campaign_projection_envelopes(scenarios)

    assert envelopes[0].minimum_scenario == "first"
    assert envelopes[0].maximum_scenario == "first"
    assert envelopes[1].minimum_scenario == "first"
    assert envelopes[1].maximum_scenario == "first"


def test_envelopes_ignore_some_none_values_and_preserve_all_none_as_undefined():
    scenarios = (
        _scenario_result("undefined", None, None),
        _scenario_result("defined", 2.0, None),
        _scenario_result("lower", -1.0, None),
    )

    assert campaign_projection_envelopes(scenarios) == (
        CampaignMetricProjectionEnvelope("iae", -1.0, "lower", 2.0, "defined"),
        CampaignMetricProjectionEnvelope("ise", None, None, None, None),
    )


def test_empty_scenario_results_have_no_metric_envelopes():
    assert campaign_projection_envelopes(()) == ()


def test_projection_envelopes_materialize_generator_inputs():
    scenarios = (
        _scenario_result(name, value, -value)
        for name, value in (("first", 1.0), ("second", 2.0))
    )

    envelopes = campaign_projection_envelopes(scenarios)

    assert envelopes[0].maximum_scenario == "second"
    assert envelopes[1].minimum_scenario == "second"


def test_envelopes_reject_blank_and_duplicate_scenario_names():
    with pytest.raises(ValueError, match="name must be non-empty"):
        campaign_projection_envelopes((_scenario_result(" ", 1.0, 2.0),))
    with pytest.raises(ValueError, match="duplicate scenario name 'same'"):
        campaign_projection_envelopes(
            (_scenario_result("same", 1.0, 2.0), _scenario_result("same", 2.0, 3.0))
        )


def test_envelopes_reject_incompatible_projection_layouts():
    incompatible = CampaignProjectionScenarioResult(
        "other",
        CampaignMetricChangeProjection(
            ("gain",), ("ise", "iae"), (1.0,), (2.0, 1.0)
        ),
    )

    with pytest.raises(ValueError, match="matching parameter and metric layouts"):
        campaign_projection_envelopes(
            (_scenario_result("first", 1.0, 2.0), incompatible)
        )


@pytest.mark.parametrize("value", [True, "bad", float("inf"), float("nan")])
def test_envelopes_reject_nonnumeric_or_nonfinite_predictions(value):
    with pytest.raises(ValueError, match=r"predicted_metric_changes\[0\].*(numeric|finite)"):
        campaign_projection_envelopes((_scenario_result("invalid", value, 1.0),))


def test_envelopes_reject_malformed_projection_metadata():
    malformed = CampaignProjectionScenarioResult(
        "malformed",
        CampaignMetricChangeProjection(("gain",), ("iae",), (), (1.0,)),
    )
    with pytest.raises(ValueError, match="parameter metadata has inconsistent lengths"):
        campaign_projection_envelopes((malformed,))

    with pytest.raises(TypeError, match="CampaignProjectionScenarioResult"):
        campaign_projection_envelopes((object(),))


def test_projection_envelopes_are_immutable_deterministic_and_detached():
    source = [
        _scenario_result("first", -1.0, 2.0),
        _scenario_result("second", 3.0, -4.0),
    ]
    first = campaign_projection_envelopes(source)
    repeated = campaign_projection_envelopes(source)

    assert first == repeated
    source.clear()
    assert first[0].minimum == -1.0
    assert first[0].maximum == 3.0
    with pytest.raises(FrozenInstanceError):
        first[0].minimum = 0.0
    with pytest.raises(TypeError):
        first[0] = first[0]
