from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from flightlab import analysis
from flightlab.analysis import (
    CampaignComparisonEntry,
    CampaignDeltaEntry,
    CampaignMetricChangeProjection,
    CampaignMetricProjectionEnvelope,
    CampaignMetricProjectionLimit,
    CampaignMetricProjectionLimitResult,
    CampaignMetricResidual,
    CampaignMetricResidualTolerance,
    CampaignMetricResidualToleranceResult,
    CampaignParameterChange,
    CampaignProjectionResiduals,
    CampaignProjectionResidualToleranceResults,
    CampaignProjectionScenario,
    CampaignProjectionScenarioResult,
    CampaignProjectionValidationCase,
    CampaignProjectionValidationResult,
    CampaignRobustnessVerdict,
    CampaignSensitivityEntry,
    CampaignSensitivityMatrix,
    SensitivityMatrixParameter,
    campaign_metric_deltas,
    campaign_projection_envelopes,
    campaign_projection_residuals,
    campaign_robustness_verdict,
    campaign_secant_sensitivities,
    campaign_sensitivity_matrix,
    check_campaign_projection_envelope_limits,
    check_campaign_projection_residual_tolerances,
    compare_campaign_runs,
    project_campaign_metric_changes,
    project_campaign_scenarios,
    validate_campaign_projection_cases,
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


def _observed_delta(run_id="observed", iae=2.0, ise=-1.0):
    return CampaignDeltaEntry(
        run_id=run_id,
        parameter_delta=1.0,
        metric_deltas=(("iae", iae), ("ise", ise)),
    )


def test_projection_residuals_compute_signed_values_and_retain_identities():
    result = campaign_projection_residuals(
        _scenario_result("stress", 2.0, 3.0),
        _observed_delta("measured", 2.0, 1.0),
    )

    assert result == CampaignProjectionResiduals(
        scenario_name="stress",
        observed_run_id="measured",
        metric_residuals=(
            CampaignMetricResidual("iae", 2.0, 2.0, 0.0),
            CampaignMetricResidual("ise", 3.0, 1.0, -2.0),
        ),
    )


def test_projection_residuals_preserve_metric_order_and_positive_residual():
    result = campaign_projection_residuals(
        _scenario_result("scenario", -3.0, 1.0),
        _observed_delta(iae=-1.0, ise=4.0),
    )

    assert tuple(item.metric_name for item in result.metric_residuals) == ("iae", "ise")
    assert tuple(item.residual for item in result.metric_residuals) == (2.0, 3.0)


@pytest.mark.parametrize(
    ("projected", "observed"),
    [(None, 1.0), (1.0, None), (None, None)],
)
def test_projection_residuals_propagate_undefined_values(projected, observed):
    result = campaign_projection_residuals(
        _scenario_result("scenario", projected, 0.0),
        _observed_delta(iae=observed, ise=0.0),
    )

    assert result.metric_residuals[0] == CampaignMetricResidual(
        "iae", projected, observed, None
    )


def test_projection_residuals_reject_incompatible_metric_layouts():
    observed = CampaignDeltaEntry(
        "observed", 1.0, (("ise", 1.0), ("iae", 2.0))
    )
    with pytest.raises(ValueError, match="match exactly in name and order"):
        campaign_projection_residuals(_scenario_result("scenario", 1.0, 2.0), observed)


@pytest.mark.parametrize(
    ("observed", "match"),
    [
        (object(), "CampaignDeltaEntry"),
        (CampaignDeltaEntry(" ", 1.0, (("iae", 1.0),)), "run_id"),
        (CampaignDeltaEntry("run", True, (("iae", 1.0),)), "numeric"),
        (CampaignDeltaEntry("run", 1.0, ()), "must not be empty"),
        (CampaignDeltaEntry("run", 1.0, (("unknown", 1.0),)), "unknown metric"),
        (
            CampaignDeltaEntry("run", 1.0, (("iae", 1.0), ("iae", 2.0))),
            "duplicate metric",
        ),
        (CampaignDeltaEntry("run", 1.0, (("iae", True),)), "numeric"),
        (CampaignDeltaEntry("run", 1.0, (("iae", float("inf")),)), "finite"),
    ],
)
def test_projection_residuals_reject_malformed_observed_entries(observed, match):
    with pytest.raises((TypeError, ValueError), match=match):
        campaign_projection_residuals(_scenario_result("scenario", 1.0, 2.0), observed)


def test_projection_residuals_reject_malformed_scenarios_and_overflow():
    with pytest.raises(TypeError, match="CampaignProjectionScenarioResult"):
        campaign_projection_residuals(object(), _observed_delta())
    with pytest.raises(ValueError, match="name must be non-empty"):
        campaign_projection_residuals(_scenario_result(" ", 1.0, 2.0), _observed_delta())
    with pytest.raises(ValueError, match="residual must be finite"):
        campaign_projection_residuals(
            _scenario_result("scenario", -1e308, 0.0),
            _observed_delta(iae=1e308, ise=0.0),
        )


def test_projection_residuals_are_immutable_deterministic_and_detached():
    scenario = _scenario_result("scenario", 1.0, -2.0)
    observed = _observed_delta("run", 3.0, 4.0)

    first = campaign_projection_residuals(scenario, observed)
    repeated = campaign_projection_residuals(scenario, observed)

    assert first == repeated
    assert first is not repeated
    assert first.metric_residuals[0] is not repeated.metric_residuals[0]
    with pytest.raises(FrozenInstanceError):
        first.scenario_name = "changed"
    with pytest.raises(FrozenInstanceError):
        first.metric_residuals[0].residual = 0.0


def _residual_result(iae=2.0, ise=-1.0):
    return CampaignProjectionResiduals(
        "scenario",
        "observed-run",
        (
            CampaignMetricResidual("iae", 1.0, 3.0, iae),
            CampaignMetricResidual("ise", 2.0, 1.0, ise),
        ),
    )


def _residual_tolerances(iae=2.0, ise=2.0):
    return (
        CampaignMetricResidualTolerance("iae", iae),
        CampaignMetricResidualTolerance("ise", ise),
    )


def test_residual_tolerances_cover_pass_fail_boundary_and_traceability():
    result = check_campaign_projection_residual_tolerances(
        _residual_result(2.0, -3.0), _residual_tolerances(2.0, 2.0)
    )

    assert result == CampaignProjectionResidualToleranceResults(
        "scenario",
        "observed-run",
        (
            CampaignMetricResidualToleranceResult("iae", 2.0, 2.0, 2.0, 0.0, True),
            CampaignMetricResidualToleranceResult(
                "ise", -3.0, 3.0, 2.0, -1.0, False
            ),
        ),
    )


def test_residual_tolerances_use_absolute_error_for_both_signs_and_keep_order():
    result = check_campaign_projection_residual_tolerances(
        _residual_result(1.5, -1.5), _residual_tolerances(2.0, 2.0)
    )

    assert tuple(item.metric_name for item in result.metric_results) == ("iae", "ise")
    assert tuple(item.absolute_residual for item in result.metric_results) == (1.5, 1.5)
    assert tuple(item.margin for item in result.metric_results) == (0.5, 0.5)
    assert all(item.passed for item in result.metric_results)


def test_residual_tolerances_handle_zero_and_undefined_residuals():
    residuals = CampaignProjectionResiduals(
        "scenario",
        "run",
        (
            CampaignMetricResidual("iae", 1.0, 1.0, 0.0),
            CampaignMetricResidual("ise", None, 1.0, None),
        ),
    )
    result = check_campaign_projection_residual_tolerances(
        residuals, _residual_tolerances(0.0, 0.0)
    )

    assert result.metric_results[0] == CampaignMetricResidualToleranceResult(
        "iae", 0.0, 0.0, 0.0, 0.0, True
    )
    assert result.metric_results[1] == CampaignMetricResidualToleranceResult(
        "ise", None, None, 0.0, None, False
    )


@pytest.mark.parametrize(
    "tolerances",
    [
        (),
        _residual_tolerances() + (CampaignMetricResidualTolerance("overshoot", 1.0),),
        (
            CampaignMetricResidualTolerance("ise", 2.0),
            CampaignMetricResidualTolerance("iae", 2.0),
        ),
        (
            CampaignMetricResidualTolerance("iae", 2.0),
            CampaignMetricResidualTolerance("iae", 2.0),
        ),
    ],
)
def test_residual_tolerances_reject_missing_extra_misordered_or_duplicate(tolerances):
    with pytest.raises(ValueError):
        check_campaign_projection_residual_tolerances(_residual_result(), tolerances)


@pytest.mark.parametrize("value", [True, "bad", float("inf"), float("nan"), -1.0])
def test_residual_tolerances_reject_invalid_tolerance_values(value):
    with pytest.raises(ValueError, match="numeric|finite|nonnegative"):
        check_campaign_projection_residual_tolerances(
            _residual_result(), _residual_tolerances(value, 2.0)
        )


@pytest.mark.parametrize(
    ("residuals", "match"),
    [
        (object(), "CampaignProjectionResiduals"),
        (CampaignProjectionResiduals(" ", "run", ()), "scenario_name"),
        (CampaignProjectionResiduals("scenario", " ", ()), "observed_run_id"),
        (
            CampaignProjectionResiduals(
                "scenario", "run", (CampaignMetricResidual("iae", 1.0, 2.0, True),)
            ),
            "numeric",
        ),
        (
            CampaignProjectionResiduals(
                "scenario",
                "run",
                (CampaignMetricResidual("iae", 1.0, 2.0, float("inf")),),
            ),
            "finite",
        ),
        (
            CampaignProjectionResiduals(
                "scenario", "run", (CampaignMetricResidual("iae", None, 2.0, 1.0),)
            ),
            "inconsistent optional",
        ),
    ],
)
def test_residual_tolerances_reject_malformed_residual_results(residuals, match):
    tolerances = (CampaignMetricResidualTolerance("iae", 2.0),)
    with pytest.raises((TypeError, ValueError), match=match):
        check_campaign_projection_residual_tolerances(residuals, tolerances)


def test_residual_tolerance_results_are_immutable_deterministic_and_detached():
    tolerances = [*_residual_tolerances()]
    first = check_campaign_projection_residual_tolerances(
        _residual_result(), tolerances
    )
    repeated = check_campaign_projection_residual_tolerances(
        _residual_result(), tolerances
    )

    assert first == repeated
    assert first is not repeated
    tolerances.clear()
    assert first.metric_results[0].metric_name == "iae"
    with pytest.raises(FrozenInstanceError):
        first.observed_run_id = "changed"
    with pytest.raises(FrozenInstanceError):
        first.metric_results[0].passed = False


def _validation_case(name="case", scenario="scenario", run_id="observed", iae=2.0):
    return CampaignProjectionValidationCase(
        name=name,
        scenario_result=_scenario_result(scenario, 1.0, 2.0),
        observed_delta=_observed_delta(run_id, iae, None),
        tolerances=_residual_tolerances(1.0, 1.0),
    )


def test_validation_cases_preserve_order_identities_and_mixed_metric_states():
    cases = (
        _validation_case("first", "nominal", "run-a", iae=2.0),
        _validation_case("second", "stress", "run-b", iae=3.0),
    )

    results = validate_campaign_projection_cases(cases)

    assert tuple(result.name for result in results) == ("first", "second")
    assert tuple(result.scenario_name for result in results) == ("nominal", "stress")
    assert tuple(result.observed_run_id for result in results) == ("run-a", "run-b")
    assert tuple(item.passed for item in results[0].tolerance_results.metric_results) == (
        True,
        False,
    )
    assert results[0].tolerance_results.metric_results[1].residual is None
    assert results[1].tolerance_results.metric_results[0].passed is False


def test_validation_cases_delegate_once_to_existing_analysis_apis(monkeypatch):
    case = _validation_case()
    residuals = CampaignProjectionResiduals("scenario", "observed", ())
    checked = CampaignProjectionResidualToleranceResults("scenario", "observed", ())
    calls = []

    def residual_api(scenario_result, observed_delta):
        calls.append(("residual", scenario_result, observed_delta))
        return residuals

    def tolerance_api(source, tolerances):
        calls.append(("tolerance", source, tolerances))
        return checked

    monkeypatch.setattr(analysis, "campaign_projection_residuals", residual_api)
    monkeypatch.setattr(
        analysis, "check_campaign_projection_residual_tolerances", tolerance_api
    )

    assert validate_campaign_projection_cases((case,)) == (
        CampaignProjectionValidationResult(
            "case", "scenario", "observed", residuals, checked
        ),
    )
    assert calls == [
        ("residual", case.scenario_result, case.observed_delta),
        ("tolerance", residuals, case.tolerances),
    ]


def test_validation_cases_materialize_generator_and_support_empty_input():
    source = (_validation_case(name) for name in ("first", "second"))

    assert tuple(result.name for result in validate_campaign_projection_cases(source)) == (
        "first",
        "second",
    )
    assert validate_campaign_projection_cases(()) == ()


def test_validation_cases_validate_all_metadata_before_evaluation(monkeypatch):
    calls = []
    monkeypatch.setattr(
        analysis,
        "campaign_projection_residuals",
        lambda *args: calls.append(args),
    )

    with pytest.raises(ValueError, match="name must be non-empty"):
        validate_campaign_projection_cases((_validation_case("valid"), _validation_case(" ")))
    assert calls == []


def test_validation_cases_reject_duplicate_names():
    with pytest.raises(ValueError, match="duplicate validation case name"):
        validate_campaign_projection_cases(
            (_validation_case("same"), _validation_case("same", "other"))
        )


@pytest.mark.parametrize(
    ("case", "match"),
    [
        (object(), "CampaignProjectionValidationCase"),
        (
            CampaignProjectionValidationCase("case", object(), _observed_delta(), ()),
            "CampaignProjectionScenarioResult",
        ),
        (
            CampaignProjectionValidationCase(
                "case", _scenario_result("scenario", 1.0, 2.0), object(), ()
            ),
            "CampaignDeltaEntry",
        ),
        (
            CampaignProjectionValidationCase(
                "case",
                _scenario_result("scenario", 1.0, 2.0),
                _observed_delta(),
                [],
            ),
            "tolerances must be a tuple",
        ),
    ],
)
def test_validation_cases_reject_malformed_members(case, match):
    with pytest.raises((TypeError, ValueError), match=match):
        validate_campaign_projection_cases((case,))


def test_validation_cases_propagate_incompatible_tolerance_failure():
    case = CampaignProjectionValidationCase(
        "case",
        _scenario_result("scenario", 1.0, 2.0),
        _observed_delta(),
        (CampaignMetricResidualTolerance("ise", 1.0),),
    )
    with pytest.raises(ValueError, match="coverage"):
        validate_campaign_projection_cases((case,))


def test_validation_case_results_are_immutable_deterministic_and_detached():
    source = [_validation_case()]
    first = validate_campaign_projection_cases(source)
    repeated = validate_campaign_projection_cases(source)

    assert first == repeated
    assert first is not repeated
    source.clear()
    assert first[0].name == "case"
    with pytest.raises(FrozenInstanceError):
        first[0].name = "changed"
    with pytest.raises(TypeError):
        first[0] = first[0]


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


def _envelope(metric_name="iae", minimum=-2.0, maximum=3.0):
    if minimum is None and maximum is None:
        return CampaignMetricProjectionEnvelope(metric_name, None, None, None, None)
    return CampaignMetricProjectionEnvelope(
        metric_name, minimum, "minimum", maximum, "maximum"
    )


def test_envelope_limit_check_clear_pass_and_exact_margins():
    result = check_campaign_projection_envelope_limits(
        (_envelope(),), (CampaignMetricProjectionLimit("iae", -5.0, 7.0),)
    )

    assert result == (
        CampaignMetricProjectionLimitResult(
            metric_name="iae",
            observed_minimum=-2.0,
            observed_maximum=3.0,
            allowable_lower=-5.0,
            allowable_upper=7.0,
            lower_margin=3.0,
            upper_margin=4.0,
            passed=True,
        ),
    )


@pytest.mark.parametrize(
    ("lower", "upper", "lower_margin", "upper_margin"),
    [
        (-1.0, 5.0, -1.0, 2.0),
        (-5.0, 2.0, 3.0, -1.0),
        (-1.0, 2.0, -1.0, -1.0),
    ],
    ids=("lower-failure", "upper-failure", "both-failure"),
)
def test_envelope_limit_check_reports_each_failure_kind(
    lower, upper, lower_margin, upper_margin
):
    result = check_campaign_projection_envelope_limits(
        (_envelope(),), (CampaignMetricProjectionLimit("iae", lower, upper),)
    )[0]

    assert result.lower_margin == lower_margin
    assert result.upper_margin == upper_margin
    assert result.passed is False


def test_envelope_limit_exact_boundaries_pass_with_zero_margins():
    result = check_campaign_projection_envelope_limits(
        (_envelope(),), (CampaignMetricProjectionLimit("iae", -2.0, 3.0),)
    )[0]

    assert result.lower_margin == 0.0
    assert result.upper_margin == 0.0
    assert result.passed is True


def test_multiple_metric_limit_checks_preserve_envelope_order():
    envelopes = (_envelope("ise", -4.0, -1.0), _envelope("iae", 2.0, 5.0))
    limits = (
        CampaignMetricProjectionLimit("ise", -5.0, 0.0),
        CampaignMetricProjectionLimit("iae", 1.0, 6.0),
    )

    results = check_campaign_projection_envelope_limits(envelopes, limits)

    assert tuple(result.metric_name for result in results) == ("ise", "iae")
    assert tuple(result.passed for result in results) == (True, True)


def test_undefined_envelope_has_no_margins_and_does_not_pass():
    result = check_campaign_projection_envelope_limits(
        (_envelope("settling_time", None, None),),
        (CampaignMetricProjectionLimit("settling_time", -1.0, 1.0),),
    )[0]

    assert result.observed_minimum is None
    assert result.observed_maximum is None
    assert result.lower_margin is None
    assert result.upper_margin is None
    assert result.passed is False


def test_empty_envelopes_and_limits_return_an_empty_tuple():
    assert check_campaign_projection_envelope_limits((), ()) == ()


def test_limit_checks_reject_missing_extra_and_misordered_limits():
    envelopes = (_envelope("iae"), _envelope("ise"))
    with pytest.raises(ValueError, match="exact metric coverage"):
        check_campaign_projection_envelope_limits(
            envelopes, (CampaignMetricProjectionLimit("iae", -5.0, 5.0),)
        )
    with pytest.raises(ValueError, match="exact metric coverage"):
        check_campaign_projection_envelope_limits(
            (_envelope("iae"),),
            (
                CampaignMetricProjectionLimit("iae", -5.0, 5.0),
                CampaignMetricProjectionLimit("ise", -5.0, 5.0),
            ),
        )
    with pytest.raises(ValueError, match=r"limits\[0\].*'iae'"):
        check_campaign_projection_envelope_limits(
            envelopes,
            (
                CampaignMetricProjectionLimit("ise", -5.0, 5.0),
                CampaignMetricProjectionLimit("iae", -5.0, 5.0),
            ),
        )


@pytest.mark.parametrize("value", [True, "bad", None, float("inf"), float("nan")])
def test_limit_checks_reject_invalid_or_nonfinite_bounds(value):
    with pytest.raises(ValueError, match=r"allowable_lower.*(numeric|finite)"):
        check_campaign_projection_envelope_limits(
            (_envelope(),), (CampaignMetricProjectionLimit("iae", value, 5.0),)
        )


def test_limit_checks_reject_reversed_bounds_and_nonfinite_margins():
    with pytest.raises(ValueError, match="allowable_lower must not exceed"):
        check_campaign_projection_envelope_limits(
            (_envelope(),), (CampaignMetricProjectionLimit("iae", 5.0, -5.0),)
        )

    huge = _envelope("iae", 1e308, 1e308)
    with pytest.raises(ValueError, match="margins must be finite"):
        check_campaign_projection_envelope_limits(
            (huge,), (CampaignMetricProjectionLimit("iae", -1e308, 1e308),)
        )


def test_limit_checks_reject_inconsistent_envelope_state():
    inconsistent = CampaignMetricProjectionEnvelope(
        "iae", None, "scenario", None, None
    )
    with pytest.raises(ValueError, match="inconsistent undefined state"):
        check_campaign_projection_envelope_limits(
            (inconsistent,), (CampaignMetricProjectionLimit("iae", -1.0, 1.0),)
        )


def test_limit_check_results_are_immutable_deterministic_and_detached():
    envelopes = [_envelope()]
    limits = [CampaignMetricProjectionLimit("iae", -5.0, 5.0)]
    first = check_campaign_projection_envelope_limits(envelopes, limits)
    repeated = check_campaign_projection_envelope_limits(envelopes, limits)

    assert first == repeated
    envelopes.clear()
    limits.clear()
    assert first[0].lower_margin == 3.0
    assert first[0].upper_margin == 2.0
    with pytest.raises(FrozenInstanceError):
        first[0].passed = False
    with pytest.raises(TypeError):
        first[0] = first[0]


def _limit_result(metric_name, lower_margin, upper_margin, passed):
    if lower_margin is None and upper_margin is None:
        return CampaignMetricProjectionLimitResult(
            metric_name, None, None, -5.0, 5.0, None, None, passed
        )
    return CampaignMetricProjectionLimitResult(
        metric_name=metric_name,
        observed_minimum=-5.0 + lower_margin,
        observed_maximum=5.0 - upper_margin,
        allowable_lower=-5.0,
        allowable_upper=5.0,
        lower_margin=lower_margin,
        upper_margin=upper_margin,
        passed=passed,
    )


def test_robustness_verdict_passes_when_all_metrics_pass():
    verdict = campaign_robustness_verdict(
        (_limit_result("iae", 1.0, 2.0, True), _limit_result("ise", 0.0, 3.0, True))
    )

    assert verdict == CampaignRobustnessVerdict(
        overall_passed=True,
        passing_metrics=("iae", "ise"),
        failing_metrics=(),
        undefined_metrics=(),
    )


def test_robustness_verdict_classifies_multiple_failures_in_metric_order():
    verdict = campaign_robustness_verdict(
        (
            _limit_result("iae", 1.0, 1.0, True),
            _limit_result("ise", -1.0, 2.0, False),
            _limit_result("settling_time", 3.0, -2.0, False),
        )
    )

    assert verdict.overall_passed is False
    assert verdict.passing_metrics == ("iae",)
    assert verdict.failing_metrics == ("ise", "settling_time")
    assert verdict.undefined_metrics == ()


def test_robustness_verdict_separates_passing_failing_and_undefined_metrics():
    verdict = campaign_robustness_verdict(
        (
            _limit_result("settling_time", None, None, False),
            _limit_result("iae", 1.0, 1.0, True),
            _limit_result("ise", -1.0, 1.0, False),
            _limit_result("overshoot_percent", None, None, False),
        )
    )

    assert verdict == CampaignRobustnessVerdict(
        overall_passed=False,
        passing_metrics=("iae",),
        failing_metrics=("ise",),
        undefined_metrics=("settling_time", "overshoot_percent"),
    )


def test_empty_limit_results_return_an_explicit_nonpassing_verdict():
    assert campaign_robustness_verdict(()) == CampaignRobustnessVerdict(
        overall_passed=False,
        passing_metrics=(),
        failing_metrics=(),
        undefined_metrics=(),
    )


def test_robustness_verdict_materializes_generator_input():
    results = (
        _limit_result(metric_name, 1.0, 1.0, True)
        for metric_name in ("iae", "ise")
    )

    assert campaign_robustness_verdict(results).passing_metrics == ("iae", "ise")


def test_robustness_verdict_rejects_blank_and_duplicate_metric_names():
    with pytest.raises(ValueError, match="metric_name must be non-empty"):
        campaign_robustness_verdict((_limit_result(" ", 1.0, 1.0, True),))
    with pytest.raises(ValueError, match="duplicate metric name 'iae'"):
        campaign_robustness_verdict(
            (
                _limit_result("iae", 1.0, 1.0, True),
                _limit_result("iae", 2.0, 2.0, True),
            )
        )


def test_robustness_verdict_rejects_inconsistent_undefined_and_pass_states():
    undefined_pass = _limit_result("iae", None, None, True)
    with pytest.raises(ValueError, match="undefined metric cannot pass"):
        campaign_robustness_verdict((undefined_pass,))

    partial = CampaignMetricProjectionLimitResult(
        "iae", None, 1.0, -5.0, 5.0, None, 4.0, False
    )
    with pytest.raises(ValueError, match="inconsistent undefined state"):
        campaign_robustness_verdict((partial,))

    impossible_pass = _limit_result("iae", -1.0, 1.0, True)
    with pytest.raises(ValueError, match="pass state is inconsistent"):
        campaign_robustness_verdict((impossible_pass,))


def test_robustness_verdict_rejects_inconsistent_margins():
    inconsistent = CampaignMetricProjectionLimitResult(
        "iae", -2.0, 3.0, -5.0, 5.0, 99.0, 2.0, True
    )

    with pytest.raises(ValueError, match="margins are inconsistent"):
        campaign_robustness_verdict((inconsistent,))


@pytest.mark.parametrize("value", [True, "bad", float("inf"), float("nan")])
def test_robustness_verdict_rejects_nonnumeric_or_nonfinite_fields(value):
    invalid = CampaignMetricProjectionLimitResult(
        "iae", value, 3.0, -5.0, 5.0, 3.0, 2.0, True
    )

    with pytest.raises(ValueError, match=r"observed_minimum.*(numeric|finite)"):
        campaign_robustness_verdict((invalid,))


def test_robustness_verdict_rejects_malformed_entries():
    with pytest.raises(TypeError, match="CampaignMetricProjectionLimitResult"):
        campaign_robustness_verdict((object(),))


def test_robustness_verdict_is_immutable_deterministic_and_detached():
    source = [
        _limit_result("iae", 1.0, 2.0, True),
        _limit_result("ise", -1.0, 2.0, False),
    ]
    first = campaign_robustness_verdict(source)
    repeated = campaign_robustness_verdict(source)

    assert first == repeated
    source.clear()
    assert first.passing_metrics == ("iae",)
    assert first.failing_metrics == ("ise",)
    with pytest.raises(FrozenInstanceError):
        first.overall_passed = True
