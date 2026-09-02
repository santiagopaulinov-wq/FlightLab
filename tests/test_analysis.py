import json
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from flightlab import analysis
from flightlab.analysis import (
    CampaignComparisonEntry,
    CampaignDeltaEntry,
    CampaignMetricChangeProjection,
    CampaignMetricProjectionEnvelope,
    CampaignMetricProjectionErrorSummary,
    CampaignMetricProjectionErrorSummaryComparison,
    CampaignMetricProjectionLimit,
    CampaignMetricProjectionLimitResult,
    CampaignMetricResidual,
    CampaignMetricResidualTolerance,
    CampaignMetricResidualToleranceResult,
    CampaignMetricValidationResidualEnvelope,
    CampaignParameterChange,
    CampaignProjectionErrorComparisonEnvelopeAssessmentCollectionReport,
    CampaignProjectionErrorComparisonEnvelopeAssessmentCollectionVerdict,
    CampaignProjectionErrorComparisonEnvelopeAssessmentReport,
    CampaignProjectionErrorComparisonEnvelopeLimitVerdict,
    CampaignProjectionErrorMetricEnvelopeLimitVerdict,
    CampaignProjectionErrorMetricFieldIdentity,
    CampaignProjectionErrorNamedAssessmentReport,
    CampaignProjectionErrorSummaryCollection,
    CampaignProjectionErrorSummaryComparisonSetResult,
    CampaignProjectionErrorSummaryDifferenceEnvelope,
    CampaignProjectionErrorSummaryDifferenceLimit,
    CampaignProjectionErrorSummaryDifferenceLimitResult,
    CampaignProjectionResiduals,
    CampaignProjectionResidualToleranceResults,
    CampaignProjectionScenario,
    CampaignProjectionScenarioResult,
    CampaignProjectionValidationCase,
    CampaignProjectionValidationResult,
    CampaignProjectionValidationVerdict,
    CampaignRobustnessVerdict,
    CampaignSensitivityEntry,
    CampaignSensitivityMatrix,
    SensitivityMatrixParameter,
    campaign_metric_deltas,
    campaign_projection_envelopes,
    campaign_projection_error_comparison_envelope_assessment_collection_record,
    campaign_projection_error_comparison_envelope_assessment_collection_report,
    campaign_projection_error_comparison_envelope_assessment_collection_verdict,
    campaign_projection_error_comparison_envelope_assessment_record,
    campaign_projection_error_comparison_envelope_assessment_report,
    campaign_projection_error_comparison_envelope_limit_verdict,
    campaign_projection_error_comparison_envelope_metric_verdicts,
    campaign_projection_error_comparison_envelope_named_assessment_records,
    campaign_projection_error_comparison_envelope_verdict_overview,
    campaign_projection_error_comparison_set_metric_envelopes,
    campaign_projection_error_summaries,
    campaign_projection_residuals,
    campaign_projection_validation_residual_envelopes,
    campaign_projection_validation_verdict,
    campaign_robustness_verdict,
    campaign_secant_sensitivities,
    campaign_sensitivity_matrix,
    check_campaign_projection_envelope_limits,
    check_campaign_projection_error_comparison_envelope_limits,
    check_campaign_projection_residual_tolerances,
    compare_campaign_projection_error_summaries,
    compare_campaign_projection_error_summary_collections,
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
    record = _bundle_record((_run("third", 3.0), _run("first", 1.0, overshoot=False)))

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
    with pytest.raises(
        ValueError, match="baseline run_id 'baseline'.*not in comparison"
    ):
        campaign_metric_deltas((), "baseline")


def test_unknown_baseline_and_duplicate_run_ids_are_rejected():
    with pytest.raises(
        ValueError, match="baseline run_id 'unknown'.*not in comparison"
    ):
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
    matrix = CampaignSensitivityMatrix(("gain",), ("iae",), ("gain-run",), ((2.5,),))

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
    matrix = CampaignSensitivityMatrix(("gain",), ("iae",), ("gain-run",), ((1.0,),))

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
    matrix = CampaignSensitivityMatrix(("gain",), ("iae",), ("gain-run",), ((2.0,),))
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

    monkeypatch.setattr(
        "flightlab.analysis.project_campaign_metric_changes", projection
    )
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

    monkeypatch.setattr(
        "flightlab.analysis.project_campaign_metric_changes", projection
    )
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
    observed = CampaignDeltaEntry("observed", 1.0, (("ise", 1.0), ("iae", 2.0)))
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
        campaign_projection_residuals(
            _scenario_result(" ", 1.0, 2.0), _observed_delta()
        )
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
            CampaignMetricResidualToleranceResult("ise", -3.0, 3.0, 2.0, -1.0, False),
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
    assert tuple(
        item.passed for item in results[0].tolerance_results.metric_results
    ) == (
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

    assert tuple(
        result.name for result in validate_campaign_projection_cases(source)
    ) == (
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
        validate_campaign_projection_cases(
            (_validation_case("valid"), _validation_case(" "))
        )
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


def _validation_result(name, iae, ise):
    case = CampaignProjectionValidationCase(
        name,
        _scenario_result(f"{name}-scenario", 1.0, 2.0),
        _observed_delta(f"{name}-run", iae, ise),
        _residual_tolerances(1.0, 1.0),
    )
    return validate_campaign_projection_cases((case,))[0]


def test_validation_verdict_passes_only_when_all_cases_pass():
    results = (
        _validation_result("first", 2.0, 3.0),
        _validation_result("second", 1.0, 2.0),
    )

    assert campaign_projection_validation_verdict(results) == (
        CampaignProjectionValidationVerdict(True, ("first", "second"), (), ())
    )


def test_validation_verdict_classifies_failures_in_case_order():
    results = (
        _validation_result("pass", 2.0, 3.0),
        _validation_result("fail-a", 3.0, 3.0),
        _validation_result("fail-b", 2.0, 4.0),
    )

    verdict = campaign_projection_validation_verdict(results)

    assert verdict.overall_passed is False
    assert verdict.passing_cases == ("pass",)
    assert verdict.failing_cases == ("fail-a", "fail-b")
    assert verdict.undefined_cases == ()


def test_validation_verdict_uses_undefined_precedence_over_failure():
    undefined_only = _validation_result("undefined", 2.0, None)
    failing_and_undefined = _validation_result("both", 3.0, None)

    verdict = campaign_projection_validation_verdict(
        (undefined_only, failing_and_undefined)
    )

    assert verdict == CampaignProjectionValidationVerdict(
        False, (), (), ("undefined", "both")
    )


def test_validation_verdict_handles_mixed_categories_and_generator_order():
    results = (
        _validation_result("fail", 3.0, 3.0),
        _validation_result("pass", 2.0, 3.0),
        _validation_result("undefined", 2.0, None),
        _validation_result("pass-two", 1.0, 2.0),
    )

    verdict = campaign_projection_validation_verdict(result for result in results)

    assert verdict.passing_cases == ("pass", "pass-two")
    assert verdict.failing_cases == ("fail",)
    assert verdict.undefined_cases == ("undefined",)


def test_validation_verdict_empty_input_is_explicitly_nonpassing():
    assert campaign_projection_validation_verdict(()) == (
        CampaignProjectionValidationVerdict(False, (), (), ())
    )


def test_validation_verdict_rejects_blank_and_duplicate_case_names():
    valid = _validation_result("valid", 2.0, 3.0)
    blank = CampaignProjectionValidationResult(
        " ",
        valid.scenario_name,
        valid.observed_run_id,
        valid.residuals,
        valid.tolerance_results,
    )
    duplicate = CampaignProjectionValidationResult(
        "valid",
        valid.scenario_name,
        valid.observed_run_id,
        valid.residuals,
        valid.tolerance_results,
    )

    with pytest.raises(ValueError, match="name must be non-empty"):
        campaign_projection_validation_verdict((blank,))
    with pytest.raises(ValueError, match="duplicate validation case name"):
        campaign_projection_validation_verdict((valid, duplicate))


def test_validation_verdict_rejects_inconsistent_identity_metadata():
    valid = _validation_result("valid", 2.0, 3.0)
    inconsistent = CampaignProjectionValidationResult(
        valid.name,
        "different",
        valid.observed_run_id,
        valid.residuals,
        valid.tolerance_results,
    )

    with pytest.raises(ValueError, match="inconsistent scenario/run metadata"):
        campaign_projection_validation_verdict((inconsistent,))


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        (
            CampaignMetricResidualToleranceResult("iae", 1.0, 1.0, 1.0, 0.0, False),
            "defined state",
        ),
        (
            CampaignMetricResidualToleranceResult("iae", 1.0, 2.0, 1.0, -1.0, False),
            "defined state",
        ),
        (
            CampaignMetricResidualToleranceResult("iae", 1.0, 1.0, -1.0, -2.0, False),
            "nonnegative",
        ),
        (
            CampaignMetricResidualToleranceResult("iae", 1.0, 1.0, 1.0, 0.0, 1),
            "boolean",
        ),
    ],
)
def test_validation_verdict_rejects_malformed_defined_tolerance_results(
    replacement, match
):
    valid = _validation_result("valid", 2.0, 3.0)
    malformed_checks = CampaignProjectionResidualToleranceResults(
        valid.scenario_name,
        valid.observed_run_id,
        (replacement, valid.tolerance_results.metric_results[1]),
    )
    malformed = CampaignProjectionValidationResult(
        valid.name,
        valid.scenario_name,
        valid.observed_run_id,
        valid.residuals,
        malformed_checks,
    )

    with pytest.raises((TypeError, ValueError), match=match):
        campaign_projection_validation_verdict((malformed,))


def test_validation_verdict_rejects_inconsistent_undefined_state_and_layout():
    undefined = _validation_result("undefined", 2.0, None)
    checks = undefined.tolerance_results.metric_results
    impossible = CampaignProjectionResidualToleranceResults(
        undefined.scenario_name,
        undefined.observed_run_id,
        (
            checks[0],
            CampaignMetricResidualToleranceResult("ise", None, 0.0, 1.0, 1.0, True),
        ),
    )
    malformed = CampaignProjectionValidationResult(
        undefined.name,
        undefined.scenario_name,
        undefined.observed_run_id,
        undefined.residuals,
        impossible,
    )
    with pytest.raises(ValueError, match="inconsistent undefined state"):
        campaign_projection_validation_verdict((malformed,))

    valid = _validation_result("valid", 2.0, 3.0)
    reordered = CampaignProjectionResidualToleranceResults(
        valid.scenario_name,
        valid.observed_run_id,
        tuple(reversed(valid.tolerance_results.metric_results)),
    )
    malformed = CampaignProjectionValidationResult(
        valid.name,
        valid.scenario_name,
        valid.observed_run_id,
        valid.residuals,
        reordered,
    )
    with pytest.raises(ValueError, match="metric layouts"):
        campaign_projection_validation_verdict((malformed,))


def test_validation_verdict_rejects_malformed_result_type():
    with pytest.raises(TypeError, match="CampaignProjectionValidationResult"):
        campaign_projection_validation_verdict((object(),))


def test_validation_verdict_is_immutable_deterministic_and_detached():
    source = [_validation_result("valid", 2.0, 3.0)]
    first = campaign_projection_validation_verdict(source)
    repeated = campaign_projection_validation_verdict(source)

    assert first == repeated
    assert first is not repeated
    source.clear()
    assert first.passing_cases == ("valid",)
    with pytest.raises(FrozenInstanceError):
        first.overall_passed = False


def test_validation_residual_envelope_handles_one_case_and_metric_order():
    result = _validation_result("only", 3.0, 1.0)

    assert campaign_projection_validation_residual_envelopes((result,)) == (
        CampaignMetricValidationResidualEnvelope(
            "iae", 2.0, "only", "only-scenario", "only-run"
        ),
        CampaignMetricValidationResidualEnvelope(
            "ise", 1.0, "only", "only-scenario", "only-run"
        ),
    )


def test_validation_residual_envelopes_select_exact_worst_signed_errors():
    results = (
        _validation_result("small", 2.0, 2.5),
        _validation_result("negative", -2.0, 4.0),
        _validation_result("positive", 3.0, -2.0),
    )

    envelopes = campaign_projection_validation_residual_envelopes(results)

    assert envelopes[0] == CampaignMetricValidationResidualEnvelope(
        "iae", 3.0, "negative", "negative-scenario", "negative-run"
    )
    assert envelopes[1] == CampaignMetricValidationResidualEnvelope(
        "ise", 4.0, "positive", "positive-scenario", "positive-run"
    )


def test_validation_residual_envelope_ties_select_first_case():
    results = (
        _validation_result("first", -2.0, 2.0),
        _validation_result("second", 4.0, 2.0),
    )

    envelope = campaign_projection_validation_residual_envelopes(results)[0]

    assert envelope.maximum_absolute_residual == 3.0
    assert envelope.validation_case_name == "first"


def test_validation_residual_envelopes_handle_mixed_and_all_undefined_metrics():
    results = (
        _validation_result("undefined", None, None),
        _validation_result("defined", 3.0, None),
    )

    assert campaign_projection_validation_residual_envelopes(results) == (
        CampaignMetricValidationResidualEnvelope(
            "iae", 2.0, "defined", "defined-scenario", "defined-run"
        ),
        CampaignMetricValidationResidualEnvelope("ise", None, None, None, None),
    )


def test_validation_residual_envelopes_reject_incompatible_metric_layouts():
    first = _validation_result("first", 2.0, 3.0)
    scenario = CampaignProjectionScenarioResult(
        "reordered-scenario",
        CampaignMetricChangeProjection(("gain",), ("ise", "iae"), (1.0,), (2.0, 1.0)),
    )
    observed = CampaignDeltaEntry("reordered-run", 1.0, (("ise", 3.0), ("iae", 2.0)))
    residuals = campaign_projection_residuals(scenario, observed)
    checked = check_campaign_projection_residual_tolerances(
        residuals,
        (
            CampaignMetricResidualTolerance("ise", 1.0),
            CampaignMetricResidualTolerance("iae", 1.0),
        ),
    )
    reordered = CampaignProjectionValidationResult(
        "reordered",
        "reordered-scenario",
        "reordered-run",
        residuals,
        checked,
    )

    with pytest.raises(ValueError, match="metric layout is incompatible"):
        campaign_projection_validation_residual_envelopes((first, reordered))


def test_validation_residual_envelopes_reject_malformed_and_nonfinite_results():
    with pytest.raises(TypeError, match="CampaignProjectionValidationResult"):
        campaign_projection_validation_residual_envelopes((object(),))

    valid = _validation_result("valid", 2.0, 3.0)
    checks = valid.tolerance_results.metric_results
    malformed_checks = CampaignProjectionResidualToleranceResults(
        valid.scenario_name,
        valid.observed_run_id,
        (
            CampaignMetricResidualToleranceResult(
                "iae", 1.0, float("inf"), 1.0, 0.0, True
            ),
            checks[1],
        ),
    )
    malformed = CampaignProjectionValidationResult(
        valid.name,
        valid.scenario_name,
        valid.observed_run_id,
        valid.residuals,
        malformed_checks,
    )
    with pytest.raises(ValueError, match="finite"):
        campaign_projection_validation_residual_envelopes((malformed,))


def test_validation_residual_envelopes_reject_blank_duplicate_and_bad_metadata():
    valid = _validation_result("valid", 2.0, 3.0)
    blank = CampaignProjectionValidationResult(
        " ",
        valid.scenario_name,
        valid.observed_run_id,
        valid.residuals,
        valid.tolerance_results,
    )
    with pytest.raises(ValueError, match="name must be non-empty"):
        campaign_projection_validation_residual_envelopes((blank,))
    with pytest.raises(ValueError, match="duplicate validation case name"):
        campaign_projection_validation_residual_envelopes((valid, valid))

    inconsistent = CampaignProjectionValidationResult(
        "other",
        "wrong",
        valid.observed_run_id,
        valid.residuals,
        valid.tolerance_results,
    )
    with pytest.raises(ValueError, match="inconsistent scenario/run metadata"):
        campaign_projection_validation_residual_envelopes((inconsistent,))


def test_validation_residual_envelopes_support_empty_and_generator_inputs():
    assert campaign_projection_validation_residual_envelopes(()) == ()
    source = (
        result
        for result in (
            _validation_result("first", 2.0, 3.0),
            _validation_result("second", 3.0, 4.0),
        )
    )

    assert campaign_projection_validation_residual_envelopes(source)[
        0
    ].validation_case_name == ("second")


def test_validation_residual_envelopes_are_immutable_deterministic_and_detached():
    source = [_validation_result("case", 3.0, 1.0)]
    first = campaign_projection_validation_residual_envelopes(source)
    repeated = campaign_projection_validation_residual_envelopes(source)

    assert first == repeated
    assert first is not repeated
    source.clear()
    assert first[0].maximum_absolute_residual == 2.0
    with pytest.raises(FrozenInstanceError):
        first[0].maximum_absolute_residual = 0.0


def test_projection_error_summary_for_one_case_has_exact_counts_and_values():
    result = _validation_result("only", 3.0, 1.0)

    assert campaign_projection_error_summaries((result,)) == (
        CampaignMetricProjectionErrorSummary("iae", 1, 1, 0, 2.0, 2.0, 2.0, 2.0, 2.0),
        CampaignMetricProjectionErrorSummary(
            "ise", 1, 1, 0, -1.0, -1.0, -1.0, 1.0, 1.0
        ),
    )


def test_projection_error_summaries_compute_signed_extrema_and_means():
    results = (
        _validation_result("positive", 4.0, 4.0),
        _validation_result("negative", -1.0, 1.0),
        _validation_result("zero", 1.0, 2.0),
    )

    summaries = campaign_projection_error_summaries(results)

    assert tuple(summary.metric_name for summary in summaries) == ("iae", "ise")
    assert summaries[0] == CampaignMetricProjectionErrorSummary(
        "iae", 3, 3, 0, -2.0, 3.0, 1.0 / 3.0, 5.0 / 3.0, 3.0
    )
    assert summaries[1] == CampaignMetricProjectionErrorSummary(
        "ise", 3, 3, 0, -1.0, 2.0, 1.0 / 3.0, 1.0, 2.0
    )


def test_projection_error_summaries_count_mixed_and_all_undefined_residuals():
    results = (
        _validation_result("undefined", None, None),
        _validation_result("defined", 3.0, None),
    )

    summaries = campaign_projection_error_summaries(results)

    assert summaries[0] == CampaignMetricProjectionErrorSummary(
        "iae", 2, 1, 1, 2.0, 2.0, 2.0, 2.0, 2.0
    )
    assert summaries[1] == CampaignMetricProjectionErrorSummary(
        "ise", 2, 0, 2, None, None, None, None, None
    )


def test_projection_error_maximum_matches_residual_envelope_semantics():
    results = (
        _validation_result("first", -2.0, 2.0),
        _validation_result("second", 4.0, 5.0),
    )

    summaries = campaign_projection_error_summaries(results)
    envelopes = campaign_projection_validation_residual_envelopes(results)

    assert tuple(item.maximum_absolute_residual for item in summaries) == tuple(
        item.maximum_absolute_residual for item in envelopes
    )


def test_projection_error_summaries_reject_incompatible_layouts():
    first = _validation_result("first", 2.0, 3.0)
    scenario = CampaignProjectionScenarioResult(
        "reordered-scenario",
        CampaignMetricChangeProjection(("gain",), ("ise", "iae"), (1.0,), (2.0, 1.0)),
    )
    observed = CampaignDeltaEntry("reordered-run", 1.0, (("ise", 3.0), ("iae", 2.0)))
    residuals = campaign_projection_residuals(scenario, observed)
    checked = check_campaign_projection_residual_tolerances(
        residuals,
        (
            CampaignMetricResidualTolerance("ise", 1.0),
            CampaignMetricResidualTolerance("iae", 1.0),
        ),
    )
    reordered = CampaignProjectionValidationResult(
        "reordered", "reordered-scenario", "reordered-run", residuals, checked
    )

    with pytest.raises(ValueError, match="metric layout is incompatible"):
        campaign_projection_error_summaries((first, reordered))


def _large_residual_validation_result(name):
    residuals = CampaignProjectionResiduals(
        f"{name}-scenario",
        f"{name}-run",
        (CampaignMetricResidual("iae", 0.0, 1e308, 1e308),),
    )
    checked = CampaignProjectionResidualToleranceResults(
        residuals.scenario_name,
        residuals.observed_run_id,
        (CampaignMetricResidualToleranceResult("iae", 1e308, 1e308, 1e308, 0.0, True),),
    )
    return CampaignProjectionValidationResult(
        name,
        residuals.scenario_name,
        residuals.observed_run_id,
        residuals,
        checked,
    )


def test_projection_error_summaries_reject_malformed_and_nonfinite_arithmetic():
    with pytest.raises(TypeError, match="CampaignProjectionValidationResult"):
        campaign_projection_error_summaries((object(),))
    with pytest.raises(ValueError, match="summary arithmetic must be finite"):
        campaign_projection_error_summaries(
            (
                _large_residual_validation_result("first"),
                _large_residual_validation_result("second"),
            )
        )


def test_projection_error_summaries_support_empty_and_generator_inputs():
    assert campaign_projection_error_summaries(()) == ()
    source = (
        result
        for result in (
            _validation_result("first", 2.0, 3.0),
            _validation_result("second", 3.0, 4.0),
        )
    )

    assert campaign_projection_error_summaries(source)[0].validation_case_count == 2


def test_projection_error_summaries_are_immutable_deterministic_and_detached():
    source = [_validation_result("case", 3.0, 1.0)]
    first = campaign_projection_error_summaries(source)
    repeated = campaign_projection_error_summaries(source)

    assert first == repeated
    assert first is not repeated
    source.clear()
    assert first[0].mean_residual == 2.0
    with pytest.raises(FrozenInstanceError):
        first[0].mean_residual = 0.0


def _error_summary(
    metric="iae",
    *,
    cases=2,
    defined=2,
    undefined=0,
    minimum=-1.0,
    maximum=3.0,
    mean=1.0,
    mean_absolute=2.0,
    maximum_absolute=3.0,
):
    return CampaignMetricProjectionErrorSummary(
        metric,
        cases,
        defined,
        undefined,
        minimum,
        maximum,
        mean,
        mean_absolute,
        maximum_absolute,
    )


def test_error_summary_comparison_identical_values_have_zero_differences():
    summary = _error_summary()

    comparison = compare_campaign_projection_error_summaries(
        "left", (summary,), "right", (summary,)
    )[0]

    assert comparison == CampaignMetricProjectionErrorSummaryComparison(
        "left", "right", "iae", summary, summary, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0
    )
    assert comparison.left_summary is not summary
    assert comparison.right_summary is not summary


def test_error_summary_comparison_reports_increases_decreases_and_counts():
    left = _error_summary(cases=3, defined=2, undefined=1)
    right = _error_summary(
        cases=4,
        defined=3,
        undefined=1,
        minimum=-2.0,
        maximum=4.0,
        mean=0.5,
        mean_absolute=2.5,
        maximum_absolute=4.0,
    )

    comparison = compare_campaign_projection_error_summaries(
        "baseline", (left,), "candidate", (right,)
    )[0]

    assert comparison.defined_residual_count_difference == 1
    assert comparison.undefined_residual_count_difference == 0
    assert comparison.minimum_residual_difference == -1.0
    assert comparison.maximum_residual_difference == 1.0
    assert comparison.mean_residual_difference == -0.5
    assert comparison.mean_absolute_residual_difference == 0.5
    assert comparison.maximum_absolute_residual_difference == 1.0


def test_error_summary_comparison_propagates_optional_none_values():
    undefined = _error_summary(
        cases=2,
        defined=0,
        undefined=2,
        minimum=None,
        maximum=None,
        mean=None,
        mean_absolute=None,
        maximum_absolute=None,
    )
    defined = _error_summary()

    comparison = compare_campaign_projection_error_summaries(
        "undefined", (undefined,), "defined", (defined,)
    )[0]

    assert comparison.defined_residual_count_difference == 2
    assert comparison.undefined_residual_count_difference == -2
    assert comparison.minimum_residual_difference is None
    assert comparison.maximum_residual_difference is None
    assert comparison.mean_residual_difference is None
    assert comparison.mean_absolute_residual_difference is None
    assert comparison.maximum_absolute_residual_difference is None


def test_error_summary_comparison_preserves_multiple_metric_order_and_generators():
    left = (
        _error_summary("iae"),
        _error_summary(
            "ise",
            minimum=0.0,
            maximum=2.0,
            mean=1.0,
            mean_absolute=1.0,
            maximum_absolute=2.0,
        ),
    )
    right = tuple(_copy for _copy in left)

    comparisons = compare_campaign_projection_error_summaries(
        "left", (item for item in left), "right", (item for item in right)
    )

    assert tuple(item.metric_name for item in comparisons) == ("iae", "ise")


@pytest.mark.parametrize(
    ("left_name", "right_name"), [("", "right"), ("left", " "), ("same", "same")]
)
def test_error_summary_comparison_rejects_invalid_collection_names(
    left_name, right_name
):
    with pytest.raises(ValueError, match="non-empty|distinct"):
        compare_campaign_projection_error_summaries(left_name, (), right_name, ())


def test_error_summary_comparison_requires_identical_metric_layouts():
    with pytest.raises(ValueError, match="identical metric names and order"):
        compare_campaign_projection_error_summaries(
            "left", (_error_summary("iae"),), "right", (_error_summary("ise"),)
        )
    with pytest.raises(ValueError, match="identical metric names and order"):
        compare_campaign_projection_error_summaries(
            "left", (), "right", (_error_summary(),)
        )


@pytest.mark.parametrize(
    ("summary", "match"),
    [
        (object(), "CampaignMetricProjectionErrorSummary"),
        (_error_summary(" "), "metric_name"),
        (_error_summary(cases=True), "nonnegative integer"),
        (_error_summary(cases=3), "inconsistent residual counts"),
        (_error_summary(minimum=float("inf")), "finite"),
        (_error_summary(mean=4.0), "mean residual"),
        (_error_summary(mean_absolute=4.0), "absolute residual summaries"),
    ],
)
def test_error_summary_comparison_rejects_malformed_summaries(summary, match):
    with pytest.raises((TypeError, ValueError), match=match):
        compare_campaign_projection_error_summaries(
            "left", (summary,), "right", (_error_summary(),)
        )


def test_error_summary_comparison_rejects_duplicate_metric_names():
    duplicates = (_error_summary(), _error_summary())
    with pytest.raises(ValueError, match="duplicate metric"):
        compare_campaign_projection_error_summaries(
            "left", duplicates, "right", duplicates
        )


def test_error_summary_comparison_rejects_nonfinite_difference():
    left = _error_summary(
        minimum=-1e308,
        maximum=1e308,
        mean=-1e308,
        mean_absolute=1e308,
        maximum_absolute=1e308,
    )
    right = _error_summary(
        minimum=-1e308,
        maximum=1e308,
        mean=1e308,
        mean_absolute=1e308,
        maximum_absolute=1e308,
    )
    with pytest.raises(ValueError, match="difference must be finite"):
        compare_campaign_projection_error_summaries("left", (left,), "right", (right,))


def test_error_summary_comparison_empty_collections_return_empty():
    assert compare_campaign_projection_error_summaries("left", (), "right", ()) == ()


def test_error_summary_comparisons_are_immutable_deterministic_and_detached():
    source = [_error_summary()]
    first = compare_campaign_projection_error_summaries("left", source, "right", source)
    repeated = compare_campaign_projection_error_summaries(
        "left", source, "right", source
    )

    assert first == repeated
    assert first is not repeated
    source.clear()
    assert first[0].metric_name == "iae"
    with pytest.raises(FrozenInstanceError):
        first[0].mean_residual_difference = 1.0


def test_error_summary_collection_comparison_handles_one_collection():
    baseline = _error_summary(mean=1.0)
    candidate = _error_summary(mean=2.0)

    result = compare_campaign_projection_error_summary_collections(
        "baseline",
        (baseline,),
        (CampaignProjectionErrorSummaryCollection("candidate", (candidate,)),),
    )

    assert len(result) == 1
    assert result[0].baseline_collection_name == "baseline"
    assert result[0].comparison_collection_name == "candidate"
    assert result[0].comparisons[0].mean_residual_difference == 1.0


def test_error_summary_collection_comparison_preserves_collection_and_metric_order():
    baseline = (_error_summary("iae"), _error_summary("ise"))
    collections = (
        CampaignProjectionErrorSummaryCollection("second", baseline),
        CampaignProjectionErrorSummaryCollection("first", baseline),
    )

    result = compare_campaign_projection_error_summary_collections(
        "baseline", baseline, collections
    )

    assert tuple(item.comparison_collection_name for item in result) == (
        "second",
        "first",
    )
    assert tuple(item.metric_name for item in result[0].comparisons) == ("iae", "ise")


def test_error_summary_collection_comparison_delegates_once_and_reuses_baseline(
    monkeypatch,
):
    baseline = (_error_summary(),)
    calls = []
    original = analysis.compare_campaign_projection_error_summaries

    def recording_comparison(left_name, left, right_name, right):
        calls.append((left_name, left, right_name, right))
        return original(left_name, left, right_name, right)

    monkeypatch.setattr(
        analysis, "compare_campaign_projection_error_summaries", recording_comparison
    )
    collections = (
        CampaignProjectionErrorSummaryCollection("a", baseline),
        CampaignProjectionErrorSummaryCollection("b", baseline),
    )

    compare_campaign_projection_error_summary_collections(
        "baseline", baseline, collections
    )

    assert len(calls) == 2
    assert calls[0][1] is calls[1][1]
    assert tuple(call[2] for call in calls) == ("a", "b")


@pytest.mark.parametrize("baseline_name", ["", " ", None])
def test_error_summary_collection_comparison_rejects_blank_baseline_name(baseline_name):
    with pytest.raises(ValueError, match="baseline_collection_name must be non-empty"):
        compare_campaign_projection_error_summary_collections(baseline_name, (), ())


@pytest.mark.parametrize(
    "collections, match",
    [
        (
            (CampaignProjectionErrorSummaryCollection(" ", ()),),
            "name must be non-empty",
        ),
        (
            (
                CampaignProjectionErrorSummaryCollection("same", ()),
                CampaignProjectionErrorSummaryCollection("same", ()),
            ),
            "duplicate comparison collection name",
        ),
        (
            (CampaignProjectionErrorSummaryCollection("baseline", ()),),
            "distinct from baseline",
        ),
    ],
)
def test_error_summary_collection_comparison_rejects_invalid_names(collections, match):
    with pytest.raises(ValueError, match=match):
        compare_campaign_projection_error_summary_collections(
            "baseline", (), collections
        )


def test_error_summary_collection_comparison_validates_all_entries_before_delegation(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        analysis,
        "compare_campaign_projection_error_summaries",
        lambda *args: calls.append(args),
    )
    collections = (
        CampaignProjectionErrorSummaryCollection("valid", (_error_summary(),)),
        object(),
    )

    with pytest.raises(TypeError, match="CampaignProjectionErrorSummaryCollection"):
        compare_campaign_projection_error_summary_collections(
            "baseline", (_error_summary(),), collections
        )
    assert calls == []


def test_error_summary_collection_comparison_rejects_malformed_and_incompatible_summaries():
    with pytest.raises(TypeError, match="CampaignMetricProjectionErrorSummary"):
        compare_campaign_projection_error_summary_collections(
            "baseline",
            (_error_summary(),),
            (CampaignProjectionErrorSummaryCollection("bad", (object(),)),),
        )
    with pytest.raises(ValueError, match="identical metric names and order"):
        compare_campaign_projection_error_summary_collections(
            "baseline",
            (_error_summary("iae"),),
            (
                CampaignProjectionErrorSummaryCollection(
                    "other", (_error_summary("ise"),)
                ),
            ),
        )


def test_error_summary_collection_comparison_propagates_delegated_failure(monkeypatch):
    def fail(*args):
        raise RuntimeError("delegated failure")

    monkeypatch.setattr(analysis, "compare_campaign_projection_error_summaries", fail)
    with pytest.raises(RuntimeError, match="delegated failure"):
        compare_campaign_projection_error_summary_collections(
            "baseline",
            (_error_summary(),),
            (CampaignProjectionErrorSummaryCollection("other", (_error_summary(),)),),
        )


def test_error_summary_collection_comparison_supports_generator_and_empty_inputs():
    assert (
        compare_campaign_projection_error_summary_collections(
            "baseline", (_error_summary(),), ()
        )
        == ()
    )
    collections = (
        item
        for item in (
            CampaignProjectionErrorSummaryCollection("a", (_error_summary(),)),
            CampaignProjectionErrorSummaryCollection("b", (_error_summary(mean=2.0),)),
        )
    )
    result = compare_campaign_projection_error_summary_collections(
        "baseline", (_error_summary(),), collections
    )
    assert tuple(item.comparison_collection_name for item in result) == ("a", "b")
    assert result[0].comparisons != result[1].comparisons


def test_error_summary_collection_comparison_is_immutable_deterministic_and_detached():
    source = [_error_summary()]
    collections = [CampaignProjectionErrorSummaryCollection("other", tuple(source))]
    first = compare_campaign_projection_error_summary_collections(
        "baseline", source, collections
    )
    repeated = compare_campaign_projection_error_summary_collections(
        "baseline", source, collections
    )

    assert first == repeated
    assert first is not repeated
    assert isinstance(first[0], CampaignProjectionErrorSummaryComparisonSetResult)
    source.clear()
    collections.clear()
    assert first[0].comparisons[0].metric_name == "iae"
    with pytest.raises(FrozenInstanceError):
        first[0].comparison_collection_name = "changed"


def _comparison_set_result(name, summaries, *, baseline=None):
    if baseline is None:
        baseline = tuple(_error_summary(summary.metric_name) for summary in summaries)
    return compare_campaign_projection_error_summary_collections(
        "baseline",
        baseline,
        (CampaignProjectionErrorSummaryCollection(name, tuple(summaries)),),
    )[0]


def test_comparison_set_envelopes_one_collection_has_same_minimum_and_maximum():
    result = _comparison_set_result("only", (_error_summary(mean=2.0),))

    envelopes = campaign_projection_error_comparison_set_metric_envelopes((result,))

    mean = next(item for item in envelopes if item.difference_field == "mean_residual_difference")
    assert mean == CampaignProjectionErrorSummaryDifferenceEnvelope(
        "iae", "mean_residual_difference", 1.0, "only", 1.0, "only"
    )
    assert len(envelopes) == 7


def test_comparison_set_envelopes_find_signed_extrema_and_preserve_metric_order():
    baseline = (_error_summary("iae"), _error_summary("ise"))
    low = _comparison_set_result(
        "low",
        (_error_summary("iae", mean=-1.0), _error_summary("ise", mean=2.0)),
        baseline=baseline,
    )
    high = _comparison_set_result(
        "high",
        (_error_summary("iae", mean=2.0), _error_summary("ise", mean=0.0)),
        baseline=baseline,
    )

    envelopes = campaign_projection_error_comparison_set_metric_envelopes((low, high))
    means = tuple(
        item for item in envelopes if item.difference_field == "mean_residual_difference"
    )

    assert tuple(item.metric_name for item in means) == ("iae", "ise")
    assert (means[0].minimum_difference, means[0].maximum_difference) == (-2.0, 1.0)
    assert (means[0].minimum_comparison_collection_name, means[0].maximum_comparison_collection_name) == ("low", "high")
    assert (means[1].minimum_difference, means[1].maximum_difference) == (-1.0, 1.0)


def test_comparison_set_envelope_ties_select_first_collection():
    first = _comparison_set_result("first", (_error_summary(mean=2.0),))
    second = _comparison_set_result("second", (_error_summary(mean=2.0),))

    mean = next(
        item
        for item in campaign_projection_error_comparison_set_metric_envelopes((first, second))
        if item.difference_field == "mean_residual_difference"
    )
    assert mean.minimum_comparison_collection_name == "first"
    assert mean.maximum_comparison_collection_name == "first"


def test_comparison_set_envelopes_ignore_none_and_preserve_all_undefined():
    undefined_summary = _error_summary(
        defined=0, undefined=2, minimum=None, maximum=None, mean=None,
        mean_absolute=None, maximum_absolute=None,
    )
    undefined = _comparison_set_result("undefined", (undefined_summary,))
    defined = _comparison_set_result("defined", (_error_summary(mean=2.0),))

    mixed = campaign_projection_error_comparison_set_metric_envelopes((undefined, defined))
    mixed_mean = next(item for item in mixed if item.difference_field == "mean_residual_difference")
    assert (mixed_mean.minimum_difference, mixed_mean.maximum_difference) == (1.0, 1.0)
    assert mixed_mean.minimum_comparison_collection_name == "defined"

    all_undefined = campaign_projection_error_comparison_set_metric_envelopes((undefined,))
    undefined_mean = next(item for item in all_undefined if item.difference_field == "mean_residual_difference")
    assert (
        undefined_mean.minimum_difference,
        undefined_mean.minimum_comparison_collection_name,
        undefined_mean.maximum_difference,
        undefined_mean.maximum_comparison_collection_name,
    ) == (None, None, None, None)


def test_comparison_set_envelopes_reject_incompatible_layouts_and_identities():
    iae = _comparison_set_result("iae-set", (_error_summary("iae"),))
    ise = _comparison_set_result("ise-set", (_error_summary("ise"),))
    with pytest.raises(ValueError, match="incompatible metric layouts"):
        campaign_projection_error_comparison_set_metric_envelopes((iae, ise))

    inconsistent = CampaignProjectionErrorSummaryComparisonSetResult(
        "other-baseline", "other", iae.comparisons
    )
    with pytest.raises(ValueError, match="inconsistent collection identities|baseline"):
        campaign_projection_error_comparison_set_metric_envelopes((inconsistent,))


def test_comparison_set_envelopes_reject_duplicate_names_and_malformed_entries():
    result = _comparison_set_result("same", (_error_summary(),))
    with pytest.raises(ValueError, match="duplicate comparison collection name"):
        campaign_projection_error_comparison_set_metric_envelopes((result, result))
    with pytest.raises(TypeError, match="ComparisonSetResult"):
        campaign_projection_error_comparison_set_metric_envelopes((object(),))


@pytest.mark.parametrize("value, match", [(True, "numeric"), (float("inf"), "finite")])
def test_comparison_set_envelopes_reject_invalid_defined_differences(value, match):
    valid = _comparison_set_result("candidate", (_error_summary(),))
    comparison = valid.comparisons[0]
    malformed_comparison = CampaignMetricProjectionErrorSummaryComparison(
        comparison.left_collection_name,
        comparison.right_collection_name,
        comparison.metric_name,
        comparison.left_summary,
        comparison.right_summary,
        value,
        comparison.undefined_residual_count_difference,
        comparison.minimum_residual_difference,
        comparison.maximum_residual_difference,
        comparison.mean_residual_difference,
        comparison.mean_absolute_residual_difference,
        comparison.maximum_absolute_residual_difference,
    )
    malformed = CampaignProjectionErrorSummaryComparisonSetResult(
        "baseline", "candidate", (malformed_comparison,)
    )
    with pytest.raises(ValueError, match=match):
        campaign_projection_error_comparison_set_metric_envelopes((malformed,))


def test_comparison_set_envelopes_reject_invalid_optional_state():
    undefined_summary = _error_summary(
        defined=0, undefined=2, minimum=None, maximum=None, mean=None,
        mean_absolute=None, maximum_absolute=None,
    )
    valid = _comparison_set_result("candidate", (undefined_summary,))
    comparison = valid.comparisons[0]
    malformed_comparison = CampaignMetricProjectionErrorSummaryComparison(
        comparison.left_collection_name, comparison.right_collection_name,
        comparison.metric_name, comparison.left_summary, comparison.right_summary,
        comparison.defined_residual_count_difference,
        comparison.undefined_residual_count_difference, 0.0,
        comparison.maximum_residual_difference, comparison.mean_residual_difference,
        comparison.mean_absolute_residual_difference,
        comparison.maximum_absolute_residual_difference,
    )
    malformed = CampaignProjectionErrorSummaryComparisonSetResult(
        "baseline", "candidate", (malformed_comparison,)
    )
    with pytest.raises(ValueError, match="invalid optional state"):
        campaign_projection_error_comparison_set_metric_envelopes((malformed,))


def test_comparison_set_envelopes_support_empty_generator_and_detached_results():
    assert campaign_projection_error_comparison_set_metric_envelopes(()) == ()
    source = [_comparison_set_result("candidate", (_error_summary(mean=2.0),))]
    first = campaign_projection_error_comparison_set_metric_envelopes(
        item for item in source
    )
    repeated = campaign_projection_error_comparison_set_metric_envelopes(source)

    assert first == repeated
    assert first is not repeated
    source.clear()
    assert first[0].metric_name == "iae"
    with pytest.raises(FrozenInstanceError):
        first[0].minimum_difference = 0.0


_DIFFERENCE_FIELDS = (
    "defined_residual_count_difference",
    "undefined_residual_count_difference",
    "minimum_residual_difference",
    "maximum_residual_difference",
    "mean_residual_difference",
    "mean_absolute_residual_difference",
    "maximum_absolute_residual_difference",
)


def _difference_envelopes(metric="iae", minimum=-2.0, maximum=3.0):
    if minimum is None:
        return tuple(
            CampaignProjectionErrorSummaryDifferenceEnvelope(
                metric, field, None, None, None, None
            )
            for field in _DIFFERENCE_FIELDS
        )
    return tuple(
        CampaignProjectionErrorSummaryDifferenceEnvelope(
            metric, field, minimum, "minimum", maximum, "maximum"
        )
        for field in _DIFFERENCE_FIELDS
    )


def _difference_limits(envelopes, lower=-5.0, upper=7.0):
    return tuple(
        CampaignProjectionErrorSummaryDifferenceLimit(
            envelope.metric_name, envelope.difference_field, lower, upper
        )
        for envelope in envelopes
    )


def test_comparison_envelope_limit_clear_pass_and_exact_margins():
    envelopes = _difference_envelopes()
    results = check_campaign_projection_error_comparison_envelope_limits(
        envelopes, _difference_limits(envelopes)
    )

    assert results[0] == CampaignProjectionErrorSummaryDifferenceLimitResult(
        "iae", _DIFFERENCE_FIELDS[0], -2.0, 3.0, -5.0, 7.0, 3.0, 4.0, True
    )


@pytest.mark.parametrize(
    "lower, upper, expected_margins",
    [
        (-1.0, 5.0, (-1.0, 2.0)),
        (-5.0, 2.0, (3.0, -1.0)),
        (-1.0, 2.0, (-1.0, -1.0)),
    ],
    ids=("lower", "upper", "both"),
)
def test_comparison_envelope_limits_report_bound_failures(
    lower, upper, expected_margins
):
    envelopes = _difference_envelopes()
    result = check_campaign_projection_error_comparison_envelope_limits(
        envelopes, _difference_limits(envelopes, lower, upper)
    )[0]
    assert (result.lower_margin, result.upper_margin) == expected_margins
    assert result.passed is False


@pytest.mark.parametrize(
    "minimum, maximum, lower, upper",
    [(-2.0, 3.0, -2.0, 3.0), (0.0, 0.0, 0.0, 0.0), (-4.0, -1.0, -4.0, -1.0)],
)
def test_comparison_envelope_limits_exact_and_zero_width_boundaries_pass(
    minimum, maximum, lower, upper
):
    envelopes = _difference_envelopes(minimum=minimum, maximum=maximum)
    result = check_campaign_projection_error_comparison_envelope_limits(
        envelopes, _difference_limits(envelopes, lower, upper)
    )[0]
    assert (result.lower_margin, result.upper_margin, result.passed) == (0.0, 0.0, True)


def test_comparison_envelope_limits_preserve_multiple_metric_and_field_order():
    envelopes = _difference_envelopes("ise") + _difference_envelopes("iae", 1.0, 2.0)
    results = check_campaign_projection_error_comparison_envelope_limits(
        envelopes, _difference_limits(envelopes)
    )
    assert tuple((result.metric_name, result.difference_field) for result in results) == tuple(
        (envelope.metric_name, envelope.difference_field) for envelope in envelopes
    )


def test_comparison_envelope_limit_undefined_has_no_margins_and_does_not_pass():
    envelopes = _difference_envelopes(minimum=None, maximum=None)
    result = check_campaign_projection_error_comparison_envelope_limits(
        envelopes, _difference_limits(envelopes)
    )[0]
    assert result.observed_minimum_difference is None
    assert result.observed_maximum_difference is None
    assert result.lower_margin is None
    assert result.upper_margin is None
    assert result.passed is False


def test_comparison_envelope_limits_reject_missing_extra_and_misordered_limits():
    envelopes = _difference_envelopes()
    limits = _difference_limits(envelopes)
    with pytest.raises(ValueError, match="exact metric and field coverage"):
        check_campaign_projection_error_comparison_envelope_limits(envelopes, limits[:-1])
    with pytest.raises(ValueError, match="exact metric and field coverage"):
        check_campaign_projection_error_comparison_envelope_limits(
            envelopes, limits + (limits[-1],)
        )
    with pytest.raises(ValueError, match=r"limits\[0\] identity"):
        check_campaign_projection_error_comparison_envelope_limits(
            envelopes, (limits[1], limits[0], *limits[2:])
        )


def test_comparison_envelope_limits_reject_malformed_envelope_layouts():
    envelopes = _difference_envelopes()
    with pytest.raises(ValueError, match="complete difference-field layouts"):
        check_campaign_projection_error_comparison_envelope_limits(
            envelopes[:-1], _difference_limits(envelopes[:-1])
        )
    malformed = (
        CampaignProjectionErrorSummaryDifferenceEnvelope(
            " ", _DIFFERENCE_FIELDS[0], -1.0, "a", 1.0, "b"
        ),
        *envelopes[1:],
    )
    with pytest.raises(ValueError, match="metric_name must be non-empty"):
        check_campaign_projection_error_comparison_envelope_limits(
            malformed, _difference_limits(malformed)
        )


@pytest.mark.parametrize("value, match", [(True, "numeric"), ("bad", "numeric"), (float("inf"), "finite"), (float("nan"), "finite")])
def test_comparison_envelope_limits_reject_invalid_bounds(value, match):
    envelopes = _difference_envelopes()
    limits = list(_difference_limits(envelopes))
    limits[0] = CampaignProjectionErrorSummaryDifferenceLimit(
        "iae", _DIFFERENCE_FIELDS[0], value, 7.0
    )
    with pytest.raises(ValueError, match=match):
        check_campaign_projection_error_comparison_envelope_limits(envelopes, limits)


def test_comparison_envelope_limits_reject_reversed_bounds_and_nonfinite_margins():
    envelopes = _difference_envelopes()
    with pytest.raises(ValueError, match="must not exceed"):
        check_campaign_projection_error_comparison_envelope_limits(
            envelopes, _difference_limits(envelopes, 2.0, -2.0)
        )
    huge = _difference_envelopes(minimum=1e308, maximum=1e308)
    with pytest.raises(ValueError, match="margins must be finite"):
        check_campaign_projection_error_comparison_envelope_limits(
            huge, _difference_limits(huge, -1e308, 1e308)
        )


def test_comparison_envelope_limits_reject_malformed_stored_values_and_states():
    envelopes = list(_difference_envelopes())
    envelopes[0] = CampaignProjectionErrorSummaryDifferenceEnvelope(
        "iae", _DIFFERENCE_FIELDS[0], float("inf"), "a", 1.0, "b"
    )
    with pytest.raises(ValueError, match="finite"):
        check_campaign_projection_error_comparison_envelope_limits(
            envelopes, _difference_limits(envelopes)
        )
    envelopes[0] = CampaignProjectionErrorSummaryDifferenceEnvelope(
        "iae", _DIFFERENCE_FIELDS[0], None, "a", None, None
    )
    with pytest.raises(ValueError, match="inconsistent optional state"):
        check_campaign_projection_error_comparison_envelope_limits(
            envelopes, _difference_limits(envelopes)
        )
    envelopes[0] = CampaignProjectionErrorSummaryDifferenceEnvelope(
        "iae", _DIFFERENCE_FIELDS[0], 2.0, "a", 1.0, "b"
    )
    with pytest.raises(ValueError, match="must not exceed"):
        check_campaign_projection_error_comparison_envelope_limits(
            envelopes, _difference_limits(envelopes)
        )


def test_comparison_envelope_limits_support_empty_generators_and_immutability():
    assert check_campaign_projection_error_comparison_envelope_limits((), ()) == ()
    envelopes = _difference_envelopes()
    limits = _difference_limits(envelopes)
    first = check_campaign_projection_error_comparison_envelope_limits(
        (item for item in envelopes), (item for item in limits)
    )
    repeated = check_campaign_projection_error_comparison_envelope_limits(
        envelopes, limits
    )
    assert first == repeated
    assert first is not repeated
    with pytest.raises(FrozenInstanceError):
        first[0].passed = False


def _difference_limit_result_block(metric="iae", states=None):
    if states is None:
        states = ("pass",) * len(_DIFFERENCE_FIELDS)
    envelopes = []
    for field, state in zip(_DIFFERENCE_FIELDS, states):
        if state == "undefined":
            envelopes.append(
                CampaignProjectionErrorSummaryDifferenceEnvelope(
                    metric, field, None, None, None, None
                )
            )
        else:
            minimum, maximum = ((0.0, 0.0) if state == "pass" else (2.0, 3.0))
            envelopes.append(
                CampaignProjectionErrorSummaryDifferenceEnvelope(
                    metric, field, minimum, "first", maximum, "first"
                )
            )
    limits = _difference_limits(envelopes, -1.0, 1.0)
    return check_campaign_projection_error_comparison_envelope_limits(envelopes, limits)


def _copy_difference_limit_result(result, **changes):
    values = {
        "metric_name": result.metric_name,
        "difference_field": result.difference_field,
        "observed_minimum_difference": result.observed_minimum_difference,
        "observed_maximum_difference": result.observed_maximum_difference,
        "allowable_minimum_difference": result.allowable_minimum_difference,
        "allowable_maximum_difference": result.allowable_maximum_difference,
        "lower_margin": result.lower_margin,
        "upper_margin": result.upper_margin,
        "passed": result.passed,
    }
    values.update(changes)
    return CampaignProjectionErrorSummaryDifferenceLimitResult(**values)


def test_comparison_envelope_limit_verdict_passes_when_all_results_pass():
    verdict = campaign_projection_error_comparison_envelope_limit_verdict(
        _difference_limit_result_block()
    )
    assert verdict.overall_passed is True
    assert tuple(identity.difference_field for identity in verdict.passing_identities) == (
        _DIFFERENCE_FIELDS
    )
    assert verdict.failing_identities == ()
    assert verdict.undefined_identities == ()


def test_comparison_envelope_limit_verdict_classifies_one_and_multiple_failures():
    one_failure = campaign_projection_error_comparison_envelope_limit_verdict(
        _difference_limit_result_block(states=("pass", "fail", *("pass",) * 5))
    )
    assert one_failure.failing_identities == (
        CampaignProjectionErrorMetricFieldIdentity("iae", _DIFFERENCE_FIELDS[1]),
    )
    assert one_failure.overall_passed is False

    multiple = campaign_projection_error_comparison_envelope_limit_verdict(
        _difference_limit_result_block(states=("fail", "pass", "fail", *("pass",) * 4))
    )
    assert tuple(item.difference_field for item in multiple.failing_identities) == (
        _DIFFERENCE_FIELDS[0],
        _DIFFERENCE_FIELDS[2],
    )


def test_comparison_envelope_limit_verdict_classifies_undefined_and_mixed_in_order():
    results = _difference_limit_result_block(
        states=("undefined", "pass", "fail", "undefined", "pass", "fail", "pass")
    )
    verdict = campaign_projection_error_comparison_envelope_limit_verdict(
        item for item in results
    )
    assert tuple(item.difference_field for item in verdict.passing_identities) == (
        _DIFFERENCE_FIELDS[1], _DIFFERENCE_FIELDS[4], _DIFFERENCE_FIELDS[6]
    )
    assert tuple(item.difference_field for item in verdict.failing_identities) == (
        _DIFFERENCE_FIELDS[2], _DIFFERENCE_FIELDS[5]
    )
    assert tuple(item.difference_field for item in verdict.undefined_identities) == (
        _DIFFERENCE_FIELDS[0], _DIFFERENCE_FIELDS[3]
    )
    assert verdict.overall_passed is False


def test_comparison_envelope_limit_verdict_preserves_metric_order_in_categories():
    results = _difference_limit_result_block("ise") + _difference_limit_result_block("iae")
    verdict = campaign_projection_error_comparison_envelope_limit_verdict(results)
    assert tuple(item.metric_name for item in verdict.passing_identities) == (
        *("ise",) * 7,
        *("iae",) * 7,
    )


def test_comparison_envelope_limit_verdict_rejects_duplicate_and_bad_ordering():
    results = _difference_limit_result_block()
    with pytest.raises(ValueError, match="duplicate limit-result metric"):
        campaign_projection_error_comparison_envelope_limit_verdict(results + results)
    reordered = (results[1], results[0], *results[2:])
    with pytest.raises(ValueError, match="difference_field must be"):
        campaign_projection_error_comparison_envelope_limit_verdict(reordered)


def test_comparison_envelope_limit_verdict_rejects_optional_and_pass_states():
    results = list(_difference_limit_result_block())
    results[0] = _copy_difference_limit_result(results[0], lower_margin=None)
    with pytest.raises(ValueError, match="inconsistent optional state"):
        campaign_projection_error_comparison_envelope_limit_verdict(results)

    undefined = list(_difference_limit_result_block(states=("undefined",) * 7))
    undefined[0] = _copy_difference_limit_result(undefined[0], passed=True)
    with pytest.raises(ValueError, match="undefined result cannot pass"):
        campaign_projection_error_comparison_envelope_limit_verdict(undefined)

    results = list(_difference_limit_result_block())
    results[0] = _copy_difference_limit_result(results[0], passed=False)
    with pytest.raises(ValueError, match="pass state is inconsistent"):
        campaign_projection_error_comparison_envelope_limit_verdict(results)


@pytest.mark.parametrize("value, match", [(float("inf"), "finite"), (True, "numeric")])
def test_comparison_envelope_limit_verdict_rejects_nonfinite_or_boolean_values(value, match):
    results = list(_difference_limit_result_block())
    results[0] = _copy_difference_limit_result(
        results[0], observed_minimum_difference=value
    )
    with pytest.raises(ValueError, match=match):
        campaign_projection_error_comparison_envelope_limit_verdict(results)


def test_comparison_envelope_limit_verdict_rejects_impossible_margins_and_entries():
    results = list(_difference_limit_result_block())
    results[0] = _copy_difference_limit_result(results[0], lower_margin=2.0)
    with pytest.raises(ValueError, match="margins are inconsistent"):
        campaign_projection_error_comparison_envelope_limit_verdict(results)
    with pytest.raises(TypeError, match="DifferenceLimitResult"):
        campaign_projection_error_comparison_envelope_limit_verdict((object(),) * 7)


def test_comparison_envelope_limit_verdict_empty_is_nonpassing_and_immutable():
    assert campaign_projection_error_comparison_envelope_limit_verdict(()) == (
        CampaignProjectionErrorComparisonEnvelopeLimitVerdict(False, (), (), ())
    )
    source = list(_difference_limit_result_block())
    first = campaign_projection_error_comparison_envelope_limit_verdict(source)
    repeated = campaign_projection_error_comparison_envelope_limit_verdict(source)
    assert first == repeated
    assert first is not repeated
    source.clear()
    assert len(first.passing_identities) == 7
    with pytest.raises(FrozenInstanceError):
        first.overall_passed = False


def test_comparison_envelope_metric_verdict_all_fields_passing():
    verdicts = campaign_projection_error_comparison_envelope_metric_verdicts(
        _difference_limit_result_block()
    )
    assert verdicts == (
        CampaignProjectionErrorMetricEnvelopeLimitVerdict(
            "iae",
            True,
            tuple(
                CampaignProjectionErrorMetricFieldIdentity("iae", field)
                for field in _DIFFERENCE_FIELDS
            ),
            (),
            (),
        ),
    )


def test_comparison_envelope_metric_verdict_one_and_multiple_failures():
    one = campaign_projection_error_comparison_envelope_metric_verdicts(
        _difference_limit_result_block(states=("pass", "fail", *("pass",) * 5))
    )[0]
    assert tuple(item.difference_field for item in one.failing_identities) == (
        _DIFFERENCE_FIELDS[1],
    )
    assert one.overall_passed is False

    multiple = campaign_projection_error_comparison_envelope_metric_verdicts(
        _difference_limit_result_block(states=("fail", "pass", "fail", *("pass",) * 4))
    )[0]
    assert tuple(item.difference_field for item in multiple.failing_identities) == (
        _DIFFERENCE_FIELDS[0], _DIFFERENCE_FIELDS[2]
    )


def test_comparison_envelope_metric_verdict_mixed_categories_preserve_field_order():
    verdict = campaign_projection_error_comparison_envelope_metric_verdicts(
        _difference_limit_result_block(
            states=("undefined", "pass", "fail", "undefined", "pass", "fail", "pass")
        )
    )[0]
    assert tuple(item.difference_field for item in verdict.passing_identities) == (
        _DIFFERENCE_FIELDS[1], _DIFFERENCE_FIELDS[4], _DIFFERENCE_FIELDS[6]
    )
    assert tuple(item.difference_field for item in verdict.failing_identities) == (
        _DIFFERENCE_FIELDS[2], _DIFFERENCE_FIELDS[5]
    )
    assert tuple(item.difference_field for item in verdict.undefined_identities) == (
        _DIFFERENCE_FIELDS[0], _DIFFERENCE_FIELDS[3]
    )
    assert verdict.overall_passed is False


def test_comparison_envelope_metric_verdicts_preserve_exact_metric_order():
    results = _difference_limit_result_block("ise") + _difference_limit_result_block("iae")
    verdicts = campaign_projection_error_comparison_envelope_metric_verdicts(results)
    assert tuple(verdict.metric_name for verdict in verdicts) == ("ise", "iae")
    assert all(verdict.overall_passed for verdict in verdicts)


def test_comparison_envelope_metric_verdicts_reject_incomplete_duplicate_and_reordered_layouts():
    results = _difference_limit_result_block()
    with pytest.raises(ValueError, match="complete difference-field layouts"):
        campaign_projection_error_comparison_envelope_metric_verdicts(results[:-1])
    with pytest.raises(ValueError, match="duplicate limit-result metric"):
        campaign_projection_error_comparison_envelope_metric_verdicts(results + results)
    with pytest.raises(ValueError, match="difference_field must be"):
        campaign_projection_error_comparison_envelope_metric_verdicts(
            (results[1], results[0], *results[2:])
        )


def test_comparison_envelope_metric_verdicts_reject_impossible_states_and_values():
    results = list(_difference_limit_result_block())
    results[0] = _copy_difference_limit_result(results[0], passed=False)
    with pytest.raises(ValueError, match="pass state is inconsistent"):
        campaign_projection_error_comparison_envelope_metric_verdicts(results)

    results = list(_difference_limit_result_block())
    results[0] = _copy_difference_limit_result(results[0], lower_margin=float("inf"))
    with pytest.raises(ValueError, match="finite"):
        campaign_projection_error_comparison_envelope_metric_verdicts(results)

    results = list(_difference_limit_result_block())
    results[0] = _copy_difference_limit_result(results[0], upper_margin=None)
    with pytest.raises(ValueError, match="inconsistent optional state"):
        campaign_projection_error_comparison_envelope_metric_verdicts(results)


def test_comparison_envelope_metric_verdicts_empty_generator_and_immutable():
    assert campaign_projection_error_comparison_envelope_metric_verdicts(()) == ()
    source = list(_difference_limit_result_block())
    first = campaign_projection_error_comparison_envelope_metric_verdicts(
        item for item in source
    )
    repeated = campaign_projection_error_comparison_envelope_metric_verdicts(source)
    assert first == repeated
    assert first is not repeated
    source.clear()
    assert len(first[0].passing_identities) == 7
    with pytest.raises(FrozenInstanceError):
        first[0].overall_passed = False


def test_comparison_envelope_assessment_report_retains_passing_views_and_results():
    source = _difference_limit_result_block()
    report = campaign_projection_error_comparison_envelope_assessment_report(source)
    assert report.overall_verdict.overall_passed is True
    assert report.metric_verdicts[0].overall_passed is True
    assert report.limit_results == source
    assert report.limit_results[0] is not source[0]


def test_comparison_envelope_assessment_report_handles_failing_and_undefined_states():
    failing = campaign_projection_error_comparison_envelope_assessment_report(
        _difference_limit_result_block(states=("fail", *("pass",) * 6))
    )
    assert failing.overall_verdict.overall_passed is False
    assert failing.metric_verdicts[0].failing_identities[0].difference_field == (
        _DIFFERENCE_FIELDS[0]
    )

    undefined = campaign_projection_error_comparison_envelope_assessment_report(
        _difference_limit_result_block(states=("undefined", *("pass",) * 6))
    )
    assert undefined.metric_verdicts[0].undefined_identities[0].difference_field == (
        _DIFFERENCE_FIELDS[0]
    )


def test_comparison_envelope_assessment_report_preserves_mixed_metric_and_field_order():
    results = _difference_limit_result_block(
        "ise", ("pass", "fail", *("pass",) * 5)
    ) + _difference_limit_result_block(
        "iae", ("undefined", "pass", "fail", *("pass",) * 4)
    )
    report = campaign_projection_error_comparison_envelope_assessment_report(results)
    assert tuple(verdict.metric_name for verdict in report.metric_verdicts) == (
        "ise", "iae"
    )
    assert tuple(
        (result.metric_name, result.difference_field) for result in report.limit_results
    ) == tuple((result.metric_name, result.difference_field) for result in results)
    assert report.overall_verdict.failing_identities == (
        report.metric_verdicts[0].failing_identities
        + report.metric_verdicts[1].failing_identities
    )


def test_comparison_envelope_assessment_report_rejects_malformed_and_incomplete_input():
    results = _difference_limit_result_block()
    with pytest.raises(ValueError, match="complete difference-field layouts"):
        campaign_projection_error_comparison_envelope_assessment_report(results[:-1])
    with pytest.raises(ValueError, match="duplicate limit-result metric"):
        campaign_projection_error_comparison_envelope_assessment_report(results + results)
    malformed = list(results)
    malformed[0] = _copy_difference_limit_result(malformed[0], passed=False)
    with pytest.raises(ValueError, match="pass state is inconsistent"):
        campaign_projection_error_comparison_envelope_assessment_report(malformed)


def test_comparison_envelope_assessment_report_rejects_inconsistent_delegated_views(
    monkeypatch,
):
    original = analysis.campaign_projection_error_comparison_envelope_limit_verdict

    def inconsistent(results):
        verdict = original(results)
        return CampaignProjectionErrorComparisonEnvelopeLimitVerdict(
            verdict.overall_passed,
            verdict.passing_identities[:-1],
            verdict.failing_identities,
            verdict.undefined_identities,
        )

    monkeypatch.setattr(
        analysis,
        "campaign_projection_error_comparison_envelope_limit_verdict",
        inconsistent,
    )
    with pytest.raises(ValueError, match="identities do not match"):
        campaign_projection_error_comparison_envelope_assessment_report(
            _difference_limit_result_block()
        )


def test_comparison_envelope_assessment_report_delegates_each_verdict_api_once(
    monkeypatch,
):
    overall = analysis.campaign_projection_error_comparison_envelope_limit_verdict
    metric = analysis.campaign_projection_error_comparison_envelope_metric_verdicts
    calls = []

    def overall_wrapper(results):
        calls.append(("overall", results))
        return overall(results)

    def metric_wrapper(results):
        calls.append(("metric", results))
        return metric(results)

    monkeypatch.setattr(
        analysis,
        "campaign_projection_error_comparison_envelope_limit_verdict",
        overall_wrapper,
    )
    monkeypatch.setattr(
        analysis,
        "campaign_projection_error_comparison_envelope_metric_verdicts",
        metric_wrapper,
    )
    source = _difference_limit_result_block()
    campaign_projection_error_comparison_envelope_assessment_report(
        item for item in source
    )
    assert tuple(name for name, _ in calls) == ("overall", "metric")
    assert calls[0][1] is calls[1][1]


def test_comparison_envelope_assessment_report_empty_immutable_and_deterministic():
    empty = campaign_projection_error_comparison_envelope_assessment_report(())
    assert empty == CampaignProjectionErrorComparisonEnvelopeAssessmentReport(
        (), CampaignProjectionErrorComparisonEnvelopeLimitVerdict(False, (), (), ()), ()
    )
    source = list(_difference_limit_result_block())
    first = campaign_projection_error_comparison_envelope_assessment_report(source)
    repeated = campaign_projection_error_comparison_envelope_assessment_report(source)
    assert first == repeated
    assert first is not repeated
    source.clear()
    assert len(first.limit_results) == 7
    with pytest.raises(FrozenInstanceError):
        first.overall_verdict = empty.overall_verdict


def test_comparison_envelope_assessment_record_serializes_passing_report():
    report = campaign_projection_error_comparison_envelope_assessment_report(
        _difference_limit_result_block()
    )
    record = campaign_projection_error_comparison_envelope_assessment_record(report)
    assert record["limit_results"][0] == {
        "metric_name": "iae",
        "difference_field": _DIFFERENCE_FIELDS[0],
        "observed_minimum_difference": 0.0,
        "observed_maximum_difference": 0.0,
        "allowable_minimum_difference": -1.0,
        "allowable_maximum_difference": 1.0,
        "lower_margin": 1.0,
        "upper_margin": 1.0,
        "passed": True,
    }
    assert record["overall_verdict"]["overall_passed"] is True
    assert record["metric_verdicts"][0]["overall_passed"] is True


def test_comparison_envelope_assessment_record_preserves_failures_and_none():
    report = campaign_projection_error_comparison_envelope_assessment_report(
        _difference_limit_result_block(
            states=("fail", "undefined", *("pass",) * 5)
        )
    )
    record = campaign_projection_error_comparison_envelope_assessment_record(report)
    assert record["limit_results"][0]["passed"] is False
    assert record["limit_results"][1]["observed_minimum_difference"] is None
    assert record["limit_results"][1]["lower_margin"] is None
    assert record["overall_verdict"]["failing_identities"] == [
        {"metric_name": "iae", "difference_field": _DIFFERENCE_FIELDS[0]}
    ]
    assert record["metric_verdicts"][0]["undefined_difference_fields"] == [
        _DIFFERENCE_FIELDS[1]
    ]


def test_comparison_envelope_assessment_record_preserves_multiple_metric_order():
    results = _difference_limit_result_block("ise") + _difference_limit_result_block("iae")
    report = campaign_projection_error_comparison_envelope_assessment_report(results)
    record = campaign_projection_error_comparison_envelope_assessment_record(report)
    assert [item["metric_name"] for item in record["metric_verdicts"]] == [
        "ise", "iae"
    ]
    assert [
        (item["metric_name"], item["difference_field"])
        for item in record["limit_results"]
    ] == [(item.metric_name, item.difference_field) for item in results]


def test_comparison_envelope_assessment_record_empty_schema_is_explicit():
    report = campaign_projection_error_comparison_envelope_assessment_report(())
    assert campaign_projection_error_comparison_envelope_assessment_record(report) == {
        "limit_results": [],
        "overall_verdict": {
            "overall_passed": False,
            "passing_identities": [],
            "failing_identities": [],
            "undefined_identities": [],
        },
        "metric_verdicts": [],
    }


def test_comparison_envelope_assessment_record_is_plain_and_json_compatible():
    report = campaign_projection_error_comparison_envelope_assessment_report(
        _difference_limit_result_block()
    )
    record = campaign_projection_error_comparison_envelope_assessment_record(report)

    def assert_plain(value):
        assert type(value) in (dict, list, str, bool, int, float, type(None))
        if isinstance(value, dict):
            assert all(type(key) is str for key in value)
            for child in value.values():
                assert_plain(child)
        elif isinstance(value, list):
            for child in value:
                assert_plain(child)

    assert_plain(record)
    json.dumps(record, allow_nan=False)


def test_comparison_envelope_assessment_record_rejects_malformed_report():
    report = campaign_projection_error_comparison_envelope_assessment_report(
        _difference_limit_result_block()
    )
    malformed_verdict = CampaignProjectionErrorComparisonEnvelopeLimitVerdict(
        True,
        report.overall_verdict.passing_identities[:-1],
        (),
        (report.overall_verdict.passing_identities[-1],),
    )
    malformed = CampaignProjectionErrorComparisonEnvelopeAssessmentReport(
        report.limit_results, malformed_verdict, report.metric_verdicts
    )
    with pytest.raises(ValueError, match="classifications disagree|classification disagrees"):
        campaign_projection_error_comparison_envelope_assessment_record(malformed)
    with pytest.raises(TypeError, match="AssessmentReport"):
        campaign_projection_error_comparison_envelope_assessment_record(object())


def test_comparison_envelope_assessment_record_is_detached_and_deterministic():
    report = campaign_projection_error_comparison_envelope_assessment_report(
        _difference_limit_result_block()
    )
    first = campaign_projection_error_comparison_envelope_assessment_record(report)
    repeated = campaign_projection_error_comparison_envelope_assessment_record(report)
    assert first == repeated
    assert first is not repeated
    first["limit_results"][0]["metric_name"] = "changed"
    first["overall_verdict"]["passing_identities"].clear()
    assert repeated["limit_results"][0]["metric_name"] == "iae"
    assert len(repeated["overall_verdict"]["passing_identities"]) == 7
    assert campaign_projection_error_comparison_envelope_assessment_record(report) == repeated


def _named_assessment_report(name="assessment", states=None, metric="iae"):
    return CampaignProjectionErrorNamedAssessmentReport(
        name,
        campaign_projection_error_comparison_envelope_assessment_report(
            _difference_limit_result_block(metric, states)
        ),
    )


def test_named_assessment_records_convert_one_report_and_preserve_name():
    records = campaign_projection_error_comparison_envelope_named_assessment_records(
        (_named_assessment_report("baseline"),)
    )
    assert len(records) == 1
    assert records[0]["name"] == "baseline"
    assert records[0]["report"]["overall_verdict"]["overall_passed"] is True


def test_named_assessment_records_preserve_multiple_caller_order():
    entries = (
        _named_assessment_report("second", metric="ise"),
        _named_assessment_report("first", states=("fail", *("pass",) * 6)),
    )
    records = campaign_projection_error_comparison_envelope_named_assessment_records(
        entries
    )
    assert [record["name"] for record in records] == ["second", "first"]
    assert records[0]["report"]["metric_verdicts"][0]["metric_name"] == "ise"
    assert records[1]["report"]["overall_verdict"]["overall_passed"] is False


def test_named_assessment_records_delegate_once_per_entry(monkeypatch):
    original = analysis.campaign_projection_error_comparison_envelope_assessment_record
    calls = []

    def recording(report):
        calls.append(report)
        return original(report)

    monkeypatch.setattr(
        analysis,
        "campaign_projection_error_comparison_envelope_assessment_record",
        recording,
    )
    entries = (_named_assessment_report("a"), _named_assessment_report("b"))
    campaign_projection_error_comparison_envelope_named_assessment_records(entries)
    assert calls == [entries[0].report, entries[1].report]


@pytest.mark.parametrize("name", ["", " ", None])
def test_named_assessment_records_reject_blank_names(name):
    with pytest.raises(ValueError, match="name must be non-empty"):
        campaign_projection_error_comparison_envelope_named_assessment_records(
            (CampaignProjectionErrorNamedAssessmentReport(name, _named_assessment_report().report),)
        )


def test_named_assessment_records_reject_duplicates_and_malformed_members():
    entry = _named_assessment_report("same")
    with pytest.raises(ValueError, match="duplicate assessment report name"):
        campaign_projection_error_comparison_envelope_named_assessment_records(
            (entry, entry)
        )
    with pytest.raises(TypeError, match="NamedAssessmentReport"):
        campaign_projection_error_comparison_envelope_named_assessment_records(
            (object(),)
        )
    with pytest.raises(TypeError, match=r"entries\[0\].report"):
        campaign_projection_error_comparison_envelope_named_assessment_records(
            (CampaignProjectionErrorNamedAssessmentReport("bad", object()),)
        )


def test_named_assessment_records_validate_complete_collection_before_conversion(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        analysis,
        "campaign_projection_error_comparison_envelope_assessment_record",
        lambda report: calls.append(report),
    )
    with pytest.raises(TypeError, match="NamedAssessmentReport"):
        campaign_projection_error_comparison_envelope_named_assessment_records(
            (_named_assessment_report("valid"), object())
        )
    assert calls == []


def test_named_assessment_records_propagate_delegated_failure(monkeypatch):
    def fail(report):
        raise RuntimeError("delegated conversion failed")

    monkeypatch.setattr(
        analysis,
        "campaign_projection_error_comparison_envelope_assessment_record",
        fail,
    )
    with pytest.raises(RuntimeError, match="delegated conversion failed"):
        campaign_projection_error_comparison_envelope_named_assessment_records(
            (_named_assessment_report(),)
        )


def test_named_assessment_records_support_empty_and_generator_inputs():
    assert campaign_projection_error_comparison_envelope_named_assessment_records(()) == []
    entries = (_named_assessment_report(name) for name in ("a", "b"))
    records = campaign_projection_error_comparison_envelope_named_assessment_records(
        entries
    )
    assert [record["name"] for record in records] == ["a", "b"]


def test_named_assessment_records_are_plain_detached_and_deterministic():
    entries = [_named_assessment_report("a"), _named_assessment_report("b")]
    first = campaign_projection_error_comparison_envelope_named_assessment_records(
        entries
    )
    repeated = campaign_projection_error_comparison_envelope_named_assessment_records(
        entries
    )
    assert first == repeated
    json.dumps(first, allow_nan=False)
    first[0]["name"] = "changed"
    first[0]["report"]["limit_results"].clear()
    entries.clear()
    assert repeated[0]["name"] == "a"
    assert len(repeated[0]["report"]["limit_results"]) == 7


def test_named_assessment_verdict_overview_one_report_preserves_stored_passes():
    overview = campaign_projection_error_comparison_envelope_verdict_overview(
        (_named_assessment_report("passing"),)
    )
    assert overview == [
        {
            "name": "passing",
            "overall_passed": True,
            "metrics": [{"metric": "iae", "passed": True}],
        }
    ]


def test_named_assessment_verdict_overview_preserves_report_and_metric_order():
    first_report = campaign_projection_error_comparison_envelope_assessment_report(
        _difference_limit_result_block("ise")
        + _difference_limit_result_block("iae", states=("fail", *("pass",) * 6))
    )
    entries = (
        CampaignProjectionErrorNamedAssessmentReport("first", first_report),
        _named_assessment_report("second", metric="settling_time"),
    )
    overview = campaign_projection_error_comparison_envelope_verdict_overview(entries)
    assert [item["name"] for item in overview] == ["first", "second"]
    assert overview[0]["overall_passed"] is False
    assert overview[0]["metrics"] == [
        {"metric": "ise", "passed": True},
        {"metric": "iae", "passed": False},
    ]
    assert overview[1]["metrics"] == [
        {"metric": "settling_time", "passed": True}
    ]


def test_named_assessment_verdict_overview_rejects_blank_and_duplicate_names():
    report = _named_assessment_report().report
    with pytest.raises(ValueError, match="name must be non-empty"):
        campaign_projection_error_comparison_envelope_verdict_overview(
            (CampaignProjectionErrorNamedAssessmentReport(" ", report),)
        )
    same = CampaignProjectionErrorNamedAssessmentReport("same", report)
    with pytest.raises(ValueError, match="duplicate assessment report name"):
        campaign_projection_error_comparison_envelope_verdict_overview((same, same))


def test_named_assessment_verdict_overview_rejects_malformed_members_and_metrics():
    with pytest.raises(TypeError, match="NamedAssessmentReport"):
        campaign_projection_error_comparison_envelope_verdict_overview((object(),))
    with pytest.raises(TypeError, match=r"entries\[0\].report"):
        campaign_projection_error_comparison_envelope_verdict_overview(
            (CampaignProjectionErrorNamedAssessmentReport("bad", object()),)
        )

    report = _named_assessment_report().report
    verdict = report.metric_verdicts[0]
    blank = CampaignProjectionErrorMetricEnvelopeLimitVerdict(
        " ", verdict.overall_passed, verdict.passing_identities,
        verdict.failing_identities, verdict.undefined_identities,
    )
    malformed = CampaignProjectionErrorComparisonEnvelopeAssessmentReport(
        report.limit_results, report.overall_verdict, (blank,)
    )
    with pytest.raises(ValueError, match="metric_name must be non-empty"):
        campaign_projection_error_comparison_envelope_verdict_overview(
            (CampaignProjectionErrorNamedAssessmentReport("blank", malformed),)
        )


def test_named_assessment_verdict_overview_rejects_duplicate_and_impossible_verdicts():
    report = _named_assessment_report().report
    duplicate = CampaignProjectionErrorComparisonEnvelopeAssessmentReport(
        report.limit_results,
        report.overall_verdict,
        report.metric_verdicts + report.metric_verdicts,
    )
    with pytest.raises(ValueError, match="per-metric verdict order"):
        campaign_projection_error_comparison_envelope_verdict_overview(
            (CampaignProjectionErrorNamedAssessmentReport("duplicate", duplicate),)
        )

    verdict = report.metric_verdicts[0]
    impossible = CampaignProjectionErrorMetricEnvelopeLimitVerdict(
        verdict.metric_name, False, verdict.passing_identities,
        verdict.failing_identities, verdict.undefined_identities,
    )
    malformed = CampaignProjectionErrorComparisonEnvelopeAssessmentReport(
        report.limit_results, report.overall_verdict, (impossible,)
    )
    with pytest.raises(ValueError, match="pass state is inconsistent"):
        campaign_projection_error_comparison_envelope_verdict_overview(
            (CampaignProjectionErrorNamedAssessmentReport("impossible", malformed),)
        )


def test_named_assessment_verdict_overview_empty_generator_plain_and_detached():
    assert campaign_projection_error_comparison_envelope_verdict_overview(()) == []
    entries = [_named_assessment_report("a"), _named_assessment_report("b")]
    first = campaign_projection_error_comparison_envelope_verdict_overview(
        entry for entry in entries
    )
    repeated = campaign_projection_error_comparison_envelope_verdict_overview(entries)
    assert first == repeated
    assert first is not repeated
    json.dumps(first, allow_nan=False)
    first[0]["name"] = "changed"
    first[0]["metrics"].clear()
    entries.clear()
    assert repeated[0] == {
        "name": "a",
        "overall_passed": True,
        "metrics": [{"metric": "iae", "passed": True}],
    }


def test_named_assessment_collection_verdict_all_reports_passing():
    verdict = campaign_projection_error_comparison_envelope_assessment_collection_verdict(
        (_named_assessment_report("first"), _named_assessment_report("second"))
    )
    assert verdict == (
        CampaignProjectionErrorComparisonEnvelopeAssessmentCollectionVerdict(
            True, ("first", "second"), (), ()
        )
    )


def test_named_assessment_collection_verdict_one_and_multiple_failures():
    one = campaign_projection_error_comparison_envelope_assessment_collection_verdict(
        (
            _named_assessment_report("pass"),
            _named_assessment_report("fail", ("fail", *("pass",) * 6)),
        )
    )
    assert one.overall_passed is False
    assert one.passing_report_names == ("pass",)
    assert one.failing_report_names == ("fail",)

    multiple = campaign_projection_error_comparison_envelope_assessment_collection_verdict(
        (
            _named_assessment_report("fail-a", ("fail", *("pass",) * 6)),
            _named_assessment_report("pass"),
            _named_assessment_report("fail-b", ("pass", "fail", *("pass",) * 5)),
        )
    )
    assert multiple.failing_report_names == ("fail-a", "fail-b")


def test_named_assessment_collection_verdict_undefined_precedes_failure():
    verdict = campaign_projection_error_comparison_envelope_assessment_collection_verdict(
        (
            _named_assessment_report(
                "undefined-and-failing",
                ("undefined", "fail", *("pass",) * 5),
            ),
        )
    )
    assert verdict.passing_report_names == ()
    assert verdict.failing_report_names == ()
    assert verdict.undefined_report_names == ("undefined-and-failing",)


def test_named_assessment_collection_verdict_mixed_categories_preserve_order():
    entries = (
        _named_assessment_report("undefined-a", ("undefined", *("pass",) * 6)),
        _named_assessment_report("fail-a", ("fail", *("pass",) * 6)),
        _named_assessment_report("pass-a"),
        _named_assessment_report("undefined-b", ("pass", "undefined", *("pass",) * 5)),
        _named_assessment_report("fail-b", ("pass", "fail", *("pass",) * 5)),
        _named_assessment_report("pass-b"),
    )
    verdict = campaign_projection_error_comparison_envelope_assessment_collection_verdict(
        entries
    )
    assert verdict.passing_report_names == ("pass-a", "pass-b")
    assert verdict.failing_report_names == ("fail-a", "fail-b")
    assert verdict.undefined_report_names == ("undefined-a", "undefined-b")


def test_named_assessment_collection_verdict_rejects_names_and_malformed_members():
    report = _named_assessment_report().report
    with pytest.raises(ValueError, match="name must be non-empty"):
        campaign_projection_error_comparison_envelope_assessment_collection_verdict(
            (CampaignProjectionErrorNamedAssessmentReport(" ", report),)
        )
    same = CampaignProjectionErrorNamedAssessmentReport("same", report)
    with pytest.raises(ValueError, match="duplicate assessment report name"):
        campaign_projection_error_comparison_envelope_assessment_collection_verdict(
            (same, same)
        )
    with pytest.raises(TypeError, match="NamedAssessmentReport"):
        campaign_projection_error_comparison_envelope_assessment_collection_verdict(
            (object(),)
        )
    with pytest.raises(TypeError, match=r"entries\[0\].report"):
        campaign_projection_error_comparison_envelope_assessment_collection_verdict(
            (CampaignProjectionErrorNamedAssessmentReport("bad", object()),)
        )


def test_named_assessment_collection_verdict_rejects_inconsistent_stored_verdicts():
    report = _named_assessment_report().report
    impossible_overall = CampaignProjectionErrorComparisonEnvelopeLimitVerdict(
        False,
        report.overall_verdict.passing_identities,
        report.overall_verdict.failing_identities,
        report.overall_verdict.undefined_identities,
    )
    malformed = CampaignProjectionErrorComparisonEnvelopeAssessmentReport(
        report.limit_results, impossible_overall, report.metric_verdicts
    )
    with pytest.raises(ValueError, match="overall verdict pass state"):
        campaign_projection_error_comparison_envelope_assessment_collection_verdict(
            (CampaignProjectionErrorNamedAssessmentReport("bad", malformed),)
        )

    undefined = _named_assessment_report(
        "undefined", ("undefined", *("pass",) * 6)
    ).report
    inconsistent = CampaignProjectionErrorComparisonEnvelopeAssessmentReport(
        undefined.limit_results,
        CampaignProjectionErrorComparisonEnvelopeLimitVerdict(
            False,
            undefined.overall_verdict.passing_identities,
            undefined.overall_verdict.failing_identities,
            (),
        ),
        undefined.metric_verdicts,
    )
    with pytest.raises(ValueError, match="identities do not match|classifications disagree"):
        campaign_projection_error_comparison_envelope_assessment_collection_verdict(
            (CampaignProjectionErrorNamedAssessmentReport("bad", inconsistent),)
        )


def test_named_assessment_collection_verdict_empty_generator_immutable_deterministic():
    assert campaign_projection_error_comparison_envelope_assessment_collection_verdict(
        ()
    ) == CampaignProjectionErrorComparisonEnvelopeAssessmentCollectionVerdict(
        False, (), (), ()
    )
    source = [_named_assessment_report("first"), _named_assessment_report("second")]
    first = campaign_projection_error_comparison_envelope_assessment_collection_verdict(
        entry for entry in source
    )
    repeated = campaign_projection_error_comparison_envelope_assessment_collection_verdict(
        source
    )
    assert first == repeated
    assert first is not repeated
    source.clear()
    assert first.passing_report_names == ("first", "second")
    with pytest.raises(FrozenInstanceError):
        first.overall_passed = False


def test_named_assessment_collection_report_retains_passing_reports_and_verdict():
    entries = (_named_assessment_report("first"), _named_assessment_report("second"))
    report = campaign_projection_error_comparison_envelope_assessment_collection_report(
        entries
    )
    assert report.collection_verdict == (
        CampaignProjectionErrorComparisonEnvelopeAssessmentCollectionVerdict(
            True, ("first", "second"), (), ()
        )
    )
    assert tuple(entry.name for entry in report.named_reports) == ("first", "second")
    assert report.named_reports[0] is not entries[0]
    assert report.named_reports[0].report is not entries[0].report


def test_named_assessment_collection_report_handles_failing_and_undefined_reports():
    entries = (
        _named_assessment_report("pass"),
        _named_assessment_report("fail", ("fail", *("pass",) * 6)),
        _named_assessment_report("undefined", ("undefined", *("pass",) * 6)),
    )
    report = campaign_projection_error_comparison_envelope_assessment_collection_report(
        entries
    )
    assert report.collection_verdict.overall_passed is False
    assert report.collection_verdict.passing_report_names == ("pass",)
    assert report.collection_verdict.failing_report_names == ("fail",)
    assert report.collection_verdict.undefined_report_names == ("undefined",)


def test_named_assessment_collection_report_preserves_nested_and_report_order():
    multi_metric = campaign_projection_error_comparison_envelope_assessment_report(
        _difference_limit_result_block("ise") + _difference_limit_result_block("iae")
    )
    entries = (
        CampaignProjectionErrorNamedAssessmentReport("multi", multi_metric),
        _named_assessment_report("single", metric="settling_time"),
    )
    report = campaign_projection_error_comparison_envelope_assessment_collection_report(
        entry for entry in entries
    )
    assert tuple(entry.name for entry in report.named_reports) == ("multi", "single")
    assert tuple(
        verdict.metric_name for verdict in report.named_reports[0].report.metric_verdicts
    ) == ("ise", "iae")
    assert tuple(
        result.difference_field
        for result in report.named_reports[0].report.limit_results[:7]
    ) == _DIFFERENCE_FIELDS


def test_named_assessment_collection_report_delegates_collection_verdict_once(
    monkeypatch,
):
    original = (
        analysis.campaign_projection_error_comparison_envelope_assessment_collection_verdict
    )
    calls = []

    def recording(entries):
        calls.append(entries)
        return original(entries)

    monkeypatch.setattr(
        analysis,
        "campaign_projection_error_comparison_envelope_assessment_collection_verdict",
        recording,
    )
    entries = (_named_assessment_report("first"), _named_assessment_report("second"))
    campaign_projection_error_comparison_envelope_assessment_collection_report(entries)
    assert len(calls) == 1
    assert calls[0] == entries


def test_named_assessment_collection_report_rejects_inconsistent_delegated_verdict(
    monkeypatch,
):
    def inconsistent(entries):
        return CampaignProjectionErrorComparisonEnvelopeAssessmentCollectionVerdict(
            True, ("second", "first"), (), ()
        )

    monkeypatch.setattr(
        analysis,
        "campaign_projection_error_comparison_envelope_assessment_collection_verdict",
        inconsistent,
    )
    with pytest.raises(ValueError, match="report-name order is inconsistent"):
        campaign_projection_error_comparison_envelope_assessment_collection_report(
            (_named_assessment_report("first"), _named_assessment_report("second"))
        )


def test_named_assessment_collection_report_rejects_bad_names_and_nested_reports():
    report = _named_assessment_report().report
    with pytest.raises(ValueError, match="name must be non-empty"):
        campaign_projection_error_comparison_envelope_assessment_collection_report(
            (CampaignProjectionErrorNamedAssessmentReport(" ", report),)
        )
    same = CampaignProjectionErrorNamedAssessmentReport("same", report)
    with pytest.raises(ValueError, match="duplicate assessment report name"):
        campaign_projection_error_comparison_envelope_assessment_collection_report(
            (same, same)
        )
    with pytest.raises(TypeError, match="NamedAssessmentReport"):
        campaign_projection_error_comparison_envelope_assessment_collection_report(
            (object(),)
        )

    malformed_overall = CampaignProjectionErrorComparisonEnvelopeLimitVerdict(
        False,
        report.overall_verdict.passing_identities,
        report.overall_verdict.failing_identities,
        report.overall_verdict.undefined_identities,
    )
    malformed = CampaignProjectionErrorComparisonEnvelopeAssessmentReport(
        report.limit_results, malformed_overall, report.metric_verdicts
    )
    with pytest.raises(ValueError, match="overall verdict pass state"):
        campaign_projection_error_comparison_envelope_assessment_collection_report(
            (CampaignProjectionErrorNamedAssessmentReport("bad", malformed),)
        )


def test_named_assessment_collection_report_empty_is_explicit_and_nonpassing():
    report = campaign_projection_error_comparison_envelope_assessment_collection_report(
        ()
    )
    assert report == CampaignProjectionErrorComparisonEnvelopeAssessmentCollectionReport(
        (),
        CampaignProjectionErrorComparisonEnvelopeAssessmentCollectionVerdict(
            False, (), (), ()
        ),
    )


def test_named_assessment_collection_report_is_immutable_deterministic_and_detached():
    source = [_named_assessment_report("first"), _named_assessment_report("second")]
    first = campaign_projection_error_comparison_envelope_assessment_collection_report(
        source
    )
    repeated = campaign_projection_error_comparison_envelope_assessment_collection_report(
        source
    )
    assert first == repeated
    assert first is not repeated
    assert first.named_reports[0] is not repeated.named_reports[0]
    assert first.named_reports[0].report.limit_results[0] is not (
        source[0].report.limit_results[0]
    )
    source.clear()
    assert tuple(entry.name for entry in first.named_reports) == ("first", "second")
    with pytest.raises(FrozenInstanceError):
        first.collection_verdict = repeated.collection_verdict


def test_named_assessment_collection_record_converts_passing_and_verdict_schema():
    report = campaign_projection_error_comparison_envelope_assessment_collection_report(
        (_named_assessment_report("first"), _named_assessment_report("second"))
    )
    record = campaign_projection_error_comparison_envelope_assessment_collection_record(
        report
    )
    assert [item["name"] for item in record["named_reports"]] == ["first", "second"]
    assert record["collection_verdict"] == {
        "overall_passed": True,
        "passing_report_names": ["first", "second"],
        "failing_report_names": [],
        "undefined_report_names": [],
    }


def test_named_assessment_collection_record_preserves_failure_undefined_and_none():
    report = campaign_projection_error_comparison_envelope_assessment_collection_report(
        (
            _named_assessment_report("fail", ("fail", *("pass",) * 6)),
            _named_assessment_report("undefined", ("undefined", *("pass",) * 6)),
        )
    )
    record = campaign_projection_error_comparison_envelope_assessment_collection_record(
        report
    )
    assert record["collection_verdict"]["overall_passed"] is False
    assert record["collection_verdict"]["failing_report_names"] == ["fail"]
    assert record["collection_verdict"]["undefined_report_names"] == ["undefined"]
    undefined_result = record["named_reports"][1]["report"]["limit_results"][0]
    assert undefined_result["observed_minimum_difference"] is None
    assert undefined_result["lower_margin"] is None


def test_named_assessment_collection_record_preserves_nested_metric_field_order():
    assessment = campaign_projection_error_comparison_envelope_assessment_report(
        _difference_limit_result_block("ise") + _difference_limit_result_block("iae")
    )
    report = campaign_projection_error_comparison_envelope_assessment_collection_report(
        (CampaignProjectionErrorNamedAssessmentReport("multi", assessment),)
    )
    record = campaign_projection_error_comparison_envelope_assessment_collection_record(
        report
    )
    plain_report = record["named_reports"][0]["report"]
    assert [item["metric_name"] for item in plain_report["metric_verdicts"]] == [
        "ise", "iae"
    ]
    assert [
        item["difference_field"] for item in plain_report["limit_results"][:7]
    ] == list(_DIFFERENCE_FIELDS)


def test_named_assessment_collection_record_delegates_single_report_conversion(
    monkeypatch,
):
    original = analysis.campaign_projection_error_comparison_envelope_assessment_record
    calls = []

    def recording(report):
        calls.append(report)
        return original(report)

    monkeypatch.setattr(
        analysis,
        "campaign_projection_error_comparison_envelope_assessment_record",
        recording,
    )
    report = campaign_projection_error_comparison_envelope_assessment_collection_report(
        (_named_assessment_report("first"), _named_assessment_report("second"))
    )
    campaign_projection_error_comparison_envelope_assessment_collection_record(report)
    assert calls == [
        report.named_reports[0].report,
        report.named_reports[1].report,
    ]


def test_named_assessment_collection_record_rejects_malformed_collection_report():
    report = campaign_projection_error_comparison_envelope_assessment_collection_report(
        (_named_assessment_report("first"),)
    )
    malformed = CampaignProjectionErrorComparisonEnvelopeAssessmentCollectionReport(
        report.named_reports,
        CampaignProjectionErrorComparisonEnvelopeAssessmentCollectionVerdict(
            True, (), ("first",), ()
        ),
    )
    with pytest.raises(ValueError, match="classification disagrees"):
        campaign_projection_error_comparison_envelope_assessment_collection_record(
            malformed
        )
    with pytest.raises(TypeError, match="AssessmentCollectionReport"):
        campaign_projection_error_comparison_envelope_assessment_collection_record(
            object()
        )


def test_named_assessment_collection_record_empty_representation_is_explicit():
    report = campaign_projection_error_comparison_envelope_assessment_collection_report(
        ()
    )
    assert campaign_projection_error_comparison_envelope_assessment_collection_record(
        report
    ) == {
        "named_reports": [],
        "collection_verdict": {
            "overall_passed": False,
            "passing_report_names": [],
            "failing_report_names": [],
            "undefined_report_names": [],
        },
    }


def test_named_assessment_collection_record_is_plain_detached_and_deterministic():
    report = campaign_projection_error_comparison_envelope_assessment_collection_report(
        (_named_assessment_report("first"),)
    )
    first = campaign_projection_error_comparison_envelope_assessment_collection_record(
        report
    )
    repeated = campaign_projection_error_comparison_envelope_assessment_collection_record(
        report
    )
    assert first == repeated
    assert first is not repeated
    json.dumps(first, allow_nan=False)

    def assert_plain(value):
        assert type(value) in (dict, list, str, bool, int, float, type(None))
        if isinstance(value, dict):
            for key, child in value.items():
                assert type(key) is str
                assert_plain(child)
        elif isinstance(value, list):
            for child in value:
                assert_plain(child)

    assert_plain(first)
    first["named_reports"][0]["name"] = "changed"
    first["collection_verdict"]["passing_report_names"].clear()
    assert repeated["named_reports"][0]["name"] == "first"
    assert repeated["collection_verdict"]["passing_report_names"] == ["first"]


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
        CampaignMetricProjectionEnvelope("iae", -3.0, "low-iae", 5.0, "high-iae"),
        CampaignMetricProjectionEnvelope("ise", -6.0, "high-iae", 4.0, "low-iae"),
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
        CampaignMetricChangeProjection(("gain",), ("ise", "iae"), (1.0,), (2.0, 1.0)),
    )

    with pytest.raises(ValueError, match="matching parameter and metric layouts"):
        campaign_projection_envelopes(
            (_scenario_result("first", 1.0, 2.0), incompatible)
        )


@pytest.mark.parametrize("value", [True, "bad", float("inf"), float("nan")])
def test_envelopes_reject_nonnumeric_or_nonfinite_predictions(value):
    with pytest.raises(
        ValueError, match=r"predicted_metric_changes\[0\].*(numeric|finite)"
    ):
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
    inconsistent = CampaignMetricProjectionEnvelope("iae", None, "scenario", None, None)
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
        _limit_result(metric_name, 1.0, 1.0, True) for metric_name in ("iae", "ise")
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
