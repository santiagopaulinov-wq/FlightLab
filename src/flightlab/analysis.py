import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

from flightlab.persistence import _METRIC_KEYS, _validate_record

_PARAMETER_CATEGORIES = ("system", "controller", "reference", "user_metadata")


@dataclass(frozen=True, slots=True)
class CampaignComparisonEntry:
    """One immutable run entry in an ordered campaign comparison."""

    run_id: str
    parameter_value: None | bool | int | float | str
    metric_values: tuple[tuple[str, float | None], ...]


@dataclass(frozen=True, slots=True)
class CampaignDeltaEntry:
    """One immutable run entry containing explicit-baseline absolute deltas."""

    run_id: str
    parameter_delta: float
    metric_deltas: tuple[tuple[str, float | None], ...]


@dataclass(frozen=True, slots=True)
class CampaignSensitivityEntry:
    """One immutable run entry containing baseline-relative secant slopes."""

    run_id: str
    parameter_delta: float
    metric_sensitivities: tuple[tuple[str, float | None], ...]


@dataclass(frozen=True, slots=True)
class SensitivityMatrixParameter:
    """One explicit named parameter column and representative sensitivity run."""

    name: str
    sensitivities: tuple[CampaignSensitivityEntry, ...]
    representative_run_id: str


@dataclass(frozen=True, slots=True)
class CampaignSensitivityMatrix:
    """Immutable metric-row by parameter-column secant sensitivity matrix."""

    parameter_names: tuple[str, ...]
    metric_names: tuple[str, ...]
    representative_run_ids: tuple[str, ...]
    values: tuple[tuple[float | None, ...], ...]


@dataclass(frozen=True, slots=True)
class CampaignParameterChange:
    """One explicit named parameter change in sensitivity-matrix column order."""

    name: str
    value: float


@dataclass(frozen=True, slots=True)
class CampaignMetricChangeProjection:
    """Immutable linear projection from secant sensitivities."""

    parameter_names: tuple[str, ...]
    metric_names: tuple[str, ...]
    parameter_changes: tuple[float, ...]
    predicted_metric_changes: tuple[float | None, ...]


@dataclass(frozen=True, slots=True)
class CampaignProjectionScenario:
    """One explicit named sensitivity-projection change vector."""

    name: str
    parameter_changes: tuple[CampaignParameterChange, ...]


@dataclass(frozen=True, slots=True)
class CampaignProjectionScenarioResult:
    """One immutable named sensitivity-projection scenario result."""

    name: str
    projection: CampaignMetricChangeProjection


@dataclass(frozen=True, slots=True)
class CampaignMetricResidual:
    """One immutable observed-versus-projected metric residual."""

    metric_name: str
    projected_change: float | None
    observed_change: float | None
    residual: float | None


@dataclass(frozen=True, slots=True)
class CampaignProjectionResiduals:
    """Immutable ordered residuals for one scenario and observed run."""

    scenario_name: str
    observed_run_id: str
    metric_residuals: tuple[CampaignMetricResidual, ...]


@dataclass(frozen=True, slots=True)
class CampaignMetricResidualTolerance:
    """One explicit maximum absolute residual tolerance for a metric."""

    metric_name: str
    maximum_absolute_residual: float


@dataclass(frozen=True, slots=True)
class CampaignMetricResidualToleranceResult:
    """Immutable per-metric absolute residual tolerance check."""

    metric_name: str
    residual: float | None
    absolute_residual: float | None
    maximum_absolute_residual: float
    margin: float | None
    passed: bool


@dataclass(frozen=True, slots=True)
class CampaignProjectionResidualToleranceResults:
    """Immutable ordered tolerance checks for one projection residual result."""

    scenario_name: str
    observed_run_id: str
    metric_results: tuple[CampaignMetricResidualToleranceResult, ...]


@dataclass(frozen=True, slots=True)
class CampaignProjectionValidationCase:
    """One explicit named projection-validation case."""

    name: str
    scenario_result: CampaignProjectionScenarioResult
    observed_delta: CampaignDeltaEntry
    tolerances: tuple[CampaignMetricResidualTolerance, ...]


@dataclass(frozen=True, slots=True)
class CampaignProjectionValidationResult:
    """Immutable residual and tolerance results for one validation case."""

    name: str
    scenario_name: str
    observed_run_id: str
    residuals: CampaignProjectionResiduals
    tolerance_results: CampaignProjectionResidualToleranceResults


@dataclass(frozen=True, slots=True)
class CampaignProjectionValidationVerdict:
    """Immutable verdict over ordered campaign projection-validation cases."""

    overall_passed: bool
    passing_cases: tuple[str, ...]
    failing_cases: tuple[str, ...]
    undefined_cases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CampaignMetricValidationResidualEnvelope:
    """Immutable worst defined absolute residual for one validation metric."""

    metric_name: str
    maximum_absolute_residual: float | None
    validation_case_name: str | None
    scenario_name: str | None
    observed_run_id: str | None


@dataclass(frozen=True, slots=True)
class CampaignMetricProjectionErrorSummary:
    """Immutable descriptive projection-error summary for one metric."""

    metric_name: str
    validation_case_count: int
    defined_residual_count: int
    undefined_residual_count: int
    minimum_residual: float | None
    maximum_residual: float | None
    mean_residual: float | None
    mean_absolute_residual: float | None
    maximum_absolute_residual: float | None


@dataclass(frozen=True, slots=True)
class CampaignMetricProjectionErrorSummaryComparison:
    """Immutable right-minus-left comparison for one projection-error metric."""

    left_collection_name: str
    right_collection_name: str
    metric_name: str
    left_summary: CampaignMetricProjectionErrorSummary
    right_summary: CampaignMetricProjectionErrorSummary
    defined_residual_count_difference: int
    undefined_residual_count_difference: int
    minimum_residual_difference: float | None
    maximum_residual_difference: float | None
    mean_residual_difference: float | None
    mean_absolute_residual_difference: float | None
    maximum_absolute_residual_difference: float | None


@dataclass(frozen=True, slots=True)
class CampaignProjectionErrorSummaryCollection:
    """One explicitly named projection-error summary collection."""

    name: str
    summaries: tuple[CampaignMetricProjectionErrorSummary, ...]


@dataclass(frozen=True, slots=True)
class CampaignProjectionErrorSummaryComparisonSetResult:
    """One named baseline-to-comparison result in an ordered comparison set."""

    baseline_collection_name: str
    comparison_collection_name: str
    comparisons: tuple[CampaignMetricProjectionErrorSummaryComparison, ...]


@dataclass(frozen=True, slots=True)
class CampaignProjectionErrorSummaryDifferenceEnvelope:
    """Finite extrema for one stored summary-difference field and metric."""

    metric_name: str
    difference_field: str
    minimum_difference: int | float | None
    minimum_comparison_collection_name: str | None
    maximum_difference: int | float | None
    maximum_comparison_collection_name: str | None


@dataclass(frozen=True, slots=True)
class CampaignProjectionErrorSummaryDifferenceLimit:
    """One explicit allowable interval for a metric summary difference."""

    metric_name: str
    difference_field: str
    allowable_minimum_difference: int | float
    allowable_maximum_difference: int | float


@dataclass(frozen=True, slots=True)
class CampaignProjectionErrorSummaryDifferenceLimitResult:
    """Immutable margins for one comparison-envelope limit check."""

    metric_name: str
    difference_field: str
    observed_minimum_difference: int | float | None
    observed_maximum_difference: int | float | None
    allowable_minimum_difference: float
    allowable_maximum_difference: float
    lower_margin: float | None
    upper_margin: float | None
    passed: bool


@dataclass(frozen=True, slots=True)
class CampaignProjectionErrorMetricFieldIdentity:
    """Immutable projection-error metric and difference-field identity."""

    metric_name: str
    difference_field: str


@dataclass(frozen=True, slots=True)
class CampaignProjectionErrorComparisonEnvelopeLimitVerdict:
    """Immutable verdict over ordered comparison-envelope limit results."""

    overall_passed: bool
    passing_identities: tuple[CampaignProjectionErrorMetricFieldIdentity, ...]
    failing_identities: tuple[CampaignProjectionErrorMetricFieldIdentity, ...]
    undefined_identities: tuple[CampaignProjectionErrorMetricFieldIdentity, ...]


@dataclass(frozen=True, slots=True)
class CampaignProjectionErrorMetricEnvelopeLimitVerdict:
    """Immutable comparison-envelope limit verdict for one metric."""

    metric_name: str
    overall_passed: bool
    passing_identities: tuple[CampaignProjectionErrorMetricFieldIdentity, ...]
    failing_identities: tuple[CampaignProjectionErrorMetricFieldIdentity, ...]
    undefined_identities: tuple[CampaignProjectionErrorMetricFieldIdentity, ...]


@dataclass(frozen=True, slots=True)
class CampaignProjectionErrorComparisonEnvelopeAssessmentReport:
    """Immutable assembly of checked differences and their verdict views."""

    limit_results: tuple[CampaignProjectionErrorSummaryDifferenceLimitResult, ...]
    overall_verdict: CampaignProjectionErrorComparisonEnvelopeLimitVerdict
    metric_verdicts: tuple[CampaignProjectionErrorMetricEnvelopeLimitVerdict, ...]


@dataclass(frozen=True, slots=True)
class CampaignProjectionErrorNamedAssessmentReport:
    """One explicitly named comparison-envelope assessment report."""

    name: str
    report: CampaignProjectionErrorComparisonEnvelopeAssessmentReport


@dataclass(frozen=True, slots=True)
class CampaignProjectionErrorComparisonEnvelopeAssessmentCollectionVerdict:
    """Immutable verdict over ordered named comparison-envelope assessments."""

    overall_passed: bool
    passing_report_names: tuple[str, ...]
    failing_report_names: tuple[str, ...]
    undefined_report_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CampaignProjectionErrorComparisonEnvelopeAssessmentCollectionReport:
    """Immutable named assessments plus their collection verdict."""

    named_reports: tuple[CampaignProjectionErrorNamedAssessmentReport, ...]
    collection_verdict: (
        CampaignProjectionErrorComparisonEnvelopeAssessmentCollectionVerdict
    )


@dataclass(frozen=True, slots=True)
class CampaignProjectionErrorNamedAssessmentCollectionReport:
    """One explicitly named assessment collection report."""

    name: str
    report: CampaignProjectionErrorComparisonEnvelopeAssessmentCollectionReport


@dataclass(frozen=True, slots=True)
class CampaignProjectionErrorNamedAssessmentCollectionReportVerdict:
    """Immutable verdict over ordered named assessment collection reports."""

    overall_passed: bool
    passing_collection_names: tuple[str, ...]
    failing_collection_names: tuple[str, ...]
    undefined_collection_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CampaignMetricProjectionEnvelope:
    """Immutable per-metric extrema across explicit projection scenarios."""

    metric_name: str
    minimum: float | None
    minimum_scenario: str | None
    maximum: float | None
    maximum_scenario: str | None


@dataclass(frozen=True, slots=True)
class CampaignMetricProjectionLimit:
    """One explicit allowable predicted-change interval for a metric."""

    metric_name: str
    allowable_lower: float
    allowable_upper: float


@dataclass(frozen=True, slots=True)
class CampaignMetricProjectionLimitResult:
    """Immutable envelope-to-limit margins and pass/fail result."""

    metric_name: str
    observed_minimum: float | None
    observed_maximum: float | None
    allowable_lower: float
    allowable_upper: float
    lower_margin: float | None
    upper_margin: float | None
    passed: bool


@dataclass(frozen=True, slots=True)
class CampaignRobustnessVerdict:
    """Immutable deterministic verdict over ordered per-metric limit checks."""

    overall_passed: bool
    passing_metrics: tuple[str, ...]
    failing_metrics: tuple[str, ...]
    undefined_metrics: tuple[str, ...]


def campaign_robustness_verdict(
    limit_results: Iterable[CampaignMetricProjectionLimitResult],
) -> CampaignRobustnessVerdict:
    """Classify validated metric limit results into one campaign verdict."""
    try:
        result_iterator = iter(limit_results)
    except TypeError as error:
        raise TypeError("limit_results must be an iterable") from error
    limit_results = tuple(result_iterator)

    names = set()
    passing = []
    failing = []
    undefined = []
    for index, result in enumerate(limit_results):
        state = _validated_limit_result(result, index)
        if result.metric_name in names:
            raise ValueError(f"duplicate metric name {result.metric_name!r}")
        names.add(result.metric_name)
        if state == "undefined":
            undefined.append(result.metric_name)
        elif result.passed:
            passing.append(result.metric_name)
        else:
            failing.append(result.metric_name)

    return CampaignRobustnessVerdict(
        overall_passed=bool(limit_results) and not failing and not undefined,
        passing_metrics=tuple(passing),
        failing_metrics=tuple(failing),
        undefined_metrics=tuple(undefined),
    )


def _validated_limit_result(result, index):
    if not isinstance(result, CampaignMetricProjectionLimitResult):
        raise TypeError(
            f"limit_results[{index}] must be a CampaignMetricProjectionLimitResult"
        )
    if type(result.metric_name) is not str or not result.metric_name.strip():
        raise ValueError(f"limit_results[{index}].metric_name must be non-empty")
    if result.metric_name not in _METRIC_KEYS:
        raise ValueError(
            f"limit_results[{index}] has unknown metric {result.metric_name!r}"
        )
    if type(result.passed) is not bool:
        raise ValueError(f"limit_results[{index}].passed must be a boolean")
    lower = _finite_numeric(
        f"limit_results[{index}].allowable_lower", result.allowable_lower
    )
    upper = _finite_numeric(
        f"limit_results[{index}].allowable_upper", result.allowable_upper
    )
    if lower > upper:
        raise ValueError(
            f"limit_results[{index}] allowable_lower must not exceed allowable_upper"
        )

    observed_undefined = (
        result.observed_minimum is None and result.observed_maximum is None
    )
    margins_undefined = result.lower_margin is None and result.upper_margin is None
    if (
        (result.observed_minimum is None) != (result.observed_maximum is None)
        or (result.lower_margin is None) != (result.upper_margin is None)
        or observed_undefined != margins_undefined
    ):
        raise ValueError(f"limit_results[{index}] has inconsistent undefined state")
    if observed_undefined:
        if result.passed:
            raise ValueError(f"limit_results[{index}] undefined metric cannot pass")
        return "undefined"

    minimum = _finite_numeric(
        f"limit_results[{index}].observed_minimum", result.observed_minimum
    )
    maximum = _finite_numeric(
        f"limit_results[{index}].observed_maximum", result.observed_maximum
    )
    if minimum > maximum:
        raise ValueError(
            f"limit_results[{index}] observed_minimum must not exceed observed_maximum"
        )
    lower_margin = _finite_numeric(
        f"limit_results[{index}].lower_margin", result.lower_margin
    )
    upper_margin = _finite_numeric(
        f"limit_results[{index}].upper_margin", result.upper_margin
    )
    expected_lower_margin = minimum - lower
    expected_upper_margin = upper - maximum
    if not math.isfinite(expected_lower_margin) or not math.isfinite(
        expected_upper_margin
    ):
        raise ValueError(f"limit_results[{index}] expected margins must be finite")
    if lower_margin != expected_lower_margin or upper_margin != expected_upper_margin:
        raise ValueError(f"limit_results[{index}] margins are inconsistent")
    expected_pass = lower_margin >= 0.0 and upper_margin >= 0.0
    if result.passed is not expected_pass:
        raise ValueError(f"limit_results[{index}] pass state is inconsistent")
    return "defined"


def check_campaign_projection_envelope_limits(
    envelopes: Iterable[CampaignMetricProjectionEnvelope],
    limits: Iterable[CampaignMetricProjectionLimit],
) -> tuple[CampaignMetricProjectionLimitResult, ...]:
    """Check ordered deterministic projection envelopes against explicit limits."""
    try:
        envelope_iterator = iter(envelopes)
    except TypeError as error:
        raise TypeError("envelopes must be an iterable") from error
    try:
        limit_iterator = iter(limits)
    except TypeError as error:
        raise TypeError("limits must be an iterable") from error
    envelopes = tuple(envelope_iterator)
    limits = tuple(limit_iterator)
    if len(envelopes) != len(limits):
        raise ValueError("limits must provide exact metric coverage")

    metric_names = set()
    limit_names = set()
    validated = []
    for index, (envelope, limit) in enumerate(zip(envelopes, limits)):
        minimum, maximum = _validated_projection_envelope(envelope, index)
        if envelope.metric_name in metric_names:
            raise ValueError(f"duplicate envelope metric {envelope.metric_name!r}")
        metric_names.add(envelope.metric_name)
        if not isinstance(limit, CampaignMetricProjectionLimit):
            raise TypeError(f"limits[{index}] must be a CampaignMetricProjectionLimit")
        if type(limit.metric_name) is not str or not limit.metric_name.strip():
            raise ValueError(f"limits[{index}].metric_name must be non-empty")
        if limit.metric_name in limit_names:
            raise ValueError(f"duplicate limit metric {limit.metric_name!r}")
        limit_names.add(limit.metric_name)
        if limit.metric_name != envelope.metric_name:
            raise ValueError(
                f"limits[{index}].metric_name must be {envelope.metric_name!r}"
            )
        lower = _finite_numeric(
            f"limits[{index}].allowable_lower", limit.allowable_lower
        )
        upper = _finite_numeric(
            f"limits[{index}].allowable_upper", limit.allowable_upper
        )
        if lower > upper:
            raise ValueError(
                f"limits[{index}] allowable_lower must not exceed allowable_upper"
            )
        validated.append((envelope.metric_name, minimum, maximum, lower, upper))

    results = []
    for metric_name, minimum, maximum, lower, upper in validated:
        if minimum is None:
            lower_margin = None
            upper_margin = None
            passed = False
        else:
            lower_margin = minimum - lower
            upper_margin = upper - maximum
            if not math.isfinite(lower_margin) or not math.isfinite(upper_margin):
                raise ValueError(f"metric {metric_name!r} margins must be finite")
            passed = lower_margin >= 0.0 and upper_margin >= 0.0
        results.append(
            CampaignMetricProjectionLimitResult(
                metric_name=metric_name,
                observed_minimum=minimum,
                observed_maximum=maximum,
                allowable_lower=lower,
                allowable_upper=upper,
                lower_margin=lower_margin,
                upper_margin=upper_margin,
                passed=passed,
            )
        )
    return tuple(results)


def _validated_projection_envelope(envelope, index):
    if not isinstance(envelope, CampaignMetricProjectionEnvelope):
        raise TypeError(
            f"envelopes[{index}] must be a CampaignMetricProjectionEnvelope"
        )
    if type(envelope.metric_name) is not str or not envelope.metric_name.strip():
        raise ValueError(f"envelopes[{index}].metric_name must be non-empty")
    if envelope.metric_name not in _METRIC_KEYS:
        raise ValueError(
            f"envelopes[{index}] has unknown metric {envelope.metric_name!r}"
        )

    values_undefined = envelope.minimum is None and envelope.maximum is None
    names_undefined = (
        envelope.minimum_scenario is None and envelope.maximum_scenario is None
    )
    if (
        (envelope.minimum is None) != (envelope.maximum is None)
        or (envelope.minimum_scenario is None) != (envelope.maximum_scenario is None)
        or values_undefined != names_undefined
    ):
        raise ValueError(f"envelopes[{index}] has inconsistent undefined state")
    if values_undefined:
        return None, None

    minimum = _finite_numeric(f"envelopes[{index}].minimum", envelope.minimum)
    maximum = _finite_numeric(f"envelopes[{index}].maximum", envelope.maximum)
    if minimum > maximum:
        raise ValueError(f"envelopes[{index}] minimum must not exceed maximum")
    for name, value in (
        ("minimum_scenario", envelope.minimum_scenario),
        ("maximum_scenario", envelope.maximum_scenario),
    ):
        if type(value) is not str or not value.strip():
            raise ValueError(f"envelopes[{index}].{name} must be non-empty")
    return minimum, maximum


def campaign_projection_envelopes(
    scenario_results: Iterable[CampaignProjectionScenarioResult],
) -> tuple[CampaignMetricProjectionEnvelope, ...]:
    """Reduce ordered named projections to deterministic per-metric extrema."""
    try:
        result_iterator = iter(scenario_results)
    except TypeError as error:
        raise TypeError("scenario_results must be an iterable") from error
    scenario_results = tuple(result_iterator)
    if not scenario_results:
        return ()

    names = set()
    expected_parameter_names = None
    expected_metric_names = None
    for index, result in enumerate(scenario_results):
        if not isinstance(result, CampaignProjectionScenarioResult):
            raise TypeError(
                f"scenario_results[{index}] must be a CampaignProjectionScenarioResult"
            )
        if type(result.name) is not str or not result.name.strip():
            raise ValueError(f"scenario_results[{index}].name must be non-empty")
        if result.name in names:
            raise ValueError(f"duplicate scenario name {result.name!r}")
        names.add(result.name)
        parameter_names, metric_names = _validate_projection(
            result.projection, f"scenario_results[{index}].projection"
        )
        if expected_parameter_names is None:
            expected_parameter_names = parameter_names
            expected_metric_names = metric_names
        elif (
            parameter_names != expected_parameter_names
            or metric_names != expected_metric_names
        ):
            raise ValueError(
                "scenario projections must have matching parameter and metric layouts"
            )

    envelopes = []
    for metric_index, metric_name in enumerate(expected_metric_names):
        minimum = None
        minimum_scenario = None
        maximum = None
        maximum_scenario = None
        for result in scenario_results:
            value = result.projection.predicted_metric_changes[metric_index]
            if value is None:
                continue
            value = float(value)
            if minimum is None or value < minimum:
                minimum = value
                minimum_scenario = result.name
            if maximum is None or value > maximum:
                maximum = value
                maximum_scenario = result.name
        envelopes.append(
            CampaignMetricProjectionEnvelope(
                metric_name=metric_name,
                minimum=minimum,
                minimum_scenario=minimum_scenario,
                maximum=maximum,
                maximum_scenario=maximum_scenario,
            )
        )
    return tuple(envelopes)


def _validate_projection(projection, name):
    if not isinstance(projection, CampaignMetricChangeProjection):
        raise TypeError(f"{name} must be a CampaignMetricChangeProjection")
    for field_name, value in (
        ("parameter_names", projection.parameter_names),
        ("metric_names", projection.metric_names),
        ("parameter_changes", projection.parameter_changes),
        ("predicted_metric_changes", projection.predicted_metric_changes),
    ):
        if type(value) is not tuple:
            raise TypeError(f"{name}.{field_name} must be a tuple")
    if len(projection.parameter_names) != len(projection.parameter_changes):
        raise ValueError(f"{name} parameter metadata has inconsistent lengths")
    if len(projection.metric_names) != len(projection.predicted_metric_changes):
        raise ValueError(f"{name} metric metadata has inconsistent lengths")
    if not projection.parameter_names and projection.metric_names:
        raise ValueError(f"{name} empty parameter layout must have no metrics")
    _validate_unique_nonblank(projection.parameter_names, f"{name} parameter name")
    _validate_unique_nonblank(projection.metric_names, f"{name} metric name")
    for metric_name in projection.metric_names:
        if metric_name not in _METRIC_KEYS:
            raise ValueError(f"{name} has unknown metric {metric_name!r}")
    for index, value in enumerate(projection.parameter_changes):
        _finite_numeric(f"{name}.parameter_changes[{index}]", value)
    for index, value in enumerate(projection.predicted_metric_changes):
        if value is not None:
            _finite_numeric(f"{name}.predicted_metric_changes[{index}]", value)
    return projection.parameter_names, projection.metric_names


def campaign_projection_residuals(
    scenario_result: CampaignProjectionScenarioResult,
    observed_delta: CampaignDeltaEntry,
) -> CampaignProjectionResiduals:
    """Compare observed metric deltas with one named scenario projection."""
    if not isinstance(scenario_result, CampaignProjectionScenarioResult):
        raise TypeError("scenario_result must be a CampaignProjectionScenarioResult")
    if type(scenario_result.name) is not str or not scenario_result.name.strip():
        raise ValueError("scenario_result.name must be non-empty")
    _, projected_metric_names = _validate_projection(
        scenario_result.projection, "scenario_result.projection"
    )
    observed_metric_names, observed_values = _validated_observed_delta(observed_delta)
    if observed_metric_names != projected_metric_names:
        raise ValueError(
            "observed and projected metric layouts must match exactly in name and order"
        )

    residuals = []
    for metric_name, projected_change, observed_change in zip(
        projected_metric_names,
        scenario_result.projection.predicted_metric_changes,
        observed_values,
    ):
        if projected_change is None or observed_change is None:
            residual = None
        else:
            residual = float(observed_change) - float(projected_change)
            if not math.isfinite(residual):
                raise ValueError(f"metric {metric_name!r} residual must be finite")
        residuals.append(
            CampaignMetricResidual(
                metric_name=metric_name,
                projected_change=(
                    None if projected_change is None else float(projected_change)
                ),
                observed_change=(
                    None if observed_change is None else float(observed_change)
                ),
                residual=residual,
            )
        )
    return CampaignProjectionResiduals(
        scenario_name=scenario_result.name,
        observed_run_id=observed_delta.run_id,
        metric_residuals=tuple(residuals),
    )


def _validated_observed_delta(observed_delta):
    if not isinstance(observed_delta, CampaignDeltaEntry):
        raise TypeError("observed_delta must be a CampaignDeltaEntry")
    if type(observed_delta.run_id) is not str or not observed_delta.run_id.strip():
        raise ValueError("observed_delta.run_id must be non-empty")
    _finite_numeric("observed_delta.parameter_delta", observed_delta.parameter_delta)
    if type(observed_delta.metric_deltas) is not tuple:
        raise TypeError("observed_delta.metric_deltas must be a tuple")
    if not observed_delta.metric_deltas:
        raise ValueError("observed_delta.metric_deltas must not be empty")

    metric_names = []
    values = []
    for index, metric_item in enumerate(observed_delta.metric_deltas):
        if type(metric_item) is not tuple or len(metric_item) != 2:
            raise ValueError(
                f"observed_delta.metric_deltas[{index}] must be a name/value pair"
            )
        metric_name, value = metric_item
        if type(metric_name) is not str or not metric_name:
            raise ValueError("observed_delta metric name must be a non-empty string")
        if metric_name not in _METRIC_KEYS:
            raise ValueError(f"observed_delta has unknown metric {metric_name!r}")
        if metric_name in metric_names:
            raise ValueError(f"observed_delta has duplicate metric {metric_name!r}")
        metric_names.append(metric_name)
        values.append(
            None
            if value is None
            else _finite_numeric(f"observed_delta metric {metric_name!r}", value)
        )
    return tuple(metric_names), tuple(values)


def check_campaign_projection_residual_tolerances(
    residuals: CampaignProjectionResiduals,
    tolerances: Iterable[CampaignMetricResidualTolerance],
) -> CampaignProjectionResidualToleranceResults:
    """Check ordered projection residuals against explicit absolute tolerances."""
    metric_names, residual_values = _validated_projection_residuals(residuals)
    try:
        tolerance_iterator = iter(tolerances)
    except TypeError as error:
        raise TypeError("tolerances must be an iterable") from error
    tolerances = tuple(tolerance_iterator)
    if len(tolerances) != len(metric_names):
        raise ValueError("tolerances must provide exact residual metric coverage")

    checked_tolerances = []
    tolerance_names = set()
    for index, (expected_name, tolerance) in enumerate(zip(metric_names, tolerances)):
        if not isinstance(tolerance, CampaignMetricResidualTolerance):
            raise TypeError(
                f"tolerances[{index}] must be a CampaignMetricResidualTolerance"
            )
        if type(tolerance.metric_name) is not str or not tolerance.metric_name.strip():
            raise ValueError(f"tolerances[{index}].metric_name must be non-empty")
        if tolerance.metric_name in tolerance_names:
            raise ValueError(f"duplicate tolerance metric {tolerance.metric_name!r}")
        tolerance_names.add(tolerance.metric_name)
        if tolerance.metric_name != expected_name:
            raise ValueError(
                f"tolerances[{index}].metric_name must be {expected_name!r}"
            )
        maximum = _finite_numeric(
            f"tolerances[{index}].maximum_absolute_residual",
            tolerance.maximum_absolute_residual,
        )
        if maximum < 0.0:
            raise ValueError(
                f"tolerances[{index}].maximum_absolute_residual must be nonnegative"
            )
        checked_tolerances.append(maximum)

    results = []
    for metric_name, residual, maximum in zip(
        metric_names, residual_values, checked_tolerances
    ):
        if residual is None:
            absolute_residual = None
            margin = None
            passed = False
        else:
            absolute_residual = abs(residual)
            if not math.isfinite(absolute_residual):
                raise ValueError(
                    f"metric {metric_name!r} absolute residual must be finite"
                )
            margin = maximum - absolute_residual
            if not math.isfinite(margin):
                raise ValueError(f"metric {metric_name!r} margin must be finite")
            passed = margin >= 0.0
        results.append(
            CampaignMetricResidualToleranceResult(
                metric_name=metric_name,
                residual=residual,
                absolute_residual=absolute_residual,
                maximum_absolute_residual=maximum,
                margin=margin,
                passed=passed,
            )
        )
    return CampaignProjectionResidualToleranceResults(
        scenario_name=residuals.scenario_name,
        observed_run_id=residuals.observed_run_id,
        metric_results=tuple(results),
    )


def _validated_projection_residuals(residuals):
    if not isinstance(residuals, CampaignProjectionResiduals):
        raise TypeError("residuals must be a CampaignProjectionResiduals")
    for name, value in (
        ("scenario_name", residuals.scenario_name),
        ("observed_run_id", residuals.observed_run_id),
    ):
        if type(value) is not str or not value.strip():
            raise ValueError(f"residuals.{name} must be non-empty")
    if type(residuals.metric_residuals) is not tuple:
        raise TypeError("residuals.metric_residuals must be a tuple")

    metric_names = []
    values = []
    for index, item in enumerate(residuals.metric_residuals):
        if not isinstance(item, CampaignMetricResidual):
            raise TypeError(
                f"residuals.metric_residuals[{index}] must be a CampaignMetricResidual"
            )
        if type(item.metric_name) is not str or not item.metric_name.strip():
            raise ValueError(
                f"residuals.metric_residuals[{index}].metric_name must be non-empty"
            )
        if item.metric_name not in _METRIC_KEYS:
            raise ValueError(f"residuals has unknown metric {item.metric_name!r}")
        if item.metric_name in metric_names:
            raise ValueError(f"residuals has duplicate metric {item.metric_name!r}")
        metric_names.append(item.metric_name)
        projected = (
            None
            if item.projected_change is None
            else _finite_numeric(
                f"residuals metric {item.metric_name!r} projected_change",
                item.projected_change,
            )
        )
        observed = (
            None
            if item.observed_change is None
            else _finite_numeric(
                f"residuals metric {item.metric_name!r} observed_change",
                item.observed_change,
            )
        )
        residual = (
            None
            if item.residual is None
            else _finite_numeric(
                f"residuals metric {item.metric_name!r} residual", item.residual
            )
        )
        if (projected is None or observed is None) != (residual is None):
            raise ValueError(
                f"residuals metric {item.metric_name!r} has inconsistent optional fields"
            )
        values.append(residual)
    return tuple(metric_names), tuple(values)


def validate_campaign_projection_cases(
    cases: Iterable[CampaignProjectionValidationCase],
) -> tuple[CampaignProjectionValidationResult, ...]:
    """Evaluate explicit named projection-validation cases in caller order."""
    try:
        case_iterator = iter(cases)
    except TypeError as error:
        raise TypeError("cases must be an iterable") from error
    cases = tuple(case_iterator)

    names = set()
    for index, case in enumerate(cases):
        if not isinstance(case, CampaignProjectionValidationCase):
            raise TypeError(
                f"cases[{index}] must be a CampaignProjectionValidationCase"
            )
        if type(case.name) is not str or not case.name.strip():
            raise ValueError(f"cases[{index}].name must be non-empty")
        if case.name in names:
            raise ValueError(f"duplicate validation case name {case.name!r}")
        names.add(case.name)
        if not isinstance(case.scenario_result, CampaignProjectionScenarioResult):
            raise TypeError(
                f"cases[{index}].scenario_result must be a "
                "CampaignProjectionScenarioResult"
            )
        if not isinstance(case.observed_delta, CampaignDeltaEntry):
            raise TypeError(
                f"cases[{index}].observed_delta must be a CampaignDeltaEntry"
            )
        if type(case.tolerances) is not tuple:
            raise TypeError(f"cases[{index}].tolerances must be a tuple")

    results = []
    for case in cases:
        residuals = campaign_projection_residuals(
            case.scenario_result, case.observed_delta
        )
        tolerance_results = check_campaign_projection_residual_tolerances(
            residuals, case.tolerances
        )
        results.append(
            CampaignProjectionValidationResult(
                name=case.name,
                scenario_name=residuals.scenario_name,
                observed_run_id=residuals.observed_run_id,
                residuals=residuals,
                tolerance_results=tolerance_results,
            )
        )
    return tuple(results)


def campaign_projection_validation_verdict(
    validation_results: Iterable[CampaignProjectionValidationResult],
) -> CampaignProjectionValidationVerdict:
    """Summarize ordered projection-validation results into one verdict."""
    validation_results, states = _validated_projection_validation_results(
        validation_results
    )

    passing = []
    failing = []
    undefined = []
    for result, state in zip(validation_results, states):
        if state == "undefined":
            undefined.append(result.name)
        elif state == "failing":
            failing.append(result.name)
        else:
            passing.append(result.name)
    return CampaignProjectionValidationVerdict(
        overall_passed=bool(validation_results) and not failing and not undefined,
        passing_cases=tuple(passing),
        failing_cases=tuple(failing),
        undefined_cases=tuple(undefined),
    )


def campaign_projection_validation_residual_envelopes(
    validation_results: Iterable[CampaignProjectionValidationResult],
) -> tuple[CampaignMetricValidationResidualEnvelope, ...]:
    """Find each metric's worst defined absolute validation residual."""
    validation_results, _ = _validated_projection_validation_results(validation_results)
    if not validation_results:
        return ()

    metric_names = tuple(
        item.metric_name
        for item in validation_results[0].tolerance_results.metric_results
    )
    for index, result in enumerate(validation_results[1:], start=1):
        result_metric_names = tuple(
            item.metric_name for item in result.tolerance_results.metric_results
        )
        if result_metric_names != metric_names:
            raise ValueError(
                f"validation_results[{index}] metric layout is incompatible"
            )

    envelopes = []
    for metric_index, metric_name in enumerate(metric_names):
        maximum = None
        attaining_result = None
        for result in validation_results:
            absolute_residual = result.tolerance_results.metric_results[
                metric_index
            ].absolute_residual
            if absolute_residual is None:
                continue
            absolute_residual = float(absolute_residual)
            if maximum is None or absolute_residual > maximum:
                maximum = absolute_residual
                attaining_result = result
        envelopes.append(
            CampaignMetricValidationResidualEnvelope(
                metric_name=metric_name,
                maximum_absolute_residual=maximum,
                validation_case_name=(
                    None if attaining_result is None else attaining_result.name
                ),
                scenario_name=(
                    None if attaining_result is None else attaining_result.scenario_name
                ),
                observed_run_id=(
                    None
                    if attaining_result is None
                    else attaining_result.observed_run_id
                ),
            )
        )
    return tuple(envelopes)


def campaign_projection_error_summaries(
    validation_results: Iterable[CampaignProjectionValidationResult],
) -> tuple[CampaignMetricProjectionErrorSummary, ...]:
    """Summarize stored validation residuals by metric in layout order."""
    validation_results, _ = _validated_projection_validation_results(validation_results)
    if not validation_results:
        return ()
    envelopes = campaign_projection_validation_residual_envelopes(validation_results)
    metric_names = tuple(envelope.metric_name for envelope in envelopes)

    summaries = []
    for metric_index, (metric_name, envelope) in enumerate(
        zip(metric_names, envelopes)
    ):
        values = tuple(
            result.tolerance_results.metric_results[metric_index].residual
            for result in validation_results
            if result.tolerance_results.metric_results[metric_index].residual
            is not None
        )
        defined_count = len(values)
        undefined_count = len(validation_results) - defined_count
        if not values:
            minimum = None
            maximum = None
            mean = None
            mean_absolute = None
        else:
            minimum = min(values)
            maximum = max(values)
            try:
                mean = math.fsum(values) / defined_count
                mean_absolute = (
                    math.fsum(abs(value) for value in values) / defined_count
                )
            except OverflowError as error:
                raise ValueError(
                    f"metric {metric_name!r} summary arithmetic must be finite"
                ) from error
            for field_name, value in (
                ("minimum residual", minimum),
                ("maximum residual", maximum),
                ("mean residual", mean),
                ("mean absolute residual", mean_absolute),
            ):
                if not math.isfinite(value):
                    raise ValueError(
                        f"metric {metric_name!r} {field_name} must be finite"
                    )
        summaries.append(
            CampaignMetricProjectionErrorSummary(
                metric_name=metric_name,
                validation_case_count=len(validation_results),
                defined_residual_count=defined_count,
                undefined_residual_count=undefined_count,
                minimum_residual=minimum,
                maximum_residual=maximum,
                mean_residual=mean,
                mean_absolute_residual=mean_absolute,
                maximum_absolute_residual=envelope.maximum_absolute_residual,
            )
        )
    return tuple(summaries)


def compare_campaign_projection_error_summaries(
    left_collection_name: str,
    left_summaries: Iterable[CampaignMetricProjectionErrorSummary],
    right_collection_name: str,
    right_summaries: Iterable[CampaignMetricProjectionErrorSummary],
) -> tuple[CampaignMetricProjectionErrorSummaryComparison, ...]:
    """Compare aligned projection-error summaries as right minus left."""
    for name, value in (
        ("left_collection_name", left_collection_name),
        ("right_collection_name", right_collection_name),
    ):
        if type(value) is not str or not value.strip():
            raise ValueError(f"{name} must be non-empty")
    if left_collection_name == right_collection_name:
        raise ValueError("collection names must be distinct")

    left_summaries = _validated_projection_error_summaries(
        left_summaries, "left_summaries"
    )
    right_summaries = _validated_projection_error_summaries(
        right_summaries, "right_summaries"
    )
    left_metric_names = tuple(summary.metric_name for summary in left_summaries)
    right_metric_names = tuple(summary.metric_name for summary in right_summaries)
    if left_metric_names != right_metric_names:
        raise ValueError(
            "left and right summaries must have identical metric names and order"
        )

    comparisons = []
    optional_fields = (
        "minimum_residual",
        "maximum_residual",
        "mean_residual",
        "mean_absolute_residual",
        "maximum_absolute_residual",
    )
    for left, right in zip(left_summaries, right_summaries):
        differences = {}
        for field_name in optional_fields:
            left_value = getattr(left, field_name)
            right_value = getattr(right, field_name)
            if left_value is None or right_value is None:
                difference = None
            else:
                difference = right_value - left_value
                if not math.isfinite(difference):
                    raise ValueError(
                        f"metric {left.metric_name!r} {field_name} difference "
                        "must be finite"
                    )
            differences[field_name] = difference
        comparisons.append(
            CampaignMetricProjectionErrorSummaryComparison(
                left_collection_name=left_collection_name,
                right_collection_name=right_collection_name,
                metric_name=left.metric_name,
                left_summary=_copy_projection_error_summary(left),
                right_summary=_copy_projection_error_summary(right),
                defined_residual_count_difference=(
                    right.defined_residual_count - left.defined_residual_count
                ),
                undefined_residual_count_difference=(
                    right.undefined_residual_count - left.undefined_residual_count
                ),
                minimum_residual_difference=differences["minimum_residual"],
                maximum_residual_difference=differences["maximum_residual"],
                mean_residual_difference=differences["mean_residual"],
                mean_absolute_residual_difference=differences["mean_absolute_residual"],
                maximum_absolute_residual_difference=differences[
                    "maximum_absolute_residual"
                ],
            )
        )
    return tuple(comparisons)


def compare_campaign_projection_error_summary_collections(
    baseline_collection_name: str,
    baseline_summaries: Iterable[CampaignMetricProjectionErrorSummary],
    comparison_collections: Iterable[CampaignProjectionErrorSummaryCollection],
) -> tuple[CampaignProjectionErrorSummaryComparisonSetResult, ...]:
    """Compare ordered named summary collections with one explicit baseline."""
    if (
        type(baseline_collection_name) is not str
        or not baseline_collection_name.strip()
    ):
        raise ValueError("baseline_collection_name must be non-empty")

    baseline_summaries = _validated_projection_error_summaries(
        baseline_summaries, "baseline_summaries"
    )
    try:
        collection_iterator = iter(comparison_collections)
    except TypeError as error:
        raise TypeError("comparison_collections must be an iterable") from error
    comparison_collections = tuple(collection_iterator)

    names = set()
    validated_collections = []
    for index, collection in enumerate(comparison_collections):
        prefix = f"comparison_collections[{index}]"
        if not isinstance(collection, CampaignProjectionErrorSummaryCollection):
            raise TypeError(
                f"{prefix} must be a CampaignProjectionErrorSummaryCollection"
            )
        if type(collection.name) is not str or not collection.name.strip():
            raise ValueError(f"{prefix}.name must be non-empty")
        if collection.name == baseline_collection_name:
            raise ValueError(
                f"{prefix}.name must be distinct from baseline_collection_name"
            )
        if collection.name in names:
            raise ValueError(
                f"duplicate comparison collection name {collection.name!r}"
            )
        names.add(collection.name)
        summaries = _validated_projection_error_summaries(
            collection.summaries, f"{prefix}.summaries"
        )
        validated_collections.append((collection.name, summaries))

    return tuple(
        CampaignProjectionErrorSummaryComparisonSetResult(
            baseline_collection_name=baseline_collection_name,
            comparison_collection_name=comparison_name,
            comparisons=compare_campaign_projection_error_summaries(
                baseline_collection_name,
                baseline_summaries,
                comparison_name,
                comparison_summaries,
            ),
        )
        for comparison_name, comparison_summaries in validated_collections
    )


_PROJECTION_ERROR_COMPARISON_DIFFERENCE_FIELDS = (
    "defined_residual_count_difference",
    "undefined_residual_count_difference",
    "minimum_residual_difference",
    "maximum_residual_difference",
    "mean_residual_difference",
    "mean_absolute_residual_difference",
    "maximum_absolute_residual_difference",
)


def campaign_projection_error_comparison_set_metric_envelopes(
    comparison_set_results: Iterable[
        CampaignProjectionErrorSummaryComparisonSetResult
    ],
) -> tuple[CampaignProjectionErrorSummaryDifferenceEnvelope, ...]:
    """Envelope stored comparison differences in metric and field order."""
    comparison_set_results, metric_names = (
        _validated_projection_error_comparison_set_results(comparison_set_results)
    )
    envelopes = []
    for metric_index, metric_name in enumerate(metric_names):
        for field_name in _PROJECTION_ERROR_COMPARISON_DIFFERENCE_FIELDS:
            minimum = None
            minimum_name = None
            maximum = None
            maximum_name = None
            for result in comparison_set_results:
                value = getattr(result.comparisons[metric_index], field_name)
                if value is None:
                    continue
                if minimum is None or value < minimum:
                    minimum = value
                    minimum_name = result.comparison_collection_name
                if maximum is None or value > maximum:
                    maximum = value
                    maximum_name = result.comparison_collection_name
            envelopes.append(
                CampaignProjectionErrorSummaryDifferenceEnvelope(
                    metric_name=metric_name,
                    difference_field=field_name,
                    minimum_difference=minimum,
                    minimum_comparison_collection_name=minimum_name,
                    maximum_difference=maximum,
                    maximum_comparison_collection_name=maximum_name,
                )
            )
    return tuple(envelopes)


def check_campaign_projection_error_comparison_envelope_limits(
    envelopes: Iterable[CampaignProjectionErrorSummaryDifferenceEnvelope],
    limits: Iterable[CampaignProjectionErrorSummaryDifferenceLimit],
) -> tuple[CampaignProjectionErrorSummaryDifferenceLimitResult, ...]:
    """Check stored comparison envelopes against explicit aligned intervals."""
    envelopes = _validated_projection_error_difference_envelopes(envelopes)
    try:
        limit_iterator = iter(limits)
    except TypeError as error:
        raise TypeError("limits must be an iterable") from error
    limits = tuple(limit_iterator)
    if len(envelopes) != len(limits):
        raise ValueError("limits must provide exact metric and field coverage")

    validated = []
    limit_identities = set()
    for index, (envelope, limit) in enumerate(zip(envelopes, limits)):
        if not isinstance(limit, CampaignProjectionErrorSummaryDifferenceLimit):
            raise TypeError(
                f"limits[{index}] must be a "
                "CampaignProjectionErrorSummaryDifferenceLimit"
            )
        for field_name in ("metric_name", "difference_field"):
            value = getattr(limit, field_name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"limits[{index}].{field_name} must be non-empty")
        identity = (limit.metric_name, limit.difference_field)
        if identity in limit_identities:
            raise ValueError(f"duplicate limit identity {identity!r}")
        limit_identities.add(identity)
        expected_identity = (envelope.metric_name, envelope.difference_field)
        if identity != expected_identity:
            raise ValueError(
                f"limits[{index}] identity must be {expected_identity!r}"
            )
        lower = _finite_numeric(
            f"limits[{index}].allowable_minimum_difference",
            limit.allowable_minimum_difference,
        )
        upper = _finite_numeric(
            f"limits[{index}].allowable_maximum_difference",
            limit.allowable_maximum_difference,
        )
        if lower > upper:
            raise ValueError(
                f"limits[{index}] allowable_minimum_difference must not exceed "
                "allowable_maximum_difference"
            )
        validated.append((envelope, lower, upper))

    results = []
    for envelope, lower, upper in validated:
        if envelope.minimum_difference is None:
            lower_margin = None
            upper_margin = None
            passed = False
        else:
            lower_margin = envelope.minimum_difference - lower
            upper_margin = upper - envelope.maximum_difference
            if not math.isfinite(lower_margin) or not math.isfinite(upper_margin):
                raise ValueError(
                    f"metric {envelope.metric_name!r} field "
                    f"{envelope.difference_field!r} margins must be finite"
                )
            passed = lower_margin >= 0.0 and upper_margin >= 0.0
        results.append(
            CampaignProjectionErrorSummaryDifferenceLimitResult(
                metric_name=envelope.metric_name,
                difference_field=envelope.difference_field,
                observed_minimum_difference=envelope.minimum_difference,
                observed_maximum_difference=envelope.maximum_difference,
                allowable_minimum_difference=lower,
                allowable_maximum_difference=upper,
                lower_margin=lower_margin,
                upper_margin=upper_margin,
                passed=passed,
            )
        )
    return tuple(results)


def campaign_projection_error_comparison_envelope_limit_verdict(
    limit_results: Iterable[CampaignProjectionErrorSummaryDifferenceLimitResult],
) -> CampaignProjectionErrorComparisonEnvelopeLimitVerdict:
    """Classify validated comparison-envelope limit results into one verdict."""
    limit_results, states = _validated_projection_error_difference_limit_results(
        limit_results
    )
    passing = []
    failing = []
    undefined = []
    for result, state in zip(limit_results, states):
        identity = CampaignProjectionErrorMetricFieldIdentity(
            result.metric_name, result.difference_field
        )
        if state == "undefined":
            undefined.append(identity)
        elif result.passed:
            passing.append(identity)
        else:
            failing.append(identity)
    return CampaignProjectionErrorComparisonEnvelopeLimitVerdict(
        overall_passed=bool(limit_results) and not failing and not undefined,
        passing_identities=tuple(passing),
        failing_identities=tuple(failing),
        undefined_identities=tuple(undefined),
    )


def campaign_projection_error_comparison_envelope_metric_verdicts(
    limit_results: Iterable[CampaignProjectionErrorSummaryDifferenceLimitResult],
) -> tuple[CampaignProjectionErrorMetricEnvelopeLimitVerdict, ...]:
    """Classify validated comparison-envelope limit results for each metric."""
    limit_results, states = _validated_projection_error_difference_limit_results(
        limit_results
    )
    field_count = len(_PROJECTION_ERROR_COMPARISON_DIFFERENCE_FIELDS)
    verdicts = []
    for block_start in range(0, len(limit_results), field_count):
        result_block = limit_results[block_start : block_start + field_count]
        state_block = states[block_start : block_start + field_count]
        passing = []
        failing = []
        undefined = []
        for result, state in zip(result_block, state_block):
            identity = CampaignProjectionErrorMetricFieldIdentity(
                result.metric_name, result.difference_field
            )
            if state == "undefined":
                undefined.append(identity)
            elif result.passed:
                passing.append(identity)
            else:
                failing.append(identity)
        verdicts.append(
            CampaignProjectionErrorMetricEnvelopeLimitVerdict(
                metric_name=result_block[0].metric_name,
                overall_passed=not failing and not undefined,
                passing_identities=tuple(passing),
                failing_identities=tuple(failing),
                undefined_identities=tuple(undefined),
            )
        )
    return tuple(verdicts)


def campaign_projection_error_comparison_envelope_assessment_report(
    limit_results: Iterable[CampaignProjectionErrorSummaryDifferenceLimitResult],
) -> CampaignProjectionErrorComparisonEnvelopeAssessmentReport:
    """Assemble validated limit results with both existing verdict views."""
    try:
        result_iterator = iter(limit_results)
    except TypeError as error:
        raise TypeError("limit_results must be an iterable") from error
    limit_results = tuple(result_iterator)

    overall_verdict = campaign_projection_error_comparison_envelope_limit_verdict(
        limit_results
    )
    metric_verdicts = campaign_projection_error_comparison_envelope_metric_verdicts(
        limit_results
    )
    _validate_projection_error_assessment_consistency(
        limit_results, overall_verdict, metric_verdicts
    )
    return CampaignProjectionErrorComparisonEnvelopeAssessmentReport(
        limit_results=tuple(
            _copy_projection_error_difference_limit_result(result)
            for result in limit_results
        ),
        overall_verdict=overall_verdict,
        metric_verdicts=metric_verdicts,
    )


def campaign_projection_error_comparison_envelope_assessment_record(report):
    """Return a detached JSON-compatible record for one assessment report."""
    _validated_projection_error_assessment_report(report)

    def identity_records(identities):
        return [
            {
                "metric_name": identity.metric_name,
                "difference_field": identity.difference_field,
            }
            for identity in identities
        ]

    return {
        "limit_results": [
            {
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
            for result in report.limit_results
        ],
        "overall_verdict": {
            "overall_passed": report.overall_verdict.overall_passed,
            "passing_identities": identity_records(
                report.overall_verdict.passing_identities
            ),
            "failing_identities": identity_records(
                report.overall_verdict.failing_identities
            ),
            "undefined_identities": identity_records(
                report.overall_verdict.undefined_identities
            ),
        },
        "metric_verdicts": [
            {
                "metric_name": verdict.metric_name,
                "overall_passed": verdict.overall_passed,
                "passing_difference_fields": [
                    identity.difference_field
                    for identity in verdict.passing_identities
                ],
                "failing_difference_fields": [
                    identity.difference_field
                    for identity in verdict.failing_identities
                ],
                "undefined_difference_fields": [
                    identity.difference_field
                    for identity in verdict.undefined_identities
                ],
            }
            for verdict in report.metric_verdicts
        ],
    }


def campaign_projection_error_comparison_envelope_named_assessment_records(entries):
    """Convert ordered named assessment reports to detached plain records."""
    entries = _validated_projection_error_named_assessment_reports(entries)
    return [
        {
            "name": entry.name,
            "report": (
                campaign_projection_error_comparison_envelope_assessment_record(
                    entry.report
                )
            ),
        }
        for entry in entries
    ]


def campaign_projection_error_comparison_envelope_verdict_overview(entries):
    """Extract stored overall and per-metric verdict states as plain values."""
    entries = _validated_projection_error_named_assessment_reports(entries)
    for entry in entries:
        _validated_projection_error_assessment_report(entry.report)
    return [
        {
            "name": entry.name,
            "overall_passed": entry.report.overall_verdict.overall_passed,
            "metrics": [
                {
                    "metric": verdict.metric_name,
                    "passed": verdict.overall_passed,
                }
                for verdict in entry.report.metric_verdicts
            ],
        }
        for entry in entries
    ]


def campaign_projection_error_comparison_envelope_assessment_collection_verdict(
    entries,
):
    """Classify ordered named assessment reports using their stored verdicts."""
    entries = _validated_projection_error_named_assessment_reports(entries)
    for entry in entries:
        _validated_projection_error_assessment_report(entry.report)

    passing = []
    failing = []
    undefined = []
    for entry in entries:
        verdict = entry.report.overall_verdict
        if verdict.undefined_identities:
            undefined.append(entry.name)
        elif verdict.overall_passed:
            passing.append(entry.name)
        else:
            failing.append(entry.name)
    return CampaignProjectionErrorComparisonEnvelopeAssessmentCollectionVerdict(
        overall_passed=bool(entries) and not failing and not undefined,
        passing_report_names=tuple(passing),
        failing_report_names=tuple(failing),
        undefined_report_names=tuple(undefined),
    )


def campaign_projection_error_comparison_envelope_assessment_collection_report(
    entries,
):
    """Assemble detached named assessments with their existing verdict."""
    entries = _validated_projection_error_named_assessment_reports(entries)
    collection_verdict = (
        campaign_projection_error_comparison_envelope_assessment_collection_verdict(
            entries
        )
    )
    _validate_projection_error_assessment_collection_consistency(
        entries, collection_verdict
    )
    return CampaignProjectionErrorComparisonEnvelopeAssessmentCollectionReport(
        named_reports=tuple(
            CampaignProjectionErrorNamedAssessmentReport(
                entry.name, _copy_projection_error_assessment_report(entry.report)
            )
            for entry in entries
        ),
        collection_verdict=collection_verdict,
    )


def campaign_projection_error_comparison_envelope_assessment_collection_record(
    report,
):
    """Return a detached plain record for one assessment collection report."""
    _validated_projection_error_assessment_collection_report(report)
    verdict = report.collection_verdict
    return {
        "named_reports": (
            campaign_projection_error_comparison_envelope_named_assessment_records(
                report.named_reports
            )
        ),
        "collection_verdict": {
            "overall_passed": verdict.overall_passed,
            "passing_report_names": list(verdict.passing_report_names),
            "failing_report_names": list(verdict.failing_report_names),
            "undefined_report_names": list(verdict.undefined_report_names),
        },
    }


def campaign_projection_error_comparison_envelope_named_assessment_collection_records(
    entries,
):
    """Convert ordered named assessment collection reports to plain records."""
    entries = _validated_projection_error_named_assessment_collection_reports(entries)
    return [
        {
            "name": entry.name,
            "report": (
                campaign_projection_error_comparison_envelope_assessment_collection_record(
                    entry.report
                )
            ),
        }
        for entry in entries
    ]


def campaign_projection_error_comparison_envelope_assessment_collection_verdict_overview(
    entries,
):
    """Extract stored named collection pass states as compact plain values."""
    entries = _validated_projection_error_named_assessment_collection_reports(entries)
    for entry in entries:
        _validated_projection_error_assessment_collection_report(entry.report)
    return [
        {
            "name": entry.name,
            "overall_passed": entry.report.collection_verdict.overall_passed,
        }
        for entry in entries
    ]


def campaign_projection_error_comparison_envelope_named_assessment_collection_verdict(
    entries,
):
    """Classify named assessment collection reports by stored verdict state."""
    entries = _validated_projection_error_named_assessment_collection_reports(entries)
    for entry in entries:
        _validated_projection_error_assessment_collection_report(entry.report)

    passing = []
    failing = []
    undefined = []
    for entry in entries:
        verdict = entry.report.collection_verdict
        if verdict.undefined_report_names:
            undefined.append(entry.name)
        elif verdict.overall_passed:
            passing.append(entry.name)
        else:
            failing.append(entry.name)
    return CampaignProjectionErrorNamedAssessmentCollectionReportVerdict(
        overall_passed=bool(entries) and not failing and not undefined,
        passing_collection_names=tuple(passing),
        failing_collection_names=tuple(failing),
        undefined_collection_names=tuple(undefined),
    )


def campaign_projection_error_comparison_envelope_named_assessment_collection_verdict_record(
    verdict,
):
    """Convert one named assessment collection verdict to a plain record."""
    _validated_projection_error_named_assessment_collection_verdict(verdict)
    return {
        "overall_passed": verdict.overall_passed,
        "passing_collection_names": list(verdict.passing_collection_names),
        "failing_collection_names": list(verdict.failing_collection_names),
        "undefined_collection_names": list(verdict.undefined_collection_names),
    }


def _validated_projection_error_named_assessment_collection_verdict(verdict):
    if not isinstance(
        verdict, CampaignProjectionErrorNamedAssessmentCollectionReportVerdict
    ):
        raise TypeError(
            "verdict must be a "
            "CampaignProjectionErrorNamedAssessmentCollectionReportVerdict"
        )
    if type(verdict.overall_passed) is not bool:
        raise ValueError("verdict.overall_passed must be a boolean")

    names = set()
    for category_name in (
        "passing_collection_names",
        "failing_collection_names",
        "undefined_collection_names",
    ):
        category = getattr(verdict, category_name)
        if type(category) is not tuple:
            raise TypeError(f"verdict.{category_name} must be a tuple")
        for index, name in enumerate(category):
            if type(name) is not str or not name.strip():
                raise ValueError(
                    f"verdict.{category_name}[{index}] must be non-empty"
                )
            if name in names:
                raise ValueError("verdict categories are not mutually exclusive")
            names.add(name)

    expected_pass = bool(verdict.passing_collection_names) and not (
        verdict.failing_collection_names or verdict.undefined_collection_names
    )
    if verdict.overall_passed is not expected_pass:
        raise ValueError("verdict overall pass state is inconsistent")
    return verdict


def _validated_projection_error_named_assessment_collection_reports(entries):
    try:
        entry_iterator = iter(entries)
    except TypeError as error:
        raise TypeError("entries must be an iterable") from error
    entries = tuple(entry_iterator)

    names = set()
    for index, entry in enumerate(entries):
        prefix = f"entries[{index}]"
        if not isinstance(
            entry, CampaignProjectionErrorNamedAssessmentCollectionReport
        ):
            raise TypeError(
                f"{prefix} must be a "
                "CampaignProjectionErrorNamedAssessmentCollectionReport"
            )
        if type(entry.name) is not str or not entry.name.strip():
            raise ValueError(f"{prefix}.name must be non-empty")
        if entry.name in names:
            raise ValueError(f"duplicate assessment collection report name {entry.name!r}")
        names.add(entry.name)
        if not isinstance(
            entry.report,
            CampaignProjectionErrorComparisonEnvelopeAssessmentCollectionReport,
        ):
            raise TypeError(
                f"{prefix}.report must be a "
                "CampaignProjectionErrorComparisonEnvelopeAssessmentCollectionReport"
            )
    return entries


def _validated_projection_error_assessment_collection_report(report):
    if not isinstance(
        report, CampaignProjectionErrorComparisonEnvelopeAssessmentCollectionReport
    ):
        raise TypeError(
            "report must be a "
            "CampaignProjectionErrorComparisonEnvelopeAssessmentCollectionReport"
        )
    if type(report.named_reports) is not tuple:
        raise TypeError("report.named_reports must be a tuple")
    entries = _validated_projection_error_named_assessment_reports(report.named_reports)
    for entry in entries:
        _validated_projection_error_assessment_report(entry.report)
    _validate_projection_error_assessment_collection_consistency(
        entries, report.collection_verdict
    )
    return report


def _validate_projection_error_assessment_collection_consistency(entries, verdict):
    if not isinstance(
        verdict,
        CampaignProjectionErrorComparisonEnvelopeAssessmentCollectionVerdict,
    ):
        raise TypeError(
            "collection verdict must be a "
            "CampaignProjectionErrorComparisonEnvelopeAssessmentCollectionVerdict"
        )
    if type(verdict.overall_passed) is not bool:
        raise ValueError("collection verdict overall_passed must be a boolean")
    source_names = tuple(entry.name for entry in entries)
    source_positions = {name: index for index, name in enumerate(source_names)}
    categories = {}
    for category_name, names in (
        ("passing", verdict.passing_report_names),
        ("failing", verdict.failing_report_names),
        ("undefined", verdict.undefined_report_names),
    ):
        if type(names) is not tuple:
            raise TypeError(f"collection verdict {category_name} names must be a tuple")
        positions = []
        for index, name in enumerate(names):
            if type(name) is not str or not name.strip():
                raise ValueError(
                    f"collection verdict {category_name} names[{index}] "
                    "must be non-empty"
                )
            if name in categories:
                raise ValueError("collection verdict categories are not mutually exclusive")
            if name not in source_positions:
                raise ValueError("collection verdict contains an unknown report name")
            categories[name] = category_name
            positions.append(source_positions[name])
        if positions != sorted(positions):
            raise ValueError("collection verdict report-name order is inconsistent")
    if set(categories) != set(source_names):
        raise ValueError("collection verdict report names do not match named reports")

    for entry in entries:
        stored_verdict = entry.report.overall_verdict
        expected_category = (
            "undefined"
            if stored_verdict.undefined_identities
            else "passing"
            if stored_verdict.overall_passed
            else "failing"
        )
        if categories[entry.name] != expected_category:
            raise ValueError(
                "collection verdict classification disagrees with named reports"
            )
    expected_pass = bool(entries) and all(
        category == "passing" for category in categories.values()
    )
    if verdict.overall_passed is not expected_pass:
        raise ValueError("collection verdict overall pass state is inconsistent")


def _copy_projection_error_assessment_report(report):
    return CampaignProjectionErrorComparisonEnvelopeAssessmentReport(
        limit_results=tuple(
            _copy_projection_error_difference_limit_result(result)
            for result in report.limit_results
        ),
        overall_verdict=CampaignProjectionErrorComparisonEnvelopeLimitVerdict(
            overall_passed=report.overall_verdict.overall_passed,
            passing_identities=tuple(
                CampaignProjectionErrorMetricFieldIdentity(
                    identity.metric_name, identity.difference_field
                )
                for identity in report.overall_verdict.passing_identities
            ),
            failing_identities=tuple(
                CampaignProjectionErrorMetricFieldIdentity(
                    identity.metric_name, identity.difference_field
                )
                for identity in report.overall_verdict.failing_identities
            ),
            undefined_identities=tuple(
                CampaignProjectionErrorMetricFieldIdentity(
                    identity.metric_name, identity.difference_field
                )
                for identity in report.overall_verdict.undefined_identities
            ),
        ),
        metric_verdicts=tuple(
            CampaignProjectionErrorMetricEnvelopeLimitVerdict(
                metric_name=verdict.metric_name,
                overall_passed=verdict.overall_passed,
                passing_identities=tuple(
                    CampaignProjectionErrorMetricFieldIdentity(
                        identity.metric_name, identity.difference_field
                    )
                    for identity in verdict.passing_identities
                ),
                failing_identities=tuple(
                    CampaignProjectionErrorMetricFieldIdentity(
                        identity.metric_name, identity.difference_field
                    )
                    for identity in verdict.failing_identities
                ),
                undefined_identities=tuple(
                    CampaignProjectionErrorMetricFieldIdentity(
                        identity.metric_name, identity.difference_field
                    )
                    for identity in verdict.undefined_identities
                ),
            )
            for verdict in report.metric_verdicts
        ),
    )


def _validated_projection_error_named_assessment_reports(entries):
    try:
        entry_iterator = iter(entries)
    except TypeError as error:
        raise TypeError("entries must be an iterable") from error
    entries = tuple(entry_iterator)

    names = set()
    for index, entry in enumerate(entries):
        prefix = f"entries[{index}]"
        if not isinstance(entry, CampaignProjectionErrorNamedAssessmentReport):
            raise TypeError(
                f"{prefix} must be a CampaignProjectionErrorNamedAssessmentReport"
            )
        if type(entry.name) is not str or not entry.name.strip():
            raise ValueError(f"{prefix}.name must be non-empty")
        if entry.name in names:
            raise ValueError(f"duplicate assessment report name {entry.name!r}")
        names.add(entry.name)
        if not isinstance(
            entry.report, CampaignProjectionErrorComparisonEnvelopeAssessmentReport
        ):
            raise TypeError(
                f"{prefix}.report must be a "
                "CampaignProjectionErrorComparisonEnvelopeAssessmentReport"
            )
    return entries


def _validated_projection_error_assessment_report(report):
    if not isinstance(
        report, CampaignProjectionErrorComparisonEnvelopeAssessmentReport
    ):
        raise TypeError(
            "report must be a "
            "CampaignProjectionErrorComparisonEnvelopeAssessmentReport"
        )
    if type(report.limit_results) is not tuple:
        raise TypeError("report.limit_results must be a tuple")
    limit_results, states = _validated_projection_error_difference_limit_results(
        report.limit_results
    )
    overall = report.overall_verdict
    if not isinstance(overall, CampaignProjectionErrorComparisonEnvelopeLimitVerdict):
        raise TypeError(
            "report.overall_verdict must be a "
            "CampaignProjectionErrorComparisonEnvelopeLimitVerdict"
        )
    if type(overall.overall_passed) is not bool:
        raise ValueError("report.overall_verdict.overall_passed must be a boolean")
    for category_name in (
        "passing_identities",
        "failing_identities",
        "undefined_identities",
    ):
        _validated_projection_error_identity_tuple(
            getattr(overall, category_name),
            f"report.overall_verdict.{category_name}",
        )

    if type(report.metric_verdicts) is not tuple:
        raise TypeError("report.metric_verdicts must be a tuple")
    for index, verdict in enumerate(report.metric_verdicts):
        prefix = f"report.metric_verdicts[{index}]"
        if not isinstance(verdict, CampaignProjectionErrorMetricEnvelopeLimitVerdict):
            raise TypeError(
                f"{prefix} must be a "
                "CampaignProjectionErrorMetricEnvelopeLimitVerdict"
            )
        if type(verdict.metric_name) is not str or not verdict.metric_name.strip():
            raise ValueError(f"{prefix}.metric_name must be non-empty")
        if type(verdict.overall_passed) is not bool:
            raise ValueError(f"{prefix}.overall_passed must be a boolean")
        for category_name in (
            "passing_identities",
            "failing_identities",
            "undefined_identities",
        ):
            _validated_projection_error_identity_tuple(
                getattr(verdict, category_name), f"{prefix}.{category_name}"
            )

    _validate_projection_error_assessment_consistency(
        limit_results, overall, report.metric_verdicts
    )
    actual_categories = {}
    for category_name, identities in (
        ("passing", overall.passing_identities),
        ("failing", overall.failing_identities),
        ("undefined", overall.undefined_identities),
    ):
        actual_categories.update((identity, category_name) for identity in identities)
    for result, state in zip(limit_results, states):
        identity = CampaignProjectionErrorMetricFieldIdentity(
            result.metric_name, result.difference_field
        )
        expected_category = (
            "undefined" if state == "undefined" else "passing" if result.passed else "failing"
        )
        if actual_categories[identity] != expected_category:
            raise ValueError("report verdict classification disagrees with limit results")
    return report


def _validated_projection_error_identity_tuple(identities, name):
    if type(identities) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    seen = set()
    for index, identity in enumerate(identities):
        prefix = f"{name}[{index}]"
        if not isinstance(identity, CampaignProjectionErrorMetricFieldIdentity):
            raise TypeError(
                f"{prefix} must be a CampaignProjectionErrorMetricFieldIdentity"
            )
        for field_name in ("metric_name", "difference_field"):
            value = getattr(identity, field_name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{prefix}.{field_name} must be non-empty")
        if identity in seen:
            raise ValueError(f"{name} has duplicate identity {identity!r}")
        seen.add(identity)
    return identities


def _validate_projection_error_assessment_consistency(
    limit_results, overall_verdict, metric_verdicts
):
    source_identities = tuple(
        CampaignProjectionErrorMetricFieldIdentity(
            result.metric_name, result.difference_field
        )
        for result in limit_results
    )
    global_categories = {}
    for category_name, identities in (
        ("passing", overall_verdict.passing_identities),
        ("failing", overall_verdict.failing_identities),
        ("undefined", overall_verdict.undefined_identities),
    ):
        for identity in identities:
            if identity in global_categories:
                raise ValueError("overall verdict categories are not mutually exclusive")
            global_categories[identity] = category_name
    if set(global_categories) != set(source_identities):
        raise ValueError("overall verdict identities do not match limit results")
    source_positions = {identity: index for index, identity in enumerate(source_identities)}
    for identities in (
        overall_verdict.passing_identities,
        overall_verdict.failing_identities,
        overall_verdict.undefined_identities,
    ):
        if tuple(source_positions[identity] for identity in identities) != tuple(
            sorted(source_positions[identity] for identity in identities)
        ):
            raise ValueError("overall verdict identity order is inconsistent")

    field_count = len(_PROJECTION_ERROR_COMPARISON_DIFFERENCE_FIELDS)
    expected_metric_names = tuple(
        limit_results[index].metric_name
        for index in range(0, len(limit_results), field_count)
    )
    if tuple(verdict.metric_name for verdict in metric_verdicts) != expected_metric_names:
        raise ValueError("per-metric verdict order does not match limit results")

    per_metric_categories = {}
    for verdict in metric_verdicts:
        for category_name, identities in (
            ("passing", verdict.passing_identities),
            ("failing", verdict.failing_identities),
            ("undefined", verdict.undefined_identities),
        ):
            for identity in identities:
                if identity.metric_name != verdict.metric_name:
                    raise ValueError("per-metric verdict has inconsistent identity")
                if identity in per_metric_categories:
                    raise ValueError(
                        "per-metric verdict categories are not mutually exclusive"
                    )
                per_metric_categories[identity] = category_name
            if tuple(source_positions[identity] for identity in identities) != tuple(
                sorted(source_positions[identity] for identity in identities)
            ):
                raise ValueError("per-metric verdict identity order is inconsistent")
        expected_pass = bool(verdict.passing_identities) and not (
            verdict.failing_identities or verdict.undefined_identities
        )
        if verdict.overall_passed is not expected_pass:
            raise ValueError("per-metric verdict pass state is inconsistent")
    if set(per_metric_categories) != set(source_identities):
        raise ValueError("per-metric verdict identities do not match limit results")
    if per_metric_categories != global_categories:
        raise ValueError("overall and per-metric verdict classifications disagree")
    expected_overall_pass = bool(limit_results) and all(
        verdict.overall_passed for verdict in metric_verdicts
    )
    if overall_verdict.overall_passed is not expected_overall_pass:
        raise ValueError("overall verdict pass state disagrees with metric verdicts")


def _copy_projection_error_difference_limit_result(result):
    return CampaignProjectionErrorSummaryDifferenceLimitResult(
        metric_name=result.metric_name,
        difference_field=result.difference_field,
        observed_minimum_difference=result.observed_minimum_difference,
        observed_maximum_difference=result.observed_maximum_difference,
        allowable_minimum_difference=result.allowable_minimum_difference,
        allowable_maximum_difference=result.allowable_maximum_difference,
        lower_margin=result.lower_margin,
        upper_margin=result.upper_margin,
        passed=result.passed,
    )


def _validated_projection_error_difference_limit_results(limit_results):
    try:
        result_iterator = iter(limit_results)
    except TypeError as error:
        raise TypeError("limit_results must be an iterable") from error
    limit_results = tuple(result_iterator)
    field_count = len(_PROJECTION_ERROR_COMPARISON_DIFFERENCE_FIELDS)
    if len(limit_results) % field_count:
        raise ValueError("limit_results must contain complete difference-field layouts")

    metric_names = set()
    states = []
    for block_start in range(0, len(limit_results), field_count):
        block = limit_results[block_start : block_start + field_count]
        metric_name = None
        for offset, (result, expected_field) in enumerate(
            zip(block, _PROJECTION_ERROR_COMPARISON_DIFFERENCE_FIELDS)
        ):
            index = block_start + offset
            prefix = f"limit_results[{index}]"
            if not isinstance(
                result, CampaignProjectionErrorSummaryDifferenceLimitResult
            ):
                raise TypeError(
                    f"{prefix} must be a "
                    "CampaignProjectionErrorSummaryDifferenceLimitResult"
                )
            if type(result.metric_name) is not str or not result.metric_name.strip():
                raise ValueError(f"{prefix}.metric_name must be non-empty")
            if metric_name is None:
                metric_name = result.metric_name
                if metric_name in metric_names:
                    raise ValueError(f"duplicate limit-result metric {metric_name!r}")
                metric_names.add(metric_name)
            elif result.metric_name != metric_name:
                raise ValueError(f"{prefix} has inconsistent metric ordering")
            if result.difference_field != expected_field:
                raise ValueError(
                    f"{prefix}.difference_field must be {expected_field!r}"
                )
            if type(result.passed) is not bool:
                raise ValueError(f"{prefix}.passed must be a boolean")
            lower = _finite_numeric(
                f"{prefix}.allowable_minimum_difference",
                result.allowable_minimum_difference,
            )
            upper = _finite_numeric(
                f"{prefix}.allowable_maximum_difference",
                result.allowable_maximum_difference,
            )
            if lower > upper:
                raise ValueError(
                    f"{prefix}.allowable_minimum_difference must not exceed "
                    "allowable_maximum_difference"
                )

            observed_undefined = (
                result.observed_minimum_difference is None
                and result.observed_maximum_difference is None
            )
            margins_undefined = (
                result.lower_margin is None and result.upper_margin is None
            )
            if (
                (result.observed_minimum_difference is None)
                != (result.observed_maximum_difference is None)
                or (result.lower_margin is None) != (result.upper_margin is None)
                or observed_undefined != margins_undefined
            ):
                raise ValueError(f"{prefix} has inconsistent optional state")
            if observed_undefined:
                if result.passed:
                    raise ValueError(f"{prefix} undefined result cannot pass")
                states.append("undefined")
                continue

            minimum = _finite_numeric(
                f"{prefix}.observed_minimum_difference",
                result.observed_minimum_difference,
            )
            maximum = _finite_numeric(
                f"{prefix}.observed_maximum_difference",
                result.observed_maximum_difference,
            )
            if minimum > maximum:
                raise ValueError(
                    f"{prefix}.observed_minimum_difference must not exceed "
                    "observed_maximum_difference"
                )
            lower_margin = _finite_numeric(
                f"{prefix}.lower_margin", result.lower_margin
            )
            upper_margin = _finite_numeric(
                f"{prefix}.upper_margin", result.upper_margin
            )
            expected_lower_margin = minimum - lower
            expected_upper_margin = upper - maximum
            if not math.isfinite(expected_lower_margin) or not math.isfinite(
                expected_upper_margin
            ):
                raise ValueError(f"{prefix} expected margins must be finite")
            if (
                lower_margin != expected_lower_margin
                or upper_margin != expected_upper_margin
            ):
                raise ValueError(f"{prefix} margins are inconsistent")
            expected_pass = lower_margin >= 0.0 and upper_margin >= 0.0
            if result.passed is not expected_pass:
                raise ValueError(f"{prefix} pass state is inconsistent")
            states.append("defined")
    return limit_results, tuple(states)


def _validated_projection_error_difference_envelopes(envelopes):
    try:
        envelope_iterator = iter(envelopes)
    except TypeError as error:
        raise TypeError("envelopes must be an iterable") from error
    envelopes = tuple(envelope_iterator)
    field_count = len(_PROJECTION_ERROR_COMPARISON_DIFFERENCE_FIELDS)
    if len(envelopes) % field_count:
        raise ValueError("envelopes must contain complete difference-field layouts")

    metric_names = set()
    for block_start in range(0, len(envelopes), field_count):
        block = envelopes[block_start : block_start + field_count]
        metric_name = None
        for offset, (envelope, expected_field) in enumerate(
            zip(block, _PROJECTION_ERROR_COMPARISON_DIFFERENCE_FIELDS)
        ):
            index = block_start + offset
            prefix = f"envelopes[{index}]"
            if not isinstance(
                envelope, CampaignProjectionErrorSummaryDifferenceEnvelope
            ):
                raise TypeError(
                    f"{prefix} must be a "
                    "CampaignProjectionErrorSummaryDifferenceEnvelope"
                )
            if type(envelope.metric_name) is not str or not envelope.metric_name.strip():
                raise ValueError(f"{prefix}.metric_name must be non-empty")
            if metric_name is None:
                metric_name = envelope.metric_name
                if metric_name in metric_names:
                    raise ValueError(f"duplicate envelope metric {metric_name!r}")
                metric_names.add(metric_name)
            elif envelope.metric_name != metric_name:
                raise ValueError(f"{prefix} has malformed metric layout")
            if envelope.difference_field != expected_field:
                raise ValueError(
                    f"{prefix}.difference_field must be {expected_field!r}"
                )

            values_undefined = (
                envelope.minimum_difference is None
                and envelope.maximum_difference is None
            )
            names_undefined = (
                envelope.minimum_comparison_collection_name is None
                and envelope.maximum_comparison_collection_name is None
            )
            if (
                (envelope.minimum_difference is None)
                != (envelope.maximum_difference is None)
                or (envelope.minimum_comparison_collection_name is None)
                != (envelope.maximum_comparison_collection_name is None)
                or values_undefined != names_undefined
            ):
                raise ValueError(f"{prefix} has inconsistent optional state")
            if values_undefined:
                continue
            minimum = _finite_numeric(
                f"{prefix}.minimum_difference", envelope.minimum_difference
            )
            maximum = _finite_numeric(
                f"{prefix}.maximum_difference", envelope.maximum_difference
            )
            if minimum > maximum:
                raise ValueError(
                    f"{prefix}.minimum_difference must not exceed maximum_difference"
                )
            for name_field in (
                "minimum_comparison_collection_name",
                "maximum_comparison_collection_name",
            ):
                name = getattr(envelope, name_field)
                if type(name) is not str or not name.strip():
                    raise ValueError(f"{prefix}.{name_field} must be non-empty")
    return envelopes


def _validated_projection_error_comparison_set_results(comparison_set_results):
    try:
        result_iterator = iter(comparison_set_results)
    except TypeError as error:
        raise TypeError("comparison_set_results must be an iterable") from error
    comparison_set_results = tuple(result_iterator)

    baseline_name = None
    comparison_names = set()
    expected_metric_names = None
    expected_baseline_summaries = None
    optional_source_fields = {
        "minimum_residual_difference": "minimum_residual",
        "maximum_residual_difference": "maximum_residual",
        "mean_residual_difference": "mean_residual",
        "mean_absolute_residual_difference": "mean_absolute_residual",
        "maximum_absolute_residual_difference": "maximum_absolute_residual",
    }
    for result_index, result in enumerate(comparison_set_results):
        prefix = f"comparison_set_results[{result_index}]"
        if not isinstance(result, CampaignProjectionErrorSummaryComparisonSetResult):
            raise TypeError(
                f"{prefix} must be a "
                "CampaignProjectionErrorSummaryComparisonSetResult"
            )
        for field_name in (
            "baseline_collection_name",
            "comparison_collection_name",
        ):
            value = getattr(result, field_name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{prefix}.{field_name} must be non-empty")
        if result.baseline_collection_name == result.comparison_collection_name:
            raise ValueError(f"{prefix} collection names must be distinct")
        if baseline_name is None:
            baseline_name = result.baseline_collection_name
        elif result.baseline_collection_name != baseline_name:
            raise ValueError("comparison set results have inconsistent baseline identities")
        if result.comparison_collection_name in comparison_names:
            raise ValueError(
                "duplicate comparison collection name "
                f"{result.comparison_collection_name!r}"
            )
        comparison_names.add(result.comparison_collection_name)
        if type(result.comparisons) is not tuple:
            raise TypeError(f"{prefix}.comparisons must be a tuple")

        metric_names = []
        baseline_summaries = []
        for metric_index, comparison in enumerate(result.comparisons):
            item_prefix = f"{prefix}.comparisons[{metric_index}]"
            if not isinstance(
                comparison, CampaignMetricProjectionErrorSummaryComparison
            ):
                raise TypeError(
                    f"{item_prefix} must be a "
                    "CampaignMetricProjectionErrorSummaryComparison"
                )
            if (
                comparison.left_collection_name != result.baseline_collection_name
                or comparison.right_collection_name
                != result.comparison_collection_name
            ):
                raise ValueError(f"{item_prefix} has inconsistent collection identities")
            _validated_projection_error_summaries(
                (comparison.left_summary,), f"{item_prefix}.left_summary"
            )
            _validated_projection_error_summaries(
                (comparison.right_summary,), f"{item_prefix}.right_summary"
            )
            if (
                comparison.metric_name != comparison.left_summary.metric_name
                or comparison.metric_name != comparison.right_summary.metric_name
            ):
                raise ValueError(f"{item_prefix} has inconsistent metric identities")
            metric_names.append(comparison.metric_name)
            baseline_summaries.append(comparison.left_summary)
            for field_name in _PROJECTION_ERROR_COMPARISON_DIFFERENCE_FIELDS:
                value = getattr(comparison, field_name)
                if field_name in optional_source_fields:
                    source_field = optional_source_fields[field_name]
                    should_be_none = (
                        getattr(comparison.left_summary, source_field) is None
                        or getattr(comparison.right_summary, source_field) is None
                    )
                    if (value is None) != should_be_none:
                        raise ValueError(f"{item_prefix}.{field_name} has invalid optional state")
                    if value is None:
                        continue
                _finite_numeric(f"{item_prefix}.{field_name}", value)

        metric_names = tuple(metric_names)
        baseline_summaries = tuple(baseline_summaries)
        if len(set(metric_names)) != len(metric_names):
            raise ValueError(f"{prefix} has duplicate metric names")
        if expected_metric_names is None:
            expected_metric_names = metric_names
            expected_baseline_summaries = baseline_summaries
        elif metric_names != expected_metric_names:
            raise ValueError("comparison set results have incompatible metric layouts")
        elif baseline_summaries != expected_baseline_summaries:
            raise ValueError("comparison set results have inconsistent baseline summaries")
    return comparison_set_results, (() if expected_metric_names is None else expected_metric_names)


def _validated_projection_error_summaries(summaries, name):
    try:
        summary_iterator = iter(summaries)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable") from error
    summaries = tuple(summary_iterator)
    metric_names = set()
    validation_case_count = None
    optional_fields = (
        "minimum_residual",
        "maximum_residual",
        "mean_residual",
        "mean_absolute_residual",
        "maximum_absolute_residual",
    )
    for index, summary in enumerate(summaries):
        prefix = f"{name}[{index}]"
        if not isinstance(summary, CampaignMetricProjectionErrorSummary):
            raise TypeError(f"{prefix} must be a CampaignMetricProjectionErrorSummary")
        if type(summary.metric_name) is not str or not summary.metric_name.strip():
            raise ValueError(f"{prefix}.metric_name must be non-empty")
        if summary.metric_name in metric_names:
            raise ValueError(f"{name} has duplicate metric {summary.metric_name!r}")
        metric_names.add(summary.metric_name)
        for field_name in (
            "validation_case_count",
            "defined_residual_count",
            "undefined_residual_count",
        ):
            value = getattr(summary, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{prefix}.{field_name} must be a nonnegative integer")
        if summary.validation_case_count == 0:
            raise ValueError(f"{prefix}.validation_case_count must be positive")
        if (
            summary.defined_residual_count + summary.undefined_residual_count
            != summary.validation_case_count
        ):
            raise ValueError(f"{prefix} has inconsistent residual counts")
        if validation_case_count is None:
            validation_case_count = summary.validation_case_count
        elif summary.validation_case_count != validation_case_count:
            raise ValueError(f"{name} has inconsistent validation case counts")

        values = tuple(getattr(summary, field_name) for field_name in optional_fields)
        if summary.defined_residual_count == 0:
            if any(value is not None for value in values):
                raise ValueError(f"{prefix} has inconsistent undefined summaries")
            continue
        if any(value is None for value in values):
            raise ValueError(f"{prefix} has incomplete defined summaries")
        checked = {
            field_name: _finite_numeric(f"{prefix}.{field_name}", value)
            for field_name, value in zip(optional_fields, values)
        }
        if checked["minimum_residual"] > checked["maximum_residual"]:
            raise ValueError(f"{prefix} minimum residual must not exceed maximum")
        if not (
            checked["minimum_residual"]
            <= checked["mean_residual"]
            <= checked["maximum_residual"]
        ):
            raise ValueError(f"{prefix} mean residual must lie within its extrema")
        if (
            checked["mean_absolute_residual"] < 0.0
            or checked["maximum_absolute_residual"] < 0.0
            or checked["mean_absolute_residual"] > checked["maximum_absolute_residual"]
            or checked["maximum_absolute_residual"]
            != max(
                abs(checked["minimum_residual"]),
                abs(checked["maximum_residual"]),
            )
        ):
            raise ValueError(f"{prefix} has inconsistent absolute residual summaries")
    return summaries


def _copy_projection_error_summary(summary):
    return CampaignMetricProjectionErrorSummary(
        metric_name=summary.metric_name,
        validation_case_count=summary.validation_case_count,
        defined_residual_count=summary.defined_residual_count,
        undefined_residual_count=summary.undefined_residual_count,
        minimum_residual=summary.minimum_residual,
        maximum_residual=summary.maximum_residual,
        mean_residual=summary.mean_residual,
        mean_absolute_residual=summary.mean_absolute_residual,
        maximum_absolute_residual=summary.maximum_absolute_residual,
    )


def _validated_projection_validation_results(validation_results):
    try:
        result_iterator = iter(validation_results)
    except TypeError as error:
        raise TypeError("validation_results must be an iterable") from error
    validation_results = tuple(result_iterator)

    names = set()
    states = []
    for index, result in enumerate(validation_results):
        if not isinstance(result, CampaignProjectionValidationResult):
            raise TypeError(
                f"validation_results[{index}] must be a "
                "CampaignProjectionValidationResult"
            )
        if type(result.name) is not str or not result.name.strip():
            raise ValueError(f"validation_results[{index}].name must be non-empty")
        if result.name in names:
            raise ValueError(f"duplicate validation case name {result.name!r}")
        names.add(result.name)
        states.append(_validated_projection_validation_result(result, index))
    return validation_results, tuple(states)


def _validated_projection_validation_result(result, index):
    for field_name in ("scenario_name", "observed_run_id"):
        value = getattr(result, field_name)
        if type(value) is not str or not value.strip():
            raise ValueError(
                f"validation_results[{index}].{field_name} must be non-empty"
            )

    residual_metric_names, residual_values = _validated_projection_residuals(
        result.residuals
    )
    tolerance_results = result.tolerance_results
    if not isinstance(tolerance_results, CampaignProjectionResidualToleranceResults):
        raise TypeError(
            f"validation_results[{index}].tolerance_results must be a "
            "CampaignProjectionResidualToleranceResults"
        )
    identities = (
        result.scenario_name,
        result.observed_run_id,
        result.residuals.scenario_name,
        result.residuals.observed_run_id,
        tolerance_results.scenario_name,
        tolerance_results.observed_run_id,
    )
    if identities != (
        result.scenario_name,
        result.observed_run_id,
        result.scenario_name,
        result.observed_run_id,
        result.scenario_name,
        result.observed_run_id,
    ):
        raise ValueError(
            f"validation_results[{index}] has inconsistent scenario/run metadata"
        )
    if type(tolerance_results.metric_results) is not tuple:
        raise TypeError(
            f"validation_results[{index}].tolerance_results.metric_results "
            "must be a tuple"
        )
    if not tolerance_results.metric_results:
        raise ValueError(f"validation_results[{index}] must contain metric results")
    if len(tolerance_results.metric_results) != len(residual_metric_names):
        raise ValueError(
            f"validation_results[{index}] metric layouts must match exactly"
        )

    has_failure = False
    has_undefined = False
    metric_names = set()
    for metric_index, (expected_name, residual, metric_result) in enumerate(
        zip(
            residual_metric_names,
            residual_values,
            tolerance_results.metric_results,
        )
    ):
        prefix = (
            f"validation_results[{index}].tolerance_results."
            f"metric_results[{metric_index}]"
        )
        if not isinstance(metric_result, CampaignMetricResidualToleranceResult):
            raise TypeError(f"{prefix} must be a CampaignMetricResidualToleranceResult")
        if (
            type(metric_result.metric_name) is not str
            or not metric_result.metric_name.strip()
        ):
            raise ValueError(f"{prefix}.metric_name must be non-empty")
        if metric_result.metric_name in metric_names:
            raise ValueError(f"duplicate metric name {metric_result.metric_name!r}")
        metric_names.add(metric_result.metric_name)
        if metric_result.metric_name != expected_name:
            raise ValueError(
                f"validation_results[{index}] metric layouts must match exactly"
            )
        maximum = _finite_numeric(
            f"{prefix}.maximum_absolute_residual",
            metric_result.maximum_absolute_residual,
        )
        if maximum < 0.0:
            raise ValueError(f"{prefix}.maximum_absolute_residual must be nonnegative")
        if type(metric_result.passed) is not bool:
            raise TypeError(f"{prefix}.passed must be a boolean")

        if residual is None:
            if (
                metric_result.residual is not None
                or metric_result.absolute_residual is not None
                or metric_result.margin is not None
                or metric_result.passed
            ):
                raise ValueError(f"{prefix} has inconsistent undefined state")
            has_undefined = True
            continue

        stored_residual = _finite_numeric(f"{prefix}.residual", metric_result.residual)
        absolute_residual = _finite_numeric(
            f"{prefix}.absolute_residual", metric_result.absolute_residual
        )
        margin = _finite_numeric(f"{prefix}.margin", metric_result.margin)
        if (
            stored_residual != residual
            or absolute_residual != abs(residual)
            or margin != maximum - absolute_residual
            or metric_result.passed != (margin >= 0.0)
        ):
            raise ValueError(f"{prefix} has inconsistent defined state")
        has_failure = has_failure or not metric_result.passed

    if has_undefined:
        return "undefined"
    if has_failure:
        return "failing"
    return "passing"


def project_campaign_scenarios(
    matrix: CampaignSensitivityMatrix,
    scenarios: Iterable[CampaignProjectionScenario],
) -> tuple[CampaignProjectionScenarioResult, ...]:
    """Apply one secant matrix to explicit named change scenarios in order."""
    _validate_sensitivity_matrix(matrix)
    try:
        scenario_iterator = iter(scenarios)
    except TypeError as error:
        raise TypeError("scenarios must be an iterable") from error
    scenarios = tuple(scenario_iterator)

    names = set()
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, CampaignProjectionScenario):
            raise TypeError(f"scenarios[{index}] must be a CampaignProjectionScenario")
        if type(scenario.name) is not str or not scenario.name.strip():
            raise ValueError(f"scenarios[{index}].name must be non-empty")
        if scenario.name in names:
            raise ValueError(f"duplicate scenario name {scenario.name!r}")
        names.add(scenario.name)
        if type(scenario.parameter_changes) is not tuple:
            raise TypeError(f"scenarios[{index}].parameter_changes must be a tuple")
        _validated_parameter_changes(matrix, scenario.parameter_changes)

    return tuple(
        CampaignProjectionScenarioResult(
            name=scenario.name,
            projection=project_campaign_metric_changes(
                matrix, scenario.parameter_changes
            ),
        )
        for scenario in scenarios
    )


def project_campaign_metric_changes(
    matrix: CampaignSensitivityMatrix,
    parameter_changes: Iterable[CampaignParameterChange],
) -> CampaignMetricChangeProjection:
    """Apply one explicit parameter-change vector to a secant matrix."""
    _validate_sensitivity_matrix(matrix)
    values = _validated_parameter_changes(matrix, parameter_changes)

    predicted = []
    for metric_name, row in zip(matrix.metric_names, matrix.values):
        if any(sensitivity is None for sensitivity in row):
            predicted.append(None)
            continue
        metric_change = 0.0
        for sensitivity, change in zip(row, values):
            term = float(sensitivity) * change
            if not math.isfinite(term):
                raise ValueError(
                    f"predicted metric {metric_name!r} contribution must be finite"
                )
            metric_change += term
            if not math.isfinite(metric_change):
                raise ValueError(
                    f"predicted metric change {metric_name!r} must be finite"
                )
        predicted.append(metric_change)

    return CampaignMetricChangeProjection(
        parameter_names=matrix.parameter_names,
        metric_names=matrix.metric_names,
        parameter_changes=values,
        predicted_metric_changes=tuple(predicted),
    )


def _validated_parameter_changes(matrix, parameter_changes):
    try:
        change_iterator = iter(parameter_changes)
    except TypeError as error:
        raise TypeError("parameter_changes must be an iterable") from error
    parameter_changes = tuple(change_iterator)
    if len(parameter_changes) != len(matrix.parameter_names):
        raise ValueError(
            "parameter_changes must match the sensitivity matrix column count"
        )

    values = []
    for index, (expected_name, change) in enumerate(
        zip(matrix.parameter_names, parameter_changes)
    ):
        if not isinstance(change, CampaignParameterChange):
            raise TypeError(
                f"parameter_changes[{index}] must be a CampaignParameterChange"
            )
        if change.name != expected_name:
            raise ValueError(
                f"parameter_changes[{index}].name must be {expected_name!r}"
            )
        values.append(
            _finite_numeric(f"parameter_changes[{index}].value", change.value)
        )
    return tuple(values)


def _validate_sensitivity_matrix(matrix):
    if not isinstance(matrix, CampaignSensitivityMatrix):
        raise TypeError("matrix must be a CampaignSensitivityMatrix")
    for name, value in (
        ("parameter_names", matrix.parameter_names),
        ("metric_names", matrix.metric_names),
        ("representative_run_ids", matrix.representative_run_ids),
        ("values", matrix.values),
    ):
        if type(value) is not tuple:
            raise TypeError(f"matrix.{name} must be a tuple")
    if len(matrix.representative_run_ids) != len(matrix.parameter_names):
        raise ValueError("matrix representative IDs must match parameter columns")
    if len(matrix.values) != len(matrix.metric_names):
        raise ValueError("matrix value rows must match metric names")

    _validate_unique_nonblank(matrix.parameter_names, "matrix parameter name")
    _validate_unique_nonblank(matrix.metric_names, "matrix metric name")
    _validate_unique_nonblank(
        matrix.representative_run_ids, "matrix representative run_id"
    )
    for metric_name in matrix.metric_names:
        if metric_name not in _METRIC_KEYS:
            raise ValueError(f"matrix has unknown metric {metric_name!r}")
    for row_index, row in enumerate(matrix.values):
        if type(row) is not tuple:
            raise TypeError(f"matrix.values[{row_index}] must be a tuple")
        if len(row) != len(matrix.parameter_names):
            raise ValueError(f"matrix.values[{row_index}] must match parameter columns")
        for column_index, sensitivity in enumerate(row):
            if sensitivity is not None:
                _finite_numeric(
                    f"matrix.values[{row_index}][{column_index}]", sensitivity
                )


def _validate_unique_nonblank(values, name):
    seen = set()
    for index, value in enumerate(values):
        if type(value) is not str or not value.strip():
            raise ValueError(f"{name} at index {index} must be non-empty")
        if value in seen:
            raise ValueError(f"duplicate {name} {value!r}")
        seen.add(value)


def campaign_sensitivity_matrix(
    parameters: Iterable[SensitivityMatrixParameter],
) -> CampaignSensitivityMatrix:
    """Assemble explicit representative secants into a metric-by-parameter matrix."""
    try:
        parameter_iterator = iter(parameters)
    except TypeError as error:
        raise TypeError("parameters must be an iterable") from error
    parameters = tuple(parameter_iterator)

    names = []
    representative_ids = []
    selected_entries = []
    expected_metric_names = None
    for index, parameter in enumerate(parameters):
        if not isinstance(parameter, SensitivityMatrixParameter):
            raise TypeError(f"parameters[{index}] must be a SensitivityMatrixParameter")
        if type(parameter.name) is not str or not parameter.name.strip():
            raise ValueError(f"parameters[{index}].name must be non-empty")
        if parameter.name in names:
            raise ValueError(f"duplicate parameter name {parameter.name!r}")
        names.append(parameter.name)
        if (
            type(parameter.representative_run_id) is not str
            or not parameter.representative_run_id.strip()
        ):
            raise ValueError(
                f"parameters[{index}].representative_run_id must be non-empty"
            )
        if parameter.representative_run_id in representative_ids:
            raise ValueError(
                f"duplicate representative run_id {parameter.representative_run_id!r}"
            )
        representative_ids.append(parameter.representative_run_id)
        metric_names, entries_by_id = _validated_sensitivity_entries(
            parameter.sensitivities,
            f"parameters[{index}].sensitivities",
        )
        if expected_metric_names is None:
            expected_metric_names = metric_names
        elif metric_names != expected_metric_names:
            raise ValueError(
                "parameter sensitivities must have matching metric layouts"
            )
        if parameter.representative_run_id not in entries_by_id:
            raise ValueError(
                f"representative run_id {parameter.representative_run_id!r} "
                f"is not in parameters[{index}].sensitivities"
            )
        selected = entries_by_id[parameter.representative_run_id]
        if selected.parameter_delta == 0.0:
            raise ValueError(
                f"representative run_id {parameter.representative_run_id!r} "
                "must have a nonzero parameter delta"
            )
        selected_entries.append(selected)

    if not parameters:
        return CampaignSensitivityMatrix((), (), (), ())

    rows = []
    for metric_index, metric_name in enumerate(expected_metric_names):
        rows.append(
            tuple(
                entry.metric_sensitivities[metric_index][1]
                for entry in selected_entries
            )
        )
    return CampaignSensitivityMatrix(
        parameter_names=tuple(names),
        metric_names=expected_metric_names,
        representative_run_ids=tuple(representative_ids),
        values=tuple(rows),
    )


def _validated_sensitivity_entries(sensitivities, name):
    if type(sensitivities) is not tuple:
        raise TypeError(f"{name} must be a tuple")
    if not sensitivities:
        raise ValueError(f"{name} must not be empty")
    entries_by_id = {}
    expected_metric_names = None
    has_baseline = False
    for index, entry in enumerate(sensitivities):
        if not isinstance(entry, CampaignSensitivityEntry):
            raise TypeError(f"{name}[{index}] must be a CampaignSensitivityEntry")
        if type(entry.run_id) is not str or not entry.run_id.strip():
            raise ValueError(f"{name}[{index}].run_id must be non-empty")
        if entry.run_id in entries_by_id:
            raise ValueError(f"duplicate run_id {entry.run_id!r} in {name}")
        parameter_delta = _finite_numeric(
            f"{name}[{index}].parameter_delta", entry.parameter_delta
        )
        if type(entry.metric_sensitivities) is not tuple:
            raise TypeError(f"{name}[{index}].metric_sensitivities must be a tuple")
        if not entry.metric_sensitivities:
            raise ValueError(f"{name}[{index}].metric_sensitivities must not be empty")

        metric_names = []
        for metric_index, metric_item in enumerate(entry.metric_sensitivities):
            if type(metric_item) is not tuple or len(metric_item) != 2:
                raise ValueError(
                    f"{name}[{index}].metric_sensitivities[{metric_index}] "
                    "must be a name/value pair"
                )
            metric_name, sensitivity = metric_item
            if type(metric_name) is not str or not metric_name:
                raise ValueError(f"{name}[{index}] metric name must be non-empty")
            if metric_name not in _METRIC_KEYS:
                raise ValueError(f"{name}[{index}] has unknown metric {metric_name!r}")
            if metric_name in metric_names:
                raise ValueError(
                    f"{name}[{index}] has duplicate metric {metric_name!r}"
                )
            metric_names.append(metric_name)
            if sensitivity is not None:
                _finite_numeric(f"{name}[{index}] metric {metric_name!r}", sensitivity)
                if parameter_delta == 0.0:
                    raise ValueError(
                        f"{name}[{index}] zero parameter delta must have "
                        "undefined sensitivities"
                    )
        metric_names = tuple(metric_names)
        if expected_metric_names is None:
            expected_metric_names = metric_names
        elif metric_names != expected_metric_names:
            raise ValueError(f"{name} entries must have matching metric layouts")
        entries_by_id[entry.run_id] = entry
        has_baseline = has_baseline or parameter_delta == 0.0
    if not has_baseline:
        raise ValueError(f"{name} must contain a zero-delta baseline entry")
    return expected_metric_names, entries_by_id


def campaign_secant_sensitivities(
    deltas: Iterable[CampaignDeltaEntry],
) -> tuple[CampaignSensitivityEntry, ...]:
    """Compute ordered baseline-relative metric-delta/parameter-delta ratios."""
    try:
        delta_iterator = iter(deltas)
    except TypeError as error:
        raise TypeError("deltas must be an iterable of entries") from error
    deltas = tuple(delta_iterator)
    _validate_delta_entries(deltas)

    sensitivities = []
    for entry in deltas:
        metric_sensitivities = []
        for metric_name, metric_delta in entry.metric_deltas:
            if entry.parameter_delta == 0.0 or metric_delta is None:
                sensitivity = None
            else:
                sensitivity = metric_delta / entry.parameter_delta
                if not math.isfinite(sensitivity):
                    raise ValueError(
                        f"run_id {entry.run_id!r} metric {metric_name!r} "
                        "sensitivity must be finite"
                    )
            metric_sensitivities.append((metric_name, sensitivity))
        sensitivities.append(
            CampaignSensitivityEntry(
                run_id=entry.run_id,
                parameter_delta=float(entry.parameter_delta),
                metric_sensitivities=tuple(metric_sensitivities),
            )
        )
    return tuple(sensitivities)


def _validate_delta_entries(deltas):
    run_ids = set()
    expected_metric_names = None
    has_baseline = False
    for index, entry in enumerate(deltas):
        if not isinstance(entry, CampaignDeltaEntry):
            raise TypeError(f"deltas[{index}] must be a CampaignDeltaEntry")
        if type(entry.run_id) is not str or not entry.run_id.strip():
            raise ValueError(f"deltas[{index}].run_id must be non-empty")
        if entry.run_id in run_ids:
            raise ValueError(f"duplicate run_id {entry.run_id!r} in deltas")
        run_ids.add(entry.run_id)
        parameter_delta = _finite_numeric(
            f"deltas[{index}].parameter_delta", entry.parameter_delta
        )
        if type(entry.metric_deltas) is not tuple:
            raise TypeError(f"deltas[{index}].metric_deltas must be a tuple")
        if not entry.metric_deltas:
            raise ValueError(f"deltas[{index}].metric_deltas must not be empty")

        metric_names = []
        baseline_candidate = parameter_delta == 0.0
        for metric_index, metric_item in enumerate(entry.metric_deltas):
            if type(metric_item) is not tuple or len(metric_item) != 2:
                raise ValueError(
                    f"deltas[{index}].metric_deltas[{metric_index}] "
                    "must be a name/value pair"
                )
            metric_name, metric_delta = metric_item
            if type(metric_name) is not str or not metric_name:
                raise ValueError(
                    f"deltas[{index}] metric name must be a non-empty string"
                )
            if metric_name not in _METRIC_KEYS:
                raise ValueError(f"deltas[{index}] has unknown metric {metric_name!r}")
            if metric_name in metric_names:
                raise ValueError(
                    f"deltas[{index}] has duplicate metric {metric_name!r}"
                )
            metric_names.append(metric_name)
            if metric_delta is not None:
                metric_delta = _finite_numeric(
                    f"deltas[{index}] metric {metric_name!r}", metric_delta
                )
                if metric_delta != 0.0:
                    baseline_candidate = False
        metric_names = tuple(metric_names)
        if expected_metric_names is None:
            expected_metric_names = metric_names
        elif metric_names != expected_metric_names:
            raise ValueError("delta entries must have matching metric layouts")
        has_baseline = has_baseline or baseline_candidate

    if deltas and not has_baseline:
        raise ValueError("deltas must contain a zero-delta baseline entry")


def campaign_metric_deltas(
    comparison: Iterable[CampaignComparisonEntry],
    baseline_run_id: str,
) -> tuple[CampaignDeltaEntry, ...]:
    """Subtract one explicit baseline from an ordered campaign comparison."""
    if type(baseline_run_id) is not str or not baseline_run_id.strip():
        raise ValueError("baseline_run_id must be a non-empty string")
    try:
        comparison_iterator = iter(comparison)
    except TypeError as error:
        raise TypeError("comparison must be an iterable of entries") from error
    comparison = tuple(comparison_iterator)
    _validate_comparison_entries(comparison)

    matches = [entry for entry in comparison if entry.run_id == baseline_run_id]
    if not matches:
        raise ValueError(f"baseline run_id {baseline_run_id!r} is not in comparison")
    baseline = matches[0]
    baseline_metrics = dict(baseline.metric_values)

    deltas = []
    for entry in comparison:
        metric_deltas = []
        for metric_name, metric_value in entry.metric_values:
            baseline_value = baseline_metrics[metric_name]
            metric_deltas.append(
                (
                    metric_name,
                    None
                    if metric_value is None or baseline_value is None
                    else _finite_delta(
                        f"metric {metric_name!r}", metric_value, baseline_value
                    ),
                )
            )
        deltas.append(
            CampaignDeltaEntry(
                run_id=entry.run_id,
                parameter_delta=_finite_delta(
                    "parameter", entry.parameter_value, baseline.parameter_value
                ),
                metric_deltas=tuple(metric_deltas),
            )
        )
    return tuple(deltas)


def _validate_comparison_entries(comparison):
    run_ids = set()
    expected_metric_names = None
    for index, entry in enumerate(comparison):
        if not isinstance(entry, CampaignComparisonEntry):
            raise TypeError(f"comparison[{index}] must be a CampaignComparisonEntry")
        if type(entry.run_id) is not str or not entry.run_id.strip():
            raise ValueError(f"comparison[{index}].run_id must be non-empty")
        if entry.run_id in run_ids:
            raise ValueError(f"duplicate run_id {entry.run_id!r} in comparison")
        run_ids.add(entry.run_id)
        _finite_numeric(f"comparison[{index}].parameter_value", entry.parameter_value)
        if type(entry.metric_values) is not tuple:
            raise TypeError(f"comparison[{index}].metric_values must be a tuple")
        if not entry.metric_values:
            raise ValueError(f"comparison[{index}].metric_values must not be empty")

        metric_names = []
        for metric_index, metric_item in enumerate(entry.metric_values):
            if type(metric_item) is not tuple or len(metric_item) != 2:
                raise ValueError(
                    f"comparison[{index}].metric_values[{metric_index}] "
                    "must be a name/value pair"
                )
            metric_name, metric_value = metric_item
            if type(metric_name) is not str or not metric_name:
                raise ValueError(
                    f"comparison[{index}] metric name must be a non-empty string"
                )
            if metric_name not in _METRIC_KEYS:
                raise ValueError(
                    f"comparison[{index}] has unknown metric {metric_name!r}"
                )
            if metric_name in metric_names:
                raise ValueError(
                    f"comparison[{index}] has duplicate metric {metric_name!r}"
                )
            metric_names.append(metric_name)
            if metric_value is not None:
                _finite_numeric(
                    f"comparison[{index}] metric {metric_name!r}", metric_value
                )
        metric_names = tuple(metric_names)
        if expected_metric_names is None:
            expected_metric_names = metric_names
        elif metric_names != expected_metric_names:
            raise ValueError("comparison entries must have matching metric layouts")
    return () if expected_metric_names is None else expected_metric_names


def _finite_numeric(name, value):
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be numeric")
    try:
        numeric_value = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite")
    return numeric_value


def _finite_delta(name, value, baseline_value):
    value = _finite_numeric(name, value)
    baseline_value = _finite_numeric(f"baseline {name}", baseline_value)
    delta = value - baseline_value
    if not math.isfinite(delta):
        raise ValueError(f"{name} delta must be finite")
    return delta


def compare_campaign_runs(
    bundle_record,
    parameter_category,
    parameter_key,
    metric_names: Iterable[str],
) -> tuple[CampaignComparisonEntry, ...]:
    """Extract one explicit provenance parameter and ordered existing metrics."""
    manifest, records = _validated_bundle_record(bundle_record)
    if parameter_category not in _PARAMETER_CATEGORIES:
        raise ValueError(f"parameter_category must be one of {_PARAMETER_CATEGORIES!r}")
    if type(parameter_key) is not str or not parameter_key.strip():
        raise ValueError("parameter_key must be a non-empty string")
    metric_names = _validated_metric_names(metric_names)

    entries = []
    for index, (run_id, record) in enumerate(zip(manifest["run_ids"], records)):
        metadata = record[parameter_category]
        if parameter_key not in metadata:
            raise ValueError(
                f"records[{index}] run_id {run_id!r} is missing parameter "
                f"{parameter_category}.{parameter_key}"
            )
        parameter_value = metadata[parameter_key]
        if parameter_value is not None and type(parameter_value) not in (
            bool,
            int,
            float,
            str,
        ):
            raise ValueError(
                f"records[{index}] parameter {parameter_category}.{parameter_key} "
                "must be scalar"
            )
        if type(parameter_value) is float and not math.isfinite(parameter_value):
            raise ValueError(
                f"records[{index}] parameter {parameter_category}.{parameter_key} "
                "must be finite"
            )

        values = []
        for metric_name in metric_names:
            metric_value = record["metrics"][metric_name]
            if metric_value is not None and type(metric_value) not in (int, float):
                raise ValueError(
                    f"records[{index}] metric {metric_name!r} must be scalar"
                )
            values.append(
                (
                    metric_name,
                    None if metric_value is None else float(metric_value),
                )
            )
        entries.append(
            CampaignComparisonEntry(
                run_id=run_id,
                parameter_value=parameter_value,
                metric_values=tuple(values),
            )
        )
    return tuple(entries)


def _validated_metric_names(metric_names):
    try:
        metric_iterator = iter(metric_names)
    except TypeError as error:
        raise TypeError("metric_names must be an iterable of metric names") from error
    metric_names = tuple(metric_iterator)
    if not metric_names:
        raise ValueError("metric_names must contain at least one metric name")
    for index, metric_name in enumerate(metric_names):
        if type(metric_name) is not str or not metric_name:
            raise ValueError(f"metric_names[{index}] must be a non-empty string")
        if metric_name not in _METRIC_KEYS:
            raise ValueError(f"unknown metric name {metric_name!r}")
    if len(set(metric_names)) != len(metric_names):
        raise ValueError("metric_names must not contain duplicates")
    return metric_names


def _validated_bundle_record(bundle_record):
    if type(bundle_record) is not dict:
        raise TypeError("bundle_record must be a dictionary")
    if set(bundle_record) != {"manifest", "records"}:
        raise ValueError("bundle_record must contain exactly manifest and records")
    manifest = bundle_record["manifest"]
    records = bundle_record["records"]
    if type(manifest) is not dict or set(manifest) != {
        "campaign_id",
        "created_at",
        "run_ids",
    }:
        raise ValueError("bundle_record manifest has invalid structure")
    if type(manifest["campaign_id"]) is not str or not manifest["campaign_id"].strip():
        raise ValueError("bundle_record campaign_id must be a non-empty string")
    if type(manifest["created_at"]) is not str:
        raise ValueError("bundle_record created_at must be a string")
    try:
        created_at = datetime.fromisoformat(manifest["created_at"])
    except ValueError:
        raise ValueError(
            "bundle_record created_at must be an aware UTC timestamp"
        ) from None
    if created_at.utcoffset() != timedelta(0):
        raise ValueError("bundle_record created_at must be an aware UTC timestamp")
    if type(manifest["run_ids"]) is not list:
        raise TypeError("bundle_record manifest run_ids must be a list")
    if type(records) is not list:
        raise TypeError("bundle_record records must be a list")
    if len(manifest["run_ids"]) != len(records):
        raise ValueError("bundle_record run_ids and records must have matching lengths")
    for index, (run_id, record) in enumerate(zip(manifest["run_ids"], records)):
        if type(run_id) is not str or not run_id.strip():
            raise ValueError(f"bundle_record run_ids[{index}] must be non-empty")
        _validate_record(record)
        if record["run_id"] != run_id:
            raise ValueError(
                f"bundle_record records[{index}] does not match run_id {run_id!r}"
            )
    return manifest, records
