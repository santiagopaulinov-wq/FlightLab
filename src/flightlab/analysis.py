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
            raise TypeError(
                f"limits[{index}] must be a CampaignMetricProjectionLimit"
            )
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
        raise ValueError(f"envelopes[{index}] has unknown metric {envelope.metric_name!r}")

    values_undefined = envelope.minimum is None and envelope.maximum is None
    names_undefined = (
        envelope.minimum_scenario is None and envelope.maximum_scenario is None
    )
    if (
        (envelope.minimum is None) != (envelope.maximum is None)
        or (envelope.minimum_scenario is None)
        != (envelope.maximum_scenario is None)
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
                f"scenario_results[{index}] must be a "
                "CampaignProjectionScenarioResult"
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
            raise TypeError(
                f"scenarios[{index}] must be a CampaignProjectionScenario"
            )
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
            raise ValueError(
                f"matrix.values[{row_index}] must match parameter columns"
            )
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
            raise TypeError(
                f"parameters[{index}] must be a SensitivityMatrixParameter"
            )
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
                "duplicate representative run_id "
                f"{parameter.representative_run_id!r}"
            )
        representative_ids.append(parameter.representative_run_id)
        metric_names, entries_by_id = _validated_sensitivity_entries(
            parameter.sensitivities,
            f"parameters[{index}].sensitivities",
        )
        if expected_metric_names is None:
            expected_metric_names = metric_names
        elif metric_names != expected_metric_names:
            raise ValueError("parameter sensitivities must have matching metric layouts")
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
                _finite_numeric(
                    f"{name}[{index}] metric {metric_name!r}", sensitivity
                )
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
        raise ValueError(
            f"parameter_category must be one of {_PARAMETER_CATEGORIES!r}"
        )
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
        raise ValueError("bundle_record created_at must be an aware UTC timestamp") from None
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
