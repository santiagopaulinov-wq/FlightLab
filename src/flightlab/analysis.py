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
