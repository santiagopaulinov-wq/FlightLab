import json
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

from flightlab.experiment import ExperimentRun

_TOP_LEVEL_KEYS = (
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

_METRIC_KEYS = (
    "final_output",
    "final_reference",
    "steady_state_error",
    "peak_output",
    "maximum_absolute_tracking_error",
    "rms_tracking_error",
    "iae",
    "ise",
    "overshoot_percent",
    "settling_time",
    "settling_tolerance",
)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id TEXT PRIMARY KEY NOT NULL,
    created_at TEXT NOT NULL,
    method TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    duration REAL NOT NULL CHECK (duration > 0.0),
    sample_count INTEGER NOT NULL CHECK (sample_count >= 2),
    initial_state_json TEXT NOT NULL,
    system_json TEXT NOT NULL,
    controller_json TEXT NOT NULL,
    reference_json TEXT NOT NULL,
    user_metadata_json TEXT NOT NULL,
    final_output REAL NOT NULL,
    final_reference REAL NOT NULL,
    steady_state_error REAL NOT NULL,
    peak_output REAL NOT NULL,
    maximum_absolute_tracking_error REAL NOT NULL,
    rms_tracking_error REAL NOT NULL,
    iae REAL NOT NULL,
    ise REAL NOT NULL,
    overshoot_percent REAL,
    settling_time REAL,
    settling_tolerance REAL NOT NULL CHECK (settling_tolerance > 0.0)
)
"""

_CREATE_LIST_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS experiment_runs_created_at_idx
ON experiment_runs (created_at DESC, run_id ASC)
"""

_INSERT_SQL = """
INSERT INTO experiment_runs (
    run_id,
    created_at,
    method,
    start_time,
    end_time,
    duration,
    sample_count,
    initial_state_json,
    system_json,
    controller_json,
    reference_json,
    user_metadata_json,
    final_output,
    final_reference,
    steady_state_error,
    peak_output,
    maximum_absolute_tracking_error,
    rms_tracking_error,
    iae,
    ise,
    overshoot_percent,
    settling_time,
    settling_tolerance
) VALUES (
    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
)
"""

_SELECT_SQL = """
SELECT
    run_id,
    created_at,
    method,
    start_time,
    end_time,
    duration,
    sample_count,
    initial_state_json,
    system_json,
    controller_json,
    reference_json,
    user_metadata_json,
    final_output,
    final_reference,
    steady_state_error,
    peak_output,
    maximum_absolute_tracking_error,
    rms_tracking_error,
    iae,
    ise,
    overshoot_percent,
    settling_time,
    settling_tolerance
FROM experiment_runs
WHERE run_id = ?
"""

_LIST_SQL = """
SELECT run_id, created_at, method, duration, sample_count
FROM experiment_runs
ORDER BY created_at DESC, run_id ASC
"""


class DuplicateRunIDError(ValueError):
    """Raised when save would overwrite an existing run identifier."""


@dataclass(frozen=True, slots=True)
class ExperimentRunSummary:
    run_id: str
    created_at: str
    method: str
    duration: float
    sample_count: int


def _invalid_record(message):
    raise ValueError(f"invalid experiment reproducibility record: {message}")


def _require_exact_keys(name, mapping, expected_keys):
    if type(mapping) is not dict:
        _invalid_record(f"{name} must be a dictionary")

    actual_keys = set(mapping)
    expected_key_set = set(expected_keys)
    missing = expected_key_set - actual_keys
    extra = actual_keys - expected_key_set
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing keys {sorted(missing)!r}")
        if extra:
            details.append(f"unexpected keys {sorted(extra)!r}")
        _invalid_record(f"{name} has {' and '.join(details)}")


def _finite_real(name, value):
    if type(value) not in (int, float):
        _invalid_record(f"{name} must be a finite real number")
    try:
        numeric_value = float(value)
    except OverflowError:
        _invalid_record(f"{name} must be a finite real number")
    if not math.isfinite(numeric_value):
        _invalid_record(f"{name} must be a finite real number")
    return numeric_value


def _validate_metadata_scalar(name, value):
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float and math.isfinite(value):
        return
    _invalid_record(f"{name} must be a supported finite simple value")


def _validate_metadata(name, metadata):
    if type(metadata) is not dict:
        _invalid_record(f"{name} must be a dictionary")
    if any(type(key) is not str for key in metadata):
        _invalid_record(f"{name} keys must be strings")
    if tuple(metadata) != tuple(sorted(metadata)):
        _invalid_record(f"{name} keys must use deterministic sorted order")

    for key, value in metadata.items():
        value_name = f"{name}[{key!r}]"
        if type(value) is list:
            for index, item in enumerate(value):
                _validate_metadata_scalar(f"{value_name}[{index}]", item)
        else:
            _validate_metadata_scalar(value_name, value)


def _validate_record(record):
    _require_exact_keys("record", record, _TOP_LEVEL_KEYS)

    run_id = record["run_id"]
    if type(run_id) is not str or not run_id.strip():
        _invalid_record("run_id must be a non-empty string")

    created_at = record["created_at"]
    if type(created_at) is not str:
        _invalid_record("created_at must be an aware UTC ISO timestamp")
    try:
        parsed_created_at = datetime.fromisoformat(created_at)
    except ValueError:
        _invalid_record("created_at must be an aware UTC ISO timestamp")
    if parsed_created_at.utcoffset() != timedelta(0):
        _invalid_record("created_at must be an aware UTC ISO timestamp")

    method = record["method"]
    if type(method) is not str or not method.strip():
        _invalid_record("method must be a non-empty string")

    start_time = _finite_real("start_time", record["start_time"])
    end_time = _finite_real("end_time", record["end_time"])
    duration = _finite_real("duration", record["duration"])
    if end_time <= start_time or duration <= 0.0:
        _invalid_record("timing values must describe a positive interval")
    if not math.isfinite(end_time - start_time) or duration != end_time - start_time:
        _invalid_record("duration must equal end_time - start_time")

    sample_count = record["sample_count"]
    if (
        type(sample_count) is not int
        or sample_count < 2
        or sample_count > 2**63 - 1
    ):
        _invalid_record("sample_count must be an SQLite-compatible integer >= 2")

    initial_state = record["initial_state"]
    if type(initial_state) is not list:
        _invalid_record("initial_state must be a list")
    for index, value in enumerate(initial_state):
        _finite_real(f"initial_state[{index}]", value)

    for name in ("system", "controller", "reference", "user_metadata"):
        _validate_metadata(name, record[name])

    metrics = record["metrics"]
    _require_exact_keys("metrics", metrics, _METRIC_KEYS)

    for name in (
        "final_output",
        "final_reference",
        "steady_state_error",
        "peak_output",
        "maximum_absolute_tracking_error",
        "rms_tracking_error",
        "iae",
        "ise",
        "settling_tolerance",
    ):
        _finite_real(f"metrics.{name}", metrics[name])

    for name in (
        "maximum_absolute_tracking_error",
        "rms_tracking_error",
        "iae",
        "ise",
    ):
        if metrics[name] < 0.0:
            _invalid_record(f"metrics.{name} must be nonnegative")

    overshoot_percent = metrics["overshoot_percent"]
    if overshoot_percent is not None:
        overshoot_percent = _finite_real(
            "metrics.overshoot_percent", overshoot_percent
        )
        if overshoot_percent < 0.0:
            _invalid_record("metrics.overshoot_percent must be nonnegative")

    settling_time = metrics["settling_time"]
    if settling_time is not None:
        settling_time = _finite_real("metrics.settling_time", settling_time)
        if not start_time <= settling_time <= end_time:
            _invalid_record("metrics.settling_time must lie in the run interval")

    if metrics["settling_tolerance"] <= 0.0:
        _invalid_record("metrics.settling_tolerance must be positive")


def _record_from_run(run):
    if not isinstance(run, ExperimentRun):
        raise TypeError("run must be an ExperimentRun")
    try:
        record = run.reproducibility_record()
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "run does not contain a valid reproducibility record"
        ) from error
    _validate_record(record)
    return record


def _encode_json(name, value):
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} cannot be serialized as deterministic JSON") from error


def _decode_json(name, value):
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"stored {name} is not valid JSON") from error


def _record_values(record):
    metrics = record["metrics"]
    return (
        record["run_id"],
        record["created_at"],
        record["method"],
        record["start_time"],
        record["end_time"],
        record["duration"],
        record["sample_count"],
        _encode_json("initial_state", record["initial_state"]),
        _encode_json("system", record["system"]),
        _encode_json("controller", record["controller"]),
        _encode_json("reference", record["reference"]),
        _encode_json("user_metadata", record["user_metadata"]),
        metrics["final_output"],
        metrics["final_reference"],
        metrics["steady_state_error"],
        metrics["peak_output"],
        metrics["maximum_absolute_tracking_error"],
        metrics["rms_tracking_error"],
        metrics["iae"],
        metrics["ise"],
        metrics["overshoot_percent"],
        metrics["settling_time"],
        metrics["settling_tolerance"],
    )


def _record_from_row(row):
    record = {
        "run_id": row["run_id"],
        "created_at": row["created_at"],
        "method": row["method"],
        "start_time": row["start_time"],
        "end_time": row["end_time"],
        "duration": row["duration"],
        "sample_count": row["sample_count"],
        "initial_state": _decode_json("initial_state", row["initial_state_json"]),
        "system": _decode_json("system", row["system_json"]),
        "controller": _decode_json("controller", row["controller_json"]),
        "reference": _decode_json("reference", row["reference_json"]),
        "user_metadata": _decode_json(
            "user_metadata", row["user_metadata_json"]
        ),
        "metrics": {
            "final_output": row["final_output"],
            "final_reference": row["final_reference"],
            "steady_state_error": row["steady_state_error"],
            "peak_output": row["peak_output"],
            "maximum_absolute_tracking_error": row[
                "maximum_absolute_tracking_error"
            ],
            "rms_tracking_error": row["rms_tracking_error"],
            "iae": row["iae"],
            "ise": row["ise"],
            "overshoot_percent": row["overshoot_percent"],
            "settling_time": row["settling_time"],
            "settling_tolerance": row["settling_tolerance"],
        },
    }
    _validate_record(record)
    return record


class SQLiteExperimentStore:
    """Store immutable experiment reproducibility records in one SQLite file."""

    def __init__(self, path):
        try:
            path = os.fspath(path)
        except TypeError as error:
            raise TypeError("path must be a filesystem path or ':memory:'") from error
        if type(path) is not str or not path:
            raise ValueError("path must be a non-empty string path or ':memory:'")

        self.path = path
        self._connection = sqlite3.connect(path, isolation_level="DEFERRED")
        self._connection.row_factory = sqlite3.Row
        self._initialized = False

    def __enter__(self):
        try:
            self.initialize()
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _open_connection(self):
        if self._connection is None:
            raise RuntimeError("SQLiteExperimentStore is closed")
        return self._connection

    def _ready_connection(self):
        connection = self._open_connection()
        if not self._initialized:
            raise RuntimeError("SQLiteExperimentStore must be initialized")
        return connection

    def initialize(self):
        connection = self._open_connection()
        with connection:
            connection.execute(_CREATE_TABLE_SQL)
            connection.execute(_CREATE_LIST_INDEX_SQL)
        self._initialized = True

    def save(self, run):
        connection = self._ready_connection()
        record = _record_from_run(run)
        self._save_records(connection, (record,))

    def save_many(self, runs):
        """Persist a finite ordered collection of runs in one transaction."""
        connection = self._ready_connection()
        try:
            run_iterator = iter(runs)
        except TypeError as error:
            raise TypeError(
                "runs must be an iterable of ExperimentRun objects"
            ) from error
        records = tuple(_record_from_run(run) for run in run_iterator)
        self._save_records(connection, records)

    @staticmethod
    def _save_records(connection, records):
        try:
            with connection:
                for record in records:
                    connection.execute(_INSERT_SQL, _record_values(record))
        except sqlite3.IntegrityError as error:
            if error.sqlite_errorcode in (
                sqlite3.SQLITE_CONSTRAINT_PRIMARYKEY,
                sqlite3.SQLITE_CONSTRAINT_UNIQUE,
            ):
                raise DuplicateRunIDError(
                    f"run_id {record['run_id']!r} already exists"
                ) from error
            raise ValueError("run violates the experiment_runs schema") from error

    def get(self, run_id):
        connection = self._ready_connection()
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")

        cursor = connection.execute(_SELECT_SQL, (str(run_id),))
        try:
            row = cursor.fetchone()
        finally:
            cursor.close()
        return None if row is None else _record_from_row(row)

    def list_runs(self):
        connection = self._ready_connection()
        cursor = connection.execute(_LIST_SQL)
        try:
            rows = cursor.fetchall()
        finally:
            cursor.close()
        return tuple(
            ExperimentRunSummary(
                run_id=row["run_id"],
                created_at=row["created_at"],
                method=row["method"],
                duration=float(row["duration"]),
                sample_count=int(row["sample_count"]),
            )
            for row in rows
        )

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            self._initialized = False
