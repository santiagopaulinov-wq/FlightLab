import json
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

from flightlab.campaign import ExperimentCampaignResult
from flightlab.experiment import ExperimentRun
from flightlab.longitudinal import LongitudinalModel
from flightlab.state_space import StateSpace

_EXAMPLE_PATH = Path(__file__).parents[1] / "examples" / "longitudinal_campaign.py"
_EXAMPLE_SPEC = spec_from_file_location("longitudinal_campaign", _EXAMPLE_PATH)
assert _EXAMPLE_SPEC is not None and _EXAMPLE_SPEC.loader is not None
_EXAMPLE = module_from_spec(_EXAMPLE_SPEC)
_EXAMPLE_SPEC.loader.exec_module(_EXAMPLE)
run_workflow = _EXAMPLE.run_workflow
workflow_summary = _EXAMPLE.workflow_summary


def test_longitudinal_campaign_composes_complete_existing_api_path(tmp_path):
    database = tmp_path / "workflow.sqlite3"

    result = run_workflow(database)

    assert isinstance(result["aircraft"], LongitudinalModel)
    assert isinstance(result["aircraft_system"], StateSpace)
    assert isinstance(result["pitch_system"], StateSpace)
    assert result["pitch_system"].n_inputs == 1
    assert result["pitch_system"].n_outputs == 1
    assert len(result["modal_families"]) == 2
    assert result["structural_analysis"].controllable is True
    assert result["structural_analysis"].observable is True
    assert result["structural_analysis"].minimal is True
    assert isinstance(result["campaign"], ExperimentCampaignResult)
    assert all(isinstance(run, ExperimentRun) for run in result["campaign"].runs)
    assert database.is_file()


def test_longitudinal_campaign_persists_ordered_runs_and_analyzes_metrics(tmp_path):
    result = run_workflow(tmp_path / "workflow.sqlite3")

    expected_ids = [
        "longitudinal-pitch-scale-0.8",
        "longitudinal-pitch-scale-1.0",
        "longitudinal-pitch-scale-1.2",
    ]
    assert result["bundle_record"]["manifest"] == {
        "campaign_id": "longitudinal-pitch-pole-scale-v1",
        "created_at": "2026-09-07T13:00:00+00:00",
        "run_ids": expected_ids,
    }
    assert [entry.parameter_value for entry in result["comparison"]] == [
        0.8,
        1.0,
        1.2,
    ]
    assert tuple(name for name, _ in result["comparison"][0].metric_values) == (
        "iae",
        "overshoot_percent",
        "settling_time",
    )
    assert [entry.parameter_delta for entry in result["deltas"]] == pytest.approx([
        -0.2,
        0.0,
        0.2,
    ])
    assert dict(result["deltas"][1].metric_deltas) == {
        "iae": 0.0,
        "overshoot_percent": 0.0,
        "settling_time": 0.0,
    }
    for record in result["bundle_record"]["records"]:
        assert record["method"] == "exact"
        assert record["system"]["aircraft_model"] == "LongitudinalModel"
        assert record["controller"]["type"] == (
            "full_state_feedback_with_reference_prefilter"
        )
        assert record["metrics"]["iae"] > 0.0


def test_longitudinal_campaign_is_deterministic_across_fresh_databases(tmp_path):
    first = workflow_summary(run_workflow(tmp_path / "first.sqlite3"))
    second = workflow_summary(run_workflow(tmp_path / "second.sqlite3"))

    assert first == second
    json.dumps(first, allow_nan=False)
    assert first["structural_analysis"] == {
        "controllable": True,
        "detectable": True,
        "minimal": True,
        "observable": True,
        "stabilizable": True,
    }
    assert first["persisted_run_ids"] == [
        "longitudinal-pitch-scale-0.8",
        "longitudinal-pitch-scale-1.0",
        "longitudinal-pitch-scale-1.2",
    ]


def test_longitudinal_campaign_script_executes_from_repository(tmp_path):
    database = tmp_path / "script.sqlite3"
    result = subprocess.run(
        [
            sys.executable,
            "examples/longitudinal_campaign.py",
            str(database),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["campaign_id"] == "longitudinal-pitch-pole-scale-v1"
    assert summary["modal_family_count"] == 2
    assert len(summary["comparison"]) == 3
    assert database.is_file()
    assert np.isfinite(summary["comparison"][0]["metrics"]["iae"])
