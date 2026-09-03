"""Representative aircraft-to-campaign FlightLab workflow.

Run from the repository root after ``uv sync --locked --dev``:

    uv run python examples/longitudinal_campaign.py flightlab-demo.sqlite3
"""

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from flightlab.analysis import campaign_metric_deltas, compare_campaign_runs
from flightlab.campaign import run_experiment_campaign
from flightlab.experiment import ExperimentCase, SISOSimulationResult
from flightlab.longitudinal import LongitudinalModel
from flightlab.persistence import SQLiteExperimentStore, campaign_bundle_record
from flightlab.state_space import StateSpace

_POLE_SCALES = (0.8, 1.0, 1.2)
_BASE_POLES = np.array([-0.3, -0.6, -1.2, -2.0])
_TIME = np.linspace(0.0, 20.0, 81)
_REFERENCE_VALUE = 1.0
_RUN_CREATED_AT = datetime(2026, 9, 7, 12, tzinfo=UTC)
_CAMPAIGN_CREATED_AT = datetime(2026, 9, 7, 13, tzinfo=UTC)
_CAMPAIGN_ID = "longitudinal-pitch-pole-scale-v1"


def _aircraft_model():
    """Return the fixed representative dimensional longitudinal aircraft."""
    return LongitudinalModel(
        trim_speed=10.0,
        trim_pitch=0.0,
        gravity=9.81,
        x_u=-0.202,
        x_w=0.413,
        x_q=-0.148,
        x_delta_e=0.0,
        z_u=-0.437,
        z_w=-3.773,
        z_q=-0.235,
        z_delta_e=0.0,
        m_u=0.32,
        m_w=-0.909,
        m_q=-3.585,
        m_delta_e=-1.0,
    )


def _experiment_case(
    pitch_system,
    pole_scale,
    structural,
    modal_family_count,
    position,
):
    desired_poles = _BASE_POLES * pole_scale
    feedback_gain = pitch_system.place_siso_poles(desired_poles)
    closed_loop = pitch_system.full_state_feedback(feedback_gain)
    prefilter = pitch_system.siso_reference_prefilter(feedback_gain)
    initial_state = np.zeros(pitch_system.n_states)
    command = np.array([prefilter * _REFERENCE_VALUE])

    def simulate():
        _, output = closed_loop.simulate(
            initial_state,
            command,
            _TIME,
            method="exact",
        )
        return SISOSimulationResult(
            time=_TIME,
            output=output[:, 0],
            reference=np.full(_TIME.size, _REFERENCE_VALUE),
        )

    return ExperimentCase(
        simulation=simulate,
        initial_state=initial_state,
        method="exact",
        system={
            "aircraft_model": "LongitudinalModel",
            "input": "delta_e",
            "output": "theta",
            "state_count": pitch_system.n_states,
            "state_order": LongitudinalModel.STATE_ORDER,
        },
        controller={
            "desired_poles": tuple(desired_poles),
            "pole_scale": pole_scale,
            "reference_prefilter": prefilter,
            "state_feedback_gain": tuple(feedback_gain.flat),
            "type": "full_state_feedback_with_reference_prefilter",
        },
        reference={"type": "unit_step", "value_rad": _REFERENCE_VALUE},
        user_metadata={
            "modal_family_count": modal_family_count,
            "structurally_controllable": structural.controllable,
            "structurally_detectable": structural.detectable,
            "structurally_minimal": structural.minimal,
            "structurally_observable": structural.observable,
            "structurally_stabilizable": structural.stabilizable,
        },
        run_id=f"longitudinal-pitch-scale-{pole_scale:.1f}",
        created_at=_RUN_CREATED_AT + timedelta(minutes=position),
    )


def run_workflow(database_path):
    """Execute and persist the deterministic representative FlightLab workflow."""
    aircraft = _aircraft_model()
    aircraft_system = aircraft.to_state_space()
    modal_families = aircraft.modal_family_characterizations()

    pitch_system = StateSpace(
        aircraft_system.A,
        aircraft_system.B,
        aircraft_system.C[3:4],
        aircraft_system.D[3:4],
    )
    structural = pitch_system.structural_analysis()
    cases = tuple(
        _experiment_case(
            pitch_system,
            pole_scale,
            structural,
            len(modal_families),
            position,
        )
        for position, pole_scale in enumerate(_POLE_SCALES)
    )

    with SQLiteExperimentStore(database_path) as store:
        campaign = run_experiment_campaign(
            cases,
            store=store,
            campaign_id=_CAMPAIGN_ID,
            created_at=_CAMPAIGN_CREATED_AT,
        )
        bundle = store.get_campaign_bundle(campaign.campaign_id)
        if bundle is None:  # pragma: no cover - persisted campaign invariant
            raise RuntimeError("persisted campaign could not be retrieved")
        bundle_record = campaign_bundle_record(bundle)

    comparison = compare_campaign_runs(
        bundle_record,
        parameter_category="controller",
        parameter_key="pole_scale",
        metric_names=("iae", "overshoot_percent", "settling_time"),
    )
    deltas = campaign_metric_deltas(
        comparison,
        baseline_run_id="longitudinal-pitch-scale-1.0",
    )
    return {
        "aircraft": aircraft,
        "aircraft_system": aircraft_system,
        "pitch_system": pitch_system,
        "modal_families": modal_families,
        "structural_analysis": structural,
        "campaign": campaign,
        "bundle_record": bundle_record,
        "comparison": comparison,
        "deltas": deltas,
    }


def workflow_summary(result):
    """Return a compact JSON-compatible summary of a completed workflow."""
    return {
        "aircraft_model": "LongitudinalModel",
        "campaign_id": result["campaign"].campaign_id,
        "comparison": [
            {
                "metrics": dict(entry.metric_values),
                "pole_scale": entry.parameter_value,
                "run_id": entry.run_id,
            }
            for entry in result["comparison"]
        ],
        "metric_deltas_from_scale_1.0": [
            {
                "metric_deltas": dict(entry.metric_deltas),
                "pole_scale_delta": entry.parameter_delta,
                "run_id": entry.run_id,
            }
            for entry in result["deltas"]
        ],
        "modal_family_count": len(result["modal_families"]),
        "persisted_run_ids": result["bundle_record"]["manifest"]["run_ids"],
        "structural_analysis": {
            "controllable": result["structural_analysis"].controllable,
            "detectable": result["structural_analysis"].detectable,
            "minimal": result["structural_analysis"].minimal,
            "observable": result["structural_analysis"].observable,
            "stabilizable": result["structural_analysis"].stabilizable,
        },
    }


if __name__ == "__main__":
    output_path = Path(sys.argv[1] if len(sys.argv) == 2 else "flightlab-demo.sqlite3")
    print(json.dumps(workflow_summary(run_workflow(output_path)), indent=2))
