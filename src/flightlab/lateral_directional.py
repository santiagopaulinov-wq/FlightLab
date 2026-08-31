from dataclasses import dataclass, fields
from typing import NamedTuple

import numpy as np

from flightlab.aircraft_modal import (
    AircraftModalFamilyCharacterization,
    filter_aircraft_modal_family_characterizations,
    interpret_modal_family_state_labels,
)
from flightlab.state_space import StateSpace


class LateralDirectionalModeEvidence(NamedTuple):
    """Immutable facts used for one lateral physical-mode decision."""

    is_oscillatory: bool
    stability: str
    oscillatory_eligible: bool
    real_mode_eligible: bool
    natural_frequency: float | None
    period: float | None
    damping_ratio: float | None
    damping_ratio_valid: bool | None
    real_rate: float | None
    real_rate_role: str | None
    real_rate_extreme_unique: bool | None
    real_rate_separation_sufficient: bool | None
    expected_state_participation: float | None
    dominant_state_consistent: bool | None
    candidate_ambiguous: bool | None


class LateralDirectionalModeIdentification(NamedTuple):
    """Conservative physical identification of one lateral modal family."""

    characterization: AircraftModalFamilyCharacterization
    mode_name: str | None
    evidence: LateralDirectionalModeEvidence | None = None


@dataclass(frozen=True)
class LateralDirectionalModel:
    """Four-state dimensional small-disturbance lateral-directional model.

    States are ordered as ``(v, p, r, phi)`` and controls as
    ``(delta_a, delta_r)``. Their units are ``(m/s, rad/s, rad/s, rad)`` and
    ``(rad, rad)``, respectively. Outputs are the full state vector.

    The right-handed body axes have x forward, y right, and z down. Thus
    positive ``v`` is rightward, positive ``p`` is right-wing-down, positive
    ``r`` turns the nose right, and positive ``phi`` is right-wing-down.
    Positive ``delta_a`` is right-aileron trailing-edge down and positive
    ``delta_r`` is rudder trailing-edge left; supplied control derivatives
    must follow these conventions. The model assumes symmetric straight trim.

    Parameters use SI units, with radians dimensionless in dimensional
    analysis:

    - ``trim_speed``: m/s; ``trim_pitch``: rad; ``gravity``: m/s^2.
    - ``y_v``: 1/s; ``y_p``, ``y_r``: m/s.
    - ``l_v``, ``n_v``: 1/(m s).
    - ``l_p``, ``l_r``, ``n_p``, ``n_r``: 1/s.
    - ``y_delta_a``, ``y_delta_r``: m/(s^2 rad).
    - ``l_delta_a``, ``l_delta_r``, ``n_delta_a``, ``n_delta_r``:
      (rad/s^2)/rad, equivalently 1/s^2.
    """

    STATE_ORDER = ("v", "p", "r", "phi")
    INPUT_ORDER = ("delta_a", "delta_r")
    OUTPUT_ORDER = STATE_ORDER

    trim_speed: float
    trim_pitch: float
    gravity: float
    y_v: float
    y_p: float
    y_r: float
    y_delta_a: float
    y_delta_r: float
    l_v: float
    l_p: float
    l_r: float
    l_delta_a: float
    l_delta_r: float
    n_v: float
    n_p: float
    n_r: float
    n_delta_a: float
    n_delta_r: float

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (str, bytes)) or not np.isscalar(value):
                raise ValueError(f"{field.name} must be a finite scalar")
            try:
                is_finite = np.isfinite(float(value))
            except (TypeError, ValueError, OverflowError):
                is_finite = False
            if not is_finite:
                raise ValueError(f"{field.name} must be a finite scalar")

    def to_state_space(self):
        cos_pitch = np.cos(self.trim_pitch)
        tan_pitch = np.tan(self.trim_pitch)

        A = np.array(
            [
                [
                    self.y_v,
                    self.y_p,
                    self.y_r - self.trim_speed,
                    self.gravity * cos_pitch,
                ],
                [self.l_v, self.l_p, self.l_r, 0.0],
                [self.n_v, self.n_p, self.n_r, 0.0],
                [0.0, 1.0, tan_pitch, 0.0],
            ]
        )
        B = np.array(
            [
                [self.y_delta_a, self.y_delta_r],
                [self.l_delta_a, self.l_delta_r],
                [self.n_delta_a, self.n_delta_r],
                [0.0, 0.0],
            ]
        )
        C = np.eye(len(self.STATE_ORDER))
        D = np.zeros((len(self.OUTPUT_ORDER), len(self.INPUT_ORDER)))

        return StateSpace(A, B, C, D)

    def modal_family_characterizations(self):
        """Return generic modal families interpreted with lateral labels."""
        characterizations = self.to_state_space().modal_family_characterizations()
        return interpret_modal_family_state_labels(
            characterizations,
            self.STATE_ORDER,
            self.INPUT_ORDER,
            self.OUTPUT_ORDER,
        )

    def physical_mode_identifications(self):
        """Identify clear Dutch-roll, roll, and spiral families conservatively."""
        characterizations = self.modal_family_characterizations()
        names = [None] * len(characterizations)
        evidence_values = []

        def has_state_evidence(index, expected_labels):
            characterization = characterizations[index]
            dominant = set(characterization.dominant_state_labels)
            participation = dict(characterization.state_participation_by_label)
            expected_participation = sum(
                participation[label] for label in expected_labels
            )
            dominant_consistent = bool(dominant) and dominant <= expected_labels
            evidence_values[index]["expected_state_participation"] = (
                expected_participation
            )
            evidence_values[index]["dominant_state_consistent"] = (
                dominant_consistent
            )
            return dominant_consistent and expected_participation >= 0.6

        oscillatory_candidates = []
        real_candidates = []
        for index, characterization in enumerate(characterizations):
            dynamics = characterization.characterization.dynamics
            damping_ratio_valid = None
            if dynamics.is_oscillatory:
                damping_ratio_valid = bool(
                    dynamics.damping_ratio is not None
                    and np.isfinite(dynamics.damping_ratio)
                    and 0.0 < dynamics.damping_ratio < 1.0
                )
                oscillatory_eligible = bool(
                    dynamics.stability == "decaying"
                    and dynamics.natural_frequency is not None
                    and dynamics.period is not None
                    and np.isfinite(dynamics.natural_frequency)
                    and np.isfinite(dynamics.period)
                    and dynamics.natural_frequency > 0.0
                    and dynamics.period > 0.0
                    and damping_ratio_valid
                )
                real_mode_eligible = False
                real_rate = None
                if oscillatory_eligible:
                    oscillatory_candidates.append(index)
            else:
                oscillatory_eligible = False
                real_mode_eligible = bool(
                    dynamics.stability in {"decaying", "growing"}
                    and dynamics.time_constant is not None
                    and np.isfinite(dynamics.real_part)
                    and np.isfinite(dynamics.time_constant)
                    and dynamics.real_part != 0.0
                    and dynamics.time_constant != 0.0
                )
                real_rate = (
                    abs(float(dynamics.real_part))
                    if real_mode_eligible
                    else None
                )
                if real_mode_eligible:
                    real_candidates.append((index, real_rate))
            evidence_values.append(
                {
                    "is_oscillatory": dynamics.is_oscillatory,
                    "stability": dynamics.stability,
                    "oscillatory_eligible": oscillatory_eligible,
                    "real_mode_eligible": real_mode_eligible,
                    "natural_frequency": dynamics.natural_frequency,
                    "period": dynamics.period,
                    "damping_ratio": dynamics.damping_ratio,
                    "damping_ratio_valid": damping_ratio_valid,
                    "real_rate": real_rate,
                    "real_rate_role": None,
                    "real_rate_extreme_unique": None,
                    "real_rate_separation_sufficient": None,
                    "expected_state_participation": None,
                    "dominant_state_consistent": None,
                    "candidate_ambiguous": None,
                }
            )

        for index in oscillatory_candidates:
            evidence_values[index]["candidate_ambiguous"] = (
                len(oscillatory_candidates) != 1
            )
        if len(oscillatory_candidates) == 1:
            dutch_roll_index = oscillatory_candidates[0]
            if has_state_evidence(dutch_roll_index, {"v", "r"}):
                names[dutch_roll_index] = "dutch_roll"

        if len(real_candidates) >= 2:
            rates = np.asarray([rate for _, rate in real_candidates])
            slow_position = int(np.argmin(rates))
            fast_position = int(np.argmax(rates))
            slow_rate = rates[slow_position]
            fast_rate = rates[fast_position]
            unique_slow = bool(
                np.count_nonzero(
                    np.isclose(rates, slow_rate, rtol=1e-7, atol=1e-12)
                )
                == 1
            )
            unique_fast = bool(
                np.count_nonzero(
                    np.isclose(rates, fast_rate, rtol=1e-7, atol=1e-12)
                )
                == 1
            )
            slow_index = real_candidates[slow_position][0]
            fast_index = real_candidates[fast_position][0]
            separation_sufficient = bool(
                unique_slow and unique_fast and fast_rate / slow_rate >= 3.0
            )
            for index, _ in real_candidates:
                evidence_values[index]["real_rate_separation_sufficient"] = (
                    separation_sufficient
                )
                evidence_values[index]["candidate_ambiguous"] = (
                    not separation_sufficient
                )
            slow_matches = np.flatnonzero(
                np.isclose(rates, slow_rate, rtol=1e-7, atol=1e-12)
            )
            fast_matches = np.flatnonzero(
                np.isclose(rates, fast_rate, rtol=1e-7, atol=1e-12)
            )
            for position in slow_matches:
                index = real_candidates[int(position)][0]
                evidence_values[index]["real_rate_extreme_unique"] = unique_slow
                if unique_slow:
                    evidence_values[index]["real_rate_role"] = "slowest"
            for position in fast_matches:
                index = real_candidates[int(position)][0]
                evidence_values[index]["real_rate_extreme_unique"] = unique_fast
                if unique_fast:
                    evidence_values[index]["real_rate_role"] = "fastest"
            if separation_sufficient:
                fast_dynamics = characterizations[
                    fast_index
                ].characterization.dynamics
                if (
                    fast_dynamics.stability == "decaying"
                    and has_state_evidence(fast_index, {"p"})
                ):
                    names[fast_index] = "roll_subsidence"
                if has_state_evidence(slow_index, {"v", "r", "phi"}):
                    names[slow_index] = "spiral"

        return tuple(
            LateralDirectionalModeIdentification(
                characterization,
                mode_name,
                LateralDirectionalModeEvidence(**evidence),
            )
            for characterization, mode_name, evidence in zip(
                characterizations, names, evidence_values, strict=True
            )
        )

    def filter_modal_family_characterizations(
        self,
        oscillatory=None,
        stability=None,
        dominant_state_labels=None,
        dominant_input_labels=None,
        dominant_output_labels=None,
        dominant_label_match="ANY",
        dominant_state_label_match=None,
        dominant_input_label_match=None,
        dominant_output_label_match=None,
        exclude_dominant_state_labels=None,
        exclude_dominant_input_labels=None,
        exclude_dominant_output_labels=None,
        exclude_dominant_label_match="ANY",
        exclude_dominant_state_label_match=None,
        exclude_dominant_input_label_match=None,
        exclude_dominant_output_label_match=None,
    ):
        """Filter interpreted modal families by existing categorical dynamics."""
        return filter_aircraft_modal_family_characterizations(
            self.modal_family_characterizations(),
            oscillatory=oscillatory,
            stability=stability,
            dominant_state_labels=dominant_state_labels,
            dominant_input_labels=dominant_input_labels,
            dominant_output_labels=dominant_output_labels,
            dominant_label_match=dominant_label_match,
            dominant_state_label_match=dominant_state_label_match,
            dominant_input_label_match=dominant_input_label_match,
            dominant_output_label_match=dominant_output_label_match,
            exclude_dominant_state_labels=exclude_dominant_state_labels,
            exclude_dominant_input_labels=exclude_dominant_input_labels,
            exclude_dominant_output_labels=exclude_dominant_output_labels,
            exclude_dominant_label_match=exclude_dominant_label_match,
            exclude_dominant_state_label_match=exclude_dominant_state_label_match,
            exclude_dominant_input_label_match=exclude_dominant_input_label_match,
            exclude_dominant_output_label_match=exclude_dominant_output_label_match,
            state_labels=self.STATE_ORDER,
            input_labels=self.INPUT_ORDER,
            output_labels=self.OUTPUT_ORDER,
        )
