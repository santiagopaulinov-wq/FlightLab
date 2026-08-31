from dataclasses import dataclass, fields
from typing import NamedTuple

import numpy as np

from flightlab.aircraft_modal import (
    AircraftModalFamilyCharacterization,
    filter_aircraft_modal_family_characterizations,
    interpret_modal_family_state_labels,
)
from flightlab.state_space import StateSpace


class LongitudinalModeIdentification(NamedTuple):
    """Conservative physical identification of one longitudinal modal family."""

    characterization: AircraftModalFamilyCharacterization
    mode_name: str | None
    evidence: "LongitudinalModeEvidence | None" = None


class LongitudinalModeEvidence(NamedTuple):
    """Immutable facts used for one longitudinal physical-mode decision."""

    is_oscillatory: bool
    natural_frequency: float | None
    period: float | None
    frequency_period_eligible: bool
    frequency_role: str | None
    frequency_separation_sufficient: bool | None
    expected_state_participation: float | None
    dominant_state_consistent: bool | None
    damping_ratio: float | None
    damping_ratio_valid: bool
    damping_order_consistent: bool | None


@dataclass(frozen=True)
class LongitudinalModel:
    """Four-state dimensional small-disturbance longitudinal model.

    States are ordered as ``(u, w, q, theta)`` and the input as
    ``(delta_e,)``. Their units are ``(m/s, m/s, rad/s, rad)`` and ``(rad,)``,
    respectively. Outputs are the full state vector in state order.

    The body axes are right-handed with x forward, y right, and z down.
    Therefore, positive ``u`` is forward and positive ``w`` is downward.
    Positive ``q`` follows the right-hand rule about the positive y-axis and is
    nose-up, while positive ``theta`` is a nose-up pitch attitude. Positive
    ``delta_e`` is elevator trailing-edge down; supplied control derivatives
    must use this convention.

    Parameters use SI units, with radians dimensionless in dimensional analysis:

    - ``trim_speed``: m/s; ``trim_pitch``: rad; ``gravity``: m/s^2.
    - ``x_u``, ``x_w``, ``z_u``, ``z_w``: 1/s.
    - ``x_q``, ``z_q``: m/s (equivalently m/(rad s)).
    - ``m_u``, ``m_w``: 1/(m s); ``m_q``: 1/s.
    - ``x_delta_e``, ``z_delta_e``: m/(s^2 rad).
    - ``m_delta_e``: (rad/s^2)/rad, equivalently 1/s^2.
    """

    STATE_ORDER = ("u", "w", "q", "theta")
    INPUT_ORDER = ("delta_e",)
    OUTPUT_ORDER = STATE_ORDER

    trim_speed: float
    trim_pitch: float
    gravity: float
    x_u: float
    x_w: float
    x_q: float
    x_delta_e: float
    z_u: float
    z_w: float
    z_q: float
    z_delta_e: float
    m_u: float
    m_w: float
    m_q: float
    m_delta_e: float

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
        sin_pitch = np.sin(self.trim_pitch)

        A = np.array(
            [
                [self.x_u, self.x_w, self.x_q, -self.gravity * cos_pitch],
                [
                    self.z_u,
                    self.z_w,
                    self.trim_speed + self.z_q,
                    -self.gravity * sin_pitch,
                ],
                [self.m_u, self.m_w, self.m_q, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        B = np.array(
            [
                [self.x_delta_e],
                [self.z_delta_e],
                [self.m_delta_e],
                [0.0],
            ]
        )
        C = np.eye(len(self.STATE_ORDER))
        D = np.zeros((len(self.OUTPUT_ORDER), len(self.INPUT_ORDER)))

        return StateSpace(A, B, C, D)

    def modal_family_characterizations(self):
        """Return generic modal families interpreted with longitudinal labels."""
        characterizations = self.to_state_space().modal_family_characterizations()
        return interpret_modal_family_state_labels(
            characterizations,
            self.STATE_ORDER,
            self.INPUT_ORDER,
            self.OUTPUT_ORDER,
        )

    def physical_mode_identifications(self):
        """Identify clear short-period and phugoid families conservatively."""
        characterizations = self.modal_family_characterizations()
        eligible = []
        evidence_values = []
        for index, characterization in enumerate(characterizations):
            dynamics = characterization.characterization.dynamics
            frequency = dynamics.natural_frequency
            period = dynamics.period
            damping = dynamics.damping_ratio
            frequency_period_eligible = bool(
                dynamics.is_oscillatory
                and frequency is not None
                and period is not None
                and np.isfinite(frequency)
                and np.isfinite(period)
                and frequency > 0.0
                and period > 0.0
            )
            damping_ratio_valid = bool(
                damping is not None
                and np.isfinite(damping)
                and 0.0 < damping < 1.0
            )
            evidence_values.append(
                {
                    "is_oscillatory": dynamics.is_oscillatory,
                    "natural_frequency": frequency,
                    "period": period,
                    "frequency_period_eligible": frequency_period_eligible,
                    "frequency_role": None,
                    "frequency_separation_sufficient": None,
                    "expected_state_participation": None,
                    "dominant_state_consistent": None,
                    "damping_ratio": damping,
                    "damping_ratio_valid": damping_ratio_valid,
                    "damping_order_consistent": None,
                }
            )
            if frequency_period_eligible:
                eligible.append((index, float(frequency)))

        names = [None] * len(characterizations)
        if len(eligible) >= 2:
            frequencies = np.asarray([frequency for _, frequency in eligible])
            slow_position = int(np.argmin(frequencies))
            fast_position = int(np.argmax(frequencies))
            slow_frequency = frequencies[slow_position]
            fast_frequency = frequencies[fast_position]
            unique_slow = np.count_nonzero(
                np.isclose(frequencies, slow_frequency, rtol=1e-7, atol=1e-12)
            ) == 1
            unique_fast = np.count_nonzero(
                np.isclose(frequencies, fast_frequency, rtol=1e-7, atol=1e-12)
            ) == 1
            slow_index = eligible[slow_position][0]
            fast_index = eligible[fast_position][0]
            separation_sufficient = bool(
                unique_slow
                and unique_fast
                and fast_frequency / slow_frequency >= 3.0
            )
            for index, _ in eligible:
                evidence_values[index]["frequency_separation_sufficient"] = (
                    separation_sufficient
                )
            if unique_slow:
                evidence_values[slow_index]["frequency_role"] = "slowest"
            if unique_fast:
                evidence_values[fast_index]["frequency_role"] = "fastest"

            if separation_sufficient:
                slow_damping = evidence_values[slow_index]["damping_ratio"]
                fast_damping = evidence_values[fast_index]["damping_ratio"]
                damping_evidence = (
                    evidence_values[slow_index]["damping_ratio_valid"]
                    and evidence_values[fast_index]["damping_ratio_valid"]
                    and slow_damping < fast_damping
                    and not np.isclose(
                        slow_damping, fast_damping, rtol=1e-7, atol=1e-12
                    )
                )
                damping_evidence = bool(damping_evidence)
                evidence_values[slow_index]["damping_order_consistent"] = (
                    damping_evidence
                )
                evidence_values[fast_index]["damping_order_consistent"] = (
                    damping_evidence
                )

            def state_evidence(index, expected_labels):
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

            if separation_sufficient:
                slow_state_evidence = state_evidence(slow_index, {"u", "theta"})
                fast_state_evidence = state_evidence(fast_index, {"w", "q"})
                if damping_evidence and slow_state_evidence:
                    names[slow_index] = "phugoid"
                if damping_evidence and fast_state_evidence:
                    names[fast_index] = "short_period"

        return tuple(
            LongitudinalModeIdentification(
                characterization,
                mode_name,
                LongitudinalModeEvidence(**evidence),
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
