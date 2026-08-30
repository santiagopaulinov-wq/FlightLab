from dataclasses import dataclass, fields

import numpy as np

from flightlab.aircraft_modal import (
    filter_aircraft_modal_family_characterizations,
    interpret_modal_family_state_labels,
)
from flightlab.state_space import StateSpace


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

    def filter_modal_family_characterizations(
        self,
        oscillatory=None,
        stability=None,
        dominant_state_labels=None,
        dominant_input_labels=None,
        dominant_output_labels=None,
    ):
        """Filter interpreted modal families by existing categorical dynamics."""
        return filter_aircraft_modal_family_characterizations(
            self.modal_family_characterizations(),
            oscillatory=oscillatory,
            stability=stability,
            dominant_state_labels=dominant_state_labels,
            dominant_input_labels=dominant_input_labels,
            dominant_output_labels=dominant_output_labels,
            state_labels=self.STATE_ORDER,
            input_labels=self.INPUT_ORDER,
            output_labels=self.OUTPUT_ORDER,
        )
