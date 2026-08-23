from dataclasses import dataclass, fields

import numpy as np

from flightlab.state_space import StateSpace


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
