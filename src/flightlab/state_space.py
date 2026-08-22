import numpy as np


class StateSpace:
    def __init__(self, A, B, C, D):
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.D = np.asarray(D, dtype=float)

        if self.A.ndim != 2 or self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must have shape (n, n)")

        self.n_states = self.A.shape[0]

        if self.B.ndim != 2 or self.B.shape[0] != self.n_states:
            raise ValueError("B must have shape (n, m)")

        self.n_inputs = self.B.shape[1]

        if self.C.ndim != 2 or self.C.shape[1] != self.n_states:
            raise ValueError("C must have shape (p, n)")

        self.n_outputs = self.C.shape[0]

        if self.D.ndim != 2 or self.D.shape != (self.n_outputs, self.n_inputs):
            raise ValueError("D must have shape (p, m)")
