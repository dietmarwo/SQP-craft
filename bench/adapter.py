"""One S2MPJ problem as seen by every solver of the harness.

`S2mpjProblem` normalizes bounds (|b| ≥ 1e20 is infinite), clips the start
into the variable bounds, derives the Jacobian sparsity pattern from the
problem structure (linear part plus element variables, verified against
evaluated Jacobians), reads the linear rows (`lincons`), and counts model
evaluations the same way for every solver: one *function evaluation* per
distinct point at which f and c are computed, one *derivative evaluation* per
distinct point at which ∇f and J are computed.
"""
from __future__ import annotations

import numpy as np

INFINITE = 1e20


def _bounds(values, n) -> np.ndarray:
    v = np.asarray(values if values is not None else np.zeros(n), dtype=float).reshape(-1)
    v = np.where(v >= INFINITE, np.inf, v)
    return np.where(v <= -INFINITE, -np.inf, v)


class S2mpjProblem:
    def __init__(self, raw, seed: int = 0):
        self.raw = raw
        self.name = str(raw.name)
        self.n = int(raw.n)
        self.m = int(getattr(raw, "m", 0) or 0)
        self.lower = _bounds(getattr(raw, "xlower", -np.inf * np.ones(self.n)), self.n)
        self.upper = _bounds(getattr(raw, "xupper", np.inf * np.ones(self.n)), self.n)
        x0 = np.asarray(raw.x0, dtype=float).reshape(-1)
        clipped = np.clip(x0, self.lower, self.upper)
        self.start_clipped = bool(np.any(clipped != x0))
        self.x0 = clipped
        if self.m:
            self.clower = _bounds(raw.clower, self.m)
            self.cupper = _bounds(raw.cupper, self.m)
        else:
            self.clower = self.cupper = np.zeros(0)
        self.has_objective = len(np.asarray(getattr(raw, "objgrps", [])).reshape(-1)) > 0
        self.linear_rows = sorted(int(i) for i in np.asarray(getattr(raw, "lincons", []))
                                  .reshape(-1))
        self.function_points: set = set()
        self.derivative_points: set = set()
        self._pattern(seed)

    # --- structure -------------------------------------------------------
    def _structural_rows(self) -> list[set]:
        raw, rows = self.raw, []
        A = getattr(raw, "A", None)
        groups = np.asarray(getattr(raw, "congrps", np.arange(self.m))).reshape(-1)
        grelt = getattr(raw, "grelt", None) or []
        elvar = getattr(raw, "elvar", None) or []
        A = A.tocsr() if A is not None else None
        for i in range(self.m):
            g = int(groups[i])
            cols: set = set()
            if A is not None and g < A.shape[0]:
                cols.update(int(j) for j in A.indices[A.indptr[g]:A.indptr[g + 1]])
            if g < len(grelt):
                for e in np.asarray(grelt[g]).reshape(-1):
                    cols.update(int(j) for j in np.asarray(elvar[int(e)]).reshape(-1))
            rows.append(cols)
        return rows

    def _pattern(self, seed: int) -> None:
        rows = self._structural_rows() if self.m else []
        self.pattern_extended = False
        if self.m:
            rng = np.random.default_rng(seed)
            points = [self.x0]
            span = np.where(np.isfinite(self.upper - self.lower), self.upper - self.lower, 2.0)
            for _ in range(2):
                trial = self.x0 + 0.1 * span * rng.uniform(-1, 1, self.n)
                points.append(np.clip(trial, self.lower, self.upper))
            for x in points:
                try:
                    J = self._jacobian(x)
                except Exception:
                    continue
                for i, j in zip(*np.nonzero(J)):
                    if int(j) not in rows[i]:
                        rows[i].add(int(j))
                        self.pattern_extended = True
        self.indptr = [0]
        self.indices: list[int] = []
        for cols in rows:
            self.indices.extend(sorted(cols))
            self.indptr.append(len(self.indices))
        self.nnz = len(self.indices)
        self._rows = np.repeat(np.arange(self.m), np.diff(self.indptr))
        self._cols = np.asarray(self.indices, dtype=int)

    # --- evaluations ------------------------------------------------------
    def _column(self, x):
        return np.asarray(x, dtype=float).reshape(-1, 1)

    def _jacobian(self, x) -> np.ndarray:
        _, J = self.raw.cJx(self._column(x))
        return J.toarray() if hasattr(J, "toarray") else np.asarray(J, dtype=float)

    def objective(self, x) -> float:
        if not self.has_objective:
            return 0.0
        return float(np.asarray(self.raw.fx(self._column(x))).reshape(-1)[0])

    def gradient(self, x) -> np.ndarray:
        if not self.has_objective:
            return np.zeros(self.n)
        _, g = self.raw.fgx(self._column(x))
        return np.asarray(g, dtype=float).reshape(-1)

    def constraints(self, x) -> np.ndarray:
        if not self.m:
            return np.zeros(0)
        return np.asarray(self.raw.cx(self._column(x)), dtype=float).reshape(-1)

    def jacobian_values(self, x) -> np.ndarray:
        if not self.m:
            return np.zeros(0)
        return self._jacobian(x)[self._rows, self._cols]

    def functions(self, x):
        self.function_points.add(tuple(np.asarray(x, dtype=float)))
        return self.objective(x), self.constraints(x)

    def derivatives(self, x):
        self.derivative_points.add(tuple(np.asarray(x, dtype=float)))
        return self.gradient(x), self.jacobian_values(x)

    def reset_counts(self) -> None:
        self.function_points.clear()
        self.derivative_points.clear()

    @property
    def counts(self) -> tuple[int, int]:
        return len(self.function_points), len(self.derivative_points)

    # --- sqp-craft --------------------------------------------------------
    def spec(self, linear: bool = False):
        from sqp_craft import ProblemSpec

        return ProblemSpec(lower=list(self.lower), upper=list(self.upper),
                           constraint_lower=list(self.clower),
                           constraint_upper=list(self.cupper),
                           jacobian_indptr=self.indptr, jacobian_indices=self.indices,
                           linear_rows=self.linear_rows if linear and self.linear_rows else None)

    def dense_jacobian(self, values) -> np.ndarray:
        J = np.zeros((self.m, self.n))
        J[self._rows, self._cols] = values
        return J
