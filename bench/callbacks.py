"""Callback accounting shared by the curvature comparison's solver adapters."""
from __future__ import annotations

from collections import Counter
import time

import numpy as np


class CallbackWork:
    """Count grouped model calls (including failed attempts), outside setup and verification."""

    def __init__(self):
        self.requests = Counter()
        self.evaluations = dict.fromkeys(("functions", "derivatives", "hessians"), 0)
        self.failures = self.evaluations.copy()
        self.seconds = self.evaluations.copy()
        self.s2mpj_hessian_calls = Counter()
        self.s2mpj_hessian_seconds = 0.0

    def evaluate(self, kind, callback, *args):
        self.evaluations[kind] += 1
        started = time.perf_counter()
        try:
            values = callback(*args)
            parts = values if isinstance(values, tuple) else (values,)
            if not all(np.all(np.isfinite(part)) for part in parts):
                raise ValueError(f"nonfinite {kind}")
            return values
        except Exception:
            self.failures[kind] += 1
            raise
        finally:
            self.seconds[kind] += time.perf_counter() - started

    def record(self):
        return dict(requests=dict(self.requests), evaluations=dict(self.evaluations),
                    failures=dict(self.failures), seconds=dict(self.seconds),
                    s2mpj_hessian_calls=dict(self.s2mpj_hessian_calls),
                    s2mpj_hessian_seconds=self.s2mpj_hessian_seconds)


def lagrangian_hessian(problem, x, sigma, multipliers, work=None) -> np.ndarray:
    """σ∇²f − Σλᵢ∇²cᵢ from S2MPJ, whose Lagrangian is f + yᵀc."""
    raw, column = problem.raw, x.reshape(-1, 1)

    def evaluate(method, *args):
        evaluate_model = getattr(raw, method)
        if work is None:
            return evaluate_model(column, *args)[2]
        work.s2mpj_hessian_calls[method] += 1
        started = time.perf_counter()
        try:
            return evaluate_model(column, *args)[2]
        finally:
            work.s2mpj_hessian_seconds += time.perf_counter() - started

    def dense(matrix):
        return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix, dtype=float)

    if problem.has_objective and problem.m and sigma == 1.0:
        return dense(evaluate("LgHxy", -multipliers.reshape(-1, 1)))
    # Weight the objective directly. Subtracting Hf from a rounded H(f+y·c)
    # can erase the constraint curvature when Hf dominates it.
    if problem.has_objective and sigma != 0.0:
        matrix = sigma * dense(evaluate("fgHx"))
    else:
        matrix = np.zeros((problem.n, problem.n))
    if np.any(multipliers):
        for weight, hessian in zip(multipliers, evaluate("cJHx"), strict=True):
            if weight != 0.0:
                matrix -= weight * dense(hessian)
    return matrix
