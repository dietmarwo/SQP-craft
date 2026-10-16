"""End-to-end use of the Python API with the dummy backend.

Smooth equality-constrained toy model; not a trajectory benchmark.
"""

from sqp_craft import Cancellation, Options, ProblemSpec, solve


class Toy:
    """minimize (x0-1)^2 + (x1-2)^2  subject to  x0 + x1 = 3,  0 <= x <= 4"""

    def functions(self, x):
        return (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2, [x[0] + x[1]]

    def derivatives(self, x):
        return [2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)], [1.0, 1.0]


spec = ProblemSpec(
    lower=[0.0, 0.0],
    upper=[4.0, 4.0],
    constraint_lower=[3.0],
    constraint_upper=[3.0],
    jacobian_indptr=[0, 2],
    jacobian_indices=[0, 1],
)
print(spec)

result = solve(
    spec,
    Toy(),
    [0.5, 2.0],
    options=Options(feasibility_tolerance=1e-9),
    cancellation=Cancellation(),
    observer=print,
)
print(result.status, "-", result.message)
print("candidate:", result.candidate)
print("evaluation:", result.evaluation)
assert result.status == "stub" and result.stationarity is None
