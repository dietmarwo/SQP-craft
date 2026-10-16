"""Native SQP on a small nonlinear problem, with a derivative check first.

minimize 0.5 * ((x0 - 2)^2 + x1^2)  subject to  x0^2 + x1^2 <= 1,  -3 <= x <= 3
Solution (1, 0) with objective 0.5 and row multiplier -0.5 (active upper bound).
"""

from sqp_craft import ProblemSpec, SqpOptions, check_derivatives, solve


class Circle:
    def functions(self, x):
        return 0.5 * ((x[0] - 2) ** 2 + x[1] ** 2), [x[0] ** 2 + x[1] ** 2]

    def derivatives(self, x):
        return [x[0] - 2, x[1]], [2 * x[0], 2 * x[1]]


spec = ProblemSpec(lower=[-3.0, -3.0], upper=[3.0, 3.0],
                   constraint_lower=[-float("inf")], constraint_upper=[1.0],
                   jacobian_indptr=[0, 2], jacobian_indices=[0, 1])
assert check_derivatives(spec, Circle(), [0.2, 0.4]).passed

result = solve(spec, Circle(), [0.2, 0.4], method="sqp",
               method_options=SqpOptions(record_history=True), observer=print)
print(result.status, "-", result.message)
print("x =", result.candidate, " multipliers =", result.multipliers)
for record in result.report.history:
    print(f"  iteration {record.iteration}: f = {record.objective:.8f}, "
          f"stationarity = {record.stationarity:.1e}, step = {record.step_length}")
assert result.status == "converged"
