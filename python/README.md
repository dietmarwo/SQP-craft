<p align="center">
  <img src="https://raw.githubusercontent.com/dietmarwo/SQP-craft/main/docs/images/sqp-craft.png" alt="SQP-craft: smooth constrained optimization in Rust">
</p>

# sqp-craft

Sequential quadratic programming (SQP) for smooth constrained nonlinear
problems, with a native Rust solver: damped BFGS updates of a dense Hessian
approximation, step QPs solved by Clarabel (interior point, the default) or a
dense active-set backend, and an ℓ1 merit line search. Results report
stationarity, complementarity and dual-sign residuals of the KKT conditions,
and `kkt_residuals` and `check_derivatives` check points and derivatives
independently.

## Install

```sh
python -m pip install sqp-craft
```

Wheels (abi3, CPython 3.10 or newer) cover Linux x86_64, Windows x64 and macOS
x86_64/arm64 and need no compiler. On other platforms pip builds the source
distribution, which needs Rust 1.85 or newer. The package depends on nothing
else at run time.

## Example

```python
from sqp_craft import ProblemSpec, check_derivatives, solve

class Model:
    """minimize (x0-2)^2 + (x1-1)^2  subject to  x0^2 - x1 <= 0,  x0 + x1 <= 2"""
    def functions(self, x):
        return (x[0] - 2) ** 2 + (x[1] - 1) ** 2, [x[0] ** 2 - x[1], x[0] + x[1]]

    def derivatives(self, x):  # gradient, then the Jacobian values in CSR order
        return [2 * (x[0] - 2), 2 * (x[1] - 1)], [2 * x[0], -1.0, 1.0, 1.0]

inf = float("inf")
spec = ProblemSpec(lower=[-5, -5], upper=[5, 5],
                   constraint_lower=[-inf, -inf], constraint_upper=[0, 2],
                   jacobian_indptr=[0, 2, 4], jacobian_indices=[0, 1, 0, 1])
assert check_derivatives(spec, Model(), [0.5, 0.5]).passed
result = solve(spec, Model(), [0.5, 0.5], method="sqp")
print(result.status, result.candidate, result.evaluation.objective)
# converged, candidate ≈ [1, 1], objective ≈ 1
```

Pass `method="sqp"`: the default `method="dummy"` only evaluates the start
point. `SqpOptions` configures the solver (iteration, callback and time
limits, KKT tolerances, `qp_solver="clarabel"` or `"active-set"`,
`elastic_mode` with the active-set backend for infeasible linearizations);
invalid values raise `ValueError`. Callbacks run serially on the calling
thread.

## Limits

- Smooth (twice continuously differentiable) objective and constraints with
  first derivatives; kinks, noise and branches break BFGS and the line search.
- At most 256 variables (dense BFGS). For larger sparse problems, an
  IPOPT-family solver fits better.
- A local method: `converged` means a first-order KKT point within the
  tolerances, not a global optimum. Use several starts for multimodal models.

## More

- [Repository](https://github.com/dietmarwo/SQP-craft), with the multistart
  examples and the source build; the Rust crates start at
  [`sqp-craft-solver`](https://crates.io/crates/sqp-craft-solver)
- [AI agent guide](https://github.com/dietmarwo/SQP-craft/blob/main/ai-context.md):
  modelling, options, results, multistart and the job API
- [Changelog](https://github.com/dietmarwo/SQP-craft/blob/main/CHANGELOG.md)
- [Shared solver-selection guide](https://github.com/dietmarwo/optimization-jobs/blob/main/ai-context.md)
  and [ecosystem overview](https://dietmarwo.github.io/optimization-jobs/):
  when black-box optimization (fcmaes) or LP/MIP (LP-IP-craft) fits better,
  and durable shell/MCP jobs through the `sqp-job-worker` (built from source)

MIT license.
