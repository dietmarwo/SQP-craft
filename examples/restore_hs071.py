"""Recover from HS071's inconsistent linearization at [1, 1, 1, 1].

Run: python examples/restore_hs071.py
The objective may increase during restoration; only the subsequent normal SQP
KKT check may declare convergence.
"""
from sqp_craft import ProblemSpec, SqpOptions, kkt_residuals, solve


class Hs071:
    def functions(self, x):
        product = x[0] * x[1] * x[2] * x[3]
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2], [product, sum(v*v for v in x)]

    def derivatives(self, x):
        product = x[0] * x[1] * x[2] * x[3]
        gradient = [x[3]*(2*x[0]+x[1]+x[2]), x[0]*x[3], x[0]*x[3]+1, x[0]*(x[0]+x[1]+x[2])]
        return gradient, [product/v for v in x] + [2*v for v in x]


def specification():
    return ProblemSpec(lower=[1.]*4, upper=[5.]*4, constraint_lower=[25., 40.],
                       constraint_upper=[float('inf'), 40.],
                       jacobian_indptr=[0, 4, 8], jacobian_indices=list(range(4))*2)


if __name__ == '__main__':
    spec = specification()
    for budget in (0, 10):
        result = solve(spec, Hs071(), [1.]*4, method='sqp',
                       method_options=SqpOptions(max_restoration_steps=budget, record_history=True))
        print(f'restoration budget={budget}: {result.status}, '
              f'iterations={result.iterations}, restoration steps={result.report.restoration_steps}')
        for step in result.report.restoration_history:
            print(f'  feasibility merit {step.merit_before:.6g} -> {step.merit_after:.6g}')
        if budget:
            assert result.status == 'converged'
            assert abs(result.evaluation.objective - 17.0140173) < 1e-6
            print('  candidate:', result.candidate)
            print('  residuals:', kkt_residuals(spec, result.candidate, result.evaluation, result.multipliers))
        else:
            assert result.status == 'linearization_infeasible'
