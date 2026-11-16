import pytest
from sqp_craft import ProblemSpec, SqpOptions, solve, UndefinedPointError, EvaluationError


def spec(n=1):
    return ProblemSpec(
        lower=[-float("inf")] * n,
        upper=[float("inf")] * n,
        constraint_lower=[],
        constraint_upper=[],
        jacobian_indptr=[0],
        jacobian_indices=[],
    )


class LargeQuadratic:
    def functions(self, x):
        return 1e12 * (x[0] - 1) ** 2, []

    def derivatives(self, x):
        return [2e12 * (x[0] - 1)], []


def test_major_step_and_curvature_scaling_recover_large_first_gradient():
    old = solve(spec(), LargeQuadratic(), [0.0], method="sqp", method_options=SqpOptions())
    r = solve(
        spec(),
        LargeQuadratic(),
        [0.0],
        method="sqp",
        method_options=SqpOptions(major_step_limit=2.0, initial_curvature_scaling=True),
    )
    assert old.status == "line_search_failure"
    assert r.status == "converged"
    assert r.candidate == pytest.approx([1.0])
    assert r.report.step_limited_iterations > 0
    assert r.report.initial_curvature_scale == pytest.approx(2e12)


def test_safeguards_leave_untriggered_runs_unchanged():
    class Small:
        def functions(self, x):
            return (x[0] - 1) ** 2, []

        def derivatives(self, x):
            return [2 * (x[0] - 1)], []

    a = solve(spec(), Small(), [0.5], method="sqp")
    b = solve(
        spec(),
        Small(),
        [0.5],
        method="sqp",
        method_options=SqpOptions(major_step_limit=2, initial_curvature_scaling=True),
    )
    assert a.status == b.status and a.candidate == b.candidate and a.counts == b.counts
    assert b.report.step_limited_iterations == 0 and b.report.initial_curvature_scale is None


@pytest.mark.parametrize("limit", [0, -1, float("inf"), float("nan")])
def test_invalid_major_step_limits_are_rejected(limit):
    with pytest.raises(ValueError, match="major_step_limit"):
        solve(
            spec(),
            LargeQuadratic(),
            [0],
            method="sqp",
            method_options=SqpOptions(major_step_limit=limit),
        )


def test_solver_projected_undefined_start_returns_no_candidate():
    class Projected:
        def functions(self, x):
            if x[0] > 0.5:
                raise UndefinedPointError("domain x <= 0.5")
            return x[0] ** 2, [x[0]]

        def derivatives(self, x):
            return [2 * x[0]], [1.0]

    s = ProblemSpec(
        lower=[-2],
        upper=[2],
        constraint_lower=[1],
        constraint_upper=[1],
        jacobian_indptr=[0, 1],
        jacobian_indices=[0],
        linear_rows=[0],
    )
    r = solve(s, Projected(), [0], method="sqp")
    assert r.status == "numerical_failure" and r.candidate is None and r.evaluation is None
    assert "projected start" in r.message


@pytest.mark.parametrize("error", [EvaluationError, UndefinedPointError])
def test_user_start_errors_remain_errors(error):
    class Broken(LargeQuadratic):
        def functions(self, x):
            raise error("model message")

    with pytest.raises(error, match="model message"):
        solve(spec(), Broken(), [0], method="sqp")


def test_callback_errors_inside_iterations_are_not_swallowed():
    class Broken:
        def functions(self, x):
            if x[0] != 0:
                raise EvaluationError("callback failure")
            return 1.0, []

        def derivatives(self, x):
            return [-1.0], []

    with pytest.raises(EvaluationError, match="callback failure"):
        solve(spec(), Broken(), [0], method="sqp")


@pytest.mark.parametrize("what", ["objective", "gradient"])
@pytest.mark.parametrize("at_trial", [False, True])
def test_nonfinite_callback_outputs_keep_evaluation_errors(what, at_trial):
    class Bad:
        def functions(self, x):
            return (
                float("nan") if what == "objective" and (not at_trial or x[0] != 0) else -x[0]
            ), []

        def derivatives(self, x):
            return [
                float("nan") if what == "gradient" and (not at_trial or x[0] != 0) else -1.0
            ], []

    with pytest.raises(EvaluationError, match="nonfinite"):
        solve(spec(), Bad(), [0], method="sqp")


def test_nonfinite_augmented_trial_returns_the_last_complete_point():
    class HugeTrial:
        def functions(self, x):
            return -x[0], [x[0] + 1e200 * (x[0] - 1) ** 2]

        def derivatives(self, x):
            return [-1.0], [1 + 2e200 * (x[0] - 1)]

    s = ProblemSpec(
        lower=[-2],
        upper=[2],
        constraint_lower=[0],
        constraint_upper=[0],
        jacobian_indptr=[0, 1],
        jacobian_indices=[0],
    )
    r = solve(
        s, HugeTrial(), [1], method="sqp", method_options=SqpOptions(merit="augmented-lagrangian")
    )
    assert r.status == "numerical_failure" and r.candidate == [1.0]
    assert "merit" in r.message and "iteration 0" in r.message
    assert r.counts.functions == 2 and r.counts.derivatives == 1


@pytest.mark.parametrize("soc", [False, True])
def test_complementarity_overflow_returns_a_status(soc):
    class HugeResidual:
        def functions(self, x):
            return 1.7e308, [1e308]

        def derivatives(self, x):
            return [-1.0], [0.0]

    s = ProblemSpec(
        lower=[-10],
        upper=[10],
        constraint_lower=[0],
        constraint_upper=[0],
        jacobian_indptr=[0, 1],
        jacobian_indices=[0],
    )
    r = solve(
        s, HugeResidual(), [0], method="sqp", method_options=SqpOptions(second_order_correction=soc)
    )
    assert r.status == "numerical_failure" and r.candidate == [0.0]
    assert "complementarity" in r.message


def test_partitioned_and_block_curvature_share_the_initial_scale():
    # A separable objective with a large first gradient, so all modes use
    # the same Shanno-Phua factor before their different BFGS updates.
    class Quadratic:
        def functions(self, x):
            return 1e10 * sum((v - 1) ** 2 for v in x), []

        def derivatives(self, x):
            return [2e10 * (v - 1) for v in x], []

    for mode in ["full", "block", "partitioned"]:
        fields = dict(major_step_limit=2.0, initial_curvature_scaling=True)
        s = ProblemSpec(
            lower=[-10.0] * 4,
            upper=[10.0] * 4,
            constraint_lower=[],
            constraint_upper=[],
            jacobian_indptr=[0],
            jacobian_indices=[],
            objective_pattern=list(range(4)),
        )
        if mode == "block":
            fields["hessian_block_size"] = 2
        if mode == "partitioned":
            fields["hessian"] = "partitioned"
        r = solve(s, Quadratic(), [0.0] * 4, method="sqp", method_options=SqpOptions(**fields))
        assert r.report.step_limited_iterations > 0
        assert r.report.initial_curvature_scale == pytest.approx(2e10, rel=1e-6)
        assert r.candidate == pytest.approx([1.0] * 4, abs=1e-6)


def test_exact_hessian_is_not_rescaled():
    class Exact:
        def functions(self, x):
            return (x[0] - 100) ** 2, []

        def derivatives(self, x):
            return [2 * (x[0] - 100)], []

        def lagrangian_hessian(self, x, lam):
            return [2.0]

    s = ProblemSpec(
        lower=[-1000.0],
        upper=[1000.0],
        constraint_lower=[],
        constraint_upper=[],
        jacobian_indptr=[0],
        jacobian_indices=[],
        hessian_indptr=[0, 1],
        hessian_indices=[0],
    )
    r = solve(
        s,
        Exact(),
        [0],
        method="sqp",
        method_options=SqpOptions(
            hessian="exact", major_step_limit=2, initial_curvature_scaling=True
        ),
    )
    assert r.status == "converged" and r.candidate == pytest.approx([100.0])
    assert r.report.step_limited_iterations > 0 and r.report.initial_curvature_scale is None
