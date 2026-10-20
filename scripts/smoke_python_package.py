"""Smoke-test an installed sqp-craft wheel or source distribution."""
import os
import tempfile
from importlib.metadata import version


def main() -> None:
    # Run outside the checkout, so the test cannot import the source tree.
    os.chdir(tempfile.mkdtemp(prefix="sqp-craft-smoke-"))
    import sqp_craft

    installed = version("sqp-craft")
    expected = os.environ.get("EXPECTED_VERSION")
    if expected is not None:
        assert installed == expected, (installed, expected)
    assert sqp_craft.__version__ == installed, (sqp_craft.__version__, installed)

    from sqp_craft import ProblemSpec, check_derivatives, solve

    class Toy:
        """minimize (x0-1)^2 + (x1-2)^2 subject to x0 + x1 = 3, 0 <= x <= 4"""
        def functions(self, x):
            return (x[0] - 1) ** 2 + (x[1] - 2) ** 2, [x[0] + x[1]]

        def derivatives(self, x):
            return [2 * (x[0] - 1), 2 * (x[1] - 2)], [1.0, 1.0]

    spec = ProblemSpec(lower=[0, 0], upper=[4, 4],
                       constraint_lower=[3], constraint_upper=[3],
                       jacobian_indptr=[0, 2], jacobian_indices=[0, 1])
    assert check_derivatives(spec, Toy(), [0.5, 0.5]).passed
    result = solve(spec, Toy(), [0.5, 0.5], method="sqp")
    assert result.status == "converged", result.status
    assert abs(result.candidate[0] - 1.0) < 1e-6 and abs(result.candidate[1] - 2.0) < 1e-6
    summary = f"candidate={list(result.candidate)}"
    print(f"sqp-craft {installed}: {summary}")


if __name__ == "__main__":
    main()
