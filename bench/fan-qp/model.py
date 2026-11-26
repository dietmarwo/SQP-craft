"""Stage data, reduced CSR coordinates and independent certificates for the fan study."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

ABS_TOL = 1e-8
COMPLEMENTARITY_TOL = 1e-6
OBJECTIVE_TOL = 1e-7


@dataclass(frozen=True)
class Stage:
    hessian: np.ndarray
    gradient: np.ndarray
    selectors: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    dynamics: np.ndarray | None = None
    offset: np.ndarray | None = None


@dataclass(frozen=True)
class Branch:
    initial: np.ndarray
    stages: tuple[Stage, ...]


@dataclass(frozen=True)
class Fixture:
    branches: tuple[Branch, ...]
    nx: int
    nu: int
    objective_constant: float = 0.0  # Only the target-based toys supply this.

    @property
    def J(self):
        return len(self.branches)

    @property
    def N(self):
        return len(self.branches[0].stages) - 1


def read_fixture(path: Path):
    with path.open(encoding="ascii") as file:
        tag, *dimensions = file.readline().split()
        if tag != "fan-ocp/1":
            raise ValueError("expected fan-ocp/1")
        J, N, nx, nu, ny, ny_terminal = map(int, dimensions)
        if min(J, nx, nu) < 1 or N < 2 or min(ny, ny_terminal) < 0:
            raise ValueError("invalid fixture dimensions")
        values = np.fromiter(map(float, file.read().split()), dtype=np.float64)
    width = nx + nu
    n_stage = nx * width + nx + width**2 + width + ny * width + 2 * ny
    n_terminal = nx**2 + nx + ny_terminal * nx + 2 * ny_terminal
    if len(values) != J * (nx + N * n_stage + n_terminal):
        raise ValueError("fixture token count does not match its header")
    offset = 0

    def read(*shape):
        nonlocal offset
        count = int(np.prod(shape))
        data = values[offset:offset + count].reshape(shape)
        offset += count
        return data

    branches = []
    for _ in range(J):
        initial = read(nx)
        stages = []
        for k in range(N + 1):
            local_width = width if k < N else nx
            local_ny = ny if k < N else ny_terminal
            dynamics, affine = (read(nx, width), read(nx)) if k < N else (None, None)
            stages.append(Stage(
                read(local_width, local_width), read(local_width),
                read(local_ny, local_width), read(local_ny), read(local_ny),
                dynamics, affine,
            ))
        branches.append(Branch(initial, tuple(stages)))
    return Fixture(tuple(branches), nx, nu)


def write_values(file, values):
    # Each call streams one vector or one upstream stage block.
    file.write(" ".join(str(x) if isinstance(x, (int, np.integer)) else
                        format(float(x), ".17g") for x in np.ravel(values)))
    file.write("\n")


def write_fixture(path, fixture):
    stages = fixture.branches[0].stages
    with path.open("w", encoding="ascii") as file:
        file.write(f"fan-ocp/1 {fixture.J} {fixture.N} {fixture.nx} {fixture.nu} "
                   f"{len(stages[0].lower)} {len(stages[-1].lower)}\n")
        for branch in fixture.branches:
            write_values(file, branch.initial)
            for stage in branch.stages:
                if stage.dynamics is not None:
                    write_values(file, stage.dynamics)
                    write_values(file, stage.offset)
                for data in (stage.hessian, stage.gradient, stage.selectors, stage.lower, stage.upper):
                    write_values(file, data)


@dataclass(frozen=True)
class Layout:
    J: int
    N: int
    nx: int
    nu: int

    @property
    def branch_width(self):
        return self.N * self.nx + (self.N - 1) * self.nu

    @property
    def root(self):
        return slice(self.J * self.branch_width, self.n)

    @property
    def n(self):
        return self.J * self.branch_width + self.nu

    @property
    def m(self):
        return self.J * self.N * self.nx

    def x(self, j, k):
        assert 1 <= k <= self.N
        first = j * self.branch_width + (k - 1) * (self.nx + self.nu)
        return slice(first, first + self.nx)

    def u(self, j, k):
        if k == 0:
            return self.root
        first = self.x(j, k).stop
        return slice(first, first + self.nu)

    def stage(self, j, k):
        if k == 0:
            return self.root
        start = self.x(j, k).start
        stop = self.u(j, k).stop if k < self.N else self.x(j, k).stop
        return slice(start, stop)

    def eq(self, j, k):
        first = (j * self.N + k) * self.nx
        return slice(first, first + self.nx)

    def labels(self):
        names = []
        for j in range(self.J):
            for k in range(1, self.N + 1):
                names.extend((j, k, "state", i) for i in range(self.nx))
                if k < self.N:
                    names.extend((j, k, "input", i) for i in range(self.nu))
        names.extend((None, 0, "shared input", i) for i in range(self.nu))
        return names


def indices(block):
    return range(block.start, block.stop)


@dataclass(frozen=True)
class Program:
    layout: Layout
    lower_hessian: csr_matrix
    gradient: np.ndarray
    jacobian: csr_matrix
    rhs: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    offset: float
    bound_origin: tuple

    def product(self, x):
        H = self.lower_hessian
        return H @ x + H.T @ x - H.diagonal() * x

    def cost(self, x):
        return float((0.5 * self.product(x) + self.gradient) @ x + self.offset)

    def write(self, path, *, gradient=None, lower=None, upper=None, rhs=None):
        H, E = self.lower_hessian, self.jacobian
        n, m = self.layout.n, self.layout.m
        with path.open("w", encoding="ascii") as file:
            file.write(f"fan-qp/1 {n} {m} {H.nnz} {E.nnz}\n")
            for matrix in (H, E):
                for entries in (matrix.indptr, matrix.indices, matrix.data):
                    write_values(file, entries)
            rhs = self.rhs if rhs is None else rhs
            for entries in (self.gradient if gradient is None else gradient,
                            self.lower if lower is None else lower,
                            self.upper if upper is None else upper, rhs, rhs):
                write_values(file, entries)


def make_csr(rows, n):
    ptr, cols, values = [0], [], []
    for row in rows:
        for col, value in sorted(row.items()):
            cols.append(col)
            values.append(value)
        ptr.append(len(cols))
    return csr_matrix((np.asarray(values, dtype=np.float64), cols, ptr), shape=(len(rows), n))


def validate_support(q):
    L = q.layout
    for j in range(L.J):
        for k in range(L.N):
            allowed = set(indices(L.stage(j, k))) | set(indices(L.x(j, k + 1)))
            for row in indices(L.eq(j, k)):
                E = q.jacobian
                actual = set(E.indices[E.indptr[row]:E.indptr[row + 1]])
                if not actual <= allowed:
                    raise ValueError(f"dynamics row {row} couples undeclared branch or stage columns")
    H = q.lower_hessian
    blocks = [L.stage(j, k) for j in range(L.J) for k in range(1, L.N + 1)] + [L.root]
    for block in blocks:
        for row in indices(block):
            cols = H.indices[H.indptr[row]:H.indptr[row + 1]]
            if len(cols) == 0 or cols[-1] != row or cols[0] < block.start:
                raise ValueError(f"Hessian row {row} leaves its declared stage or lacks its diagonal")


def assemble(fixture):
    L = Layout(fixture.J, fixture.N, fixture.nx, fixture.nu)
    n, m, nx = L.n, L.m, L.nx
    gradient = np.zeros(n)
    lower, upper = np.full(n, -np.inf), np.full(n, np.inf)
    H, E, rhs = [{} for _ in range(n)], [], np.empty(m)
    offset = fixture.objective_constant
    bound_origin = []
    for j, branch in enumerate(fixture.branches):
        if not np.array_equal(branch.initial, fixture.branches[0].initial):
            raise ValueError("branches must share the fixed initial state")
        if len(branch.stages) != L.N + 1 or not np.isfinite(branch.initial).all():
            raise ValueError("invalid branch data")
        source = []
        for k, stage in enumerate(branch.stages):
            cols = L.stage(j, k)
            h, g = stage.hessian, stage.gradient
            if not (np.isfinite(h).all() and np.isfinite(g).all() and np.array_equal(h, h.T)):
                raise ValueError("cost must be finite and symmetric")
            if k == 0:
                x0 = branch.initial
                offset += float(x0 @ (0.5 * h[:nx, :nx] @ x0 + g[:nx]))
                gradient[cols] += g[nx:] + h[nx:, :nx] @ x0
                h = h[nx:, nx:]
            else:
                gradient[cols] = g
            for local, row in enumerate(indices(cols)):
                for column, value in enumerate(h[local, :local + 1], start=cols.start):
                    if value != 0.0 or row == column:
                        H[row][column] = H[row].get(column, 0.0) + value
            if k < L.N:
                ab, b = stage.dynamics, stage.offset
                if not (np.isfinite(ab).all() and np.isfinite(b).all()):
                    raise ValueError("nonfinite dynamics")
                rhs[L.eq(j, k)] = b + ab[:, :nx] @ branch.initial if k == 0 else b
                transition = ab[:, nx:] if k == 0 else ab
                for i, successor in enumerate(indices(L.x(j, k + 1))):
                    row = {c: -v for c, v in zip(indices(cols), transition[i]) if v != 0}
                    row[successor] = 1.0
                    E.append(row)
            stage_rows = []
            for selector, lo, hi in zip(stage.selectors, stage.lower, stage.upper, strict=True):
                nonzero = np.flatnonzero(selector)
                if len(nonzero) != 1 or not np.isfinite(selector).all():
                    raise ValueError("fixture assembler requires coordinate selectors")
                if np.isnan(lo) or np.isnan(hi) or lo > hi or lo == np.inf or hi == -np.inf:
                    raise ValueError("invalid stage bounds")
                i = int(nonzero[0])
                sign = selector[i]
                if k == 0 and i < nx:
                    if not lo <= sign * branch.initial[i] <= hi:
                        raise ValueError("fixed x0 violates a root-state bound")
                    stage_rows.append((-1, sign))
                    continue
                column = cols.start + i - (nx if k == 0 else 0)
                interval = (lo / sign, hi / sign) if sign > 0 else (hi / sign, lo / sign)
                lower[column] = max(lower[column], interval[0])
                upper[column] = min(upper[column], interval[1])
                stage_rows.append((column, sign))
            source.append(tuple(stage_rows))
        bound_origin.append(tuple(source))
    if (lower > upper).any():
        raise ValueError("empty intersection of shared bounds")
    q = Program(L, make_csr(H, n), gradient, make_csr(E, n), rhs, lower, upper,
                offset, tuple(bound_origin))
    validate_support(q)
    return q


def stage_vectors(fixture, layout, x, j):
    branch = fixture.branches[j]
    for k in range(layout.N + 1):
        state = branch.initial if k == 0 else x[layout.x(j, k)]
        yield state if k == layout.N else np.r_[state, x[layout.u(j, k)]]


def forward_rollout(fixture, layout, x):
    result = x.copy()
    for j, branch in enumerate(fixture.branches):
        state = branch.initial
        for k, stage in enumerate(branch.stages[:-1]):
            state = stage.dynamics @ np.r_[state, x[layout.u(j, k)]] + stage.offset
            result[layout.x(j, k + 1)] = state
    return result


def native_candidate(fixture, q, data):
    if data["schema"] != "fan-qpalm-solution/1":
        raise ValueError("unknown native solution")
    L = q.layout
    if len(data["branches"]) != L.J or len(data["shared_input"]) != L.nu:
        raise ValueError("wrong native result size")
    point = np.empty(L.n)
    point[L.root] = data["shared_input"]
    dynamics_dual = np.empty(L.m)
    native_bounds = []
    root_copy = {"value": 0.0, "at": None}
    initial_copy = {"value": 0.0, "at": None}
    for j, br in enumerate(data["branches"]):
        trajectory = np.asarray(br["trajectory"])
        values = trajectory[:L.N * (L.nx + L.nu)].reshape(L.N, L.nx + L.nu)
        for at, error in enumerate(np.abs(values[0, L.nx:] - point[L.root])):
            update_worst(root_copy, error, (j, 0, at))
        for at, error in enumerate(np.abs(values[0, :L.nx] - fixture.branches[j].initial)):
            update_worst(initial_copy, error, (j, 0, at))
        for k in range(1, L.N):
            point[L.stage(j, k)] = values[k]
        point[L.x(j, L.N)] = trajectory[L.N * (L.nx + L.nu):]
        # Upstream uses +λ term with rows Ax+Bu+b−x_next. The generic
        # step uses −λ term with the opposite row, so λ itself is preserved.
        dynamics_dual[j * L.N * L.nx:(j + 1) * L.N * L.nx] = br["dynamics_dual"][L.nx:]
        y = np.asarray(br["inequality_dual"])
        splits = np.cumsum([len(s.lower) for s in fixture.branches[j].stages])
        if y.shape != (splits[-1],):
            raise ValueError("wrong native constraint dual size")
        native_bounds.append(np.split(y, splits[:-1]))
    return point, dynamics_dual, tuple(native_bounds), root_copy, initial_copy


def update_worst(worst, value, at):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"nonfinite residual at {at}")
    if value > worst["value"]:
        worst.update(value=value, at=at)


def maximum(values, labels):
    if not np.isfinite(values).all():
        raise ValueError("nonfinite residual vector")
    index = int(np.argmax(np.abs(values)))
    return {"value": float(abs(values[index])), "at": labels[index]}


def bound_dual_check(comp, sign, value, lo, hi, multiplier, at):
    # SQP convention is positive toward the lower side.
    if multiplier == 0:
        return
    side = lo if multiplier > 0 else hi
    if np.isfinite(side):
        update_worst(comp, abs(multiplier * (value - side)), at)
    else:
        update_worst(sign, abs(multiplier), at)


def evaluate(fixture, L, x, dual=None, box_dual=None, native_duals=None):
    """Independent evaluation from untransformed stage blocks and transition duals."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (L.n,) or not np.isfinite(x).all():
        raise ValueError("invalid primal")
    if dual is not None and (np.shape(dual) != (L.m,) or not np.isfinite(dual).all()):
        raise ValueError("invalid dynamics dual")
    if box_dual is not None and (np.shape(box_dual) != (L.n,) or not np.isfinite(box_dual).all()):
        raise ValueError("invalid signed bound dual")
    equality = np.empty(L.m)
    objective = fixture.objective_constant
    stationarity = np.zeros(L.n)
    worst = {name: {"value": 0.0, "at": None} for name in
             ("inequality", "complementarity", "dual_sign")}
    for j, branch in enumerate(fixture.branches):
        vector = list(stage_vectors(fixture, L, x, j))
        for k, (stage, z) in enumerate(zip(branch.stages, vector, strict=True)):
            hz = stage.hessian @ z
            objective += float((0.5 * hz + stage.gradient) @ z)
            grad = hz + stage.gradient
            outputs = stage.selectors @ z
            if native_duals is not None:
                y = native_duals[j][k]
                if not np.isfinite(y).all():
                    raise ValueError("nonfinite native box dual")
                grad = grad + stage.selectors.T @ y
            for i, (yval, lo, hi) in enumerate(zip(outputs, stage.lower, stage.upper, strict=True)):
                at = (j, k, i)
                update_worst(worst["inequality"], max(0.0, lo - yval, yval - hi), at)
                if native_duals is not None:
                    bound_dual_check(worst["complementarity"], worst["dual_sign"],
                                     yval, lo, hi, -y[i], at)
            if k < L.N:
                eq = L.eq(j, k)
                equality[eq] = vector[k + 1][:L.nx] - stage.dynamics @ z - stage.offset
                if dual is not None:
                    grad = grad + stage.dynamics.T @ dual[eq]
                    stationarity[L.x(j, k + 1)] -= dual[eq]
            stationarity[L.stage(j, k)] += grad[L.nx:] if k == 0 else grad
    if box_dual is not None:
        stationarity -= box_dual
    return objective, equality, stationarity, worst


def certificate(fixture, q, x, dual, *, box_dual=None, native_duals=None):
    x = np.asarray(x, dtype=np.float64)
    L = q.layout
    objective, equality, stationarity, worst = evaluate(fixture, L, x, dual, box_dual, native_duals)
    labels = L.labels()
    eq_labels = [(j, k, i) for j in range(L.J) for k in range(L.N) for i in range(L.nx)]
    if box_dual is not None:
        for i, (value, lo, hi, z) in enumerate(zip(x, q.lower, q.upper, box_dual, strict=True)):
            bound_dual_check(worst["complementarity"], worst["dual_sign"], value, lo, hi, z, labels[i])
    worst.update(equality=maximum(equality, eq_labels), stationarity=maximum(stationarity, labels))
    result = {"objective": objective, **worst}
    result["root_stationarity"] = maximum(stationarity[L.root], labels[L.root])
    result["forward_rollout"] = maximum(x - forward_rollout(fixture, L, x), labels)
    result["passed"] = (np.isfinite(objective) and
                        all(v["value"] <= (COMPLEMENTARITY_TOL if k == "complementarity" else ABS_TOL)
                            for k, v in worst.items()))
    return result


def aggregate_native_bounds(q, native_duals):
    z = np.zeros(q.layout.n)
    for origins, stages in zip(q.bound_origin, native_duals, strict=True):
        for rows, multipliers in zip(origins, stages, strict=True):
            for (column, coefficient), y in zip(rows, multipliers, strict=True):
                if column >= 0:
                    z[column] -= coefficient * y
    return z
