//! PyO3 binding. The public Python API lives in `python/sqp_craft/__init__.py`;
//! this module only converts data and runs the Rust solver.
//!
//! Native phases release the interpreter. Grouped callbacks reattach on the
//! calling thread; each solve owns its callback state.

use pyo3::create_exception;
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyString, PyTuple};
use sqp_craft::{
    Bounds, CancellationToken, Control, CsrPattern, DummySolver, Error, NlpProblem, Observer,
    Options, ProblemSpec, Progress, SqpSolver,
};

create_exception!(
    sqp_craft,
    EvaluationError,
    PyRuntimeError,
    "The model failed or returned malformed or non-finite values."
);
create_exception!(
    sqp_craft,
    UndefinedPointError,
    EvaluationError,
    "Raised by a model that is not defined at the requested point."
);

fn to_py_err(e: Error) -> PyErr {
    let message = e.message().to_owned();
    match e {
        Error::InvalidInput(_) => PyValueError::new_err(message),
        Error::Evaluation(_) => EvaluationError::new_err(message),
        Error::UndefinedPoint(_) => UndefinedPointError::new_err(message),
        Error::Unsupported(_) => PyNotImplementedError::new_err(message),
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}

fn reject_text(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<()> {
    if obj.is_instance_of::<PyString>() || obj.is_instance_of::<PyBytes>() {
        return Err(PyTypeError::new_err(format!(
            "{what}: expected a sequence of numbers, got {}",
            obj.get_type().name()?
        )));
    }
    Ok(())
}

/// Copies an iterable of numbers (list, tuple, 1-D NumPy array, ...) into
/// `out`, returning how many items it had. Items beyond `out.len()` are counted
/// but not stored.
fn copy_floats(obj: &Bound<'_, PyAny>, out: &mut [f64], what: &str) -> PyResult<usize> {
    reject_text(obj, what)?;
    let mut count = 0;
    for item in obj.try_iter()? {
        let value: f64 = item?.extract()?;
        if let Some(slot) = out.get_mut(count) {
            *slot = value;
        }
        count += 1;
    }
    Ok(count)
}

fn copy_exact(obj: &Bound<'_, PyAny>, out: &mut [f64], what: &str) -> PyResult<()> {
    let count = copy_floats(obj, out, what)?;
    if count != out.len() {
        return Err(EvaluationError::new_err(format!(
            "{what}: expected {} values, got {count}",
            out.len()
        )));
    }
    Ok(())
}

fn floats(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<f64>> {
    reject_text(obj, what)?;
    obj.try_iter()?.map(|item| item?.extract()).collect()
}

fn indices(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<usize>> {
    reject_text(obj, what)?;
    obj.try_iter()?.map(|item| item?.extract()).collect()
}

fn pair<'py>(
    ret: &Bound<'py, PyAny>,
    what: &str,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    match ret.cast::<PyTuple>() {
        Ok(t) if t.len() == 2 => Ok((t.get_item(0)?, t.get_item(1)?)),
        _ => Err(PyTypeError::new_err(format!(
            "{what} must return a tuple of two items, got {}",
            ret.get_type().name()?
        ))),
    }
}

/// Validated problem structure, shared by any number of solves.
#[pyclass(frozen, module = "sqp_craft", name = "ProblemSpec")]
struct PyProblemSpec {
    inner: ProblemSpec,
}

#[pymethods]
impl PyProblemSpec {
    #[new]
    #[pyo3(signature = (*, lower, upper, constraint_lower, constraint_upper, jacobian_indptr, jacobian_indices))]
    fn new(
        lower: &Bound<'_, PyAny>,
        upper: &Bound<'_, PyAny>,
        constraint_lower: &Bound<'_, PyAny>,
        constraint_upper: &Bound<'_, PyAny>,
        jacobian_indptr: &Bound<'_, PyAny>,
        jacobian_indices: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let variables = Bounds::new(floats(lower, "lower")?, floats(upper, "upper")?);
        let constraints = Bounds::new(
            floats(constraint_lower, "constraint_lower")?,
            floats(constraint_upper, "constraint_upper")?,
        );
        let jacobian = CsrPattern {
            rows: constraints.len(),
            cols: variables.len(),
            indptr: indices(jacobian_indptr, "jacobian_indptr")?,
            indices: indices(jacobian_indices, "jacobian_indices")?,
        };
        let inner = ProblemSpec::new(variables, constraints, jacobian).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Number of variables.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    /// Number of constraints.
    #[getter]
    fn m(&self) -> usize {
        self.inner.m()
    }

    /// Number of structural Jacobian nonzeros.
    #[getter]
    fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    #[getter]
    fn lower(&self) -> Vec<f64> {
        self.inner.variables().lower.clone()
    }

    #[getter]
    fn upper(&self) -> Vec<f64> {
        self.inner.variables().upper.clone()
    }

    #[getter]
    fn constraint_lower(&self) -> Vec<f64> {
        self.inner.constraints().lower.clone()
    }

    #[getter]
    fn constraint_upper(&self) -> Vec<f64> {
        self.inner.constraints().upper.clone()
    }

    #[getter]
    fn jacobian_indptr(&self) -> Vec<usize> {
        self.inner.jacobian().indptr.clone()
    }

    #[getter]
    fn jacobian_indices(&self) -> Vec<usize> {
        self.inner.jacobian().indices.clone()
    }

    /// Supports pickling, so a spec can be sent to worker processes.
    fn __getnewargs_ex__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyTuple>, Bound<'py, PyDict>)> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("lower", self.lower())?;
        kwargs.set_item("upper", self.upper())?;
        kwargs.set_item("constraint_lower", self.constraint_lower())?;
        kwargs.set_item("constraint_upper", self.constraint_upper())?;
        kwargs.set_item("jacobian_indptr", self.jacobian_indptr())?;
        kwargs.set_item("jacobian_indices", self.jacobian_indices())?;
        Ok((PyTuple::empty(py), kwargs))
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<PyProblemSpec>()
            .is_ok_and(|o| o.get().inner == self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "ProblemSpec(n={}, m={}, nnz={})",
            self.inner.n(),
            self.inner.m(),
            self.inner.nnz()
        )
    }
}

/// Cooperative cancellation flag; checked between model calls.
#[pyclass(frozen, module = "sqp_craft")]
#[derive(Default)]
struct Cancellation {
    inner: CancellationToken,
}

#[pymethods]
impl Cancellation {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Requests cancellation. Idempotent.
    fn cancel(&self) {
        self.inner.cancel();
    }

    /// Whether cancellation has been requested.
    #[getter]
    fn cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }

    fn __repr__(&self) -> String {
        format!("Cancellation(cancelled={})", self.cancelled())
    }
}

/// A Python exception raised by a callback, kept together with the Rust error
/// that stood in for it while the solver ran.
struct Failure {
    stand_in: Error,
    error: PyErr,
}

impl Failure {
    fn new(error: PyErr, observer: bool) -> Self {
        let stand_in = Python::attach(|py| {
            let message = error.value(py).to_string();
            if !observer && error.is_instance_of::<UndefinedPointError>(py) {
                Error::UndefinedPoint(message)
            } else {
                Error::Evaluation(message)
            }
        });
        Self { stand_in, error }
    }
}

/// Maps a solver result to Python. If the solver failed with the stand-in of
/// the exception raised by the most recent callback, that original exception
/// (type, identity, traceback) is raised. A recorded exception the solver
/// recovered from, such as an `UndefinedPointError` at a line-search trial,
/// does not match a successful result or a later error and is dropped.
fn reraise<T>(
    result: sqp_craft::Result<T>,
    failures: impl IntoIterator<Item = Option<Failure>>,
) -> PyResult<T> {
    match result {
        Ok(value) => Ok(value),
        Err(e) => {
            for failure in failures.into_iter().flatten() {
                if failure.stand_in == e {
                    return Err(failure.error);
                }
            }
            Err(to_py_err(e))
        }
    }
}

/// Adapts the two Python callbacks. Each call reattaches to the interpreter on
/// the calling thread and keeps only the exception of the most recent call.
struct PythonModel {
    functions: Py<PyAny>,
    derivatives: Py<PyAny>,
    failure: Option<Failure>,
}

impl PythonModel {
    fn new(functions: Py<PyAny>, derivatives: Py<PyAny>) -> Self {
        Self {
            functions,
            derivatives,
            failure: None,
        }
    }

    fn fail(&mut self, error: PyErr) -> Error {
        let failure = Failure::new(error, false);
        let stand_in = failure.stand_in.clone();
        self.failure = Some(failure);
        stand_in
    }
}

impl NlpProblem for PythonModel {
    fn functions(&mut self, x: &[f64], constraints: &mut [f64]) -> sqp_craft::Result<f64> {
        self.failure = None;
        let result = Python::attach(|py| {
            py.check_signals()?;
            let ret = self.functions.bind(py).call1((PyList::new(py, x)?,))?;
            let (objective, values) = pair(&ret, "functions(x)")?;
            copy_exact(&values, constraints, "functions(x) constraint values")?;
            objective.extract::<f64>()
        });
        result.map_err(|e| self.fail(e))
    }

    fn derivatives(
        &mut self,
        x: &[f64],
        gradient: &mut [f64],
        jacobian: &mut [f64],
    ) -> sqp_craft::Result<()> {
        self.failure = None;
        let result = Python::attach(|py| {
            py.check_signals()?;
            let ret = self.derivatives.bind(py).call1((PyList::new(py, x)?,))?;
            let (g, jac) = pair(&ret, "derivatives(x)")?;
            copy_exact(&g, gradient, "derivatives(x) gradient")?;
            copy_exact(&jac, jacobian, "derivatives(x) Jacobian values")
        });
        result.map_err(|e| self.fail(e))
    }
}

/// Calls a Python function `observe(progress: dict) -> bool` whose result
/// means "stop". The public facade adapts user observers to this form.
struct PythonObserver {
    callback: Py<PyAny>,
    failure: Option<Failure>,
}

impl Observer for PythonObserver {
    fn observe(&mut self, p: &Progress) -> sqp_craft::Result<Control> {
        self.failure = None;
        let result = Python::attach(|py| {
            py.check_signals()?;
            let data = PyDict::new(py);
            data.set_item("iteration", p.iteration)?;
            data.set_item("objective", p.objective)?;
            data.set_item("primal_violation", p.primal_violation)?;
            self.callback.bind(py).call1((data,))?.extract::<bool>()
        });
        match result {
            Ok(true) => Ok(Control::Stop),
            Ok(false) => Ok(Control::Continue),
            Err(e) => {
                let failure = Failure::new(e, true);
                let stand_in = failure.stand_in.clone();
                self.failure = Some(failure);
                Err(stand_in)
            }
        }
    }
}

/// Runs the selected backend (`"dummy"` or `"sqp"`). Use `sqp_craft.solve` instead of calling this directly.
#[pyfunction]
#[pyo3(signature = (spec, functions, derivatives, x0, feasibility_tolerance, threads, cancellation, observer=None, method="dummy", method_options=None))]
#[allow(clippy::too_many_arguments)]
fn solve_native<'py>(
    py: Python<'py>,
    spec: &Bound<'py, PyProblemSpec>,
    functions: Bound<'py, PyAny>,
    derivatives: Bound<'py, PyAny>,
    x0: &Bound<'py, PyAny>,
    feasibility_tolerance: f64,
    threads: usize,
    cancellation: &Bound<'py, Cancellation>,
    observer: Option<Bound<'py, PyAny>>,
    method: &str,
    method_options: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    for (name, f) in [("functions", &functions), ("derivatives", &derivatives)] {
        if !f.is_callable() {
            return Err(PyTypeError::new_err(format!(
                "model.{name} must be callable"
            )));
        }
    }
    if observer.as_ref().is_some_and(|o| !o.is_callable()) {
        return Err(PyTypeError::new_err("observer must be callable"));
    }
    let x0 = floats(x0, "x0")?;
    let options = Options {
        feasibility_tolerance,
        threads,
    };
    let mut model = PythonModel::new(functions.unbind(), derivatives.unbind());
    let mut monitor = observer.map(|callback| PythonObserver {
        callback: callback.unbind(),
        failure: None,
    });
    let inner = &spec.get().inner;
    let token = cancellation.get().inner.clone();
    let result = match method {
        "dummy" => {
            if method_options.is_some_and(|d| !d.is_empty()) {
                return Err(PyNotImplementedError::new_err(
                    "dummy does not support SQP settings",
                ));
            }
            py.detach(|| {
                DummySolver.solve(
                    inner,
                    &mut model,
                    &x0,
                    &options,
                    &token,
                    monitor.as_mut().map(|m| m as &mut dyn Observer),
                )
            })
        }
        "sqp" => {
            #[cfg(feature = "sqp")]
            {
                let mut settings = sqp_craft_solver::SqpOptions::default();
                let mut qp_solver = String::from("clarabel");
                if let Some(d) = method_options {
                    for (key, value) in d.iter() {
                        let key: String = key.extract()?;
                        match key.as_str() {
                            "max_iterations" => settings.max_iterations = value.extract()?,
                            "max_backtracks" => settings.max_backtracks = value.extract()?,
                            "max_callback_calls" => {
                                settings.max_callback_calls = value.extract()?
                            }
                            "time_limit_seconds" => {
                                settings.time_limit_seconds = value.extract()?
                            }
                            "stationarity_tolerance" => {
                                settings.stationarity_tolerance = value.extract()?
                            }
                            "complementarity_tolerance" => {
                                settings.complementarity_tolerance = value.extract()?
                            }
                            "dual_sign_tolerance" => {
                                settings.dual_sign_tolerance = value.extract()?
                            }
                            "dense_size_limit" => settings.dense_size_limit = value.extract()?,
                            "initial_hessian_scale" => {
                                settings.initial_hessian_scale = value.extract()?
                            }
                            "max_hessian_resets" => {
                                settings.max_hessian_resets = value.extract()?
                            }
                            "qp_tolerance" => settings.qp_tolerance = value.extract()?,
                            "qp_max_iterations" => settings.qp_max_iterations = value.extract()?,
                            "hessian_block_size" => {
                                settings.hessian_block_size = value.extract()?
                            }
                            "qp_reuse_workspace" => {
                                settings.qp_reuse_workspace = value.extract()?
                            }
                            "qp_warm_start" => settings.qp_warm_start = value.extract()?,
                            "elastic_mode" => settings.elastic_mode = value.extract()?,
                            "elastic_weight" => settings.elastic_weight = value.extract()?,
                            "second_order_correction" => {
                                settings.second_order_correction = value.extract()?
                            }
                            "qp_solver" => qp_solver = value.extract()?,
                            "record_qp_limit" => settings.record_qp_limit = value.extract()?,
                            "record_history" => settings.record_history = value.extract()?,
                            "max_restoration_steps" => {
                                settings.max_restoration_steps = value.extract()?
                            }
                            "restoration_step_limit" => {
                                settings.restoration_step_limit = value.extract()?
                            }
                            "restoration_regularization" => {
                                settings.restoration_regularization = value.extract()?
                            }
                            _ => {
                                return Err(PyValueError::new_err(format!(
                                    "unknown SQP setting: {key}"
                                )));
                            }
                        }
                    }
                }
                if !matches!(qp_solver.as_str(), "clarabel" | "active-set") {
                    return Err(PyValueError::new_err(format!(
                        "qp_solver must be \"clarabel\" or \"active-set\", got {qp_solver:?}"
                    )));
                }
                py.detach(|| {
                    let observer = monitor.as_mut().map(|m| m as &mut dyn Observer);
                    if qp_solver == "active-set" {
                        sqp_craft_solver::NativeSqp {
                            qp: sqp_craft_activeset::ActiveSetQp::default(),
                            settings,
                        }
                        .solve(inner, &mut model, &x0, &options, &token, observer)
                    } else {
                        sqp_craft_solver::NativeSqp {
                            qp: sqp_craft_clarabel::ClarabelQp::default(),
                            settings,
                        }
                        .solve(inner, &mut model, &x0, &options, &token, observer)
                    }
                })
            }
            #[cfg(not(feature = "sqp"))]
            {
                return Err(PyNotImplementedError::new_err(
                    "native SQP was not compiled",
                ));
            }
        }
        _ => return Err(PyValueError::new_err(format!("unknown method: {method}"))),
    };
    let r = reraise(result, [model.failure, monitor.and_then(|m| m.failure)])?;

    let out = PyDict::new(py);
    out.set_item("status", r.status.as_str())?;
    out.set_item("candidate", r.candidate)?;
    out.set_item("iterations", r.iterations)?;
    out.set_item("stationarity", r.stationarity)?;
    out.set_item("dual_sign", r.dual_sign)?;
    let report = PyDict::new(py);
    report.set_item("method", &r.report.method)?;
    report.set_item("qp_backend", &r.report.qp_backend)?;
    report.set_item("qp_status", &r.report.qp_status)?;
    report.set_item("qp_solves", r.report.qp_solves)?;
    report.set_item("qp_iterations", r.report.qp_iterations)?;
    report.set_item("rejected_trials", r.report.rejected_trials)?;
    report.set_item("undefined_trials", r.report.undefined_trials)?;
    report.set_item("hessian_resets", r.report.hessian_resets)?;
    report.set_item("block_hessian_fallbacks", r.report.block_hessian_fallbacks)?;
    report.set_item(
        "qp_not_positive_definite",
        r.report.qp_not_positive_definite,
    )?;
    report.set_item("elastic_steps", r.report.elastic_steps)?;
    report.set_item("elastic_weight", r.report.elastic_weight)?;
    report.set_item(
        "elastic_weight_increases",
        r.report.elastic_weight_increases,
    )?;
    report.set_item(
        "second_order_corrections",
        r.report.second_order_corrections,
    )?;
    report.set_item("second_order_steps", r.report.second_order_steps)?;
    report.set_item("fixed_step_corrections", r.report.fixed_step_corrections)?;
    report.set_item("bound_step_corrections", r.report.bound_step_corrections)?;
    report.set_item("step_norm", r.report.step_norm)?;
    report.set_item("total_seconds", r.report.total_seconds)?;
    report.set_item("callback_seconds", r.report.callback_seconds)?;
    report.set_item("qp_seconds", r.report.qp_seconds)?;
    report.set_item("qp_setups", r.report.qp_setups)?;
    report.set_item("qp_updates", r.report.qp_updates)?;
    report.set_item("qp_cold_retries", r.report.qp_cold_retries)?;
    report.set_item("qp_retry_reason", &r.report.qp_retry_reason)?;
    report.set_item("qp_conversion_seconds", r.report.qp_conversion_seconds)?;
    report.set_item("qp_setup_seconds", r.report.qp_setup_seconds)?;
    report.set_item("qp_update_seconds", r.report.qp_update_seconds)?;
    report.set_item("qp_solve_seconds", r.report.qp_solve_seconds)?;
    report.set_item("qp_check_seconds", r.report.qp_check_seconds)?;
    report.set_item("bfgs_seconds", r.report.bfgs_seconds)?;
    let trace = PyList::empty(py);
    for q in &r.report.qp_trace {
        let d = PyDict::new(py);
        d.set_item("gradient", &q.gradient)?;
        d.set_item("lower", &q.variables.lower)?;
        d.set_item("upper", &q.variables.upper)?;
        d.set_item("row_lower", &q.rows.lower)?;
        d.set_item("row_upper", &q.rows.upper)?;
        for (name, a) in [("hessian_lower", &q.hessian_lower), ("matrix", &q.matrix)] {
            let m = PyDict::new(py);
            m.set_item("rows", a.pattern.rows)?;
            m.set_item("cols", a.pattern.cols)?;
            m.set_item("indptr", &a.pattern.indptr)?;
            m.set_item("indices", &a.pattern.indices)?;
            m.set_item("values", &a.values)?;
            d.set_item(name, m)?;
        }
        trace.append(d)?;
    }
    report.set_item("qp_trace", trace)?;
    let history = PyList::empty(py);
    for h in &r.report.history {
        let d = PyDict::new(py);
        d.set_item("iteration", h.iteration)?;
        d.set_item("objective", h.objective)?;
        d.set_item("primal_violation", h.primal_violation)?;
        d.set_item("stationarity", h.stationarity)?;
        d.set_item("complementarity", h.complementarity)?;
        d.set_item("qp_step_norm", h.qp_step_norm)?;
        d.set_item("step_length", h.step_length)?;
        d.set_item("trials", h.trials)?;
        d.set_item("max_penalty", h.max_penalty)?;
        d.set_item("qp_iterations", h.qp_iterations)?;
        d.set_item("second_order_correction", h.second_order_correction)?;
        history.append(d)?;
    }
    report.set_item("history", history)?;
    report.set_item("restoration_attempts", r.report.restoration_attempts)?;
    report.set_item("restoration_steps", r.report.restoration_steps)?;
    report.set_item("restoration_seconds", r.report.restoration_seconds)?;
    let restoration = PyList::empty(py);
    for h in &r.report.restoration_history {
        let d = PyDict::new(py);
        d.set_item("iteration", h.iteration)?;
        d.set_item("objective_before", h.objective_before)?;
        d.set_item("objective_after", h.objective_after)?;
        d.set_item("primal_before", h.primal_before)?;
        d.set_item("primal_after", h.primal_after)?;
        d.set_item("merit_before", h.merit_before)?;
        d.set_item("merit_after", h.merit_after)?;
        d.set_item("step_length", h.step_length)?;
        d.set_item("trials", h.trials)?;
        d.set_item("qp_iterations", h.qp_iterations)?;
        restoration.append(d)?;
    }
    report.set_item("restoration_history", restoration)?;
    out.set_item("report", report)?;
    out.set_item("complementarity", r.complementarity)?;
    out.set_item("message", r.message)?;
    let counts = PyDict::new(py);
    counts.set_item("functions", r.counts.functions)?;
    counts.set_item("derivatives", r.counts.derivatives)?;
    out.set_item("counts", counts)?;
    match r.evaluation {
        Some(e) => {
            let d = PyDict::new(py);
            d.set_item("objective", e.objective)?;
            d.set_item("constraints", e.constraints)?;
            d.set_item("gradient", e.gradient)?;
            d.set_item("jacobian_values", e.jacobian_values)?;
            d.set_item("primal_violation", e.primal_violation)?;
            d.set_item("feasible_candidate", e.feasible_candidate)?;
            out.set_item("evaluation", d)?;
        }
        None => out.set_item("evaluation", py.None())?,
    }
    match r.multipliers {
        Some(mult) => {
            let d = PyDict::new(py);
            d.set_item("constraints", mult.constraints)?;
            d.set_item("variables", mult.variables)?;
            out.set_item("multipliers", d)?;
        }
        None => out.set_item("multipliers", py.None())?,
    }
    Ok(out)
}

/// Bound-aware derivative check using the same grouped Python callback adapter.
#[pyfunction]
#[pyo3(signature=(spec, functions, derivatives, x, steps, absolute_tolerance, relative_tolerance, max_directions))]
#[allow(clippy::too_many_arguments)]
fn check_derivatives<'py>(
    py: Python<'py>,
    spec: &Bound<'py, PyProblemSpec>,
    functions: Py<PyAny>,
    derivatives: Py<PyAny>,
    x: &Bound<'py, PyAny>,
    steps: Vec<f64>,
    absolute_tolerance: f64,
    relative_tolerance: f64,
    max_directions: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let x = floats(x, "x")?;
    let inner = &spec.get().inner;
    let options = sqp_craft::derivative_check::DerivativeOptions {
        steps,
        absolute_tolerance,
        relative_tolerance,
        max_directions,
    };
    let mut model = PythonModel::new(functions, derivatives);
    let result = py
        .detach(|| sqp_craft::derivative_check::check_derivatives(inner, &mut model, &x, &options));
    let r = reraise(result, [model.failure])?;
    let out = PyDict::new(py);
    out.set_item("passed", r.passed)?;
    out.set_item("inconclusive", r.inconclusive)?;
    out.set_item("checked", r.checked)?;
    out.set_item("fixed", r.fixed)?;
    out.set_item("skipped", r.skipped)?;
    let counts = PyDict::new(py);
    counts.set_item("functions", r.counts.functions)?;
    counts.set_item("derivatives", r.counts.derivatives)?;
    out.set_item("counts", counts)?;
    let issues = PyList::empty(py);
    for issue in r.issues {
        let d = PyDict::new(py);
        d.set_item("variable", issue.variable)?;
        d.set_item("output", issue.output)?;
        d.set_item("analytic", issue.analytic)?;
        d.set_item("estimated", issue.estimated)?;
        d.set_item("scaled_error", issue.scaled_error)?;
        issues.append(d)?;
    }
    out.set_item("issues", issues)?;
    Ok(out)
}
/// Independent original-unit NLP KKT residuals.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn kkt_residuals<'py>(
    py: Python<'py>,
    spec: &Bound<'py, PyProblemSpec>,
    x: Vec<f64>,
    objective: f64,
    constraints: Vec<f64>,
    gradient: Vec<f64>,
    jacobian_values: Vec<f64>,
    row_multipliers: Vec<f64>,
    variable_multipliers: Vec<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let inner = &spec.get().inner;
    let e = sqp_craft::Evaluation {
        objective,
        constraints,
        gradient,
        jacobian_values,
        primal_violation: 0.0,
        feasible_candidate: false,
    };
    let m = sqp_craft::Multipliers {
        constraints: row_multipliers,
        variables: variable_multipliers,
    };
    let r = py
        .detach(|| sqp_craft::kkt::nlp_residuals(inner, &x, &e, &m))
        .map_err(to_py_err)?;
    let d = PyDict::new(py);
    d.set_item("primal", r.primal)?;
    d.set_item("stationarity", r.stationarity)?;
    d.set_item("complementarity", r.complementarity)?;
    d.set_item("dual_sign", r.dual_sign)?;
    Ok(d)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_class::<PyProblemSpec>()?;
    m.add_class::<Cancellation>()?;
    m.add("EvaluationError", py.get_type::<EvaluationError>())?;
    m.add("UndefinedPointError", py.get_type::<UndefinedPointError>())?;
    m.add_function(wrap_pyfunction!(solve_native, m)?)?;
    m.add_function(wrap_pyfunction!(check_derivatives, m)?)?;
    m.add_function(wrap_pyfunction!(kkt_residuals, m)?)?;
    Ok(())
}
