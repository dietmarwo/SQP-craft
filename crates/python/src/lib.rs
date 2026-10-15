//! PyO3 binding. The public Python API lives in `python/sqp_craft/__init__.py`;
//! this module only converts data and runs the Rust solver.
//!
//! Model callbacks run serially on the calling thread with the interpreter
//! attached. No Python object is ever handed to another thread.

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

/// Adapts the two Python callbacks. The first Python exception is kept so the
/// caller sees the original type and traceback.
struct PythonModel<'py> {
    functions: Bound<'py, PyAny>,
    derivatives: Bound<'py, PyAny>,
    error: Option<PyErr>,
}

impl PythonModel<'_> {
    fn keep(&mut self, err: PyErr) -> Error {
        let py = self.functions.py();
        let message = err.value(py).to_string();
        let kind = if err.is_instance_of::<UndefinedPointError>(py) {
            Error::UndefinedPoint(message)
        } else {
            Error::Evaluation(message)
        };
        self.error = Some(err);
        kind
    }
}

impl NlpProblem for PythonModel<'_> {
    fn functions(&mut self, x: &[f64], constraints: &mut [f64]) -> sqp_craft::Result<f64> {
        let py = self.functions.py();
        let result = (|| {
            let ret = self.functions.call1((PyList::new(py, x)?,))?;
            let (objective, values) = pair(&ret, "functions(x)")?;
            copy_exact(&values, constraints, "functions(x) constraint values")?;
            objective.extract::<f64>()
        })();
        result.map_err(|e| self.keep(e))
    }

    fn derivatives(
        &mut self,
        x: &[f64],
        gradient: &mut [f64],
        jacobian: &mut [f64],
    ) -> sqp_craft::Result<()> {
        let py = self.derivatives.py();
        let result = (|| {
            let ret = self.derivatives.call1((PyList::new(py, x)?,))?;
            let (g, jac) = pair(&ret, "derivatives(x)")?;
            copy_exact(&g, gradient, "derivatives(x) gradient")?;
            copy_exact(&jac, jacobian, "derivatives(x) Jacobian values")
        })();
        result.map_err(|e| self.keep(e))
    }
}

/// Calls a Python function `observe(progress: dict) -> bool` whose result
/// means "stop". The public facade adapts user observers to this form.
struct PythonObserver<'py> {
    callback: Bound<'py, PyAny>,
    error: Option<PyErr>,
}

impl Observer for PythonObserver<'_> {
    fn observe(&mut self, p: &Progress) -> sqp_craft::Result<Control> {
        let py = self.callback.py();
        let result = (|| {
            let data = PyDict::new(py);
            data.set_item("iteration", p.iteration)?;
            data.set_item("objective", p.objective)?;
            data.set_item("primal_violation", p.primal_violation)?;
            self.callback.call1((data,))?.extract::<bool>()
        })();
        match result {
            Ok(true) => Ok(Control::Stop),
            Ok(false) => Ok(Control::Continue),
            Err(e) => {
                let message = e.value(py).to_string();
                self.error = Some(e);
                Err(Error::Evaluation(format!("observer failed: {message}")))
            }
        }
    }
}

/// Runs the dummy backend. Use `sqp_craft.solve` instead of calling this directly.
#[pyfunction]
#[pyo3(signature = (spec, functions, derivatives, x0, feasibility_tolerance, threads, cancellation, observer=None))]
#[allow(clippy::too_many_arguments)]
fn solve_dummy<'py>(
    py: Python<'py>,
    spec: &Bound<'py, PyProblemSpec>,
    functions: Bound<'py, PyAny>,
    derivatives: Bound<'py, PyAny>,
    x0: &Bound<'py, PyAny>,
    feasibility_tolerance: f64,
    threads: usize,
    cancellation: &Bound<'py, Cancellation>,
    observer: Option<Bound<'py, PyAny>>,
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
    let mut model = PythonModel {
        functions,
        derivatives,
        error: None,
    };
    let mut monitor = observer.map(|callback| PythonObserver {
        callback,
        error: None,
    });
    let result = DummySolver.solve(
        &spec.get().inner,
        &mut model,
        &x0,
        &options,
        &cancellation.get().inner,
        monitor.as_mut().map(|m| m as &mut dyn Observer),
    );
    if let Some(e) = model.error {
        return Err(e);
    }
    if let Some(e) = monitor.and_then(|m| m.error) {
        return Err(e);
    }
    let r = result.map_err(to_py_err)?;

    let out = PyDict::new(py);
    out.set_item("status", r.status.as_str())?;
    out.set_item("candidate", r.candidate)?;
    out.set_item("iterations", r.iterations)?;
    out.set_item("stationarity", r.stationarity)?;
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

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_class::<PyProblemSpec>()?;
    m.add_class::<Cancellation>()?;
    m.add("EvaluationError", py.get_type::<EvaluationError>())?;
    m.add("UndefinedPointError", py.get_type::<UndefinedPointError>())?;
    m.add_function(wrap_pyfunction!(solve_dummy, m)?)?;
    Ok(())
}
