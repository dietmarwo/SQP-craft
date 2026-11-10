//! Element-wise BFGS for partially separable Lagrangians.

use super::*;
use hessian::{HessianModel, Update};
use nalgebra::DMatrix;
use sqp_craft::{CurvatureElementReport, StructureReport, sparse};

struct Layout {
    rows: Vec<usize>,
    columns: Vec<usize>,
}

/// Group rows by maximal nested supports, with stable tie breaking.
fn row_elements(spec: &ProblemSpec, nonlinear: impl Fn(usize) -> bool) -> Vec<Layout> {
    let j = spec.jacobian();
    if let Some(groups) = spec.elements() {
        return groups
            .iter()
            .filter_map(|group| {
                let rows: Vec<_> = group.iter().copied().filter(|&i| nonlinear(i)).collect();
                let mut columns: Vec<_> = rows
                    .iter()
                    .flat_map(|&i| j.row(i).iter().copied())
                    .collect();
                columns.sort_unstable();
                columns.dedup();
                (!rows.is_empty()).then_some(Layout { rows, columns })
            })
            .collect();
    }
    let mut rows: Vec<_> = (0..spec.m()).filter(|&i| nonlinear(i)).collect();
    rows.sort_unstable_by_key(|&i| (std::cmp::Reverse(j.row(i).len()), i));
    let mut groups: Vec<Layout> = Vec::new();
    let mut owners = vec![Vec::<usize>::new(); spec.n()];
    for i in rows {
        let support = j.row(i);
        let &rarest = support
            .iter()
            .min_by_key(|&&col| owners[col].len())
            .expect("nonempty row");
        if let Some(g) = owners[rarest].iter().copied().find(|&g| {
            support
                .iter()
                .all(|col| groups[g].columns.binary_search(col).is_ok())
        }) {
            groups[g].rows.push(i);
        } else {
            for &col in support {
                owners[col].push(groups.len());
            }
            groups.push(Layout {
                rows: vec![i],
                columns: support.to_vec(),
            });
        }
    }
    for g in &mut groups {
        g.rows.sort_unstable();
    }
    groups
}

struct Element {
    columns: Vec<usize>,
    initial_diagonal: Vec<f64>,
    matrix: DMatrix<f64>,
    work: bfgs::Workspace,
    scatter: Vec<usize>,
    jacobian: CsrPattern,
    sources: Vec<usize>,
    change: Vec<f64>,
    departure: Vec<f64>,
    s: Vec<f64>,
    y: Vec<f64>,
    report: CurvatureElementReport,
}

impl Element {
    fn new(layout: Layout, diagonal: Vec<f64>, h: &CsrPattern, j: &CsrPattern) -> Self {
        let Layout { rows, columns } = layout;
        let k = columns.len();
        let mut scatter = Vec::with_capacity(k * (k + 1) / 2);
        for (i, &row) in columns.iter().enumerate() {
            for col in &columns[..=i] {
                scatter
                    .push(h.indptr[row] + h.row(row).binary_search(col).expect("element clique"));
            }
        }
        let mut indices = Vec::new();
        let mut sources = Vec::new();
        let mut indptr = vec![0];
        for &i in &rows {
            for a in j.indptr[i]..j.indptr[i + 1] {
                indices.push(columns.binary_search(&j.indices[a]).expect("row support"));
                sources.push(a);
            }
            indptr.push(indices.len());
        }
        let matrix = DMatrix::from_diagonal(&nalgebra::DVectorView::from_slice(&diagonal, k));
        Self {
            columns,
            initial_diagonal: diagonal,
            matrix,
            work: bfgs::Workspace::new(k, true),
            scatter,
            jacobian: CsrPattern {
                rows: rows.len(),
                cols: k,
                indptr,
                indices,
            },
            change: vec![0.0; sources.len()],
            sources,
            departure: vec![0.0; rows.len()],
            s: vec![0.0; k],
            y: vec![0.0; k],
            report: CurvatureElementReport {
                rows,
                ..Default::default()
            },
        }
    }

    fn reset(&mut self) {
        self.matrix.fill(0.0);
        for (i, &v) in self.initial_diagonal.iter().enumerate() {
            self.matrix[(i, i)] = v;
        }
    }

    fn update(&mut self, data: &Update<'_>) -> CurvatureUpdate {
        for (local, &global) in self.columns.iter().enumerate() {
            self.s[local] = data.step[global] / data.scales.variables[global];
        }
        self.y.fill(0.0);
        if self.report.rows.is_empty() {
            for (i, &j) in self.columns.iter().enumerate() {
                self.y[i] = data.next.gradient[j] - data.previous.gradient[j];
            }
        }
        let augmented = data.strategy == HessianUpdate::Augmented && !self.report.rows.is_empty();
        for (local, &row) in self.report.rows.iter().enumerate() {
            let mut predicted = 0.0;
            let lambda = data.multipliers.constraints[row];
            for a in self.jacobian.indptr[local]..self.jacobian.indptr[local + 1] {
                let source = self.sources[a];
                let col = self.jacobian.indices[a];
                let global = self.columns[col];
                let before = data.previous.jacobian_values[source];
                let delta = data.next.jacobian_values[source] - before;
                // The repository's convention is ∇f − Jᵀλ − z.
                self.y[col] -= lambda * delta;
                if augmented {
                    predicted += before * data.step[global];
                    self.change[a] =
                        delta * (data.scales.rows[row] * data.scales.variables[global]);
                }
            }
            if augmented {
                self.departure[local] =
                    (data.next.constraints[row] - data.previous.constraints[row] - predicted)
                        * data.scales.rows[row];
            }
        }
        for (y, &j) in self.y.iter_mut().zip(&self.columns) {
            *y *= data.scales.variables[j];
        }
        let repair = augmented.then_some(bfgs::Repair {
            pattern: &self.jacobian,
            change: &self.change,
            departure: &self.departure,
        });
        let update = self.work.update(
            &mut self.matrix,
            &self.s,
            &self.y,
            repair.as_ref().map(|r| (r, 0)),
        );
        match update {
            Some(CurvatureUpdate::Negligible) => self.report.skipped += 1,
            Some(kind) => {
                self.report.updates += 1;
                self.report.damped += u64::from(kind == CurvatureUpdate::Damped);
                self.report.modified += u64::from(kind == CurvatureUpdate::Modified);
            }
            None => {
                self.reset();
                self.report.resets += 1;
            }
        }
        update.unwrap_or(CurvatureUpdate::Reset)
    }
}

pub(super) struct Partitioned {
    elements: Vec<Element>,
    lower: CsrMatrix,
    ridge: Vec<(usize, f64)>,
    report: StructureReport,
    fallback: bool,
}

impl Partitioned {
    pub fn new(spec: &ProblemSpec, settings: &SqpOptions) -> Result<Self> {
        let n = spec.n();
        let limit = settings.structured_memory_limit_bytes;
        if n > settings.dense_size_limit && spec.objective_pattern().is_none() {
            return Err(Error::InvalidInput(
                "partitioned solves above the dense limit need an explicit objective_pattern (empty for affine)"
                    .into(),
            ));
        }
        // Jacobian, local row maps, solver vectors, and bounds. Check before
        // constructing groups, and charge element matrices separately below.
        let mut fixed = spec
            .nnz()
            .saturating_mul(64)
            .saturating_add(n.saturating_add(spec.m()).saturating_mul(192));
        sparse::check_memory(fixed, limit)?;
        let bounds = spec.constraints();
        let mut groups = row_elements(spec, |i| {
            !spec.linear_rows()[i]
                && !spec.jacobian().row(i).is_empty()
                && (bounds.lower[i].is_finite() || bounds.upper[i].is_finite())
        });
        let objective = spec
            .objective_pattern()
            .map_or_else(|| (0..n).collect(), <[usize]>::to_vec);
        if !objective.is_empty() {
            groups.push(Layout {
                rows: vec![],
                columns: objective,
            });
        }
        let mut degree = vec![0_usize; n];
        for group in &groups {
            let k = group.columns.len();
            fixed = fixed
                .saturating_add(k.saturating_mul(k).saturating_mul(40))
                .saturating_add(384);
            sparse::check_memory(fixed, limit)?;
            for &j in &group.columns {
                degree[j] += 1;
            }
        }
        let budget = limit - fixed;
        let mut graph = sparse::Graph::new(n, budget)?;
        for group in &groups {
            graph.clique(&group.columns)?;
        }
        let pattern = graph.into_pattern();
        if n > settings.dense_size_limit && pattern.nnz() == n.saturating_mul(n + 1) / 2 {
            return Err(Error::Unsupported(
                "structured Hessian pattern is dense".into(),
            ));
        }
        // Trace entries retain full QP models, including the bound arrays.
        let trace_bytes = settings.record_qp_limit.saturating_mul(
            pattern
                .nnz()
                .saturating_add(spec.nnz())
                .saturating_add(n)
                .saturating_add(spec.m())
                .saturating_mul(24),
        );
        fixed = fixed.saturating_add(trace_bytes);
        sparse::check_memory(fixed, limit)?;
        let budget = limit - fixed;
        let hessian_factor_nnz = sparse::Symbolic::of(&pattern, budget)?.nnz();
        let kkt_factor_nnz =
            sparse::kkt_fill(&pattern, spec.jacobian(), bounds, spec.variables(), budget)?;
        let estimated_memory_bytes =
            fixed.saturating_add(sparse::factor_bytes(hessian_factor_nnz.max(kkt_factor_nnz)));
        let scale = settings.initial_hessian_scale;
        let elements = groups
            .into_iter()
            .map(|group| {
                let diagonal: Vec<_> = group
                    .columns
                    .iter()
                    .map(|&j| scale / degree[j] as f64)
                    .collect();
                if diagonal.contains(&0.0) {
                    return Err(Error::InvalidInput(
                        "initial_hessian_scale underflows when split across overlapping elements"
                            .into(),
                    ));
                }
                Ok(Element::new(group, diagonal, &pattern, spec.jacobian()))
            })
            .collect::<Result<_>>()?;
        let ridge = degree
            .iter()
            .enumerate()
            .filter(|&(_, &d)| d == 0)
            .map(|(i, _)| (pattern.indptr[i + 1] - 1, scale))
            .collect();
        let report = StructureReport {
            hessian_nnz: pattern.nnz(),
            hessian_factor_nnz,
            kkt_factor_nnz,
            estimated_memory_bytes,
            ..Default::default()
        };
        let mut model = Self {
            elements,
            lower: CsrMatrix {
                values: vec![0.0; pattern.nnz()],
                pattern,
            },
            ridge,
            report,
            fallback: n <= settings.dense_size_limit,
        };
        model.assemble();
        Ok(model)
    }

    fn assemble(&mut self) {
        self.lower.values.fill(0.0);
        for &(slot, scale) in &self.ridge {
            self.lower.values[slot] = scale;
        }
        for e in &self.elements {
            let mut scatter = e.scatter.iter();
            for i in 0..e.columns.len() {
                for j in 0..=i {
                    self.lower.values[*scatter.next().expect("element scatter")] +=
                        e.matrix[(i, j)];
                }
            }
        }
    }
}

impl HessianModel for Partitioned {
    fn lower_csr(&self) -> CsrMatrix {
        self.lower.clone()
    }

    fn refresh_lower(&self, lower: &mut CsrMatrix) {
        lower.values.copy_from_slice(&self.lower.values);
    }

    fn product(&self, p: &[f64], out: &mut [f64]) {
        self.lower.symmetric_lower_product(p, out);
    }

    fn update(&mut self, data: &Update<'_>) -> Result<Option<CurvatureUpdate>> {
        if let Some(pattern) = data.spec.objective_pattern() {
            for (j, (&before, &after)) in data
                .previous
                .gradient
                .iter()
                .zip(&data.next.gradient)
                .enumerate()
            {
                if before != after && pattern.binary_search(&j).is_err() {
                    return Err(Error::InvalidInput(format!(
                        "objective_pattern: objective gradient changed outside the declared support at column {j}"
                    )));
                }
            }
        }
        let mut kind = CurvatureUpdate::Negligible;
        let rank = |kind| match kind {
            CurvatureUpdate::Negligible => 0,
            CurvatureUpdate::Exact => 1,
            CurvatureUpdate::Damped => 2,
            CurvatureUpdate::Modified => 3,
            _ => 4,
        };
        for element in &mut self.elements {
            let updated = element.update(data);
            if rank(updated) > rank(kind) {
                kind = updated;
            }
        }
        self.assemble();
        if self.lower.values.iter().any(|v| !v.is_finite()) {
            return Ok(None);
        }
        Ok(Some(kind))
    }

    fn reset(&mut self) {
        for element in &mut self.elements {
            element.reset();
        }
        self.assemble();
    }

    fn can_fallback(&self) -> bool {
        self.fallback
    }

    fn structured_pattern(&self) -> Option<&CsrPattern> {
        Some(&self.lower.pattern)
    }

    fn structure_report(&self) -> Option<StructureReport> {
        Some(StructureReport {
            elements: self.elements.iter().map(|e| e.report.clone()).collect(),
            ..self.report.clone()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disjoint_cliques_match_uniform_blocks_on_identical_secants() {
        let n = 4;
        let spec = ProblemSpec::new(
            Bounds::free(n),
            Bounds::fixed(vec![0.0; 2]),
            CsrPattern {
                rows: 2,
                cols: n,
                indptr: vec![0, 2, 4],
                indices: vec![0, 1, 2, 3],
            },
        )
        .unwrap()
        .with_objective_pattern(&[])
        .unwrap();
        let eval = |jacobian_values| Evaluation {
            objective: 0.0,
            constraints: vec![0.0; 2],
            gradient: vec![0.0; n],
            jacobian_values,
            primal_violation: 0.0,
            feasible_candidate: true,
        };
        let old = eval(vec![0.0; n]);
        let m = Multipliers {
            constraints: vec![-1.0; 2],
            variables: vec![0.0; n],
        };
        let scales = Scales::identity(n, 2);
        let settings = SqpOptions::default();
        let mut sparse = Partitioned::new(&spec, &settings).unwrap();
        let mut blocks = Legacy::new(n, 2, 1.0);
        assert_eq!(sparse.lower_csr(), blocks.lower_csr());
        for (s, y) in [
            ([1., 1., 1., 1.], vec![2., 4., 1., 3.]),
            ([0., 1., -1., 1.], vec![1., 2., -2., 5.]),
            ([1., 0., 2., -1.], vec![2., 1., 2., -5.]),
        ] {
            let next = eval(y);
            let update = Update {
                spec: &spec,
                previous: &old,
                next: &next,
                multipliers: &m,
                step: &s,
                scales: &scales,
                scaled_identity: true,
                strategy: HessianUpdate::Damped,
            };
            assert_eq!(
                sparse.update(&update).unwrap(),
                blocks.update(&update).unwrap()
            );
            assert_eq!(sparse.lower_csr(), blocks.lower_csr());
        }
    }
}
