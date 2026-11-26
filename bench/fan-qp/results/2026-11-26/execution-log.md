# SQP-craft and FanCyQPALM spring–mass experiment

**26 November 2026**

The sparse representation passed on the published four-branch spring–mass case. SQP-craft’s Clarabel adapter and the pinned FanCyQPALM reference both passed the independent, original-coordinate KKT checks. Their recomputed objectives differed by **4.36 × 10⁻¹⁰ relative**. The first exact-Hessian SQP step used the expected sparse matrices and translated bounds.

Keeping the shared input until last produced the **same symbolic fill as unrestricted minimum degree**. Using a fixed reverse-stage order increased factor entries by **17.5%** on the same graph. The representation is ready for an opt-in prototype and the controlled four- and eight-core study. This run establishes correctness and storage behavior on one convex instance.

## Fixture, revisions and build

| Item | Executed configuration |
|---|---|
| SQP-craft | Workspace 0.20.0 at `a87fde4ef8904198f8ba959d047348f3934a7d25`, with the included benchmark patch |
| FanCyqlone | `06f4bcfbab56d0fa9bbd82ad871929a3f0a464b1`, with the separate fixture-export patch |
| Published case | Figure 3, `wang-boyd-2008`, 12 masses, 32 intervals, 4 branches, seed 0. Shared perturbed initial state from the C++ benchmark wrapper. |
| Reduced model | 3,822 variables, 3,072 dynamics equalities, 2,286 two-sided boxes, one shared six-input root |
| Reference run | Scalar binary64, serial branches, one worker pinned to logical CPU 0 of the EPYC guest. Zero-start FanCyQPALM. |
| Toolchain | GCC 14.2, CMake 4.4.5, Ninja 1.13.2, reference BLAS/LAPACK 3.12.1. Rust 1.98.0, Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0. |

The original toy-only experiment had stopped during the batmat build. This run compiled the reference with scalar double kernels and used the native fixture generator directly. The first system-BLAS build needed CBLAS and LAPACK headers, then exposed the triangular-GEMM spelling difference. The reproducible build uses the corresponding 3.12.1 header files and maps guanaqo’s `gemmt` calls to the installed reference BLAS `gemmtr` symbols. The numerical solver code runs from the pinned upstream revision. The source patch and build command are under `reference/` in the archive.

The exporter saved the evaluated stage arrays, including the branch-specific discretizations. Its fixture SHA-256 is `3df5ee99261f33abfbd82e592b0a9a66206ccd337112f61bfa0111005cccbf6a`. The branch costs are summed as in the source. All six shared-input Hessian diagonal entries are 4. The eliminated initial-state cost contributes 90.45122306451248 to each reconstructed **total** objective. [Paper and Figure 3](https://arxiv.org/abs/2611.17221v1) · [Pinned benchmark driver](https://github.com/kul-optec/fancyqlone/blob/06f4bcfbab56d0fa9bbd82ad871929a3f0a464b1/benchmarks/fancyqpalm/spring-mass.cpp)

## Numerical verification

The checker takes the two primal–dual solutions back to the original stage blocks. It evaluates the stored dynamics, positions and input bounds, branch-weighted costs and shared-root stationarity. Equality multipliers retain their value when translating from upstream’s dynamics residual orientation to SQP-craft’s opposite row and opposite Lagrangian sign. Upstream inequality multipliers are converted to SQP-craft’s signed box convention. Duplicate first-input bounds contribute their summed dual to the root.

| Measurement | Clarabel through SQP-craft | FanCyQPALM, original stage rows | First exact SQP step |
|---|---:|---:|---:|
| Recomputed objective | 1778.031254933646 | 1778.031254158380 | 1778.031254933746 |
| Dynamics residual, ∞-norm | 1.71 × 10⁻¹³ | 1.78 × 10⁻¹⁵ | 3.78 × 10⁻¹⁴ |
| Box violation, ∞-norm | 1.30 × 10⁻¹⁴ | 3.73 × 10⁻⁹ | 0 |
| Stationarity, ∞-norm | 1.23 × 10⁻¹³ | 1.79 × 10⁻¹³ | 1.12 × 10⁻¹² |
| Complementarity, ∞-norm | 2.25 × 10⁻¹⁰ | 2.30 × 10⁻⁸ | 4.40 × 10⁻¹⁰ |
| Dual-sign violation, ∞-norm | 0 | 0 | 0 |
| Root stationarity, ∞-norm | 1.42 × 10⁻¹⁴ | 6.44 × 10⁻¹⁴ | 1.42 × 10⁻¹⁴ |

The acceptance limits were 10⁻⁸ for the equality, box and stationarity residuals, 10⁻⁶ for complementarity, and 10⁻⁷ for relative objective difference. Clarabel returned `Solved` and passed `QpModel::accepts` after 13 iterations. FanCyQPALM returned `Converged` with 8 outer iterations, 18 inner iterations and 17 factorizations. All branch copies of the first input and the initial state matched exactly. FanCyQPALM’s complementarity after merging root bounds was 3.64 × 10⁻⁸. The largest differences between the two solutions were **8.41 × 10⁻⁸ in state** and **2.27 × 10⁻⁷ in control**. The largest difference between their independent forward rollouts was 8.41 × 10⁻⁸.

The exact SQP test started from a box-feasible zero-input rollout with nonzero states. The initial equality residual was 1.78 × 10⁻¹⁵. The recorded QP matched the expected CSR patterns, entries, gradient and translated bounds exactly. Its only accepted step passed the independent certificate, with zero shifts, restoration attempts and cold retries. The driver reported `iteration_limit` at the deliberately set one-step cap. [Exact-curvature interface](https://github.com/dietmarwo/SQP-craft/blob/a87fde4ef8904198f8ba959d047348f3934a7d25/docs/structured-sqp.md)

## Sparse assembly and the common KKT graph

| Assembly case | Variables | Dynamics rows | Dynamics entries | Python dynamics CSR bytes |
|---|---:|---:|---:|---:|
| Published, four branches and 32 intervals | 3,822 | 3,072 | 92,928 | 1,127,428 |
| Double branches to eight | 7,638 | 6,144 | 185,856 | 2,254,852 |
| Double horizon to 64 | 7,662 | 6,144 | 188,160 | 2,282,500 |

Direct CSR assembly, wire transfer, import and verification keep branch dynamics sparse. All rows have at most 31 entries. The largest stage dynamics input block supplied by upstream is 24×30. Both variants in the table were assembly checks. The nominal published dynamics CSR represents **1.08 MiB**, compared with **89.58 MiB** for a single dense dynamics array. The two imported Rust CSR payloads use 1,603,168 bytes including 64-bit indices. The text wire takes 2,624,065 bytes. Python assembly peaked at about 12.6 MiB of traced allocations, excluding its already loaded stage inputs and process overhead.

The box-reduced symbolic graph has **6,894 vertices** and **99,822 lower-triangle entries** including the diagonal. The test used symmetric symbolic elimination without numerical pivoting.

| Ordering on the common graph | Factor entries incl. diagonal | Fill edges | Largest front incl. pivot |
|---|---:|---:|---:|
| Exact minimum degree | 204,333 | 104,511 | 48 |
| Exact minimum degree with shared input deferred | 204,333 | 104,511 | 48 |
| Fixed reverse-stage order, shared input last | 240,081 | 140,259 | 54 |
| Same stage order, shared input first | 243,522 | 143,700 | 97 |

Clarabel’s AMD-ordered QDLDL diagnostic reports **199,659 off-diagonal factor entries**. The dimension from its finite cone rows is 11,466, including the 4,572 box sides. Presolve was enabled and there were no cold retries. SQP-craft’s dual-first budget estimate reports **276,201 entries** and the Hessian-only Cholesky reports **3,822**. The three computations treat boxes and factor diagonals according to their own conventions.

On an identically configured upstream scalar linear-solver object, the selected buffers totaled **326,823 logical scalar slots** and **384,786 allocated slots**. This inventory includes the four packed stage Jacobian buffers with 69,120 allocated entries for local inequalities and a 432-entry dense root constraint buffer. The root Schur factor itself has 21 logical triangular entries. The 144 allocated root-factor buffer entries include storage for all four contributions. The inventory is a sample of the factor and workspace buffers and uses a scalar build. Peak RSS and SIMD storage remain for the controlled performance study. [SQP-craft symbolic accounting](https://github.com/dietmarwo/SQP-craft/blob/a87fde4ef8904198f8ba959d047348f3934a7d25/crates/core/src/sparse.rs) · [Native root and branch factorization](https://github.com/kul-optec/fancyqlone/blob/06f4bcfbab56d0fa9bbd82ad871929a3f0a464b1/src/fancyqlone/include/fancyqlone/fancyqlone/implementation/fan-factor.tpp)

## Regression and replay results

- All 12 original toy cases passed before and after the sparse bridge change. The post-change maximum stationarity was 1.16 × 10⁻¹³. The bounded and unbounded two-branch toy cases also passed native round trips through upstream `LinearOCPStorage` and actual FanCyQPALM solves. The importer preserved the nonzero initial state and finite or infinite sides.
- Stage and CSR objective, gradient and dynamics agreed at zero, a seeded arbitrary point and a shared-control perturbation. Branch reversal preserved the mapping. The checks caught an injected off-branch entry, a mismatched copy of the shared input, and flipped active-bound and dynamics multipliers.
- The sparse bridge rejected seven malformed input variants. The symbolic counter agreed with dense elimination on the tiny graph and with the hand count for a star.
- Locked Rust workspace tests passed **236** tests, with one documentation census marked ignored by the repository. Rust formatting and Clippy with warnings denied passed. The Python suite passed **220** tests with the pinned S2MPJ checkout present.
- Both patches applied in clean checkouts. Rebuilding FanCyqlone with the supplied script and rerunning the full comparison reproduced the fixture, sparse input, both raw solutions, certificates and fill records byte for byte. An independent offline replay matched the same six files. The archive contains the fixture, both raw solutions, factor permutations, allocation inventory and a hash manifest.

## Decision

The shared first input and stage sparsity survive the SQP-craft exact-QP interface. A root-last ordering can preserve generic fill for this instance when it uses minimum degree within the eligible vertices. FanCyQPALM’s packed inequality blocks show which storage a sparse integration would need to avoid. The next performance question is whether independent branch work pays for mapping, packing and root synchronization over the paired four- and eight-core series on the home machine. This execution used the one-core correctness phase of the plan.

The patch ZIP includes application and reproduction commands in `README.md`, the separate upstream instrumentation and build script, and results under `evidence/`.

Patch archive SHA-256 `86c3951b12fb6aafd1d1e375e4b0d08f5f570c04adc23940c7367a84a0e22968`.
