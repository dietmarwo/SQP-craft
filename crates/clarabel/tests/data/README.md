# EPIGRIDS step-QP regressions

These are original-coordinate SQP subproblems derived from the 250-bus
`Easy/Winter12Generous.m` case in EPIGRIDS v1.0.0, by Jonathan Snodgrass
and collaborators, WISPO-POP/SyntheticElectricNetworkModels.

Source: <https://github.com/WISPO-POP/SyntheticElectricNetworkModels/tree/v1.0.0>

Data license: CC BY 4.0, <https://creativecommons.org/licenses/by/4.0/>.
The derived QPs preserve that attribution and license. Repository source
code remains MIT licensed. Derivation changes: nonlinear AC equations were
linearized at SQP iterates, the objective divided by 10,000, and sparse
partitioned BFGS curvature assembled into the step objective. Original
MATPOWER data were not edited.

`epigrids-reference-step.txt` is the independently rejected QP captured in
the original November 16 campaign. `epigrids-later-step.txt` comes from the
November 17 reference-start run after the accuracy retry was introduced;
its initially guessed active set contains a voltage upper bound whose
polished multiplier has the wrong sign.

The whitespace format is: `n m`, gradient, variable lower/upper bounds,
row lower/upper bounds, then Hessian lower CSR and constraint CSR. Each
matrix has `nnz`, row pointers, column indices and values. Bound infinities
are explicit. Unit tests parse the files without additional dependencies
and accept only under the unchanged original-coordinate QP check.
