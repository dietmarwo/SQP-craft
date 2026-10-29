# Structured application tutorials

Each chapter is a Python script that writes its model as SQP-craft callbacks, checks the derivatives, solves with the native solver (`method="sqp"`) and checks the result against an independent reference. Run one with `python tutorials/<chapter>/run.py --output /new/dir`. Start with the first two chapters, then pick an application.

| ID | Chapter |
|---|---|
| SQP1 | [Constrained calibration and checked derivatives](constrained-calibration/README.md) |
| SQP2 | [Multistart, units and QP backends](multistart-scaling/README.md) |
| SQP3 | [Hanging chain: separate mesh and solver error](hanging-chain/README.md) |
| SQP4 | [Optimal control by collocation and independent replay](collocation/README.md) |
| SQP5 | [Earth-Mars rendezvous with pykep-core](low-thrust-pykep/README.md) |
| SQP6 | [Reaction kinetics and identifiability](reaction-kinetics/README.md) |
| X2 | [Power dispatch: DC LP versus AC physics](power-dispatch/README.md) |

The [shared setup](https://github.com/dietmarwo/optimization-jobs/blob/main/tutorials/structured/README.md) in optimization-jobs runs the whole twenty-chapter series (LP, SQP, fcmaes and cross-solver chapters) directly or as trusted command jobs. The [job API tutorial](job-api/README.md) shows the `sqp-job-worker` profiles and a registered custom model.
