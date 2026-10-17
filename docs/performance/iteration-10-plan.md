# Cache compatible conic conversion data and scatter maps

Hypothesis: every QP rebuilds identical conic row maps, CSC patterns, cones and
allocation even when only numeric data change. Cache one converted model and
source-index/sign map per adapter. A hit requires exact Hessian/Jacobian CSR
pattern equality and identical finite/equality bound classes. Replace every
P/A/b numeric entry on a hit, including explicit zeros; gradient stays borrowed
from the current model. Keep the SPD gate and independent checks. Clones and
structural changes start cold; failed/nonoptimal calls do not retain conversion.

Compare refreshed CSC/b/mapping data bitwise with fresh conversion after
numeric changes, and test invalidation after equality/free-side changes and
pattern changes. Run all release tests, four full-BFGS benchmarks and all six
explicit block experiments, including the recorded orbit failures.
