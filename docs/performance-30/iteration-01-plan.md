# Iteration 01: Use the sorted CSR row endpoint for range checks

Written before the implementation. Previous retained source: sqp-perf30-00.

## Plan

After checking strict row ordering, its last index is the maximum. Check only that index on valid input; scan for the original diagnostic only on the invalid path. This removes one full pass over every valid sparse pattern without caching or trusting mutable data.

Target measurements: validate kernel; QP and KKT conversion costs. Compare the same fixed work and both original NLP callbacks/acceptance rules. Preserve public validation failures, finite arithmetic checks and existing numerical tolerances.

## Implement and test

Apply the change only to the identified hot path. Add a numerical or cache-invalidation fixture when the risk warrants it; use the existing malformed-input, finite-arithmetic, random QP/Clarabel and NLP acceptance tests as regression gates. Run the full release Rust workspace tests before measurement. Measure previous/candidate in serial, pinned processes after builds, with warmups and three rotating samples. Retain raw results and explain work reduction; revise or revert any unexplained correctness regression. Final stress/holdout and paired baseline comparisons remain mandatory.
