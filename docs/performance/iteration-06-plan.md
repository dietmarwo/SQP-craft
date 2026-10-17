# Recycle derivative callback output buffers

Hypothesis: accepted evaluations can hand their retired gradient/Jacobian
allocations to the next callback. Keep two alternating evaluation-buffer sets
per solve. Refill every reused entry with NaN before invoking the callback so
an incomplete write still fails. Recycle restoration evaluations too; preserve
function-before-derivative ordering, callback counts and complete-point limits.

Add a stale-output regression that would fail if recycled values were accepted.
Run full release tests and the fixed-callback benchmarks. Python receives the
same fresh input-list contract; no Python objects are retained or reused.
