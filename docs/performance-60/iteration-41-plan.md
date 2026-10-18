# Iteration 41: Parallel dense transformed normals

Only dense/large row normals need substantial J^T a work. Share an owned execution handle with Sources and compute independent transformed components concurrently when n>=256 and row density>=half; variable and singleton normals stay scalar. Keep the original index/value iteration and sum in each dot. Test variable, singleton and dense normals with both signs against scalar at128/256. Mark the old reflector wrapper test-only to keep production warnings clean. Full release suites and paired eight-CPU-budget benchmarks.
