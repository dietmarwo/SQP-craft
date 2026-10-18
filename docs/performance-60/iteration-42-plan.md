# Iteration 42: Parallel KKT projection dots

The KKT polishing projection c=J^T gd has independent output components. Pass execution resources explicitly to kkt_solve and distribute those dots for n>=256, retaining dense::dot exact scalar reduction order and serial triangular multiplier solves. Test complete c/x/u state equality against one thread at127/256, varied active rank and zero coefficients. Run full release tests and paired NLP/batch timings. No parallel reduction across coordinates is introduced.
