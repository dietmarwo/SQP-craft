# Iteration 50: Parallel bounded dense-row norms

Extract row-normal preparation and parallelize only when stored row work is large and there are enough bounded rows. Free rows stay zero and do not count toward dispatch work; sparse/small inputs retain the original serial iterator. Each row uses the unchanged dense::norm scalar reduction. Test dense bounded/free mixtures against scalar output at the actual dispatch size. Full release suites and paired existing benchmark fixtures; this targets larger many-row QPs rather than claiming gains on tiny sparse NLP rows.
