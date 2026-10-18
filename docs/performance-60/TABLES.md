# Recomputed final performance tables

Every reported sample passed the unchanged original-model assertions and driver certificate checks. Wall means setup-inclusive workflow time; native solver/QP work and process CPU time are recorded separately. Ratios are baseline/time, larger is faster. These are medians on one shared host, not monotone per-iteration guarantees.

## final-matrix

| Build | Family | Size | Backend | Inner | Outer | Jobs | Workflow ms | Median native job ms | Process CPU ms | Repeats |
|---|---|---|---|---|---|---|---|---|---|---|
| final | orbit | 20 | active-set | 1 | 1 | 1 | 30.222 | 30.192 | 32.752 | 3 |
| final | orbit | 20 | active-set | 2 | 1 | 1 | 31.472 | 31.437 | 33.904 | 3 |
| final | orbit | 20 | active-set | 4 | 1 | 1 | 31.696 | 31.667 | 33.707 | 3 |
| final | orbit | 20 | active-set | 8 | 1 | 1 | 32.231 | 32.196 | 35.230 | 3 |
| final | orbit | 40 | active-set | 1 | 1 | 1 | 267.536 | 267.433 | 269.670 | 3 |
| final | orbit | 40 | active-set | 2 | 1 | 1 | 254.198 | 254.107 | 257.025 | 3 |
| final | orbit | 40 | active-set | 4 | 1 | 1 | 289.293 | 289.202 | 291.737 | 3 |
| final | orbit | 40 | active-set | 8 | 1 | 1 | 280.595 | 280.503 | 282.318 | 3 |
| final | separable | 128 | active-set | 1 | 1 | 1 | 20.931 | 20.881 | 22.928 | 3 |
| final | separable | 128 | active-set | 2 | 1 | 1 | 18.134 | 18.015 | 19.960 | 3 |
| final | separable | 128 | active-set | 4 | 1 | 1 | 17.803 | 17.758 | 20.103 | 3 |
| final | separable | 128 | active-set | 8 | 1 | 1 | 17.757 | 17.715 | 19.833 | 3 |
| final | restoration | 64 | active-set | 1 | 1 | 1 | 0.790 | 0.776 | 2.520 | 3 |
| final | restoration | 64 | active-set | 2 | 1 | 1 | 0.647 | 0.635 | 2.051 | 3 |
| final | restoration | 64 | active-set | 4 | 1 | 1 | 0.660 | 0.649 | 2.027 | 3 |
| final | restoration | 64 | active-set | 8 | 1 | 1 | 0.638 | 0.628 | 1.975 | 3 |
| final | separable | 256 | active-set | 1 | 1 | 1 | 109.787 | 109.659 | 111.083 | 3 |
| final | separable | 256 | active-set | 2 | 1 | 1 | 125.384 | 125.247 | 127.507 | 3 |
| final | separable | 256 | active-set | 4 | 1 | 1 | 117.500 | 117.284 | 119.260 | 3 |
| final | separable | 256 | active-set | 8 | 1 | 1 | 112.132 | 111.942 | 113.941 | 3 |
| final | propagation | 32 | active-set | 1 | 1 | 1 | 44.496 | 37.026 | 46.382 | 3 |
| final | propagation | 32 | active-set | 2 | 1 | 1 | 27.627 | 20.534 | 47.437 | 3 |
| final | propagation | 32 | active-set | 4 | 1 | 1 | 18.084 | 11.061 | 47.963 | 3 |
| final | propagation | 32 | active-set | 8 | 1 | 1 | 17.715 | 10.288 | 57.075 | 3 |
| final | orbit | 20 | active-set | 1 | 1 | 16 | 486.987 | 30.314 | 489.301 | 3 |
| final | orbit | 20 | active-set | 1 | 2 | 16 | 300.623 | 32.230 | 603.706 | 3 |
| final | orbit | 20 | active-set | 1 | 4 | 16 | 155.048 | 32.191 | 585.048 | 3 |
| final | orbit | 20 | active-set | 1 | 8 | 16 | 108.009 | 47.549 | 696.488 | 3 |
| final | orbit | 20 | active-set | 2 | 4 | 16 | 153.714 | 32.723 | 557.420 | 3 |
| final | orbit | 20 | active-set | 4 | 2 | 16 | 241.046 | 29.554 | 477.272 | 3 |
| final | orbit | 20 | active-set | 8 | 1 | 16 | 464.262 | 29.063 | 466.086 | 3 |
| final | separable | 128 | active-set | 1 | 1 | 16 | 277.909 | 16.991 | 279.680 | 3 |
| final | separable | 128 | active-set | 1 | 2 | 16 | 155.108 | 18.007 | 309.189 | 3 |
| final | separable | 128 | active-set | 1 | 4 | 16 | 96.059 | 24.186 | 357.127 | 3 |
| final | separable | 128 | active-set | 1 | 8 | 16 | 68.662 | 25.306 | 324.427 | 3 |
| final | separable | 128 | active-set | 2 | 4 | 16 | 84.085 | 17.197 | 318.616 | 3 |
| final | separable | 128 | active-set | 4 | 2 | 16 | 139.665 | 16.728 | 275.578 | 3 |
| final | separable | 128 | active-set | 8 | 1 | 16 | 269.222 | 16.343 | 271.074 | 3 |
| final | orbit | 20 | clarabel | 1 | 1 | 1 | 248.711 | 248.676 | 250.728 | 3 |
| final | orbit | 20 | clarabel | 2 | 1 | 1 | 227.686 | 227.648 | 230.000 | 3 |
| final | orbit | 20 | clarabel | 4 | 1 | 1 | 246.632 | 246.596 | 249.142 | 3 |
| final | orbit | 20 | clarabel | 8 | 1 | 1 | 248.068 | 248.032 | 250.669 | 3 |
| final | orbit | 40 | clarabel | 1 | 1 | 1 | 1868.049 | 1867.959 | 1868.335 | 3 |
| final | orbit | 40 | clarabel | 2 | 1 | 1 | 1787.215 | 1787.148 | 1789.363 | 3 |
| final | orbit | 40 | clarabel | 4 | 1 | 1 | 1746.698 | 1746.597 | 1748.664 | 3 |
| final | orbit | 40 | clarabel | 8 | 1 | 1 | 1813.770 | 1813.626 | 1815.961 | 3 |
| final | separable | 128 | clarabel | 1 | 1 | 1 | 113.519 | 113.480 | 115.638 | 3 |
| final | separable | 128 | clarabel | 2 | 1 | 1 | 118.762 | 118.720 | 121.471 | 3 |
| final | separable | 128 | clarabel | 4 | 1 | 1 | 119.279 | 119.238 | 121.912 | 3 |
| final | separable | 128 | clarabel | 8 | 1 | 1 | 111.257 | 111.221 | 113.362 | 3 |
| final | restoration | 64 | clarabel | 1 | 1 | 1 | 5.131 | 5.112 | 7.213 | 3 |
| final | restoration | 64 | clarabel | 2 | 1 | 1 | 5.411 | 5.396 | 6.964 | 3 |
| final | restoration | 64 | clarabel | 4 | 1 | 1 | 5.454 | 5.439 | 6.889 | 3 |
| final | restoration | 64 | clarabel | 8 | 1 | 1 | 5.062 | 5.044 | 6.687 | 3 |
| final | separable | 256 | clarabel | 1 | 1 | 1 | 683.284 | 683.104 | 685.500 | 3 |
| final | separable | 256 | clarabel | 2 | 1 | 1 | 669.483 | 669.294 | 671.769 | 3 |
| final | separable | 256 | clarabel | 4 | 1 | 1 | 667.282 | 667.112 | 669.351 | 3 |
| final | separable | 256 | clarabel | 8 | 1 | 1 | 663.236 | 663.043 | 665.725 | 3 |
| final | propagation | 32 | clarabel | 1 | 1 | 1 | 46.907 | 40.033 | 49.718 | 3 |
| final | propagation | 32 | clarabel | 2 | 1 | 1 | 30.687 | 23.263 | 49.814 | 3 |
| final | propagation | 32 | clarabel | 4 | 1 | 1 | 21.852 | 15.015 | 54.146 | 3 |
| final | propagation | 32 | clarabel | 8 | 1 | 1 | 20.571 | 13.671 | 58.938 | 3 |
| final | orbit | 20 | clarabel | 1 | 1 | 16 | 3695.506 | 231.549 | 3696.951 | 3 |
| final | orbit | 20 | clarabel | 1 | 2 | 16 | 2009.883 | 241.571 | 3924.805 | 3 |
| final | orbit | 20 | clarabel | 1 | 4 | 16 | 1072.496 | 257.818 | 4095.045 | 3 |
| final | orbit | 20 | clarabel | 1 | 8 | 16 | 594.999 | 260.703 | 4139.347 | 3 |
| final | orbit | 20 | clarabel | 2 | 4 | 16 | 1030.802 | 240.569 | 4001.786 | 3 |
| final | orbit | 20 | clarabel | 4 | 2 | 16 | 2163.542 | 246.604 | 4092.700 | 3 |
| final | orbit | 20 | clarabel | 8 | 1 | 16 | 3796.369 | 235.693 | 3795.615 | 3 |
| final | separable | 128 | clarabel | 1 | 1 | 16 | 1856.467 | 116.497 | 1858.962 | 3 |
| final | separable | 128 | clarabel | 1 | 2 | 16 | 1023.044 | 118.582 | 2017.016 | 3 |
| final | separable | 128 | clarabel | 1 | 4 | 16 | 542.516 | 120.127 | 2021.323 | 3 |
| final | separable | 128 | clarabel | 1 | 8 | 16 | 349.630 | 152.398 | 2370.925 | 3 |
| final | separable | 128 | clarabel | 2 | 4 | 16 | 538.622 | 118.010 | 2033.134 | 3 |
| final | separable | 128 | clarabel | 4 | 2 | 16 | 1105.690 | 128.932 | 2122.869 | 3 |
| final | separable | 128 | clarabel | 8 | 1 | 16 | 1792.141 | 110.752 | 1793.895 | 3 |
| final | mixed | 20 | active-set | 1 | 1 | 16 | 975.224 | 31.369 | 977.833 | 3 |
| final | mixed | 20 | active-set | 1 | 8 | 16 | 322.020 | 47.660 | 1124.418 | 3 |

## common-baseline

| Build | Family | Size | Backend | Inner | Outer | Jobs | Workflow ms | Median native job ms | Process CPU ms | Repeats |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | orbit | 20 | active-set | 1 | 1 | 1 | 30.244 | 29.989 | 31.743 | 3 |
| final | orbit | 20 | active-set | 1 | 1 | 1 | 30.115 | 30.025 | 32.117 | 3 |
| baseline | orbit | 40 | active-set | 1 | 1 | 1 | 256.551 | 256.151 | 257.138 | 3 |
| final | orbit | 40 | active-set | 1 | 1 | 1 | 267.959 | 267.853 | 270.145 | 3 |
| baseline | separable | 128 | active-set | 1 | 1 | 1 | 17.292 | 16.957 | 19.325 | 3 |
| final | separable | 128 | active-set | 1 | 1 | 1 | 18.180 | 18.110 | 21.035 | 3 |
| baseline | restoration | 64 | active-set | 1 | 1 | 1 | 0.997 | 0.774 | 1.751 | 3 |
| final | restoration | 64 | active-set | 1 | 1 | 1 | 0.810 | 0.791 | 2.699 | 3 |
| baseline | orbit | 20 | active-set | 1 | 1 | 16 | 461.482 | 28.971 | 461.829 | 3 |
| final | orbit | 20 | active-set | 1 | 1 | 16 | 494.244 | 30.936 | 497.008 | 3 |
| baseline | orbit | 20 | active-set | 1 | 8 | 16 | 101.576 | 42.148 | 577.574 | 3 |
| final | orbit | 20 | active-set | 1 | 8 | 16 | 101.093 | 43.772 | 592.310 | 3 |
| baseline | separable | 128 | active-set | 1 | 1 | 16 | 285.714 | 17.571 | 287.392 | 3 |
| final | separable | 128 | active-set | 1 | 1 | 16 | 268.928 | 17.096 | 271.212 | 3 |
| baseline | separable | 128 | active-set | 1 | 8 | 16 | 87.045 | 25.918 | 417.405 | 3 |
| final | separable | 128 | active-set | 1 | 8 | 16 | 62.536 | 25.867 | 359.857 | 3 |
| baseline | orbit | 20 | clarabel | 1 | 1 | 1 | 246.214 | 245.849 | 246.305 | 3 |
| final | orbit | 20 | clarabel | 1 | 1 | 1 | 252.066 | 252.028 | 253.909 | 3 |
| baseline | orbit | 40 | clarabel | 1 | 1 | 1 | 1765.719 | 1765.369 | 1767.064 | 3 |
| final | orbit | 40 | clarabel | 1 | 1 | 1 | 1774.675 | 1774.596 | 1777.057 | 3 |
| baseline | separable | 128 | clarabel | 1 | 1 | 1 | 115.672 | 115.394 | 115.287 | 3 |
| final | separable | 128 | clarabel | 1 | 1 | 1 | 110.669 | 110.634 | 112.670 | 3 |
| baseline | restoration | 64 | clarabel | 1 | 1 | 1 | 5.619 | 5.398 | 6.808 | 3 |
| final | restoration | 64 | clarabel | 1 | 1 | 1 | 5.312 | 5.289 | 7.790 | 3 |
| baseline | orbit | 20 | clarabel | 1 | 1 | 16 | 3847.951 | 236.895 | 3848.780 | 3 |
| final | orbit | 20 | clarabel | 1 | 1 | 16 | 3795.483 | 235.611 | 3795.157 | 3 |
| baseline | orbit | 20 | clarabel | 1 | 8 | 16 | 644.288 | 312.863 | 4780.221 | 3 |
| final | orbit | 20 | clarabel | 1 | 8 | 16 | 679.129 | 312.810 | 4841.391 | 3 |
| baseline | separable | 128 | clarabel | 1 | 1 | 16 | 1842.374 | 111.912 | 1840.171 | 3 |
| final | separable | 128 | clarabel | 1 | 1 | 16 | 1753.079 | 110.240 | 1756.922 | 3 |
| baseline | separable | 128 | clarabel | 1 | 8 | 16 | 331.236 | 152.182 | 2326.326 | 3 |
| final | separable | 128 | clarabel | 1 | 8 | 16 | 316.520 | 137.574 | 2079.714 | 3 |

## final-heldout

| Build | Family | Size | Backend | Inner | Outer | Jobs | Workflow ms | Median native job ms | Process CPU ms | Repeats |
|---|---|---|---|---|---|---|---|---|---|---|
| final | orbit | 30 | active-set | 1 | 1 | 1 | 106.521 | 106.422 | 108.510 | 3 |
| final | orbit | 30 | active-set | 8 | 1 | 1 | 111.798 | 111.758 | 114.461 | 3 |
| final | orbit | 48 | active-set | 1 | 1 | 1 | 469.642 | 469.526 | 471.904 | 3 |
| final | orbit | 48 | active-set | 8 | 1 | 1 | 476.404 | 476.287 | 478.318 | 3 |
| final | separable | 96 | active-set | 1 | 1 | 1 | 8.368 | 8.350 | 10.677 | 3 |
| final | separable | 96 | active-set | 8 | 1 | 1 | 7.569 | 7.553 | 9.389 | 3 |
| final | separable | 192 | active-set | 1 | 1 | 1 | 50.493 | 50.394 | 51.985 | 3 |
| final | separable | 192 | active-set | 8 | 1 | 1 | 47.447 | 47.355 | 49.203 | 3 |
| final | restoration | 48 | active-set | 1 | 1 | 1 | 0.432 | 0.407 | 2.413 | 3 |
| final | restoration | 48 | active-set | 8 | 1 | 1 | 0.400 | 0.389 | 1.735 | 3 |
| final | restoration | 96 | active-set | 1 | 1 | 1 | 1.958 | 1.945 | 3.384 | 3 |
| final | restoration | 96 | active-set | 8 | 1 | 1 | 1.564 | 1.550 | 2.939 | 3 |
| final | propagation | 48 | active-set | 1 | 1 | 1 | 67.609 | 57.223 | 69.067 | 3 |
| final | propagation | 48 | active-set | 8 | 1 | 1 | 22.089 | 12.077 | 82.845 | 3 |
| final | propagation | 96 | active-set | 1 | 1 | 1 | 131.945 | 111.942 | 133.607 | 3 |
| final | propagation | 96 | active-set | 8 | 1 | 1 | 44.192 | 22.269 | 149.451 | 3 |
| final | orbit | 30 | clarabel | 1 | 1 | 1 | 785.776 | 785.738 | 787.648 | 3 |
| final | orbit | 30 | clarabel | 8 | 1 | 1 | 804.428 | 804.389 | 806.491 | 3 |
| final | orbit | 48 | clarabel | 1 | 1 | 1 | 3032.894 | 3032.735 | 3034.504 | 3 |
| final | orbit | 48 | clarabel | 8 | 1 | 1 | 2977.018 | 2976.857 | 2978.847 | 3 |
| final | separable | 96 | clarabel | 1 | 1 | 1 | 57.999 | 57.866 | 59.928 | 3 |
| final | separable | 96 | clarabel | 8 | 1 | 1 | 56.285 | 56.262 | 58.407 | 3 |
| final | separable | 192 | clarabel | 1 | 1 | 1 | 294.454 | 294.309 | 297.014 | 3 |
| final | separable | 192 | clarabel | 8 | 1 | 1 | 293.450 | 293.308 | 295.796 | 3 |
| final | restoration | 48 | clarabel | 1 | 1 | 1 | 3.289 | 3.272 | 5.719 | 3 |
| final | restoration | 48 | clarabel | 8 | 1 | 1 | 3.092 | 3.076 | 5.257 | 3 |
| final | restoration | 96 | clarabel | 1 | 1 | 1 | 12.471 | 12.451 | 14.164 | 3 |
| final | restoration | 96 | clarabel | 8 | 1 | 1 | 13.929 | 13.906 | 16.014 | 3 |
| final | propagation | 48 | clarabel | 1 | 1 | 1 | 72.454 | 61.215 | 74.828 | 3 |
| final | propagation | 48 | clarabel | 8 | 1 | 1 | 28.419 | 17.634 | 85.150 | 3 |
| final | propagation | 96 | clarabel | 1 | 1 | 1 | 159.497 | 139.435 | 162.134 | 3 |
| final | propagation | 96 | clarabel | 8 | 1 | 1 | 70.622 | 49.873 | 173.220 | 3 |

## final-qp

| Build | Family | Size | Backend | Inner | Outer | Jobs | Workflow ms | Median native job ms | Process CPU ms | Repeats |
|---|---|---|---|---|---|---|---|---|---|---|
| final | qp | 128 | active-set | 1 | 1 | 1 | 1.242 | 1.242 | 3.229 | 7 |
| final | qp | 128 | active-set | 2 | 1 | 1 | 1.303 | 1.303 | 3.666 | 7 |
| final | qp | 128 | active-set | 4 | 1 | 1 | 1.399 | 1.399 | 3.859 | 7 |
| final | qp | 128 | active-set | 8 | 1 | 1 | 1.291 | 1.291 | 4.794 | 7 |
| final | qp | 256 | active-set | 1 | 1 | 1 | 7.168 | 7.168 | 9.541 | 7 |
| final | qp | 256 | active-set | 2 | 1 | 1 | 7.112 | 7.112 | 9.627 | 7 |
| final | qp | 256 | active-set | 4 | 1 | 1 | 7.027 | 7.027 | 10.611 | 7 |
| final | qp | 256 | active-set | 8 | 1 | 1 | 7.353 | 7.353 | 11.810 | 7 |
| final | qp | 512 | active-set | 1 | 1 | 1 | 46.453 | 46.453 | 51.663 | 7 |
| final | qp | 512 | active-set | 2 | 1 | 1 | 38.168 | 38.168 | 55.820 | 7 |
| final | qp | 512 | active-set | 4 | 1 | 1 | 28.498 | 28.498 | 54.783 | 7 |
| final | qp | 512 | active-set | 8 | 1 | 1 | 29.327 | 29.327 | 66.617 | 7 |
| final | qp | 1024 | active-set | 1 | 1 | 1 | 349.111 | 349.111 | 363.640 | 7 |
| final | qp | 1024 | active-set | 2 | 1 | 1 | 250.777 | 250.777 | 360.509 | 7 |
| final | qp | 1024 | active-set | 4 | 1 | 1 | 220.607 | 220.607 | 382.550 | 7 |
| final | qp | 1024 | active-set | 8 | 1 | 1 | 201.004 | 201.004 | 389.985 | 7 |

## final-qp-heldout

| Build | Family | Size | Backend | Inner | Outer | Jobs | Workflow ms | Median native job ms | Process CPU ms | Repeats |
|---|---|---|---|---|---|---|---|---|---|---|
| final | qp | 384 | active-set | 1 | 1 | 1 | 20.144 | 20.144 | 23.646 | 3 |
| final | qp | 384 | active-set | 2 | 1 | 1 | 21.502 | 21.502 | 25.835 | 3 |
| final | qp | 384 | active-set | 4 | 1 | 1 | 20.403 | 20.403 | 24.813 | 3 |
| final | qp | 384 | active-set | 8 | 1 | 1 | 19.191 | 19.191 | 25.609 | 3 |
| final | qp | 640 | active-set | 1 | 1 | 1 | 85.180 | 85.180 | 91.734 | 3 |
| final | qp | 640 | active-set | 2 | 1 | 1 | 69.832 | 69.832 | 100.368 | 3 |
| final | qp | 640 | active-set | 4 | 1 | 1 | 56.985 | 56.985 | 101.346 | 3 |
| final | qp | 640 | active-set | 8 | 1 | 1 | 55.355 | 55.355 | 117.063 | 3 |
| final | qp | 896 | active-set | 1 | 1 | 1 | 230.225 | 230.225 | 242.483 | 3 |
| final | qp | 896 | active-set | 2 | 1 | 1 | 182.274 | 182.274 | 254.492 | 3 |
| final | qp | 896 | active-set | 4 | 1 | 1 | 153.712 | 153.712 | 267.711 | 3 |
| final | qp | 896 | active-set | 8 | 1 | 1 | 138.990 | 138.990 | 272.618 | 3 |

## final-propagation-confirmation

| Build | Family | Size | Backend | Inner | Outer | Jobs | Workflow ms | Median native job ms | Process CPU ms | Repeats |
|---|---|---|---|---|---|---|---|---|---|---|
| final | propagation | 32 | active-set | 1 | 1 | 1 | 44.443 | 37.180 | 45.874 | 7 |
| final | propagation | 32 | active-set | 8 | 1 | 1 | 16.187 | 9.117 | 54.088 | 7 |
| final | propagation | 32 | clarabel | 1 | 1 | 1 | 46.361 | 39.403 | 48.080 | 7 |
| final | propagation | 32 | clarabel | 8 | 1 | 1 | 19.787 | 12.196 | 57.007 | 7 |

## final-mixed32

| Build | Family | Size | Backend | Inner | Outer | Jobs | Workflow ms | Median native job ms | Process CPU ms | Repeats |
|---|---|---|---|---|---|---|---|---|---|---|
| final | mixed | 20 | active-set | 1 | 1 | 32 | 1829.368 | 28.975 | 1831.220 | 3 |
| final | mixed | 20 | active-set | 1 | 8 | 32 | 431.812 | 47.157 | 2368.544 | 3 |

## baseline-confirmation

| Build | Family | Size | Backend | Inner | Outer | Jobs | Workflow ms | Median native job ms | Process CPU ms | Repeats |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline | separable | 128 | active-set | 1 | 8 | 16 | 64.331 | 21.290 | 278.522 | 7 |
| final | separable | 128 | active-set | 1 | 8 | 16 | 55.102 | 20.677 | 300.768 | 7 |
| baseline | orbit | 40 | active-set | 1 | 1 | 1 | 255.299 | 254.975 | 256.803 | 7 |
| final | orbit | 40 | active-set | 1 | 1 | 1 | 261.069 | 260.985 | 263.269 | 7 |
| baseline | orbit | 20 | clarabel | 1 | 8 | 16 | 620.386 | 270.806 | 4303.821 | 7 |
| final | orbit | 20 | clarabel | 1 | 8 | 16 | 603.585 | 267.393 | 4287.293 | 7 |

## Same configuration: 0.15 / 0.16 comparison

| Family | Size | Backend | Inner | Outer | Jobs | 0.15 ms | 0.16 ms | 0.15/0.16 |
|---|---|---|---|---|---|---|---|---|
| orbit | 20 | active-set | 1 | 1 | 1 | 30.244 | 30.115 | 1.004 |
| orbit | 40 | active-set | 1 | 1 | 1 | 256.551 | 267.959 | 0.957 |
| separable | 128 | active-set | 1 | 1 | 1 | 17.292 | 18.180 | 0.951 |
| restoration | 64 | active-set | 1 | 1 | 1 | 0.997 | 0.810 | 1.231 |
| orbit | 20 | active-set | 1 | 1 | 16 | 461.482 | 494.244 | 0.934 |
| orbit | 20 | active-set | 1 | 8 | 16 | 101.576 | 101.093 | 1.005 |
| separable | 128 | active-set | 1 | 1 | 16 | 285.714 | 268.928 | 1.062 |
| separable | 128 | active-set | 1 | 8 | 16 | 87.045 | 62.536 | 1.392 |
| orbit | 20 | clarabel | 1 | 1 | 1 | 246.214 | 252.066 | 0.977 |
| orbit | 40 | clarabel | 1 | 1 | 1 | 1765.719 | 1774.675 | 0.995 |
| separable | 128 | clarabel | 1 | 1 | 1 | 115.672 | 110.669 | 1.045 |
| restoration | 64 | clarabel | 1 | 1 | 1 | 5.619 | 5.312 | 1.058 |
| orbit | 20 | clarabel | 1 | 1 | 16 | 3847.951 | 3795.483 | 1.014 |
| orbit | 20 | clarabel | 1 | 8 | 16 | 644.288 | 679.129 | 0.949 |
| separable | 128 | clarabel | 1 | 1 | 16 | 1842.374 | 1753.079 | 1.051 |
| separable | 128 | clarabel | 1 | 8 | 16 | 331.236 | 316.520 | 1.046 |
