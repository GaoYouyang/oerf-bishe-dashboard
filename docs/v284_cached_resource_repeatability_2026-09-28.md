# v284 Cached Resource Repeatability: Post-Open Result

Date: 2026-09-28  
Scope: descriptive process-resource comparison on the same already-opened v284 roster. This is not a new accuracy evaluation or held-out test.

## Protocol

Five paired fresh-process blocks compare the frozen learned v284 path with the frozen certified AMG-PCGLS path. Each process replays the same 1,010 rows across two geometry paths, including process startup, cached-artifact loading, prediction/certification, and output sealing. Two warmups are excluded. Ten completion receipts and raw operating-system timing records were independently checked; every replay returned successfully and reproduced its archived predictions.

The comparison excludes learned fitting (about 122,255 seconds in the archived fit record) and non-free AMG hierarchy/certificate construction. Accuracy was not rescored. This is cached inference-path timing only, not full-pipeline or amortized lifecycle timing.

## Results

| Measure | Learned v284 | Certified AMG-PCGLS |
|---|---:|---:|
| Median process wall time | 3,226.10 s | 3,662.70 s |
| Median peak RSS | 2.356 GiB | 1.757 GiB |

The median paired wall-time ratio (learned / AMG) is 0.8808; paired-block ratios range from 0.8733 to 0.8822. A paired log-ratio interval of 0.8751-0.8840 describes timing repeatability on this one workload and machine only. It is not uncertainty across trajectories or geometries.

The certified AMG comparator retains the pointwise exact-call advantage: it passed the four accuracy gates on all 505 opened queries and used fewer A and A^T calls on each paired query. Thus, on this cached execution path, fewer operator calls did not imply lower process wall time. Conversely, the learned path had higher peak RSS, and its large fit cost plus AMG setup were excluded.

## Interpretation Boundary

This result adds a measured resource tradeoff, not a net speedup or a change to the frozen v284 verdict `FAIL_SOLVER_IN_LOOP_LOTO_STRICT_COST` (453/505 strict pointwise cost wins; 1/5 complete trajectories). It establishes neither total-pipeline advantage, external generalization, real BOST, nor an algorithmic breakthrough. No model was retrained, no threshold was changed, and no sealed condition was opened.

The complete paired-process readback remains local to the reproducibility record; this public summary intentionally omits private paths, hashes, and raw artifacts.
