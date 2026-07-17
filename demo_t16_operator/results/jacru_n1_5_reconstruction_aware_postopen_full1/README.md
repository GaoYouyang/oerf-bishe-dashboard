# N1.5-B post-open reconstruction-aware screen

Status: **POSTOPEN HYPOTHESIS ONLY**. This run cannot establish an algorithm gain.

The N1.5-A visible curvature ridge reduced forward mismatch but failed its no-harm gate. This follow-up asks whether any correction improves the reconstructed field under an explicit call budget.

- Corrected candidates: CGLS-12 warm start + one visible low projection + warm-start CGLS-12 = 25 low forward and 24 low adjoint calls.
- Strong reference: low-order CGLS-25 = 25 low forward and 25 low adjoint calls.
- Direct fourth-order CGLS-25 is included as a strong physics comparator.
- Exact mismatch is evaluator-only and excluded from selection.
- The future confirmation hypothesis was chosen on calibration only: `damping_high_order_b0p75` (beta=0.75).
- Its already-opened development mean field gain is 0.047991, H1 gain is 0.108988, and worst-case field gain is 0.016548. These are exploratory numbers, not confirmation evidence.
- OOD/fresh/final and real BOST claims remain closed.
