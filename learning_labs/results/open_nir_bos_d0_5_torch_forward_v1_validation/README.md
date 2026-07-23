# Independent D0.5 validation

- Status: `VALID_PASS_D0_5_IMPLEMENTATION_ONLY`
- Checks: 88 / 88
- Independent heterogeneous value probe: 1.217e-16 relative L2
- PyTorch autograd gradcheck: `true`
- Public replay rerun: 768 rays

The validator recomputes checksums, thresholds, verdicts, comparison statistics,
the public value replay, a different-shape value probe, and an autograd gradcheck.
It validates implementation evidence only and authorizes no reconstruction,
generalisation, algorithm-comparison, paper, or breakthrough claim.
