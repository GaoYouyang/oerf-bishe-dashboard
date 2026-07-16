# TV/Huber SupPCG scale smoke

Status: `POSTOPEN_SUPPCG_BUDGET_EFFICIENCY_NO_GO`

This post-open scale smoke compares 32 superiorized PCGLS candidates and five
classical references on two already opened replicates. The physical detector
geometry is real PSU geometry; reaction fields, correlated flow-on noise, and
covariance truth are synthetic.

The best candidate is `sup_huber_k3_n1_g0p32_a0p7`. It has a small +0.124%
mean signal relative to graph-PCGLS at the same three stages, but its extra
residual-rebuild forward projection changes the fair call-budget reference to
graph-PCGLS-4. Against that reference it obtains -6.016% mean, -10.299% p10,
87.5% harm over one percent, and -15.551% worst field gain.

The result is a scale smoke, not a confirmatory comparison. It does not
authorize a full opened grid, fresh seeds, neural training, or an algorithm
superiority claim.
