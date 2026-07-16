# Observable covariance-stopping rule audit

Status: `OBSERVABLE_POOLED_STOPPING_RULE_NO_GO`

The audit evaluates 348 bounded, interpretable stopping rules on already opened
graph-whitened PCGLS trajectories. No one-threshold rule passes the selection
gates. Five rollback/continue rules pass the opened selection half.

The selected rule uses stage-4 beta, relative volume update, and marginal
residual reduction. It obtains +3.76% mean field gain with a -1.77% worst field
on the selection half, but fails the opened diagnostic half: +3.34% mean, p10
-1.75%, 12.5% harm over 1%, and -17.53% worst field.

Fresh preregistration is not authorized. The result argues against increasing
controller capacity on these six pooled scalar features. The next development
candidate must either preserve per-camera spatial structure or impose a
front-protecting regularizer before any learned controller is reconsidered.
