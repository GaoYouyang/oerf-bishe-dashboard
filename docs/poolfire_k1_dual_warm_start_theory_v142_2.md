# K1-dual pair-depth warm start: exact guarantees and claim limits

Status: result-independent mathematical note. This document does not use the
unfinished v142.1 full-run metrics and does not authorize a learned predictor.

## Setting

Let the reported linear BOS proxy be

\[
  A : X \to Y, \qquad y \in Y,
\]

with Euclidean inner products. The reconstruction residual is
\(r(x)=y-Ax\). All statements below concern this fixed *reported* operator.
They do not establish correctness for an unknown true camera model, curved
rays, experimental noise, or real BOST.

The zero-start CGLS recurrence stores a field iterate \(x_k\), residual
\(r_k\), and a detector-space dual \(z_k\). The v142 shell builds a proposed
detector correction \(\widehat{\delta z}\), lifts it exactly through
\(A^T\), performs an observable scalar line search, and optionally takes one
literal restarted CGLS step.

## Proposition 1: detector-dual lift identity

In exact arithmetic, every stored zero-start CGLS iterate satisfies

\[
  x_k = A^T z_k.
\]

**Proof.** Initially \(x_0=A^Tz_0=0\). Let the CGLS field direction be
\(p_k\), and define the detector direction \(q_k\) by the same recurrence:
\(q_0=r_0\), and \(q_{k+1}=r_{k+1}+\beta_kq_k\). Since
\(p_0=A^Tr_0=A^Tq_0\), induction gives

\[
 p_{k+1}=A^Tr_{k+1}+\beta_kp_k=A^Tq_{k+1}.
\]

The paired updates
\(x_{k+1}=x_k+\alpha_kp_k\) and
\(z_{k+1}=z_k+\alpha_kq_k\) therefore preserve
\(x_{k+1}=A^Tz_{k+1}\). This establishes a Range\((A^T)\)-consistent
representation without an extra online operator call.

The production recurrence is in
`learning_labs/nine_view_poolfire_full_trajectory_v67.py`; the independent
dense-operator test checks the identity at K1--K4.

## Proposition 2: observable correction cannot increase measurement residual

For any finite proposed dual \(\widehat{\delta z}\), define

\[
  d=A^T\widehat{\delta z}, \qquad p=Ad,
\]

and, when \(\lVert p\rVert_2>0\), choose

\[
  \alpha_* = \frac{\langle r_1,p\rangle}{\lVert p\rVert_2^2}.
\]

Then \(x_0=x_1+\alpha_*d\) minimizes
\(\lVert r_1-\alpha p\rVert_2\) over all scalar \(\alpha\), and

\[
  \lVert r_1\rVert_2^2-\lVert r(x_0)\rVert_2^2
  =\frac{\langle r_1,p\rangle^2}{\lVert p\rVert_2^2}\ge 0,
\]

with \(\langle r(x_0),p\rangle=0\). The proof follows by expanding the
one-dimensional quadratic and setting its derivative to zero.

This is an observation-space guarantee only. It does **not** imply lower field
error, lower full-gradient error, lower interior-gradient error, convergence,
or robustness to a wrong operator. Those quantities remain veto gates in the
experiment.

## Proposition 3: the exact K4 dual difference is an offline capacity witness

Let \(z_1,z_4\) be duals from one exact zero-start CGLS trajectory and set
\(\delta z_*=z_4-z_1\). Proposition 1 yields

\[
 A^T\delta z_*=x_4-x_1.
\]

Moreover, \(p_*=A(x_4-x_1)=r_1-r_4\). CGLS orthogonality gives
\(\langle r_4,p_*\rangle=0\), hence

\[
 \langle r_1,p_*\rangle=\lVert p_*\rVert_2^2
 \quad\Longrightarrow\quad \alpha_*=1.
\]

Thus an exact supplied teacher correction reproduces K4 from K1 after one
lift/projection pair. This is useful for asking whether a representation has
enough capacity. It is not an online shortcut: producing \(z_4-z_1\) already
requires the expensive K4 reference, and v142.1 additionally fits coefficients
offline. A deployable method must predict those coefficients using only K1
state and reported geometry on a held-out trajectory.

## Proposition 4: one literal restart also decreases measurement residual

At the initializer state, let \(n=A^Tr\) and \(p=An\). The implemented
restart uses

\[
  \eta=\frac{\lVert n\rVert_2^2}{\lVert p\rVert_2^2}.
\]

By adjointness,
\(\langle r,p\rangle=\langle A^Tr,n\rangle=\lVert n\rVert_2^2\), so
\(\eta\) is exactly the scalar residual minimizer from Proposition 2. Therefore
the restarted step cannot increase the reported measurement residual in exact
arithmetic. It is a restart, not a claim that the prior CGLS conjugacy has been
preserved.

## Exact online call ledger

The candidate path is counted from zero start:

| State | Exact A | Exact A^T | Reason |
| --- | ---: | ---: | --- |
| CGLS K1 state | 1 | 1 | one ordinary CGLS step |
| initializer K0 | 2 | 2 | K1 plus exact lift and projection |
| warm restart K1 | 3 | 3 | initializer plus one restarted step |
| Zero-CGLS K3 control | 3 | 3 | equal-cost deterministic control |
| Zero-CGLS K4 reference | 4 | 4 | matched-accuracy reference |

Offline basis projection and K4-teacher construction are excluded from an
eventual online ledger only if a separately trained predictor replaces them.
They must remain included in training/setup and end-to-end reporting.

## Camera reordering

For a coherent camera-block permutation \(P\), the reordered operator is
\(A_P=PA\). If \(y_P=Py\) and \(z_P=Pz\), then

\[
  A_P^Tz_P=A^TP^TPz=A^Tz.
\]

Hence fields are invariant while detector duals and residuals co-permute.
The pair-depth construction enumerates ordered target/peer camera pairs and
co-permutes those directions. This establishes invariance to relabeling or
input order. It does not establish robustness to adding/removing cameras or
changing their calibration; those are separate empirical strata.

## What remains unproved

Even if the full v142.1 capacity gate passes, the following remain false until
separate evidence is produced:

- the coefficients are predictable from deployment-visible inputs;
- complete-trajectory generalization succeeds;
- the candidate beats every equal- or lower-cost control;
- exact-call savings become fresh-process wall or RSS savings;
- the straight-ray proxy transfers to curved rays or real BOST;
- the method is globally novel, state of the art, or publication ready.

The executable certificates are in
`learning_labs/poolfire_k1_dual_theory_v142_2.py`; randomized production-path
tests are in `learning_labs/test_poolfire_k1_dual_theory_v142_2.py`.
