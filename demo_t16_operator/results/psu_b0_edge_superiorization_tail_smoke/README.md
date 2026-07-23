# TV/Huber SupPCG tail smoke

Status: `POSTOPEN_SUPPCG_BUDGET_EFFICIENCY_NO_GO`

This preregistered post-open extension tests 48 deeper/larger TV and Huber
superiorization candidates plus seven classical references on two already
opened replicates. It was the only authorized extension after the initial
scale smoke.

The best call-budget candidate is `sup_huber_k6_n1_g2p56_a0p7`. It is already
-2.278% relative to graph-PCGLS at the same six stages. Against the
call-budget floor graph-PCGLS-8 it obtains:

- mean field gain: -8.518%;
- p10: -25.463%;
- harm over one percent: 68.75%;
- worst field gain: -26.411%.

The frozen stopping rule therefore closes the SupPCG performance branch.
Fresh seeds and neural proximal training remain sealed. The next classical
candidate must use one forward and one adjoint per iteration, such as a
primal-dual TV/Huber solver, before any bounded learned proximal map is
considered.

The four-panel audit is available as
`psu_b0_edge_superiorization_no_go.png` and `.pdf`.
