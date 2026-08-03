# v111 Formal Stage A：脱敏公开结果

**更新时间 / Updated:** 2026-08-03

**Formal status:** `PASS_V111_FORMAL_STAGE_A_ALL_SEEDS`

**Independent status:** `PASS_INDEPENDENT_RECOMPUTATION_V111_FORMAL_STAGE_A`

## 中文摘要

Formal Stage A 在当前公开三维 PoolFire CFD straight-ray proxy 上完成了正式多轨迹评估。15 个 trajectory-seed summary 全部通过；总计 2970 个 candidate cells、990 个 control cells、270 个 predictions 和 90 个 input bundles。独立程序重新实现指标与判决并完成复算，最大数值差为 0；它仍共享冻结的 physics kernels，因此没有证明端到端物理实现完全独立。

当前可确认的结论是：在冻结的八项精度门下，DRC-Warm candidate 在这批公开 proxy 的正式评估中保持通过，且当前没有简单 frozen control 对 candidate 形成支配。这是从单轨迹 pilot 走向正式多轨迹精度证据的实质增量。

## What this does not prove

这一步没有证明：parent controls 已经完整比较；truth-mutation non-interference 已完成；端到端 wall/RSS 有收益；跨数据集或跨几何泛化成立；方法已经迁移到 OERF 真实 BOST；或者已经形成论文突破。公开状态仍为 `algorithm_breakthrough=false`、`paper_success=false`。

## English summary

Formal Stage A is a formal multi-trajectory evaluation on the current public 3D PoolFire straight-ray proxy. All 15 trajectory-seed summaries pass, covering 2,970 candidate cells, 990 control cells, 270 predictions, and 90 input bundles. An independent program reimplemented the metrics and decision and reproduced the result with maximum numeric difference 0. It still shares the frozen physics kernels, so end-to-end physics independence is not proven.

The supported conclusion is narrow: under the frozen eight accuracy gates, the DRC-Warm candidate passes the formal evaluation on this public proxy, and no simple frozen control currently dominates it. This is a substantive step from the single-trajectory pilot to formal multi-trajectory accuracy evidence.

## What remains open

Parent controls, truth-mutation non-interference, fresh wall/RSS resource measurement, external reacting-flow evaluation, and transfer to real OERF BOST remain open. The public status remains `algorithm_breakthrough=false` and `paper_success=false`.

## Reproducibility boundary

This public document contains aggregate counts and claims only. It intentionally excludes private execution paths, hashes, checkpoints, model parameters, raw data, credentials, and restricted laboratory material.
