# v206：九相机流式势函数正规路径通过 fresh wall/RSS 资源门

## 这次真正回答了什么

v205 已经证明，一个固定的势函数正规方程缓存可以数值复现稠密 full-DCT K1，但正式 setup 仍会瞬时构造全相机稠密响应，而且当时没有 fresh-process wall/RSS 证据。v206 只回答下一层问题：**把 setup 改成按相机流式构造后，九相机路径能否在保持同一输出的同时，稳定降低端到端时间和整流水线内存？**

正式执行与完全独立审裁共同得到：

`PASS_STREAMING_COMPACT_FRESH_RESOURCE_V206`

`PASS_INDEPENDENT_ADJUDICATION_STREAMING_COMPACT_FRESH_RESOURCE_V206`

## 做了什么，以及为什么这样做

v206 保留 v205 的固定 1009 维非直流 DCT 表示和未修改 K1，只把几何 setup 改成按相机流式累积势函数正规方程所需状态。这样不再瞬时形成全相机稠密响应，同时不改变候选物理输出。

setup 独立复算覆盖五/九相机共 26 个几何 setup、2626 个单元。相对正式实现的最大坐标差为 `1.48e-12`，因子重建差为 `2.07e-13`，正则项差为 `1.25e-13`；Helmert gauge 与 stationarity 也通过冻结门。这一步先证明流式构造不是用数值漂移换内存。

## Fresh-process 资源结果

资源审计仅使用历史上已经暴露的 p14 九相机开发轨迹。13 套标定、每套 101 帧，共完成 `39` 个 reference worker、`429` 个 timed worker、`143` 个随机相邻完整区组和 `468` 条原始 worker receipt。所有 worker 均为独立 fresh process，setup 放在 worker 内，输出相对稠密 K1 的最大差为 `6.02e-13`。

相对稠密 full-DCT K1：

- outer wall p50 / p90-higher：`0.8603 / 0.8729`
- setup wall p50 / p90-higher：`0.7801 / 0.7973`
- worker-self RSS p50 / p90-higher：`0.6886 / 0.7160`
- sampled worker-tree RSS p50 / p90-higher：`0.6907 / 0.7192`
- sampled whole-pipeline RSS p50 / p90-higher：`0.7100 / 0.7370`

相对 dense K2 reference：

- outer wall p50 / p90-higher：`0.7395 / 0.7503`
- setup wall p50 / p90-higher：`0.7804 / 0.7991`
- worker-self RSS p50 / p90-higher：`0.6883 / 0.7124`
- sampled worker-tree RSS p50 / p90-higher：`0.6934 / 0.7149`
- sampled whole-pipeline RSS p50 / p90-higher：`0.7122 / 0.7339`

全部全局门和 13 套标定的逐组 p50 门均通过。这里的比值以对照为 1，越低越好。

## 精确调用账没有被隐藏

每帧逻辑账仍为：

- 流式紧凑 K1：`2A+2A^T`
- 稠密 full-DCT K1：`2A+1A^T`
- dense K2 reference：`3A+2A^T`

因此，流式路径相对稠密 K1 **多一次精确伴随**；本轮 wall/RSS 正结果来自流式 setup 与更小工作状态，而不是相对稠密 K1 的 exact-call 减少。相对 K2，它少一次 forward、伴随次数相同。

## 准确率与适用范围

v206 不重新包装准确率。九相机继承 `1313/1313` 个严格单元与 `13/13` 个完整组；五相机仍只有 `1268/1313` 与 `3/13`。所以本轮没有建立五/九相机都通过的可变基数算法。

成功的是：在 p14 九相机已暴露开发条件上，流式紧凑路径以约 `1e-12` 的输出等价性，稳定降低了 fresh outer wall、setup wall、worker RSS、process-tree RSS 和 sampled whole-pipeline RSS。

没有成功的是：这不是 fresh validation、外部泛化、曲线光路或真实 BOST；也不是全局资源加速结论。下一门必须在结果前冻结一个此前未打开的独立公开反应流工况，同时复核九相机 matched-accuracy 与资源收益。五相机仍是单独未解决的准确率门。

`algorithm_breakthrough=false`

`global_resource_speedup_claim=false`

---

# v206: the all-nine streamed potential-normal path clears the fresh wall/RSS gate

## What this run actually answers

v205 showed that a fixed potential-normal cache can numerically reproduce dense full-DCT K1, but formal setup still transiently formed the all-camera dense response and no fresh-process wall/RSS evidence existed. v206 asks only the next question: **after making setup camera-streamed, can the all-nine path preserve the same output while consistently reducing end-to-end time and whole-pipeline memory?**

The formal run and fully independent adjudication jointly seal:

`PASS_STREAMING_COMPACT_FRESH_RESOURCE_V206`

`PASS_INDEPENDENT_ADJUDICATION_STREAMING_COMPACT_FRESH_RESOURCE_V206`

## What was done and why

v206 retains v205's fixed 1,009-dimensional non-DC DCT representation and unchanged K1. It changes only geometry setup, accumulating the potential-normal state camera by camera instead of transiently forming the all-camera dense response. Candidate physical output is unchanged.

Independent setup recomputation covers 26 five/all-nine geometry setups and 2,626 cells. Maximum coordinate difference to formal is `1.48e-12`, factor-reconstruction difference is `2.07e-13`, and regularization difference is `1.25e-13`; the frozen Helmert gauge and stationarity checks also pass. This establishes that memory is not reduced by accepting numerical drift.

## Fresh-process resource result

The resource audit uses only the historically exposed p14 all-nine development trajectory. Across 13 calibrations and 101 frames each, it completes `39` reference workers, `429` timed workers, `143` randomized adjacent complete blocks, and `468` raw worker receipts. Every worker is a fresh process with setup inside the worker. Maximum output difference to dense K1 is `6.02e-13`.

Versus dense full-DCT K1:

- outer-wall p50 / p90-higher: `0.8603 / 0.8729`
- setup-wall p50 / p90-higher: `0.7801 / 0.7973`
- worker-self RSS p50 / p90-higher: `0.6886 / 0.7160`
- sampled worker-tree RSS p50 / p90-higher: `0.6907 / 0.7192`
- sampled whole-pipeline RSS p50 / p90-higher: `0.7100 / 0.7370`

Versus the dense K2 reference:

- outer-wall p50 / p90-higher: `0.7395 / 0.7503`
- setup-wall p50 / p90-higher: `0.7804 / 0.7991`
- worker-self RSS p50 / p90-higher: `0.6883 / 0.7124`
- sampled worker-tree RSS p50 / p90-higher: `0.6934 / 0.7149`
- sampled whole-pipeline RSS p50 / p90-higher: `0.7122 / 0.7339`

Every global threshold and every per-calibration p50 check passes. Ratios use the control as 1, so lower is better.

## The exact-call ledger remains visible

Per frame:

- streamed compact K1: `2A+2AT`
- dense full-DCT K1: `2A+1AT`
- dense K2 reference: `3A+2AT`

The streamed path therefore uses **one extra exact adjoint** versus dense K1. Its wall/RSS result comes from streamed setup and a smaller working state, not fewer exact calls than dense K1. Versus K2, it saves one forward and uses the same number of adjoints.

## Accuracy and scope

v206 does not repackage accuracy. All-nine inherits `1,313/1,313` strict cells and `13/13` complete groups; five cameras remain at only `1,268/1,313` and `3/13`. Stable variable-cardinality success has not been established.

What succeeded: on the exposed p14 all-nine development condition, the streamed compact path preserves output near `1e-12` while consistently reducing fresh outer wall, setup wall, worker RSS, process-tree RSS, and sampled whole-pipeline RSS.

What did not: this is not fresh validation, external generalization, curved-ray validation, or real BOST, and it is not a global resource-speedup claim. The next gate must preregister a previously unopened independent public reacting-flow condition and jointly recheck all-nine matched accuracy and resource gains. Five-camera accuracy remains a separate unresolved gate.

`algorithm_breakthrough=false`

`global_resource_speedup_claim=false`
