# v256：Galerkin 粗到细冷启动因 residual 独立一致性越界而保持不确定

## 为什么做

v255 关闭后，Case 19 首帧唯一绝对阻塞仍是一个 interior-gradient 尾部，而且把同一 fine-grid 迭代从 K14 加深到 K16 并没有改善。v256 因此不再微调旧信任混合，而是检验一个物理上不同的经典多层机制：先在 `16x8x8` 粗网格运行无预条件 CGLS K4，再用 cell-centered 分片线性张量插值提升到 `32x16x16`，最后运行未修改的 fine-grid geometry-Jacobi PCGLS K10。

结果前固定了粗细网格、常值端点延拓、一层 fine 零边界、active-cell 均值 gauge、精确转置 restriction、唯一 K4→K10 深度、两个便宜 control、K16 reference、四指标绝对门、K16-matched 门、调用账和独立数值容差。候选不读取 CFD 真值、rig 标签或失败标签，训练参数为 0。

## 唯一独立失败门

正式运行完成 13 个已开封 Case 19 九相机首帧，预测屏障和物理 replay 都封存。完全独立的第二实现通过冻结检查中的 `19/20`，唯一失败项是 `residuals_agree`：最大 residual 相对差为 `5.91005e-7`，高于冻结的 `2.00000e-7`，即 `2.95502×`。

其余连续量都在各自冻结界内：coarse field 与 initializer 相对差约 `1.50e-15 / 1.42e-15`，final field 相对差 `7.68e-9` 对 `2e-8` 界；逐单元指标和汇总差为 `5.38e-10 / 8.19e-10` 对 `2e-8` 界；观测差和相机乱序差为 0。transfer、物理 replay 与残差方程的绝对闭环也在 `1e-15` 量级。

这些通过项不能覆盖一个明确越界的预注册检查。正式状态因此是 `INCONCLUSIVE_INDEPENDENT_RECOMPUTATION_CASE19_GALERKIN_PYRAMID_FRAME_ZERO_V256`，科学判决为 `INCONCLUSIVE_INVALID_CASE19_GALERKIN_PYRAMID_FRAME_ZERO_V256`。

## 13/13 只能作诊断

两套实现的离散判决一致。无效合同下的诊断计数是：primary 绝对门与 K16-matched 都为 `13/13`；zero geometry-J PCGLS K14 control 为绝对 `7/13`、matched `0/13`；normalized-BP geometry-J PCGLS K13 control 为绝对 `6/13`、matched `0/13`；K16 reference 自身绝对为 `12/13`。

这些数字不能写成算法 headroom 或通过。`19/20` 也不是“几乎通过”：独立连续数值合同是科学门的一部分，不能由离散门一致替代。

## 成本与路线动作

primary 的名义单帧账为 `15A+14A^T`，K16 reference 为 `16A+16A^T`，合计调用算术差为 `9.375%`。由于结果无效且只运行了首帧，这不是有效 exact-call 减少，也不授权完整 429 单元序列、wall time、RSS、外部门、训练或 GPU。

当前精确 K4→K10 Galerkin 金字塔关闭；不放宽 residual 容差、不重跑、不搜索深度、transfer、边界或 gauge。这个结论不否定所有 Galerkin 或多重网格方法，也不关闭整条 C 路线。下一步只能来自新的配对真实 BOST 信息，或一个结果前唯一冻结、部署可见、可独立证伪且物理上真正不同的机制。

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

# v256: Galerkin coarse-to-fine cold start remains inconclusive after residual agreement fails

## Motivation

After v255 closes, the sole absolute Case 19 frame-zero blocker remains an interior-gradient tail, and deepening the same fine-grid solve from K14 to K16 does not improve it. v256 therefore avoids retuning the old trust blend and tests a physically distinct classical multilevel mechanism: unpreconditioned CGLS K4 on a `16x8x8` grid, a cell-centered piecewise-linear tensor lift to `32x16x16`, then unchanged fine-grid geometry-Jacobi PCGLS K10.

The coarse and fine grids, constant endpoint extension, one-layer fine zero boundary, active-cell mean gauge, exact-transpose restriction, unique K4-to-K10 schedule, two cheap controls, K16 reference, four absolute gates, K16-matched gates, call ledger, and independent numerical tolerances were fixed before results. The candidate reads no CFD truth, rig label, or failure label and has zero trainable parameters.

## The sole independent failure

Formal execution completes the 13 opened Case 19 nine-camera frame-zero cells, with the prediction barrier and physical replay sealed. A fully independent implementation passes `19/20` frozen checks. The sole failure is `residuals_agree`: maximum residual relative disagreement is `5.91005e-7` against a frozen `2.00000e-7` limit, or `2.95502x`.

All other continuous quantities remain within their frozen limits. Coarse-field and initializer relative differences are about `1.50e-15 / 1.42e-15`; final-field disagreement is `7.68e-9` against `2e-8`. Cell-metric and summary differences are `5.38e-10 / 8.19e-10` against `2e-8`. Observation and camera-permutation differences are zero, while transfer, physical replay, and residual-equation closure remain near `1e-15` in absolute terms.

Passing these checks cannot override one explicit preregistered failure. The formal status is therefore `INCONCLUSIVE_INDEPENDENT_RECOMPUTATION_CASE19_GALERKIN_PYRAMID_FRAME_ZERO_V256`, with scientific decision `INCONCLUSIVE_INVALID_CASE19_GALERKIN_PYRAMID_FRAME_ZERO_V256`.

## The 13/13 count is diagnostic only

The two implementations agree on discrete decisions. Under the invalid contract, diagnostic counts are `13/13` for both primary absolute and K16-matched gates; `7/13` absolute and `0/13` matched for the zero-start geometry-Jacobi PCGLS K14 control; `6/13` absolute and `0/13` matched for the normalized-BP geometry-Jacobi PCGLS K13 control; and `12/13` absolute for the K16 reference itself.

None of these counts is admissible as algorithmic headroom or a pass. `19/20` does not mean “almost passed”: continuous independent agreement is part of the science gate and cannot be replaced by matching discrete decisions.

## Cost and route action

The nominal primary frame ledger is `15A+14A^T`, versus `16A+16A^T` for K16, an arithmetic combined-call difference of `9.375%`. Because the result is invalid and covers frame zero only, this is not effective exact-call reduction and does not authorize the full 429-cell sequence, wall time, RSS, an external gate, training, or GPU use.

The exact K4-to-K10 Galerkin pyramid closes without relaxing residual tolerance, rerunning, or searching depth, transfer, boundary, or gauge choices. This does not reject all Galerkin or multigrid methods and does not close the C route. Any next step requires new paired real-BOST information or one uniquely preregistered, deployment-visible, independently falsifiable, and physically distinct mechanism.

`algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, and `real_bost=false`.
