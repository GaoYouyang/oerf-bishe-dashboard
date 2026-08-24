# v221：精确行空间 lift 没有保住 Low-64 暖启动信息，当前构造关闭

## 结论

v220.2 已经说明，固定 Low-64 起点在 Case 5 有局部效果，但跨到 Case 2 后不稳定。v221 检验一个物理上不同、可直接证伪的解释：问题是否来自 Low-64 场中对当前观测不可见的近零空间成分。

候选先把 direct Low-64 场经过一次精确 `A` 和一次精确 `A^T` 投回 `range(A^T)`，再用仅由当前观测确定的单个缩放系数，接未修改的 geometry-Jacobi PCGLS K10。完整在线账固定为 `12A+11A^T`，与 direct Low-64 K11 对照完全同价。

正式运行和完全独立第二实现重放 Case 5 与 Case 2 共 `1261` 个单元。独立 `32/32` 项检查全部通过，科学判决为：

`FAIL_LOW64_EXACT_ROWSPACE_LIFT_V221`

## 结果

| 工况 | 绝对严格安全单元 | K16 matched 单元 | 完整几何 | 最大 matched ratio |
|---|---:|---:|---:|---:|
| Case 5 | `202/546` | `0/546` | `0/13` | `1.94921` |
| Case 2 | `670/715` | `0/715` | `0/13` | `2.14816` |

Case 5 的主要绝对失败来自内部梯度：`344/546` 个单元越过逐单元门，完整梯度另有 3 个失败。Case 2 有 `45/715` 个内部梯度失败。更关键的是，两种工况所有单元都至少有一项指标超过 K16 的 `1.05` matched 门，因此不是完整几何聚合方式造成的表面失败。

同价 direct Low-64 K11 在 Case 5 仍是 `546/546、13/13`，说明精确 lift 反而抹掉了有用信息；在 Case 2，direct Low-64 K11 虽通过 `715/715` 绝对门，但 matched 只有 `518/715、0/13`。Zero-start K16 在两种工况均为 `13/13`，所以 reference 本身充分。

## 可观测缩放没有卡在边界

缩放系数结果前固定在 `[0,2]`。Case 5 范围为 `0.02872-0.03859`，Case 2 为 `0.01739-0.03698`，上下界命中均为 0。这排除了“只是 clip 边界选坏了”的解释。小幅缩放后的行空间状态更像略优于 Zero K11 的起点，但没有保住 direct Low-64 的跨迭代价值。

## 独立复算

独立程序不导入正式 runner 或初始化 helper，自行重建公开场预处理、几何、Low-64 表示、精确 forward-adjoint-forward lift、可观测缩放、缓存起点 PCGLS、四项指标、调用账与相机乱序哨兵。

正式/独立场、初始化器、逐单元指标和缩放系数最大差分别为 `3.03e-9`、`8.03e-15`、`1.49e-10` 和 `1.60e-16`；相机乱序场差为 `7.56e-14`，调用账差为 0。共享冻结 physics kernels 仍然存在，所以不宣称端到端物理完全独立，但预注册数值门全部通过。

## 路线动作

1. 关闭当前“Low-64 -> 精确 `A^T A` 行空间 lift -> 单缩放 -> PCGLS K10”构造；
2. 不调整 alpha 边界、PCGLS 深度或 Low-64 秩，不增加方向；
3. 不训练 CNN、FNO、UNO、DeepONet，不租 GPU；
4. 不打开 Case 4/6，不运行 fresh wall/RSS；
5. 后续只接受物理上真正不同、结果前冻结的机制，或映射完整的配对真实 BOST 位移数据。

这次成功完成了机制证伪和独立复算，但没有算法突破、论文成功、资源加速、外部泛化、曲线光路或真实 BOST 证据。`algorithm_breakthrough=false`。

---

# v221: The Exact Row-Space Lift Does Not Preserve the Useful Low-64 Warm-Start Signal

## Conclusion

v220.2 showed that the fixed Low-64 start has local value in Case 5 but is unstable after transfer to Case 2. v221 tests a physically distinct, falsifiable explanation: whether the failure comes from near-nullspace components of the Low-64 field that are invisible to the current observation.

The candidate sends the direct Low-64 field through one exact `A` and one exact `A^T` to place it in `range(A^T)`, applies one observation-only scale, and runs unchanged geometry-Jacobi PCGLS K10. Its complete online ledger is fixed at `12A+11A^T`, exactly matching the direct Low-64 K11 control.

The formal and fully independent implementations replay all `1261` Case 5 and Case 2 cells. All `32/32` independent checks pass. The scientific decision is:

`FAIL_LOW64_EXACT_ROWSPACE_LIFT_V221`

## Results

| Condition | Absolute strict-safe cells | K16-matched cells | Complete rigs | Maximum matched ratio |
|---|---:|---:|---:|---:|
| Case 5 | `202/546` | `0/546` | `0/13` | `1.94921` |
| Case 2 | `670/715` | `0/715` | `0/13` | `2.14816` |

Case 5 absolute failures are dominated by interior gradient: `344/546` cells cross the cellwise limit, with three additional full-gradient failures. Case 2 has `45/715` interior-gradient failures. More importantly, every cell in both conditions exceeds the `1.05` K16-matched limit in at least one metric, so this is not an artifact of complete-rig aggregation.

The equal-cost direct Low-64 K11 control remains at `546/546 and 13/13` in Case 5, showing that the exact lift removes useful information. In Case 2, direct Low-64 K11 clears all `715/715` absolute cells but reaches only `518/715 and 0/13` under matched accuracy. Zero-start K16 reaches `13/13` in both conditions, so the reference itself is adequate.

## The observable scale is not stuck at a boundary

The scale is preregistered on `[0,2]`. Its range is `0.02872-0.03859` in Case 5 and `0.01739-0.03698` in Case 2, with zero lower- or upper-bound hits. This rules out a simple clipping-boundary explanation. The scaled row-space state behaves like a modestly improved zero-K11 start but does not preserve the cross-iteration value of direct Low-64.

## Independent recomputation

The independent program imports neither the formal runner nor its initializer helper. It rebuilds public-field preprocessing, geometry, the Low-64 representation, exact forward-adjoint-forward lift, observable scale, cached-start PCGLS, all four metrics, call ledgers, and camera-permutation sentinels.

Maximum formal-independent differences in field, initializer, cell metric, and scale are `3.03e-9`, `8.03e-15`, `1.49e-10`, and `1.60e-16`. The camera-permutation field difference is `7.56e-14`, and the call-ledger difference is zero. Frozen physics kernels are still shared, so end-to-end physics independence is not claimed, but every preregistered numerical check passes.

## Route action

1. close the current Low-64 to exact `A^T A` row-space lift to one-scale to PCGLS K10 construction;
2. do not retune the scale bounds, PCGLS depth, Low-64 rank, or direction count;
3. do not train a CNN, FNO, UNO, or DeepONet and do not rent a GPU;
4. do not open Case 4/6 or run fresh wall/RSS;
5. continue only with a physically distinct preregistered mechanism or fully mapped paired real-BOST displacement data.

The mechanism was successfully falsified and independently recomputed, but this is not an algorithm breakthrough, paper success, resource speedup, external generalization, curved-ray validation, or real BOST. `algorithm_breakthrough=false`.
