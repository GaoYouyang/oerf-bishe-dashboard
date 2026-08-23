# v218.1：新初始化器失败，但 Low-64 K11 建立确定性调用余量

## 结论

v217.1 已把 geometry-Jacobi PCGLS K16 固定为 Case 5 上最低可靠的全局 reference。v218.1 在同一批已开封的 `42` 帧、`13` 套虚拟九相机几何和 `546` 个单元上，正式重放一个只读取二维观测与已知几何的 potential-normal warm initializer，并在 K1 到 K14 的未修改 PCGLS 后端上检验绝对精度与 K16 matched-accuracy。

独立第二实现确认，主候选在 K14 仍只有：

- 绝对门：`0/546` 单元、`0/13` 完整几何；
- K16 matched 门：`0/546` 单元、`0/13` 完整几何；
- 实际逻辑调用：`15A+15A^T`。

科学判决为：

`FAIL_POTENTIAL_NORMAL_PCGLS_WARM_INSUFFICIENT_V218_1`

这不是边缘失败。K14 的逐几何 p90 范围为 field `2.016-2.224`、完整梯度 `3.327-3.691`、内部梯度 `7.876-9.037`、observation `0.175-0.211`。因此 current potential-normal 表示关闭，不再调阈值、秩、深度，也不用更大的网络挽救。

## 同场出现的确定性正结果

同一冻结实验还重放了结果前已固定的便宜 controls。这里出现了一个真正改变路线优先级的结果：existing Low-64 observation-only control 在 PCGLS K11 首次完整通过。

| Arm | `A/A^T` | 绝对门单元 | 绝对门完整几何 | Matched 单元 | Matched 完整几何 | 最大 matched ratio |
|---|---:|---:|---:|---:|---:|---:|
| Potential-normal + K14 | `15/15` | `0/546` | `0/13` | `0/546` | `0/13` | `14.8000` |
| Low-64 + K10 | `11/10` | `546/546` | `13/13` | `164/546` | `0/13` | `1.16435` |
| **Low-64 + K11** | **`12/11`** | **`546/546`** | **`13/13`** | **`546/546`** | **`13/13`** | **`1.02190`** |
| Normalized BP + K14 | `15/15` | `546/546` | `13/13` | `0/546` | `0/13` | `1.20003` |
| Geometry-Jacobi PCGLS K16 | `16/16` | `546/546` | `13/13` | `546/546` | `13/13` | `1.00000` |

Low-64 K10 已通过绝对门，却只有 `164/546` matched 单元、`0/13` 完整几何；K11 才是冻结 roster 中第一个同时达到 `546/546` 和 `13/13` matched 的深度。它相对 K16 reference：

- A 从 `16` 降到 `12`，减少 `25%`；
- A^T 从 `16` 降到 `11`，减少 `31.25%`；
- 总逻辑精确调用从 `32` 降到 `23`，减少 `28.125%`。

也就是从 `16A+16A^T` 降到 `12A+11A^T`。

Low-64 K11 的逐几何 p90 范围为 field `0.237-0.261`、完整梯度 `0.483-0.529`、内部梯度 `0.501-0.536`、observation `0.0499-0.0518`，最大 matched ratio 为 `1.02190`。

这是同一已开封 Case 5 虚拟九相机代理上的**确定性 control headroom**。它不是 learned algorithm，也还没有在结果前未开的工况上确认，更没有 fresh wall/RSS 或真实 BOST 证据。

## 独立复算与审裁修正

正式重放完成 `546/546` 个单元并封存。首轮独立 validator 的所有逐 arm 指标、主候选物理场、调用账和离散判决均一致，但它额外使用了未在协议中冻结的 reference 场容差，并用刚重算的浮点 K16 指标替代已封存 K16 指标做 matched 分母，因此按规则保持 inconclusive。

修正只作用于 validator 的 reference 审裁：正式数组、候选、controls、阈值、求解器和数据全部不变，首次 inconclusive 证据也原样保留。修正后的独立实现重新构造预测、物理重放、四项指标、逐几何尾部、调用账和 K16 绑定，全部检查通过：

- 预测坐标最大相对差 `2.17e-13`；
- 预测场最大相对差 `2.79e-13`；
- 主候选正式/独立场最大相对差 `9.34e-10`；
- 逐单元指标最大差 `2.37e-9`；
- 汇总最大差 `3.23e-11`；
- 相机乱序 K14 场最大相对差 `4.40e-10`；
- 调用账、离散判决与封存输入全部一致。

独立状态为：

`PASS_INDEPENDENT_RECOMPUTATION_POTENTIAL_NORMAL_PCGLS_WARM_V218_1_1`

## 路线动作与证据边界

v218.1 同时给出一个明确负结果和一个值得保留的确定性 control 正结果：

1. potential-normal 表示立即关闭；
2. Low-64 K11 固定为下一次确认候选，不再改变表示或深度；
3. 只有它在一个结果前未开的等价公开工况上再次通过 matched-accuracy 和独立复算，才运行 fresh wall/RSS；
4. 当前不训练神经模型、不租 GPU，也不打开真实 BOST 性能主张。

本结果不是外部泛化、wall/RSS 加速、曲线光路验证或真实 BOST。`algorithm_breakthrough=false`。

---

# v218.1: The New Initializer Fails, While Low-64 K11 Establishes Deterministic Call Headroom

## Conclusion

v217.1 fixes geometry-Jacobi PCGLS K16 as the lowest reliable global Case 5 reference. On the same `42` opened frames, `13` virtual nine-camera geometries, and `546` cells, v218.1 physically replays a potential-normal warm initializer that reads only 2D observations and known geometry, followed by unchanged PCGLS depths K1 through K14. It evaluates both absolute accuracy and K16-matched accuracy.

The independent second implementation confirms that the primary still reaches only:

- absolute gates: `0/546` cells and `0/13` complete rigs;
- K16-matched gates: `0/546` cells and `0/13` complete rigs;
- logical exact-call ledger: `15A+15A^T`.

The scientific decision is:

`FAIL_POTENTIAL_NORMAL_PCGLS_WARM_INSUFFICIENT_V218_1`

This is not a borderline miss. At K14, per-rig p90 ranges are `2.016-2.224` for field, `3.327-3.691` for full gradient, `7.876-9.037` for interior gradient, and `0.175-0.211` for observation. The current potential-normal representation is therefore closed. Its threshold, rank, depth, and network size will not be extended.

## Deterministic Positive Result in the Same Replay

The same frozen experiment replays preregistered cheap controls. It reveals one result that materially changes route priority: the existing observation-only Low-64 control first clears the full gate at PCGLS K11.

| Arm | `A/A^T` | Absolute cells | Complete absolute rigs | Matched cells | Complete matched rigs | Maximum matched ratio |
|---|---:|---:|---:|---:|---:|---:|
| Potential-normal + K14 | `15/15` | `0/546` | `0/13` | `0/546` | `0/13` | `14.8000` |
| Low-64 + K10 | `11/10` | `546/546` | `13/13` | `164/546` | `0/13` | `1.16435` |
| **Low-64 + K11** | **`12/11`** | **`546/546`** | **`13/13`** | **`546/546`** | **`13/13`** | **`1.02190`** |
| Normalized BP + K14 | `15/15` | `546/546` | `13/13` | `0/546` | `0/13` | `1.20003` |
| Geometry-Jacobi PCGLS K16 | `16/16` | `546/546` | `13/13` | `546/546` | `13/13` | `1.00000` |

Low-64 K10 clears the absolute gates but reaches only `164/546` matched cells and `0/13` complete matched rigs. K11 is the first depth in the frozen roster to reach both `546/546` matched cells and `13/13` complete rigs. Relative to the K16 reference, it reduces:

- A from `16` to `12`, or `25%`;
- A^T from `16` to `11`, or `31.25%`;
- total logical exact calls from `32` to `23`, or `28.125%`.

That is a reduction from `16A+16A^T` to `12A+11A^T`.

Low-64 K11 per-rig p90 ranges are `0.237-0.261` for field, `0.483-0.529` for full gradient, `0.501-0.536` for interior gradient, and `0.0499-0.0518` for observation. Its maximum matched ratio is `1.02190`.

This is **deterministic control headroom** on the same opened Case 5 virtual-nine proxy. It is not a learned algorithm, has not been confirmed on a previously unopened condition, and carries no fresh wall/RSS or real-BOST evidence.

## Independent Recomputation and Adjudication Correction

The formal replay completes and seals all `546/546` cells. In the first independent validator, all per-arm metrics, primary physical fields, call ledgers, and discrete decisions agree. However, that validator adds a reference-field tolerance not frozen in the protocol and uses freshly recomputed floating-point K16 metrics instead of the sealed K16 metrics as matched denominators. It therefore correctly remains inconclusive.

The correction changes only validator-side reference adjudication. Formal arrays, candidates, controls, thresholds, solver, and data remain unchanged, and the first inconclusive evidence is preserved. The corrected independent implementation rebuilds predictions, physical replays, four metrics, per-rig tails, call ledgers, and K16 binding. Every check passes:

- maximum prediction-coordinate relative difference: `2.17e-13`;
- maximum prediction-field relative difference: `2.79e-13`;
- maximum formal-independent primary-field relative difference: `9.34e-10`;
- maximum cell-metric difference: `2.37e-9`;
- maximum summary difference: `3.23e-11`;
- maximum K14 camera-permutation field difference: `4.40e-10`;
- call ledgers, discrete decisions, and sealed inputs all agree.

The independent status is:

`PASS_INDEPENDENT_RECOMPUTATION_POTENTIAL_NORMAL_PCGLS_WARM_V218_1_1`

## Route Action and Evidence Boundary

v218.1 provides both a decisive negative and a deterministic-control positive:

1. close the potential-normal representation immediately;
2. fix Low-64 K11 as the next confirmation candidate without changing its representation or depth;
3. run fresh wall/RSS only if it again clears matched accuracy and independent recomputation on an equivalent previously unopened public condition;
4. do not train a neural model, rent a GPU, or make a real-BOST performance claim now.

This result does not establish external generalization, wall/RSS speedup, curved-ray validation, or real BOST. `algorithm_breakthrough=false`.
