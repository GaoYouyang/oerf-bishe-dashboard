# v215：观测代理 Warm Replay 因 K16 参考不足而不可判定

## 结论

v215 把 v214 的 observation/geometry-only low-64 代理场真正接入未修改 CGLS，并在已开封 Case 5 的 `13` 套虚拟九相机几何、`42` 帧、共 `546` 个单元上重放完整物理链。实验同时运行 Zero-CGLS、归一化 BP+CGLS 和精确几何 Jacobi-PCGLS 控制，并记录实际 `A/A^T` 账。

但预注册的 Zero-CGLS K16 参考没有通过自身充分性门，所以协议在比较 warm start 之前停止。封存科学判决为：

`INCONCLUSIVE_INVALID_OBSERVATION_PROXY_WARM_REPLAY_V215`

这不是“代理 warm start 已失败”，也不是“代理 warm start 已成功”；当前数据不能给出合法比较。

## 为什么参考无效

Zero-CGLS K16 在 `546` 个单元中只有 `466` 个同时通过四项绝对门，完整通过的几何只有 `1/13`。全部 `80` 个失败都只来自内部梯度；field、完整梯度与 observation 没有单元越线。

| 参考充分性证据 | 结果 |
|---|---:|
| 完整通过的几何 | `1/13` |
| 通过全部四项绝对门的单元 | `466/546` |
| 内部梯度越线单元 | `80` |
| 其他三项越线单元 | `0` |
| 逐几何内部梯度 p90 范围 | `0.71296-0.78644` |
| 冻结内部梯度单元 / p90 门 | `0.75000` |

唯一完整通过的是 rig 7。其余几何要么存在个别单元高于 `0.75`，要么逐几何 p90 也超过 `0.75`。因此 K16 不能作为本轮 matched-accuracy 的合格 reference。

## 独立复算

独立第二实现重新读取原始 42 帧密度与网格，重建 low-64 物理基、13 套虚拟九相机算子、二维观测、代理初值、CGLS/BP/PCGLS 递推、四项指标、调用账和相机换序审计。它没有用正式侧代理场构造候选，并以 `M^T M` 特征分解替代正式侧求解路径。

- 正式与独立物理场最大相对差：`3.42e-9`
- 四项指标最大绝对差：`1.43e-10`
- 逐几何汇总最大差：`5.75e-11`
- 二维观测最大绝对差：`8.88e-16`
- 相机换序重放指标最大差：`7.66e-15`

所有数组、调用账、汇总和科学判决一致。独立状态保持 inconclusive，是因为它同样复现了 K16 参考不足，而不是因为两套实现不一致。

## 这项结果改变了什么

v214 证明二维观测与已知几何足以复现谱对齐判别；v215 进一步证明这条代理已经可以机械地进入完整 warm-start 物理重放链。与此同时，它暴露了更靠前的阻塞：当前 Zero-K16 reference 在内部梯度上不够充分，因此不能用它裁决 warm start。

## 它不意味着什么

本轮没有选出 proxy 深度，也没有裁决便宜控制是否支配代理。它没有建立 matched accuracy、exact-call 减少、wall/RSS、外部泛化、真实 BOST 或可部署学习算法。

`algorithm_breakthrough=false`

## 下一门

另行结果前冻结 reference qualification，而不是在 v215 内事后改 K、改门或挑一个更好看的控制。只有新的 reference 在同一 13 套几何与 42 帧上先独立通过 field、完整梯度、内部梯度和 observation 充分性门，才允许重新裁决 observation-only warm start。

---

# v215: Observation-Proxy Warm Replay Is Inconclusive Because the K16 Reference Is Inadequate

## Conclusion

v215 feeds the v214 observation/geometry-only low-64 proxy field into unchanged CGLS and replays the full physical chain on `13` virtual nine-camera geometries and `42` opened Case 5 frames, for `546` cells. Zero-CGLS, normalized BP+CGLS, and exact-geometry Jacobi-PCGLS controls run alongside it with actual `A/A^T` ledgers.

The preregistered Zero-CGLS K16 reference does not pass its own adequacy gate, so the protocol stops before comparing warm starts. The sealed scientific decision is:

`INCONCLUSIVE_INVALID_OBSERVATION_PROXY_WARM_REPLAY_V215`

This is neither a successful nor a failed proxy warm start. The current experiment cannot make that comparison validly.

## Why the Reference Is Invalid

Zero-CGLS K16 passes all four absolute gates in only `466/546` cells and completes only `1/13` geometries. All `80` failing cells fail interior gradient only; no field, full-gradient, or observation cell crosses its threshold.

| Reference-adequacy evidence | Result |
|---|---:|
| Complete geometries passed | `1/13` |
| Cells passing all four absolute gates | `466/546` |
| Interior-gradient violating cells | `80` |
| Violations in the other three metrics | `0` |
| Per-geometry interior-gradient p90 range | `0.71296-0.78644` |
| Frozen interior-gradient cell / p90 gate | `0.75000` |

Only rig 7 passes completely. Every other geometry has at least one cell above `0.75`, and some also have a per-geometry p90 above `0.75`. K16 is therefore not an adequate matched-accuracy reference for this run.

## Independent Recalculation

The independent second implementation rereads all 42 raw density frames and grids, rebuilds the low-64 physical basis, 13 virtual-nine operators, 2D observations, proxy initializers, CGLS/BP/PCGLS recurrences, four metrics, call ledgers, and camera-permutation audit. It does not consume formal proxy fields to construct candidates and uses an eigendecomposition of `M^T M` instead of the formal solve path.

- maximum formal-versus-independent physical-field relative difference: `3.42e-9`
- maximum four-metric absolute difference: `1.43e-10`
- maximum per-geometry summary difference: `5.75e-11`
- maximum 2D-observation absolute difference: `8.88e-16`
- maximum camera-permutation replay-metric difference: `7.66e-15`

Every array, call ledger, summary, and scientific decision agrees. The independent status remains inconclusive because it independently reproduces K16 reference inadequacy, not because the two implementations disagree.

## What This Changes

v214 shows that 2D observations and known geometry reproduce the spectral-alignment decision. v215 shows that the proxy can be mechanically connected to a complete warm-start physical replay. It also exposes an earlier blocker: the current Zero-K16 reference is inadequate in interior gradient and therefore cannot adjudicate the warm start.

## What It Does Not Mean

No proxy depth is selected, and no cheap-control dominance claim is adjudicated. This establishes no matched accuracy, exact-call reduction, wall/RSS result, external generalization, real BOST, or deployable learned algorithm.

`algorithm_breakthrough=false`

## Next Gate

Freeze a separate reference-qualification experiment before results instead of changing K, thresholds, or controls post hoc inside v215. A new reference must independently pass field, full-gradient, interior-gradient, and observation adequacy across the same 13 geometries and 42 frames before the observation-only warm start can be adjudicated again.
