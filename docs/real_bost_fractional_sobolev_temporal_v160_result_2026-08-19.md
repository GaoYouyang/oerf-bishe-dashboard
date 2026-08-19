# v160：放松高频惩罚没有救回五相机梯度，反而把四个时间层全部拖出门

更新：2026-08-19

## 先说结论

v159.1 的固定 H1 正则只差一个五相机梯度尾部。为了检验它是否“把高频压得太狠”，v160 在看新结果前只冻结一个改变：把整数阶 H1 的平方频率惩罚换成半阶 Sobolev 的频率模长惩罚。DCT1024 表示、固定倍数 `0.03`、四个时间点、5/7/9 相机、物理算子和全部精度门都不变；没有扫描别的阶数或倍数。

正式运行与独立第二实现得到相同判决：`FAIL_FRACTIONAL_SOBOLEV_TEMPORAL_V160`。

- 七、九相机的八个分层全部通过；
- 五相机的四个时间层全部失败；
- 半阶先验在四个五相机层的 gradient p90 分别是 `0.777364 / 0.770968 / 0.809636 / 0.772459`，全部高于 `0.750000` 门；
- 相比原 H1，这四个 gradient p90 分别变差 `0.068832 / 0.075093 / 0.050997 / 0.060425`；
- `t=0.25` 和 `t=1.0` 的 field p90 也略高于 `0.50` 门。

因此，v159.1 的五相机缺口不能用“少平滑一点”修复。更可信的解释是：五相机下某些方向本来就欠定，放松高频惩罚会把不可辨成分放大。

## 四个时间点的结果

下表列出半阶主策略的 p90，顺序为 field / gradient / observation。

| 归一化时间 | 5 相机 | 7 相机 | 9 相机 |
| ---: | :--- | :--- | :--- |
| 0.00 | 0.497 / **0.777** / 0.133，失败 | 0.434 / 0.625 / 0.141，通过 | 0.311 / 0.532 / 0.150，通过 |
| 0.25 | **0.513** / **0.771** / 0.117，失败 | 0.443 / 0.615 / 0.135，通过 | 0.322 / 0.535 / 0.147，通过 |
| 0.75 | 0.483 / **0.810** / 0.119，失败 | 0.420 / 0.633 / 0.132，通过 | 0.317 / 0.560 / 0.144，通过 |
| 1.00 | **0.502** / **0.772** / 0.120，失败 | 0.425 / 0.618 / 0.139，通过 | 0.320 / 0.533 / 0.148，通过 |

冻结 p90 门分别是 field `0.50`、gradient `0.75`、observation `0.20`。所有 worst 门、九相机对 CGLS K16 的 no-harm 门以及真实调用账也被逐项检查。

## 为什么这个负结果可信

正式程序覆盖 `39` 个 operator setups、`1,404` 个 cells 和 `4,212` 条三臂结果。除了半阶主策略，它还在同一次物理重放中重新计算原 H1 `0.03` 和 CGLS K16。

原 H1 对 v159.1 的逐 cell 最大复现差是 `1.64e-11`，汇总最大差是 `7.12e-12`。这说明结果变化来自惩罚的物理形式，而不是换了数据、相机、算子或指标。

独立程序改用解析余弦基、另一套相机射线和稀疏算子实现、独立计算的半阶频率权重以及另一种特征分解，重算全部结果。`19/19` 项独立检查通过；正式与独立逐 cell 指标最大差 `1.64e-11`，汇总最大差 `1.36e-11`，算子数值最大差 `4.19e-11`。

## 接下来怎么走

半阶先验和围绕已开封结果继续扫 Sobolev 阶数都停止。现有数据下仍可检验一个物理上不同的方向：只根据 active cameras 的横向梯度灵敏度，构造相机置换不变的各向异性 H1 度量，对观测较弱的空间方向给予更强约束。它必须单独结果前冻结，并沿用同一四时间 × 三档相机门和独立复算。

这仍是代码生成的受控虚拟代理，不是配对实验 BOST。真实迁移仍需二维双分量位移及相机、帧、标定、噪声和认可基线。

`algorithm_breakthrough=false`、`paper_success=false`、`resource_speedup=false`、`real_bost=false`、`gpu_rental_authorized=false`。

---

# v160: weaker high-frequency attenuation does not repair the five-camera gradient tail

Updated: 2026-08-19

v160 preregisters one physically distinct test of the lone v159.1 miss. It replaces the homogeneous H1 squared-frequency penalty with a homogeneous half-order Sobolev frequency-magnitude penalty, while keeping DCT1024, multiplier `0.03`, four normalized times, 5/7/9 cameras, the forward model, and every accuracy threshold unchanged. No other order or multiplier is evaluated.

The formal run and independent implementation agree on `FAIL_FRACTIONAL_SOBOLEV_TEMPORAL_V160`. All eight seven- and nine-camera strata pass, but all four five-camera strata fail. Their gradient p90 values are `0.777364 / 0.770968 / 0.809636 / 0.772459`, each above the frozen `0.750000` gate and each worse than the corresponding H1 control.

The formal run covers 39 operator setups, 1,404 cells, and 4,212 arm rows. The frozen H1 control reproduces v159.1 to `1.64e-11` per-cell and `7.12e-12` at summary level. A second implementation rebuilds analytic cosine fields, rays, sparse operators, half-order weights, eigensystems, metrics, and all decisions. All `19/19` checks pass; maximum formal-versus-independent per-cell, summary, and operator differences are `1.64e-11`, `1.36e-11`, and `4.19e-11`.

The result rejects the oversmoothing explanation on this controlled proxy. Relaxing high-frequency attenuation amplifies the sparse-view error, which instead supports directional underdetermination or noise amplification. Half-order and post-hoc Sobolev-order search are closed. A future controlled-proxy test must be physically different and preregistered, such as a geometry-only anisotropic metric derived from active-camera transverse-gradient sensitivity.

This is not paired experimental BOST, a learned predictor, external generalization, a wall/RSS result, or an algorithm breakthrough.
