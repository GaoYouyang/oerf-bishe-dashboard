# v161：几何各向异性保住 11/12 分层，但没有救回原五相机缺口

更新：2026-08-19

## 先说结论

v161 检验一个与 v160 物理上不同、且不需要训练的解释：五相机失守是否来自三个世界坐标轴的观测灵敏度不均衡。结果前只冻结一个公式：从 active cameras 实际进入 forward 的单位射线方向计算 `s_j = mean(1-d_j^2)`，再取 `w_j = geometric_mean(s)/max(s_j,1e-12)` 形成对角各向异性 H1 惩罚。固定倍数仍为 `0.03`，没有搜索权重、floor、clip、阶数或倍数。

正式运行与独立第二实现得到相同判决：`FAIL_GEOMETRY_ANISOTROPIC_H1_TEMPORAL_V161`。

- 12 个时间×相机分层通过 `11/12`；
- 唯一失败仍是 `t=0.75`、五相机；
- 该层 field / gradient / observation p90 为 `0.417905 / 0.768197 / 0.119424`；
- gradient p90 门是 `0.750000`；
- 同一层冻结的各向同性 H1 gradient p90 为 `0.758639`，新方案反而变差 `0.009558`。

因此，active-ray 三轴横向灵敏度不均衡本身不是缺失机制。当前对角几何各向异性路线关闭，不围绕已见结果继续改权重或 lambda。

## 十二个冻结分层

下表列出各向异性主策略的 p90，顺序为 field / gradient / observation。

| 归一化时间 | 5 相机 | 7 相机 | 9 相机 |
| ---: | :--- | :--- | :--- |
| 0.00 | 0.429 / 0.723 / 0.133，通过 | 0.365 / 0.580 / 0.141，通过 | 0.239 / 0.500 / 0.149，通过 |
| 0.25 | 0.447 / 0.719 / 0.116，通过 | 0.358 / 0.584 / 0.134，通过 | 0.247 / 0.510 / 0.146，通过 |
| 0.75 | 0.418 / **0.768** / 0.119，失败 | 0.347 / 0.618 / 0.131，通过 | 0.240 / 0.541 / 0.143，通过 |
| 1.00 | 0.430 / 0.728 / 0.120，通过 | 0.347 / 0.580 / 0.138，通过 | 0.234 / 0.504 / 0.148，通过 |

冻结 p90 门分别是 field `0.50`、gradient `0.75`、observation `0.20`。对应 worst 门、CGLS K16 对照和调用账也被逐项检查。

## 为什么这个负结果可信

正式程序覆盖 `39` 个 operator setups、`1,404` 个 cells 和 `4,212` 条三臂结果。三轴灵敏度范围是 `0.172859–0.998331`，权重范围是 `0.524955–3.025801`；这说明方法确实产生了明显各向异性，并非退化成原 H1。

相机换序后的灵敏度、权重和惩罚最大相对差为 `9.49e-15`，因此机制满足相机置换不变。独立程序重新构建射线、灵敏度、权重、DCT 惩罚、特征分解、候选场、观测与全部指标。`19/19` 项检查通过；正式与独立逐 cell、汇总和算子数值最大差分别为 `1.64e-11 / 7.65e-12 / 1.59e-11`。

第一次执行曾因一个实现附加检查误用绝对容差而在解释结果前 fail-closed；只修正该检查与冻结协议的相对容差一致性后才产生这里报告的正式结果。物理公式、数据、门和指标都没有改变。

## 接下来怎么走

停止围绕当前对角几何各向异性继续调权重或 lambda，也不训练 CNN、FNO、UNO 或 DeepONet 来挽救这个经典机制。现有受控代理若没有一个真正不同、能在结果前写清并可证伪的物理机制，就停在这里等待新物理信息。

真实 BOST 的下一关键输入仍是逐工况配对的二维双分量位移，以及相机、帧、标定、checkpoint、时间、单位、符号、crop/resize/mask、重复背景噪声和组内认可基线。

`algorithm_breakthrough=false`、`paper_success=false`、`resource_speedup=false`、`real_bost=false`、`gpu_rental_authorized=false`。

---

# v161: geometry-derived diagonal anisotropy does not repair the original five-camera miss

Updated: 2026-08-19

v161 preregisters one physically different, training-free explanation for the sparse-view failure. Active reported world-frame ray directions define `s_j = mean(1-d_j^2)`, and the fixed diagonal H1 weights are `w_j = geometric_mean(s)/max(s_j,1e-12)`. The multiplier stays at `0.03`; no weight, floor, clipping, order, or multiplier search is performed.

The formal run and independent implementation agree on `FAIL_GEOMETRY_ANISOTROPIC_H1_TEMPORAL_V161`. The primary clears 11/12 frozen time-by-camera strata. The sole miss remains five cameras at `t=0.75`, where field / gradient / observation p90 are `0.417905 / 0.768197 / 0.119424`. The frozen gradient gate is `0.750000`, and the isotropic H1 control reaches `0.758639`; the proposed anisotropy therefore worsens the missed gradient tail by `0.009558`.

This is a nondegenerate test: axis sensitivities span `0.172859–0.998331`, and axis weights span `0.524955–3.025801`. Camera reordering changes the sensitivity, weights, and penalty by at most `9.49e-15` relatively. A second implementation rebuilds rays, weights, DCT penalties, eigensystems, fields, observations, and all decisions. All `19/19` checks pass; maximum formal-versus-independent per-cell, summary, and operator differences are `1.64e-11`, `7.65e-12`, and `1.59e-11`.

The result closes the current diagonal geometry-anisotropy mechanism and any post-hoc tuning of its weights or lambda. It is a controlled virtual straight-ray proxy, not a learned predictor, paired experimental BOST, external generalization, a wall/RSS result, or an algorithm breakthrough. Further work waits for paired experimental two-component displacements or a genuinely different preregistered physical mechanism.
