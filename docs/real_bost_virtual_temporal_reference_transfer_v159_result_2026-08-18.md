# v159.1：虚拟时序配对可由代码生成，但固定正则仍在一个五相机分层失守

更新：2026-08-18

## 先说结论

师兄进一步明确了现有模型的使用语义：第一个输出作为三维重建密度场，时间输入归一化到 `0–1`；在虚拟数据阶段，相机、三维场和二维双分量投影可以由代码受控组合，不要求它们具有真实实验中的一一配对关系。

这项澄清足以继续做**受控虚拟代理**。v159.1 因而没有索取不存在的真实对应表，而是直接检验 v158 中仅作诊断的固定正则 `0.03` 能否跨时间工作。结果前固定四个新时间点 `t=0 / 0.25 / 0.75 / 1`，并在每个时间点检查 5/7/9 相机。

正式运行与独立第二实现得到相同判决：

- `12` 个时间×相机数分层中，`11` 个通过全部冻结门；
- 唯一失败是 `t=0.75`、5 相机；
- 该分层 field p90 为 `0.357930`、observation p90 为 `0.118507`，均通过；
- gradient p90 为 `0.758639`，比 `0.750000` 门高 `0.008639`；
- 合同要求 `12/12` 全部通过，因此判决为 `FAIL_TEMPORAL_REFERENCE_TRANSFER_V159_1`。

这是接近通过的负结果，不是突破。现在不能事后改用 `0.1`，也不能把 `11/12` 写成时序泛化成功。

## 四个时间点的结果

下表列出固定 `0.03` 主策略的 p90；括号内为 field / gradient / observation。

| 归一化时间 | 5 相机 | 7 相机 | 9 相机 |
| ---: | :--- | :--- | :--- |
| 0.00 | 0.383 / 0.709 / 0.133，通过 | 0.290 / 0.573 / 0.141，通过 | 0.214 / 0.494 / 0.148，通过 |
| 0.25 | 0.379 / 0.696 / 0.115，通过 | 0.291 / 0.570 / 0.133，通过 | 0.212 / 0.510 / 0.146，通过 |
| 0.75 | 0.358 / **0.759** / 0.119，gradient p90 失败 | 0.287 / 0.593 / 0.130，通过 | 0.208 / 0.539 / 0.142，通过 |
| 1.00 | 0.371 / 0.712 / 0.119，通过 | 0.292 / 0.572 / 0.137，通过 | 0.205 / 0.505 / 0.147，通过 |

冻结 p90 门分别是 field `0.50`、gradient `0.75`、observation `0.20`。每个分层的 worst 门也被逐项检查；唯一失败仍是上面的 gradient p90。

## 做了什么，以及为什么这样做

本轮保留 v158 的九个可执行三维场、十三套九相机标定、DCT1024 表示和 H1 Tikhonov 解法。固定正则 `0.03` 是在已经开封的 `t=0.5` 开发诊断中得到的线索；四个新时间点出现结果前，它已被单独冻结为唯一主假设。

相机与三维场的组合、5/7/9 相机子集以及二维双分量观测全部由代码生成。这正是师兄所说的虚拟数据构建方式，但它不等于真实实验配对。

共检查 `39` 个 operator setups、`1,404` 个 cells 和 `2,808` 条 arm rows。主策略在几何缓存后的逻辑在线账是每 cell `1A+1A^T`；同时公开披露 `13,299` 次几何 basis setup 投影和离线真值/候选投影，未把这些成本写成免费部署收益。

## 独立复算

独立程序没有调用正式 v159.1 的算子或求解辅助函数。它改用解析余弦基、另一套相机射线与稀疏算子实现，以及另一种稳定特征分解，重新生成全部 `1,404` cells、`2,808` arm rows、`12` 个分层和最终判决。

`17/17` 项检查全部通过。逐 cell 指标最大差为 `1.64e-11`，汇总最大差为 `7.12e-12`，算子数值最大差为 `1.60e-11`；所有离散判决一致。

最初的 v159.0 在评分前错误地把“密度输出”解释成“每个重建体素必须严格为正”。已有封存审计早已表明某个重建模型在两个时间点有少量非正值，因此该协议在读取逆问题结果前失效。v159.1 只修正这项已知语义错误：要求 finite，记录非正值，但不裁剪、不平移、不把符号作为有效性门；时间点、相机数、正则、指标和阈值均未变化。

## 证据边界与下一步

这次结果确认了师兄所说的虚拟数据生成逻辑可以直接执行，但也否定了“固定 `0.03` 已经具备严格时序稳健性”。当前不训练 predictor，不租 GPU，不运行 fresh wall/RSS，也不把代码生成的二维观测称作真实实验 BOST。

真实迁移仍需要逐工况实验二维双分量位移，以及相机/帧/标定对应、分量顺序与符号、图像处理约定、重复测量噪声和认可基线。若继续研究新的虚拟机制，也必须与本次固定正则在物理上不同，并在看结果前冻结，不能围绕这个唯一失败分层继续调参。

`algorithm_breakthrough=false`、`paper_success=false`、`resource_speedup=false`、`real_bost=false`、`gpu_rental_authorized=false`。

---

# v159.1: virtual temporal pairings can be generated in code, but fixed regularization misses one five-camera stratum

Updated: 2026-08-18

The senior collaborator clarified that the first model output is the reconstructed density field, time is normalized to `[0,1]`, and camera-field pairings plus two-component projections may be generated in code for a controlled virtual dataset. They need not claim one-to-one correspondence with a real experiment.

v159.1 therefore tests one preregistered temporal-transfer hypothesis directly. The fixed multiplier `0.03`, identified only as a diagnostic on the already opened `t=0.5` development result, is frozen before evaluating `t=0 / 0.25 / 0.75 / 1` under 5/7/9 cameras.

Eleven of twelve time-by-camera strata pass every frozen field, gradient, observation, and worst-case gate. The only failure is five cameras at `t=0.75`: field p90 is `0.357930`, gradient p90 is `0.758639`, and observation p90 is `0.118507`. Gradient p90 exceeds its `0.750000` gate by `0.008639`, so the all-strata decision is `FAIL_TEMPORAL_REFERENCE_TRANSFER_V159_1`.

The formal run covers 39 operator setups, 1,404 cells, and 2,808 arm rows. An independent implementation rebuilds the analytic cosine basis, camera rays, sparse operators, stable eigensystems, physical fields, metrics, and all 12 decisions. All `17/17` checks pass; maximum per-cell, summary, and operator-numeric differences are `1.64e-11`, `7.12e-12`, and `1.60e-11`.

This confirms that the clarified virtual-data mechanics are executable, but it does not establish strict temporal transfer. The result cannot be rescued after the fact by switching to multiplier `0.1`. No paired experimental 2D displacement, learned predictor, wall/RSS result, external generalization, curved-ray validation, real BOST, or algorithm breakthrough is present.
