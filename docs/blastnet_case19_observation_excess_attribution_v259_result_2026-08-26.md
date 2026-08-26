# v259：Case 19 剩余观测误差具有相机局部结构

## 这次只回答一个问题

v258 已经是有效负结果：Krylov 正交补热修正通过 13/13 个绝对门，但相对 K16 的 matched-accuracy 为 0/13，唯一系统性阻塞来自观测误差。v259 不生成新候选、不改场、不调参数，也不增加任何 `A/A^T` 调用；它只把 v258 相对 K16 多出的观测残差能量按相机、位移分量和二维频带做加性分解，判断阻塞究竟是局部结构还是弥散误差。

结果前固定的局部门为：某种分组的主导份额至少为 75%，并且至少 10/13 套 rig 达到该门。若多种分组同时满足，裁决优先级固定为相机、分量、频带、弥散，不能看到结果后改选。

## 独立结果

完全独立的第二实现重新读取封存的 v258/K16 残差，独立完成相机切分、双分量切分、正交二维 DCT 频带切分、Parseval 闭环和全部汇总。`18/18` 项检查全真；父观测指标重放最大差为 `5.00e-16`，归一化数组最大差为 `2.82e-9`，能量数组最大相对差为 `8.75e-10`，汇总最大差为 `2.75e-9`，Parseval / 加性闭环误差为 `1.97e-15`。

v258 相对 K16 的剩余超额中：

- 前三相机的正超额份额在 `10/13` 套 rig 达到 75% 门；其中位份额为 `0.811`，最差 rig 为 `0.610`。
- 第二个位移分量在 `13/13` 套 rig 达到门，中位正超额份额为 `1.000`。
- 高频带在 `12/13` 套 rig 达到门，中位正超额份额为 `0.924`。
- 总残差能量相对 K16 的 p50 / p90-higher / worst 比为 `1.366 / 1.504 / 1.869`。

两个对照说明这不是所有残差差异都会产生的通用图样。被正交补移除的线性热部分只呈现分量集中，相机与频带均为 `0/13`；raw K14 相对 K16 则在相机、分量和频带上全部为 `0/13`，按合同属于弥散。

## 判决与边界

封存判决为 `POST_OPEN_CASE19_CAMERA_LOCAL_OBSERVATION_EXCESS_V259`。按结果前优先级，只授权下一轮另行冻结一个相机局部、部署可见且可独立证伪的诊断；分量与高频集中是重要的描述性支持，但不能在事后替代 primary 决策。v259 没有新增场、物理 replay 或算子调用，也没有运行完整序列、训练、GPU、wall/RSS 或外部门。

这是一条改变下一步方向的机制归因，不是算法通过。`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`real_bost=false`。

# v259: the remaining Case 19 observation excess is camera-local

## The single question

v258 is already a valid negative result. Its Krylov-complement heat correction clears all 13 absolute cells but reaches 0/13 under K16-matched accuracy, with observation as the only systematic blocker. v259 creates no new candidate, changes no field, tunes no parameter, and adds `0A+0A^T`. It only decomposes the observation-residual energy in excess of K16 by camera, detector-displacement component, and orthonormal 2D frequency band to distinguish localized structure from diffuse error.

The preregistered localization gate requires a dominant share of at least 75% in at least 10 of 13 rigs. If several groupings satisfy that gate, the frozen adjudication priority is camera, component, frequency, then diffuse; it cannot be changed after seeing the result.

## Independent result

A fully independent implementation rereads the sealed v258 and K16 residuals and rebuilds camera groups, two-component groups, orthonormal 2D DCT bands, Parseval closure, and every summary. All `18/18` checks pass. Maximum differences are `5.00e-16` for parent observation-metric replay, `2.82e-9` for normalized arrays, `8.75e-10` relative for energy arrays, and `2.75e-9` for summaries. Parseval and additive closure error is `1.97e-15`.

For the remaining v258-over-K16 excess:

- The top three cameras clear the 75% gate in `10/13` rigs, with median share `0.811` and worst-rig share `0.610`.
- The second displacement component clears the gate in `13/13` rigs, with median positive-excess share `1.000`.
- The high-frequency band clears the gate in `12/13` rigs, with median positive-excess share `0.924`.
- Total residual-energy ratios over K16 are `1.366 / 1.504 / 1.869` at p50 / p90-higher / worst.

The controls show that this is not a generic pattern of every residual difference. The linear-heat portion removed by the complement is component-local only, with camera and frequency localization both at `0/13`. Raw K14 versus K16 is `0/13` for camera, component, and frequency localization and is therefore diffuse under the frozen contract.

## Verdict and boundary

The sealed decision is `POST_OPEN_CASE19_CAMERA_LOCAL_OBSERVATION_EXCESS_V259`. Under the preregistered priority, it authorizes exactly one separately frozen camera-local, deployment-visible, independently falsifiable diagnostic. Component and high-frequency concentration are important descriptive support, not post-hoc replacement primaries. v259 adds no field, physical replay, or operator calls and runs no full sequence, training, GPU, wall/RSS, or external gate.

This attribution changes the next scientific question; it is not an algorithmic pass. `algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `real_bost=false`.
