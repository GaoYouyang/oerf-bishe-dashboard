# N2-PVGR N5-D4b 32-cell 场导数普查结果审计

## 一句话结论

N5-D4b 把 D4 的四单元实现证书扩展到 N4/D3 冻结的完整 32-cell 开发总体后，没有通过预注册的逐上下文全门：`254/256` 个 map context 通过，`128/128` 个结构控制通过，但只有 `58/64` 个 ordered-topology context 稳定。独立 validator 重新生成冻结输入、重算全部导数数组和 960 个拓扑签名后，保持同一判决：

`D4B_DERIVATIVE_CONTEXT_CHANGED_FAIL_CLOSED`

这是一份**可信的否定证据**。它说明 tiny D4 的成功不能直接外推到完整开发 census；它不授权 decoder 链、三维重建、神经算子训练、真实数据或论文算法主张。

## 1. 固定证据链

- 协议提交：`cba4f285f03cc1bc791684d240d87d0c74cbd6cb`
- 原子输入证明包提交：`c8464d0`
- 正式结果与独立验证提交：`a897aed3609357735b30a02dc8e0b9c1bca89444`
- 冻结输入归档 SHA-256：`e40fc69356b7b86455ba51a11f094cb33b41de401df9d9fc0c146f7882e50d75`
- 数值环境：CPU float64
- 场网格：`17 x 17 x 17`
- 每个 cell：4 条冻结 Sobol-prefix rays、2 个新方向、2 个新 cotangent
- 每个方向：4 种 map、7 个中心有限差分步长、15 个 ordered topology signatures
- 总体：32 cells、16 matched pairs、5 field units；不是 32 个 iid 样本

证明包在隐藏 staging 中完整写入、fsync 并互验后，以一次目录 rename 发布。正式运行没有 retry，也没有残留 `_work` 目录。

## 2. 总体结果

| 证据门 | 正式结果 | 预注册要求 | 判定 |
|---|---:|---:|---|
| map contexts | `254 / 256` | `256 / 256` | 失败 |
| structural controls | `128 / 128` | `128 / 128` | 通过 |
| combined structure contexts | `64 / 64` | `64 / 64` | 通过 |
| ordered topology contexts | `58 / 64` | `64 / 64` | 失败 |
| 最大 dot relative defect | `1.841685e-10` | `<= 1e-10` | 失败 |
| 最小 dot signal | `7.511429e-10` | `>= 1e-16` | 通过 |
| 最坏 best-h FD relative-L2 | `6.183022e-8` | `<= 1e-6` | 通过 |
| 三个强制 h 中最坏 FD relative-L2 | `5.254664e-7` | 每个 `<= 1e-5` | 通过 |
| 最小 domain margin | `0.397157` | `>= 0` | 通过 |
| 最小 stencil margin | `0.395157` | `>= 0` | 通过 |
| 最小 frustum margin | `0.00280746` | 符号稳定 | 数值余量为正 |
| retry | `0` | `0` | 通过 |

不能把 `254/256` 四舍五入成总体成功。D4b 在结果前已经承诺任何单一 map 或 topology context 失败都整体 fail closed。

## 3. 两个 map 失败在哪里

两个失败都来自同一个上下文：

- cell：`wrinkled-s3163-orientation_22-wide__stress_1`
- pair：`p14`
- role：`n3_failure`
- direction：`0`，平滑平均池化方向
- map：`raw_curved_minus_straight` 与 `paired_neumaier_residual`

| map | dot relative defect | absolute defect | dot signal | best-h FD | required-h 最坏 FD |
|---|---:|---:|---:|---:|---:|
| raw residual | `1.841685e-10` | `1.383369e-19` | `7.511429e-10` | `2.884459e-8` | `1.270058e-7` |
| paired residual | `1.534310e-10` | `1.152486e-19` | `7.511429e-10` | `2.881320e-8` | `1.270334e-7` |

这里有限差分、重复输出、非退化信号和三个强制步长都通过，失败只落在 residual map 的相对 dot 门。absolute defect 约为 `1e-19`，但预注册的是 relative gate，不能在看见结果后用 absolute defect 把它救回。

**当前机理判断仅为推断：**raw 与 paired 两条残差路线同时在同一方向失败，而 curved/straight 单项和 FD 仍通过，提示“小残差上的收缩/点积累计误差”值得单独审计；它尚未证明根因是浮点求和，也未证明 VJP 实现错误。下一诊断应在固定保存的 JVP/VJP、direction 和 cotangent 上比较 float64、补偿点积与更高精度 contraction，但该 post-open 诊断不能改变 D4b 判决。

## 4. 六个拓扑上下文为什么失败

六个失败 context 分布为：

| field / geometry | stress | direction | 发生变化的扰动 |
|---|---:|---:|---|
| `smooth-s1871 / orientation_58 / wide` | 1 | 0 | `+h`, `h=0.01` |
| 同上 | 3 | 0 | `-h`, `h=0.01` |
| 同上 | 10 | 0 | `+h`、`-h`, `h=0.01` |
| `smooth-s2131 / orientation_58 / narrow` | 3 | 0 | `+h`、`-h`, `h=0.01` |
| 同上 | 10 | 0 | `+h`、`-h`, `h=0.01` |
| 同上 | 10 | 1 | `+h`, `h=0.01` |

这 10 个 changed perturbation 中，interpolation-cell 哈希和 frustum-margin-sign 哈希保持不变，改变的是 support-bit 哈希；`h <= 0.003` 的对应 topology signatures 均保持稳定。

**讲人话：**在最大的有限差分步长上，少量采样点跨过了当前 hard support 判定边界。此时正负扰动不再执行完全相同的离散程序分支，所以不能把差商强行解释为同一个光滑局部 Jacobian。

这不等于物理折射场本身不连续。support bit 可能来自数值计算域、光阑/视场、mask 或其他程序条件；在确认其物理语义前，不能随意用 sigmoid 平滑，也不能直接删掉 `h=0.01`。

## 5. 哪些东西其实很稳

- `128/128` 结构控制全部通过：raw 始终等于 curved minus straight，paired 与 raw 保持冻结累计误差合同。
- `256/256` map 的 finite、nondegenerate、repeat-output 与多步长 FD 主体没有发现失效。
- 最坏 best-h FD 仍比 `1e-6` 门低约 16 倍；最坏 required-h FD 比 `1e-5` 门低约 19 倍。
- 全部 domain/stencil margin 为正，direction norm error 最坏 `2.22e-16`。
- 失败没有散布到所有 wrinkled 或所有 high-stress cells：map failure 只在 p14；topology failure 只在两个 smooth field units。

这些事实有助于缩小问题，但不能抵消严格全门失败。

## 6. 成本账本

| 项目 | 数量 |
|---|---:|
| D4b logical field-point queries | `12,558,336` |
| interpolation dispatches | `1,899,392` |
| map closure invocations | `4,096` |
| JVP sweeps | `256` |
| VJP sweeps | `256` |
| finite-difference forwards | `3,584` |
| topology signatures | `960` |
| lineage queries（含一次性 D4 toy） | `12,561,696` |
| failed/retried calls | `0` |
| formal wall time | `333.070 s` |
| peak RSS | `498,417,664 bytes` |

host synchronization 次数与 autograd saved-tensor bytes 仍未仪器化，结果明确写成 `not_instrumented_not_claimed_zero`。本层 CPU 审计在普通 Mac 上约 5.6 分钟完成，当前瓶颈不是网络或 GPU。

## 7. 独立 validator 的证据上限

validator 没有导入 D4b runner、gate helper、冻结输入构造器或 topology serializer。它独立完成：

1. 核对协议提交、原子证明包、D3/N4 population identity 与顺序；
2. 重生成 32 个 field、rays、64 个 directions 与 cotangents；
3. 重算 256 组 JVP/VJP/FD map audits 与全部 derivative arrays；
4. 重算 960 个 ordered topology signatures；
5. 重建成本、聚类非独立账本、判决和 claim boundary；
6. 检查结果 manifest 与 `2880 x 1620` 非空图像。

最终 `validation_report.json` 为 `valid: true`，并保持 fail-closed 判决。它仍共享 PyTorch 和底层 forward primitive，不是独立软件栈或独立研究者复现。

## 8. 现在真正值得研究的两个问题

### A. Active-set aware / topology-certified differentiable renderer

目标不是把 hard support 悄悄改软，而是先识别 support 的物理语义，再设计三种可比较合同：

1. hard support + 可计算的 local stability radius，半径外 fail closed；
2. 若有限孔径、PSF 或像素积分本来就提供连续权重，使用由真实光学模型推出的 smooth support；
3. 对确有离散遮挡/裁剪边界的情况，研究 active-set 或广义导数，而不是假装全局光滑。

这条支线的最小贡献可以是“可微 BOST renderer 的拓扑证书与拒答机制”。它比直接换一个 FNO 更贴合当前真实失败。

### B. Cancellation-aware adjoint contraction

目标是解释 residual map 的两个 dot 失败是否来自 contraction/representation，而不是先训练网络掩盖它。下一份结果前协议应固定：

1. 使用已保存 JVP/VJP 的 float64、pairwise、Neumaier 与 extended-precision contraction 对照；
2. 分别报告 curved、straight、raw residual 和 paired residual 的 absolute/relative dot defect；
3. 保留原 `1e-10` gate 作为 D4b 历史判决，不追认；
4. 若 contraction 不能解释，回到同一 forward primitive 的自定义 VJP 或程序图逐段对账。

只有先分清“数学导数错”与“极小残差上的数值 contraction 误差”，后续 decoder-chain 才有意义。

## 9. 下一步合法顺序

1. 冻结 D4b 为 fail-closed，不重跑、不换 seed、不删 cell、不改阈值。
2. 做只读 post-open forensics：定位 support flip 和 residual dot contraction；结果只能解释机制，不能救回 D4b。
3. 结果前预注册 D4b-R：明确 hard/physical-smooth support 合同、local stable radius、逐上下文全门与停止规则。
4. D4b-R 通过后才允许预注册 decoder-chain D4c；否则 decoder 与网络训练继续关闭。
5. D4c 通过后才进入 6-train-view / 2-held-out-view deterministic reconstruction。
6. 真实 displacement 单位、flow-off repeats、mask/confidence/covariance 到位后，才映射 synthetic tail 到实验 noise floor。
7. 最后才做 Picard-1、DeepONet、FNO/FFNO 与小型 residual operator 的同 split、同调用预算、同停止规则比较。

## 10. 需要问何远哲师兄

- 组内 renderer 的 support/mask 是计算域裁剪、光阑/视场、背景纹理置信度，还是实际遮挡？
- NeRIF/TDBOST 是否把越界 ray 硬置零，还是有连续像素/孔径积分？
- 组内最常见的最大 displacement/deflection 范围是多少，`h=0.01` 在真实单位里是否有意义？
- 现有代码能否导出固定小例子的 `F(x)`、`Jv`、`J^Tq` 与 support mask，供逐段对账？
- decoder 是 voxel、Fourier-feature MLP、hash grid 还是 tensor decomposition？
- 能否提供同 rig 的 flow-off repeats、mask/confidence、camera geometry、pixel pitch、magnification 与 Gladstone-Dale/密度换算？

## 11. 禁止主张

- 不能说 32 个独立样本或总体泛化；这里只有 5 个 field units。
- 不能说 D4b 基本通过；预注册要求是全门，机器判决明确失败。
- 不能把 topology 失败直接解释成真实光学不连续。
- 不能把 absolute dot defect 很小写成历史 relative gate 已通过。
- 不能进入 decoder、三维重建或 neural operator 训练并把它当作同一证据链的下一授权步骤。
- 不能宣称优于 NeRIF、TDBOST、DeepONet、FNO 或 FFNO。

## 12. 可复核入口

- [结果前预注册](n2_pvgr_n5_d4b_population_field_derivative_preregistration_2026-07-19.md)
- [机器结果](../demo_t16_operator/results/n2_pvgr_n5_d4b_population_field_derivative_v1/result.json)
- [独立验证报告](../demo_t16_operator/results/n2_pvgr_n5_d4b_population_field_derivative_v1/validation_report.json)
- [结果图](../demo_t16_operator/results/n2_pvgr_n5_d4b_population_field_derivative_v1/n2_pvgr_n5_d4b_population_field_derivative.png)
- [逐上下文结果](../demo_t16_operator/results/n2_pvgr_n5_d4b_population_field_derivative_v1/rows.json)
- [成本与摘要](../demo_t16_operator/results/n2_pvgr_n5_d4b_population_field_derivative_v1/summary.md)
