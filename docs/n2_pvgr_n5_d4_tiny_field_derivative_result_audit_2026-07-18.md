# N2-PVGR N5-D4 四单元场导数结果审计

## 一句话结论

N5-D4 在结果前固定的四个合成单元、八个场方向上下文中，通过了 detector、straight、raw residual 和 paired residual 的 JVP/VJP 对账、多步长中心有限差分、结构恒等式与有序拓扑门。独立 validator 重新生成输入并重算全部导数后给出：

`D4_TINY_SELECTED_FIELD_DERIVATIVE_CONTRACT_VALID`

这是一张**小范围网格场导数实现证书**。它允许下一份结果前预注册的 32 单元导数扩展，不允许写成三维重建成功、神经算子成功、真实 BOST 成功或论文算法成功。

## 1. 结果前固定了什么

- 协议提交：`47278d1ad2a551bd87d40e364ae7db6c15bd95c3`
- 证明提交：`b9bc2195bdb56c377e45bf5ec49408746aa4e675`
- 结果提交：`2484cd7`
- 数值环境：CPU float64
- 场网格：`17 x 17 x 17`
- 光线：每单元 4 条，来自冻结的 256-ray Sobol 序列前缀
- 积分：16 步显式 RK4
- 单元：p04/p05 的 wide failure 与 narrow matched control，共 4 个
- 场方向：每单元 2 个，分别为平滑方向和 raw Gaussian 高频方向
- 余切：等幅、全观测分量非零的 Rademacher 方向
- 有限差分步长：`1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5`
- 冻结 NPZ：478,846 bytes，SHA-256 `091e9354228b200c2ec7685be0ef0e9997e15fa78c237086f0da60a2bd585d28`

全部四格都来自 `smooth-s1871 / orientation_58`。因此它们不是四个独立流场，只是孔径和 stress 的小型实现压力测试。

## 2. 为什么需要四种 map

| map | 它检查什么 | 不能据此声称什么 |
|---|---|---|
| `curved_detector` | 场变化是否通过曲光线轨迹和积分传到探测器 | 曲光线物理已被真实相机验证 |
| `straight_detector` | 冻结直路径上的 Born-style 场导数 | 少视角反问题已可解 |
| `raw_curved_minus_straight` | 两个完整 detector 输出相减后的导数接线 | raw subtraction 是最优积分器 |
| `paired_neumaier_residual` | 同节点先减被积函数再补偿累加的独立对照 | paired route 已在 32 格统一等价 |

raw residual 按定义就是 curved minus straight，所以它的结构恒等式只算 wiring invariant。paired-Neumaier 才提供不同累加顺序的独立控制。

## 3. 正式结果

| 证据门 | 结果 | 冻结门槛 |
|---|---:|---:|
| map contexts | `32 / 32` 通过 | 每个都必须通过 |
| structural controls | `16 / 16` 通过 | 每个都必须通过 |
| ordered topology contexts | `8 / 8` 稳定 | 不允许平均 |
| 最大 dot relative defect | `2.844646e-11` | `<= 1e-10` |
| 最小 dot signal | `4.183844e-9` | `>= 1e-16` |
| 最坏 best-h FD relative-L2 | `3.062104e-8` | `<= 1e-6` |
| 三个强制 h 中最坏 FD relative-L2 | `1.484802e-7` | `<= 1e-5` |
| 最小 FD signal / roundoff floor | `1.294446e9` | `> 1` |
| 最小 domain margin | `0.428653` | `>= 0` |
| 最小 stencil margin | `0.426653` | `>= 0` |
| 最小 frustum margin | `0.00460248` | 符号不能变化 |
| full-path vs frozen-path toy JVP relative difference | `0.0455239` | `>= 1e-4` |

最接近门槛的是 `wide / stress 1 / 高频方向 / paired residual` 的 dot defect，仍低于冻结门约 3.5 倍。这个余量足以让 tiny gate 通过，但并不宽裕，也是 D4b 必须扩展更多 field/geometry 的原因。

## 4. 成本账本

| 项目 | 数量 |
|---|---:|
| field-point logical queries | `1,573,152` |
| interpolation dispatches | `238,110` |
| map closure invocations | `512` |
| JVP sweeps | `32` |
| VJP sweeps | `32` |
| finite-difference forwards | `448` |
| topology signatures | `120` |
| failed/retried calls | `0` |
| wall time | `42.997 s` |
| peak RSS | `399,572,992 bytes` |

host synchronization 次数和 autograd saved-tensor bytes 尚未仪器化，因此结果明确写成 `not_instrumented_not_claimed_zero`。

## 5. 独立 validator 做了什么

validator 不导入 D4 runner、gate helper、冻结输入构造器或程序签名 serializer，也不复用 N4/N0 runner 的展开、seed 和 rig helper。它独立完成：

1. 核对协议提交与全部 attested file 哈希；
2. 从上游配置重建 p01-p16 配对语义，并选出四个冻结上下文；
3. 重生成 field、rays、directions 和 cotangents，逐字节比较 NPZ；
4. 重算 32 个 map derivative audits 和所有结构误差；
5. 重算 120 个有序 topology signatures；
6. 重建成本账本、机器判决和 claim boundary；
7. 检查图像为 `2860 x 1739` 且非空。

最终 `validation_report.json` 为 `valid: true`。

## 6. 讲人话

可以把 forward 想成一台机器：放进三维折射率场，吐出每条光线在探测器上的两个偏折分量。

- JVP 问：“如果三维场沿某个方向轻轻变化，所有探测器读数会怎样一起变化？”
- VJP 问：“如果探测器上有一组误差，怎样一次把责任分配回三维网格？”
- dot test 检查这两个方向是否真的是同一个 Jacobian 的正反两面。
- finite difference 用实际加减一个小扰动，检查 autograd 没有悄悄断图。
- topology gate 防止扰动跨进另一个插值单元、support 或 frustum 分支后，还拿光滑导数公式硬比。

D4 的意义是：后面做 Gauss-Newton、Landweber、NeRIF 或神经算子 warm start 时，至少不是拿一条明显错误的梯度链训练。但它没有证明逆问题唯一，也没有证明四条光线足够恢复三维场。

## 7. 仍然锁住的结论

- `population_field_jvp_vjp = false`
- `three_dimensional_reconstruction = false`
- `neural_operator_training = false`
- `neural_operator_superiority = false`
- `real_data = false`
- `generalization = false`
- `paper_claim = false`

尤其要注意：primitive 是 `field[17,17,17] -> detector[4,2]`。它还没有检查 `decoder(theta) -> field -> detector` 的链式 JVP/VJP，因此不能把“网格场可微”写成“NeRIF 参数化已可训练”。

## 8. 下一步唯一合法顺序

1. **D4b 32 单元扩展：**覆盖 N4 的 16 failure-control pairs、两个 field families、多个 seeds、orientation/aperture 与 stress，不改 D4 门槛。
2. **decoder 链式导数门：**先用小 Fourier feature MLP 或低秩 basis 检查 `theta -> field -> y` 的 dot/FD。
3. **6+2 最小重建：**6 个 train views 优化，2 个 held-out views 只用于重投影审计；先用 deterministic Tikhonov/Landweber/Gauss-Newton。
4. **真实单位与 noise gate：**把 synthetic deflection 映射到组内 pixel/angle/length，并用 flow-off repeats 冻结 covariance。
5. **之后才训练模型：**Picard-1、DeepONet、FNO/FFNO 和候选 residual operator 共用相同 split、调用预算、停止规则与指标。

## 9. 需要向何远哲师兄确认

- 组内 forward 的输出到底是 exit angle、path-integrated deflection 还是 pixel displacement？
- NeRIF/TDBOST 当前用的是直接 voxel/grid field，还是 MLP/tensor decomposition 参数化？
- 是否已有匹配的 `F`、`Jv`、`J^Tq/VJP` 或可供对账的小例子？
- 能否先给 1-3 帧脱敏数据，以及 camera geometry、mask/confidence、单位和 flow-off repeats？
- 6+2 view 的 held-out cameras 在组内是否物理可行，还是必须改成 leave-one-camera/case？

## 10. 可复核入口

- [结果前协议](n2_pvgr_n5_d4_tiny_field_derivative_preregistration_2026-07-18.md)
- [机器结果](../demo_t16_operator/results/n2_pvgr_n5_d4_tiny_field_derivative_v1/result.json)
- [独立验证报告](../demo_t16_operator/results/n2_pvgr_n5_d4_tiny_field_derivative_v1/validation_report.json)
- [结果图](../demo_t16_operator/results/n2_pvgr_n5_d4_tiny_field_derivative_v1/n2_pvgr_n5_d4_tiny_field_derivative.png)
- [结果摘要](../demo_t16_operator/results/n2_pvgr_n5_d4_tiny_field_derivative_v1/summary.md)
