# N2-PVGR N5-D4b：32-cell 场导数总体扩展预注册

日期：2026-07-19

状态：结果产生前预注册

候选：`N2-PVGR-N5-D4B-CLOSED-POPULATION-FIELD-DERIVATIVE`

## 1. 这一步在证明什么

D4 已在 4 个经过选择的 synthetic cell 上验证了离散前向算子

\[
F:n[17,17,17]\longrightarrow y[4,2]
\]

的 JVP、VJP、中心有限差分和有序程序拓扑合同。D4 的通过只允许设计一个新的 32-cell 扩展，不能直接进入三维重建或神经算子训练。

D4b 的唯一问题是：**同一个导数实现合同能否在 N4/D3 已冻结的完整 32-cell 开发审计总体上逐项成立？**

这不是模型性能实验，也不是重建实验。它是把“4 个选中上下文里导数代码看起来正确”推进到“完整的封闭开发 census 中没有发现上下文依赖的导数错误”。

## 2. 为什么不能把 32 写成 32 个独立样本

32 个 cell 由 16 个 matched pair 组成；每一对共享场、stress 等条件，只改变预先指定的几何因素。32 个 cell 进一步聚成 5 个 field unit：

| field unit | cell 数 |
|---|---:|
| `smooth-s1729` | 6 |
| `smooth-s1871` | 6 |
| `smooth-s1987` | 2 |
| `smooth-s2131` | 6 |
| `wrinkled-s3163` | 12 |

因此，D4b 必须同时给出 cell、pair cluster 和 field-unit cluster 三层账本。方向、map 和有限差分步长都是同一上下文内的诊断，不得作为新的统计样本。D4b 不做显著性检验，也不报告把 32 当 iid 的置信区间。

## 3. 冻结总体

选择规则只有一条：使用 D3 `reference_pack.json` 中按冻结顺序排列的全部 32 个 cell，不做任何结果驱动筛选。

- pair：`p01` 至 `p16`，每对顺序固定为 `n3_failure`、`matched_control`；
- family：20 个 smooth、12 个 wrinkled；
- stress：12 个 stress 1、10 个 stress 3、10 个 stress 10；
- D3 cell-order SHA-256：`80640caed7bad515bc7449993ea0e18c7556be03b83bf26e28d2f83ee7ffa79e`；
- 每个 cell 保留 D3 identity、reference method、source kind 和 selection basis，只作为来源追踪，不把 mixed reference pack 称作外部真值。

N4 的原始收敛门并没有授权全部 32 个最终参考；后续 D3 通过预注册的自适应路线形成了 32-cell mixed reference pack。D4b 不会抹去这段来源差异。导数审计本身在 step 16 前向程序上进行，并不使用 D3 的数值参考去决定通过与否。

## 4. 新方向而非复制 D4

每个 cell 固定两个场方向：

1. `smooth_avg_pool3d_kernel3_stride1_padding1`；
2. `raw_torch_randn`。

两者均将两层边界置零并做 L2 归一化。每个方向配一个归一化 Rademacher detector cotangent。D4b 使用未搜索的新 seed base：

- direction：`66001`；
- cotangent：`66101`；
- seed rule：`stable_seed(base, cell_id, direction_index)`。

所有实际 field、4-ray prefix、direction 和 cotangent 数组都在正式结果之前写入一次性冻结 NPZ，并在 attestation 中记录逐数组 SHA-256。

NPZ、attestation 与 `READY.json` 不分别暴露到最终路径。构建器先在同一文件系统的隐藏 staging 目录内完整写入并 fsync 三者，校验互相的 SHA-256 后，以一次目录 rename 原子发布整个 preregistration bundle。若进程在最终 rename 前中断，正式路径仍完全不存在；只有带有效 `READY.json` 的完整 staging bundle 可以由固定 recovery 命令原样晋升，不能重算或替换其中任何字节。不完整 staging 必须保留并另行预注册恢复处理。

## 5. 四个 map

对每个 cell、每个方向都审计：

1. `curved_detector`；
2. `straight_detector`；
3. `raw_curved_minus_straight`；
4. `paired_neumaier_residual`。

同时检查两条结构控制：

- raw = curved - straight；
- paired 与 raw 在既定累计误差门内一致。

结构控制是代码接线/累计顺序证据，不是额外物理模型证据。

## 6. 阈值完全继承 D4

D4b 不根据 D4b 结果调阈值。以下门逐字继承已 attested 的 D4：

| 门 | 阈值 |
|---|---:|
| dot relative defect | `<= 1e-10` |
| absolute dot signal | `>= 1e-16` |
| best-h FD relative L2 | `<= 1e-6` |
| required-h FD relative L2 | 每个 `<= 1e-5` |
| repeated primal relative L2 | `<= 2e-12` |
| direction norm error | `<= 1e-12` |
| domain/stencil margin | `>= 0` |

有限差分固定为 symmetric central：

`[1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]`

其中前三个 h 都必须通过 required-h 门；不能只挑最好看的 h。每个 h 还必须具有足够的 symmetric-difference signal，并通过基于 float64 epsilon 和 output scale 的 roundoff-aware absolute gate。

## 7. 有序程序拓扑

对每个 field direction，记录 base 以及 7 个 h 的正负扰动，共 15 次 signature。signature 覆盖：

- 每条 ray 的每个 RK4 stage；
- curved/straight path midpoint；
- central stencil 的固定 offset 顺序；
- interpolation cell；
- support bit；
- frustum-margin sign。

任一正负扰动相对 base 改变 signature，判定 `D4B_DERIVATIVE_CONTEXT_CHANGED_FAIL_CLOSED`。这避免把跨越离散分支边界后的有限差分误称为同一局部导数。

## 8. 精确计数和成本

| 项目 | D4b 预注册值 |
|---|---:|
| cell | 32 |
| pair cluster | 16 |
| field unit | 5 |
| direction context | 64 |
| map context | 256 |
| structural control | 128（每个 direction 两条） |
| topology context | 64 |
| topology signature | 960 |
| JVP sweep | 256 |
| VJP sweep | 256 |
| finite-difference forward | 3,584 |
| map closure invocation | 4,096 |
| derivative logical queries | 8,257,536 |
| topology logical queries | 4,300,800 |
| D4b logical queries | 12,558,336 |
| D4b interpolation dispatches | 1,899,392 |

D4 已经完成的 strong full-path versus frozen-path toy control 不重复运行。若把该一次性 parent control 计入整条证据链，lineage 账本是 12,561,696 logical queries 和 1,900,078 dispatches。host sync 与 saved-tensor bytes 仍未仪器化，必须写成“未测”，不能写成零。

## 9. 严格判决

通过需要：

- 256/256 map context 通过；
- 128/128 structural control 通过；
- 64/64 topology context 稳定；
- 32 个 cell、16 个 pair、5 个 field unit 全部保留；
- 正式执行没有 retry；
- 独立 validator 重建全部冻结输入、重算全部导数数组、拓扑、成本、聚类账本和声明边界。

这里的“独立”是指 validator 不导入 D4b runner、导数 gate helper、D4b frozen-input builder 或 program-signature serializer，并自行重算审计逻辑；它仍共享 PyTorch 和底层前向物理原语，不等于独立软件栈或独立研究者复现。

任一 map/cell/direction/structure/topology 失败都不能被均值抵消。

可能的机器判决：

- `D4B_CLOSED_32_CELL_FIELD_DERIVATIVE_CONTRACT_VALID`；
- `D4B_POPULATION_FIELD_DERIVATIVE_FAIL_CLOSED`；
- `D4B_DERIVATIVE_CONTEXT_CHANGED_FAIL_CLOSED`。

输入、代码、阈值、seed、h、map order 或 parent identity 漂移时，在正式结果之前停止，不生成一个“看起来失败但实际协议已变”的结果。

## 10. 通过后仍然不能说什么

即便 D4b 通过，也只有 `closed_n4_32_cell_field_derivative_contract` 可以变为 true。以下仍必须为 false：

- 独立总体/外部分布上的 field JVP/VJP；
- `decoder(theta) -> field -> detector` 参数链；
- 三维重建；
- neural operator training 或 superiority；
- real data；
- generalization；
- paper claim。

通过后只允许预注册 D4c decoder-parameter-chain derivative gate。没有 D4c，就不能把场空间导数自动等同于可训练网络参数的端到端梯度。

## 11. 预先承诺的失败处理

- 不删除 `_work` 目录后偷偷重跑；
- 不删除或拼补 preregistration staging bundle；完整 staging 只能经哈希核对后原子晋升，不完整 staging 进入单独 recovery protocol；
- 不因边缘失败改 seed、加方向、删 cell、换 h 或放宽阈值；
- 不把部分成功写成总体成功；
- 若运行中断，保留工作目录并另行预注册 recovery protocol；
- 若 validator 与 runner 不一致，保留冲突并 fail closed。

这一步的价值在于把后续重建训练最基础、也最容易被忽略的一件事钉牢：我们优化的梯度确实属于所声明的离散 BOST 前向程序，而不是偶然在少数场景里对上的数值影子。
