# v24：三条公开 CFD 外部代理复现通过

更新时间：2026-07-28

## 一句话结论

我把已经固定的 **Observable Reduced Residual Warm K1** 原样放到三条在结果前
没有进入项目数据根目录的公开 PoolFire 轨迹上。三条轨迹逐条通过预先固定的
field、gradient、observation 兼容门和资源门；独立实现把候选场复算到
`2.20e-15` 以内，指标差不超过 `2.22e-16`，benchmark 统计差为 `0`。

相对每帧都运行完整 proposal 的父模型，101 帧在线账从
`202A + 202A^T = 404` 降为 `202A + 152A^T = 354`，少 50 次精确伴随；
三条轨迹 fresh-process wall 中位降幅分别为 **13.48% / 11.81% / 12.80%**。

这是目前最重要的 **公开 CFD 代理复现里程碑**。它不是算法突破、不是有效官方
盲测、不是独立外部数据集，也不是组内真实 BOST 或论文成功。

## 实际做了什么

方法没有在看见三条轨迹的结果后改模型、改 rank、改 solver 或改阈值：

1. 101 帧中的 51 个偶数帧运行已经固定的 compact dual proposal。
2. 每个 proposal 经过精确 `A^T` lift、可见观测解析 alpha 和未修改的
   strict CGLS K1。
3. 50 个奇数帧不再运行 proposal 和 `A^T`；只用当前可见 observation residual，
   在五条 fit trajectory 建立的 rank-199 低秩包中恢复 warm lift。
4. 每条外部代理轨迹都与相同的 Zero-CGLS K4、完整父模型、兼容包络和成本账比较。
5. 三条轨迹全部完成预测后才统一评分；随后由不同数值路径重新构造低秩包、候选场、
   指标与 benchmark。

这里真正被节省的是 **50 次 proposal 和 50 次精确 `A^T`**，不是给候选少算误差，
也不是用更差终点换速度。

## 三条轨迹的逐条结果

### 精度和坏尾部

| 外部代理轨迹 | all-frame match | skipped match | joint harm | severe harm |
|---|---:|---:|---:|---:|
| `p=33kw_size=05` | 101/101 | 50/50 | 0 | 0 |
| `p=45kw_size=01` | 101/101 | 50/50 | 0 | 0 |
| `p=58kw_size=05` | 99/101 | 48/50 | 0 | 0 |

`p58-s05` 的两处未匹配只发生在 observation 指标，field 和 gradient 仍匹配：

| 帧 | candidate observation | Zero-K4 | 冻结容差 | 超出 match 线 |
|---:|---:|---:|---:|---:|
| 77 | 0.286727 | 0.259892 | 0.025989 | 0.000846 |
| 79 | 0.301481 | 0.269967 | 0.026997 | 0.004517 |

它们让 all-frame match 从 100% 变成 98.02%，但没有越过 harm 线，因此整条轨迹
仍按原合同通过。页面保留这两帧，是为了防止“三条都通过”掩盖最接近失败的位置。

### p90 精度

| 轨迹 | 指标 | Reduced K1 | Zero-K4 | Reduced / Zero |
|---|---|---:|---:|---:|
| p33-s05 | field | 0.677943 | 0.677119 | 1.0012 |
|  | gradient | 1.025329 | 1.045570 | 0.9806 |
|  | observation | 0.323804 | 0.320514 | 1.0103 |
| p45-s01 | field | 0.708273 | 0.708872 | 0.9992 |
|  | gradient | 1.471311 | 1.498751 | 0.9817 |
|  | observation | 0.378907 | 0.380009 | 0.9971 |
| p58-s05 | field | 0.730187 | 0.731234 | 0.9986 |
|  | gradient | 1.003334 | 1.026264 | 0.9777 |
|  | observation | 0.358724 | 0.346775 | 1.0345 |

“兼容”表示它们落在结果前冻结的单侧同精度包络内，不表示所有指标都逐项优于
Zero-K4，更不表示当前 proxy 的绝对重建误差已经足够低。

## 时间、调用与内存

| 轨迹 | fresh wall 降幅 | paired 中位降幅 | paired bootstrap 95% 区间 | primary 更快 |
|---|---:|---:|---:|---:|
| p33-s05 | 13.48% | 13.70% | 13.10%–14.09% | 99/101 |
| p45-s01 | 11.81% | 11.83% | 11.05%–13.31% | 93/101 |
| p58-s05 | 12.80% | 13.52% | 11.94%–14.51% | 91/101 |

交替执行顺序后，各轨迹的 primary-first / primary-second 中位降幅分别是：

- p33-s05：13.51% / 13.83%
- p45-s01：12.38% / 11.52%
- p58-s05：12.97% / 13.52%

因此 wall 信号不是只由一种 arm 顺序制造。三条轨迹等权 fresh wall 降幅中位数
为 **12.80%**，均值为 **12.70%**。

RSS p90 ratio 分别为 `1.0161 / 1.0123 / 1.0198`。它们都通过不超过 1.05 的
no-harm 门，但这表示 **内存没有明显恶化**，不是内存下降。

## 完整成本账

| 方法，每 101 帧 | A | A^T | 总调用 |
|---|---:|---:|---:|
| Observable Reduced Warm K1 | 202 | 152 | 354 |
| Full-parent Warm K1 | 202 | 202 | 404 |
| Zero-CGLS K4 | 404 | 404 | 808 |

相对完整父模型：

- 总 `A+A^T` 减少 12.38%
- `A^T` 减少 24.75%

相对 Zero-K4：

- 总 `A+A^T` 减少 56.19%

rank-199 package 的离线 setup 是 `200A + 200A^T = 400`，没有藏入在线账。
若把 `A` 与 `A^T` 视为同成本且 package 可以复用，需 **8 条 101 帧序列，
即 808 帧** 才能相对完整父模型摊平 setup。真实 BOST 上还要重新测 setup、
I/O 和相机处理，不能直接沿用这个盈亏点。

## 独立复算

正式路径使用 direct thin SVD 和 measurement-space SVD ridge；独立路径改用
sample-space symmetric eigendecomposition、field QR 和 normal-equation ridge，
并重新执行所有 K1、Zero-K4、指标和资源统计。

| 复核项 | p33-s05 | p45-s01 | p58-s05 |
|---|---:|---:|---:|
| candidate 最大绝对差 | 2.03e-15 | 1.49e-15 | 2.19e-15 |
| metric 最大绝对差 | 2.22e-16 | 2.22e-16 | 1.11e-16 |
| reference 最大绝对差 | 0 | 0 | 0 |
| benchmark 统计最大差 | 0 | 0 | 0 |

三条 pair 绑定和完整结果树在独立验证前后都没有变化。最终独立状态：

```text
PASS_INDEPENDENT_EXTERNAL_HOLDOUT_RECOMPUTATION_V24
PASS_EXTERNAL_PROXY_REPLICATION
```

## 为什么这次比 v23 更有意义

v23 的数值信号来自两条已经因执行错误打开的 official-test 轨迹，只能算 post-open
支持性诊断。v24 在结果前冻结了三条此前不在项目数据根目录中的公开 PoolFire
train-split 轨迹，并规定三条必须逐条过门，不能用平均值掩盖一条失败。

三条轨迹是新的功率/尺寸组合，但功率值和尺寸值都曾在 fit 数据出现。因此这次可以
说“跨三个新组合的公开外部 proxy 复现”，不能说 unseen-power、unseen-size、
geometry OOD、官方确认集或独立外部数据集。

## 排除“只是轨迹特别平滑”的解释

三条 holdout 的相邻帧变化量 p90 为 `0.282 / 0.206 / 0.338`，其中 p58-s05
略高于五条 fit trajectory 中原有的最大值 `0.333`，仍在 50 个 skipped frames
中通过 48 个。这削弱了“只对异常平滑序列有效”的解释，但不能证明任意高动态流场
都能泛化。

## 原创性边界

截至 2026-07-28 的一级来源核查没有找到与本方法完全同构的公开方案，但每个组件
都有强近邻：

- warm start 和网络-迭代耦合已有系统研究；
- WB-IPM 已把学习方向与投影/Krylov refinement 结合；
- FCG-NO、NeurKItt、Learned ReSeSOp 等已覆盖学习预条件、子空间或迭代修正；
- NeRIF、NeDF、Pyramid-BOST 已覆盖神经场、BOST 反演和粗到细初始化。

因此不能声称“第一个 learned warm start”“第一个 neural-Krylov”或“第一个
AI-BOST”。当前可防守的差异是：

> 面向 BOST 的观测条件化 measurement-space 低秩 proposal，经精确 `A^T`
> range lift、可见观测解析标量和未修改 strict CGLS K1，在 field / gradient /
> observation 同精度包络下按完整算子调用与 wall 统一记账。

这是一条限定日期和来源范围的差异性陈述，不是全球唯一性证明；仍需何远哲师兄
核对组内未公开方案、专利和代码。

## 成功、失败与突破判断

**成功：**

- 三条冻结的公开外部 proxy 轨迹逐条通过精度与资源门；
- 三条均无 harm、无 severe harm；
- 减少 50 次 proposal 和 50 次精确 `A^T`，wall 降幅稳定在 11.81%–13.48%；
- 不同数值路径在浮点误差内完整复算。

**仍未成功：**

- 没有有效 official confirmatory test；
- 没有接组内真实 BOST 位移、相机标定、重复实验和噪声分布；
- 没有证明物理同精度、任意工况泛化、真实设备 wall 或全流程内存优势；
- 当前绝对 field / gradient 误差仍高，proxy 正演仍可能存在 inverse crime。

**突破性进展判定：**

```text
important_proxy_replication_milestone=true
algorithm_breakthrough=false
official_confirmatory_test=false
valid_untouched_official_test_count=0
real_BOST=false
global_generalization_proven=false
paper_success=false
```

下一项真正能改变论文等级的证据，不是再加一个公开 CFD 工况，而是把冻结方法迁移到
组内真实 BOST：用重复采集定义实验噪声，用标定不确定度定义“同精度”，再比较组内
forward / reconstruction 的完整 `A/A^T`、wall 和全流程峰值内存。

脱敏机器可读数据见
`docs/poolfire_c_observable_external_v24_public_summary.json`。
