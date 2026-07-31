# v74.1：加载工件在完整多轨迹 fresh 资源门中正式通过

## 一句话结论

在 v73 已经证明 `1,515/1,515` 个重建单元保持 field、gradient、
interior-gradient 与 observation 精度以后，v74.1 把加载工件候选和
Zero-CGLS K4 放进完整 fresh-process 对照：

```text
30 reference workers
330 timing workers
165 randomized adjacent paired blocks
5 PoolFire trajectories
3 known nine-view geometries
101 frames per worker
```

独立 validator 重建输入、重放 reference、核对调用账并重算全部区组后，正式状态为：

```text
PASS_LOADED_ARTIFACT_FRESH_RESOURCE_STAGE_C_V74_1
PASS_INDEPENDENT_VALIDATION_LOADED_ARTIFACT_FRESH_RESOURCE_V74_1
integrity_complete=true
stage_c_pass=true
algorithm_breakthrough=false
```

候选保持 `2A + 2A^T`，Zero-K4 为 `4A + 4A^T`。全局 fresh wall ratio
为 `p50=0.56507`、`p90=0.57440`，相当于中位耗时约下降 `43.49%`。
worker-self、sampled worker-tree、sampled pipeline RSS 的全局 p90 ratio
分别为 `1.03456 / 1.01971 / 1.01409`，全部低于冻结的 `1.05` 门。

这是真正改变主线判断的资源里程碑，但仍只属于已打开 PoolFire fit family、
已知几何、无噪声 straight-ray 代理。

## 为什么 v74.0 没有被采用

第一次正式批次 v74.0 虽然完成了 30 个 reference、330 个 timing worker 和
165 个区组，但独立 validator 发现第 263 个 timing receipt 中：

```text
float monotonic duration
比同源 integer monotonic-nanosecond duration
短 0.056 nanoseconds
```

这是浮点表示边界，不是算法结果，但旧合同要求严格不小于，因此整批被判：

```text
INCONCLUSIVE_INVALID_EXECUTION_LOADED_ARTIFACT_FRESH_RESOURCE_STAGE_C_V74
```

没有从旧批次复用任何 reference、timing row 或 formal result。

v74.1 在结果前重新冻结三项修复：

1. 同时保存 raw float duration 与 integer monotonic duration；
2. 两者相差超过 1 微秒就 fail closed；
3. 计分用的 canonical wall 必须精确等于两者最大值。

旧批次的 checksum seal、formal/READY、invalid validation、VALIDATED_READY
和单次 attempt registry 也被组成私有 predecessor closure。新 runner 与独立
validator 都重新核验它，证明旧结果没有被替换或偷偷混入。v74.1 随后从零重跑
全部 360 个 worker。

## 固定候选与公平对照

### 候选

```text
known nine-view geometry
-> offline compiled rank-8 detector-space factor
-> fresh worker verifies and read-only maps artifact
-> four cheap detector actions
-> exact A^T lift
-> unchanged CGLS K1
```

每个 101 帧重建 worker 的在线精确账为：

```text
candidate    2A + 2A^T
Zero-K4      4A + 4A^T
```

加载、工件校验、Python 进程启动和完整 101 帧计算都包含在 fresh wall 与 RSS
统计内。离线工件构造单独报告，不藏进在线账，也不重复算进每条序列。

### 精度前提

资源门没有重新调精度阈值。它绑定 v73 已独立验证的完整结果：

```text
evaluated cells                    1515
compatibility PASS                 1515 / 1515
loaded/canonical metric max diff   0
failed cells                       0
```

每个 fresh worker 的输出 digest 与同一 arm、同一 trajectory、同一 geometry
的 reference 必须一致。独立 validator 重放了 30 个 reference，共 3,030 帧；
30 个 digest、30 个实际调用账全部匹配，norm-sum 最大差为 `0`。

## 正式资源结果

### Fresh wall

| 统计 | loaded q8-K1 | Zero-K4 | 配对比值 |
|---|---:|---:|---:|
| p50 | 8.9251 s | 15.7548 s | 0.56507 |
| p90 | 9.4585 s | 16.5906 s | 0.57440 |
| worst | 10.6783 s | 19.6692 s | 0.63788 |

冻结门为：

```text
global wall p50 <= 0.90
global wall p90 <= 1.05
each trajectory p50 <= 1.05
each geometry p50 <= 1.05
```

五条轨迹的 wall p50 ratio 落在 `0.56350` 到 `0.56716`；三档几何落在
`0.56501` 到 `0.56566`。没有依靠某一条特别容易的轨迹把 pooled 结果拉下来。

### RSS

| 资源比值 | p50 | p90 | worst | 冻结 p90 门 |
|---|---:|---:|---:|---:|
| worker-self high-water RSS | 1.00628 | 1.03456 | 1.08731 | <= 1.05 |
| sampled worker-tree RSS | 0.99274 | 1.01971 | 1.07281 | <= 1.05 |
| sampled pipeline RSS | 0.99611 | 1.01409 | 1.03349 | <= 1.05 |

这里必须讲清楚两件事：

- **通过的是预注册的全局 p90、逐轨迹 p50 和逐几何 p50 门。**
- worker-self 与 worker-tree 的最坏单次 ratio 仍高于 `1.05`；worst 不是本轮
  的判决门，但必须披露，不能说“每一次都更省内存”。

所以准确说法是：加载工件把旧方案的系统性 RSS 高尾伤害压回容许范围，资源门
通过；不是“内存显著下降”。候选与 Zero 的全局 pipeline p50 近乎相同，
`0.99611`。

### 离线构造与摊销

三档几何的离线构造分别为 `2.2213 / 2.2409 / 2.5578 s`。对应每条 101 帧
序列的在线中位节省量为 `6.8575 / 6.8488 / 6.9506 s`，因此每档几何在当前
口径下都能在第一条 trajectory batch 内摊销。

离线 cold peak RSS 不参与摊销，也没有被包装成在线内存优势。

## 独立 validator 做了什么

它没有导入正式 controller 或 worker，而是：

1. 重新打开五条公开 PoolFire pair truth；
2. 独立重建 1,515 帧 observation；
3. 手工解析并核验三份只读 factor artifact；
4. 重放 30 个 reference、3,030 帧；
5. 核对 360 个 worker 的 schema、输出 digest、调用账、读取边界和时间区间；
6. 重组 165 个随机相邻完整区组；
7. 独立重算 global、per-trajectory、per-geometry 和 per-stratum 分位数；
8. 再次核验旧 v74.0 invalid predecessor、v73 parent、源码与全部输入未漂移。

结果：

```text
canonical reference digest matches       30 / 30
actual call-ledger matches                30 / 30
reference norm maximum difference         0
all worker intervals serial/nonoverlap    true
minimum RSS samples per worker            378
maximum reported sample gap               0.08850 s
formal payload unchanged                  true
source / pair / observation unchanged     true
```

## 成功了什么

v74.1 首次把这条 PoolFire C 路线的三层证据接通：

1. **精度层**：v73 的 1,515/1,515 完整兼容性成立；
2. **调用层**：精确 `A/A^T` 各减半；
3. **资源层**：在完整 fresh worker 中 wall 稳定下降，三类 RSS 的冻结高尾门
   同时通过。

这比单条开发序列、理论调用减少或只看平均时间强得多。它说明“把固定几何的
低秩 detector factor 离线编译并在线加载”确实解决了此前
“算法快但每个 worker 重新构造 factor 导致内存失败”的工程物理瓶颈。

## 尚未成功什么

本轮仍不能声称：

- 这是神经算子或 learned operator 的成功；
- 已经跨独立反应流数据族泛化；
- 已经处理 curved ray、折射非线性、标定误差或观测噪声；
- 已经在实验室真实 BOS 位移图上通过；
- 已经证明全流水线瞬时 peak RSS 的数学精确值。

RSS monitor 保存的是每个 worker 的 coverage summary 与峰值回执，没有保存 raw
sampling trace。独立 validator 能核验 sealed summary、时间覆盖和下游比值，
不能重新生成操作系统采样轨迹。因此：

```text
raw_rss_samples_persisted=false
rss_sampling_trace_independently_recomputed=false
whole_pipeline_peak_memory_exactly_proven=false
external_family_transfer=false
real_bost_result=false
operator_learning_result=false
algorithm_breakthrough=false
paper_success=false
```

## 下一条唯一有效门

下一步不是回头扩大 PoolFire 模型，也不是马上写“突破”。应当在结果访问前冻结
一个独立公开反应流族外门：

- 数据族不能参与当前 factor、阈值或候选选择；
- 使用相同三档九视角 geometry 与同一 observation 生成合同；
- 先过完整 field / gradient / observation matched-accuracy；
- 再按同样 `2A+2A^T` 对 `4A+4A^T` 比较 fresh wall 与三层 RSS；
- 任一 accuracy、trajectory tail、wall 或 RSS 门失败，都记录外部迁移负结果；
- 外部门通过后，才向何远哲师兄申请真实位移图、相机标定、重复测量噪声与认可
  基线，进入真实 BOST 迁移。

这条顺序能检验当前正结果究竟是一个可迁移机制，还是只对 PoolFire fit family
成立的固定几何优化。
