# v10.4.2：完整 K3 重启能过，有限 B96 表示仍过不了

状态：`PASS_INDEPENDENT_RECOMPUTATION_V10_4_2`

科学判决：

`POSTOPEN_DIAGNOSTIC_B96_SEARCH_NO_HEADROOM_FOUND_NOT_IMPOSSIBILITY_PROOF`

这轮只使用已经打开的 `p14-s01` 101 帧做开发机制诊断。它不训练可部署模型，
不打开 fresh、stopping validation 或 test，也不证明真实 BOST、泛化、速度或论文成功。

## 一句话结论

问题不在于 `2A+2A^T` 的 K3 dual restart 外壳完全没有能力：保留完整 K3 dual
certificate 再做一次 CGLS，确实通过了冻结的三指标 compatibility 门。

但把 K3 所需的 dual correction 压进逐通道 `4x4` 低频 DCT、总计 96 个系数后，
无论保留 50% 半径约束还是去掉约束，三次冻结初始化、每次 800 步的有限搜索都没有
找到通过门的候选。它只说明当前 B96 搜索没有 headroom，不是数学不可能证明。

## 三个最重要的数字

| 方法 | field p90 | gradient p90 | observation p90 | joint matched | joint harm |
|---|---:|---:|---:|---:|---:|
| Zero-CGLS K4 | 0.6332 | 1.2180 | 0.3330 | 100% | 0% |
| 完整 K3 dual restart + K1 | 0.6205 | 1.0923 | 0.3466 | 100% | 0% |
| best capped B96 + K1 | 0.6663 | 0.8725 | 0.4601 | 0% | 100% |
| best uncapped B96 + K1 | 0.6355 | 0.9072 | 0.4245 | 0% | 100% |

完整 K3 重启牺牲了少量 observation p90，但仍处在冻结容差内，同时 field 和
gradient 更好。两个 B96 最优候选的 gradient 很低，却都让每一帧的 observation
指标受损，说明只盯某一个物理量会制造假胜利。

## 为什么先查表示，而不是继续加宽网络

K3 dual correction 在逐通道均匀低频 DCT 中的能量保留率是：

| 每通道频率块 | 总 rank | minimum | p10 | p50 |
|---|---:|---:|---:|---:|
| `4x4` | 96 | 37.11% | 38.52% | 42.12% |
| `6x6` | 216 | 68.66% | 69.83% | 72.62% |
| `8x8` | 384 | 78.99% | 80.01% | 82.36% |
| `12x12` | 864 | 86.20% | 87.94% | 89.26% |
| `14x14` 近完整 | 约 2072 | 99.02% | 99.18% | 99.37% |

这里专门纠正一个旧字段：v10.4.2 原始机器结果沿用了“误差越低越好”的通用 summary，
把 capture 的最大值写在名为 `worst` 的字段里。capture 越高越好，所以真正的尾部
应看 minimum 和低侧 p10。上表已经从封存数组重新计算；这个修正不改变 B96 失败的
科学判决，但避免把表示覆盖说得比实际更好。

而六个 view-channel 的 p50 能量份额约为：

```text
8.35%, 0.64%, 8.95%, 0.65%, 42.04%, 39.29%
```

也就是说，均匀给每个通道 16 个系数很可能浪费大量容量，同时漏掉主要通道的中高频
结构。下一门因此只比较：

1. fit-only 自适应频率与通道联合分配；
2. 严格同秩的逐通道 `4x4 / 6x6 / 8x8` 控制；
3. rank `96 / 216 / 384` 的 projection support 与 teacher-oracle headroom；
4. 完整 rank 2072 对完整 K3 restart 的数值恒等控制。

这仍不能把差异归因于“通道分配”一个因素，因为自适应方案同时改变了频率位置和通道
计数。

## 独立验证做了什么

- 三个 capped 和三个 uncapped 搜索全部有限完成；
- 独立 NumPy 中央差分方向导数检查全部通过，误差量级约 `1e-11`；
- validator 重算全部候选、指标、cap、目标与判决；
- 输入封印在运行前后未改变；
- 最终状态只允许写“有限 B96 搜索未找到 headroom”。

## 下一门的防偷看顺序

1. 五条 fit 轨迹只用 observation、K3/K4 teacher 做留一轨迹选基与 teacher
   deficiency，不读真值；
2. 用全部五条 fit 重算唯一最终 basis 并封存；
3. 在 p14 不读 truth 的阶段生成并封存全部 projection/oracle 候选；
4. 独立验证 candidate seal 后，使用一次性 score token 才读取 p14 truth；
5. projection pass 与 oracle-only pass 分开解释；
6. 若 `96/216/384` 全失败而完整 rank 控制仍通过，停止 DCT decoder 训练，再冻结
   一个有成本上限的新表示或 Krylov-state oracle。

当前始终保持：

```text
deployable_model=false
outer_generalization=false
real_BOST=false
wall_time_speedup=false
algorithm_breakthrough=false
paper_result=false
```

脱敏聚合数据：
`docs/poolfire_c_dual_representation_ceiling_v10_4_2_public_summary.json`。
