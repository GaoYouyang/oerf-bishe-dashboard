# PoolFire C v10.7 rank96 跨轨迹可达性判决

## 一句话结论

`selected-rank96 coefficient distillation` 在训练模型前就被证伪了。

五个完整 fit trajectory 做 leave-one-trajectory-out：每一折只用另外四条轨迹选择
rank96 detector-DCT basis，再在留出轨迹上逐帧寻找对 K4 teacher 最有利的系数。最终
只有 `1/5` 轨迹通过冻结的 field / gradient / observation 兼容性门；其余四条的
joint matched 都是 `0%`、joint harm 都是 `100%`，总计出现 `90` 个 severe-harm
帧。

因此失败发生在“这个表示有没有合格答案”，还没有走到“网络能否预测答案”。按事前
规则，后续 20 份训练标签、ridge、MLP 和大模型全部停止。

```text
T1_training_target_generation_authorized=false
raw_rank96_coefficient_distillation_stopped=true
algorithm_breakthrough=false
```

## 为什么先做这个实验

v10.5 曾在已见 p14 上发现：使用全部五条 fit trajectory 选出的 selected-rank96
basis，teacher-oracle 可以通过。这只能证明一个特定开发条件下“房间里有答案”，
不能证明从其他工况选出的同秩房间能容纳新的完整 trajectory。

v10.7 把两个问题按顺序拆开：

1. **T0 可达性：**fold-train 选出的 rank96 basis，在 held-out fit trajectory 上
   是否至少存在兼容的 K1 warm start？
2. **T1 可预测性：**只有 T0 至少 `4/5` 通过且 severe harm 为 0，才生成训练标签并
   比较 ridge、线性模型和小 MLP。

T0 没过，所以 T1 没有运行。这避免了用更大网络拟合一个本身就不具跨轨迹可达性的
目标。

## 执行与独立复核

- 五个 fold 各生成 `101` 帧的四候选 target：三个固定起点优化结果加一个未修改
  K3 projection fallback；
- 每帧独立选最低 teacher-deficiency `q`，没有跨帧 top-k、时间耦合或随机帧切分；
- 五个 target 在真值读取前均通过独立逐数组复算，真值读取状态均为 `false`；
- 五个 target 全部封存后才一次性读取 proxy truth；
- 正式 scorer 之外，另写逐帧四步 CGLS 和独立 field / gradient / observation
  指标公式复算，所有公开判决量的最大绝对差为 `0.0`。

独立复核没有复用正式 compatibility helper。第一次尝试因底层算子拒绝批量输入而在
指标计算前 fail closed；改为逐帧求解后得到与正式结果完全一致的 `1/5`。

## 五条轨迹结果

| held-out trajectory | joint matched | joint harm | field harm | observation harm | severe | 判决 |
|---|---:|---:|---:|---:|---:|---|
| p=14kw_size=05 | 0% | 100% | 50.50% | 100% | 24 | FAIL |
| p=22kw_size=03 | 0% | 100% | 87.13% | 100% | 21 | FAIL |
| p=33kw_size=01 | **100%** | **0%** | **0%** | **0%** | **0** | **PASS** |
| p=45kw_size=05 | 0% | 100% | 15.84% | 100% | 0 | FAIL |
| p=58kw_size=03 | 0% | 100% | 86.14% | 100% | 45 | FAIL |

gradient harm 在五条轨迹上都是 0，但这不能抵消 observation 与 field 的系统性伤害。
尤其 observation harm 在四条失败轨迹上都是 100%，说明候选虽然在 rank96 空间里
尽力逼近 K4 teacher，仍无法保持冻结的数据一致性前沿。

## 失败到底来自哪里

结果后做了一个不改变正式判决的覆盖诊断：

- 五个 fold 的 rank96 basis 两两 Jaccard 中位数为 `0.8286`；
- 五折 basis 的并集只有 `115` 个 atom；
- 所以失败不是五折选出了完全互不相干的频率；
- 唯一通过的 p33，其 held-out K3 correction 的 rank96 能量捕获 p50 为
  `85.47%`；
- 四条失败轨迹的对应 p50 只有 `68.28%–78.86%`。

把同一 mean-energy 选择规则探索性扩到 rank384，五条 held-out 的 p50 捕获升到
`88.22%–96.95%`。这只是“更宽表示可能有容量”的信号，不是 rank384 oracle、
可预测性或兼容性成功，也不能据此声称应该训练 384 输出的大网络。

## 对算法路线的真实影响

### 已经停止

- raw selected-rank96 oracle coefficient label generation；
- 96 维 ridge / 全线性 / MLP 蒸馏；
- 用更大 MLP 挽救相同表示；
- 把 p14 单点 oracle headroom 写成跨工况表示成功。

### 新的研究对象

下一候选改为 **coverage-adaptive / full-view dual proposal**：

1. 模型读取完整三视图 detector observation，而不是只保留 96 个固定频率；
2. 输出 detector-space dual correction 或 observation-conditioned basis
   coefficient；
3. 仍用精确 `A^T` 把它提升到 `Range(A^T)`；
4. 仍用 observable-only `alpha`；
5. 仍只运行未修改 strict CGLS K1；
6. 训练目标直接看 K1 后的 teacher / proxy 物理误差，避免把任意一个可能不唯一的
   raw coefficient 当成唯一真值。

这个转向针对本轮真实故障：固定 rank96 的跨轨迹覆盖不足。它不是“换一个更大网络”，
而是取消被数据证伪的固定瓶颈，同时保留可纠正性和完整成本账。

## 与公开工作的边界

- [Learning to Warm-Start Fixed-Point Optimization Algorithms](https://www.jmlr.org/papers/v25/23-1174.html)
  已经说明“网络输出初值，再通过固定迭代器训练”不是新概念；
- [Rethinking Warm-Starts with Predictions](https://proceedings.mlr.press/v202/sakaue23a.html)
  指出多解时应关注到最优解集合的距离，而不是拟合任意一个坐标标签；
- [NeRIF](https://arxiv.org/abs/2409.14722) 已经把神经场用于 BOST 三维重建。

因此潜在贡献只能来自完整组合：BOST 多视图 detector dual proposal、精确
`Range(A^T)` lift、短程 CGLS、轨迹级非劣与真实部署成本，而不能把 warm start、
神经网络或 BOST 单独写成创新。

## 仍未成立

- 可部署 learned initializer；
- fresh trajectory 泛化；
- A/A^T、wall time 或 whole-pipeline RSS 优势；
- 真实 BOST、曲线射线、噪声和标定误差闭环；
- SOTA、全球唯一、论文成功或算法突破。

机器可读脱敏结果见
[`poolfire_c_dual_coefficient_attainability_v10_7_public_summary.json`](poolfire_c_dual_coefficient_attainability_v10_7_public_summary.json)。
