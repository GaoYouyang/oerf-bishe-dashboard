# PoolFire C v10.5 正式 p14 机制判决

## 一句话结论

fit-selected detector-DCT 子空间并没有失败：teacher-oracle 在 rank
`96 / 216 / 384` 全部通过冻结兼容性门，最小通过 rank 只有 `96`；但是把 K3
dual correction 直接投影进该子空间的候选全部失败。同秩 uniform oracle 也全部
失败。

因此当前真正的问题已经从“低秩表示装不下答案”转成：

> 能否只根据 observation，预测 selected basis 中那组能通过兼容性门的系数？

这仍不是可部署预测器、算法加速或论文成功。

## 如何保证这次判决没有边跑边改

正式过程分成两段：

1. Stage A 只拿到 p14 observation、冻结 basis、几何和代码身份，生成
   `selected / uniform × rank 96 / 216 / 384 × projection / oracle` 共 12 臂；
2. 全部候选封存后，一次性评分令牌先被消费，独立 validator 才加载 p14 truth，
   重算 proposal、精确 `A^T`、observable alpha、strict CGLS K1 和全部指标。

full-rank identity、完整 K3 restart reference、输入角色、单次令牌和独立复算五类
执行门全部通过，hard gate failure 为 0。这里证明的是本地制品链的可信度，不是操作
系统级隔离。

## 12 臂结果

冻结兼容性门要求 joint matched fraction 至少 90%、joint harm 不超过 5%，且不能
出现 severe harm。表中 objective 是相对 K4 teacher 的完整轨迹目标，越低越好。

| 轴 | basis | rank | joint matched | joint harm | severe | 判决 |
|---|---|---:|---:|---:|---:|---|
| projection | selected | 96 | 0.00% | 100.00% | 101 | FAIL |
| projection | selected | 216 | 0.00% | 100.00% | 101 | FAIL |
| projection | selected | 384 | 0.00% | 100.00% | 101 | FAIL |
| projection | uniform | 96 | 0.00% | 100.00% | 56 | FAIL |
| projection | uniform | 216 | 0.00% | 100.00% | 101 | FAIL |
| projection | uniform | 384 | 0.00% | 100.00% | 101 | FAIL |
| teacher-oracle | selected | 96 | **93.07%** | **0.00%** | **0** | **PASS** |
| teacher-oracle | selected | 216 | **100.00%** | **0.00%** | **0** | **PASS** |
| teacher-oracle | selected | 384 | **100.00%** | **0.00%** | **0** | **PASS** |
| teacher-oracle | uniform | 96 | 0.00% | 100.00% | 0 | FAIL |
| teacher-oracle | uniform | 216 | 0.00% | 57.43% | 0 | FAIL |
| teacher-oracle | uniform | 384 | 8.91% | 0.00% | 0 | FAIL |

selected oracle 的 teacher objective 随 rank 下降：

```text
rank 96  -> 0.369901
rank 216 -> 0.277132
rank 384 -> 0.180834
```

但 rank 96 已经通过 truth-side compatibility，所以“继续增加 rank”不是当前最便宜
的下一步。更高价值的问题是 rank 96 系数能否从 observation 预测出来。

## 这次发现了什么，没发现什么

### 已经成立

- 当前 fit-selected basis 在已见 p14 上具有明确的 teacher-side oracle headroom；
- 这个 headroom 不是 uniform 低频 basis 自动带来的；
- 简单投影并不能把 observation 送到 oracle 所在位置；
- v10.3 的失败不能再简单归因于“所有 rank-96 detector 表示都装不下”。

### 仍未成立

- oracle 系数可由 observation 稳定预测；
- rank-96 模型能在未见完整 trajectory 上通过；
- 最终精度等价时真的减少 `A/A^T`；
- wall time 或整流程峰值内存下降；
- 结果能迁移到真实 BOST、曲线射线、噪声和标定误差；
- 算法突破、SOTA、全球唯一或论文成功。

## 下一步只做一个问题

下一门是 **selected-rank96 oracle-coefficient distillation**：

1. 只用五条 fit 轨迹生成 rank96 oracle coefficient targets；
2. trajectory-level 留一，任何 normalization、basis、模型选择和 early stopping 都只
   来自 fold-train；
3. 先比较 ridge、全线性、时间持续性和旧同规模 MLP，再只训练一个最小模型；
4. held-out fit 必须同时报告 coefficient error 和经过同一
   `A^T -> alpha -> strict K1` 后的 field / gradient / observation 尾部与 harm；
5. fit-only nested gate 过关后，才允许封存一个 predictor 做一次 post-open p14
   筛选；仍不打开 fresh 和 test。

机器可读的脱敏 12 臂数据见
[`poolfire_c_dual_spectral_v10_5_p14_public_summary.json`](poolfire_c_dual_spectral_v10_5_p14_public_summary.json)。

```text
algorithm_breakthrough=false
learned_predictor_success=false
fresh_trajectory_generalization=false
real_bost_validation=false
```
