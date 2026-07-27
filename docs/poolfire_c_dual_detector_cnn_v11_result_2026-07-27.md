# PoolFire C v11.2：非线性 Dual Warm Start 的严格 fit-only 五折结果

## 一句话结论

这是一次**重大阶段性进展，但还不是算法突破**。

在五条已经进入开发池的完整 PoolFire trajectory 上，77,020 参数的奇对称多视角
detector CNN 做到了严格 leave-one-trajectory-out（LOTO）`5/5` 通过。每折只用
另外四条 trajectory 训练，第五条的 101 帧全部留出；候选在 field、gradient 和
observation 三类指标上的 joint matched fraction 均为 `100%`，harm fraction 均为
`0%`，severe-harm frame 均为 `0`。

候选每帧实测调用账为 `2A + 2A^T`，参照 Zero-CGLS K4 为 `4A + 4A^T`。因此当前
证据支持：

> 在这五条 fit-only 公开代理轨迹及冻结兼容性门下，结构化非线性 dual proposal
> 可以用一半的完整算子调用，达到 Zero-CGLS K4 的兼容精度包络。

它**尚未**证明 fresh trajectory 泛化、墙钟加速、峰值内存优势、真实 BOST 有效性、
SOTA 或论文成功。因此正式状态仍是：

```text
algorithm_breakthrough=false
fresh_opened=false
stopping_validation_opened=false
test_opened=false
real_bost=false
wall_time_speedup=false
whole_pipeline_rss_speedup=false
```

## 实际做了什么

模型输入是冻结三视角 observation，打包成六个中心化的 `14x30` detector 分量通道。
网络使用 width-32 输入卷积、四个 dilation 为 `1/2/4/1` 的残差块、全局通道上下文、
输出卷积和六个可学习 skip gain，共 77,020 个参数。它不读取 trajectory 标签，也不
把物理算子藏进网络。

模型只生成 observation-space dual proposal：

```text
three-view observation
  -> odd multiview detector CNN
  -> dual proposal z_theta
  -> exact A^T lift
  -> observable-only alpha
  -> unchanged CGLS K1
```

选择这个结构不是因为“CNN 流行”，而是此前两条可复现证据共同指向它：

1. v10.8.2 证明完整 K3 dual target 在五条 fit trajectory 上 `5/5` 可行。
2. v10.9 的 full-linear KRR 即使允许 outer oracle 选择正则仍是 `0/5`。
3. K3 dual 映射满足奇对称和尺度齐次，但对输入通常不可加；40 对 observation 的
   non-additivity p50/p90/worst 为 `4.15% / 6.77% / 9.33%`。

所以 v11 使用奇对称、尺度齐次的局部多尺度非线性，并把 loss 直接放在 exact
`A^T -> alpha -> K1` 后的物理缺口上，而不是只拟合一个可能不安全的 certificate
MSE。每折固定训练 120 epoch，不做 held-out checkpoint selection。

## 五条完整轨迹结果

下表中“候选 p90”和“K4 p90”的顺序均为
`field / gradient / observation`。匹配不是要求逐项严格更小，而是按预先冻结的
单侧 compatibility envelope 判断；任何材料性 harm 都会让该轨迹失败。

| 留出轨迹 | 候选 p90 | Zero-K4 p90 | joint match | harm | severe |
|---|---:|---:|---:|---:|---:|
| P14-S05 | 0.568922 / 0.985755 / 0.288382 | 0.569342 / 0.988922 / 0.288154 | 100% | 0% | 0 |
| P22-S03 | 0.500261 / 0.961537 / 0.262536 | 0.499868 / 0.969621 / 0.259811 | 100% | 0% | 0 |
| P33-S01 | 0.677289 / 1.355534 / 0.365899 | 0.683349 / 1.382053 / 0.365459 | 100% | 0% | 0 |
| P45-S05 | 0.702366 / 1.018988 / 0.308148 | 0.698407 / 1.033856 / 0.301599 | 100% | 0% | 0 |
| P58-S03 | 0.593840 / 0.950112 / 0.279869 | 0.590762 / 0.948369 / 0.279135 | 100% | 0% | 0 |

这张表不能解释成“CNN 每个指标都更准”。例如 P22 的 field/observation 和 P45 的
field/observation 略差于 K4，但都位于事前冻结的兼容包络内，且没有触发任何 harm。
当前真正成立的是**匹配精度下的算子调用减少候选**，不是普遍精度提升。

## 为什么这次结果可以信

v11.0 曾得到同样的 `5/5` 数值，但证据边界不够严，不能当正式结果。v11.1 在独立
红队发现问题后只生成三个 checkpoint 就中止，没有评分。v11.2 修复后重新完整运行：

- 几何只由冻结坐标构建，不借读 `rho` 或时间数组。
- 正式输入以同一文件描述符读入、摘要和解析，避免“校验一份、计算另一份”。
- 五个 checkpoint 全部先完成并冻结，之后才一次性授权读取留出 truth。
- 每帧 `A/A^T` 调用由运行时账本记录，505 帧合计
  `1010 A + 1010 A^T`。
- 独立 validator 不复用正式评分 helper，重新做 NumPy 推理、物理链和全部指标。
- 独立复算的最大科学数值差为 `2.220446049250313e-16`。
- 正式运行前独立红队结论为 `P0=0 / P1=0`。

因此，这不是一次训练日志里的偶然好看数字，而是一项已通过独立重放的 fit-only
候选结果。

## 科学判断发生了什么变化

此前最强判断是“完整 dual target 存在，但固定低秩和冻结线性映射跨工况失败”。现在
可以把它推进为：

> 在当前 PoolFire straight-ray 代理中，保留 detector 局部结构、奇对称和
> 下游物理约束的非线性映射，确实跨越了已测试 full-linear KRR 的失败门，并在五条
> 完整 fit trajectory 的严格 LOTO 上达到 matched-accuracy。

这说明“非线性结构化 dual proposal”值得继续，而不是继续扩大无结构线性层或盲目换
FNO/UNO/DeepONet。但它仍只是**开发池内跨轨迹证据**：架构、损失与研究方向都曾受
这五条数据启发，不能把 LOTO 改写成真正未见数据泛化。

## 下一道不可跳过的结论门

下一次科学判断必须来自一个在数据打开前冻结的 full-fit checkpoint 和 fresh release，
然后只在预先选定的一条项目管理 fresh trajectory 上执行一次。需要同时报告：

1. 与 Zero-K4 的 field/gradient/observation matched-accuracy 与逐帧 harm。
2. 实际完整 `A/A^T` 调用。
3. fresh-process 端到端 wall time。
4. whole-pipeline peak RSS。

只有 fresh 仍通过、调用优势保留且 wall/RSS 不被模型推理抵消，才可把
`algorithm_breakthrough` 从 `false` 重新送审。之后还必须用组内真实 BOST 数据闭合
折射率、相机标定、噪声和曲光线误差，才具备论文主张资格。
