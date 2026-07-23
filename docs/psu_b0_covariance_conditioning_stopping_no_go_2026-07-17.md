# Covariance-conditioned PCGLS：重调有大收益，但 pooled 早停规则严格 NO-GO

日期：2026-07-17
状态：`OBSERVABLE_POOLED_STOPPING_RULE_NO_GO`

## 一句话结论

在真实 PSU detector geometry、解析反应场和合成 graph-correlated noise 上，
重新调 Sobolev 预条件器能显著降低三维重建误差；graph covariance 在强
`component s3-k4` 基线上仍有约 1.4% 的平均增量。但稀有场最坏回退达到
7.92%，仅凭六个 pooled PCGLS 轨迹标量设计的早停/回退规则不能迁移，fresh
实验因此继续封存。

## 1. 为什么最初的 25% 不能写成 covariance 创新

冻结网格包含：

- 5 个 detector 空间 covariance tempering 指数；
- 5 个 Sobolev strengths；
- 2/3/4/5 个 PCGLS stages；
- full fitted graph covariance anchor；
- 16 个已经打开的 replicates，前 8 个选择、后 8 个诊断。

原始规则选择 `full_graph_s3_k4`。相对旧 `component_s5_k4`：

| 分区 | mean | p10 | >1% harm | front-F1 gain |
|---|---:|---:|---:|---:|
| opened selection | +25.33% | +9.72% | 1.56% | +0.2562 |
| opened diagnostic | +24.27% | +9.38% | 0% | +0.2436 |

但 `component_s3_k4` 自身已经得到约 24% 的收益。也就是说，大部分漂亮数字
来自经典预条件器重调，不是 graph covariance。若把它归功于新方法，会形成
弱基线偏差。

## 2. 公平基线后的真实 covariance 信号

将 `full_graph_s3_k4` 与同 strength、同 stage 的 `component_s3_k4` 配对：

- pooled mean：`+1.406%`；
- replicate-clustered 95% CI：`[+1.235%, +1.578%]`；
- pooled p10：`+0.166%`；
- `>1%` harm：`2.34%`；
- worst field：`-7.920%`。

均值和 p10 都说明 covariance 不是完全无效，但 annular、oblique shock 和少数
multi-plume 场出现强烈的 stage-dependent semiconvergence。正确 covariance
改变了 data metric 和 Krylov trajectory；它不能与原先固定的停止深度独立
调参。

## 3. 轨迹复用与调用账本

120 个候选原本需要 6,784 对 forward/adjoint 调用。固定 covariance 与
Sobolev strength 后，2/3/4/5 stages 共享同一条 PCGLS 前缀轨迹：

- 逻辑候选调用：6,784 对；
- 实际物理调用：2,464 对；
- 减少：63.68%；
- float64 单元测试逐值验证共享 checkpoint 与独立求解一致；
- 全仓 745 项测试通过后才运行网格。

这只是执行优化，不改变每个候选的逻辑预算，也不构成算法收益。

## 4. 可观测早停规则为什么失败

每个场提取 stage 2–5 的六个部署可见量：

1. whitened residual objective；
2. 上一 stage 到当前 stage 的 residual reduction；
3. PCGLS alpha；
4. previous beta；
5. relative volume update；
6. gradient-to-field norm。

真值只用于给规则打分，不进入规则输入。正式审计包含 348 条规则：

- one-threshold stage-4/5 rules：108 条，选择门通过 0 条；
- rollback/continue rules：240 条，选择门通过 5 条。

选中的规则为：

```text
if beta4 >= 0.675:
    if relative_update4 >= 0.325: use stage 3
    else: use stage 5
else:
    if residual_reduction4 <= 0.03: use stage 5
    else: use stage 4
```

| 指标 | opened selection | opened diagnostic check |
|---|---:|---:|
| mean field gain | +3.765% | +3.340% |
| p10 | +0.178% | -1.746% |
| >1% harm | 3.125% | 12.5% |
| worst field | -1.775% | -17.532% |
| mean stage | 4.234 | 4.266 |

后半区同时失败 p10、harm、worst 和平均调用门。不能因为 mean 仍为正就宣布
成功，也不能继续放宽阈值。

## 5. 对算子学习方向的真实含义

这个 NO-GO 不是“神经网络没用”，而是说明当前输入表示不够：

- pooled 标量看不到是哪一台相机、哪个 detector 邻域产生冲突；
- 同样的 residual reduction 可对应 front 恢复，也可对应噪声拟合；
- morphology-dependent semiconvergence 不能由一个全局阈值可靠识别；
- 增加 MLP 宽度只会更容易拟合 64 个选择场。

下一候选只能沿两条更有物理约束的支线推进：

### A. 逐相机结构保留的 controller

输入每台相机的 graph residual map、局部 Jacobian/front energy、camera pose 和
covariance spectrum，以 permutation-equivariant set encoder 输出有限动作：
`component fallback / graph stage 4 / graph stage 5`。必须先做
leave-one-replicate、leave-one-morphology 和 held-out camera，不得直接上大模型。

### B. front-protecting deterministic baseline

先实现 TV/Huber-superiorized PCGLS 或 edge-preserving proximal correction，
检查 annular/shock 尾部能否在不依赖 selector 的情况下收缩。只有它成为强
基线后，才让 neural operator 学习 bounded proximal map 或参数，而不是直接
生成三维场。

当前优先级更推荐 B：它更容易证伪，也能防止神经模型把一个经典正则化收益
包装成创新。

## 6. 仍需向何远哲确认

1. 每台相机同一固定条件是否有至少 50 张未经平均、带时间顺序的 flow-off？
2. 是否能提供一套 annular/shock/front 失败 case，而不只给平均重建结果？
3. 组内更在意 field-L2、front 位置/宽度、held-out displacement 还是 PIV 补偿？
4. NeRIF/TDBOST 当前 loss 是否能接收逐相机 covariance-weighted residual？
5. 是否允许封存一条 camera 或一个 session，任何模型选择都不能看？
6. 真实 forward 是否已有可调用的 `F` 和 `F^T/J^T`，以及其有限孔径假设？

## 7. 证据边界

- 真实：PSU detector/ray geometry；
- 合成：反应场、flow-on noise、covariance truth；
- 已见：所有 16 replicates；
- 未使用：真实 temporal flow-off、实验三维真值、OERF 私有数据；
- 未做：NeRIF/FNO/DeepONet/FFNO 公平比较；
- 不宣称：算法优越、实验有效或可投稿结论；
- fresh：未打开，且本轮 NO-GO 后不授权打开。

## 8. 可复现入口

- 冻结网格：
  `demo_t16_operator/configs/psu_b0_covariance_conditioned_pcgls_diagnosis_v1.json`
- 网格报告：
  `demo_t16_operator/results/psu_b0_covariance_conditioned_pcgls_diagnosis/report.json`
- 独立验证：
  `demo_t16_operator/results/psu_b0_covariance_conditioned_pcgls_diagnosis/validation_report.json`
- 轨迹配置：
  `demo_t16_operator/configs/psu_b0_covariance_stopping_feature_audit_v1.json`
- 规则配置：
  `demo_t16_operator/configs/psu_b0_covariance_stopping_rule_audit_v1.json`
- 规则报告：
  `demo_t16_operator/results/psu_b0_covariance_stopping_rule_audit/report.json`
- 四联图：
  `demo_t16_operator/results/psu_b0_covariance_stopping_rule_audit/psu_b0_covariance_conditioning_audit.png`
