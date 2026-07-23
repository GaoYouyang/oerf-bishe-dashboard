# RQ-OGSE-PCGLS 开发判决：场误差主门出现正信号，反应前沿安全仍未闭合

**更新日期：** 2026-07-17
**证据等级：** post-open synthetic development diagnostic
**最终判决：** `HOLD；不生成独立 repeat，不宣称算法成功`

## 一句话结论

把 OGSE 的全专家 softmax 混合改成
`baseline -> single expert` 的固定正定路径后，mean-only 路由相对
static Sobolev-PCGLS-4 的 field relative L2 在 validation / calibration
分别改善：

- `+3.321%`，95% bootstrap CI `[+1.732%, +5.071%]`；
- `+2.907%`，95% bootstrap CI `[+1.259%, +4.858%]`。

预先写入配置的八项 field/risk 主门全部通过，但 calibration 仍有一个场
恶化 `-1.897%`，而 front top-10% F1 的均值为 `-0.261%`、最坏下降
`-30.876%`。因此这只能说明**单专家有限动作空间比旧 softmax 混合更有
效用**，不能说明反应流结构重建已经更可靠。

加入 field quantile、field harm 和 front-risk 双头后，已知 field harm
可以清零，但严格多目标版本只剩 `+1.192% / +1.382%` field gain，且
calibration front 均值仍为 `-0.060%`。它没有通过 field `2%` 主门，也没有
通过 front 安全审计。

## 1. 这一轮真正实现了什么

### 1.1 有限动作缓存

四个 train-only 固定 SPD 专家不再任意 softmax 混合。动作被离散为：

```text
exact baseline
baseline -> expert 1 at tau = 0.25 / 0.50 / 0.75 / 1.00
baseline -> expert 2 at tau = 0.25 / 0.50 / 0.75 / 1.00
baseline -> expert 3 at tau = 0.25 / 0.50 / 0.75 / 1.00
```

每个动作只重建一次 `risk_train`。648 个 RQ 路由配置和 576 个
field+front 路由配置随后复用逐样本动作误差查表，不再重复跑三维求解。

这把开发成本从：

```text
route candidates x reconstruction
```

降为：

```text
finite actions x reconstruction + route candidates x table lookup
```

最新公开摘要记录 RQ 全流程 10.96 秒、多目标补充实验 2.98 秒；两者
峰值 RSS 均低于 450 MB。运行时间会随缓存与系统负载浮动。

### 1.2 固定正定单专家路径

对基线 multiplier `M0` 和候选专家 `Me`，使用：

\[
\log M_\tau=(1-\tau)\log M_0+\tau\log M_e,
\]

随后减去频谱均值，使几何均值保持为 1。于是：

- `M_tau > 0`；
- 拒绝时逐值等于 `M0`；
- 一个 solve 内 multiplier 固定；
- 仍保持每个样本 `4F + 4AT`；
- 没有用额外 forward/adjoint 偷预算。

### 1.3 三类风险目标

现有 44 个共享首伴随场特征分别拟合：

```text
mean field gain
lower quantile of field gain
P(field gain < -1%)
lower quantile of absolute front-F1 delta
P(front-F1 absolute delta < -0.02)
```

线性分位头使用带 L1 slope penalty 的 pinball-loss 线性规划；harm 头使用
logistic ridge。所有超参数只在 family-stratified `risk_train` OOF 上筛选。

## 2. 有限动作本身告诉了什么

任何一个非基线专家全局应用都不安全。以相对温和的 expert 1 为例：

| 插值 `tau` | train mean field gain | `>1%` field harm |
|---:|---:|---:|
| 0.25 | `+1.164%` | `12.5%` |
| 0.50 | `+1.234%` | `31.3%` |
| 0.75 | `≈0%` | `50.0%` |
| 1.00 | `-2.011%` | `60.4%` |

另外两个专家的全局平均更差。真正的空间不在“找到一个更好的全局
preconditioner”，而在**准确识别少数适用样本并限制动作幅度**。

## 3. RQ 四路消融

所有路线都由 `risk_train` OOF 单独选择，再原样迁移到 post-open
validation/calibration。

| Selector | Validation field gain | Calibration field gain | Field harm V/C |
|---|---:|---:|---:|
| mean-only | `+3.321%` | `+2.907%` | `0% / 3.33%` |
| quantile-only | `+1.933%` | `+1.946%` | `4.17% / 0%` |
| quantile + harm | `+1.933%` | `+1.777%` | `4.17% / 0%` |
| mean + quantile + harm | `+1.979%` | `+1.777%` | `0% / 0%` |

这给出两个机制结论：

1. quantile/harm 头不是“没有工作”：联合路线确实把已知 field harm 清零；
2. 现有特征下，安全性通过降低 coverage 和拒绝高收益样本换来，因此两层
   field mean 都低于 `2%`。

mean-only 的 field 主门通过，但不能单独作为 GO，因为它牺牲了前沿质量：

| Split | Gradient mean gain | Front-F1 mean gain | Front-F1 p10 | Worst front |
|---|---:|---:|---:|---:|
| validation | `+0.537%` | `+1.177%` | `-3.219%` | `-27.404%` |
| calibration | `+0.384%` | `-0.261%` | `-3.515%` | `-30.876%` |

最坏 front 场是 correlated-noise oblique shock。field-L2 改善并不等价于
shock/front 定位更好。

## 4. 多目标 front-risk 路由为什么仍是 NO-GO

严格多目标 OOF 路由选择 `tau=0.25`，train coverage `33.3%`，train field
harm 和 front harm 都为 0。迁移后：

| Split | Field gain | Bootstrap 95% CI | Field harm | Front mean |
|---|---:|---:|---:|---:|
| validation | `+1.192%` | `[+0.294%, +2.204%]` | `0%` | `+0.375%` |
| calibration | `+1.382%` | `[+0.493%, +2.478%]` | `0%` | `-0.060%` |

它验证了 front veto 可以降低激进行为，但没有形成足够的可迁移辨识力。
继续把阈值收紧只会趋向全回退；继续扩大线性头网格会增加 post-open
过拟合。当前应停止阈值搜索。

## 5. 这对下一算法意味着什么

现有 44 维特征来自**所有视角反投影之和**。求和会把以下信息抹掉：

- 某一相机是否与其他相机方向冲突；
- correlated-noise 是否只污染少数视角；
- shock/front 在不同视角的高频响应是否一致；
- 一个看似合理的全局 `g0` 是否由互相抵消的 per-view 分量组成。

下一候选应改为 `View-Decomposed Risk Router`：

```text
per-view whitened observation/residual
  -> per-view spectral summaries
  -> per-view adjoint contribution g0,v
  -> permutation-invariant camera-set encoder
  -> field utility head + front-preservation head
  -> baseline / one bounded expert action
```

优先新增：

1. 每视角白化 displacement 的径向频谱、方向性和高频比例；
2. `||g0,v|| / ||sum_v g0,v||`、两两 cosine、角向离散度；
3. camera-correlated residual 的最大特征值与有效秩；
4. 图像平面 ridge/front coherence，而不是只看三维总谱；
5. leave-one-family-out 与 leave-one-noise-profile-out 双层 OOF。

### 重要计算预算修正

“第一步 residual contraction”不能直接用于当前初始 preconditioner，因为
它只有跑完第一步后才可观察。若使用该特征，必须明确选择：

- 先用 baseline 跑一步，再 restart/FCG 选择后三步；
- 或增加额外 probe calls 并如实记账。

不能一边声称固定 `4F+4AT`，一边在第一步之前读取第一步之后才存在的量。

## 6. 文献边界

- [Regression Quantiles](https://doi.org/10.2307/1913643)：pinball loss
  估计条件分位，而非条件均值。
- [Conformalized Quantile Regression](https://proceedings.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html)：
  held-out residual 可校准预测区间，但保证依赖交换性。
- [Selective classification risk-coverage](https://jmlr.org/papers/v11/el-yaniv10a.html)：
  拒绝更多通常降低风险，因此必须同时报告 coverage。
- [Learn then Test](https://arxiv.org/abs/2110.01052)：可用 calibration
  对低维决策参数做风险控制，但当前 validation/calibration 已经 post-open，
  不能倒推为新的有限样本保证。

本轮分位头是 empirical OOF quantile regression，不是 conformal guarantee；
front 阈值是在看到第一版 front 结果后，把网页中已有的“不得实质恶化”
要求数值化，因此只作 post-selection safety audit。

## 7. 当前可对师兄说的话

> 我们已经把固定 SPD 专家动作、收益路由和 PCGLS 公平预算闭合。单专家
> mean-only 路线在 post-open validation/calibration 的 field-L2 为
> +3.32%/+2.91%，但 calibration 有一个 -1.90% field harm，shock-front
> 最坏下降约 31%。quantile/harm 与 front-risk 路由能清除 field harm，
> 但收益降到 2% 以下，说明当前全局首伴随特征不足以同时识别 utility 和
> front risk。下一步拟做 per-view backprojection/residual set encoder。

不能说：

- 已经优于 NeRIF、DeepONet、FNO、UNO-CG 或组内方法；
- 已经在 OERF 实验数据上验证；
- 已经拥有 conformal 风险保证；
- field-L2 主门通过就等于反应流结构重建成功。

## 8. 需要何远哲确认

1. 组内 forward/adjoint 能否返回每个相机的 adjoint contribution，而不是只
   返回视角求和？
2. 对当前实验，front/shock 位置、密度场 L2、held-out displacement、
   PIV 补偿速度误差，哪个才是主物理指标？
3. 是否有 flow-off repeats 可估计 per-camera covariance 和相关噪声？
4. 真实数据能否固定一组 reconstruction cameras，再留至少一台 camera
   完全不参与训练、停止和路由？
5. NeRIF/TDBOST 当前最常见失败是否集中在 shock、thin front、低频 plume，
   还是有限孔径与相机标定？

## 9. 复现

```bash
cd "${REPO_ROOT}"

PYTHONPATH=. .venv/bin/python \
  site_tools/run_psu_b0_rq_ogse_pcgls_development.py \
  --config demo_t16_operator/configs/psu_b0_rq_ogse_pcgls_development_v1.json \
  --development-report private_library/external_datasets/psu_bost_flight_body/b0_residual_risk_development_v1/private_report.json \
  --probe-private-report private_library/external_datasets/psu_bost_flight_body/b0_observable_morphology_probe_v1/private_report.json \
  --view-root private_library/external_datasets/psu_bost_flight_body/all_view_geometry_audit \
  --device mps \
  --private-output private_library/external_datasets/psu_bost_flight_body/b0_rq_ogse_pcgls_development_v1/private_report.json \
  --public-output docs/psu_b0_rq_ogse_pcgls_development_public_summary.json

PYTHONPATH=. .venv/bin/python \
  site_tools/run_psu_b0_mo_rq_ogse_pcgls_development.py \
  --config demo_t16_operator/configs/psu_b0_mo_rq_ogse_pcgls_development_v1.json \
  --development-report private_library/external_datasets/psu_bost_flight_body/b0_residual_risk_development_v1/private_report.json \
  --probe-private-report private_library/external_datasets/psu_bost_flight_body/b0_observable_morphology_probe_v1/private_report.json \
  --rq-private-report private_library/external_datasets/psu_bost_flight_body/b0_rq_ogse_pcgls_development_v1/private_report.json \
  --view-root private_library/external_datasets/psu_bost_flight_body/all_view_geometry_audit \
  --device mps \
  --private-output private_library/external_datasets/psu_bost_flight_body/b0_mo_rq_ogse_pcgls_development_v1/private_report.json \
  --public-output docs/psu_b0_mo_rq_ogse_pcgls_development_public_summary.json
```

![RQ-OGSE 与 front-risk 四联图](../demo_t16_operator/results/psu_b0_rq_ogse_pcgls_development/psu_b0_rq_ogse_pcgls_development_figure.png)
