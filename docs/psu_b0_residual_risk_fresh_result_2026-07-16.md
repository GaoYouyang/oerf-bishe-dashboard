# PSU B0 可观测残差风险门：fresh 合成压力测试判决

更新日期：2026-07-16

判决：`RESIDUAL_RISK_FRESH_CANDIDATE_PASS_SYNTHETIC_ONLY`

> **2026-07-16 post-open 解释更新：**后续逐值审计发现 development /
> calibration 在 support 投影后提取方向特征，而冻结 deployment 在投影前提取。
> 两种顺序使 504 条 fresh candidate rows 中 7 条 accept/fallback 决策改变。
> 本文记录的经验重建指标、3/3 metric gate 与 4 条 accepted harm 仍可复现，
> 但 split-conformal 解释已经撤回。请同时阅读
> [特征契约诊断](psu_b0_residual_risk_postopen_diagnosis_2026-07-16.md)。

这不是“算法已经优于 FNO、DeepONet 或现有 BOST 方法”的判决。它只说明：
在预注册的真实 PSU support 几何、解析反应流形态代理和七组 fresh 压力测试
上，Observable Conformal Residual-Risk Gate（OCRRG）满足了本轮事先冻结的
候选门。

## 1. 时间边界与防泄漏

fresh 配置、冻结 checkpoint、development 风险模型、阈值与哈希先在提交
`cd5d4a0ffa997eb1d4f3820d0bc5e0aa626287df` 中提交并推送，随后才执行一次
fresh audit。

- 预注册配置：
  `demo_t16_operator/configs/psu_b0_residual_risk_fresh_prereg_v1.json`
- 预注册 SHA-256：
  `0ee26d502fc5e9920067507c30581525b673269cb942d4d23e0c082f0ca141db`
- 公开摘要 SHA-256：
  `64499946b238c7a908c552c39c1330257caaf553e3ed868d475c8539ecc2814e`
- 私有完整报告 SHA-256：
  `74677e9df67d52c071caba329e590ea7bf7b7bf34e460ab6ad69058649214083`
- fresh support mask 与全部 development mask 不重叠；
- development rotation 40 与 final audit 均未打开；
- 1,176 条逐样本方法指标由独立 validator 重新聚合。

## 2. 方法到底做了什么

候选重建器仍是冻结的 2,227 参数正谱预条件器，强回退基线仍是
validation-selected inverse-Sobolev `p=5`。两者都通过精确伴随梯度和解析
线搜索运行固定四步，部署预算相同：

```text
4 forward calls + 4 adjoint calls
```

OCRRG 不先完整运行两种重建再挑答案。它只在第一步读取部署时可观测的 16
个特征，包括：

- active-view fraction；
- 白化 residual 的分布统计和两个像移分量关系；
- 精确伴随梯度的频谱质心、高频占比和各向异性；
- candidate 与 Sobolev 方向的夹角、范数比和相对修正量；
- 学习谱增益跨度及 controller 系数尺度。

标准化 ridge 模型预测候选相对 Sobolev 的 field-gain，再用每个冻结模型
种子独立的 one-sided split-conformal overprediction quantile 构造保守下界。
只有同时满足 6 至 9 视角硬支持、特征距离和增益下界三门时才启用学习器；
否则整条四步求解回退 Sobolev。

## 3. fresh 结果

每个 split 有 24 个全新解析场，三种冻结模型种子分别评分。下表是三种子
均值；`harm` 表示相对 Sobolev 的 field error 恶化超过 1%。

| fresh split | raw gain | raw p10 | raw harm | gated gain | gated p10 | gated harm | coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| support IID | +1.273% | -4.311% | 25.000% | +1.375% | 0.000% | 2.778% | 36.111% |
| held-out morphology | +2.815% | -3.322% | 12.500% | +1.043% | 0.000% | 0.000% | 26.389% |
| strong correlated noise | +1.802% | -0.850% | 8.333% | +1.307% | -0.223% | 2.778% | 43.056% |
| morphology + noise | +3.252% | -0.528% | 5.556% | +1.408% | 0.000% | 0.000% | 27.778% |
| exact-operator control | +2.741% | +0.044% | 2.778% | +1.663% | 0.000% | 0.000% | 47.222% |
| 3-5-view geometry OOD | +0.410% | -2.537% | 25.000% | approximately 0 | 0.000% | 0.000% | 0.000% |
| 3-5-view joint OOD | +0.233% | -2.813% | 29.167% | approximately 0 | approximately 0 | 0.000% | 0.000% |

三种冻结模型种子都满足预注册门，因此为 `3/3`。支持域外的 candidate
coverage 精确为 0；与 Sobolev 的最大逐指标绝对差为
`1.1920928955078125e-7`，而逻辑调用仍是 `4F+4A^T`。

## 4. 必须保留的失败样本

通过候选门不等于没有伤害。独立 validator 找到 4 条“被接受但恶化超过
1%”的记录，来自两个源样本：

| split / sample | family / noise / views | seed | field gain |
|---|---|---:|---:|
| `fresh_iid_support-012` | plume / iid / 6 | 20261741 | -2.553% |
| `fresh_iid_support-012` | plume / iid / 6 | 20261743 | -2.682% |
| `fresh_correlated_noise_ood-011` | oblique shock / strong correlated / 6 | 20261742 | -5.702% |
| `fresh_correlated_noise_ood-011` | oblique shock / strong correlated / 6 | 20261743 | -4.471% |

这两个病例说明当前 pooled risk model 对“最低支持视角数 + 特定形态/噪声”
仍不够保守。当前结果最多支持 selective safety mechanism 的继续研究，
不能支持逐样本安全保证，也不能支持任意 OOD 的 conformal 保证。

## 5. 结果的现实意义

这轮最有价值的结论不是平均增益本身，而是得到了一条可执行的算法设计
原则：

> 学习预条件器可以只在可观测证据足够时接管求解，其余样本精确退回强
> 经典基线；但风险模型必须按物理条件分组审计，不能只依赖 pooled 均值。

它贴合 BOST 的真实困难：实验中没有三维真值可供在线选择，视角数、相关
相机噪声、薄前沿与积分算子失配又会共同变化。OCRRG 只读取 residual、
伴随梯度、视角 mask 和候选方向，因此比用真值挑模型更接近可部署流程。

## 6. 证据边界

- 几何来自公开 PSU 数据，但没有使用 PSU 实验像移值；
- 三维真值是 32^3 解析形态代理，不是 CFD 或实验场；
- 噪声是合成模型，不是 flow-off repeatability；
- 没有与 FNO、DeepONet、FCG-NO、NeRIF、TV 或完整 CGLS 前沿做公平比较；
- 没有打开 rotation 40 development 或 70-view final audit；
- conformal 校准的交换性假设不能外推到任意形态、装置或噪声；
- 因而不授权算法优越性、实验场精度或 OERF 应用主张。

## 7. 下一轮优先级

1. **独立重复。** 更换风险 train/calibration/fresh seeds，完整重建风险模型，
   检验 3/3 是否可重复。
2. **组条件风险。** 按 active-view count、形态族与噪声级做 groupwise 或
   worst-case 校准，重点修复 6-view plume 和强相关 oblique-shock 病例。
3. **留一形态族开发。** 在 risk model 训练时轮流留出整个 morphology family，
   防止 16 个特征只记住当前解析场生成器。
4. **真实噪声替换。** 向何远哲请求 flow-off repeats，估计逐相机 covariance、
   row/column correlation 和漂移，再冻结实验风险特征。
5. **公平强基线。** 同 calls / wall time 比较 validation-selected Sobolev、
   CGLS、Tikhonov、TV、FCG-NO、DeepONet/FNO 类基线；没有真实 field truth 时
   分开报告 support fit、held-out reprojection 和 repeatability。
6. **最后才开真实留出。** 只有真实噪声门与 calibration perturbation 通过，
   才允许打开 rotation 40 development；模型冻结后再触碰 final audit。

## 8. 需要向师兄确认

1. 是否能提供至少 20 至 50 次 flow-off 位移重复，以及每个相机的曝光、
   background pattern、PIV/BOS preprocessing 设置？
2. 真实任务最常见的是 6、7、8 还是 9 个有效视角？是否存在相机临时失效？
3. 当前最担心的形态是薄火焰面、激波、羽流、多核结构还是强时间突变？
4. 组内最终更关心 density/refractive-index field、held-out reprojection，
   还是 PIV 折射补偿后的 velocity error？
5. 可否提供一个独立 calibration phantom 或 session，专门用于风险门校准，
   与最终展示数据分开？

## 9. 复现入口

- [公开结果 JSON](psu_b0_residual_risk_fresh_public_summary.json)
- [fresh 预注册说明](psu_b0_residual_risk_preregistration_2026-07-16.md)
- [Conformal Risk Control](https://research.google/pubs/conformal-risk-control/)
- [Learning Preconditioners for Conjugate Gradient PDE Solvers](https://proceedings.mlr.press/v202/li23e.html)
- [DeepONet Based Preconditioning Strategies](https://epubs.siam.org/doi/10.1137/24M162861X)
- [Learned Operator Correction](https://epubs.siam.org/doi/10.1137/20M1338460)
