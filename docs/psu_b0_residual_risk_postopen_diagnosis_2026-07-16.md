# OCRRG post-open 诊断：先修特征契约，再做双支路风险门

更新日期：2026-07-16

状态：`POSTOPEN_DIAGNOSIS_ONLY_NEXT_CANDIDATE_NOT_FROZEN`

## 1. 结论先说清楚

冻结的 OCRRG fresh audit 仍然可以逐样本复现其经验指标：五个支持域 split
保留非零 coverage，3 至 5 视角精确回退，且最终记录的 4 条 accepted harm
仍来自两个 6-view 源样本。

但本轮代码审计发现一个此前未报告的**特征契约缺陷**：

- development / calibration 在计算方向相关风险特征前，先把 candidate 与
  Sobolev direction 乘以体素 support mask；
- 冻结部署门在乘 support mask **之前**计算同一组方向特征；
- 因而 calibration score 与 deployment score 不是同一个确定性函数。

这不抹掉已经真实跑出的重建误差、coverage 或 fallback 指标，但会使
split-conformal 的交换性解释失效。旧结果现在应准确表述为：

> 一个冻结的经验 selective gate 通过了其 synthetic metric gate；由于
> calibration / deployment 特征函数不一致，它不能再被描述为已获得有效的
> conformal 风险解释。

## 2. 缺陷如何被验证

诊断脚本重新生成了 `7 splits × 24 fields × 3 checkpoints = 504` 条第一步
特征，不重新调阈值，也不把 opened fresh 当新确认集。

使用部署实际采用的“support 前特征”时：

- 与冻结报告 prediction 的最大绝对差为 `8.24e-5` 个百分点；
- 与冻结报告 feature distance 的最大绝对差为 `2.87e-6`；
- 因而已逐值复现真实部署评分。

把同一方向改回 development 的“support 后特征”时：

- 504 条中有 7 条 accept / fallback 决策改变，占 `1.389%`；
- prediction 最大移动 `0.826` 个百分点，平均绝对移动 `0.095` 个百分点；
- 这 7 条都不是当前 4 条 harmful rows，所以旧 fresh 的四个反例数量不变；
- 但同一 scoring-function 前提已经被机械反例否定。

此前的 equivalence probe 只检查了 validation 的前 6 个样本是否最终选择
同一 candidate / fallback，未检查两条路径的 prediction 与 feature 是否逐值
一致，因此没有暴露这个问题。

## 3. 两个真正不同的漏判机制

### 3.1 低频 plume + 修正过猛

`fresh_iid_support-012` 在 checkpoints `1741 / 1743` 上分别退化
`-2.553% / -2.682%`。

可观测特征共同表现为：

- gradient spectral centroid 与 high-frequency fraction 偏低；
- candidate correction magnitude、direction norm ratio 与 gain span 偏高；
- residual component correlation 并不异常。

通俗地说，平滑羽流把能量放在较难由梯度积分观测约束的低频方向；网络的
频谱修正看起来能继续压低数据项，却可能把三维场沿弱可辨识方向推远。

### 3.2 强相关相机噪声 + shock-like gradient

`fresh_correlated_noise_ood-011` 在 checkpoints `1742 / 1743` 上分别退化
`-5.702% / -4.471%`。

可观测特征共同表现为：

- 两个位移分量的白化 residual correlation 约为训练均值上方 `2.85σ`；
- gradient RMS、white residual scale 异常偏低；
- gradient anisotropy 偏高，形态又像 oblique shock。

这说明 IID 风格的逐相机 RMS whitening 没有消除 row / column / component
相关结构。结构化噪声可能与激波型梯度同向，让 pooled linear predictor
误以为它是“容易且有正增益”的样本。

## 4. 为什么只按视角分组仍不够

当前 accepted rows 的风险按视角分层如下：

| active views | accepted rows | mean gain | p10 | minimum | harm >1% |
|---:|---:|---:|---:|---:|---:|
| 6 | 80 | +3.715% | -0.318% | -5.702% | 4 / 80 |
| 7 | 41 | +4.187% | +1.136% | +0.442% | 0 / 41 |
| 8 | 9 | +2.276% | +0.132% | -0.092% | 0 / 9 |

6-view 确实是风险层，但把全部 6-view 一刀切回退会丢掉大量真实正增益。
更关键的是 development calibration 本身严重失衡：

| views | train fields | validation fields | calibration fields |
|---:|---:|---:|---:|
| 6 | 32 | 15 | 20 |
| 7 | 13 | 7 | 9 |
| 8 | 2 | 2 | 1 |
| 9 | 1 | 0 | 0 |

post-open exact-view quantile probe 将 pooled quantile 与同视角 quantile取更
保守者，仍然拒不掉 4 条 harmful rows；同时 8/9-view 因校准样本不足只能
机械拒答。结论是：**view count 是必要条件，不是足够的风险表征。**

## 5. 下一候选：Observable Multi-Veto Residual-Risk Gate

这只是 opened-fresh 启发的开发假设，还不是可宣称的新算法结果。

```text
accept =
  hard view support
  AND pooled lower-gain vote
  AND NOT spectral/correction stress veto
  AND NOT correlated-camera stress veto
  AND six-view backoff passes
```

所有否决都必须在第一步使用部署可观测量；任一否决触发时，四步求解逐值
回退 validation-selected inverse-Sobolev。

两个候选 stress score 的物理含义是：

```text
spectral stress =
  mean(z(correction), z(gain span), -z(centroid), -z(high-frequency))

camera stress =
  mean(z(component correlation), -z(gradient RMS),
       -z(white residual RMS), z(anisotropy))
```

这些具体组合由 opened fresh 启发，不能在旧 fresh 上调到好看后再声称验证。
正确流程是：

1. 新建唯一的 canonical feature function，support 顺序固定并被单元测试锁死；
2. 6/7/8/9 视角按字段数平衡生成，避免 mask sampler 自然偏向 6-view；
3. development selection 做 leave-one-morphology-family-out；
4. 在同一 development data 上比较 pooled ridge、exact-view Mondrian、
   二次交互 ridge、浅树与双支路 veto；
5. 使用 Learn then Test 或明确的多重检验规则选择有限个阈值；
6. 冻结代码、阈值、seeds、tie-break、coverage 与 harm failure gate；
7. 只在全新 morphology / noise / mask / seed 的 independent repeat 上判决。

后续 development-only 有限网格筛选已经执行：correlated-camera veto 拒掉了
opened fresh 中两条 shock harm，但低频 plume 的两条 harm 仍保留，因此
候选不允许冻结。详见
[Multi-Veto v2 开发筛选](psu_b0_multiveto_development_screen_2026-07-16.md)。

## 6. 理论边界

- [Conformal Risk Control](https://research.google/pubs/conformal-risk-control/)
  允许对声明的单调风险作有限样本控制，但依赖同一评分规则和交换性；
- [Learn then Test](https://arxiv.org/abs/2110.01052)说明多个候选阈值需要作为
  风险控制选择问题处理，不能在 audit set 上挑最好看的组合；
- [Automatically Adaptive Conformal Risk Control](https://proceedings.mlr.press/v258/blot25a.html)
  提供 difficulty-adaptive conditioning 思路，但当前 BOST 校准量不足以支撑
  任意细分条件；
- [Confidence on the Focal](https://academic.oup.com/jrsssb/article/87/4/1239/8113856)
  强调选择过程会改变目标保证，risk | accept 不能被 marginal coverage 代替。

本项目还没有实验三维真值、真实 flow-off covariance 或任意 OOD 保证。

## 7. 给基础还不牢时的学习顺序

1. 先手算 support projection：为什么 solver 实际搜索方向是 `S p`，而不是 `p`；
2. 理解 split conformal 需要 calibration 与 test 使用同一 nonconformity score；
3. 用二维散点画出“一个线性分数无法同时包住两个不同坏尾部”；
4. 学 selective prediction 的 risk-coverage curve，不只看 overall mean；
5. 再学习 Mondrian / group-conditional calibration 与多重检验；
6. 最后把合成 camera correlation 换成师兄提供的 flow-off covariance。

## 8. 请何远哲优先确认

1. 实验部署中最常见的有效视角数究竟是 6、7、8 还是 9？相机失效是否常见？
2. 能否给每台相机至少 20 至 50 次 flow-off repeat，用于估计 row/column 与
   两个位移分量的 covariance？
3. 羽流低频漂移和激波/薄前沿，哪一种是组内更常见、更高代价的失败？
4. 风险门最应保护 field error、held-out reprojection，还是 PIV 折射补偿后的
   velocity error？
5. 是否可以预留独立 calibration session 与最终 audit session，禁止调参交叉？

## 9. 可复现入口

- [公开诊断 JSON](psu_b0_residual_risk_postopen_diagnosis_public_summary.json)
- [四联诊断图](../demo_t16_operator/results/psu_b0_residual_risk_postopen_diagnosis/psu_b0_residual_risk_postopen_diagnosis_figure.png)
- `site_tools/analyze_psu_b0_residual_risk_postopen.py`
- `site_tools/plot_psu_b0_residual_risk_postopen.py`

私有报告保留全部 504 条 feature pair，但不进入 GitHub Pages。
