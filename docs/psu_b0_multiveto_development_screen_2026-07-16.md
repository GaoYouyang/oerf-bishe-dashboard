# Observable Multi-Veto v2：开发筛选只修复了一半坏尾部

更新日期：2026-07-16

状态：`DEVELOPMENT_SCREEN_PARTIAL_MECHANISM_SIGNAL_NOT_READY_TO_FREEZE`

## 1. 这轮做了什么

旧 OCRRG 的 support-order 特征契约被否掉后，新建了独立 v2 模块：

- candidate / fallback direction 必须先乘 support，再计算所有方向特征；
- spectral stress 与 correlated-camera stress 只使用部署可观测量；
- 6-view 可以增加额外 gain-margin，但不能从 opened fresh 事后指定；
- 任一 veto 触发时精确回退 inverse-Sobolev。

本轮只用旧 development rows 选择有限网格，不让 opened fresh 参与阈值选择。
之后才把 development-selected 版本放回 opened fresh 做诊断；该诊断不构成
新的 fresh 证据。

## 2. 有限网格与选择结果

网格包含：

- train spectral stress 的 `70/80/90/95/97.5/100%` 分位数与 no-veto；
- train camera stress 的同一组分位数与 no-veto；
- 6-view extra margin：`0/0.25/0.5/1/1.5/2%`。

共 294 个候选，253 个满足 development validation 的：

- coverage 至少 20%；
- overall harm 不超过 5%；
- 任一 view stratum harm 不超过 10%。

最终 tie-break 选择：

| parameter | selected |
|---|---:|
| spectral stress threshold | 1.3165 |
| camera stress threshold | 1.0266 |
| 6-view extra margin | 0% |

它在 validation 上 coverage 为 52.78%、mean selected gain 为 +2.066%、harm
为 0；calibration 上 coverage 为 51.11%、mean gain 为 +1.596%、harm 为 0。

但这个选择接近“宽松 veto”，且 6-view backoff 被选为 0。这说明旧
development 并没有包含足够强的低频 plume 失败证据，不能从 validation
自动推导出事后看起来有效的 6-view 阈值。

## 3. 放回 opened fresh 后发生什么

先把旧 deployment 的 support-order 缺陷修为 canonical pooled gate，再叠加
development-selected multi-veto：

| split | canonical coverage / gain / harm | multi-veto coverage / gain / harm |
|---|---|---|
| support IID | 36.11% / +1.369% / 2.78% | 27.78% / +1.010% / 2.78% |
| held-out morphology | 29.17% / +1.105% / 0 | 26.39% / +0.955% / 0 |
| strong correlated noise | 45.83% / +1.348% / 2.78% | 27.78% / +0.915% / 0 |
| morphology + noise | 27.78% / +1.408% / 0 | 23.61% / +1.167% / 0 |
| exact operator | 45.83% / +1.655% / 0 | 45.83% / +1.655% / 0 |

它拒掉了 correlated-noise oblique shock 的两条 harmful rows，但
`fresh_iid_support-012` 的两条低频 plume harm 仍被接受。

因此判决不是“v2 成功一半”，而是：

> correlated-camera stress 是值得带进新 development 的机制候选；当前
> spectral stress 和旧 development 不足以形成可冻结的 plume 风险门。

## 4. 为什么不能把 6-view margin 调到刚好抓住 plume

两个 plume 反例的 lower gain bound 约为 `-0.57% / -0.47%`。在旧
`-1%` margin 上事后增加约 `0.6%`，确实可能同时拒掉它们。

但这个数是看过 fresh 后得到的。如果直接写进下一协议，就是典型 audit
leakage。严肃做法是先生成包含低频羽流、弱梯度、不同 correction magnitude
的新 development fields，让阈值从 development 风险中自然选出；若仍选不出，
就否掉该支路，而不是强行保住想法。

## 5. 下一轮 development 必须改变什么

1. 使用 `balanced_view_masks()` 平衡 6/7/8/9-view 字段数；
2. 允许不可避免的 mask pattern 重复，独立单位改为 field/noise/session；
3. 增加 smooth broad plume、stratified low-frequency layer 与 weak-gradient
   drift，不复用 opened fresh seeds；
4. 增加 component-correlated、row/column drift、view bias 与信号相关噪声；
5. 用 leave-one-family-out 与 leave-one-noise-regime-out 选择 veto；
6. 比较 pooled ridge、exact-view Mondrian、quadratic ridge、浅树与
   multi-veto，全部使用同一 canonical feature contract；
7. 若 spectral branch 只能靠 opened-fresh-inspired margin 工作，正式 NO-GO；
8. development 通过后再冻结全新 independent repeat。

## 6. 当前判决

- canonical v2 implementation skeleton：**通过单元测试，尚未端到端 fresh**；
- correlated-camera veto：**有 post-open 机制信号**；
- spectral/plume veto：**未修复 opened failure**；
- candidate freeze：**不允许**；
- conformal guarantee：**无**；
- superiority / OERF claim：**无**。

## 7. 复现入口

- [公开筛选 JSON](psu_b0_multiveto_development_screen_public_summary.json)
- `demo_t16_operator/psu_b0_multiveto_risk.py`
- `demo_t16_operator/configs/psu_b0_multiveto_development_screen_v1.json`
- `site_tools/screen_psu_b0_multiveto_development.py`
