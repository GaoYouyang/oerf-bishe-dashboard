# v174：同成本归因确认相机选择器本身存在开发集 headroom

更新：2026-08-21

## 先说结论

v173 已经证明，选中五相机后的固定 H1 初始化在不运行 CGLS 时就能通过全部门。因此 v174 把后端完全固定：四种相机选择策略都使用同一个 `H1-K0` 重建、同一个 `1A+1A^T` 逻辑在线账，并各自与同一子集上的 Zero-K4 比较。这样只剩一个变量：**相机是怎样选出来的。**

独立复算后的结果是：

- v172 selector：严格安全 `468/468`，完整标定 / 三维场 / 时间为 `13/13 · 9/9 · 4/4`；
- fit-static：`323/468`，完整组为 `1/13 · 0/9 · 0/4`；
- v169 低模态 D-opt：`192/468`，完整组为 `0/13 · 0/9 · 0/4`；
- 新增的结果不可见 ray-axis maximin：`455/468`，完整组为 `9/13 · 4/9 · 1/4`。

三个对照都没有完整通过，因此科学判决是：

`PASS_POSTOPEN_SELECTOR_ONLY_HEADROOM_V174`

精确含义是：**在已经开封的受控 straight-ray 代理上，v172 相机选择策略的完整通过不能由这三个同成本冻结对照解释。** 这是选择器归因的实质正证据，但仍不是一个已经完成训练和部署验证的算法。

## 同成本公平比较

| 相机选择策略 | 严格安全单元 | 完整标定 / 场 / 时间 | field / gradient / observation p90 | exact `A / A^T` | 完整判决 |
| :--- | ---: | :---: | :--- | :---: | :---: |
| v172 selector | 468 / 468 | 13 / 13 · 9 / 9 · 4 / 4 | 0.327496 / 0.621204 / 0.118422 | 1 / 1 | PASS |
| Fit-static | 323 / 468 | 1 / 13 · 0 / 9 · 0 / 4 | 0.338375 / 0.857127 / 0.131662 | 1 / 1 | FAIL |
| v169 low-mode D-opt | 192 / 468 | 0 / 13 · 0 / 9 · 0 / 4 | 0.324087 / 0.882023 / 0.134096 | 1 / 1 | FAIL |
| Ray-axis maximin | 455 / 468 | 9 / 13 · 4 / 9 · 1 / 4 | 0.274764 / 0.712584 / 0.128909 | 1 / 1 | FAIL |

ray-axis maximin 是看结果前冻结的便宜确定性对照。它只使用实际进入 forward 的报告相机世界坐标射线，最大化五相机集合的最小轴向分离；不读取三维真值、误差结果或算子投影。它在全局 p90 上表现不差，甚至 field p90 更低，但仍有 13 个单元越线，并且无法守住全部标定、场和时间。这正说明本研究的判决不是只看一个平均数。

## 为什么这一步比 v172 更扎实

v172 已经做了标定、完整场和时间三重隔离，但后续 v173 发现额外 CGLS K1 并非必要。v174 把所有策略的后端和 exact-call 成本都统一为 H1-K0，消除了“是不是某个策略只是多做了一步求解”的解释。

v172 selector 的 field / gradient / observation p90 为 `0.327496 / 0.621204 / 0.118422`，相对各自同子集 Zero-K4 的无害门全部通过，harm 为零。三个对照没有一个同时通过逐单元绝对门、完整轴尾部门和 matched-reference 门。

## 独立复算

完全独立第二实现使用不同的稀疏 forward/adjoint 与解析 DCT 路径，重新构造相机轴、四种选择、H1-K0 场、各自 Zero-K4、二维残差、全部尾部门与调用账。

`27/27` 项检查全部通过：

- 逐单元指标最大差 `1.62e-11`；
- 策略汇总最大差 `8.66e-12`；
- 候选场与残差最大相对差 `1.47e-11 / 2.78e-11`；
- ray-axis 行最大差 `5.55e-15`，相机换序不改变选择；
- adjoint identity 最大相对误差 `1.78e-16`；
- exact-call 差为 `0`；
- v173 父指标重放差为 `0`；
- 所有离散选择、通过/失败和最终判决完全一致。

## 下一步与突破判断

v174 只授权一个很窄的下一步：冻结最小共享参数、只读 deployment-visible observation 与 reported geometry 的 CPU 选择器，在完整轨迹隔离下复现安全选择，再用同一 H1-K0、`1A+1A^T` 做物理重放和同成本对照。通过以前不训练大规模 neural operator，也不租 GPU。

当前选择器的风险标签来自已经开封的受控候选结果，所以这不是 fresh 外部泛化。也没有工况匹配的实验二维位移、fresh wall/RSS、curved-ray 或真实 BOST 证据。

因此，本轮是可喜的科学增量，但不是突破性结论：

`algorithm_breakthrough=false`、`paper_success=false`、`external_generalization=false`、`resource_speedup=false`、`curved_ray_validated=false`、`real_bost=false`。

---

# v174: equal-cost attribution establishes selector-only development headroom

Updated: 2026-08-21

v173 showed that the fixed H1 initializer on the selected five-camera subset already passes every gate without CGLS. v174 therefore fixes the entire reconstruction backend: every camera-selection policy uses the same `H1-K0` reconstruction, the same logical `1A+1A^T` online ledger, and its own same-subset Zero-K4 reference. Camera selection is the only changing factor.

After independent recomputation, the v172 selector is strict-safe on `468/468` cells and clears `13/13` calibrations, `9/9` complete 3D fields, and `4/4` times. Fit-static reaches `323/468` and `1/13 · 0/9 · 0/4`; frozen v169 low-mode D-opt reaches `192/468` and `0/13 · 0/9 · 0/4`; the result-free ray-axis maximin control reaches `455/468` and `9/13 · 4/9 · 1/4`. No control passes completely.

Decision: `PASS_POSTOPEN_SELECTOR_ONLY_HEADROOM_V174`.

The exact conclusion is that, on the already opened controlled straight-ray proxy, the complete v172 selector result is not explained by these three frozen equal-cost controls. This is substantive selector-attribution evidence, but it is not yet a trained and deployment-validated algorithm.

The result-free ray-axis maximin control uses only the reported world-frame rays that actually enter the forward model. It maximizes the minimum axial separation of the selected five-camera set and reads no 3D truth, outcome metric, or operator projection. Its global field / gradient / observation p90 values are `0.274764 / 0.712584 / 0.128909`, but thirteen cells still fail and complete calibration, field, and time axes do not all survive. A favorable global average is therefore insufficient.

A fully independent implementation rebuilds camera axes, all four selections, H1-K0 fields, each policy's Zero-K4 reference, 2D residuals, tail gates, and exact-call receipts through different sparse-operator and analytic-DCT code paths. All `27/27` checks pass. Maximum metric and policy-summary differences are `1.62e-11` and `8.66e-12`; candidate-field and residual relative differences are `1.47e-11` and `2.78e-11`; ray-axis rows differ by at most `5.55e-15`; the adjoint identity error is `1.78e-16`; exact-call difference is zero; and every discrete decision agrees.

v174 authorizes only one narrow next gate: freeze the smallest shared-parameter CPU selector that reads deployment-visible observations and reported geometry, reproduce safe selections under complete-trajectory isolation, and physically replay the same H1-K0 `1A+1A^T` reconstruction against the same controls. A large neural operator and GPU rental remain unauthorized until that gate passes.

The selector-risk labels still come from opened controlled candidate outcomes. No fresh external condition, condition-matched experimental 2D displacement, fresh wall/RSS measurement, curved-ray validation, or real-BOST result is present.

This is encouraging scientific progress, not a breakthrough claim: `algorithm_breakthrough=false`, `paper_success=false`, `external_generalization=false`, `resource_speedup=false`, `curved_ray_validated=false`, `real_bost=false`.
