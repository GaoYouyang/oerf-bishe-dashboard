# v154：扩大公开训练覆盖后，跨轨迹支持仍未过门

更新：2026-08-16

## 先说结论

v154 把四条剩余的、此前已经在开发阶段打开过的完整 PoolFire 训练候选加入覆盖审计。公开训练轨迹由 6 条增至 10 条，样本由 4,440 增至 7,400，active camera rows 由 36,630 增至 61,050。

结果仍然没有通过冻结的 90% 跨轨迹支持门：全局支持率为 **87.07%**，10 条完整轨迹中只有 **7 条**通过汇总门。`p45-s05`、`p58-s03` 和新增的 `p58-s05` 分别只有 **16.79% / 77.62% / 87.13%**。

正式执行和独立第二实现得到同一判决：

- 正式状态：`PASS_FORMAL_BROADER_TRAIN_COVERAGE_EXECUTION_V154`
- 独立状态：`PASS_INDEPENDENT_RECOMPUTATION_BROADER_TRAIN_COVERAGE_V154`
- 科学判决：`FAIL_BROADER_TRAIN_COVERAGE_V154`

因此，当前“跨轨迹系数预测”路线停止。不训练 predictor，不做物理 replay，不用更大的 CNN/FNO/UNO/DeepONet 补救，也不租 GPU。

## 为什么做这一步

v153 已经排除了固定仿射和单调坐标规范化。剩下最直接的解释是：训练工况覆盖还不够。v154 不改表示、不加模型，只问一个更干净的问题：把所有仍可用于 post-open 开发的完整公开训练轨迹加进来，原始 deployment-visible 表示能否获得足够跨轨迹邻域支持？

角色审计确认，新增的 `p33-s05`、`p45-s01`、`p45-s03`、`p58-s05` 都已被早期开发工作打开，不能再充当未来 fresh/confirmatory holdout，只能用于 post-open 训练覆盖诊断。validation、stopping-validation 和两条 untouched test 始终未读；只有五帧的旧开发轨迹也没有冒充完整轨迹。

## 冻结方法

v154 完全继承 v152 的 raw deployment-visible state、complete-trajectory leave-one-out、fold-train-only normalization、90% 支持阈值、5/7/9/12 相机和 clean/noise/pose/intrinsic/combined 分层。

它没有运行 v153 的仿射或单调 warp，没有新增候选，没有调阈值，没有读取 Krylov target 或 CFD truth，也没有拟合预测器。

## 结果明细

| 留出轨迹 | 汇总支持率 | 5 相机 | 7 相机 | 9 相机 | 12 相机 |
| --- | ---: | ---: | ---: | ---: | ---: |
| p45-s05 | 16.79% | 16.32% | 20.62% | 17.96% | 13.87% |
| p58-s03 | 77.62% | 77.41% | 79.77% | 77.48% | 76.58% |
| p58-s05 | 87.13% | 91.03% | 82.47% | 89.97% | 86.08% |

新增轨迹中，`p33-s05`、`p45-s01`、`p45-s03` 的汇总支持率分别为 **98.98% / 99.72% / 97.94%**，但 `p58-s05` 没有过门。更关键的是，`p45-s05` 和 `p58-s03` 的 clean 分层也只有 **21.21%** 和 **80.61%**，说明主要缺口并不是加入噪声或标定误差才出现。

总计有 **11 个 trajectory × camera-count 分层**和 **35 个 trajectory × perturbation 分层**低于冻结阈值。扩大公开覆盖带来局部帮助，但没有建立全轨迹、全相机数、全扰动层级的支持闭环。

## 独立复算

独立程序重新构造四条新增轨迹的 exact-K1 state，再独立完成十轨迹 leave-one-out 支持审计。20/20 项检查全部为真：

- added-state 最大绝对差：`7.11e-15`；
- 数值数组最大差：`3.11e-15`；
- 汇总最大差：`0`；
- 正式与独立判决完全一致。

两条实现仍共享冻结 physics kernels，因此没有证明端到端物理独立性。

## 成本和边界

四条新增轨迹的离线 state 构造使用 `2960A + 2960A^T`；支持审计本身新增 `0A + 0A^T`。这是离线诊断账，不是部署节省，也不是重建结果。

本结果不证明算法成功、matched accuracy、加速、内存优势、外部泛化、curved ray 或真实 BOST。`algorithm_breakthrough=false`，`paper_success=false`。

## 下一步

当前 raw cross-trajectory coefficient predictor 路线到此关闭。只有两类新证据值得继续：

1. 真正更广的公开或组内真实工况数据；
2. 结果前单独冻结、物理上不同且仍只读 deployment-visible observation/geometry 的表示。

在这两类证据到来前，不继续扩模、不租 GPU，validation/test 继续封存。

---

# v154: broader public training coverage still fails the support gate

Updated: 2026-08-16

## Verdict

v154 adds all four remaining full PoolFire fit candidates that had already been opened by earlier development work. The public training audit expands from six to ten trajectories, from 4,440 to 7,400 samples, and from 36,630 to 61,050 active-camera rows.

The frozen 90% cross-trajectory support gate still fails. Global support is **87.07%**, and only **7 of 10** complete trajectories pass in aggregate. `p45-s05`, `p58-s03`, and the newly added `p58-s05` reach only **16.79% / 77.62% / 87.13%**.

Formal execution and an independent second implementation agree:

- Formal: `PASS_FORMAL_BROADER_TRAIN_COVERAGE_EXECUTION_V154`
- Independent: `PASS_INDEPENDENT_RECOMPUTATION_BROADER_TRAIN_COVERAGE_V154`
- Scientific decision: `FAIL_BROADER_TRAIN_COVERAGE_V154`

The current cross-trajectory coefficient-prediction route therefore stops. No predictor fitting, physical replay, larger CNN/FNO/UNO/DeepONet rescue, or GPU rental is authorized.

## Why this was tested

v153 ruled out fixed affine and monotone coordinate canonicalization. The next direct explanation was insufficient training-condition coverage. v154 changes neither the representation nor the model. It asks whether adding every remaining full public trajectory available for post-open development gives the raw deployment-visible representation enough cross-trajectory support.

The role audit confirms that `p33-s05`, `p45-s01`, `p45-s03`, and `p58-s05` were already opened by earlier development work and cannot serve as future fresh or confirmatory holdouts. Validation, stopping-validation, and both untouched tests remain unread. The old five-frame development sample is not treated as a full trajectory.

## Frozen method

v154 inherits the v152 raw deployment-visible state, complete-trajectory leave-one-out evaluation, fold-train-only normalization, frozen 90% threshold, 5/7/9/12-camera strata, and clean/noise/pose/intrinsic/combined perturbation strata.

It does not reuse the v153 affine or monotone warp, add a candidate, tune the threshold, read a Krylov target or CFD truth, or fit a predictor.

## Detailed result

| Held-out trajectory | Aggregate support | 5 cameras | 7 cameras | 9 cameras | 12 cameras |
| --- | ---: | ---: | ---: | ---: | ---: |
| p45-s05 | 16.79% | 16.32% | 20.62% | 17.96% | 13.87% |
| p58-s03 | 77.62% | 77.41% | 79.77% | 77.48% | 76.58% |
| p58-s05 | 87.13% | 91.03% | 82.47% | 89.97% | 86.08% |

Among the added trajectories, `p33-s05`, `p45-s01`, and `p45-s03` reach **98.98% / 99.72% / 97.94%** aggregate support, while `p58-s05` fails. More importantly, the clean strata for `p45-s05` and `p58-s03` are only **21.21%** and **80.61%**, so the dominant gap is not created only by synthetic noise or calibration perturbations.

In total, **11 trajectory × camera-count strata** and **35 trajectory × perturbation strata** fall below the frozen threshold. Broader public coverage helps locally but does not close support across all trajectories, camera counts, and perturbation levels.

## Independent recomputation

The independent program rebuilds all four added exact-K1 states and independently reruns the ten-trajectory leave-one-out support audit. All 20/20 checks pass:

- maximum added-state absolute difference: `7.11e-15`;
- maximum numeric-array difference: `3.11e-15`;
- maximum summary difference: `0`;
- formal and independent scientific decisions match exactly.

Both implementations still share frozen physics kernels, so end-to-end physics independence is not proven.

## Cost and claim boundary

Offline state construction for the four added trajectories uses `2960A + 2960A^T`; the support audit itself adds `0A + 0A^T`. This is offline diagnostic accounting, not a deployment saving or reconstruction result.

This result does not establish an algorithm, matched accuracy, speedup, memory advantage, external generalization, curved-ray validity, or real BOST transfer. `algorithm_breakthrough=false` and `paper_success=false`.

## Route action

The current raw cross-trajectory coefficient-prediction route closes here. Further work requires either genuinely broader or real operating conditions, or a separately preregistered, physically different representation that still reads only deployment-visible observations and geometry. Until then, do not scale the model or rent a GPU; validation and test remain sealed.
