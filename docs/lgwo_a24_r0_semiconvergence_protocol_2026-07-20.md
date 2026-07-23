# LGWO-A24 R0：逐样本半收敛上限预注册

冻结日期：2026-07-20  
状态：`FROZEN_BEFORE_FIRST_TRAJECTORY_RUN`  
问题：**PSU-C1 首方向门失败后，零修正 CGLS 的逐样本最佳停止步，相对 validation 选出的统一停止步，是否仍有足够大且足够多样的场误差改进空间？**

## 1. 为什么先做 R0，而不是立刻训练停止网络

停止网络只能学习一个已经存在、并且能从观测量识别的差异。如果所有样本的场误差都在近似相同的 `k` 最小，或者 oracle 相对统一 `k` 只有一两个百分点，那么继续设计 DeepONet、FNO 或 MLP 选择器只会把模型复杂度加到一个没有足够效应量的问题上。

R0 因此分两层问：

1. **truth ceiling**：每个样本单独看真值，最好的 `k` 能比 validation 固定 `k` 好多少？最佳 `k` 是否真的随 morphology、noise、mask 和 operator mismatch 改变？
2. **observable proxy ceiling**：只看 inactive-view clean residual 的 oracle，是否至少不会系统性伤害场误差？

只有两层都通过，后续 R1 才允许做一个很小的 observable stopping screen。R0 无论结果如何都不授权神经网络成功、真实 BOST 成功或论文创新结论。

## 2. 与 E72 的非重复边界

| 维度 | E72 | 本 R0 |
|---|---|---|
| 物理输入 | 同一个真实 PSU 实测流场 | 公开 PSU 九视角射线几何 + 解析反应场 + 合成噪声 |
| 真值 | 无实验 3D truth | 有生成器 truth，但不是实验 truth |
| 几何变化 | 0/50/90 度 rotation transfer | 每例不同 active-view mask；qmc32/qmc8 mismatch control |
| checkpoint | `{0,1,2,3,4,6,8,12}` | 每一步 `k=1..24` |
| 核心问题 | 两个 rotation 能否为第三个 rotation 安全选早停 | 是否存在值得学习的逐样本停止上限与 observable proxy |
| 证据等级 | real measured flow 上的 post-open NO-GO | public-geometry synthetic mechanism rehearsal |

E72 已经说明跨 rotation 的平均安全性不能保护第三组相机尾部。R0 不重开这项结论，也不把合成 truth 当作实验体积真值。

## 3. 冻结数据合同

- 几何：Penn State flight-body BOS 数据集公开九视角 support ray bundles，DOI `10.26208/1VE2-5C19`；本实验不用其实验 deflection 做训练或评分。
- 生成场：已有 `psu_b0_reaction_phantoms` 的 deterministic analytic morphology。
- validation：24 例 `plume/wavy_front`，只用于选择统一 `k`。
- 描述性 test：IID、family-OOD、noise-OOD、geometry-OOD、joint-OOD、exact-operator control，各 24 例。
- 所有 source splits 与 PSU-C1 结果在本协议冻结前均已打开；因此 test 只作 post-open 描述，不能写成 fresh/final。
- 每个 mask 至少留一个 inactive PSU view；若不存在 B coverage，实验无效。

## 4. 冻结求解器与预算

唯一求解轨迹是：

```text
zero-start fully reorthogonalized CGLS
proposal = None
correction radius = 0
checkpoints = 1,2,...,24
reorthogonalization passes = 2
dtype = float32
```

每例完整轨迹必须精确使用 `24 A + 24 A^T`。评分时把 24 个场批量送入 nominal operator，记为 `1` 次 API forward、`24` 次 case-equivalent forward；评分调用不混入算法预算。

另外，为每个 split 生成 inactive-view clean target 的 true/nominal operator forward 必须单列 API calls、case-equivalent calls 和 wall time，不能藏在 synthetic data generation 或算法成本中。`qmc8` exact-operator control 必须用 nominal operator 生成 clean projection，其他 `qmc32` split 用 true operator。

同时记录：

- field relative-L2；
- gradient relative-L2；
- top-10% front F1；
- active measured weighted residual；
- active clean weighted residual；
- inactive-view clean residual；
- weighted residual / synthetic sensor-noise norm；
- 正交缺陷、残差增量、breakdown、wall time、RSS 与调用账。

## 5. 八种冻结选择器

| 选择器 | 用途 | 泄漏/部署边界 |
|---|---|---|
| `fixed_k24` | 最大预算参考 | 固定可执行 |
| `historical_fixed_k4` | 与 E72 的上下文参考 | 固定可执行，不是本数据选择出的主基线 |
| `validation_global_fixed_k` | validation 平均 field-L2 最小，平局取更小 `k` | 标准离线 truth 调参；test 不用 truth |
| `active_measured_min_k` | active residual 最小 | 不用 truth，但必须跑完整轨迹，通常偏向最后一步；只作诊断 |
| `sensor_noise_discrepancy_first_crossing` | residual 首次低于 synthetic noise norm | 不用 truth；sigma 不含 model bias，不能平移为 OERF 规则 |
| `heldout_b_clean_oracle_k` | inactive-view clean residual 最小 | 泄漏生成 truth；只作 observable proxy ceiling |
| `truth_gradient_oracle_k` | gradient error 最小 | evaluator-only ceiling |
| `truth_field_oracle_k` | field error 最小 | 主 instance-specific ceiling |

所有逐样本 gain 先相对同一例的 validation-fixed 计算，再汇总；`k24` 只作第二参考。

## 6. 预注册通过门

主 split 为 IID 与 family-OOD。每个 split 均需同时满足：

### Truth headroom

- 24 例完整覆盖；
- truth-field oracle 平均 field gain `>= 5%`；
- 20,000 次 case bootstrap 的均值 gain 95% 下界 `>= 0%`；
- oracle 至少出现 3 个不同 checkpoint；
- oracle checkpoint IQR `>= 2`；
- 相对 validation-fixed 的 `gain < -1%` harm rate 必须为 0。

### Observable proxy

- heldout-B oracle 的平均 field gain `>= 0%`；
- heldout-B oracle 的 field harm rate `<= 10%`。

### 数值与账本

- 每例所有 24 checkpoint 都存在且唯一；
- 精确 `24F/24AT`，评分精确 `1` API F / `24` case-equivalent F；
- measurement-direction orthogonality defect `<= 2e-5`；
- squared residual 最大增加 `<= 2e-5`；
- 保存 residual 与重新 forward residual 的绝对范数差，除以两者中的 recomputed residual 与 synthetic sensor-noise norm 较大者，必须 `<= 2e-5`；
- 不允许 breakdown，且第 `k` 步必须累计接受 `k` 个方向；
- 每例 heldout-B 有效。

只有 truth headroom、checkpoint diversity、heldout-B proxy 和结构门全部通过，才可进入单独预注册的 R1。即便通过，也只表示“值得筛一个可观测规则”，不表示规则已存在，更不表示神经模型已成功。

## 7. 结果状态词

- `INVALID_R0_SEMICONVERGENCE_REHEARSAL`：轨迹、调用、数值或覆盖合同失败；不得解释科学结果。
- `VALID_NO_GO_STOPPING_HEADROOM_OR_DIVERSITY_ABSENT_POSTOPEN`：上限太小或最佳 `k` 不够多样；关闭 learned stopping。
- `VALID_TRUTH_HEADROOM_BUT_HELDOUT_PROXY_NO_GO_POSTOPEN`：truth 上限存在，但可观测 B 代理无法保护 field；暂不训练。
- `VALID_GO_R1_OBSERVABLE_STOPPING_SCREEN_POSTOPEN`：只授权一个后续小型 observable screen。

## 8. 独立复核

正式运行后，`validate_lgwo_a24_r0_semiconvergence.py` 不导入运行器，而从 sealed CSV 重新：

1. 校验完整文件集与 SHA-256；
2. 检查每例 `k=1..24`；
3. 重选 validation 固定 `k` 与每例八种 selector；
4. 重算所有配对 gain、ratio 与 20,000 次 bootstrap；
5. 重建 gate/status 并精确比对 summary；
6. 检查所有禁止 claim 保持 `false`。

还要将 12 个求解、数据、算子、评分与 validator 源文件的 SHA-256 绑定到 result 中记录的 source commit；每例记录 truth、observation、clean projection、mask/sigma 与整条 field/residual/projection trajectory 的 tensor hash。独立 validator 会从本地校验过的 PSU geometry 重建 sample ID、family、mask、sigma、noise realization、truth operator、analytic truth、observation、clean projection 和 operator mismatch，并逐例比对四类 input tensor hash。

为避免把自证强度说大：result package 不重复保存约 0.5 GB 的全部 3D checkpoints，所以 independent validator **不从体数据重算底层 field/gradient/F1**。它会独立重建全部 input tensors，并重算 selector、gain、bootstrap 和 gate；trajectory tensors 则由 hash、source commit 和冻结的 metric code 绑定。这是明确的审计边界，不写成第二次独立体场重建。

复现命令在协议提交后才运行：

```bash
.venv/bin/python site_tools/run_lgwo_a24_r0_semiconvergence.py --device mps
.venv/bin/python site_tools/validate_lgwo_a24_r0_semiconvergence.py
```

## 9. 无论结果如何都不能写什么

- 不能写成新的 stopping algorithm；
- 不能写成训练过神经网络；
- 不能写成优于 DeepONet、FNO、NeRIF 或 TDBOST；
- 不能写成实验 PSU 3D truth、OERF 验证或 unseen-rig 泛化；
- 不能把 synthetic mean gain 写成算法成功；
- 不能使用“突破”“SOTA”“高质量论文已完成”等词。

正式机器输出必须保留失败 artifact；结果出来后只按冻结状态词更新学习日志、聚焦网页与论文边界。
