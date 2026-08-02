# v95.1：九维 observation-only 小模型把严格通过提高到 81/90，但没有过全单元门

## 一句话结论

v95.1 得到的是一个有局部进步、但必须判失败的结果：只看 observation 与 known geometry 的线性 ridge，把父 K1 的严格通过数从 `78/90` 提高到 `80/90`；加上折内校准的可观测选择器后达到 `81/90`，并把最大门的中位数从 `-0.04173` 改善到 `-0.06344`。可是严格合同要求 `90/90`，最佳策略仍有 9 个失败，因此不能称算法成功。

科学状态为 `FAIL_NO_STRICT_OBSERVATION_ONLY_PHYSICAL_BALL_SELECTOR_V95`，`algorithm_breakthrough=false`。

## 为什么做这一步

v91 已经把九维全局二阶方向写成基不变的物理能量球，并在 truth-aware 搜索中达到 `89/90`。v94 又排除了“围绕最后一个 F30 单帧继续增加局部窗口”的路线。于是最直接的问题变成：

> 不再增加方向，只用部署时能看到的 observation 和已知几何，能否预测九维物理球坐标，并用置信度对少数风险帧回退？

这一步若成功，才有理由训练更强模型或开独立外门；若小模型和回退连已开封开发工况都守不住，就不该租 GPU 堆网络。

## 结果前冻结的合同

- 数据：已经开封的 Case 6，30 个物理帧、3 个九视角几何，共 90 个单元；
- 外折：5 个连续六帧块，外层和内层都带一帧 embargo，同一物理帧的三个几何永远同角色；
- 输入：81 维 compact 与 177 维 enriched observation-only 特征，不使用绝对帧号、未来帧、真值或上一帧真系数；
- 目标：v91 封存的九维物理球坐标；
- 模型：scenario mean、linear ridge、RBF-KRR，共 33 个冻结候选；
- 选择器：只用折内 OOF 最近训练特征距离校准阈值，在“预测候选”和封存父 K1 之间选择；
- 成本：两条分支都必须真实回放为 `2A + 2A^T`；
- 主门：90 个单元逐一通过 field、full-gradient、interior-gradient、observation 的同成本非劣和相对 Zero-K4 no-harm，共八门；平均数不能掩盖一个失败。

## 一次必要的结果前修订

第一次执行在生成正式结果前 fail closed：我把九维 `beta=0` 误认成父 `always-K1`。诊断证明两者最大场差为 `0.210861`、残差差为 `0.214274`。

原因是父 K1 已经包含四维 spatial-beta 修正；九维 `beta=0` 会把这部分也删掉。修订只把回退改成封存的四维 spatial beta 加五个零，并要求逐数组复现父 K1。模型、折分、阈值、八门和成本都没有改变。修订后独立复算的父字段差仅约 `2.84e-15`，身份闭合。

两次错误身份执行都没有创建正式输出，也没有看到模型成绩；它们不能作为科学结果。

## 正式结果

### 1. 最佳是线性，不是 RBF

| 方法 | 严格通过 | 说明 |
|---|---:|---|
| 父 always-K1 | 78 / 90 | 封存同成本候选 |
| ungated mean | 74 / 90 | 直接预测 |
| ungated linear ridge | 80 / 90 | 最佳单一预测器 |
| ungated RBF-KRR | 79 / 90 | 没有超过线性 |
| selected mean | 74 / 90 | 置信回退没有净提升 |
| selected linear ridge | **81 / 90** | 全部方法中最好 |
| selected RBF-KRR | 79 / 90 | 选择器没有提升 |

这说明 observation 中确实存在可利用的低复杂度信号：线性模型净修复了部分父 K1 失败，而且没有被 RBF 的非线性复杂度反超。但 81/90 仍远离全单元门。

### 2. 平均质量改善，尾部仍失败

最佳 selected linear 的 maximum-gate 分布为：

- mean：`-0.05386`
- p50：`-0.06344`
- p90-higher：`0.005757`
- worst：`0.071484`

负值表示有安全余量，正值表示至少一道门越线。父 K1 的 p50 是 `-0.04173`，因此最佳策略的典型帧确实更好；但 p90 和 worst 仍为正，严格尾部没有消失。

逐几何通过数为：F12 `30/30`、F15 `26/30`、F30 `25/30`。9 个失败全部来自内部梯度：5 个没有达到同成本 Zero-K2 非劣，4 个超过 Zero-K4 的 `1.01` no-harm 线，其他七门没有产生这 9 个失败。

### 3. 置信度没有识别风险尾部

线性策略接受 `69/90`，三个几何各接受 23 帧。但 9 个最终失败全部被选择器接受，没有一个被回退挡住。最近训练特征距离能帮助少量单元，却没有成为可靠风险量。

更关键的是，事后只做候选集合上限检查：

- 真值神谕在父 K1 与线性候选之间逐单元挑选，最多 `82/90`；
- 真值神谕在父 K1、mean、linear、RBF 四个候选中挑选，最多 `83/90`；
- 仍有 7 个共同失败：F30 4 个、F15 3 个、F12 0 个。

所以即使置信分类器完美，也无法把当前候选集合变成 90/90。主要瓶颈不是“门没学会”，而是七个尾部根本没有可选的通过候选。

## 独立复算

独立 validator 没有导入 v95 模型模块或正式 runner，重新构建九维物理上下文、81/177 维特征、全部折分、33 个候选、阈值和 540 次精确回放：

- 状态：`PASS_INDEPENDENT_RECOMPUTATION_PHYSICAL_BALL_OBSERVATION_SELECTOR_V95`
- selection rows 最大差：`0`
- q、field、residual、metrics、gates 最大差：全部 `0`
- exact receipt 失败：`0`
- heldout label mutation 输出差：`0`
- 重生成 observation 与封存 observation 的最大差：`2.95e-14`，在冻结 float64 容差内。

上游仍共享 pre-v95 物理与回归内核；进程确实读取了完整 q 文件，因此只证明 API 级 heldout mutation noninterference，不声称 process-level never-read。

## 成功了什么，失败了什么

成功的是：

- 证明 observation 对九维物理球系数存在真实线性可预测信号；
- 同成本严格通过从 78 提高到 81，典型 maximum-gate 余量也改善；
- 用神谕候选上限把“继续调置信分类器”排除掉；
- 所有正式预测、分支、成本和结果均由独立实现逐项复算。

失败的是：

- 没有任何冻结小模型策略达到 90/90；
- 最近距离置信度没有挡住九个失败；
- 当前四个候选即使真值神谕选择也只有 83/90；
- 没有授权外门、wall/RSS、GPU、大网络或论文成功主张。

## 策略调整

固定九维坐标上的小模型与“只修置信分类器”到此关闭。下一门不是换更大的 RBF、MLP、FNO 或 U-Net，而是先冻结一个**observation-adaptive direction capacity** 诊断：

1. 新方向只能由部署可见 residual 与 known geometry 生成，真值不能定义方向；
2. 必须精确嵌套现有九维物理球，避免丢掉已经通过的单元；
3. 先只问七个共同失败能否在 truth-aware 系数搜索下被修复，同时全 90 单元保持八门；
4. 在线 exact 预算仍为 `2A + 2A^T`，不能靠多做一次 A 或 A^T 买结果；
5. 容量达不到 90/90 就立即关线；只有容量通过才训练新的小 predictor。

这一步仍适合本机 CPU，当前不租 GPU。

## 声明边界

本结果是已开封 Case 6、noise-free known-geometry straight-ray proxy 上的 post-open 开发证据。它不是神经算子优势、外部泛化、曲折光线、真实 BOST、wall/RSS 加速或论文突破。当前仍为：

- `algorithm_breakthrough=false`
- `paper_success=false`
- `external_generalization=false`
- `resource_advantage=false`
- `real_bost=false`

## 公开配套

- 脱敏摘要：`docs/nine_view_v94_case6_physical_ball_observation_predictor_v95_public_summary.json`
- 可视化：`assets/nine_view_v94_case6_physical_ball_observation_predictor_v95.png`
