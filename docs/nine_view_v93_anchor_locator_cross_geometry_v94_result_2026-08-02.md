# v94：F30 局部缺口没有在 F12 形成普遍净伤害，关闭局部窗口修补路线

## 一句话结论

v94 没有得到新算法成功，而是得到一个会改变下一步的可靠负结果：v93 在 F30+/12 发现的局部梯度缺口，**没有在 F12+ 的 30 帧中形成任何一帧净内部梯度伤害**。30 帧的候选场都严格优于同成本 Zero-CGLS K2，因此不应继续围绕 anchor 建局部窗口或训练局部 predictor。

科学状态为 `FAIL_LOCAL_MECHANISM_NOT_PREVALENT_V94`，`algorithm_breakthrough=false`。

## 为什么要做 v94

v93 对 F30+/12 的分解显示，候选场相对 Zero-K2 多出的内部梯度误差高度集中、以 z 负侧低频结构为主。预注册的 initial-normal locator 没有找到它，但一个事后 control `anchor_gradient_energy` 表现较好。

这只能产生假设，不能把 control 当作通过。真正要问的是：

> F30 看到的局部净伤害，是否会在另一个几何 F12 中稳定复现，值得为它设计一个可观测局部修补器？

如果不先回答这一点，继续做局部网络会把一个单帧特例误当成普遍机制。

## 结果前冻结的门

v94 只用已经开封的 Case 6 做 post-open 机制诊断，并把 F30+/12 的事后 anchor 线索固定为待确认假设。F12+ 的 30 帧用于确认，F15+ 不生成、也不评分局部 target / saliency 图。

一帧只有同时满足以下条件，才算“值得局部修补”的 eligible frame：

1. 候选场相对 Zero-K2 的内部梯度平方误差比大于 `1.01`；
2. 候选减 Zero-K2 的总 signed excess 为实质正值；
3. 正误差质量非退化；
4. 最高 10% 体素承载至少 50% 正误差；
5. 解释 80% 正误差所需体素不超过 30%。

路线继续还要求至少 `8/30` 帧 eligible，且三个连续十帧区间各至少出现一帧。门槛在读取 F12 局部图前冻结。

## 正式结果

### 1. 普遍性门为 0/30

- eligible materially harmful localized frames：`0 / 30`
- 三个十帧区间：`0 / 0 / 0`
- 候选 / Zero-K2 内部梯度误差比：
  - minimum：`0.845318`
  - median：`0.917441`
  - p90-higher：`0.942557`
  - worst：`0.979357`
- 误差比大于 `1`：`0 / 30`
- 误差比大于 `1.01`：`0 / 30`
- 总 signed excess 为正：`0 / 30`

这意味着 F12 的 30 帧中，v92 truth-aware capacity witness 在内部梯度上全部优于同成本 Zero-K2。这里没有需要 anchor 局部修补器挽救的净伤害。

### 2. “局部有正误差”不等于“总体更差”

30 帧里有 `22` 帧的 componentwise positive map 呈现空间集中，但它们的总 signed excess 仍全部为负。换句话说，局部确实能找到“候选在某些体素更差”的区域，可候选在其他区域获得了更大的改善，最终整体内部梯度误差更低。

这一区分很关键：如果只盯热图的红色局部，容易训练一个修补器去修复并不存在的总体问题，还可能破坏已经取得的净收益。

## 独立复算

独立 validator 没有导入正式 v94 runner 或 v94 数值 core，重新构建判决并核对局部图、标量和门：

- 状态：`PASS_INDEPENDENT_RECOMPUTATION_ANCHOR_LOCATOR_V94`
- 局部图最大绝对差：`0`
- 判决最大差：`0`
- 标量最大绝对差：`1.11e-16`
- F15 局部 target / saliency 图生成或评分：`false`

上游仍共享冻结的 pre-v94 physics 与数据加载内核，因此这不是端到端物理独立性证明。

## 成本账怎么解释

本轮执行核对了完整诊断账：所有几何 setup 为 `960A + 480A^T`，F12 候选 shell 为 `60A + 60A^T`，实际总账为 `1020A + 540A^T`。如果从一开始只有 F12 能力，反事实 setup + shell 为 `380A + 220A^T`；父结果物化后，局部 map 的边际 exact 调用为 `0A + 0A^T`。

这些数字只说明诊断实际花了多少算子调用，**不是部署成本，也不是 wall-time 或内存加速结果**。

## 成功了什么，失败了什么

成功的是：

- 用跨几何确认排除了“F30 单帧局部缺口是普遍局部伤害机制”的错误解释；
- 阻止后续把算力投入局部窗口、局部 U-Net 或 anchor predictor；
- F15 局部图继续保留，没有为挽救假设而追加开封；
- 发现 F12 的 30 帧中，truth-aware witness 对 Zero-K2 的内部梯度是严格净改善。

失败的是：

- anchor 局部窗口修补路线没有获得普遍性支持；
- 没有产生可部署 initializer；
- 没有授权大网络、资源测试或论文成功主张。

## 策略调整

局部窗口路线到此关闭。下一条更有价值的问题是：

> v91 已证明存在的九维 physical-ball witness，能否只由 observation 与 known geometry 预测，并用校准置信度对少数异常帧 fail closed 回退？

这比继续添加空间基函数更贴近最终 C 路线，因为它直接研究“已有低调用 witness 是否可部署预测”，同时把 rare outlier 当成安全回退问题，而不是假设每帧都有同一种局部缺口。

## 声明边界

本结果是已开封 Case 6、straight-ray proxy、truth-aware witness 上的 post-open 路线选择证据。它不是神经算子、外部泛化、曲折光线、真实 BOST、速度/内存优势或论文突破。当前仍为：

- `algorithm_breakthrough=false`
- `paper_success=false`
- `external_generalization=false`
- `real_bost=false`

## 公开配套

- 脱敏摘要：`docs/nine_view_v93_anchor_locator_cross_geometry_v94_public_summary.json`
- 可视化：`assets/nine_view_v93_anchor_locator_cross_geometry_v94.png`
