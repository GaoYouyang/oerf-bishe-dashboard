# v3g DeepONet 容量审计与下一步算法决策

> 面向何远哲师兄的阶段性审核稿。结论来自本机 Apple Silicon MPS 上的合成 K=6 development protocol；不是组内真实 BOST 证据，也不是论文 superiority 结论。

## 一句话结论

在预先限定的 `8 架构 × 3 学习率 × 3 种子 = 72` 次 validation-only screen 中，`rank 64 / pool 4×4×4 / lr 0.002` 是 24-epoch 冠军；但把它按三种优化协议延长到 240 epochs 后，最优验证误差 `0.176094` 仍略差于 v3f 的 `rank 48` 参考 DeepONet `0.175725`，也明显差于 FNO `0.094139`。因此不把 rank 64 升格为强基线，暂时冻结 v3f rank 48，并把自研重心放在 **FNO 三维空间主干 + acquisition-geometry conditioner**，而不是继续扩大纯 pooled DeepONet。

## 1. 这轮到底排除了什么质疑

v3f 的 DeepONet 只有一个 `rank 48 / pool 4×4×4` 架构。审稿人完全可能问：是不是 DeepONet 被故意配弱？v3g 给出一个有边界、可复查的回答：

- 输入、ridge anchor、loss、K=6 相机预算、train/validation split、三个种子和实际 batch order 与 FNO 协议一致。
- 扫描 `rank 32/48/64/96`，同时改变三维池化分配，检查 branch 信息压缩是否偏向错误空间轴。
- 每个可训练变体都在 `lr = 0.001 / 0.002 / 0.004` 下固定训练 24 epochs，不早停。
- 参数上限预先锁为参考 DeepONet 的 `1.5×`。`pool 8×4×4` 和 `4×8×8` 分别有 `90,273`、`172,193` 参数，因越界在训练前排除。
- 只允许三种子 mean validation 最低的一个 cell 进入 240-epoch 延长；dev2 与 Q_audit 在 selection commit 之后才读取。
- 全部筛选耗时 `153.69 s`；其中 `147.28 s` 是未进入最终曲线的搜索成本，不能藏在“训练只用了 20 秒”之外。

这证明当前 DeepONet 已获得最低限度的架构与学习率补强，但不证明所有 DeepONet 设计都弱。

## 2. 关键数字

### 2.1 24-epoch screen 的前三名非常接近

| 排名 | 架构 | lr | 三种子 mean best validation L2 | 参数 |
|---:|---|---:|---:|---:|
| 1 | rank 64, pool 4×4×4 | 0.002 | 0.185196 | 51,393 |
| 2 | rank 48, pool 2×8×4 | 0.002 | 0.185217 | 49,313 |
| 3 | rank 96, pool 4×4×4 | 0.002 | 0.185231 | 55,553 |
| 6 | v3f reference: rank 48, pool 4×4×4 | 0.002 | 0.186146 | 49,313 |

冠军与第二名只差约 `0.011%` 相对误差，属于非常脆弱的短程排序。冠军相对 reference 在 24 epochs 好约 `0.51%`，但这个优势没有延续到 240 epochs。

### 2.2 240-epoch 长程结果翻转

| 方法 / 协议 | mean validation L2 | 平均累计训练时间 / seed |
|---|---:|---:|
| v3g rank 64 + carry Adam + long cosine | 0.176094 | 20.34 s |
| v3f rank 48 + carry Adam + restart cosine | **0.175725** | 20.51 s |
| FNO + carry Adam + restart cosine | **0.094139** | 46.46 s |

- rank 64 相对 rank 48 **差 0.210%**，因此不能替换 reference。
- FNO 最终验证误差相对 rank 64 低约 `46.54%`。
- rank 64 的长程冠军是 `carry Adam + long cosine`；这与 v3f rank 48、FNO 的 `carry Adam + restart cosine` 冠军不同，说明 optimizer schedule 与架构存在交互。
- rank 64 的长程冠军在当前 plateau rule 下于 endpoint 96 起进入 plateau，但 plateau 不等于质量足够。

### 2.3 冻结后复用 dev2 也不支持替换

rank 64 相对 v3f rank 48：

- domain-equal mean field superiority：`-0.160%`
- field-cluster 95% CI：`[-0.339%, +0.006%]`
- p10：`-0.766%`
- 伤害率（field superiority < -1%）：`8.59%`
- 三个模型种子均值：`-0.296% / -0.343% / +0.136%`

CI 跨 0，四域中三域均值为负，三个种子中两个为负；development gate 明确失败。这里的 CI 先把三个模型种子折叠到每个物理场，再按 128 个场分层重采样，不能冒充 seed-level 不确定性。

## 3. 对初学者最重要的物理与模型解释

### 3.1 rank 在 DeepONet 里是什么

固定网格 DeepONet 写成

```text
x_hat(xi) = x_ridge(xi) + support(xi) * sum_k b_k(y, g) t_k(xi)
```

`branch` 从多视角观测 `y` 与几何元数据 `g` 产生系数 `b_k`；`trunk` 从查询坐标 `xi=(z,y,x)` 产生空间基函数 `t_k`。`rank` 是可组合基函数的数量。提高 rank 只增加可分离表示容量，不能自动修复 branch 在池化阶段丢掉的局部 ray 信息，也不能获得 FNO 的三维空间混合能力。

### 3.2 pool shape 为什么不是普通超参数

原始场大小是 `8×16×16`。`pool 4×4×4` 把每个 camera lift 和 ridge 压成 64 个区域均值；`pool 8×2×2` 保留全部 depth bins，却更强地压缩横向结构。它隐含了“哪个空间轴值得保留”的归纳偏置。v3g 没看到某一种池化分配形成足够稳定的优势，所以不能用几何故事反向解释微小差异。

### 3.3 为什么 24-epoch 冠军会在 240 epochs 输掉

短程验证同时混合了三件事：最终可达到的误差、早期优化速度、scheduler 与架构的交互。rank 64 可能只是前 24 epochs 收敛略快；rank 48 在 restart-cosine 延长中最终更低。以后不能用廉价短程 screen 的单一冠军直接宣称架构更好，至少要报告：

1. screen 冠军与次优的差距；
2. 排名在多个 epoch 是否稳定；
3. 只延长冠军造成的 survivor bias；
4. 搜索总成本，而不仅是最终模型成本。

## 4. 现在怎样停止基线调参

本轮按预声明规则只延长短程冠军，因此**不能**说 `rank 48` 在所有被列出的架构中全局最优；第二名 `pool 2×8×4` 和第三名 `rank 96` 没有 240-epoch 曲线。这是明确的 threat to validity。

建议把决策拆成两档：

- **毕设执行档**：冻结 v3f rank 48 作为 DeepONet control。它已有学习率、三优化协议、三种子、240 epochs、成本和 dev2 配对统计；继续追逐 0.01% 的短程排名会吞掉自研模型时间。
- **投稿补充档**：只有师兄明确认为 reviewer 会要求时，再一次性预注册 top-3 multi-fidelity extension；不得看 dev2 选择 finalist，不得在结果不满意后继续加第四个结构。

冻结不等于宣称它是世界最强 DeepONet，而是给 baseline tuning 设定可解释的预算停止点。

## 5. 对自有算法的直接启发

### 保留什么

- 保留 FNO 作为三维空间混合主干；它在当前 synthetic K=6 上是稳定强 control。
- 保留 DeepONet / set encoder 的“观测与查询分开”思想，但把 branch 降级为 acquisition conditioner，而不是主重建器。
- 保留 ridge residual 与 support mask，让网络只学习传统重建的结构化修正。

### 不把什么包装成创新

- 提高 rank、改变 pooling、加普通 branch/trunk，不是论文贡献。
- 低秩谱核、LoRA、frequency adapter 和一般 geometry-informed operator 已有最近邻。
- “比未充分训练的 DeepONet 好”不能作为自有方法主结论。

### 下一版 GC-SRO 的最小模型

1. `FNO trunk`：继续负责三维场的局部与全局空间混合。
2. `acquisition branch`：共享编码每个 camera 的 angle、mask、ray coverage、calibration/noise statistics。
3. `permutation-invariant aggregation`：相机集合求和/attention，不依赖输入顺序。
4. `zero-init modulation`：只调制少量 spectral groups；初始化时精确回退到冻结 FNO。
5. `physics residual`：输出仍限制在 support 内，并保留 used-view 与 held-out Q_audit。

最低机制对照必须同场出现：

| ID | 对照 | 要排除的解释 |
|---|---|---|
| C0 | locked FNO | 没有新分支时的强基线 |
| C1 | parameter-matched wider FNO | 收益只是多参数 |
| C2 | static frequency adapter | 收益只是 adapter |
| C3 | mask-only conditioner | 收益只来自缺相机模式 |
| C4 | shuffled geometry | 模型没有真的使用正确几何 |
| M1 | correct acquisition geometry | 唯一保留的机制候选 |

主假设不是“M1 比 FNO 平均好”，而是：**M1 同时胜 C1-C4，且在 geometry drift / missing-camera / OOD family 上保持 field CI、p10、harm 与 Q_audit 门槛。**

## 6. Gao Youyang 下一周应亲手完成

1. 从 `v3g_variant_manifest.csv` 手算 reference、rank 64 和两个排除项的参数量。
2. 从 `v3g_screen_summary.csv` 复画 8 架构 × 3 学习率图，并解释为什么 `0.011%` 的冠军差距不稳定。
3. 从 `v3g_validation_comparison.csv` 复画 24/60/120/180/240 曲线，不看网页现成图。
4. 逐行读 `GridDeepONetResidual.forward`，为每个 tensor 写 shape，解释池化在哪一步丢信息。
5. 写一个最小 1D toy：固定 trunk，只比较 rank 16/32/64 的早期与长程排序是否翻转。
6. 画 GC-SRO 接口图，只写输入、输出和 control，不先写大模型。

达到“能向师兄口述”的标准：能在三分钟内说明为什么 v3g 是有价值的失败、为什么不继续盲目扩 DeepONet、为什么 acquisition conditioner 仍有独立可检验的物理假设。

## 7. 请师兄优先回答

1. 真实实验中 camera intrinsics/extrinsics、ray paths、mask、标定误差是否跨 case 变化？如果完全固定，geometry conditioner 的论文价值会明显下降。
2. 组内能否先提供一个脱敏的 `displacement + mask + camera/ray geometry + units` 样例，只做接口健康检查？
3. 是否认可把 v3f rank 48 冻结为毕设 DeepONet control，停止继续追逐 pooled DeepONet 的微小 validation 排名？
4. 若面向投稿，是否需要一次性追加 top-3 long-horizon audit，还是优先把算力放到 GC-SRO controls 与真实 geometry？
5. 新的 untouched blind final 应由谁生成、何时封存、真实/合成各需要多少独立 flow cases？

## 8. 可核验入口

- [公开实验图](./demo_t16_operator/results/v3g_deeponet_capacity_audit/t16_v3g_deeponet_capacity_audit.png)
- [72-run screen](./demo_t16_operator/results/v3g_deeponet_capacity_audit/v3g_screen.csv)
- [架构与参数清单](./demo_t16_operator/results/v3g_deeponet_capacity_audit/v3g_variant_manifest.csv)
- [长程 validation 对照](./demo_t16_operator/results/v3g_deeponet_capacity_audit/v3g_validation_comparison.csv)
- [冻结后 dev2 配对统计](./demo_t16_operator/results/v3g_deeponet_capacity_audit/v3g_cross_baseline_pairwise.csv)
- [selection commit](./demo_t16_operator/results/v3g_deeponet_capacity_audit/v3g_selection_commit.json)
- [机器报告](./demo_t16_operator/results/v3g_deeponet_capacity_audit/v3g_deeponet_capacity_report.json)
- [训练脚本](./demo_t16_operator/run_v3g_deeponet_capacity_audit.py)
- [独立 validator](./demo_t16_operator/validate_v3g_deeponet_capacity_results.py)

公开目录不含 `.pt/.pth` 权重、`.npz` 数据或任何受限 PDF；这些材料也不会推送到 GitHub Pages。
