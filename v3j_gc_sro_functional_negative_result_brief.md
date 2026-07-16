# v3j GC-SRO 几何机制功能实验

> **停止判决**：当前 GC-SRO 的“正确几何”不能在未见 validation 布局上胜过 static、K-cardinality 或确定性 shuffled geometry。通用频谱残差分支有效，但采集几何没有产生可用的独立收益。停止当前结构的扩参和论文主张。

## 严格比较了什么

- 数据：v3i 的 328 个唯一场、28 布局、42 通道私有数据。
- 底座：同一个已锁定 240-epoch FNO checkpoint，预计算输出后全程冻结。
- 方法：`static`、`K-cardinality`、`shuffled geometry`、`correct geometry`；每个都是 45,226 总参数、1,023 可训练参数。
- 预算：4 方法 × 3 种子 × 24 epochs；同随机种子下初始化和 DataLoader 顺序匹配。
- 选择：每个方法/种子仅用 4 个未见 validation 布局选最佳 epoch；12 次训练全部结束后才统一读 test/OOD。
- 错误几何：在每个 geometry partition 内做无固定点循环错配，28 个布局变化率 100%，与 batch 顺序无关。

## 主结果

### 通用 adapter 有效

validation 平均 field relative L2：

| 方法 | L2 |
|---|---:|
| locked fixed-layout FNO | `0.254784` |
| static adapter | `0.245527` |
| K-cardinality | `0.245527` |
| shuffled geometry | `0.245524` |
| correct geometry | `0.245525` |

static adapter 相对 locked FNO 改善 `+3.6336%`。因此负结果不是“残差分支什么都没学到”，而是“它学到的几乎全是通用修正”。

### 正确几何未通过机制闸门

| correct geometry 相对 | 平均场收益 | field-cluster 95% CI | 正向种子 | audit 收益 |
|---|---:|---:|---:|---:|
| static | `-0.00019%` | `[-0.00297%, +0.00236%]` | `1/3` | `-0.00045%` |
| K-cardinality | `-0.00008%` | `[-0.00284%, +0.00246%]` | `1/3` | `-0.00006%` |
| shuffled geometry | `-0.01628%` | `[-0.04387%, +0.01128%]` | `1/3` | `-0.02376%` |

三个预注册比较全部失败。不允许使用“geometry-aware 提升”、“优于 FNO”或“变几何泛化”表述。

## 失效在哪一层

在已训练的 `correct geometry` 模型中保持所有权重不变，只把 validation 几何替换为错配几何：

| 量 | 三种子/全场平均 |
|---|---:|
| embedding swap L2 | `0.01898` |
| modulation swap L2 | `0.00942` |
| correction 相对变化 | `0.17848%` |
| 正确 descriptor 的场误差收益 | `-0.00015%` |
| 正向种子 | `1/3` |

几何已经在 set encoder 中被区分，但当前只用一个全局向量调制 6 个 adapter channels；空间分支本身只看 base prediction、ridge/support/K 和坐标，不看逐视角 ray feature map。因此布局信息在“embedding → spatial correction”之间被强烈衰减。

这与 [FiLM](https://aaai.org/papers/11671-film-visual-reasoning-with-a-general-conditioning-layer/) 的角色边界一致：FiLM 是通用 feature-wise affine conditioning，但它不保证一个全局条件能解释 BOST 中随位置变化的 ray coverage。[Deep Sets](https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html) 保证/支持集合不变表示，也不意味该表示会被下游任务有效使用。[VIDON](https://arxiv.org/abs/2205.11404) 则更贴近本问题：它把可变传感器位置与观测值联合编码，而不是只生成一个脱离观测内容的全局调制向量。

## 下一步决策树

### v3k-A：先查数据反事实监督（最低风险）

不改模型容量，将每个 train field 在每个 epoch 确定性重采样到多个布局，或为每场生成 2–4 个布局对。同一 field 与噪声下的布局变化才提供直接 counterfactual supervision。

- 若 correct 仍不胜 shuffled：主要是结构失效，进入 v3k-B。
- 若只在多布局对下胜出：主要是 v3i one-field/one-geometry 监督不足，保留当前小结构。

### v3k-B：再查空间条件化（中风险）

不先增加更多 hidden channels，而是让每个体素的修正看到“逐视角反投影 + 角度”：

1. 用共享轻量 encoder 得到每视角体素特征。
2. 用角度/布局条件权重汇总 active views，生成 mean/variance/disagreement 体场。
3. 只对汇总后的少量体场做低频谱残差，避免恢复 v3e 中昂贵的完整 per-view 3D trunk。

该路线必须继续保留 static、shuffled、参数匹配与 wall-time controls。

## 现在不应该做的事

- 不把 descriptor hidden 从 16 扩到 64/128 再广搜。
- 不将 static adapter 的 `+3.63%` 写成 geometry-aware 收益。
- 不打开 blind final，不训练论文级 matched variable-geometry FNO。
- 不在没有真实 camera/ray calibration 时声称可用于课题组 TDBOST 数据。

## 请师兄判断

1. 是否同意先做“同场多布局重采样”，用一次有界实验区分数据与结构原因？
2. 组内的每个相机是否有可构造体素级 ray/backprojection feature 的标定矩阵？
3. 如果真实布局不会变化，是否应停止 geometry 主创新，转向“强 static operator → NeRIF/TDBOST warm start”？

## 复现

- 运行：`python demo_t16_operator/run_v3j_gc_sro_functional_pilot.py`
- 验证：`python demo_t16_operator/validate_v3j_gc_sro_functional_results.py`
- 公开包含 288 条曲线、4,920 条逐场指标、24 组机制比较、120 条同模型 swap 指标和 checksum。
- 12 个 checkpoint 只留本机；公开页无 NPZ、PT/PTH、PDF 或受限材料。
