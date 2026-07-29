# BLASTNet H2-air phi=0.5 六维连续 oracle v53：无结论与 fixed-step 体素切换诊断

日期：2026-07-30

数据角色：已打开的 BLASTNet phi=0.5、snapshot 2，仅作 post-open 表示与数值机理诊断

正式状态：`INCONCLUSIVE_CONTINUOUS_SIX_SPACE_ORACLE_S2_V53`

突破状态：`algorithm_breakthrough=false`

## 一句话结论

正式 v53 的 12 起点连续约束 oracle 经第二套程序完整重放，但没有得到可用于有界负
结论的终点，也没有找到完整门 witness；随后 12 起点 Powell 和 1,024 点 Sobol
后验探测同样没有找到正见证。更重要的是，双侧扰动定位到当前 fixed-step 曲线射线
积分中的单个插值单元切换：在 `198,912` 个采样位置中，只需一个 midpoint 从相邻
体素跨界，就能让 objective 单侧跳变约 `4.07e-5`。因此当前 smooth AD/KKT 不能
单独裁决连续六维空间。

这不是连续六维无解证明，也不是算法成功。科学决策是停止给固定六方向壳层继续增加
通用光滑优化预算，转而把更有表达力、只看部署可见 observation/BP 的 warm
initializer 作为一个全新的、需重新冻结合同的假设。

## 1. 正式 v53 做了什么

v52 只在六维真值可行域里评分了固定候选表。v53 进一步在同一个六方向空间中运行
12 个事前冻结起点，每个起点最多 256 次 exact curved request；objective 是精确
curved observation 相对误差平方，field 和 gradient 作为真值约束。

runner 与不导入正式优化器的 validator 都完整执行请求序列并重算终点、KKT、指标和
判门：

```text
independent validation   PASS
frozen starts            12
exact curved F           3,088
reverse-equivalent       3,070
complete-gate passes     0
robust passes            0
```

正式请求中最好的 minimum-gate-margin 点是：

```text
field / Direct-K4        0.982639
gradient / Direct-K4     1.010000
observation / Direct-K4  1.033843
```

但是 11 个起点耗尽 256 请求预算；唯一由 SciPy 报告成功的起点也没有通过冻结 KKT
stationarity 门。并且不是全部 12 个终点都满足有界负结论条件，所以正式状态只能是：

```text
INCONCLUSIVE_CONTINUOUS_SIX_SPACE_ORACLE_S2_V53
```

不能把 `0 pass` 改写成“六维空间无解”。

## 2. 为什么 smooth KKT 在这里不够可信

后验诊断把 gradient 真值约束的椭球边界精确参数化成五维球面，在终点沿球面最速下降
方向做双侧扰动。除 `N=96` 外，积分步数 `48, 64, 80, 112, 128, 160, 192,
256, 384, 512` 的 autograd directional derivative 与 central difference
相对误差都在约 `1e-6` 以下；`N=96` 却出现：

```text
autograd directional derivative  -1.176097e-4
central directional derivative    2.035896e+2
relative derivative error         1.000001
positive-side objective jump       4.071792e-5
negative-side objective change     1.176280e-11
```

这不是笼统的“浮点误差”。程序记录每次三线性插值使用的 lower cell：

```text
N=96 value/gradient calls          576
N=96 sample positions              198,912
positive-side changed cells        1
negative-side changed cells        0
changed location                   view 0, step 77, midpoint, ray 259
cell transition                    [26,19,41] -> [26,19,40]
raw z                              41.000000000011 -> 40.999999995919
```

把同一个 N=96 终点改用 `N=128` forward 时，该局部双侧扰动没有 cell switch，导数
恢复一致；但用 `N=128` 重新优化后，新终点又在另一个位置出现一个单侧 cell switch。

**解释：**当前“对 field 做三线性插值，再用其分段常数空间梯度推进 fixed-step
ray”的离散 forward 是 piecewise smooth。改变积分步数可以移动切换面，但没有消除
这种结构。优化器可能被吸到切换面附近，所以单侧 AD 与光滑 KKT 不是足够的最终
裁判。

## 3. 不依赖梯度的正见证搜索

发现不连续性后，没有用它替失败路线找借口，而是保留同一 forward 和同一完整门，
改用只负责寻找正 witness 的 derivative-free 检查。

### 12 起点 Powell

每个正式起点先投到精确 gradient 椭球边界，再用确定性五维 gnomonic chart 做
Powell 搜索：

```text
starts                    12
exact curved F             2,973
SciPy success              2 / 12
complete-gate witnesses    0
best field / K4            0.982228
best gradient / K4         1.010000
best observation / K4      1.033484
```

它相对正式最佳点有小幅 improvement，但 observation 仍比 `1.01` 门高约 2.35 个
百分点。

### 固定种子 Sobol 球面探测

使用 seed `20260730` 在完整五维球面生成 `1,024` 个 Sobol 点；其中 `434` 个通过
box/真值可行检查并执行 exact curved F，`590` 个在 F 前拒绝：

```text
complete-gate witnesses    0
best observation / K4      1.107221
```

Sobol 粗全局点明显差于 Powell 的窄局部盆地。两项检查都只支持“没有找到正见证”，
不支持全局不存在性。

## 4. 这次真正排除了什么

1. 不能继续把“增加 SLSQP/CG 请求数”当作高价值下一步。
2. 不能用当前 fixed-step curved forward 的 smooth KKT 单独证明局部最优。
3. 不能训练一个只预测六个旧方向系数的网络，因为当前仍没有合格标签族。
4. 不能把 post-open 一快照结果写成外部泛化、真实 BOST、速度或论文成功。

同时，它没有排除更有表达力、部署时只读 observation 的三维 warm initializer。

## 5. 下一项最小科学实验

下一假设不再输出六个旧系数，而是复用已经存在的 strict dual-range 外壳：

```text
deployment-visible y
  -> small dual proposal z_theta(y)
  -> exact A^T lift
  -> observable-only alpha
  -> unchanged CGLS K1
```

接受分支目标账为 `2A + 2A^T`。训练只在 fit trajectories 使用 truth；部署输入不含
truth；不通过当前 fixed-step curved forward 反向传播。必须与 Zero/BP/CGLS、
PCGLS、dual-ridge、旧 DCT-MLP 和 direct-field control 使用相同 inverse、相同
solver、相同 matched-accuracy 与完整 wall/RSS 口径。

这是一项新表示假设，不是 v53 已经“授权成功”。先冻结最小模型、数据角色、loss、
checkpoint、种子和失败动作，再运行 fit/validation；任何轨迹的 field、gradient 或
observation harm 越门都记失败，不靠临时扩大网络挽救。

## 6. 证据边界

```text
formal_v53_independently_replayed=true
formal_v53_outcome=INCONCLUSIVE
postopen_cell_switch_diagnostic_independently_validated=false
powell_positive_witness_found=false
sobol_positive_witness_found=false
continuous_six_space_nonexistence_proven=false
global_optimality_proven=false
six_coefficient_selector_training_authorized=false
new_initializer_hypothesis_may_be_preregistered=true
matched_accuracy=false
speedup=false
external_generalization=false
real_bost=false
paper_success=false
algorithm_breakthrough=false
```

公开图：
`assets/blastnet_h2air_phi05_six_space_continuous_oracle_v53.png`

脱敏机器摘要：
`docs/blastnet_h2air_phi05_six_space_continuous_oracle_v53_public_summary.json`
