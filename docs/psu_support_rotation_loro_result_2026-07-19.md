# PSU support-rotation LORO：正式结果、负证据与下一算法门

## 一句话结论

**32³ 在固定 `k=4` 上没有通过比 16³ 更好的三折安全门；但六个“rotation fold × grid”都复现了同一种半收敛：继续从 `k=4` 算到 `k=12` 时，训练观测误差继续下降，留出 rotation 的至少一项关键误差反而上升。**

这是一条可信的负证据和一个可研究的真实逆问题，不是算法胜利：

- 没有 volumetric truth，不能比较三维场 relative-L2；
- 三个 rotation block 来自同一个物理流场，不是三个独立实验重复；
- 不能说 16³ 普遍优于 32³，也不能把全部退化归因于“高频过拟合”；
- 没有训练神经算子，没有击败 FNO、DeepONet、MgNO 或 NeRIF；
- 当前最值得做的是**不访问 outer heldout 的 rotation-aware early stopping / regularization**。

机器状态：

```text
SUPPORT_ROTATION_LORO_TRAJECTORY_COMPLETE_NO_FIELD_TRUTH
```

固定 `k=4` 判决：

```text
screen_pass = false
equal-rotation mean camera-macro improvement (16 - 32) = -0.0081779103
```

## 1. 这次到底算了什么

数据是 PSU 开放的高速飞行体 BOS tomography 数据集，DOI `10.26208/1VE2-5C19`。作者脚本与逐块 manifest 已在 E70 中把九个 support views 绑定为：

| fold | 训练 rotations | 整组留出 rotation | 留出 views | 训练 / 留出 active rays |
|---|---|---|---|---:|
| A | 50°、90° | 0° | 0、1、2 | 6,991,372 / 3,637,450 |
| B | 0°、90° | 50° | 3、4、5 | 6,965,700 / 3,663,122 |
| C | 0°、50° | 90° | 6、7、8 | 7,300,572 / 3,328,250 |

每个 fold、每个 `16³/32³` 网格都从零场运行一次 float64 CGLS，固定保存：

```text
k = 0, 1, 2, 3, 4, 6, 8, 12
```

heldout observations / store 只在 solver snapshots 固定后载入。每条 trajectory 的预算是 12 次训练 forward、13 次训练 adjoint；八个 checkpoints 的评分另用 7 次全视角 forward、8 次训练 adjoint、8 次 heldout adjoint。多分辨率归因每 fold 再用 21 次 32³ full forward。

## 2. 固定 k=4 的分辨率门没有通过

以下 improvement 定义为 `16³ error - 32³ error`；正值才表示 32³ 更低。

| heldout rotation | pooled `32 - 16` | equal-camera macro `16 - 32` | worst camera `16 - 32` | p95 ratio `32/16` | 结论 |
|---:|---:|---:|---:|---:|---|
| 0° | +0.004861 | -0.003613 | +0.003643 | 1.011288 | pooled 与 macro/p95 冲突，逐相机门失败 |
| 50° | +0.012289 | -0.017657 | -0.012962 | 1.016034 | pooled、macro、worst、p95 均退化 |
| 90° | -0.005293 | -0.003264 | +0.010318 | 1.002664 | pooled 改善，但 macro 与 camera 2/p95 退化 |

九台“rotation × camera”条件中，32³ 的 camera relative-L2 改善范围为：

```text
minimum = -0.0366156
maximum = +0.0136831
```

因此六个固定门全部为 false：

- equal-rotation mean macro improvement 未达到 `+0.01`；
- 不是每个 fold pooled 都 no-harm；
- 不是每个 heldout camera 都 no-harm；
- 不是每个 fold worst camera 都 no-harm；
- 不是每个 fold group p95 都 no-harm；
- 不是每个 heldout camera p95 都 no-harm。

这不是“32³ 永远没用”。更准确的解释是：**在当前零起点、固定 CGLS4、同一 QMC16 观测合同下，增加自由度没有形成跨 rotation、跨 camera 的安全增益。**

## 3. 六个单元都出现半收敛

表中是 `k12 - k4` 的 equal-camera macro relative-L2。负值表示训练误差继续改善，正值表示留出误差恶化。

| heldout rotation | grid | train change | heldout change | screen |
|---:|---:|---:|---:|---|
| 0° | 16³ | -0.054371 | +0.018441 | positive |
| 0° | 32³ | -0.072117 | +0.005954 | positive |
| 50° | 16³ | -0.049880 | +0.128739 | positive |
| 50° | 32³ | -0.063356 | +0.063316 | positive |
| 90° | 16³ | -0.058726 | +0.069861 | positive |
| 90° | 32³ | -0.061448 | +0.022239 | positive |

机器的正式 screen 不只看 macro；它检查 pooled、macro、worst camera 与 group p95 中是否至少一项在 `k4 → k12` 恶化。六个单元都是：

```text
train improved = true
heldout any required metric worsened = true
screen_positive = true
```

这是目前最稳的研究信号。它不证明 `k=4` 是最优停止步，也不提供物理显著性；它证明**仅靠训练 residual 单调下降来决定“继续算”会系统性误导这组跨角度观测迁移。**

## 4. coarse / orthogonal 归因没有给出统一故事

对 `x32 - Ux16` 做 active-DOF Euclidean 正交投影：

\[
x_{32}-Ux_{16}=P(x_{32}-Ux_{16})+(I-P)(x_{32}-Ux_{16}).
\]

这里第二项只能叫 **fine-grid orthogonal complement**，不能叫真实高频误差。固定 `k=4` 的 heldout normalized squared-error Shapley 为：

| heldout rotation | coarse-range disagreement | orthogonal complement | net change | 方向 |
|---:|---:|---:|---:|---|
| 0° | +0.004393 | -0.003529 | +0.000864 | 两项抵消，最终轻微伤害 |
| 50° | +0.013011 | +0.002322 | +0.015333 | 两项都伤害 |
| 90° | -0.027768 | -0.004525 | -0.032293 | 两项都改善 |

负 contribution 表示降低平方误差，正 contribution 表示增加平方误差。三折方向不一致，所以现在不能说“只删 orthogonal component”或“只学 orthogonal component”就会普遍改善。更合理的结构必须依赖 geometry / rotation support，并且 fail closed。

## 5. 数值与隐私审计

正式运行：

| 项目 | 值 |
|---|---:|
| protocol commit | `5e4e523f68825406a29a3c5f05257716bb4e7751` |
| wall time | 2,884.90 s |
| process max RSS | 9,222,799,360 bytes |
| torch threads | 8 |
| public summary SHA-256 | `08b76f597b3dd8a48fb7936dfea60e553ecbf8c7032ea3bf692a865ca09c9ca8` |
| run ID | `ab994bec2beb06a2617137d753d3d107ce4e8160dffccce40c670c0f706b36e7` |

全部数值门通过：cache hash、subset adjoint、train/heldout chunk isolation、direct-vs-recurrence、projection idempotence / normal equation / Pythagorean closure、Shapley closure、调用预算、wall-time cap 与内存 cap。

独立 validator 不导入正式 runner；它重新推导 macro/worst、k=4 分辨率差、九个 camera 差、六个半收敛单元、调用预算、fold 身份与 Shapley closure，并拒绝：

- 改 summary 但不改 binding；
- 改 binding 后伪造论文/算法声明；
- 改决策数字或 checkpoint；
- 改半收敛计数；
- 塞入本地私有路径、payload 文件名、私有报告 hash 或额外文件。

```bash
PYTHONPATH=. .venv/bin/python site_tools/validate_psu_support_rotation_loro_public.py
```

当前验证结果：`PUBLIC_SUPPORT_LORO_PACKAGE_VALID`，9 项 validator tests 通过。

## 6. 下一算法：E72 nested rotation-aware stopping

### 要解决的真实问题

经典 CGLS 在训练观测上继续收敛，却把不可稳定识别的方向逐步带入重建，导致未参与求解的 camera/rotation 观测恶化。目标不是事后寻找一个漂亮的 `k`，而是设计一个**不读取 outer heldout 分数**的停止规则。

### 最小、可执行的数据结构

现有 outer trajectories 已完成。下一步只需要补三种单 rotation 训练轨迹：

```text
train rotation 0  -> score rotations 50 and 90
train rotation 50 -> score rotations 0 and 90
train rotation 90 -> score rotations 0 and 50
```

每种都跑 16³/32³、同一 checkpoints。不是 12 个新 solver，而是 **3 rotations × 2 grids = 6 条 trajectory**；同一单-rotation solver 可以评分另外两组 rotation。

### 外层不泄漏选择

对 outer fold `r_out`，剩下两个 rotations 互相作为 inner validation：

\[
\hat{k}_{-r_{out}}=
\arg\min_{k\in K}\max_{r\in R\setminus\{r_{out}\}}
\left[
E_{macro}^{(r)}(k),
E_{worst}^{(r)}(k),
Q_{0.95}^{(r)}(k)
\right].
\]

三个量必须先按各自 `k=0` 或冻结尺度无量纲化；聚合方式、tie-break 和 no-harm 阈值必须在结果前固定。建议 tie-break 取更早 `k`，并要求两条 inner direction 都不比固定 `k=4` 的 camera/p95 envelope 更差，否则回退 `k=4`。

随后只在 outer heldout rotation 上比较：

1. fixed `k=4`；
2. train-residual minimum（通常会走向 `k=12`，作为反例）；
3. L-curve / GCV / discrepancy 等经典停止；
4. nested rotation-aware selector；
5. heldout oracle `k`，仅作为不可部署上界。

### E72 通过门

先把它当一个方法候选，不当论文结论。最低通过条件：

- 三个 outer folds 的 equal-rotation mean macro 优于 fixed `k=4`；
- 每个 outer fold pooled、worst camera、group p95 均 no-harm；
- 九个 heldout cameras 和其 p95 均 no-harm；
- outer heldout 从未参与 `k` 的选择或阈值拟合；
- 报告全部 solver / scoring A、Aᵀ 调用和 wall time；
- 与 H1/TV、pyramid/coarse-to-fine 经典强基线比较后仍有增量，才进入神经 correction。

若 nested selector 失败，结果仍有价值：说明仅靠三组 view support 不足以稳定选择停止步，下一步应优先增加 flow-off/noise repeatability、H1/TV 或真实多工况数据，而不是扩大网络。

## 7. 对自有算子模型的约束

当前允许保留的候选结构是：

\[
x_f^{cand}=Ux_c+g_{safe}(\mathcal G,\rho,\text{camera tails})
P_{DC}\,\delta_\theta(Ux_c,r,\mathcal G,h).
\]

- `x_c`：nested stopping / H1 / TV 选出的经典稳定主体；
- `delta_theta`：MgNO、CNO、FNO 或坐标网络提出的 correction；
- `P_DC`：数据一致或近零空间约束；
- `g_safe`：只由训练侧 calibration 学得的 rotation/camera-tail gate；
- 任一关键门失败，输出回退 `x_c`。

本轮证据要求网络同时回答两个问题：为什么比经典 early stopping/H1/TV 多出可迁移信息，以及为什么不会在某个 camera geometry 上造成 tail harm。只赢 pooled mean 不够。

## 8. 给师兄审核的五个问题

1. 实验室实际 BOST reconstruction 是多 rotation 同一稳态流场，还是高速 4D 中每个时刻只有固定相机阵列？这决定 rotation-aware stopping 能否直接迁移。
2. 能否提供 flow-off / repeated background / repeated steady-flow 数据，用来估计 discrepancy principle 的噪声与 repeatability floor？
3. 最终任务更重视折射率/密度场本身，还是对下游 PIV compensation / flow structure 的观测一致性？
4. 真实 pipeline 中 H1、TV、pyramid 或其他 regularization 的当前强基线和默认参数是什么？
5. 可否把若干工况完全留作 final flow-regime audit，而不参与模型、停止步、阈值或网络结构选择？

## 9. 证据入口

- [机器 summary](../demo_t16_operator/results/psu_support_rotation_loro_public_v1/summary.json)
- [公开 finalization](../demo_t16_operator/results/psu_support_rotation_loro_public_v1/finalization.json)
- [独立 validator](../site_tools/validate_psu_support_rotation_loro_public.py)
- [四联图 PNG](../assets/figures/psu_support_rotation_loro_result.png)
- [四联图 SVG](../assets/figures/psu_support_rotation_loro_result.svg)
- [冻结协议](psu_support_rotation_loro_protocol_2026-07-19.md)
- [E70 输入身份/cache 前置审计](psu_support_rotation_loro_preflight_2026-07-19.md)

## 10. 仍然不能声称什么

- 不能声称 16³ 的真实三维场更准确；
- 不能声称 32³ 普遍过拟合或 fine orthogonal complement 普遍有害；
- 不能把三折当三次独立物理复现实验，也不能给统计置信区间；
- 不能把 image-space relative-L2 写成 field relative-L2；
- 不能声称 nested stopping 已成功，它还是下一项待证伪实验；
- 不能声称神经算子优于经典方法、跨流态泛化或已经达到论文发表门槛。
