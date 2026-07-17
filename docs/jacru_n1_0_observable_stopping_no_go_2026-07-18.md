# JACRU-N1.0：只看可观测残差的早停上界判决

**日期：** 2026-07-18  
**总状态：** `N1_0_OBSERVABLE_DISCREPANCY_STOPPING_NO_GO`  
**证据边界：** 已打开的 12³ synthetic T0；复用冻结的 M2.7 K=0--10 轨迹；没有重新训练，没有 fresh/final，没有真实 flow-off 标定。exact camera-block 仍是 setup 不计入预算的 `1001F-equiv` oracle。

## 1. 先说结论

N1.0 没有得到可授权的早停规则，但它把 M2.8 留下的疑问进一步封住了：

1. 我们冻结并比较了 37 个 stopping specs，其中 26 个是只依赖 residual 的非固定迭代候选；两种网络、180 条方法/种子/样本轨迹共生成 6,660 行选择结果。
2. JACRU 的 26 个可观测候选中，6 个保护了 field tail，11 个保护了 continuous-clean-target consistency，**联合安全候选为 0**。该 clean 指标仍使用同一个 voxel `A`，不是 independent renderer。
3. pooled CNN 的 26 个可观测候选中，0 个保护 field tail，8 个保护 continuous-clean-target consistency，**联合安全候选仍为 0**。
4. 早停在 K 约 1--2 时可以保住部分界面场，但 continuous-clean-target residual 仍过大；继续到 K 约 4--6 后该一致性变好，原先的尾部伤害重新出现。
5. 因此，当前冲突不是找一个更聪明的标量阈值就能解决。下一步需要独立 flow-off 标定、covariance whitening 与 held-out camera fail-closed；不能直接训练 stopping MLP。

这是一个有价值的负结果：它阻止我们把 opened synthetic 上调出来的阈值伪装成可部署算法，也说明 N1 的核心不是“什么时候停”，而是“用什么噪声模型定义数据一致性”。

## 2. 冻结实验合同

### 输入轨迹

每条轨迹复用 M2.7 的 exact camera-block affine PCG：

```text
b = A x_net - y
(A A^T) z = b
x_K = x_net - A^T z_K,  K = 0, ..., 10
```

选择器只允许读取冻结结果中的可观测 residual 字段，不得读取 field truth、continuous clean target、OOD 标签或 case family。若阈值直到 K=10 都未命中，返回 prepared CGLS-12 base，但仍计入完整尝试预算。

### 37 个候选规格

| family | 数量 | 第一命中条件 | 部署边界 |
|---|---:|---|---|
| simulator noise-floor multiple | 10 | measured residual 低于 synthetic noise/bias 标量的倍数 | 不是 flow-off covariance，不可部署 |
| base residual multiple | 9 | candidate residual / CGLS-12 residual 低于倍数 | truth-free，但只用一个标量 |
| system residual fraction | 7 | measurement-space linear-system residual 低于初值比例 | 仅因本轨迹 damping=0 才可这样解释 |
| fixed K comparator | 11 | 固定 K=0--10 | 只作参考，不计入 26 个 observable candidates |

预算为 feature preparation `13F/13A` 加所选 projection 的 `K+1` 次 forward、`K` 次 adjoint，最大 `24F/23A`。exact camera-block 的 dense setup 单列为 `1001F-equiv`，不属于部署或效率证据。

## 3. 两个安全区没有交点

### JACRU

| 代表候选 | 平均 K | mean field gain | harm / worst | continuous-clean-target ratio mean / max | 判决 |
|---|---:|---:|---:|---:|---|
| `base_residual_x4` | 1.889 | +44.11% | 2.78% / -1.98% | 1.639x / 3.160x | tail-safe，clean-target 不安全 |
| `base_residual_x1.5` | 3.972 | +41.52% | 8.33% / -7.55% | 1.096x / 1.298x | clean-target-safe，tail 不安全 |

`base_residual_x4` 已是 6 个 tail-safe 候选中 clean-target consistency 最好的一个，但 mean clean ratio 仍为 `1.639x`，最坏达到 `3.160x`；明显越过 `1.10x / 1.50x` development 门。`base_residual_x1.5` 已进入 clean-target 安全区，却恢复为 `8.33%` harm，最坏 field gain `-7.55%`；越过 `5% / -5%` 尾部门。

### pooled CNN

pooled CNN 没有任何 tail-safe 可观测候选。clean-target-safe 中 field tail 最好的 `system_fraction_0.02` 平均选择 K=6.389，mean field gain 为 `+40.00%`，clean ratio 为 `0.904x / 1.195x`，但 harm 仍为 `8.33%`，worst field gain 为 `-11.78%`。

所以这不是 JACRU 独有故障，也不能归因于一次模型训练。两种 learned proposal 都显示同一个结构性现象：越强地服从含噪 `y`，越可能把不可辨认的 bias/noise 写回欠定三维场。

## 4. 为什么这不是 Morozov discrepancy principle 的真实验证

经典 discrepancy principle 需要独立、可信的噪声水平或协方差，使 whitened residual 与统计噪声尺度可比较。N1.0 没有这个条件：

- synthetic `noise_relative_std` 与 `camera_bias_relative_std` 是 simulator metadata，不是 flow-off 估计；
- camera bias 在当前 full-row-rank `A` 下可以被某个 field correction 精确解释，单帧 target 无法区分二者；
- exact camera-block preconditioner 是 `(AA^T)_camera^-1`，不是 detector noise covariance；
- development split 已参与 T0 model early stopping，不能再冒充独立 calibration/lock split；
- continuous clean target 只用于事后评分，没有进入 stopping rule；该指标是同一个 voxel `A` 的预测与 continuous target 之差，不是独立 renderer 或 held-out camera。

因此 N1.0 只能被称为 **post-open observable stopping ceiling**。它没有验证 flow-off whitening，也没有证明真实 BOST 早停失败。

## 5. 真正的 N1.1 应怎样做

### 数据合同

1. 每台相机同一 geometry 至少 50 帧未经平均的 flow-off/reference displacement repeats，保留时间顺序、confidence、bad-pixel mask、曝光和温漂信息。
2. 按时间块分成 covariance fit、calibration envelope、rule selection 和 lock audit；四者不能复用。
3. 永久留一台 camera 或一组 ray blocks，不进入 reconstruction、whitener fitting 或 stopping selection。
4. 至少有一种 independent evidence：clean/high-fidelity renderer、phantom、CFD slice、PIV/pressure 或第二 session。

### 数学接口

先用 flow-off residual 拟合低参数 camera-block whitener `W`，再求解

```text
B = W A
r = W (A x - y)
min_x rho(r) + beta R(x)
```

其中 `rho` 先比较固定 Gaussian、Huber 与 Student-t；`R` 先比较 Tikhonov/TV/H1。只有经典固定方法在 lock split 上出现 field/held-out/tail/cost 的联合可行区，才学习 `beta`、proximal step 或 bounded stopping operator。

### fail-closed 门

候选只有在未用于拟合的 held-out camera/rays 上改善，并且 residual feature 落在 flow-off calibration envelope 内时接管；否则返回冻结的 robust baseline。任何阈值都必须在 lock 前冻结，不能看 final truth 调整。

## 6. 现在停止什么

- 不再搜索更多 `residual < scalar threshold` 的阈值网格；26 个可观测候选已经给出两个无交点安全区。
- 不训练 alpha/stopping MLP；M2.8 truth-aware interpolation ceiling 与 N1.0 observable ceiling 都未授权它。
- 不把 exact camera-block 叫作 noise whitening 或部署加速器。
- 不复用 development 做 covariance calibration，也不打开 fresh/final。
- 不把 mean field gain 写成算法胜利；joint-safe count 对两种网络都是 0。

## 7. 可复现入口

- 冻结配置：`demo_t16_operator/configs/jacru_n1_0_observable_discrepancy_stopping_postopen_v1.json`
- runner：`site_tools/run_jacru_n1_0_observable_discrepancy_stopping.py`
- unit tests：`site_tools/test_run_jacru_n1_0_observable_discrepancy_stopping.py`
- 独立 validator：`site_tools/validate_jacru_n1_0_evidence.py`
- 结果包：`demo_t16_operator/results/jacru_n1_0_observable_discrepancy_stopping_postopen_public/`
- 机器摘要：`demo_t16_operator/results/jacru_n1_0_observable_discrepancy_stopping_postopen_public/summary.json`
- 一级来源审计：`docs/jacru_n1_primary_literature_audit_2026-07-18.md`
- 协议红队：`docs/jacru_n1_protocol_red_team_2026-07-18.md`

## 8. 当前可以怎样写进论文

可以写：

> 在打开的有限孔径 BOST synthetic fixture 上，基于 measured residual、base-relative residual 与 linear-system residual 的 26 个可观测早停候选均无法同时满足同一 voxel 算子下的 continuous-clean-target consistency 与 field-tail safety。该结果说明，对含 camera bias 的欠定仿射目标，标量 residual stopping 不足以解除数据一致性与尾部恢复之间的冲突，并支持转向独立 flow-off covariance、稳健数据项与 held-out fail-closed validation。这里没有 independent renderer 证据。

不能写：提出了新早停算法、优于 DeepONet/FNO/NeRIF/TDBOST、真实 BOST 泛化成功、flow-off whitening 已验证、可部署效率已证明或高水平论文已经完成。
