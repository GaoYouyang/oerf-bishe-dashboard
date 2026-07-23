# 三维 BOST `q_cal` 几何辨识力剖面协议

> 状态：`POSTOPEN_DEVELOPMENT_PROTOCOL`
>
> 总判决默认值：`NO-GO_FOR_DATA_ONLY_QCAL`
>
> 目的：区分 known-field 几何敏感性、纯数据几何辨识力与正则先验条件下的 profile curvature。

> 审计修订：正式产物的 evaluator teacher 改为与 nuisance field 相同的 voxel forward family；三相机选择合同限定为六相机 pilot 辅助的下一次布置，不称当帧 subset-only 部署。

## 1. 为什么要做这一轮

相机残差一致性回放只能形成 `q_rel`，不能称为几何可观测性。若要构造 `q_cal`，至少要计算测量对几何参数的 Jacobian，并把未知三维场作为 nuisance variable 消去。否则 raw `J^T J` 默认场已知且固定，会把场变化可吸收的几何响应也算作“标定信息”。

本协议在前两轮 synthetic 结果已经打开后形成，只能叫 post-open development diagnostic。它不会产生 fresh、真实 BOST、跨 rig 泛化或论文成功证据。

## 2. 局部模型与五个参数

对当前 reported pose 附近的局部模型：

```text
y = A(eta) x + epsilon
C_j = d(A(eta)x) / d eta_j
```

`eta` 是五维无量纲参数，分别沿既有 synthetic mode 改变 yaw、pitch、roll、detector shift-u 和 shift-v。一单位参数沿用原实验的物理量级：

| 参数 | 一单位 mode 的 RMS 物理尺度 |
|---|---:|
| yaw | 约 0.7895 degree |
| pitch | 约 0.5831 degree |
| roll | 约 0.5358 degree |
| shift-u | 约 0.006218 domain unit |
| shift-v | 约 0.005774 domain unit |

这些不是 OERF 实验相机的不确定度，只是当前 synthetic 参数坐标。A/D/E-optimal 排名会随参数尺度改变；得到真实标定协方差后必须重新冻结。

## 3. 两套几何 Jacobian

### 3.1 pilot-assisted estimated-field Jacobian

每个 synthetic frame 用六相机 noisy pilot observation、reported geometry 和同一 ridge solver 得到 `x_hat`，再用 matrix-free 中心差分：

```text
C_est[:, j] = (A(+h_j e_j) x_hat - A(-h_j e_j) x_hat) / (2 h_j)
```

它不读取 field truth。噪声尺度使用 observation norm 与冻结噪声比例估计，不使用 clean observation。因为 `x_hat` 使用了六相机，候选三相机子集只能被解读为 future-acquisition design；不允许声称子集自身已能完成当帧重建。

### 3.2 evaluator-only voxel teacher Jacobian

将 analytic phantom 采样成冻结 voxel truth，再经与 nuisance operator 相同的 voxel forward family 在同一几何步长上生成 `C_teacher`。连续 analytic renderer 只用来生成 noisy pilot observation，不再与离散 voxel `A` 混成一个假的 Schur likelihood。teacher 只评价 Jacobian 误差、相机子集排序和 D-efficiency；所有 pilot-assisted 决策必须先冻结。

## 4. 有限差分门

每个 mode 独立测试步长 ladder：

```text
0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125,
0.0015625, 0.00078125, 0.000390625
```

选择第一个连续两档 Jacobian 相对差低于 `5e-3` 的较小步长。中心差分信号必须高于 `100 * eps_machine` 的 cancellation floor。没有稳定平台时整帧返回 `INVALID_FD_JACOBIAN`，不得静默使用默认步长。

## 5. 三种信息量必须同时报告

白化后令 `B=A/sigma_hat`、`C=J/sigma_hat`。

### 5.1 known-field raw Gram

```text
G_raw = C^T C
```

它只回答“场固定时几何变化是否改变测量”，不回答 joint reconstruction 中几何是否可辨。

### 5.2 data-only profile Gram

对 `B=U Sigma V^T`：

```text
S_0 = C^T (I - U_r U_r^T) C
```

它把自由 voxel field 的数据切空间消掉。当前 `B` 预计满行秩，因此 `U_r U_r^T=I`、`S_0=0`。这不是数值失败，而是当前离散模型下的不可辨识性结论：单靠同一 measurement，未知自由体场可以吸收局部几何变化。

### 5.3 regularized profile curvature

对 ridge prior：

```text
S_lambda = C^T lambda (B B^T + lambda I)^(-1) C
```

它必须称为 prior-conditioned/regularized profile curvature，不能称为纯数据 Fisher information。实现同时用 SVD 形式复核，禁止显式求逆；对称、PSD、`S_lambda <= G_raw` 与 dual/SVD 一致性都要过数值门。

## 6. 正则强度扫描

相对 ridge 强度固定扫描：

```text
1e-6, 1e-4, 2e-3, 1e-2, 1
```

其中 `2e-3` 是上一实验的参考强度。每档重新计算 profile curvature 与相机子集排名。若 trace retention 或最优子集随 alpha 大幅改变，必须写成 prior sensitivity，不能挑一个最好看的 alpha 当作算法证据。

## 7. 固定预算相机选择

只比较 `k=3` 台相机，枚举六台相机的全部 `20` 个子集。比较对象：

1. `reported_azimuth_spread`：最大化最小方位角间隔；
2. `estimated_raw_d_optimal`：用 `C_est^T C_est` 选 D-optimal；
3. `estimated_regularized_profile_d_optimal`：用 `C_est` 的 `S_lambda` 选 D-optimal；
4. `voxel_teacher_regularized_profile_oracle`：用 `C_teacher` 选最优，只作 evaluator ceiling。

若 Gram rank 小于 5，返回 `UNIDENTIFIABLE`，不得用 jitter 强行产生排名。D-efficiency 定义为：

```text
exp((logdet(S_selected) - logdet(S_oracle)) / 5)
```

## 8. 冻结门与总体判决

### 数值有效门

- finite-difference 最大相对差不高于 `5e-3`；
- data-only `S_0`、regularized `S_lambda` 的相对对称缺陷与 PSD 缺陷不高于 `1e-10`；
- dual/SVD `S_lambda` 相对差不高于 `1e-8`；
- 所有候选子集的 estimated/teacher raw 和 regularized profile Gram rank 均为 5；全部非有限时 fail closed。

### 参考 alpha 的 post-open ranking signal

仅作机制线索，要求：

- 六个 rig 的 estimated-vs-analytic profile ranking Spearman 最低不小于 `0.75`；
- estimated profile selector 的 oracle D-efficiency 中位数不小于 `0.90`、最差不小于 `0.75`。

### 数据辨识力总门

只要 full-voxel `S_0` 在任一 rig 的相对 rank 不是 5，或 trace/max-eigenvalue retention 不高于 `1e-10`，总体状态必须保持 `NO-GO_FOR_DATA_ONLY_QCAL`。另外单独报告 `any_identifiable_direction`，不允许将 rank `1/5` 写成全几何可辨识。即使 reference alpha 的正则化排名过门，也只能叫 prior-conditioned mechanism signal。

## 9. 允许与禁止的结论

允许：

- 当前 overparameterized voxel BOST proxy 下，raw geometry sensitivity 与 data-only identifiability 不等价；
- 正则先验可产生正的 profile curvature，其大小和相机排名依赖 alpha；
- estimated-field profile ranking 可成为未来先验条件模型必须击败的经典基线。

禁止：

- 已证明真实 BOST 几何不可辨；
- 已证明 NeRIF/TDBOST 的神经场先验正确或充分；
- 已得到通用 observability score；
- 已完成自动标定、新算法、跨 rig 泛化或论文结果；
- 已证明三相机 subset-only 当帧重建可行；
- 用 post-open 六个 rig 的均值宣称突破。

## 10. 下一道真实门

需要何远哲师兄确认真实 callable 是否提供 geometry JVP/VJP、场表示是 voxel/implicit/tensor、哪些帧共享几何和场先验、是否有独立 calibration target/reference、真实噪声协方差和组内认可基线。只有这些事实到位，才能判断 `q_cal` 应来自已知标定目标、低维物理场、4D 时间耦合，还是明确标注的 neural-field tangent prior。
