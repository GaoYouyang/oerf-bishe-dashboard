# SRCO / DF-gated SRCO：少查询结构化残差算子校准假设

日期：2026-07-16

状态：**POST-OPEN TOY HYPOTHESIS；未预注册；未解锁任何科学主张**

与 V6B 的关系：V6B gate-only 协议保持不变；本文件记录其模型外失败后生成的 V6C/V6D 后续假设。

## 1. 为什么出现这个分支

V6B 协议自检故意设置了两种 truth：

1. **in-class**：真实算子完全属于 27-channel gate 家族；
2. **misspecified**：在 gate 家族之外再加入一个低秩残差。

gate-only 在 in-class 中可恢复到数值精度，但在 misspecified、K=32 时仍有约 11.88% hidden-action relative L2，并被同预算 minimum-norm calibration 的 10.80% 反超。这不是调参失败，而是表示族之外仍有结构。

因此提出一个后续问题：能否让 ray-local gate 负责全空间的有限孔径结构，再让 K 次校准中尚未解释的残差只在已探测子空间内做低秩补丁？

## 2. 算法定义

设：

- `A0`：nominal thin-ray operator；
- `Ag`：用 K 次 forward 拟合 27 gates 后的 ray-local operator；
- `X in R^(P x K)`：预冻结 calibration probes；
- `Y in R^(M x K)`：高保真 oracle 返回的 `A_star X`；
- `E = Y - Ag X`：gate 未解释的校准残差。

正则化残差校准为

\[
C_\lambda
=E(X^TX+\lambda I)^{-1}X^T,
\qquad
A_{\mathrm{SRCO}}=A_g+C_\lambda.
\]

工作名 **Structured Residual Calibration Operator, SRCO**。这里使用从原点出发的输入输出对，不应严格叫 quasi-Newton secant；只有改用 `Delta X, Delta Y` 或局部 Jacobian 时，才是标准差分 secant 语义。

同一更新的解析伴随为

\[
A_{\mathrm{SRCO}}^T z
=A_g^Tz
+X(X^TX+\lambda I)^{-1}E^Tz.
\]

实现不需要 materialize `M x P` 矩阵：先算 K 维分析系数，再乘 `E`；forward 和 adjoint 都是 rank-at-most-K 的矩阵链。

## 3. 这条公式能证明什么

1. `rank(C_lambda) <= K`；低秩来自查询预算，不是新颖网络结构。
2. `lambda=0` 且 X 满列秩时，`C=E X^dagger` 是满足 `CX=E` 的最小 Frobenius 范数更新。
3. `lambda>0` 时是 shrinkage calibration，不再严格插值 K 个观测。
4. 对 `x` 在 `span(X)` 的正交补上，`C x=0`；未查询方向没有自动改进保证。
5. 使用上述解析 transpose，可严格通过 dot-product；但“候选内部共轭”不等于“接近真实 adjoint”。
6. 它不增加高保真预算：gate 和 SRCO 共享同一个 `X,Y` cache，仍是 K forward、0 truth-adjoint。

因此真正困难从“怎样写一个低秩公式”转移到：probe 是否覆盖物理残差、噪声是否会被插值、更新是否保持 BOST 几何/nullspace，以及 inverse 是否真的改善。

## 4. V6C：always-on SRCO 不是统一胜者

固定配置：64 维输入、96 维输出、每层 12 rigs、`K=4/8/16/32`、0.2% calibration noise、12% 模型外 Frobenius 残差。结果目录：[V6C report](../demo_t16_operator/results/v6c_hybrid_gate_secant_postopen/report.json)。

| K=32 hidden-action 中位数 | nominal | 同预算 calibration | 27-gate | always-on SRCO |
|---|---:|---:|---:|---:|
| in-class + noise | 0.10088 | 0.07304 | **0.000163** | 0.001970 |
| misspecified + noise | 0.15770 | 0.10972 | 0.11999 | **0.08383** |

SRCO 在模型外层明显更好，却把纯 in-class 的校准噪声当成信号，误差比 gate 放大约 12 倍。判决是 `POST_OPEN_HYBRID_NOT_DOMINANT`，不是候选胜利。

## 5. V6D：用噪声地板控制补丁幅度

always-on 的失败给出第二阶段 post-open 假设。真实方法需要从独立 flow-off repeats 估计校准噪声 covariance；toy 暂时使用 generator-known 对角 covariance。gate ridge 的 hat matrix 与有效自由度记为

\[
H_g=G(G^TG+\lambda_g I)^{-1}G^T,
\qquad df_g=\operatorname{tr}(H_g).
\]

令 (R=I-H_g)。一般 covariance 下应扣除的残差噪声能量是

\[
E_{noise,res}=\operatorname{tr}(R\Sigma R^T).
\]

对固定设计和同方差噪声，残差噪声的精确自由度是

\[
\nu_{res}=\operatorname{tr}[(I-H_g)^T(I-H_g)]
=MK-2\operatorname{tr}(H_g)+\operatorname{tr}(H_g^TH_g),
\]

此时 (E_{noise,res}=\sigma^2\nu_{res})。V6D toy 的噪声按 probe 列异方差，因此实现保留 design row 的已知对角方差，并直接计算 (\operatorname{tr}(R\Sigma_{diag}R^T))；不再把总噪声能量按同方差自由度平均缩放。

定义无需 hidden label 的正部信号比例：

\[
\alpha
=\left[1-\frac{E_{noise,res}}{\|Y-A_gX\|_F^2}\right]_+,
\qquad
A_{DF\text{-}SRCO}=A_g+\alpha C_\lambda.
\]

V6D 使用全新固定 seed、每层 16 rigs，其他难度保持一致。结果目录：[V6D report](../demo_t16_operator/results/v6d_df_gated_srco_postopen/report.json)。

| K=32 hidden-action 中位数 | 27-gate | 同预算 calibration | always-on SRCO | DF-gated SRCO |
|---|---:|---:|---:|---:|
| in-class + noise | **0.00017815** | 0.070129 | 0.0019968 | 0.00017827 |
| misspecified + noise | 0.121422 | 0.112605 | 0.0876786 | **0.0876779** |

激活行为：

- in-class K=32：alpha 中位数 0，p90 0.0115；
- misspecified K=32：alpha 中位数 0.99971，p10 0.99969；
- in-class 相对 gate 的 p90/max 退化为 0.78%/1.73%，绝对最大变化 `3.41e-6`；
- misspecified 16/16 rigs 优于 gate，逐 rig 最差仍改善 25.0%；
- misspecified 中位数相对 gate 改善 28.78%，相对同预算 calibration 改善 24.38%；
- float64 dot-product 最大 defect `2.88e-13`，高保真 truth-adjoint 调用为 0。

这说明噪声地板可能把“是否启用模型外补丁”转成可测判据。上述对角 covariance trace 对固定 toy smoother 是精确的；真实 BOS 的相关噪声仍需 whitening，或由 flow-off repeats 估计 `tr(R Sigma R^T)`。它仍不是 fresh 证据，因为公式是在看过 V6C 后提出，且 covariance、失配结构和所有 rigs 都来自内部 toy generator。

## 6. 文献碰撞：公式新意关闭

| 已有谱系 | 已覆盖概念 | 对本项目的限制 |
|---|---|---|
| [Dennis & Schnabel, least-change secant updates](https://epubs.siam.org/doi/10.1137/1021091) | 满足 secant 条件的最小改动 Jacobian/operator update | 不能声称首次少量输入输出对低秩更新 |
| [Lunz et al., learned operator correction](https://epubs.siam.org/doi/10.1137/20M1338460) | 近似 forward 的 learned correction；forward-only 不保证 inverse/gradient | 必须比较真实残差梯度与最终重建 |
| [Multifidelity Deep Neural Operators](https://doi.org/10.1103/PhysRevResearch.4.023210) | 低保真 operator + 少量高保真 residual learning | 不能声称首次组合多保真与残差 operator |
| [Physics-guided correction under model misspecification](https://arxiv.org/abs/2606.03469) | misspecified physics prior + additive learnable correction | 2026 近邻已覆盖广义“物理算子 + 修正”叙事 |
| [OED with correlated observations](https://epubs.siam.org/doi/10.1137/21M1418666) | 有限预算观测设计与相关误差 covariance | 不能声称首次主动选择有限观测；必须处理 flow-off correlation |
| [Active learning for neural operators](https://proceedings.mlr.press/v267/kim25m.html) | 主动选择高保真轨迹/时间步以降低数据成本 | 样本 acquisition 可启发 probe，但不等于目标-rig 输入向量校准 |

禁止声称：

- 首次将物理算子与数据驱动残差结合；
- 首次低秩算子校准或少查询适配；
- 首次用 Broyden/multisecant 做 operator learning；
- K 次查询保证恢复完整未知算子；
- toy 中位数证明真实 BOS/OERF 有效。

## 7. 可能可守的新意

若后续要形成论文，贡献至少要落在以下三项中的两项，而不是公式本身：

1. **BOST 结构保持。** 更新保持已知 mask、ray-local support、尺度/零空间、相机分块和严格 forward-adjoint pair。
2. **主动物理 probe。** 比较随机正交、反应场 POD、thin-front/shock、leverage/D-optimal、最大 predicted residual 等查询；证明 K 次预算怎样覆盖有限孔径残差，并在相关噪声下做 whitening 或 covariance-weighted 设计。该点只能定位为 BOST 目标-rig vector-query 的应用假设，不能泛称首次 active operator learning。
3. **噪声与失配分解。** 给出 unqueried residual tail、noise amplification、ridge bias、gate representation bias 的误差上界。
4. **部署时严格预算。** 训练阶段不接触目标 rig；部署只 K 次高保真 forward；主网络不更新；每个 baseline 同 query/cost。
5. **inverse 一致性。** dot/finite-difference 只是 G1；还要在真实残差梯度、PBB/CGLS field、held-out view 和 runtime 上胜过强基线。
6. **跨离散泛化。** 在 32^3、64^3 乃至不同相机分辨率下保持相同的函数空间解释，不能只在 64 维 toy 成立。

## 8. 下一次真正有效的实验

### 8.1 先冻结，不再看 V6C/V6D 调参数

- SRCO 的 lambda、alpha 公式、probe family、K 曲线和 B* 只能在新的 opened-development generator/renderer 上一次冻结；
- fresh 首开前写死 48 rigs、统计门槛和 config hash；
- V6C/V6D 的 seed、rigs 与 hidden probes 永久归档为 post-open hypothesis data，不得回收进 fresh。

### 8.2 必须比较的同预算方法

1. `A0`；
2. gate-only；
3. origin-based ridge calibration / `lambda=0` pseudoinverse；
4. truncated-SVD、recursive Broyden/multisecant；
5. always-on SRCO；
6. DF-gated SRCO；
7. 相同 K 对上的 kernel/GP discrepancy；
8. 相同 K 与步数的 LoRA/last-layer adaptation；
9. full high-fidelity / best-rank-K oracle 只作不可部署上界。

### 8.3 数据层级

- **PSU fixed rig**：验证 MAT schema、单位、mask、reprojection；不能验证跨 aperture。
- **`photon` external synthetic**：生成独立 aperture/layout；需 CUDA，并补一致 adjoint 或只做 forward gate。
- **OERF**：flow-off repeats 估噪声；同场多 f-number/focus 或 calibration phantom；独立 session 封存。

## 9. 给何远哲的最小确认问题

1. flow-off/reference 是否有足够 repeats，可估逐相机/逐像素 covariance？
2. 是否有 calibration phantom、paired thin/cone simulation，能定义 K 个已知输入场 `x_k`？
3. 如果真实实验无法主动施加 `x_k`，K 个“query”在装置上究竟对应什么可操作测量？
4. 更现实的 probe 是数值 high-fidelity renderer 调用，还是实验标定场？二者的论文主张必须分账。
5. 组内最关心 aperture/focus、相机布局、held-out view，还是 TDBOST 时序事件？若孔径失配不重要，应停止 SRCO，转 TRAIL-4D。

## 10. 可复核入口

```bash
PYTHONPATH=. .venv/bin/python demo_t16_operator/run_v6b_protocol_conformance.py
PYTHONPATH=. .venv/bin/python demo_t16_operator/run_v6c_hybrid_gate_secant_postopen.py
PYTHONPATH=. .venv/bin/python demo_t16_operator/run_v6d_df_gated_srco_postopen.py

PYTHONPATH=. .venv/bin/python -m pytest -q \
  demo_t16_operator/test_limited_query_calibration.py \
  demo_t16_operator/test_v6b_protocol_conformance.py \
  demo_t16_operator/test_v6c_hybrid_gate_secant_postopen.py \
  demo_t16_operator/test_v6d_df_gated_srco_postopen.py
```

当前唯一合规结论：**DF-gated SRCO 是值得带到独立 renderer 重新预注册的候选；尚不是论文结果。**
