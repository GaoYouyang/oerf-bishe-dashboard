# PSU 有限孔径正定谱预条件器：首轮 NO-GO

> 证据等级：真实 PSU 九视角 support 几何 + 解析反应流形态真值
> 判决：`CANDIDATE NO-GO`
> 禁止外推：实验三维真值、真实 held-out 泛化、FNO/DeepONet 优越性

## 1. 这轮真正问了什么

当前 B0 线性参考把折射率或密度扰动映射为两分量 BOS 位移：

\[
y=A x+\epsilon,\qquad
A=\text{finite aperture}\circ\text{projection}\circ\nabla_h.
\]

普通 CGLS 只利用一般线性代数，不显式利用 `A` 中的离散梯度结构。首轮候选因此不是让网络直接生成三维场，而是限制网络只能修改精确伴随梯度的频谱：

\[
g_k=A^\top W(y-Ax_k),\qquad
d_k=P_\theta(r_k,\sigma,m)\,g_k.
\]

其中 `Pθ` 是严格为正、有界的实数 Fourier multiplier。每一步随后执行解析加权最小二乘线搜索：

\[
\alpha_k=
\frac{\langle A^\top Wr_k,d_k\rangle}
{\|W^{1/2}Ad_k\|_2^2}.
\]

因此网络：

- 不能跳过名义 `A/Aᵀ`；
- 不能在 Krylov/梯度更新之外直接“画场”；
- 每步仍只使用一次完整 `A` 和一次完整 `Aᵀ`；
- 在离散误差内保持数据项单调不增。

## 2. 为什么先建立强 Sobolev 对手

BOS 观测空间梯度。低频标量场在 `AᵀA` 中天然较弱，四步普通最速下降和 CGLS 很难恢复。验证集只用于选择固定谱预条件器

\[
P_p(k)\propto (0.05+\|k\|_2^2)^{-p}.
\]

冻结网格为 `p=0,1,...,6`，验证 combined loss 依次为：

| p | validation loss |
|---:|---:|
| 0 | 1.21360 |
| 1 | 0.88988 |
| 2 | 0.63817 |
| 3 | 0.51725 |
| 4 | 0.46308 |
| **5** | **0.44419** |
| 6 | 0.44471 |

所以正式对手是 `p=5` 的逆 Sobolev 方向，而不是弱的 identity 或四步 CGLS。学习模块从这个方向精确零初始化，只学习有界各向异性修正。

## 3. 数据合同

- 几何：PSU 公开数据九个 support views，每视角 256 条确定性 active rays；
- 真值：32³ 解析 plume / wavy-front，OOD 使用 thin-front / double-front；
- high-fidelity synthetic measurement：QMC-32 有限孔径；
- reconstruction operator：QMC-8 名义有限孔径；
- train 72、validation 24；
- 六个 24-field audit：IID、family、noise、view-count、joint、exact-operator control；
- train/validation camera masks 互斥；每个 audit split 都排除全部 train/validation masks；
- 三个训练种子；每个模型 2,227 参数；
- Sobolev 与 learned 都是 `4F + 4Aᵀ`；CGLS 是 `4F + 5Aᵀ`。

这些场只是在真实相机几何上的解析形态代理，不是 CFD，也不是 PSU 实验的三维真值。

## 4. 冻结结果

### 4.1 正信号

相对 validation-selected Sobolev：

| split | seed-1 gain | seed-2 gain | seed-3 gain | 主要观察 |
|---|---:|---:|---:|---|
| IID | +4.36% | +4.62% | +4.26% | 三种子 p10 都超过 +2%，无 >1% harm |
| noise OOD | +4.28% | +4.46% | +4.44% | 单独提高噪声仍稳定 |
| view OOD | +1.41% | +1.51% | +1.77% | 4–5 视角单独变化仍有小收益 |
| exact operator control | +4.56% | +4.83% | +4.55% | 正信号不只来自 QMC mismatch |
| family OOD | +0.88% | +0.67% | +0.81% | 均值略正，但 p10 为负 |

这说明一个 2,227 参数的受限谱修正能够学到重复的分布内各向异性，而不是依靠大网络生成完整场。

### 4.2 一票否决

joint OOD 同时使用：

- thin-front / double-front；
- 4–5 个 active views；
- 8–12.5% relative noise；
- QMC-32 truth / QMC-8 nominal mismatch。

三种子 field gain 分别为：

- `-0.432%`
- `-0.368%`
- `-0.199%`

对应 p10 为 `-4.71% / -4.52% / -4.41%`，`>1% harm` 都是 `33.3%`。同时 candidate measurement residual 约 `0.404–0.410`，劣于 Sobolev 的 `0.355`。

预注册要求至少两个种子在 joint OOD 均值不退化，因此通过种子为 `0/3`，正式判决：

> **当前无门控正定谱修正不进入论文 superiority 主张。**

## 5. 这次负结果解释了什么

1. 反应流 BOST 的首要数值结构不是 FNO，而是梯度观测导致的频谱病态；强 inverse-Sobolev 对手必须进入所有后续比较。
2. 仅用 train-support 内的 residual/noise/mask 特征，控制器会持续施加学到的修正；当形态、噪声和视角同时越界时，它不会自动知道何时退回物理基线。
3. 数据项逐步单调不增只保证每一步沿本方向下降，不保证该轨迹优于另一条 Sobolev 轨迹，更不保证三维场误差下降。
4. family OOD 中“场误差略好、重投影明显更差”提示：无实验 3D truth 时不能凭 synthetic field gain 选择真实模型。

## 6. 下一候选必须怎样改变

当前 audit splits 从现在起降级为 development，不能再称 blind。下一轮需要新种子和新形态族。

首选升级是 **Support-Enveloped Spectral Correction**：

\[
P_{\theta,\tau}=P_{\mathrm{Sobolev}}
+\tau(z)\bigl(P_\theta-P_{\mathrm{Sobolev}}\bigr),
\qquad 0\le\tau\le1.
\]

最低要求：

1. `τ=0` 必须精确回到 validation-selected Sobolev；
2. 超出训练 noise / active-view support 时，默认降低 trust，而不是外推满强度修正；
3. 用训练内 stress augmentation 学习 thin-front、correlated noise 和 camera dropout；
4. 加入相对 Sobolev 的 residual-risk penalty，不能只优化 field truth；
5. 新 audit 使用未见 oblique-shell / triple-front、相关噪声和 3-view 压力；
6. 仍按同 calls、wall time、p10、harm rate 判决。

这条升级的目标不是靠门控“把失败样本藏掉”，而是把可拒答范围写成算法合同。若 fresh audit 仍只有 IID 正收益，则该方向应降级为机制负结果，不再继续扩大网络。

## 7. 可复核入口

- [严格公开摘要](psu_b0_spectral_preconditioner_pilot_public_summary.json)
- [四联结果图 PNG](../demo_t16_operator/results/psu_b0_spectral_preconditioner_pilot/psu_b0_spectral_preconditioner_pilot_figure.png)
- [四联结果图 PDF](../demo_t16_operator/results/psu_b0_spectral_preconditioner_pilot/psu_b0_spectral_preconditioner_pilot_figure.pdf)
- [冻结配置](../demo_t16_operator/configs/psu_b0_spectral_preconditioner_pilot_v1.json)
- [算法实现](../demo_t16_operator/psu_b0_spectral_preconditioner.py)

私有 report 和三个 checkpoint 保存在本地 `private_library`，不进入 GitHub Pages。
