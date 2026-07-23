# Jump-Aware Cone-Ray Unrolling：原创性红队与最小可发表差异

## 先把不新颖的部分划掉

以下组件都不能单独作为论文主创新：

1. **smooth background + level set。** Kadu 等已经把连续变化背景与 level-set anomaly 联合
   写成双层交替层析优化：[Parametric level-set tomography](https://arxiv.org/abs/1704.00568)。
2. **phase-field / perimeter regularization。** binary tomography 中已有 phase-field relaxation、
   收敛与参数研究：[Binary recovery via phase field regularization](https://arxiv.org/abs/1811.02865)。
3. **给 DeepONet 增加界面坐标。** 移动不连续面会使线性 reconstruction 受到下界限制，
   shift-DeepONet 与 FNO 这类 nonlinear reconstruction 已是明确对手：
   [Nonlinear reconstruction for discontinuities](https://arxiv.org/abs/2210.01074)。
4. **BOST 坐标网络。** NeRIF 已经直接表示折射率与梯度、进行随机射线采样，并以 AD/ND
   一致性和双位移损失训练：[NeRIF](https://arxiv.org/html/2409.14722v2)。
5. **finite-aperture forward。** depth-of-field/cone-ray 对 BOST 正反问题的影响已经被单独
   建模：[Finite-aperture BOST](https://arxiv.org/html/2402.15954)。
6. **tensor + lightweight network 的 4D BOST。** 何远哲的 TDBOST 已公开张量分解实现，
   面向 4D refractive-index field 的内存与精度问题：
   [官方代码](https://github.com/Hyz617/TDBOST)、[ACM DOI](https://doi.org/10.1145/3809488)。
7. **重建后用 level set 求冲击波压力。** 2025 Optics Letters 已在 time-resolved BOS 后结合
   level set、Gladstone-Dale 与 Rankine-Hugoniot 推压力，并用压力传感器对照：
   [Instantaneous 3D pressure mapping](https://doi.org/10.1364/OL.571452)。

因此，`phase field + FNO + unrolling` 的简单拼装很容易被审稿人判为“已有组件堆叠”。

## 仍值得检验的研究假设

工作名暂定 **Jump-Aware Cone-Ray Unrolling (JACRU)**。这个名称和组合都只是研究假设，
尚未完成穷尽式查新，不能写“首次”。

用两个侧场和一个界面表示折射率：

```text
n(x) = n-(x) + H_epsilon(phi(x)) [n+(x) - n-(x)]
```

其梯度不是普通平滑场：

```text
grad n = (1-H) grad n- + H grad n+
       + [n] delta_epsilon(phi) grad phi.
```

最后一项把跳跃强度、界面法向和相机射线方向直接耦合起来。**潜在最小差异**不是用了
level set，而是让展开迭代分别更新：

1. 两侧 smooth fields；
2. interface geometry `phi`；
3. jump amplitude `[n]`；
4. camera-conditioned bias / reliability；
5. 以上变量经过同一个可微 finite-aperture cone-ray forward 后才比较观测。

## 最小算法结构

```text
unordered {(camera geometry, displacement, mask, reliability)}
                    |
             set geometry encoder
                    |
        n-0, n+0, phi0, jump0, camera bias0
                    |
      exact cone-ray residual and Jacobian-transpose
                    |
  split update: smooth branch | interface branch | bias branch
                    |
          repeat K=3..5 fixed-budget layers
                    |
     n, interface, jump, uncertainty + reprojection
```

FNO 可以做 smooth-field proximal，但不应独立承担非规则射线反演。Set encoder 负责相机
顺序和数量变化；exact forward 始终留在循环中。第一版先不用 Euler/PDE，把变量、证据和
失败原因控制在本科毕设可解释范围内。

## 物理范围必须先收窄

首篇最稳妥的是：**单一激波、已知上游状态、直线光/finite aperture 合同明确**。

- 激波：需要守恒跳跃与熵可容许性；仅变尖锐不等于物理正确。
- 接触面：压力和法向速度连续，不能套用激波跳跃损失。
- 火焰面：有有限反应厚度；分辨率不足时才可近似 sharp interface。
- 爆轰：前导激波与反应区并存，单 level set 不能自动代表完整 ZND 结构。
- 多组分反应流：Gladstone-Dale 常数会随组分变化，仅折射率通常不能唯一推出温度。

只有折射率/密度且没有上游状态、EOS、速度或压力时，Rankine-Hugoniot 只能作受限后处理，
不能包装成完整流场恢复。

## 必须击败的基线

| 层次 | 最低对比 |
|---|---|
| 经典 | CGLS-Tikhonov、CGLS-TV/Huber、cone-ray NIRT+TV、RBF/ART+TV |
| 结构先验 | adapted PaLS、phase-field、重建后 level-set 提面 |
| 神经隐式 | NeRIF、NeDF、Neural Refractive Index Primitives |
| 算子学习 | DeepONet、FNO/F-FNO、shift-DeepONet、SetONet/NIO |
| 展开 | Learned Primal-Dual、MoDL/Variational Network、FNO proximal unrolling |
| 4D | TDBOST，以及去掉 interface branch 的同预算 tensor baseline |

所有模型必须使用同一 train/validation/fresh 父场拆分、同一相机预算、同一 forward calls、
同一噪声与几何扰动；否则“更好”没有意义。

## 预注册失败门

这些阈值是本项目拟定的工程门，不是文献公认定理：

1. 至少 3 种 geometry、3 种 view count、5 seeds；field error 平均改善 `>=5%`。
2. ASSD/HD95 至少改善 `>=10%`，F1@1dx 至少 `+0.03`。
3. held-out reprojection 不得差于最强基线 `1.05x`；只变尖而观测变差视为 hallucination。
4. 0-interface smooth control 的 field harm 不超过 `1%`，且 false surfaces 必须为 0 或低于
   冻结容忍率。
5. geometry/view/noise OOD 至少保留一半域内增益；否则不能写“通用算子”。
6. finite aperture、pose 和 optical-flow bias 扰动后若优势消失，判 forward-mismatch 脆弱。
7. 真实数据只能报告 held-out image-space consistency；无 volumetric truth 时禁止写实验
   field-L2。
8. 多初始化得到同样重投影却给出不同界面时，必须报告不可辨识区间，不能挑最好看的图。

## 当前代码身份

`demo_t16_operator/phase_interface_bost.py` 是一个**可证伪原型**：支持 0–2 interface、
smooth background、Eikonal/curvature/gate regularization 和 fixed forward budget。它用于
检验参数化与失败模式，不是 JACRU 的最终创新，也没有胜过任何基线。

下一实现应先在 PSU-S16 的 0/1/2-interface phantoms 上比较：

```text
CGLS / TV / phase-only optimization / current prototype / split-update JACRU
```

只有 split update 明显优于 phase-only optimization，才能说明收益来自新更新机制，而不是
level set 这个已有表示。

## 给何远哲确认的五个问题

1. 师兄更希望第一篇限定 shock，还是优先对齐 turbulent flame/spray？
2. 组内 forward 能否暴露 finite-aperture residual 的 Jacobian-transpose/VJP？
3. 是否有独立 camera/session、flow-off repeats 或 pressure/PIV，用于真实外部验证？
4. TDBOST 当前最失败的形态是 shock、薄火焰面、几何变化，还是噪声/位姿漂移？
5. 论文是否接受“synthetic volumetric truth + real held-out reprojection”的双证据结构？

在这五项未回答前，最合理动作是继续做公开数据与解析 phantom 的最小机制证伪，不租大卡。
