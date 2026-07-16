# BOST-GC-SPD-PCGLS 研究合同

**日期：** 2026-07-16
**状态：** 开发原型；不构成 fresh、实验或算法优越性证据
**当前基线：** validation-selected Sobolev-PCGLS-4，`4F + 4AT`，0 参数

## 1. 为什么换主线

当前 2,227 参数 learned spectral steepest direction 只证明了“学习频谱修正
可以优于固定 Sobolev 梯度下降”。补上同预算 Sobolev-PCGLS-4 后，经典
Krylov 方法在 validation 和未参与选参的 calibration 上分别把场误差再降低
约 5.00% 和 4.94%。

因此旧问题“怎样学一个更好的单步梯度方向”已经被否掉。新问题改成：

> 在保留固定预条件 PCGLS 的共轭结构、精确 BOST forward/adjoint 和同调用
> 预算的前提下，能否根据相机集合、噪声和有限孔径条件，生成比全局静态
> Sobolev 更合适的正定预条件器？

## 2. 现实物理困境

BOST 观测与普通标量层析不同：

1. 测量主要响应折射率或密度场的横向梯度，低频和绝对幅值更难恢复；
2. 可用相机集合会因光学通道、遮挡、坏点和同步情况变化；
3. 各相机噪声、背景纹理和 optical-flow 置信度不相同；
4. 有限孔径和景深使 nominal forward 与真实成像链存在 operator mismatch；
5. 少视角下测量 residual 降低不代表三维场更接近真值。

静态 Sobolev 预条件器对全部场、相机子集和噪声使用同一频谱。它已经很强，
但不一定对所有测量条件都最合适。这里才是条件化学习可能存在的空间。

## 3. 最小算法

### 3.1 可观察输入

第一版只使用部署时可获得的信息：

- 每个 active view 的相机平面向量与视线法向摘要；
- 每视角 ray scale 的均值和离散度；
- active view mask 与 active view count；
- 每视角声明噪声尺度；
- 零初值时的白化 displacement residual RMS、分量 RMS 与分量相关性。

禁止读取：

- morphology/family 标签；
- 真实三维场；
- 当前样本相对 baseline 的真实 gain；
- held-out camera measurement；
- fresh split 身份。

### 3.2 低维正频谱

网络只输出七个频谱 basis 系数：

\[
\phi(k)=
\left[
k_x^2,\;k_y^2,\;k_z^2,\;
k_x^2k_y^2,\;k_x^2k_z^2,\;k_y^2k_z^2,\;
(k_x^2+k_y^2+k_z^2)^2
\right].
\]

以 validation-selected Sobolev-PCGLS-4 为基底：

\[
M_0(k)=(0.05+\|k\|^2)^{-4}.
\]

条件化 multiplier 为：

\[
\log C_\theta(z,k)
=c\tanh\left(\sum_j a_{\theta,j}(z)\phi_j(k)\right)
-\operatorname{mean}_k(\cdot),
\]

\[
M_\theta(z,k)=M_0(k)\exp(\log C_\theta(z,k)).
\]

因此：

- `M_theta > 0`，不会产生显式负频谱增益；
- correction 的几何均值为 1，避免靠整体步长缩放作弊；
- 最后一层零初始化时，逐值等于静态 PCGLS-4；
- 只有七个输出系数，便于解释和跨分辨率迁移。

### 3.3 固定 PCGLS

`M_theta` 在进入求解前只生成一次，随后四步完全固定：

```text
z <- observable camera/noise/initial-residual features
M <- positive bounded spectral multiplier from z
r0 <- masked and whitened observation
s0 <- AT r0
p0 <- M s0

for k = 0, 1, 2, 3:
    qk <- A pk
    alpha <- <sk, M sk> / ||qk||^2
    x <- x + alpha pk
    r <- r - alpha qk
    if k < 3:
        s_next <- AT r
        beta <- <s_next, M s_next> / <sk, M sk>
        p <- M s_next + beta p
```

固定四步只需要 `4F + 4AT`。最后的 `AT r4` 不会生成下一搜索方向，因此不
计算。

## 4. 为什么不立即做逐步自适应

如果网络每一步都根据新 residual 改变预条件器，标准 PCG 的共轭性论证不再
适用。Notay 的 Flexible CG 需要显式处理变化预条件器和搜索方向正交化：

- [Flexible Conjugate Gradients](https://epubs.siam.org/doi/abs/10.1137/S1064827599362314)

所以路线分层为：

1. **V0：**有限条件查表或线性映射；
2. **V1：**低维 MLP，solve 内固定；
3. **V2：**置换不变 camera-set encoder，solve 内固定；
4. **V3：**加入可观察 residual/noise summary，solve 内固定；
5. **V4：**只有 V1-V3 过门后，才比较 Flexible CG 自适应版。

## 5. 数据防火墙

| 层 | 用途 | 允许做什么 |
|---|---|---|
| `risk_train` | 参数学习 | 反向传播 |
| `risk_validation` | checkpoint/超参数 | 选择 epoch、宽度和 correction bound |
| `risk_calibration` | 单次开发确认 | 三种子冻结后一次性评分 |
| 已打开 fresh | 历史诊断 | 禁止训练、早停、阈值和结构选择 |
| 新 independent repeat | 后续确认 | 仅在开发门通过后生成 |
| PSU/OERF held-out camera | 实验迁移 | 不能进入训练和停止 |

当前开发结果即使通过，也只能授权“生成新的 independent repeat”，不能直接
写成算法优越性。

## 6. 必须击败的基线

### 同预算主基线

- static isotropic Sobolev-PCGLS-4；
- static anisotropic Sobolev-PCGLS-4；
- PCGLS-3，作为更低调用预算对手；
- learned stopping only，不学习 multiplier。

### 前沿保持基线

- TV-superiorized PCGLS；
- H1 / L2 Tikhonov；
- early-stopped CGLS。

### 学习预条件器碰撞

- [Learning Preconditioners for Conjugate Gradient PDE Solvers](https://proceedings.mlr.press/v202/li23e.html)
- [Neural incomplete factorization](https://arxiv.org/abs/2305.16368)
- [UNO-CG](https://arxiv.org/abs/2508.02681)

这些工作意味着“神经网络生成预条件器”不是创新点。可能的独立贡献只能来自：

1. BOST 梯度观测对应的频谱病态；
2. 可变相机集合的置换不变条件化；
3. finite-aperture/operator mismatch 和 flow-off covariance；
4. 同调用预算、精确伴随和 held-out camera 证据；
5. 对 thin front、plume 和 shock 的分层可靠性。

## 7. 指标

### 三维有真值的 synthetic 层

- field relative L2；
- gradient relative L2；
- front top-10% F1；
- measurement relative L2；
- 每场相对 PCGLS-4 gain；
- `>1%` harm rate；
- paired bootstrap 95% interval；
- 逐 family / noise / active-view 层结果。

### 实验无三维真值层

- held-out camera displacement RMS / p95；
- ambient 区域误差；
- flow-off repeatability-normalized gain；
- calibration perturbation 稳定性；
- 若有 PIV-BOST：折射补偿后的 velocity error；
- wall time、peak memory、F/AT calls。

## 8. 当前开发门

一个 seed 必须同时满足：

1. validation mean field gain `>= 2%`；
2. calibration mean field gain `>= 2%`；
3. 两层 paired bootstrap 95% 下界都 `> 0`；
4. 两层 `>1% harm rate <= 5%`；
5. PCGLS data objective、正增益和几何均值合同全部通过。

三个种子至少两个通过，才允许：

- 冻结模型与基线；
- 创建全新 field/noise/view-mask seeds；
- 运行 independent repeat。

## 9. 消融矩阵

| 消融 | 回答的问题 |
|---|---|
| geometry only | 相机集合本身是否决定频谱 |
| noise only | 收益是否只是噪声自适应平滑 |
| residual only | 初始测量难度是否足够 |
| geometry + noise | 是否有组合增量 |
| geometry + noise + residual | 完整低维候选 |
| isotropic basis | 各向异性是否必要 |
| no quartic terms | 高频/薄前沿项是否必要 |
| fixed vs learned stopping | 收益来自 preconditioner 还是迭代次数 |
| fixed PCGLS vs Flexible CG | 逐步自适应是否值得额外复杂度 |

## 10. 失败怎样解释

### validation 就失败

说明当前可观察特征或低维 basis 没有足够 headroom。停止增加网络容量，先做
条件 oracle 和 TV baseline。

### validation 正、calibration 负

说明模型选择过拟合。不能打开 fresh；降低容量、增加训练结构多样性，或关闭
学习预条件器。

### field-L2 正、front F1 负

说明模型主要平滑场而损伤薄前沿。优先比较 TV-superiorized PCGLS，不能只报
均值误差。

### synthetic 正、held-out camera 无增量

说明解析形态或噪声代理与真实装置不一致。结论降级为 synthetic development，
请求 flow-off covariance、真实 aperture 和 calibration perturbation。

## 11. 本机与服务器门

当前 32 立方、九视角子集、低维 controller 可在 Apple M5 / 32 GB 上训练。
强频谱 183 候选 screen 约 40 秒，当前无需租 GPU。

只有满足以下条件才迁 CUDA：

1. validation 与 calibration 已有稳定正信号；
2. 需要 64-128 立方、多模型、多种子；
3. 需要外部 finite-aperture renderer；
4. 服务器成本不会导致 baseline 调用预算失配。

## 12. 需要问何远哲

1. 组内 BOST forward 是否能提供精确 `F` 与 `FT/VJP`？
2. 每套实验实际可用相机集合是否变化？变化原因是什么？
3. 是否有 flow-off repeats，可估计每相机/每像素 covariance？
4. 有没有一条 camera 可以永久封存，只作 held-out reprojection？
5. finite aperture、f-number、焦距、物距和焦平面是否有记录？
6. 主终点更看重 field、thin front、held-out displacement，还是 PIV velocity correction？
7. NeRIF/TDBOST 的训练、validation、实验 case 是否允许按 session 或 rotation 分块？
8. 若没有实验 3D truth，师兄是否接受“synthetic field-L2 + experimental held-out camera”双层证据？

## 13. 当前诚实结论

已经实现并测试的是：

- 低维 geometry/noise-conditioned positive multiplier；
- solve 内固定的 materialization；
- 零初始化逐值回退静态 PCGLS；
- `4F + 4AT` 调用合同；
- 正定性、几何均值和梯度回传测试。

尚未证明的是：

- 条件化模型优于 PCGLS-4；
- 对 fresh、PSU measurement 或 OERF 实验有效；
- 优于 TV、UNO-CG、NeuralIF、FNO 或 DeepONet；
- 可以发表高水平论文。

下一判决完全由 validation/calibration 的三种子训练结果决定。
