# N2-PVGR 下一轮算法候选：从当前 NO-GO 往哪里走

## 排序结论

| 排名 | 候选 | 现在做什么 | 可行性 | 主要风险 |
|---|---|---|---|---|
| 1 | `N0.1` 证书复用与向量化 | 先把 `2.5×` wall-time 负担降下来 | 高 | 只解决工程成本，不产生论文创新 |
| 2 | `N1` 一阶轨迹变分残差预测器 | 直接预测 `H-M`，替代当前弱 risk proxy | 中 | 曲光线线性化已有先例，必须靠 BOST 门控与实证定位 |
| 3 | `N2` cross-fitted residual router | 用独立校准集学习 `pi`，保持 HT 纠偏 | 中 | 数据量不足、跨 rig 失效、router 成本 |
| 4 | `N3` validated event certificate | interval Newton/Bernstein + outward rounding | 低到中 | 很重，且实验室若没有动态 mask 就不值当 |
| 5 | 宿主 NeRIF/4D 重建集成 | 把算子放进真实训练与 held-out 视角 | 数据到位后高 | 当前没有实验数据合同 |

建议立即并行做 `N0.1 + N1`，不要先训练大网络。

## 候选 1：证书复用与向量化

当前 certificate 重新计算 endpoints、midpoints、automatic gradients 和逐 interval cell tube；而 `M` 已经在直线路径上计算中心差分。下一版应把 renderer 变成共享 state object：

```text
StraightState
  positions
  scalar n
  central gradient
  projection axes
  cell ids / margins
```

证书、`M` 输出和 risk feature 都从同一 state 读取。实现目标：

1. cell derivative bounds 全部预计算成 tensor；
2. ray × interval 索引用 gather 向量化，消除 Python 双层循环；
3. support 未启用时不计算 root certificate；
4. certificate 只在场或相机状态改变后刷新，不要每个 estimator replica 重跑；
5. 分开报告 amortization horizon `1/64/1024`。

冻结门：3/3 基础 rig 的端到端 p90 / full-high p10 `<0.90`；若做不到，当前路由在 Mac 和类似 GPU kernel 上都不应宣传加速。

## 候选 2：一阶轨迹变分残差预测器

当前风险分数只看曲率大小，没有预测输出残差的方向和抵消。沿 straight medium path 写：

\[
F(r,d)=\frac{(I-dd^T)\nabla n(r)}{n(r)}.
\]

在 `(r0,d0)` 线性化：

\[
\delta r'=\delta d,
\qquad
\delta d'=F_0+A(s)\delta r+B(s)\delta d,
\]

其中 `A=partial F/partial r`、`B=partial F/partial d`。沿直线积分这个小型变分系统，得到预测修正：

\[
\widehat{\Delta}_{traj}
=\int P[A\delta r+B\delta d]ds.
\]

两种用法：

- `risk_i = ||Delta_hat_traj,i||`，只改 routing probability；
- `M1=M+Delta_hat_traj`，把它直接当更强控制变量，再用 HT 无偏纠正 `H-M1`。

必须比较：零阶曲率分数、当前 local-deviation proxy、变分 proxy、oracle `|H-M|`。冻结开发门：

- 每 rig Spearman `rho≥0.5`；
- proxy / constant-pi exact variance `≤0.90`；
- `Var(H-M1)/Var(H-M)≤0.5`；
- predictor 成本计入后 work-variance 仍改善。

曲光线伴随和轨迹线性化并不新。潜在贡献只能是把它做成 BOST 低成本 residual predictor，并与 fail-closed 有效性门和无偏训练合同连起来。

## 候选 3：cross-fitted residual router

若解析变分 proxy 不够，使用小模型，不先上 DeepONet/FNO。特征只允许来自 high 之前：

- `M-L0` 的向量、范数与方向；
- straight path 的曲率矩、Hessian tube bound；
- aperture radius、view angle、frustum/domain margin；
- support crossing count 与 transversality margin；
- pixel/PSF/path sampling level。

建议模型顺序：单调 GAM / isotonic binning → 小 MLP → operator router。训练与评估必须分离：

1. calibration phantom 拟合 residual quantile；
2. 独立 validation 冻结 inflation factor；
3. fresh test 只运行一次；
4. `pi` 对物理参数 stop-gradient；
5. unsafe/OOD/NaN 一律 `pi=1`；
6. 报逐 rig false-safe 与 Q95，不 pool 掉坏 rig。

只有当输入本身是跨实例函数（整幅 flow-off 图、相机函数或时间历史）时，DeepONet/FNO 才成为合理 router；现在逐 ray 小特征不需要先上算子网络。

## 候选 4：validated event certificate

如果何远哲现有代码的 mask/occupancy 会随网络更新，support crossing 真正改变执行路径，才值得做：

1. 沿 low ray 按网格 cell 分段；
2. 对 `f-tau` 与 `f+tau` 用 Bernstein/interval Newton 隔离全部根；
3. 根区间外证明不会新增 crossing；
4. 根区间内证明端点异号、非 grazing、唯一根；
5. 认证 event 顺序、AABB face、background hit 和连续 pupil box；
6. 使用 outward-rounded interval arithmetic 或 validated ODE enclosure。

当前 float64 sampled certificate 只能叫 development proxy。若实验室 renderer 没有活动 mask，这条支线应暂停。

## 候选 5：真实重建宿主

最终论文不能停留在 pupil-ray mean。需要将 candidate 放入 NeRIF/何远哲 4D BOST 宿主，并比较：

- `H` high-only curved cone-ray；
- `M` straight central；
- constant-pi HT；
- physics-validity-gated proxy；
- oracle upper bound；
- NeRIF / Neural RI Primitives / 何远哲 4D 表示。

主指标必须是 field relative-L2/H1、front geometry、held-out reprojection、逐 rig Q95、high/JVP/VJP 次数、端到端时间和显存。forward mean variance不能代替重建结果。

## 现在发给何远哲的七个问题

1. 当前 3D/4D BOST renderer 是否真的使用场依赖曲光线，还是 straight/paraxial ray？
2. mask/occupancy/support 是训练前冻结，还是随网络参数更新？它是否会改变采样或分支？
3. 每个 pixel 实际使用多少 pupil、PSF 和 path sample？forward/JVP/VJP 的主要耗时在哪里？
4. 能否给米制 ROI、背景距离、焦距、f-number、pixel pitch、相机位姿与畸变标定？
5. 最大偏折工况是多少 pixel？flow-off 重复图像、光流和标定的不确定度是多少 pixel？
6. 是否有故意留出的 view/time、独立温度/密度/PLIF 或其他物理端点？
7. 可以先给一组最小数据合同吗：一份 field/checkpoint、16–64 条 ray、当前 forward 输出和一个 VJP cotangent？

没有这些答案，下一轮只能继续做数值机制，不能判断现实意义。

## 论文标题暂定

比“Topology-Certified”更稳妥的标题是：

**Physics-Validity-Gated Multifidelity Ray Operators for Neural Background-Oriented Schlieren Tomography**

投稿前仍需做前向/后向引文链和专利检索；当前只能写“未检出完全重合工作”，不能写“首次”。
