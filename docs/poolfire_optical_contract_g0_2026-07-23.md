# PoolFire → BOST 的 G0 光学合同

> 当前判定：`G0-SMOKE = GO / G0-PHYSICS = HOLD`
>
> 推荐重建变量：`Δn`
>
> 本文允许固定空气 Gladstone-Dale 常数用于程序调试，但禁止据此宣称 PoolFire synthetic BOST 已经物理可信。

## 1. 为什么 warm start 先预测 Δn

C 路线的核心问题是：算子学习给三维 BOST 物理求解器提供初值，能否在相同终点精度下降低重建成本。

BOST 直接响应的是折射率梯度，不是 CFD 文件中的绝对密度。先把学习接口冻结为

\[
W_\theta(y,\mathcal G,\lambda,\text{reference})
\longrightarrow \Delta n_0,
\]

然后由正式物理 solver 从 `Δn0` 收敛到同一终点精度，可以把两件事分开：

1. warm start 是否真的减少物理迭代；
2. `rho ↔ n` 的组分、波长和参考态假设是否正确。

如果一开始直接预测 rho，尚未确认的 Gladstone-Dale 假设会污染算法结论。

## 2. rho → n 不能只写一个常数

稀薄透明气体常用 Gladstone-Dale 近似：

\[
n-1 \approx \sum_i \rho_i K_i(\lambda)
= \rho\sum_i Y_i K_i(\lambda)
= \rho K_{\rm mix}(\lambda,\mathbf Y).
\]

因此

\[
\nabla n
= K_{\rm mix}\nabla\rho
+ \rho\nabla K_{\rm mix}.
\]

固定空气常数只保留第一项，会删除反应流中由组分变化产生的第二项。参考态也应写成

\[
\Delta n
= \rho K_{\rm mix}
-\rho_{\rm ref}K_{\rm ref},
\]

只有 `Kmix = Kref = Kconst` 时，才可化为 `Kconst(ρ-ρref)`。

PoolFire 当前只列出 `CH4/CO2/H2O/O2` 四个通道，但尚未确认它们是质量分数、摩尔分数、分密度还是其他预处理量，也没有权威说明 `N2/CO/其他组分` 如何闭合。因此不能擅自令 `N2 = 1-sum(...)`。

关键来源：

- [Gardiner et al., Refractivity of combustion gases](https://doi.org/10.1016/0010-2180(81)90124-3)：燃烧组分在常见激光波长下的摩尔折射率与色散；
- [Qin et al., premixed methane flame interferometry](https://www.sciencedirect.com/science/article/pii/S0010218001003388)：局部混合物折射率与组分假设会进入火焰反演；
- [Wanstall et al., real-gas refractive-index models](https://doi.org/10.1016/j.combustflame.2019.12.023)：高压/近临界条件下理想混合律的适用边界；
- [NIST air refractive-index model](https://www.nist.gov/publications/index-refraction-air)：适合环境空气参考，不是高温燃烧产物模型。

## 3. 从折射率场到观测

曲线射线的几何光学方程可写为

\[
\frac{d}{ds}(n\mathbf t)=\nabla n,
\qquad
\frac{d\mathbf t}{ds}
=(\mathbf I-\mathbf t\mathbf t^\mathsf T)\nabla\ln n,
\]

其中 `s` 是弧长，`\mathbf t` 必须是单位切向量。

在小偏折、直线射线近似下：

\[
\boldsymbol\varepsilon_\perp
\approx \frac{1}{n_{\rm ref}}
\int
(\mathbf I-\mathbf t_0\mathbf t_0^\mathsf T)
\nabla\Delta n\,ds.
\]

薄相位物体近似下，偏折角还要通过背景距离、放大率和有效像元尺寸转换成像素位移。必须区分：

- 偏折角；
- 背景板端点位移；
- 传感器平面长度；
- 像素位移；
- 归一化图像坐标。

更稳妥的三维生成端合同是：对每条光线直接比较有流场与 reference 场在背景平面的端点，再用背景平面到像素的标定 Jacobian 转成图像位移。这样不需要把整个三维分布体压成一个模糊的全局 `ZD`。

参考：

- [Raffel, Background-oriented schlieren techniques](https://doi.org/10.1007/s00348-015-1927-5)
- [Grauer, Bayesian Methods for Gas-Phase Tomography](https://uwspace.uwaterloo.ca/items/7134a15f-578a-4e08-bd15-2bda1bc48b27)
- [NeRIF 开放全文](https://arxiv.org/html/2409.14722v2)

## 4. Straight-ray 不是默认真理

NeRIF 的数值生成使用四阶 Runge-Kutta 曲线追迹；其重建部分使用预计算投影与反向采样。对 PoolFire 不能预设直线射线一定足够，应在同一 `Δn` 上比较 straight 与 curved：

1. 比较背景端点或像素位移差的 p50/p95/max；
2. 与光流不确定度 `σflow` 比较，而不是自定一个脱离装置的百分比；
3. 将积分步长减半，检查端点和偏折角收敛；
4. 缩放 `Δn → 0`，验证 curved 结果趋近 straight；
5. 统计离开背景板、超出视场、映射折叠和 caustic；
6. 只有 curved-straight 差异稳定低于测量不确定度，straight-ray 才能通过正式 G0。

曲线射线会让光路依赖未知折射率场，正问题因此变成非线性。邻近参考：

- [CVPR 2024: Single View Refractive Index Tomography with Neural Fields](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_Single_View_Refractive_Index_Tomography_with_Neural_Fields_CVPR_2024_paper.html)
- [Adjoint Nonlinear Ray Tracing](https://imaging.cs.cmu.edu/adjoint_nonlinear_tracing/)

## 5. 两级 forward 证据

### Tier A：程序与公式调试

- 固定且明确标注的空气 `K`；
- 常数 `Δn` 场偏折为零；
- 线性 `Δn` 场的偏折方向、符号和尺度符合解析式；
- RK4 步长收敛；
- forward replay 与 adjoint/JVP/VJP 数值检查。

Tier A 可以启动，但结果只能写成 smoke/debug。

### Tier B：物理可信 synthetic BOST

- high-resolution、组分相关 `Δn`；
- 独立 curved-ray 生成器；
- 背景纹理渲染；
- 畸变、PSF、裁剪、采样、传感器噪声与遮挡；
- 从参考/受扰图像运行光流；
- 使用不同网格、积分步长或实现进行三维反演。

只给位移加 Gaussian noise 不能消除 inverse crime。生成和反演使用同一离散矩阵、网格、插值与无噪声投影时，只能算单元测试。

参考：

- [Wirgin, The inverse crime](https://arxiv.org/abs/math-ph/0401050)
- [Kaipio & Somersalo, discretization and model reduction in statistical inverse problems](https://research.aalto.fi/en/publications/statistical-inverse-problems-discretization-model-reduction-and-i/)

## 6. 师兄工具当前为什么只能做 smoke

私有 notebook 初审显示：

- 有九视角相机位姿和曲线积分；
- 可保存光线路径和二维偏折类输出；
- `2.48/10000` 看起来像 Gladstone-Dale 量级，但来源、介质和单位未确认；
- `2.2/3200` 看起来像物理长度/像素比例，但语义未确认；
- ray direction 未明确单位归一，near/far 参数未必是物理弧长；
- 没有确认最终背景 warping/图像渲染；
- 不同数据分支对 field callable 的语义不一致。

所以允许用它做常数场、线性场和步长收敛 smoke，不允许直接批量生成论文训练标签。

## 7. 必须问何远哲师兄

可直接发送：

> 师兄，我准备把 PoolFire → synthetic BOST → 3D warm start 的物理合同锁死，需要确认：
>
> 1. 组里最终重建变量是 n、Δn、rho 还是 Δrho？
> 2. 使用哪套 Gladstone-Dale 系数，来源和单位是什么？固定空气还是组分相关？
> 3. 光源/滤光片的中心波长和带宽是多少？
> 4. rho_ref/n_ref 是 flow-off、环境边界还是指定常数？
> 5. 模拟程序输出的是像素位移、传感器长度、背景板长度、偏折角还是射线端点？
> 6. Lb/Ld/ZD/ZA 的精确定义和单位是什么？每个视角是否不同？
> 7. 能否提供内参、畸变、外参、背景平面、像元尺寸、裁剪缩放和轴约定？
> 8. 正式 forward 是 straight-ray 还是 curved-ray？积分器、步长和插值是什么？
> 9. 真实流程是直接用位移图，还是先渲染 BOS 图像再跑光流？
> 10. 现有重建 callable、A/A^T 或 JVP/VJP、基线 solver 和停止条件是什么？
> 11. 典型位移范围和光流标定误差是多少？straight-ray 是否验证过？
> 12. warm start 最终按哪些量验收：终点 field error、迭代数、运行时间、显存、调用次数还是 held-out reprojection？

PoolFire 自身的单位、domain、组分和生成版本若师兄不知道，应转问 REALM 作者，而不是由项目猜测。

## 8. 当前门禁

允许：

- 用固定 `K` 做明确标注的 Tier A smoke；
- 优先实现 `y, geometry, reference → Δn0` 的模型接口；
- 为 straight/curved、step size、单位和端点语义写测试。

禁止：

- 把固定空气 `K` 当作 PoolFire 火焰的已验证光学模型；
- 把 `2.48×10^-4` 的量级相似当作来源证明；
- 直接把四个未确认组分通道拼成 `Kmix`；
- 把偏折角、端点位移和像素位移混为一谈；
- 在 Tier B 与独立反演算子完成前声称物理重建成功。
