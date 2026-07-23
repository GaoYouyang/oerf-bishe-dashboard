# v5b 可辨识性与共享校准：零基础学习导图

> **项目状态：`DESIGN_ONLY_NOT_PREREGISTERED`。**
>
> **总判决 `[E0-v5b]`：v5b 目前只是由 v5a 失败导出的可证伪设计；当前没有可引用的预注册、
> 冻结、独立验证或首开性能/安全证据。即使存在并行原型代码，也不等于通过门槛。本文是
> 学习路线和实验前置说明，不是成功报告。**

面向读者：学过大学物理和微积分，但还不熟悉层析重建、统计推断与自动微分的物理本科生。

建议用时：连续 4 周，每天 1.5--2.5 小时。先手算，再写最小代码；不以“看懂文字”代替
“能独立算出来、说出来、测出来”。

---

## 0. 先学会看证据标签

本文所有**项目研究结论**都带证据等级。等级只在所写明的范围内有效，不能跨范围升级。

| 等级 | 含义 | 可以怎样写 | 不能怎样写 |
| --- | --- | --- | --- |
| `[E4]` | 数学恒等式、标准定义，或已有公开来源直接支持的物理/方法事实 | “Schur complement 是标准块消元工具” | “所以 v5b 一定有效” |
| `[E3]` | 本仓库预注册后首开的锁定合成实验事实 | “v5a 在该锁集五项门槛只过两项” | “真实 BOST 中也必然如此” |
| `[E2]` | 本仓库锁后诊断、开发实验或有限合成观察 | “诊断与 morphology confounding 一致” | “已经证明唯一因果机制” |
| `[E1]` | 有数学/物理动机、但尚待新实验检验的研究假设或协议推断 | “共享参数可能减少单场混淆” | “方法已改善风险” |
| `[E0]` | 没有可引用的冻结结果，或关键真实数据/接口未知 | “待验证、待确认” | 任何成功、优越或安全声明 |

> **范围规则 `[E4]`：** 高等级不等于大范围。一个 `[E3]` 的合成锁集事实，不能自动变成真实
> 装置结论；一个 `[E4]` 的数学公式，也不能自动证明输入模型与真实光学相符。

### 当前最重要的五条证据

1. **有限孔径会让理想单光线模型变成对一束子光线的积分/平均**，并引入景深相关失配。
   `[E4：公开有限孔径 BOS 来源支持]`
2. v5a 明确使用了 `A_truth != A_recon`，其锁集五项预注册门槛只通过两项；总体
   `>1%` harm 为 `25%`，接受条件 harm 为 `37.5%`。`[E3：v5a 合成锁集]`
3. v5a 锁后诊断显示孔径估计强烈撞向候选端点，并随流场 family 改变；这与
   “体场形态和孔径相混淆”一致，但不是唯一因果证明。`[E2：锁后诊断]`
4. “多个场共享一个装置参数 + profile objective + regularized profile-Schur + held-out
   camera”是
   v5b 的待检验方案。`[E1：协议假设]`
5. v5b 是否能估准孔径、改善三维场、降低尾部伤害或迁移到真实 BOST，目前全部未知。
   `[E0-v5b]`

---

## 1. 一张总导图：从相机孔径走到“接受还是回退”

```text
真实装置
  有限孔径、焦平面、机架参数 theta_*
        |
        v
真实观测 y_s = A_truth(theta_*) x_s + noise
        |
        | 电脑只能调用近似 A_recon(theta)
        v
对每个候选 theta，先重建每个场 x_s(theta)
        |
        v
profile objective Phi(theta)
  “允许 x_s 各自调整后，哪个共享 theta 仍最合理？”
        |
        v
Jacobian / JVP / VJP
  分别测“改 x”与“改 theta”怎样改变观测
        |
        v
regularized profile Gauss--Newton / Schur complement
  去掉能被 x_s 变化解释的方向
        |
        +---- 信息太小 ----> ABSTAIN_UNIDENTIFIABLE
        |                       回退 nominal cone / pinhole
        v
outer held-out camera 检查候选是否优于 fallback
        |
        +---- 不稳定/变差 ---> 回退
        |
        v
只在可辨识且外层一致时接受 candidate
        |
        v
报告 coverage 与 conditional risk = P(harm | accept)
        |
        v
final audit camera 首开；仍不能把重投影当作三维真值
```

**这张图表达的是实验协议，不是已跑通的算法。`[E1：设计链条；E0：尚无 v5b 结果]`**

---

## 2. 第一块地基：BOS、BOST 与有限孔径

### 2.1 BOS/BOST 在测什么

- **BOS（Background-Oriented Schlieren）**：先拍背景点阵参考图，再拍光线穿过非均匀
  折射率场后的图。图案位移反映光线偏折。
- **BOST（BOS Tomography）**：从多个相机/角度取得位移观测，反演三维折射率或密度场。
- 把三维场摊平成向量 `x`，把所有相机位移摊平成向量 `y`，最简线性写法是

  \[
  y=Ax+\epsilon.
  \]

这里 `A` 不是“一个神经网络名字”，而是从三维场到相机观测的 **forward operator**。

**研究边界 `[E4]`：** 上式是弱偏折/离散化下的入门模型。真实 BOST 可能包含非线性光路、
空间变化 PSF、遮挡和更复杂噪声；不能因会算 `Ax` 就声称复现了真实装置。

### 2.2 针孔为什么只是近似

理想针孔模型把一个 detector pixel 对应成一条代表性光线：

\[
y_i\approx \int_{\text{one ray }i} g(x(\mathbf r))\,d\mathbf r.
\]

有限孔径下，不同入口瞳位置会产生不同子光线，pixel 接收到的是一束光的综合：

\[
y_i\approx
\frac{1}{Q}\sum_{q=1}^{Q}
\int_{\text{sub-ray }(i,q,\theta)}g(x(\mathbf r))\,d\mathbf r.
\]

`theta` 可以包含 effective aperture、焦平面偏移或少量 pupil correction。孔径越大并不等于
“所有信息都更差”；它改变的是不同深度、不同子光线对观测的混合方式。

**物理结论 `[E4：公开有限孔径 BOS 工作]`：** 有限孔径和景深会造成相对于理想针孔模型的
forward-model mismatch。**项目结论 `[E0]`：** 当前尚不知道真实 OERF 装置里这种 mismatch
的大小、参数是否完整记录、或是否足以支持 v5b 校准。

### 2.3 为什么必须写 `A_truth != A_recon`

合成实验可以明确区分：

\[
y=A_{\text{truth}}(\theta_*)x_*+\epsilon,
\qquad
\hat x=\operatorname{Recon}(y;A_{\text{recon}}(\theta)).
\]

- `A_truth`：负责生成观测，代表更密子光线、更细路径采样或独立代码路径。
- `A_recon`：重建器真正看得到、算得动的近似模型。
- 两者完全相同，会形成 inverse crime：实验可能只证明算法能反解自己生成的数据。
- 两者不同，也不自动等于“真实”；它只把一种模型失配显式放进测试。

**本仓库事实 `[E3]`：** v5a 已在规定的线性弱偏折小模型中分离 `A_truth` 与
`A_recon`。**边界 `[E3]`：** 这不是完整 cone-ray、曲线光路或真实 BOST 复现。

### 2.4 一个两维反例：残差为零，场仍然错

设

\[
A_{truth}=\begin{bmatrix}0.95&0.05\\0.05&0.95\end{bmatrix},\quad
A_{recon}=I,\quad
x_*=\begin{bmatrix}1\\0\end{bmatrix}.
\]

于是 `y=[0.95,0.05]^T`。使用 `A_recon=I` 重建得到 `x_hat=y`：

- 重投影残差：`||y-I x_hat||=0`；
- 场误差：`||x_hat-x_*||=sqrt(0.05^2+0.05^2)>0`。

**数学结论 `[E4]`：** 对错误算子，观测拟合很好不推出未知场正确。这个反例也解释了为何
held-out residual 有价值但不能冒充三维真值。

---

## 3. 第二块地基：nuisance parameter 与共享校准

### 3.1 nuisance parameter 不是“没用的参数”

**nuisance parameter（干扰参数）**的定义取决于当前问题：它会影响观测，但不是这一步最想
报告的量。

- 当问题是“估计装置孔径 `theta`”时，每个未知三维场 `x_s` 是高维 nuisance。
- 当最终科学问题是“重建三维场 `x_s`”时，未知光学参数 `theta` 又可被视为 nuisance。

所以 nuisance 不是噪声，也不是可以删除的变量。正确做法是联合估计、积分掉、profile 掉，
或把它的不确定性传播到最后结论。

**定义 `[E4]`：** nuisance 是“影响数据但不是当前目标”的未知量；角色会随问题改变。

### 3.2 为什么 v5b 想让 `theta` 在多个场之间共享

同一 acquisition block 内写成

\[
y_s=A(\theta)x_s+\epsilon_s,\qquad s=1,\ldots,S.
\]

`x_1,...,x_S` 可以随流场变化，而 f-number、焦距和固定机架几何通常应在硬件没有漂移的
时段内共享。单场若允许 `x_s` 与 `theta_s` 一起自由变化，两者容易互相代偿；多场共享
`theta`，相当于要求同一个光学解释同时面对多种场。

**v5b 假设 `[E1]`：** 装置级共享可能减少“每个场都猜出不同孔径”的混淆。
**未知项 `[E0]`：** 组内硬件是否真的在多个场/时间帧间稳定、漂移时间尺度多长、哪些参数
应该共享，尚需元数据和实验记录确认。

### 3.3 “共享”不等于“所有东西相同”

应共享的是有硬件依据的低维参数；不应强行共享每个场的三维结构、噪声 realization 或坏点
mask。建议在 manifest 中显式记录：

```text
rig_id / acquisition_block_id / camera_id / timestamp
f_number / focal_length / object_distance / focus_setting
field_id / family / noise_seed / mask_id
```

**协议建议 `[E1]`：** 共享边界必须先由硬件与采集流程定义，再由 leave-one-block-out 检查，
不能看完性能后才重划 acquisition block。

---

## 4. 第三块地基：profile objective

### 4.1 先“内层重建场”，再“外层比较光学参数”

对固定 `theta`，先让每个场取得自己的最优重建：

\[
\hat x_s(\theta)=\arg\min_{x_s\ge 0}
\left\|W_s^{1/2}\bigl(y_s-A(\theta)x_s\bigr)\right\|_2^2
+\lambda R(x_s).
\]

再把这些内层最优值放入外层：

\[
\Phi(\theta)=
\sum_{s=1}^{S}
\left[
\left\|W_s^{1/2}\bigl(y_s-A(\theta)\hat x_s(\theta)\bigr)\right\|_2^2
+\lambda R\bigl(\hat x_s(\theta)\bigr)
\right]
+\gamma\|\theta-\theta_{meta}\|_{\Sigma_{meta}^{-1}}^2.
\]

这就是 **profile objective** 的直觉：对每个 `theta`，允许 nuisance `x_s` 做到它能做到的
最好，再比较剩下的目标值。它不是把 `x_s` 假装已知。

### 4.2 三个必须分开的量

1. **data profile curvature**：数据在去掉 `x` 的可解释方向后，还剩多少 `theta` 信息。
2. **metadata prior curvature**：f-number、PSF 等先验对 `theta` 的约束有多强。
3. **总 posterior/regularized curvature**：前两者相加后的数值稳定性。

**方法边界 `[E4]`：** 若只因强先验而曲率变大，不能写成“数据辨识出了孔径”。v5b 实验应
并列报告 data-only 与 prior-augmented 结果。**是否足以形成有效 gate `[E0]`：尚未测试。**

### 4.3 profile objective 也可能选错

profile 只保证“在规定模型、正则和候选集合里完成了嵌套优化”。若 `A_recon` 整体错、
内层优化没收敛、正则过强，或多组 `theta` 有近似相同的 profile 值，外层最小点仍可能错。

**数学/方法结论 `[E4]`：** objective 最小不等于物理真值已被识别。
**v5b 结论 `[E0]`：** 当前没有 profile curvature、孔径误差或跨 family 稳定性的运行结果。

---

## 5. 第四块地基：Jacobian、JVP 与 VJP

设 forward 为

\[
F(x,\theta)=A(\theta)x,\qquad F:\mathbb R^{n+p}\to\mathbb R^m.
\]

### 5.1 Jacobian：一张“局部变化说明书”

在当前点附近，微小改变量满足

\[
\delta y\approx J_x\delta x+J_\theta\delta\theta,
\]

其中

\[
J_x=\frac{\partial F}{\partial x}=A(\theta),\qquad
J_\theta\delta\theta=
\left(\frac{\partial A}{\partial\theta}[\delta\theta]\right)x.
\]

- `J_x` 的列：某个体素轻微变化，会怎样改变所有相机观测。
- `J_theta` 的列：某个光学参数轻微变化，会怎样改变所有相机观测。
- 若两类列空间高度重合，改场和改孔径在观测上难以区分。

### 5.2 JVP：给一个输入方向，看输出怎样动

\[
\operatorname{JVP}(v)=Jv.
\]

用途：不形成完整 Jacobian，也能计算“沿 `v` 扰动参数，观测的一阶变化”。可用中心有限差分
检查：

\[
Jv\approx\frac{F(z+h v)-F(z-h v)}{2h},\qquad z=(x,\theta).
\]

### 5.3 VJP：把观测端信号反传到输入端

\[
\operatorname{VJP}(u)=J^Tu.
\]

若损失为 `L=1/2 ||W^(1/2)(F-y)||^2`，梯度需要

\[
\nabla_z L=J^T W(F-y),
\]

这正是 VJP/adjoint 的典型用途。

### 5.4 二者绝不能混叫

| 工具 | 输入 | 输出 | 物理问题 |
| --- | --- | --- | --- |
| JVP `Jv` | 参数/体场方向 `v` | 观测方向 | “我这样改输入，图像怎样变？” |
| VJP `J^T u` | 观测残差/权重 `u` | 参数/体场梯度 | “这份图像误差该归因到哪些输入？” |

**数学定义 `[E4]`：** JVP 与 VJP 是同一 Jacobian 的不同乘法方向，不是同一个量。
**接口证据 `[E0-v5b]`：** 当前没有可引用的、已冻结且经独立检查的真实 NeRIF/TDBOST
forward 对 `x` 和 `theta` 的 JVP/VJP 证据；原型函数的存在不能替代
finite-difference/adjoint check。

### 5.5 最小手算例子

令

\[
F(x_1,x_2,\theta)=
\begin{bmatrix}
\theta x_1+x_2\\
x_1+\theta x_2
\end{bmatrix}.
\]

则

\[
J=
\begin{bmatrix}
\theta&1&x_1\\
1&\theta&x_2
\end{bmatrix}.
\]

在 `(x1,x2,theta)=(1,2,0.5)`：

\[
J=\begin{bmatrix}0.5&1&1\\1&0.5&2\end{bmatrix}.
\]

- 对 `v=(0.2,-0.1,0.3)`，应算得 `Jv=(0.3,0.75)`。
- 对 `u=(2,-1)`，应算得 `J^T u=(0,1.5,0)`。

---

## 6. 第五块地基：Fisher、Schur complement 与 profile 信息

### 6.1 从加权最小二乘的局部曲率开始

令 `W` 为噪声精度矩阵。局部 Gauss--Newton/Fisher 型块矩阵可写为

\[
G=
\begin{bmatrix}
G_{xx}&G_{x\theta}\\
G_{\theta x}&G_{\theta\theta}
\end{bmatrix}
=
\begin{bmatrix}
J_x^T WJ_x+\lambda H_R & J_x^T WJ_\theta\\
J_\theta^T WJ_x & J_\theta^T WJ_\theta
\end{bmatrix}.
\]

若把 `x` 当 nuisance，消去 `x` 后，`theta` 的局部 profile curvature 是 Schur complement：

\[
G_{\theta\mid x}
=G_{\theta\theta}
-G_{\theta x}G_{xx}^{-1}G_{x\theta}.
\]

也可写成投影形式：

\[
G_{\theta\mid x}
=J_\theta^T W
\left[I-J_x(J_x^TWJ_x+\lambda H_R)^{-1}J_x^TW\right]
J_\theta.
\]

严格来说，加入正则 Hessian 后更稳妥的称呼是“regularized local profile curvature”；只有在
对应概率模型与正则解释成立时，才直接称 Fisher information。

**数学结论 `[E4]`：** Schur complement 表示先允许 nuisance 方向作最优局部调整，再看目标
参数还剩多少二阶信息。**数值提醒 `[E4]`：** 实现中通常解线性方程，不显式求逆矩阵。

### 6.2 一个一眼看懂的二维手算

取 `W=I`、无正则，且只有一个 `x` 与一个 `theta`。

**情况 A：仍可区分**

\[
J_x=\begin{bmatrix}1\\0\end{bmatrix},\qquad
J_\theta=\begin{bmatrix}1\\1\end{bmatrix}.
\]

则

\[
G_{\theta\mid x}=2-1\times1^{-1}\times1=1.
\]

`theta` 的第二个观测分量不能被 `x` 变化解释，所以还剩信息。

**情况 B：完全混淆**

\[
J_x=\begin{bmatrix}1\\0\end{bmatrix},\qquad
J_\theta=\begin{bmatrix}2\\0\end{bmatrix}.
\]

则

\[
G_{\theta\mid x}=4-2\times1^{-1}\times2=0.
\]

改 `theta` 的观测效果完全落在改 `x` 的方向里，局部上不可辨识。

### 6.3 多场共享时怎样合并

若每个场有自己的 `x_s`，但共享 `theta`，可以对每场先消去 `x_s`，再累计 data-only
profile 信息：

\[
G_{\theta\mid x_{1:S}}^{data}
=\sum_{s=1}^{S}G_{\theta\mid x_s}^{data}.
\]

不同场若提供互补的观测方向，累计信息可能增加；若所有场都沿同一混淆方向变化，单纯增加
场数也不会创造新方向。

**机制假设 `[E1]`：** 多形态共享校准可能提供互补方向并改善 practical identifiability。
**当前证据 `[E0-v5b]`：** 尚无 v5b paired multi-scene 结果证明信息确实增加或能排序风险。

---

## 7. 第六块地基：identifiability 到底是什么

### 7.1 四个不要混在一起的层次

| 层次 | 问题 | 入门检查 |
| --- | --- | --- |
| structural/global identifiability | 无噪声、无限精度时，是否只有一个参数解释数据？ | 对称性、等价参数、全局搜索 |
| local identifiability | 当前解附近，小扰动是否能区分？ | Jacobian 秩、profile 最小特征值 |
| practical identifiability | 有限视角、噪声和算力下，估计是否稳定？ | bootstrap CI、扰动稳定性、boundary hit |
| model validity | 候选 forward 家族是否包含真实装置行为？ | 独立 renderer、真实 f-stop sweep、残差结构 |

高 regularized profile-Schur curvature 最多直接支持局部、模型内的可辨识性；它不自动证明
全局唯一、真实装置正确，
也不保证最终三维场误差小。

**方法结论 `[E4]`：** local identifiability、global identifiability 与 model validity 是不同命题。
**v5b 目标 `[E1]`：** 先把局部不可辨识样本拒绝，再用外层视角与新装置检查模型外风险。
**达成状态 `[E0]`：尚未达成。**

### 7.2 v5b 应看到什么才算“机制有信号”

以下都是**待预注册的候选判据**，不是已有结果：

1. profile score 与 aperture MAE、raw gain、harm 在未见 family/rig 上保持稳定排序；`[E1]`
2. 低 score 样本确实更常撞参数边界、跨 fold 不稳定或伤害更大；`[E1]`
3. shared `theta` 不再能被 field family 轻易预测；`[E1]`
4. 显式小矩阵与 matrix-free JVP/VJP 版在冻结容差内一致；`[E1]`

任何一条未由冻结实验运行并审计前都只能写“要检验”，不能写“已证明”。`[E0-v5b]`

---

## 8. 第七块地基：held-out camera 不是三维真值

### 8.1 三层相机拆分

```text
inner cameras       拟合 x 与 theta
outer held-out      比较 candidate 与 fallback；可用于冻结 gate
final audit camera  从不调参，只在首开时报告
```

若一条相机已经参与选 `theta`、选阈值或选模型，它就不再是同一次声明里的独立 final audit。

对 outer fold `H_f`，可比较白化重投影差：

\[
d_{s,f}=
\chi^2_{H_f}(\hat x_s,\hat\theta)
-\chi^2_{H_f}(x_{fallback},\theta_{fallback}).
\]

`d<0` 表示 candidate 在这条留出相机上比 fallback 拟合得更好。

### 8.2 held-out camera 能证明什么、不能证明什么

- 能检查：候选是否只记住 inner views；新视角观测一致性是否改善。
- 不能单独检查：三维场是否唯一正确；所有相机是否共享同一种 forward bias；下游物理量
  是否准确。
- 因此真实数据还应尽量加入 reference/flow-off、重复采集、边界/积分量、PLIF/front 或
  其他独立 trace。

**方法结论 `[E4]`：** 病态逆问题中，留出观测一致性不是未知场真值的逻辑等价物。
**v5b 设计 `[E1]`：** held-out camera 只作为安全证据的一层，不单独决定成功。
**真实可用性 `[E0]`：** 当前尚未确认组内能否封存足够的 inner/outer/final 相机。

---

## 9. 第八块地基：conditional risk 与选择性回退

### 9.1 先定义 accept，再谈风险

设 `gain>0` 表示 candidate 比 fallback 好，定义

\[
H=\mathbf 1[gain<-1\%],\qquad A=\mathbf 1[\text{accept candidate}].
\]

必须同时报告：

\[
\text{coverage}=P(A=1),
\]

\[
\text{overall harm}=P(H=1),
\]

\[
\text{conditional harm}=P(H=1\mid A=1).
\]

若大量危险样本被“回退=零增益”稀释，overall harm 会显得较小，而真正采用候选时的风险
可能很高。

### 9.2 用 v5a 的真实数字算一遍

v5a 锁集中：总样本 `36`，接受 `24`，其中 `9` 个接受样本退化超过 `1%`。

\[
coverage=24/36=66.7\%,
\]

\[
overall\ harm=9/36=25.0\%,
\]

\[
conditional\ harm=9/24=37.5\%.
\]

**锁集事实 `[E3-v5a]`：** v5a 的接受条件风险远未达到候选安全门槛，不能被平均
`+1.753%` gain 掩盖。**迁移边界 `[E3]`：** 这是规定合成锁集上的失败，不是 v5b 结果。

### 9.3 risk-coverage 的正确读法

- 阈值更严，coverage 通常下降；conditional risk 可能下降，也可能因 score 不会排序而不降。
- coverage `0%` 可以让“采用候选后的经验伤害”没有样本可算，但这不叫算法成功。
- 只报 accepted-only mean 也不够，还要报 p10、最大伤害、worst group 与置信上界。
- 观察到 `0` 个 harm 不等于真实 harm probability 为 `0`。若 `n` 个独立接受样本中零伤害，
  单侧 95% exact 上界为 `1-0.05^(1/n)`；要低于 `5%`，至少约 `59` 个独立接受样本。

**统计结论 `[E4]`：** 分母、独立样本单位与有限样本置信上界必须显式报告。
**v5b 安全性 `[E0]`：** 尚无接受样本，更没有确认性 conditional-risk 上界。

---

## 10. 四个最小代码锚点

下面代码只用于学习公式。它们不是 v5b 实现，也不产生研究证据。

### 10.1 看见 `A_truth != A_recon`

```python
import numpy as np

A_pinhole = np.eye(2)
A_shifted = np.array([[0.9, 0.1], [0.1, 0.9]])
A_truth = 0.5 * (A_pinhole + A_shifted)  # 两条子光线的玩具平均
A_recon = A_pinhole

x_true = np.array([1.0, 0.0])
y = A_truth @ x_true
x_hat = np.linalg.solve(A_recon, y)

print("reprojection residual:", np.linalg.norm(y - A_recon @ x_hat))
print("field error:", np.linalg.norm(x_hat - x_true))
```

预期：第一项为 `0`，第二项约为 `0.07071`。

### 10.2 用有限差分检查 JVP

```python
import numpy as np

def F(z):
    x1, x2, theta = z
    return np.array([theta * x1 + x2, x1 + theta * x2])

z = np.array([1.0, 2.0, 0.5])
v = np.array([0.2, -0.1, 0.3])
J = np.array([[0.5, 1.0, 1.0], [1.0, 0.5, 2.0]])

h = 1e-6
jvp_fd = (F(z + h * v) - F(z - h * v)) / (2 * h)
print(jvp_fd, J @ v)
```

### 10.3 用内积检查 VJP/adjoint

```python
u = np.array([2.0, -1.0])
left = u @ (J @ v)
right = (J.T @ u) @ v
print(left, right, abs(left - right))
```

双精度小问题应接近机器精度；真实非线性 forward 还必须做多方向、多步长检查。

### 10.4 手算 Schur，再让代码复核

```python
Jx = np.array([[1.0], [0.0]])
Jtheta = np.array([[1.0], [1.0]])
Gxx = Jx.T @ Jx
Gtt = Jtheta.T @ Jtheta
Gtx = Jtheta.T @ Jx
profile = Gtt - Gtx @ np.linalg.solve(Gxx, Gtx.T)
print(profile)  # [[1.0]]
```

---

## 11. 4 周逐日学习与练习

### 第 1 周：先把 BOST 看成一个会失配的逆问题

本周产物：一页手绘成像链、一个 2D toy notebook、一次不看稿的 5 分钟口述。

| 天 | 学习任务 | 手算练习 | 代码练习 | 当日通过标准 |
| --- | --- | --- | --- | --- |
| Day 1 | 画出背景点阵、折射率场、相机和位移观测；区分 BOS 与 BOST | 给 `A` 标出 `m×n`、`x` 的 `n×1`、`y` 的 `m×1` | 用 NumPy 计算一个 `3×2` 矩阵乘向量 | 不看稿解释 forward 与 inverse 各是什么 |
| Day 2 | 理解针孔单光线与有限孔径子光线束 | 两条子光线贡献为 `1.0, 0.6`，算等权平均和 `3:1` 加权平均 | 构造 2 个 `2×2` 子光线矩阵并求平均 operator | 能说出“孔径变化改变混合”而非只说“加噪声” |
| Day 3 | 学矩阵的列空间、秩与 null space | 求 `[[1,1],[2,2]]` 的秩，并找一个非零 null vector | 用 `np.linalg.svd` 打印 singular values | 能解释为何不同 `x` 可能给相同 `y` |
| Day 4 | 学最小二乘、残差与重建误差的区别 | 完成第 2.4 节两维反例 | 跑第 10.1 节代码，另换一个 `x_true` | 明确说出 residual 小不推出 field error 小 `[E4]` |
| Day 5 | 学噪声协方差与白化残差 | 对标准差 `1` 与 `2` 的两个观测，算 `W=diag(1,1/4)` 下的加权平方残差 | 比较普通 L2 与 whitened L2 的排序 | 权重大的一项确实对应更可信观测 |
| Day 6 | 理解 `A_truth != A_recon` 与 inverse crime | 自己设计一个 residual 为零但 field error 非零的 `2×2` 例子 | 扫描 mismatch 强度 `0--0.2`，画 field error 曲线 | 图中轴、单位、truth/recon 标签完整 |
| Day 7 | 周复盘；阅读 v5a 首开复盘第 1、4、5 节 | 复算 `9/36`、`9/24`、`24/36` | 把 v5a 三个比例写成 5 行小脚本 | 能准确说“v5a 失败 `[E3]`，v5b 无可引用结果 `[E0]`” |

### 第 2 周：把未知场当 nuisance，理解 profile 与共享参数

本周产物：一个 scalar/profile grid toy、一张共享/不共享参数对照表。

| 天 | 学习任务 | 手算练习 | 代码练习 | 当日通过标准 |
| --- | --- | --- | --- | --- |
| Day 8 | 区分 target parameter 与 nuisance parameter | 分别以“估孔径”和“估场”为目标，写出谁是 nuisance | 写两个函数名 `estimate_theta_profile`、`reconstruct_x_given_theta`，标清输入输出 | 不再把 nuisance 解释成 noise |
| Day 9 | 理解内层 `x_hat(theta)` 和外层 `Phi(theta)` | 对 `min_x (y-theta*x)^2 + lambda*x^2` 推出 `x_hat(theta)` | 在 101 个 `theta` 网格上画 profile curve | 手算极小点与代码一致到 `1e-6` |
| Day 10 | 理解正则为何会改变 profile | 分别取 `lambda=0,0.1,1` 算 `x_hat` | 叠画三条 profile curve | 能说明“更稳定”不等于“更接近 truth” |
| Day 11 | 理解多个场共享一个 `theta` | 给两场各算独立 `theta_s`，再算共享目标的最小点 | 合成 5 个 `x_s`，比较 per-scene 与 shared estimate 方差 | 不把共享 `theta` 写成共享 `x_s` |
| Day 12 | 学 metadata prior 与物理可行集合 | 算 `gamma(theta-theta_meta)^2` 如何移动极小点 | 分别输出 data-only 与 prior-augmented curve | 能指出“先验锁定”不等于“数据辨识” `[E4]` |
| Day 13 | 学 boundary hit 与参数化 | 比较截断到 `[0,0.2]` 和 sigmoid 参数化的结果 | 记录 100 次噪声实验的 boundary-hit rate | boundary hit 单独报告，不包装成高置信度 |
| Day 14 | 周小项目：shared profile grid | 手算一个两场、三个候选 `theta` 的完整表 | 输出每场 inner loss、总 profile、估计 `theta` 与真值误差 | 能从原始表复核最终选择；不声称这是 v5b 结果 |

### 第 3 周：用 JVP/VJP 和 Schur complement 判断局部可辨识性

本周产物：finite-difference 测试、adjoint 测试、两种 Schur 算法一致性报告。

| 天 | 学习任务 | 手算练习 | 代码练习 | 当日通过标准 |
| --- | --- | --- | --- | --- |
| Day 15 | 对标量函数、向量函数求导；确认 Jacobian 形状 | 推导第 5.5 节 `J` | 手写 analytic `J` 并与数值差分比较 | 每个矩阵维度都能口头解释 |
| Day 16 | 学 JVP 与 directional derivative | 算第 5.5 节 `Jv=(0.3,0.75)` | 扫 `h=1e-1...1e-8`，画差分误差 U 形趋势 | 找到误差下降后受舍入影响的区间 |
| Day 17 | 学 VJP、adjoint 与梯度 | 算 `J^T u=(0,1.5,0)` | 随机 20 组 `u,v` 检查 `<u,Jv>=<J^Tu,v>` | 双精度 toy 最大相对误差 `<1e-10` |
| Day 18 | 从噪声模型理解 Fisher/Gauss--Newton | 对一个 `J=[1,2]^T` 算 `J^TWJ` | 改变噪声标准差，观察信息量变化 | 能说出低噪声观测为何权重大 |
| Day 19 | 学块矩阵与 Schur complement | 完成第 6.2 节两种情况 | 用 `solve` 复核，不写 `inv(Gxx)` | 可辨识例得 `1`，混淆例得 `0` |
| Day 20 | 区分 local、global、practical identifiability | 为 `y=theta^2` 分析 `theta` 与 `-theta` 的全局歧义 | 画 `theta^2` 在 `0` 与 `1` 附近的局部导数 | 能解释局部有信息仍可能全局多解 |
| Day 21 | 周小项目：显式版 vs matrix-free 思维 | 画出“解 `Gxx z=Gxθ`，再算 Schur”的步骤 | 随机小矩阵比较显式 Schur 与投影/线性求解版 | 相对差 `<1e-8`；报告 condition number |

### 第 4 周：held-out、条件风险与诚实 Go/No-Go

本周产物：一个无泄漏 split manifest、risk-coverage 图、5 分钟模拟答辩。

| 天 | 学习任务 | 手算练习 | 代码练习 | 当日通过标准 |
| --- | --- | --- | --- | --- |
| Day 22 | 区分 inner、outer、final audit camera | 给 6 条相机设计 3 个 outer folds，并封存 1 条 final audit | 写索引断言：三类 view 集合两两不交 | 任一相机重复使用时测试失败 |
| Day 23 | 学 held-out whitened residual | 对 candidate/fallback 各算两条 outer camera 的 `d_f` | 输出 per-fold 差值，不只输出平均 | 能解释 held-out 不是 field GT `[E4]` |
| Day 24 | 学 coverage、overall risk、conditional risk | 复算 v5a 的 `66.7%/25%/37.5%` | 用布尔数组复算三者 | 分母完全正确；明确这是 v5a `[E3]` |
| Day 25 | 学 risk-coverage curve 与拒绝 | 给 6 个 score/gain，手动按阈值排序 | 画 coverage 对 conditional harm/p10 | 不把 coverage `0%` 标成成功 |
| Day 26 | 学 paired factorial 与 cluster 独立性 | 为 `rig×family×aperture×K` 画配对表，圈出独立 cluster | 生成 manifest 并检查 development/lock 的 rig/family/seed 互斥 | 不把同一 field 的多噪声副本当独立样本 |
| Day 27 | 做一次盲 Go/No-Go 演练 | 根据一张虚构结果表逐项判 Gate A/B | 把阈值写死后再读取模拟 lock；失败不得重调 | 能保留失败结论，不移动门槛 |
| Day 28 | 总复盘与模拟答辩 | 不看稿画出第 1 节总导图和第 6.2 节 Schur 例子 | 一键运行所有 toy tests，保存终端摘要 | 通过第 12 节全部标准；结尾必须说 v5b `[E0]` |

---

## 12. 四周通过标准

这些是**学习门槛**，不是 v5b 性能门槛，也不能作为论文结果。

### 12.1 概念通过

在不看文档时，能用 8 分钟连贯解释：

1. 有限孔径为何不是简单“再加一点高斯噪声”；
2. `A_truth != A_recon` 与 inverse crime 的关系；
3. 为什么估 `theta` 时 `x_s` 是 nuisance；
4. profile objective 的内层和外层分别优化什么；
5. JVP 与 VJP 的输入/输出方向；
6. Schur complement 去掉了哪类变化；
7. local identifiability 为什么不等于真实正确；
8. held-out camera 为什么不是三维真值；
9. coverage 与 `P(harm|accept)` 为什么要一起报；
10. 为什么 v5a 失败不能被改名成 v5b 成功。

通过线：10 题至少 9 题无关键概念错误；第 5、6、8、9、10 题必须全对。

### 12.2 手算通过

- 独立完成第 2.4 节 residual/field-error 反例；
- 独立算对第 5.5 节 JVP 与 VJP；
- 独立算对第 6.2 节两个 Schur 例子；
- 独立算对 v5a 的 coverage、overall harm、conditional harm；
- 能从 `0` harm、`n=59` 算出单侧 95% 上界约为 `4.95%`。

通过线：五项全部正确，并能说明每个数的分母和适用范围。

### 12.3 代码通过

- toy JVP 的中心差分相对误差在合理步长下 `<1e-5`；
- 20 组 toy adjoint inner-product check 最大相对误差 `<1e-10`；
- 显式 Schur 与线性求解版相对差 `<1e-8`；
- inner/outer/final view 索引两两不交；
- risk 代码同时输出 sample count、accepted count、harm count 与三个比例；
- 任意改变 final audit camera 数据，不得影响拟合、阈值和 accept 决策。

这些容差只用于双精度小矩阵教学。真实规模容差必须根据离散化、迭代误差和精度单独冻结。
`[E4：数值分析边界；E0：v5b 尚无冻结实现容差]`

### 12.4 研究诚实性通过

看到任何结果表时，能先写证据等级，再写结论；至少主动指出：

- synthetic/real 的范围；
- development/lock 的角色；
- overall/accepted-only 的分母；
- mean/tail/worst-group 的区别；
- candidate/fallback/oracle 的角色；
- v5b 当前仍是 `[E0]`。

---

## 13. 常见误解与纠正

| 常见误解 | 为什么错 | 正确说法 |
| --- | --- | --- |
| “有限孔径就是多一点噪声。” | 它会系统性改变光线混合和 forward operator | 它首先是 operator mismatch；噪声是另一层 `[E4]` |
| “`A_truth` 就是真实实验的绝对真理。” | 合成 truth 仍由某个 renderer 定义 | 它只是生成端相对于重建端的已知参考 `[E4]` |
| “重投影 residual 最小，孔径就最准。” | 错误体场可替错误孔径吸收误差 | residual 是必要诊断，不是参数真值证明 `[E4]` |
| “nuisance parameter 就是噪声。” | nuisance 是未知但系统性影响观测的变量 | 估孔径时，整个三维场都可能是 nuisance `[E4]` |
| “profile 掉 `x` 就等于知道了 `x`。” | profile 是对每个 `theta` 重优化 `x` | 它承认 `x` 未知，并比较内层最优值 `[E4]` |
| “强 metadata prior 得到窄 CI，说明数据很可辨识。” | 窄区间可能主要来自先验 | data-only 与 prior-augmented curvature 分开报 `[E4]` |
| “JVP 和 VJP 都是自动微分，所以一样。” | 一个算 `Jv`，一个算 `J^Tu` | 用形状和 inner-product test 区分 `[E4]` |
| “必须存完整 Jacobian 才能做 Fisher。” | 大问题可用 JVP/VJP、HVP 与迭代线性求解 | 先在小矩阵验证，再 matrix-free `[E4]` |
| “profile-Schur 大就保证全局唯一。” | 它通常只是当前点、当前模型内的局部曲率，而且会受正则影响 | 还要查正则尺度、全局多解、边界、噪声与模型失配 `[E4]` |
| “held-out camera 就是 ground truth。” | 多个相机可能共享同一 forward bias | 它是独立观测检查，不是三维真值 `[E4]` |
| “outer camera 和 final audit camera 可以复用。” | 用于调 gate 后就不再独立 | inner/outer/final 的声明角色必须分开 `[E4]` |
| “总体 harm 低就安全。” | 回退样本会稀释真正采用候选时的风险 | 同时报 overall harm 与 accepted-conditional harm `[E4]` |
| “零接受、零伤害就是完美安全。” | 方法没有提供任何 candidate 覆盖 | coverage `0%` 是拒绝一切，不是优越性 `[E4]` |
| “v5a 平均 gain 为正，所以差一点成功。” | p10、harm 和 accepted-only tail 明显失败 | v5a 五项只过两项，判定失败 `[E3-v5a]` |
| “换个阈值就能救 v5a。” | 在已打开 lock 上调阈值是 test overfitting | 新规则必须新命名、新 development、新 lock `[E3/E4]` |
| “多个场共享 `theta`，v5b 就会成功。” | 共享假设可能被硬件漂移或同向混淆破坏 | 这只是待证伪机制 `[E1]`，性能仍为 `[E0]` |
| “用了 Fisher/Schur 就有算法创新。” | 它们是标准数学工具 | 贡献若存在，只能来自 BOST 特定证据与验证 `[E4/E1]` |
| “NeRIF/TDBOST 会自动修复错误光学算子。” | 表示能力不等于模型正确 | 必须另验 forward、JVP/VJP 与真实成像失配 `[E1/E0]` |
| “oracle true operator 可以部署。” | oracle 使用实验时不可得的真值信息 | 它只给 headroom，不是部署基线 `[E4]` |
| “60 场 pilot 没出事故就证明安全。” | 接受样本数和置信上界可能远远不够 | pilot 先证伪；确认性安全需独立样本与上界 `[E4]` |

---

## 14. v5b 的 Go/No-Go 应怎样读

以下是协议目标，不是完成状态。

| Gate | 真正要问的问题 | 当前状态 | 允许的结论 |
| --- | --- | --- | --- |
| A：机制 | profile 信息能否在新 family/rig 上排序孔径误差与 harm？ | `[E0]` 无冻结结果 | 只能写“待检验” |
| B：pilot | 在有限 coverage 下，mean、p10、conditional harm、outer audit 是否同时有信号？ | `[E0]` 无冻结结果 | 通过也只能称 pilot signal |
| C：合成确认锁 | 新锁上是否复现，并给出 accepted harm 的单侧上界？ | `[E0]` 无可引用确认锁 | 未首开前不能写确认性安全 |
| D：真实 f-stop sweep | 至少 3 个固定 f-stop、reference/flow-off、outer/audit 是否支持真实迁移？ | `[E0]` 数据与权限待确认 | 无该门槛只能称 synthetic surrogate |

**截至 2026-07-16 的项目结论 `[E0-v5b]`：没有任何一个 v5b Gate 已通过。**

停止条件也属于结果：若 Gate A 没有稳定信号，应停止 blind/profile calibration，转向
已知 metadata cone-ray 或稳健 pinhole baseline；若 Gate B/C 尾部风险不过，应保留失败，
而不是继续从同一锁集挑阈值。`[E1：预设决策协议]`

---

## 15. 学完后应能向师兄问清的八个问题

1. 真实 forward 是 thin ray、cone ray、bilinear ray tracing，还是经验 distortion correction？
2. 是否保存 f-number、焦距、物距、focus setting、相机型号和采集时间？
3. reference/flow-off 原图能否估计 background-dot PSF/blur？
4. 哪些场/帧可合理共享同一个 `theta`，硬件漂移时间尺度多长？
5. forward 能否分别对三维场与光学参数提供 JVP/VJP，或至少支持可重复 finite difference？
6. 能否封存不参与拟合的 outer camera 和从不调参的 final audit camera？
7. 无三维真值时，组内最认可的独立 trace 是 front、积分量、PIV compensation 还是重复实验？
8. 能否做同一装置至少 3 个固定 f-stop 的最小控制实验？

这些问题的答案目前不能由本文代填。`[E0：待组内确认]`

---

## 16. 证据入口与阅读顺序

1. [v5a 有限孔径盲标定首开复盘](../v5a_blind_aperture_first_open_review.md)：先读锁集失败、
   accepted conditional risk 与锁后诊断。`[E3/E2]`
2. [v5a 预锁协议](../demo_t16_operator/v5a_blind_aperture_prelock_protocol.md)：核对冻结规则、
   数据泄漏修复与失败门槛。`[E3]`
3. [v5a 锁集 report.json](../demo_t16_operator/results/v5a_blind_aperture_calibration/report.json)：
   从机器可读结果复核数字。`[E3]`
4. [v5b 共享 Profile-Schur 设计协议](../demo_t16_operator/v5b_rig_shared_profile_calibration_protocol.md)：
   只作为待实现协议阅读。`[E1/E0]`
5. [研究学习日志第五轮](./算子学习与三维重构_研究学习日志.md)：用本科生语言回顾为什么
   v5a 失败和为什么下一步先问 identifiability。`[E3/E2/E1]`
6. [有限孔径 BOS forward/inverse modeling](https://arxiv.org/abs/2402.15954)：有限孔径与
   depth-of-field 的外部物理来源。`[E4：外部来源；不等于本项目复现]`
7. [Variable Projection](https://doi.org/10.1137/0710036)：profile/variable projection 的
   经典数值来源。`[E4：方法来源]`
8. [Risk-Coverage / Selective Prediction](https://jmlr.org/papers/v11/el-yaniv10a.html) 与
   [Learn then Test](https://arxiv.org/abs/2110.01052)：选择性风险与有限样本校准背景。
   `[E4：方法来源；不构成 v5b 安全证据]`

---

## 17. 最终自检句

完成任何课堂汇报、周报或论文草稿前，逐字检查这句话是否仍然成立：

> **v5a 在规定的合成锁集上失败 `[E3]`；v5b 是为检验共享校准与 nuisance-orthogonal
> identifiability 而提出的设计 `[E1]`；截至 2026-07-16，v5b 没有可引用的冻结、独立验证
> 或首开成功证据 `[E0]`。**

只要最后一项仍是 `[E0]`，标题、摘要、图注和口头汇报里就不得出现“v5b 已成功”“已证明
安全”“优于现有方法”或“解决真实有限孔径 BOST”等表述。
