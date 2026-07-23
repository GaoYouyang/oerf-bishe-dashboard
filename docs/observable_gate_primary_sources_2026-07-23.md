# Observable gate 与逐视角影响候选：一级来源边界

## 0. 怎么用这张表

这不是“看过论文名就拼模型”的清单。每篇文献只承担一个明确角色：物理终点、反问题结构、拒答评估、风险校准或算子表示。其他主张不从题名推出。

## 1. 何远哲 / OERF 直接物理主线

### 1.1 He et al. (2025), Neural refractive index field

- 一级来源：[arXiv 开放全文](https://arxiv.org/html/2409.14722v2)，[AIP/DOI](https://doi.org/10.1063/5.0250899)
- 文章：*Physics of Fluids* 37, 017143。
- 读取重点：neural implicit RI field；同时输出 refractive index 和 gradient；random ray sampling；automatic/numerical differentiation consistency；数值 phantom 的 field 真值评分；实验中 8 视角重建 + 1 视角 reprojection validation。
- 对当前研究的作用：说明留出视角是真实 NeRIF 验证链的一部分，也说明最终重建是三维折射率/密度场，不是一个投影分数。
- 不能推出：单个 held-out reprojection 对三维 field error 单调；它也没有给出 risk--coverage 或自动拒答证书。

### 1.2 Zheng et al. (2025), simultaneous PIV-BOST

- 一级来源：[Springer 正式页](https://link.springer.com/article/10.1007/s00348-025-04093-y)，[DOI](https://doi.org/10.1007/s00348-025-04093-y)
- 文章：*Experiments in Fluids* 66, 164。
- 读取重点：1-to-9 endoscope BOST；neural-network 3D RI reconstruction；中心平面 PIV；双系统同步；折射率梯度导致的 PIV 像素偏移；小非驻定 Bunsen flame 中约 `+/-2%` 的瞬时速度误差量级与逆向补偿。
- 对当前研究的作用：把“三维 RI 场是否值得接受”连到一个可测的下游终点：补偿后速度误差。这比继续增加平滑先验更贴近师兄真实需求。
- 不能推出：目前 synthetic energy gate 已对 PIV 有效；当前仓库还没有这篇的真实同步数据。

## 2. 跨模态物理约束邻居

### 2.1 Cai et al. (2021), Flow over an espresso cup

- 一级来源：[Cambridge/JFM 正式页](https://doi.org/10.1017/jfm.2021.135)
- 文章：*Journal of Fluid Mechanics* 915, A102。
- 读取重点：从 Tomo-BOS 三维温度 snapshot 中，用 Navier--Stokes 和热传递 PDE residual 推断连续的三维速度与压力；使用独立 PIV 实验验证中心平面速度。
- 对当前研究的作用：支持“不要只用同一 optical residual 审计 optical reconstruction”；可以引入独立速度/PDE 终点。
- 不能推出：PINN 一定比 NeRIF/FNO 好；也不能用 PDE residual 自动代替真实测量不确定度。

## 3. 拒答与风险校准

### 3.1 Geifman & El-Yaniv (2019), SelectiveNet

- 一级来源：[PMLR 正式页与开放 PDF](https://proceedings.mlr.press/v97/geifman19a.html)
- 读取重点：预测和 reject option 端到端联合训练；covered domain 上的 selective risk；risk--coverage trade-off。
- 对当前研究的作用：规定页面必须同时显示 precision/risk 和 coverage/retention。只有 `1/1` accepted 正例不是好 gate。
- 不能推出：SelectiveNet 在 BOST 上有现成安全性；原文的通用分类/回归实验不是三维折射率场验证。

### 3.2 Angelopoulos et al. (2024), Conformal Risk Control

- 一级来源：[ICLR Proceedings 正式页与 PDF](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html)
- 读取重点：对任意单调 loss 的期望值控制；与 split conformal 的关系；约 `O(1/n)` 紧性；distribution shift 等扩展。
- 对当前研究的作用：若未来要声称 false-accept risk 受控，必须先定义单调 loss、独立 calibration unit 和有效的 exchangeability/shift 假设。
- 不能推出：当前 12 phantom 的 leave-one-group-out ridge 已经 conformal；它没有独立 final audit，且零事件上界仍为 0.95。

### 3.3 Anzengruber & Ramlau (2010), Morozov discrepancy principle

- 一级来源：[IOP/DOI 正式页](https://doi.org/10.1088/0266-5611/26/2/025001)
- 读取重点：非线性反问题中 Tikhonov-type convex penalty 的事后参数选择；噪声水平趋零下的存在性、收敛与 Bregman-distance rate。
- 对当前研究的作用：强调 residual/discrepancy 门需要噪声水平和 forward 假设。
- 不能推出：只要 holdout residual 下降就可以应用 Morozov；当前 neural renderer 下的 model mismatch 和三维零空间并未被该定理自动消除。

## 4. 反问题与算子学习实现形状

### 4.1 Adler & Oktem (2018), Learned Primal-Dual Reconstruction

- 一级来源：[IEEE DOI](https://doi.org/10.1109/TMI.2018.2799231)，[author arXiv](https://arxiv.org/abs/1707.06474)
- 读取重点：把可能非线性的 forward operator 放入 unrolled primal--dual network；从 raw measurement 端到端训练；不依赖 FBP 初值。
- 对当前研究的作用：说明可学习的部分可以嵌入显式 forward/adjoint 交互，不必直接用黑箱预测三维场。逐视角 influence 也应优先保留真 renderer/JVP/VJP。
- 不能推出：在 BOST 中直接套用 learned primal-dual 就是创新。

### 4.2 Lu et al. (2021), DeepONet

- 一级来源：[Nature MI DOI](https://doi.org/10.1038/s42256-021-00302-5)，[author arXiv](https://arxiv.org/abs/1910.03193)
- 读取重点：branch net 编码固定 sensors 上的输入函数，trunk net 编码输出坐标；operator approximation 的 sensor 数和输入函数依赖。
- 对当前研究的作用：是“从逐视角观测函数到修正场/风险函数”的一个基线表示。
- 不能推出：固定 camera sensors 训练的 DeepONet 自动跨相机数和 rig 泛化。

### 4.3 Li et al. (2021), Fourier Neural Operator

- 一级来源：[ICLR/OpenReview](https://openreview.net/forum?id=c8P9NQVtmnO)，[author arXiv](https://arxiv.org/abs/2010.08895)
- 读取重点：在 Fourier 空间参数化 integral kernel；从函数到函数的映射；Burgers/Darcy/Navier--Stokes 与 zero-shot super-resolution 设置。
- 对当前研究的作用：可作为规则网格上的高容量 operator baseline，但必须同输入、同参数/调用预算与逐视角 set model 比较。
- 不能推出：FNO 在视角缺失、非规则 ray 或新 rig 上自动泛化。

## 5. 当前可以做的文献结论

在上述已核对的一级来源中，没有看到一个 OERF/何远哲已发方法同时完成下列三件事：

1. 用逐视角影响特征预测三维修正是否有益；
2. 对接受/拒答给出 group-level risk--coverage 证据；
3. 在独立 rig/session 上以 field 或 PIV velocity endpoint 确认。

这只是一个精确限定的“已核对来源中未见”，不是全世界新颖性证明。论文题名、摘要和 novelty claim 必须等师兄的真实 callable、数据合同和新一轮一级来源检索到位后再冻结。

## 6. 初学者阅读顺序

1. 先读 NeRIF 的 BOST forward 和实验留视角，把“三维场”与“二维投影”分开。
2. 再读 PIV-BOST 的速度误差传播，明白为什么重建应服务下游测量。
3. 读 SelectiveNet，手算 coverage、selective risk、precision 和 false-negative。
4. 读 Conformal Risk Control 的假设，不要先套 conformal 名字。
5. 读 learned primal-dual，理解为什么真 forward/adjoint 要留在网络里。
6. 最后对比 DeepONet/FNO：它们分别假设什么输入结构，为什么不能把新 rig 泛化当成默认属性。

## 7. 访问与公开边界

- 本文只链接 publisher/proceedings/official archive/DOI。
- 公开 GitHub Pages 不上传订阅 PDF、学校 VPN 内容、凭据或私有数据。
- 需要订阅的全文只可保留在本地 `private_library/`，公开页仅放 DOI 和自写摘要。
