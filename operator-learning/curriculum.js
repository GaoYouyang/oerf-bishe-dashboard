window.OPERATOR_LEARNING_GUIDE = {
  version: "2026.07.23-c-warmstart-locked",
  updated: "2026-07-23",
  foundationChecks: [
    {
      id: "python-array",
      domain: "Python",
      label: "我能用 NumPy 创建、切片和可视化三维数组",
      evidence: "独立生成 32³ Gaussian phantom，并画出 xy/xz/yz 三个切片。",
      week: "W0"
    },
    {
      id: "torch-autograd",
      domain: "PyTorch",
      label: "我能解释 tensor、batch、channel 和 autograd",
      evidence: "训练一个 MLP 拟合二维函数，并求输出对坐标的梯度。",
      week: "W4"
    },
    {
      id: "linear-map",
      domain: "线性代数",
      label: "我能区分列空间、零空间、秩和伪逆",
      evidence: "对一个小型少视角投影矩阵做 SVD，画奇异值并验证 A d ≈ 0。",
      week: "W1"
    },
    {
      id: "fourier",
      domain: "Fourier",
      label: "我能解释 FFT、频率模式和截断对薄前缘的影响",
      evidence: "对平滑场和薄层场做 FFT，比较保留不同 modes 后的重建误差。",
      week: "W5"
    },
    {
      id: "fluid-chain",
      domain: "流体力学",
      label: "我能把 T、ρ、参考态、Δn、位移和重建场连成物理链",
      evidence: "不看资料画出 T → ρ → ρ_ref → Δn → 光线偏折 → 背景位移，并解释为什么绝对常数偏置不可观。",
      week: "W2"
    },
    {
      id: "forward-inverse",
      domain: "反问题",
      label: "我能区分 forward、adjoint/backprojection 和 inverse",
      evidence: "用自己的话解释为什么重投影误差小不等于三维场正确。",
      week: "W3"
    },
    {
      id: "radon-toy",
      domain: "层析",
      label: "我能跑通 Radon/FBP/SART 最小实验",
      evidence: "比较 8/16/32/64 投影角和两档噪声的 field/reprojection 误差。",
      week: "W3"
    },
    {
      id: "operator-definition",
      domain: "算子学习",
      label: "我能说清函数到函数的映射与普通回归的区别",
      evidence: "写出 BOST 中 y → Δn(x,y,z) 的输入、输出、reference/gauge、几何元数据和离散化。",
      week: "W5"
    },
    {
      id: "ood-split",
      domain: "机器学习",
      label: "我能独立设计 train/val/IID/OOD 切分",
      evidence: "给 phantom family、view、noise、geometry 做无泄漏 manifest，并解释每个 split 在验证什么。",
      week: "W7"
    },
    {
      id: "paper-reading",
      domain: "论文阅读",
      label: "我能从论文中提取任务、数据、baseline、指标和失败边界",
      evidence: "用统一模板完成 NeRIF、FNO、DeepONet 三张阅读卡。",
      week: "W6"
    }
  ],
  concepts: [
    {
      id: "operator",
      title: "算子不是更大的神经网络",
      plain: "普通回归常学向量到向量；神经算子要学一整族输入函数到输出函数的映射。",
      formula: "G: y(s) → Δn(x,y,z)",
      bost: "BOST 的多视角位移/投影是输入函数，相对已知背景的三维折射率扰动是输出函数。",
      trap: "固定 24³ 网格上训练一个 CNN，不会因为名字叫 FNO 就自动具有分辨率泛化。"
    },
    {
      id: "forward",
      title: "正问题是可验收的物理层",
      plain: "给定三维场和相机几何，预测会观测到什么位移。",
      formula: "y = A_g(x) + ε",
      bost: "A_g 包含折射率梯度、光线路径、投影与相机布局 g。",
      trap: "只看 field loss 会忽略物理不一致；只看 reprojection 也可能选中错误的零空间解。"
    },
    {
      id: "inverse",
      title: "反问题的核心是非唯一与不稳定",
      plain: "不同的三维场可能产生几乎一样的少视角观测，小噪声也可能放大成很大的场误差。",
      formula: "x̂ = argmin ||A_g x-y||² + λR(x)",
      bost: "传统正则化、NeRIF 隐式表示和神经算子都在不同方式中注入先验。",
      trap: "网络不会消除不可观测性，只会根据训练分布选一类解。"
    },
    {
      id: "gauge",
      title: "BOS 看得到梯度，看不到任意常数",
      plain: "给整个折射率场加同一个常数，偏折可能不变；所以必须先定义环境参考值或统一的边界/零均值条件。",
      formula: "x = Δn = n-n_ref",
      bost: "环境值、flow-off 或已知边界应在 validation 前冻结，并以同样方式提供给经典基线与学习器。",
      trap: "若只让网络从工况标签猜背景均值，再用绝对场 L2 比较，它会得到观测本身不支持的不公平优势。"
    },
    {
      id: "svd-null",
      title: "SVD 把可观测和不可观测方向分开",
      plain: "大奇异值方向容易被相机看到，小奇异值或零奇异值方向难以从 support cameras 区分。",
      formula: "A = UΣVᵀ,  d_N = (I-A⁺A)d",
      bost: "QC-SNCO 只允许学习修正进入 support nullspace，再由 query camera 标定幅度。",
      trap: "零空间一致只能保证不破坏 support，不能保证修正方向对。"
    },
    {
      id: "adjoint",
      title: "Adjoint lift 是起点，不是答案",
      plain: "把每个观测沿光线‘撒回’三维空间，得到一个可解释但模糊的初值。",
      formula: "x₀ = A_gᵀ y",
      bost: "Residual FNO 学的是 x-x₀，Absolute FNO 直接输出 x，两者在不同 OOD 域会翻转。",
      trap: "把 adjoint lift 通道输入网络不等于网络满足数据一致性。"
    },
    {
      id: "evaluator-sentinel",
      title: "先审参考答案，再比较算法",
      plain: "高精度数值解也不是天然真值；如果 H256 到 H512 还在变化，候选与它的差可能只是 evaluator 偏差。",
      formula: "e_H = ||F_H-F_2H|| / max(||F_2H||, ε)",
      bost: "N4.1 的 30 格与 D2 的 2 格 H8192 已在 D3 按原 cell order 组成 32×256×2 参考包；独立 validator 重算了映射、数组哈希、Merkle root 和成本账本。",
      trap: "D3 是 30 格 raw + 2 格 paired-Neumaier 的 mixed pack；paired 等价覆盖只有 4/32，不能写成 fresh 或统一 paired 32/32。"
    },
    {
      id: "cancellation-aware-reference",
      title: "小残差需要相消感知的尺子",
      plain: "当 curved 和 straight 两个完整量很接近时，它们的差很小；同样的绝对离散误差会被小分母放大。",
      formula: "r_H = F_H - S_H,  e_r = ||r_H-r_2H|| / ||r_2H||",
      bost: "D1 排除累加顺序，D2 把四格加密到 H8192，D3 再用 23/7/2 分配封装混合参考包；后续导数实验必须继承每格路由标签。",
      trap: "排除浮点相消并找到 selected 二阶尾部，仍不等于真实实验可辨认；必须继续冻结 observable、flow-off noise floor 和 fresh/independent 边界。"
    },
    {
      id: "field-jvp-vjp",
      title: "Field JVP/VJP 是三维重建的发动机",
      plain: "JVP 告诉你场沿某个方向变化时观测怎样变；VJP 把所有观测残差一次拉回三维参数空间。",
      formula: "Jv = dF(theta)[v],  grad L = J^T W^T W(F-y)",
      bost: "D4 tiny gate 通过，但 D4b 扩到完整 32-cell census 后只有 254/256 maps、58/64 topology contexts 通过；post-open 又排除了最终求和顺序并定位 21 个 support-set flips。",
      trap: "D4b 历史上必须 fail closed；但当前 smoothstep forward 没有 hard mask，不能把协议 support-set bit 直接说成真实程序分支。"
    },
    {
      id: "mixed-scale-adjoint",
      title: "低信号伴随不能只看一个相对分母",
      plain: "两个接近的大观测相减后，residual signal 会缩得很小；同一个绝对闭合误差会被相对指标放大很多倍。",
      formula: "e_norm = |<Jv,w>-<v,J^Tw>| / (||Jv||||w||+||v||||J^Tw||)",
      bost: "p14 的 component dot signal 比 raw residual 大 14,466.67 倍；精确 contraction 仍不过原门，说明最终 reduction 不是根因。",
      trap: "不能看完 p14 就改用 normwise gate。mixed-scale 规则必须在独立 development population 上冻结，再用 fresh fields 验证。"
    },
    {
      id: "support-transversality",
      title: "support 等值面可以平滑移动而不必产生拓扑事件",
      plain: "采样点跨过人为阈值，不一定代表 forward 不可微；若 crossing 是唯一且非切触的 simple root，它可以随场扰动连续移动。",
      formula: "phi(s,e)=|f_e(r_e(s))|-tau,  ds*/de = -(partial_e phi)/(partial_s phi)",
      bost: "D4b 的 21 flips 只在 h=0.01，cell/frustum 不变且 24/24 map gates 通过；下一步比较 exact-bit 与 transversality-aware certificate。",
      trap: "simple-root 证书不是 soft mask。根生成、消失、grazing、危险 cell/domain/frustum 事件或区间证明失败都必须拒答。"
    },
    {
      id: "physics-residual-operator",
      title: "物理预条件残差算子只学没算准的部分",
      plain: "先用 Picard-1 给出便宜且物理可解释的近似，再让小模型学习高精度 H 与 P1 之间的结构化差。",
      formula: "F_hat(theta,g) = P1(theta,g) + R_phi(theta,g,P1-state)",
      bost: "N3 提示整体误差已经很小但少数 ray 的 Q95 仍会变差，残差模型应针对尾部而不是重做全部 forward。",
      trap: "只有 H-P1 高于 evaluator 误差和实验噪声底才值得学；P2-P1 不是免费特征，fallback 也必须计入成本。"
    },
    {
      id: "call-frontier",
      title: "绝对冠军不等于同预算冠军",
      plain: "一个方法误差更低，可能只是多调用了 forward/adjoint；必须先画误差对算子调用数的前沿。",
      formula: "cost(T)=N_A(T)+N_A^T(T)",
      bost: "v3k-D 的 quadratic candidate 误差 0.143855，但需 193 calls；128-call 前沿仍是 geometry Landweber 的 0.147083。",
      trap: "不能拿更多 A/A^T 调用得到的 1% 精度改善，写成模型结构 superiority。"
    },
    {
      id: "semi-convergence",
      title: "残差下降不等于三维场一直变好",
      plain: "含噪逆问题常先恢复稳定的大奇异值方向，随后开始拟合噪声；迭代次数本身就是正则化参数。",
      formula: "t* = first t with r_t^T Sigma_e^(-1) r_t / m <= tau",
      bost: "v3k-F 的 discrepancy 对 noise OOD 相对 fixed Landweber 平均 +7.10%，但仍有 12.5% 字段伤害；相对同 PBB endpoint 的 joint OOD 仍略退化。",
      trap: "平均改善不能放行 learned stop；noise/covariance proxy、tau、fallback 和 fresh lock 都必须在 validation 冻结。"
    },
    {
      id: "fno",
      title: "FNO 在频域中学全局核",
      plain: "FFT 把空间场分解成不同尺度的波，FNO 学习低频模式间的映射，再回到空间域。",
      formula: "v' = σ(Wv + F⁻¹(R·Fv))",
      bost: "全局感受野适合体场，但 Fourier truncation 可能过度平滑 thin-front。",
      trap: "必须与等参数 CNN/U-Net 和局部算子对照，不能把容量差异当成算子优势。"
    },
    {
      id: "deeponet",
      title: "DeepONet 把观测函数和查询坐标分开",
      plain: "branch 编码整个观测，trunk 编码要查询的三维坐标，两者内积给出场值。",
      formula: "G(y)(ξ) = Σ b_k(y)t_k(ξ)",
      bost: "适合任意坐标查询，但标准 branch 对可变视角数和顺序并不天然友好。",
      trap: "不要在第一个月同时实现可变传感器、attention 和任意几何。"
    },
    {
      id: "multi-fidelity-screen",
      title: "短程筛选只是低保真证据",
      plain: "少量 epoch 便宜，但它把早期收敛速度、学习率适配和最终容量混在一起；短程冠军不保证长程最优。",
      formula: "argmin_h E_val(h, 24) ≠ argmin_h E_val(h, 240)",
      bost: "v3g 的 rank-64 DeepONet 赢得 24-epoch screen，却在 240 epochs 略逊 rank-48 reference。",
      trap: "不能只延长一个短程冠军后，就声称其余结构都更差；必须声明 survivor bias、冠军差距和搜索停止预算。"
    },
    {
      id: "query-camera",
      title: "Query camera 不参与起点重建",
      plain: "support cameras 建立物理锚点，query camera 只检查某个零空间修正方向是否值得走。",
      formula: "α_Q = clip(⟨A_Qd,r_Q⟩ / ||A_Qd||², 0, 1)",
      bost: "v2c 的 adaptive query 曾给出 +0.746% 相对 S-only 的微弱均值；v2d 加入同重建预算 direct 与锁定 Q_audit 后，当前 checkpoint 路径在 K=4/6/8 均未通过。",
      trap: "query 能让 S-only 变好，不代表 query calibration 有附加价值；必须与把同一个 Q 直接加入重建比较。"
    },
    {
      id: "uq",
      title: "不确定度必须能对失败排序",
      plain: "好的不确定度不只是数字大小，而是误差真的大时它也更大，拒答后剩余风险会下降。",
      formula: "risk(c) = E[error | confidence ≥ c]",
      bost: "M3B 的 combined UQ 目前 Spearman/AUC 不足，所以不能宣称已有可靠拒答。",
      trap: "一个看起来平滑的 uncertainty heatmap 不是校准证据。"
    }
  ],
  weeks: [
    {
      id: "W0", phase: "起点", week: "第 0 周", title: "建环境，只跑最小闭环", hours: "4-6h", depends: [],
      learn: ["Python 环境、NumPy shape/dtype/axis", "Git 只掌握 status/diff/add/commit", "三维场与三张正交切片"],
      build: ["生成 32³ Gaussian + thin-shell phantom", "保存 npz、PNG 和一份 manifest", "记录重现命令和运行时间"],
      pass: ["换一个 seed 能得到可解释的另一个 phantom", "能解释 (B,C,D,H,W) 每一维", "不手工修图就能重跑产出"],
      resources: ["scipy-lectures", "pytorch-basics"],
      paper: "这一周不追论文数量，只建可复现习惯。"
    },
    {
      id: "W1", phase: "数学地基", week: "第 1 周", title: "SVD、伪逆与零空间", hours: "8-10h", depends: ["W0"],
      learn: ["线性映射、列空间、nullspace", "SVD 和条件数", "least squares、Tikhonov、伪逆"],
      build: ["随机低秩矩阵反演 toy", "在噪声下扫描截断秩/正则强度", "构造 d_N 并数值验证 A d_N ≈ 0"],
      pass: ["能画出奇异值谱", "能解释为什么小奇异值放大噪声", "能用一页笔记解释 QC-SNCO 的硬约束"],
      resources: ["mit-linear-algebra", "kak-slaney"],
      paper: "为 Deep Null Space Learning 和 query calibration 补足必要数学语言。"
    },
    {
      id: "W2", phase: "物理地基", week: "第 2 周", title: "从流场到背景位移", hours: "8-10h", depends: ["W0"],
      learn: ["连续介质、质量/动量守恒的物理意义", "温度-密度-Gladstone-Dale 关系", "折射率梯度、光线偏折与图像位移"],
      build: ["画 T→ρ→n→偏折→位移图", "对一维折射率梯度做数值积分", "用 300 字解释 BOST 与 PIV 各自测什么"],
      pass: ["不混淆速度场、密度场和折射率场", "能标出每个物理量的单位", "能给非本方向同学讲清 BOS 观测链"],
      resources: ["mit-206-fluid-dynamics", "paper-library-core"],
      paper: "只读 NeRIF 引言和方法中的物理链，不追网络细节。"
    },
    {
      id: "W3", phase: "反问题", week: "第 3 周", title: "Radon、FBP/SART 与少视角失败", hours: "10-12h", depends: ["W1", "W2"],
      learn: ["forward/adjoint/inverse", "Radon transform 与 sinogram", "投影不完备、正则化与误差分叉"],
      build: ["跑 scikit-image Radon/FBP/SART 官方例子", "扫描 8/16/32/64 角与噪声", "同时画 field error 和 held-out reprojection"],
      pass: ["能找到至少一个 reprojection 更好但 field 更差的反例", "所有 split 在数据生成前确定", "一条命令复现图表"],
      resources: ["skimage-radon", "kak-slaney"],
      paper: "这个反例是后续论文讲非唯一性的第一张证据图。"
    },
    {
      id: "W4", phase: "计算地基", week: "第 4 周", title: "PyTorch、autograd 和三维训练工程", hours: "10-12h", depends: ["W0"],
      learn: ["Dataset/DataLoader", "forward/loss/backward/optimizer", "normalization、checkpoint、seed 和显存"],
      build: ["坐标 MLP 拟合三维 phantom", "用 autograd 求 ∇n", "写最小 train/val 循环并保存 history.csv"],
      pass: ["能从 shape mismatch 报错定位到输入轴", "训练/验证归一化不泄漏", "重启进程能加载 checkpoint 继续验证"],
      resources: ["pytorch-basics", "pytorch-autograd", "scipy-lectures"],
      paper: "开始建立统一 experiment manifest，后面每个模型共用。"
    },
    {
      id: "W5", phase: "算子入门", week: "第 5 周", title: "从 spectral layer 到 FNO", hours: "12-14h", depends: ["W1", "W4"],
      learn: ["函数空间映射的直觉", "FFT、spectral convolution、modes", "FNO 的 lifting、operator blocks、projection"],
      build: ["手写 1D spectral layer", "跑 NeuralOperator Darcy CPU tutorial", "对 thin-front toy 扫描 modes 并画频谱误差"],
      pass: ["能从张量形状说出 FFT 在哪些轴上", "比较 L2 与 H1/gradient 指标", "不宣称未实测的 resolution invariance"],
      resources: ["neuraloperator-intro", "neuraloperator-fno", "neuraloperator-darcy", "fno-paper"],
      paper: "阅读 FNO 时提取模型假设、分辨率实验和数据规模，不只看架构图。"
    },
    {
      id: "W6", phase: "算子入门", week: "第 6 周", title: "DeepONet 与传感器函数", hours: "10-12h", depends: ["W4", "W5"],
      learn: ["branch/trunk 分解", "aligned/unaligned query", "固定传感器与可变视角的冲突"],
      build: ["跑 DeepXDE antiderivative demo", "用固定 5 视角 toy 做 branch-to-coordinate 查询", "与同预算 MLP/FNO 比较"],
      pass: ["能解释 branch input 是函数离散样本而非单个坐标", "能识别相机顺序伪特征", "有一份等预算表"],
      resources: ["deeponet-paper", "deepxde-operator"],
      paper: "DeepONet 是第二 operator family，目的是检查结论是否依赖 FNO。"
    },
    {
      id: "W7", phase: "BOST 基线", week: "第 7 周", title: "数据 manifest、物理 lift 与传统基线", hours: "12-16h", depends: ["W2", "W3", "W4"],
      learn: ["phantom family 与实验分布", "camera geometry 元数据", "adjoint/SIRT/Tikhonov 的角色"],
      build: ["生成 family×view×noise×geometry manifest", "跑 physics lift、SIRT 和 3D U-Net", "锁定 IID/view/noise/joint/family/geometry split"],
      pass: ["split 之间没有 phantom/geometry 泄漏", "所有基线共用同一指标函数", "报告 runtime、参数、显存和误差"],
      resources: ["kak-slaney", "skimage-radon", "t16-readme"],
      paper: "没有这一周的物理/数值基线，后面的网络改进没有可归因性。"
    },
    {
      id: "W8", phase: "真实数据桥", week: "第 8 周", title: "PoolFire 与师兄 BOS 工具接口审计", hours: "12-16h", depends: ["W2", "W4", "W7"],
      learn: ["CFD 数据中 rho/T/组分的含义", "数组 axis order、spacing、units 与 trajectory split", "Gladstone-Dale 与 rho_ref→Δn", "BOS 的常数 gauge", "相机内外参、ray near/far 与偏折单位"],
      build: ["低内存读取 PoolFire metadata 与真实 trajectory header", "流式保存 full-resolution rho 并检查全部 101 帧 finite/正值/统计，不使用可疑 YAML", "核对坐标降序、domain 尺度、单位与 cell center/edge", "用冻结的抗混叠/体积平均算子另建低分辨率副本", "整理师兄 notebook 的输入、输出、常数、相机与保存格式", "用常数场和线性梯度场做低分辨率 BOS smoke", "冻结 Δrho/Δn reference、axis、coordinate 和 output manifest"],
      pass: ["不加载 9.31 GB 解压 payload 也能说明 member、shape、dtype、order 与 split", "trajectory 与 metadata 的预期 SHA、ZIP CRC、rho checksums 和 READY 全通过", "full-resolution rho 全部 finite 且严格为正", "坐标方向与物理尺度冲突已解决", "常数场偏折接近零且线性场方向正确", "reference/gauge 不读取 test truth 且对所有基线相同", "师兄确认 rho/n/n-1、两个比例常数和 XYdeflection 单位", "私有 notebook 不进入 GitHub"],
      resources: ["c-route-lock", "poolfire-rho-bridge", "poolfire-data", "nerif-paper", "he-data-contract", "pytorch-autograd"],
      paper: "这一周只建立可信 forward 数据链；模拟器能跑不等于生成的数据已经物理正确。"
    },
    {
      id: "W9", phase: "成本基线", week: "第 9 周", title: "Zero/BP/PCGLS 与同精度计时器", hours: "12-18h", depends: ["W1", "W3", "W8"],
      learn: ["CGLS/CGNE recurrence", "非零初值为什么多一次 warm projection", "部署可见停止与 oracle time-to-target 的区别", "wall time warm-up、同步和内存测量"],
      build: ["验证 A/A^T dot test", "复用 fixed-warm CGLS shell", "跑 Zero-CGLS、A^T y-CGLS、fixed-SPD PCGLS", "分别输出 deployable-stop 主账与 oracle-headroom 辅账"],
      pass: ["零初值 K 步调用账与公式一致", "非零初值额外 forward 明确记账", "部署停止规则与等价区间只由 validation 冻结", "test truth 只事后画 headroom，不能控制主账停止"],
      resources: ["c-route-lock", "fair-budget-audit", "v3k-d-controls", "v3k-f-stopping", "discrepancy-hanke"],
      paper: "没有这一周，任何“快了多少”都可能只是隐藏特征成本或更差终点。"
    },
    {
      id: "W10", phase: "C0 最小算法", week: "第 10 周", title: "Adjoint-Residual Warm Start", hours: "14-20h", depends: ["W5", "W9"],
      learn: ["A^T y 为什么是模糊但物理可解释的起点", "非线性曲光线时的 J^T residual", "residual learning 与 direct field prediction", "FNO/3D U-Net 容量匹配", "field 与 measurement 双损失"],
      build: ["线性阶段训练 A^T y + coordinates/geometry → x0；非线性阶段改用参考态 J^T residual", "比较 residual FNO、matched 3D U-Net 与 direct operator", "接同一个物理 refinement", "画 per-case error-call 与 error-time 曲线"],
      pass: ["不使用 truth/SVD/family label 作为部署输入", "模型只消耗一次共享 A^T y 或参考态 J^T residual", "deployable 主账与 oracle headroom 分开报告 median/p90", "若没有 headroom 则不扩大模型"],
      resources: ["l2ws-paper", "inverse-acoustic-warmstart", "nows-paper", "super-fidelity-paper", "fno-paper", "v3e-compute-accounting", "v3f-deeponet-fno"],
      paper: "C0 主要验证研究假设；它可以成为完整毕设，但单独未必足够构成高质量论文创新。"
    },
    {
      id: "W11", phase: "C1 机制创新", week: "第 11 周", title: "Observable Krylov / Rig-Amortized Warm Start", hours: "14-20h", depends: ["W9", "W10"],
      learn: ["Krylov 子空间与谱滤波", "Lanczos/randomized basis", "可观测方向与近零空间", "一次性 setup 和多帧摊销"],
      build: ["实现 q0=A^T y 与 q1=A^T A q0 的小字典", "网络只预测有界系数或空间门控", "比较单向量/双向量/no-geometry 消融", "给 rig basis 报 setup、break-even frames 与 mismatch fallback"],
      pass: ["basis 不接触 test truth", "每个额外 A/A^T 调用进入成本账", "measurement residual 和坏尾部不劣于 C0", "简单 BP/PCGLS 已解释收益时主动停止"],
      resources: ["c-route-lock", "nows-paper", "v3k-c-adjoint", "v3k-d-controls", "nio-paper"],
      paper: "这是当前最值得争取的机制贡献：BOST 可观测子空间约束，而不是换一个更大的神经网络。"
    },
    {
      id: "W12", phase: "独立迁移", week: "第 12 周", title: "PoolFire 组合留出与独立 BOST 样例", hours: "12-20h", depends: ["W8", "W10"],
      learn: ["combination holdout 与真正参数 OOD 的区别", "trajectory/geometry/noise shift", "synthetic optical bias", "zero-shot、few-shot 与 per-instance refinement", "保留相机评价"],
      build: ["封存功率-尺寸组合留出 trajectory，不冒充未见参数 OOD", "扫描 view/noise/geometry shift", "若主张参数 OOD，另建超出训练取值范围的工况", "读取一份公开或组内最小 BOST 样例", "在不重训时先画重投影与失败 case"],
      pass: ["相邻帧和同工况不跨 split", "统计单位是 trajectory，时间帧只做块 bootstrap", "不用 test truth 调阈值或停止步", "独立 forward/domain 的结果单独标注", "失败个例与健康检查完整"],
      resources: ["poolfire-data", "open-bos-dataset", "nerif-paper", "public-data-transfer", "he-data-contract"],
      paper: "PoolFire 仍是 digital-twin/synthetic BOST；独立光学链是进入可投稿证据的必要升级。"
    },
    {
      id: "W13", phase: "可靠性", week: "第 13 周", title: "Fail-closed fallback 与成本风险", hours: "10-14h", depends: ["W10", "W12"],
      learn: ["部署可见 confidence proxy", "calibration/test 边界", "risk-coverage", "warm start 失败时回退经典初值"],
      build: ["用 initial measurement residual/geometry/noise proxy 排序失败", "画 risk-coverage 与 cost-coverage", "比较 always-warm、gated-warm、always-classical", "验证 fallback 是否降低最坏伤害"],
      pass: ["gate 不读取 field truth", "calibration/test 不混用", "p90/worst/harm 与节省成本同时报告", "proxy 失效时明确关闭部署主张"],
      resources: ["operator-uq", "conformal-deeponet", "selectivenet", "v3k-f-stopping"],
      paper: "可靠性贡献必须减少真实失败，不能只增加一张置信度热力图。"
    },
    {
      id: "W14", phase: "论文工程", week: "第 14 周", title: "C2 trajectory loss、强基线与统计封口", hours: "14-20h", depends: ["W10", "W11", "W12"],
      learn: ["短程 unroll 与 stop-gradient", "matched capacity/compute", "paired statistics 与 effect size", "机制消融"],
      build: ["仅在 C0/C1 有 headroom 时比较 field-only、field+measurement、1/2/4-step loss", "冻结 Zero/BP/PCGLS/direct FNO/DeepONet 主表", "生成 calls-time-error 前沿、三维切片、等值面和失败图", "保存 config/checksum"],
      pass: ["C2 未用更多测试信息或隐藏调用", "每个创新点有一对一消融", "筛选与最终训练成本分开", "统计单位是 trajectory/工况而非 voxel"],
      resources: ["nows-paper", "super-fidelity-paper", "nio-paper", "v3g-deeponet-capacity", "t16-readme"],
      paper: "只有 C0/C1 的稳定 headroom 存在，C2 才是合理升级；否则不靠复杂化挽救负结果。"
    },
    {
      id: "W15", phase: "投稿决策", week: "第 15-16 周", title: "可展示系统、论文包与师兄审核", hours: "16-24h", depends: ["W14"],
      learn: ["论文叙事与证据等级", "可复现包和许可边界", "投稿、降级和停止决策"],
      build: ["三维切片/等值面/重投影与 time-to-target 互动展示", "paper figures + tables + appendix + code README", "一页师兄/蔡老师审核材料和十分钟演示", "公开包剔除私有 notebook、路径和组内数据"],
      pass: ["新环境一条命令跑通主表", "页面不暗示 PoolFire 等于真实实验", "师兄确认 forward、同精度和成本口径", "所有私有资产保持本机边界"],
      resources: ["c-route-lock", "t16-evidence", "t16-readme", "paper-library-core"],
      paper: "只有全部硬门槛通过才冲论文；否则稳定收束为完整本科毕设。"
    }
  ],
  resources: [
    {id:"mit-linear-algebra",stage:"foundation",level:"零基础",type:"课程",title:"MIT 18.06SC Linear Algebra",url:"https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/",local:"",read:"优先看 subspaces、orthogonality、least squares、eigenvalues、SVD 和 pseudoinverse。",output:"一张 SVD-可观测性概念图 + 一个 NumPy toy。",verified:"MIT OCW 官方页"},
    {id:"kak-slaney",stage:"foundation",level:"入门",type:"开放电子书",title:"Kak & Slaney: Principles of Computerized Tomographic Imaging",url:"https://engineering.purdue.edu/~malcolm/pct/",local:"",read:"先读 projection、Fourier slice theorem、backprojection 和 algebraic reconstruction 章节。",output:"用自己的图重画 forward/inverse 链。",verified:"Purdue 作者电子版，仅个人使用"},
    {id:"skimage-radon",stage:"foundation",level:"零基础",type:"可运行教程",title:"scikit-image Radon / FBP / SART example",url:"https://scikit-image.org/docs/stable/auto_examples/transform/plot_radon_transform.html",local:"",read:"跑完 forward Radon、FBP 和 SART，再改投影角数与噪声。",output:"视角数-场误差-重投影误差三联图。",verified:"scikit-image 0.26 官方文档"},
    {id:"pytorch-autograd",stage:"foundation",level:"零基础",type:"官方教程",title:"PyTorch Automatic Differentiation",url:"https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html",local:"",read:"搞清 requires_grad、grad_fn、backward 和梯度累加。",output:"坐标 MLP + 空间梯度可视化。",verified:"PyTorch 官方教程"},
    {id:"neuraloperator-intro",stage:"operator",level:"入门",type:"理论导读",title:"Neural Operators: an Introduction",url:"https://neuraloperator.github.io/dev/theory_guide/neural_operators.html",local:"",read:"提取 function-space map、discretization 和 operator approximation 三个概念。",output:"用 BOST 变量重写一遍算子定义。",verified:"NeuralOperator 2.0 官方文档"},
    {id:"neuraloperator-fno",stage:"operator",level:"入门",type:"理论导读",title:"Fourier Neural Operators theory guide",url:"https://neuraloperator.github.io/dev/theory_guide/fno.html",local:"",read:"提取 spectral convolution、mode truncation、lifting/projection 和分辨率实验。",output:"1D spectral layer 和 modes ablation。",verified:"NeuralOperator 2.0 官方文档"},
    {id:"neuraloperator-darcy",stage:"operator",level:"入门",type:"可运行教程",title:"Training an FNO on Darcy Flow",url:"https://neuraloperator.github.io/dev/auto_examples/models/plot_FNO_darcy.html",local:"",read:"跑完 data loader、FNO、Lp/H1 loss、trainer 和 resolution test。",output:"一份能在 CPU 复现的 notebook 与训练曲线。",verified:"NeuralOperator 2.0 官方例子"},
    {id:"fno-paper",stage:"operator",level:"核心",type:"论文",title:"Fourier Neural Operator for Parametric PDEs",url:"https://iclr.cc/virtual/2021/poster/3281",local:"../paper_library/paper_detail.html?id=fno-li-2021",read:"式 (4)-(6)、spectral kernel、Navier-Stokes data split、resolution experiment 和 baseline 预算。",output:"一张角色阅读卡：可迁移什么，不可迁移什么。",verified:"ICLR 2021 官方页"},
    {id:"deeponet-paper",stage:"operator",level:"核心",type:"论文",title:"Learning nonlinear operators via DeepONet",url:"https://www.nature.com/articles/s42256-021-00302-5",local:"../paper_library/paper_detail.html?id=deeponet-lu-2021",read:"branch/trunk、operator universal approximation、sensor/query 格式和实验任务。",output:"画出 BOST branch/trunk 接口图。",verified:"Nature Machine Intelligence 官方页"},
    {id:"deepxde-operator",stage:"operator",level:"入门",type:"可运行教程",title:"DeepXDE operator-learning demos",url:"https://deepxde.readthedocs.io/en/latest/demos/operator.html",local:"",read:"先跑 antiderivative aligned/unaligned，再看 PI-DeepONet Poisson。",output:"一个 branch/trunk 张量形状日志。",verified:"DeepXDE 官方文档"},
    {id:"nio-paper",stage:"research",level:"进阶",type:"论文",title:"Neural Inverse Operators for Solving PDE Inverse Problems",url:"https://proceedings.mlr.press/v202/molinaro23a.html",local:"../paper_library/paper_detail.html?id=neural-inverse-operator-molinaro-2023",read:"问题定义、DeepONet/FNO 组合、训练分布和适用逆问题类型。",output:"对照 BOST 的 measurement-to-field 定义，列出三个结构差异。",verified:"ICML 2023 PMLR 官方页"},
    {id:"nerif-paper",stage:"bost",level:"核心",type:"何远哲主线",title:"Neural Refractive Index Field",url:"https://arxiv.org/abs/2409.14722",local:"../paper_library/paper_detail.html?id=nerif-he-2025",read:"BOST forward model、neural field 输出、loss、sampling、simulation 与 Bunsen-flame experiment。",output:"公式-代码-图表三列对照笔记。",verified:"arXiv 作者公开版 + Physics of Fluids 正式版"},
    {id:"bost-grauer-2018",stage:"bost",level:"核心",type:"三维 BOST 基线",title:"Instantaneous 3D flame imaging by BOST",url:"https://doi.org/10.1016/j.combustflame.2018.06.022",local:"",read:"提取 23-camera Bunsen flame 观测链、projection matrix、Tikhonov/TV 先验和 LES phantom 验证逻辑。",output:"画出传统 BOST forward/inverse 链，并列出 NeRIF 替代了哪一层。",verified:"Combustion and Flame 2018 出版商正式页，DOI 10.1016/j.combustflame.2018.06.022"},
    {id:"nirt-zhao-cvpr2024",stage:"current",level:"进阶",type:"可微折射场",title:"Single View Refractive Index Tomography with Neural Fields",url:"https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_Single_View_Refractive_Index_Tomography_with_Neural_Fields_CVPR_2024_paper.html",local:"",read:"只读 curved-ray differentiable tracing、neural field 参数化、观测歧义和 light-source prior；不把 single-view 设定直接套给多视角 BOST。",output:"写一页“可迁移的导数结构 / 不可迁移的观测先验”。",verified:"CVPR 2024 / CVF 官方开放版"},
    {id:"adjoint-nonlinear-ray",stage:"current",level:"进阶",type:"导数核心",title:"Adjoint Nonlinear Ray Tracing",url:"https://imaging.cs.cmu.edu/adjoint_nonlinear_tracing/",local:"",read:"重点看 nonlinear curved-ray constraint、adjoint-state derivative 和内存随步数恒定的原因。",output:"对照当前 PyTorch reverse-mode，画出轨迹存储与 constant-memory adjoint 的成本表。",verified:"CMU Imaging 作者项目页 + ACM TOG 2022 论文/代码入口"},
    {id:"diff-refraction-lyu",stage:"current",level:"支撑",type:"可微光线",title:"Differentiable Refraction-Tracing for Mesh Reconstruction",url:"https://arxiv.org/abs/2009.09144",local:"",read:"学习如何把背景匹配、折射光路与形状参数联起来；注意它是固体界面折射，不是连续气体场。",output:"列出 mesh/interface derivative 与 volumetric field derivative 的三个根本差异。",verified:"ACM TOG / SIGGRAPH Asia 2020，arXiv 作者公开版"},
    {id:"pearlmutter-jvp",stage:"current",level:"支撑",type:"自动微分基础",title:"Fast Exact Multiplication by the Hessian",url:"https://www.bcl.hamilton.ie/~barak/papers/nc-hessian.pdf",local:"",read:"不追二阶优化细节，只理解 R-operator 如何在不显式存储 Jacobian/Hessian 时计算向量乘积。",output:"用当前 forward 的变量写出 Jv 与 Jᵀw 的 shape 表。",verified:"Neural Computation 1994 作者公开 PDF"},
    {id:"residual-error-correction-cao",stage:"research",level:"进阶",type:"残差算子灵感",title:"Residual-based error correction for neural operators",url:"https://arxiv.org/abs/2210.03008",local:"",read:"提取神经算子近似误差为何会污染反问题，以及用物理 residual 解一个线性变分校正的条件。",output:"把论文的 PDE residual correction 与本项目 H-P1 optical residual 严格区分，写出需要重新证明的假设。",verified:"JCP 2023 DOI 10.1016/j.jcp.2023.112104 + arXiv 作者公开版"},
    {id:"deep-null-space",stage:"research",level:"进阶",type:"论文",title:"Deep Null Space Learning for Inverse Problems",url:"https://arxiv.org/abs/1806.06137",local:"../paper_library/paper_detail.html?id=deep-null-space-learning-schwab-2019",read:"Id-A⁺A 投影、data consistency、收敛假设和适用的线性设定。",output:"在 T16 小矩阵上数值验证硬零空间泄漏。",verified:"Inverse Problems 2019 DOI + arXiv 作者版"},
    {id:"neural-correction",stage:"research",level:"进阶",type:"论文",title:"Neural Correction Operator",url:"https://arxiv.org/abs/2507.18875",local:"../paper_library/paper_detail.html?id=neural-correction-operator-bhat-2026",read:"有限步传统重建 + learned correction 的分解、EIT 基线和失败边界。",output:"列出与 BOST adjoint/SIRT + correction 的相同和不同。",verified:"JCP 2026 DOI + arXiv；EIT 证据不直接迁移到 BOST"},
    {id:"gino-paper",stage:"geometry",level:"进阶",type:"论文",title:"Geometry-Informed Neural Operator",url:"https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html",local:"../paper_library/paper_detail.html?id=gino-li-2023",read:"irregular geometry 到 latent grid 的 GNO-FNO-GNO 组合和 3D 扩展成本。",output:"把 camera/ray geometry 写成 point cloud 接口草图。",verified:"NeurIPS 2023 官方页"},
    {id:"vidon-paper",stage:"geometry",level:"进阶",type:"论文",title:"Variable-Input Deep Operator Networks",url:"https://arxiv.org/abs/2205.11404",local:"../paper_library/index.html?q=VIDON",read:"可变数量传感器的 permutation-invariant 编码与误差分解。",output:"一个 3/5/7-view 可变输入 toy，与 padding+mask 对照。",verified:"arXiv 作者公开版"},
    {id:"operator-uq",stage:"reliability",level:"进阶",type:"论文",title:"Calibrated Uncertainty Quantification for Operator Learning",url:"https://openreview.net/forum?id=cGpegxy12T",local:"../paper_library/index.html?q=Calibrated%20UQ",read:"functional calibration、independent calibration set、coverage 对象和 finite-sample guarantee。",output:"为 BOST 定义 pointwise/fieldwise 两种 coverage。",verified:"TMLR OpenReview 官方页"},
    {id:"conformal-deeponet",stage:"reliability",level:"进阶",type:"论文",title:"Conformalized-DeepONet",url:"https://arxiv.org/abs/2402.15406",local:"../paper_library/paper_detail.html?id=conformalized-deeponet-moya-2025",read:"split conformal interval、exchangeability 假设与 DeepONet 输出的校准方式。",output:"一张 calibration/test 严格分层表。",verified:"Physica D DOI + arXiv 作者版"},
    {id:"selectivenet",stage:"reliability",level:"支撑",type:"论文",title:"SelectiveNet",url:"https://proceedings.mlr.press/v97/geifman19a.html",local:"../paper_library/paper_detail.html?id=selectivenet-geifman-2019",read:"coverage constraint、selective risk 与 reject option 评估。",output:"T16/M3B risk-coverage 曲线和拒答后系统误差。",verified:"ICML 2019 PMLR 官方页"},
    {id:"open-bos-dataset",stage:"bost",level:"核心",type:"开放数据",title:"Open-source BOS tomography dataset of high-speed flow",url:"https://link.springer.com/article/10.1007/s00348-026-04189-z",local:"../paper_library/paper_detail.html?id=open-bos-dataset-molnar-2026",read:"数据格式、相机布局、标定、GT/参考、许可和 benchmark 切分。",output:"只做数据健康报告和 baseline 重跑，不先调新模型。",verified:"Experiments in Fluids 2026 官方页"},
    {id:"t16-readme",stage:"local",level:"必做",type:"本项目执行包",title:"T16 reproducible operator workspace",url:"../document_reader.html?doc=demo_t16_operator%2FREADME.md",local:"",read:"按 README 复跑 smoke、validator、checksums 和逐样本统计。",output:"一份独立复跑记录，包含机器、版本、时间和差异。",verified:"本仓库可复跑工作区"},
    {id:"t16-evidence",stage:"local",level:"必做",type:"本项目证据",title:"T16 geometry/nullspace/query evidence",url:"../operator_nullspace_evidence_dashboard.html",local:"",read:"按 shared→independent→oracle→learned-null→query 证据梯形阅读。",output:"一页‘每个负结果如何改变下一步’的口述稿。",verified:"本仓库公开证据页"},
    {id:"qc-red-team",stage:"local",level:"必做",type:"本项目审计",title:"QC-SNCO red-team audit",url:"../document_reader.html?doc=qc_snco_red_team_audit.md",local:"",read:"逐条核对统计单元、相机预算、query 选择、Q_fit/Q_audit、inverse crime 与强基线。",output:"把四个当前不能声称的结论和下一轮判决阈值讲给师兄听。",verified:"本仓库方法学红队审计"},
    {id:"fair-budget-audit",stage:"local",level:"必做",type:"可复现实验",title:"Equal-reconstruction-budget v2d verdict",url:"../operator_fair_budget_dashboard.html",local:"../document_reader.html?doc=fair_camera_budget_review_brief.md",read:"先读 S/Q_fit/Q_audit 协议，再比较 direct、learned correction 与 numerical update 的均值、CI、p10 和 harm rate。",output:"用三句话解释为什么当前 QC-SNCO 不再支持扩模型，以及 K=6 max-gap 留下了什么几何线索。",verified:"19,008 method rows；88 fields；3 seeds collapsed；20,000 stratified bootstraps；checksum validator"},
    {id:"ridge-fno-v3a",stage:"local",level:"必做",type:"可复现实验",title:"Ridge-FNO → NeRIF v3a execution home",url:"../ridge_fno_nerif_roadmap.html",local:"../document_reader.html?doc=ridge_fno_nerif_review_brief.md",read:"比较 FBP-lift 与 ridge 起点、U-Net 与 FNO、三档相机预算、四类测试域和 Q_audit。",output:"讲清 ridge-FNO 为什么是自有算法必须击败的强基线，而不是最终贡献。",verified:"5,184 sample rows；1,728 field clusters；96 fields；3 seeds；20,000 bootstraps；28 tests"},
    {id:"own-algorithm-v3b",stage:"local",level:"必做",type:"自有算法实验",title:"v3b-v3k-D 自有算子竞技场与强数值门槛",url:"../own_algorithm_lab.html",local:"../document_reader.html?doc=v3k_d_strong_numerical_controls_brief.md",read:"按 v3h 可辨识性→v3i 数据合同→v3j 全局失败→v3k-A 等曝光→v3k-B 局部负结果→v3k-C 伴随闸门→v3k-D 强数值对照阅读。",output:"不看稿解释 adjoint residual、geometry/global/quadratic/lookup、双起点、A/A^T 前沿和 projected BB 下一门槛。",verified:"v3k-D: 19 tests；474 validation cells；672 pairs；8,064 sample rows；135 pairwise cells；12 checksums；base drift 0；independent validator"},
    {id:"v3h-geometry-gate",stage:"local",level:"必做",type:"几何可辨识性",title:"v3h 28-layout geometry identifiability gate",url:"../own_algorithm_lab.html#v3h-geometry-identifiability",local:"../document_reader.html?doc=v3h_gc_sro_geometry_gate_brief.md",read:"核对几何熵、maximum angular gap、operator condition number、field spread 和 correct/shuffled 可辨识性。",output:"用一个 4-camera toy 证明固定 mask 下 geometry conditioning 无法被对照识别。",verified:"28 masks；1,120 field-geometry rows；64.95% mean-error range；independent validator"},
    {id:"v3i-variable-dataset",stage:"local",level:"必做",type:"可训练数据合同",title:"v3i balanced variable-geometry dataset",url:"../own_algorithm_lab.html#v3i-variable-geometry-dataset",local:"../document_reader.html?doc=v3i_variable_geometry_dataset_brief.md",read:"核对 one-field/one-geometry、SHA-256 均衡分配、共享噪声、Q_audit 零通道和 geometry partitions。",output:"从 assignment CSV 独立复算每个 split 的几何熵与计数，再画出五组功能 pilot 的信息差异。",verified:"328 one-to-one assignments；16/4/4/4 geometries；42-channel contract；20.3 MiB private NPZ；checksum validator"},
    {id:"v3j-functional-negative",stage:"local",level:"必做",type:"机制负结果",title:"v3j GC-SRO descriptor-mechanism falsification",url:"../own_algorithm_lab.html#v3j-gc-sro-functional-result",local:"../document_reader.html?doc=v3j_gc_sro_functional_negative_result_brief.md",read:"先分开 static adapter 对 locked FNO 的通用收益与 correct geometry 对 static/shuffled 的独立收益，再追踪 embedding→modulation→correction→field error 四层 swap 敏感性。",output:"画一张‘信息在哪层衰减’图，并手写 v3k-A 反事实数据与 v3k-B 空间 ray-set 分支的 Go/No-Go 条件。",verified:"12 matched runs；288 history rows；4,920 sample rows；120 same-model swaps；base drift 0；independent validator"},
    {id:"v3k-counterfactual",stage:"local",level:"必做",type:"等曝光机制审计",title:"v3k-A same-field multi-layout counterfactual supervision",url:"../own_algorithm_lab.html#v3k-a-counterfactual-result",local:"../document_reader.html?doc=v3k_a_counterfactual_supervision_brief.md",read:"先核对 M1/M4 各 640 行、每布局 40 行和 source-field 聚类，再比较 correct-vs-shuffled、M4-M1 interaction 与 same-model swap。",output:"独立复算 validation 的 +0.0675% 和 interaction +0.0349%，再手画 voxel-level ray-set token/query/aggregation/correction。",verified:"24 matched runs；576 history rows；20,160 sample rows；960 same-model swaps；24 private checkpoints；base drift 0；checksum validator"},
    {id:"v3k-b-ray-set",stage:"local",level:"必做",type:"局部机制负结果",title:"v3k-B voxel-local ray-set and pairing falsification",url:"../own_algorithm_lab.html#v3k-b-voxel-ray-set",local:"../document_reader.html?doc=v3k_b_voxel_ray_set_negative_result_brief.md",read:"先检查 correct/shuffled 是否拥有完全相同的 ray 集合，再比较 geometry-only、pooled-static、pairing-shuffle 和同模型 swap。",output:"手推 grad 1/2||Ax-y||^2=A^T(Ax-y)，并说明 correction 阶段为什么需要 adjoint residual 而不是继续加 angle attention。",verified:"12 matched runs；288 history rows；10,080 sample rows；480 swaps；915 trainable parameters；base drift 0；checksum validator"},
    {id:"v3k-c-adjoint",stage:"local",level:"必做",type:"伴随数值闸门",title:"v3k-C projected Landweber and conditional-step launch gate",url:"../own_algorithm_lab.html#v3k-c-adjoint-gate",local:"../document_reader.html?doc=v3k_c_adjoint_landweber_gate_brief.md",read:"先对照两次红队修复，再检查 adjoint/gradient、field-only selection commit、feasible-FNO control、五域 field/Q_audit 与 worst-layout tail。",output:"亲手从 112 格 screen 选回 beta=1.9/T=64，复算 validation CI 与最差 layout，并写出 global-step/line-search/lookup/ridge-start 四个必要对手。",verified:"10 unit tests；28 geometries；112 screen rows；672 pairs；2,688 sample rows；20 layout cells；11 checksums；checkpoint drift 0"},
    {id:"v3k-d-controls",stage:"local",level:"必做",type:"强数值对照",title:"v3k-D call-matched strong numerical controls",url:"../strong_numerical_controls_lab.html",local:"../document_reader.html?doc=v3k_d_strong_numerical_controls_brief.md",read:"先比较 validation absolute champion 与 0/128/193-call frontier，再读 quadratic-vs-geometry 的 noise OOD 反向、双起点和 lookup 退化。",output:"从 474 行 screen 找回所有 winner，重画调用前沿，并从 secant equation 推导 projected BB1/BB2。",verified:"19 tests；474 screen rows；672 pairs；8,064 sample rows；60 summaries；135 pairwise rows；10 quadratic audits；60 ledger rows；12 checksums"},
    {id:"v3k-e-pbb",stage:"local",level:"必做",type:"同预算机制门",title:"v3k-E projected BB and noise-stopping gate",url:"../projected_bb_noise_gate_lab.html",local:"../document_reader.html?doc=v3k_e_projected_bb_review_brief.md",read:"先对照 strict 裁剪退化与 wide alternating 加速，再切换五个域查看 mean/p10/harm 翻转，最后读 semi-convergence 与下一轮 stop gate。",output:"复跑 13 个公式测试和 validator；手画 residual/field-error 随迭代分叉，并预注册 discrepancy tau、SPG 预算和 fresh-lock 阈值。",verified:"13 unit tests；180 validation cells；672 pairs；6,720 sample rows；45 pairwise cells；20 step audits；50 ledger rows；12 checksums；private assets 0"},
    {id:"bb1988",stage:"numerical",level:"核心",type:"论文",title:"Barzilai-Borwein two-point spectral steps",url:"https://doi.org/10.1093/imanum/8.1.141",local:"",read:"只提取 BB1/BB2 两点步长、无约束设定和数值动机。",output:"从 secant relation 推导两式，并逐数值对齐 v3k-E 第二步单元测试。",verified:"IMA Journal of Numerical Analysis 1988 官方 DOI"},
    {id:"spg2000",stage:"numerical",level:"进阶",type:"论文",title:"Nonmonotone Spectral Projected Gradient",url:"https://doi.org/10.1137/S1052623497330963",local:"",read:"提取 projected spectral direction、GLL nonmonotone line search、trial projection/call 与收敛假设。",output:"实现一版全调用记账 SPG，并列出其理论为何不能直接覆盖无 line-search PBB。",verified:"SIAM Journal on Optimization 2000 官方 DOI"},
    {id:"discrepancy-hanke",stage:"inverse",level:"核心",type:"书章",title:"Hanke: The Discrepancy Principle",url:"https://doi.org/10.1137/1.9781611974942.ch13",local:"",read:"理解 noisy Landweber 为什么必须早停，以及 first residual crossing 需要什么 noise 假设。",output:"写出 white、heteroscedastic、correlated noise 三种 normalized discrepancy，并在 validation 冻结 tau。",verified:"SIAM 2017 官方书章"},
    {id:"v3d-fno-plateau",stage:"local",level:"必做",type:"baseline 闸门",title:"K=6 FNO 24→96 epoch validation-plateau audit",url:"../own_algorithm_lab.html#v3d-fno-saturation",local:"../document_reader.html?doc=v3d_fno_validation_plateau_brief.md",read:"区分 validation-only 停止判决和后置 dev2 diagnostics；核对每 12-epoch block 的 mean/max-seed 改善。",output:"用图解释为什么 +16.54% dev2 改善仍不能拿 test 选 epoch，以及为什么新 adapter 必须等待 plateau。",verified:"288 history rows；21 seed-checkpoints；2,688 per-seed field rows；128 independent fields；checksum validator"},
    {id:"v3d-optimizer-audit",stage:"local",level:"必做",type:"固定 epoch 闸门",title:"K=6 FNO 三优化协议 24→240 epoch audit",url:"../own_algorithm_lab.html#v3d-optimizer-audit",local:"../document_reader.html?doc=v3d_fno_optimizer_protocol_brief.md",read:"比较 continuation blocks 间 restart/carry Adam moments 与 block/long cosine；base optimizer moments 未恢复。",output:"复画 error–epoch curve，解释 long-cosine 为何已 plateau 却不是最强 baseline，并衔接 v3e time-to-target。",verified:"2,016 history rows；171 checkpoints；1,536 per-seed dev2 rows；batch-order/provenance hashes；checksum validator"},
    {id:"v3e-compute-accounting",stage:"local",level:"必做",type:"跨架构成本闸门",title:"五架构参数 / FLOPs-v1 / MPS 时间与内存账本",url:"../own_algorithm_lab.html#v3e-compute-accounting",local:"../document_reader.html?doc=v3e_compute_accounting_brief.md",read:"先读 FLOPs-v1 的覆盖与排除项，再核对 15 个 fresh-worker trials、聚合成本和 FNO time-to-target。",output:"手算 adapter/FNO 的参数、训练时间和内存比，并口述为什么 4,988 个 trainable parameters 仍对应约 11.0× FNO 训练步。",verified:"5 models × 3 fresh workers；40 inference + 20 full training steps/worker；provenance hashes；独立 validator"},
    {id:"v3f-deeponet-fno",stage:"local",level:"必做",type:"matched 前沿",title:"DeepONet / FNO 24→240 error–time frontier",url:"../own_algorithm_lab.html#v3f-deeponet-fno-frontier",local:"../document_reader.html?doc=v3f_deeponet_fno_frontier_brief.md",read:"核对四学习率 screen、三优化协议、样本等权 validation、selection commit、60/120/180/240 前沿与复用 dev2 边界。",output:"复画 error–epoch/error–time 两张图，并给师兄讲清 GC-SRO 为什么保留 FNO 主干、只让 branch 编码采集几何。",verified:"2 architectures × 4 checkpoints；3 seeds；128 reused dev fields；selection commit；独立 validator"},
    {id:"v3g-deeponet-capacity",stage:"local",level:"必做",type:"baseline 容量审计",title:"DeepONet rank / 3D pooling / LR bounded audit",url:"../own_algorithm_lab.html#v3g-deeponet-capacity-audit",local:"../document_reader.html?doc=v3g_deeponet_capacity_audit_brief.md",read:"核对 1.5× 参数上限、两个预先排除项、8×3×3 screen、短程冠军差距、240-epoch 排名翻转和 selection commit。",output:"手算三个变体参数，复画 screen/长程曲线，并用三分钟解释为什么这轮负结果要求停止纯 rank 扩表。",verified:"72 fixed-epoch screen runs；3 optimizer protocols；3 seeds；128 reused dev fields；field-cluster CI；独立 validator"},
    {id:"v3d-geometry-manifest",stage:"local",level:"必做",type:"组内接口",title:"BOST geometry/data Go-No-Go manifest",url:"../document_reader.html?doc=v3d_geometry_data_manifest.md",local:"",read:"逐项确认 displacement、mask、intrinsics/extrinsics、ray coverage、calibration uncertainty、audit view 和坐标约定。",output:"请师兄勾选 geometry 是否跨 case 变化；固定且无缺失/漂移需求时停止条件分支。",verified:"公开模板；不含组内数据或私密材料"},
    {id:"cg-pdno-research",stage:"research",level:"必做",type:"自有算法主页",title:"Base-Correction CG-PDNO research lab",url:"../general_operator_research_lab.html",local:"../document_reader.html?doc=cg_pdno_guarded_smoke_review_brief.md",read:"按 E0-E4 读正负证据，再看 Base-Correction 结构、调用账本和 Go/No-Go。",output:"能不看稿解释为什么 noise-only trust 在前驱模型有效、在 CG-PDNO 上却应停止。",verified:"8 tests；736 + 120 + 144 rows；3 independent validators；superiority locked"},
    {id:"v3k-f-stopping",stage:"inverse",level:"必做",type:"确定性停止审计",title:"v3k-F deployable noise-stopping audit",url:"../document_reader.html?doc=v3k_f_noise_stopping_review_brief.md",local:"",read:"比较 fixed Landweber、fixed PBB、discrepancy、oracle，重点看 mean 与 tail 为何分叉。",output:"手写 Morozov first-crossing、调用记账和不放行 learned stop 的三个理由。",verified:"24 tests；6,048 method rows；4,704 stopping rows；independent validator"},
    {id:"public-data-transfer",stage:"data",level:"必做",type:"公开数据地图",title:"OpenBOST / CFD / reacting-flow transfer map",url:"../document_reader.html?doc=public_dataset_transfer_map.md",local:"",read:"分清 analytic truth、independent CFD 和 real optical chain 各自能证明什么。",output:"为每个数据源写 loader contract、allowed claim 和 forbidden claim。",verified:"OpenBOST official Data Commons；RealPDEBench official card；Michigan Deep Blue；SDRBench；FiReSMOKE"},
    {id:"he-data-contract",stage:"bost",level:"必做",type:"师兄沟通",title:"何远哲最小数据合同",url:"../document_reader.html?doc=he_yuanzhe_minimum_data_contract.md",local:"",read:"只要 1-3 帧的 observation、mask、ray、grid/unit、flow-off、baseline 和权限边界。",output:"把短消息发给师兄，并将回复转成 manifest，不口头猜接口。",verified:"本仓库公开合同模板；不含组内数据"},
    {id:"n3-grouped-audit",stage:"current",level:"必做",type:"正式结果审计",title:"N3 96 条件 grouped factorial NO-AUTH",url:"../general_operator_research_lab.html#n2-pvgr-n3",local:"../document_reader.html?doc=docs%2Fn2_pvgr_n3_grouped_factorial_result_audit_2026-07-18.md",read:"先读总门和 evaluator 失败，再读 Picard-1 的 8/8 强信号、Q95 反例、盲态恢复和禁止表述。",output:"不看页面复述为什么 P1 是下一强基线、却仍不能叫算法成功。",verified:"结果前预注册；96/96 checkpoints；41-file manifest；独立 validator；机器判决 NO-AUTH"},
    {id:"n4-evaluator-audit",stage:"current",level:"必做",type:"正式结果审计",title:"N4.1 H1024/H2048 evaluator convergence NO-AUTH",url:"../document_reader.html?doc=docs%2Fn2_pvgr_n4_1_evaluator_convergence_result_audit_2026-07-18.md",local:"../demo_t16_operator/results/n2_pvgr_n4_1_evaluator_convergence_v1/summary.md",read:"先看 32/32 output/topology 通过，再逐项读 23/32 H1024、9 格 H2048 和两个 narrow controls 的 residual-relative 失败。",output:"手算两失败格的 residual/full-output 比、absolute difference 和 contraction，并解释为什么不能事后改 0.125% 门。",verified:"双预注册与恢复披露；105 个 hash-locked checkpoints；30/32 final references；两层 validator valid；figure nonblank；机器判决 NO-AUTH"},
    {id:"n5-reference-plan",stage:"current",level:"必做",type:"正式机制与尾差审计",title:"N5-D1/D2 cancellation 与 H8192 tail",url:"../document_reader.html?doc=docs%2Fn2_pvgr_n5_d1_d2_result_audit_2026-07-18.md",local:"../demo_t16_operator/results/n2_pvgr_n5_d2_extreme_refinement_v1/summary.md",read:"先读 D1 为什么排除累加顺序，再逐格核对 D2 的 final adjacent、收缩比、观测阶、raw/paired 占比和禁止主张。",output:"手算四格 contraction 与 p，并画出 23×H1024 + 7×H2048 + 2×H8192 的下一 reference ledger。",verified:"D1/D2 均结果前预注册；42+6 tests；array-level independent validators valid；D2 4/4 selected tail resolved；模型与真实数据仍未授权"},
    {id:"n5-d3-reference-pack",stage:"current",level:"必做",type:"正式数值资产审计",title:"N5-D3 32-cell mixed adaptive reference pack",url:"../document_reader.html?doc=docs%2Fn2_pvgr_n5_d3_result_audit_2026-07-18.md",local:"../demo_t16_operator/results/n2_pvgr_n5_d3_adaptive_reference_v1/summary.md",read:"核对 23/7/2 映射、stacked/cell-order hash、N4 Merkle root、source/assembly query 区别和 4/32 paired coverage。",output:"不看答案解释为什么“validator valid”不等于“统一 reference 或模型成功”。",verified:"32/32 unique；23/7/2 allocation；zero-query assembly；independent validator valid；所有广义 claim 仍 false"},
    {id:"n5-d4-field-derivative",stage:"current",level:"必做",type:"正式导数实现审计",title:"N5-D4 tiny selected field JVP/VJP certificate",url:"../document_reader.html?doc=docs%2Fn2_pvgr_n5_d4_tiny_field_derivative_result_audit_2026-07-18.md",local:"../demo_t16_operator/results/n2_pvgr_n5_d4_tiny_field_derivative_v1/summary.md",read:"先看四格/八方向边界，再核对 worst dot、best/required-h FD、signal floor、ordered topology、strong detach control 和查询账本。",output:"不看结论解释为什么 32/32 map pass 只授权 D4b 32-cell expansion。",verified:"结果前协议 47278d1；32/32 maps、16/16 structures、8/8 topology；independent validator valid；reconstruction/model/real claims false"},
    {id:"n5-d4b-population-derivative",stage:"current",level:"必做",type:"正式总体导数审计",title:"N5-D4b 32-cell derivative census FAIL-CLOSED",url:"../document_reader.html?doc=docs%2Fn2_pvgr_n5_d4b_population_field_derivative_result_audit_2026-07-19.md",local:"../demo_t16_operator/results/n2_pvgr_n5_d4b_population_field_derivative_v1/summary.md",read:"先看 32 cells/16 pairs/5 field units 的非独立账本，再定位 p14 两个 residual dot failures 与六个 h=0.01 support-topology failures。",output:"画一张 map failure 与 topology failure 分开的因果图，并说明为什么 post-open contraction 或 stable-radius 诊断不能救回 D4b。",verified:"结果前协议 cba4f28；254/256 maps、128/128 structures、58/64 topology；independent validator valid；机器判决 FAIL-CLOSED；全部授权 false"},
    {id:"n5-d4b-postopen-forensics",stage:"current",level:"必做",type:"只读失败取证与下一算法设计",title:"N5-D4b low-signal / support-set post-open forensics",url:"../document_reader.html?doc=docs%2Fn2_pvgr_n5_d4b_postopen_forensics_2026-07-19.md",local:"../demo_t16_operator/results/n2_pvgr_n5_d4b_postopen_forensics_v1/summary.md",read:"先核对七种 contraction 为何都不过门，再看 14,466.67× residual signal suppression、21 个逐位 flips 和 topology-changed/stable FD 分布。",output:"手算一个 p14 normwise defect，任选三个 flips 还原索引，并写出 mixed-scale 与 simple-root 两个互相独立的 fresh protocol 草案。",verified:"saved arrays only；90 signature replays；21 flips；all frozen hashes reproduced；7 tests；historical D4b unchanged；no authorization"},
    {id:"n3-field-adjoint",stage:"current",level:"必做",type:"封存的后续设计",title:"Field JVP/VJP 到 6+2 view 三维重建接口",url:"../document_reader.html?doc=docs%2Fn2_pvgr_field_jvp_vjp_reconstruction_interface_design_2026-07-18.md",local:"",read:"重点读四个现有模块的 detach 审计、冻结 row layout、tensor-only forward、dot/FD 双门和 held-out view 边界。",output:"基于 D4b 失败写出必须先闭合的 topology-certified renderer 门，再标出 decoder 与 6+2 重建仍锁定。",verified:"D4 tiny valid；D4b fail closed；decoder chain 与三维重建未授权"},
    {id:"n3-recovery-disclosure",stage:"audit",level:"进阶",type:"研究诚信",title:"N3 KeyError 盲态分析恢复披露",url:"../document_reader.html?doc=docs%2Fn2_pvgr_n3_blind_analysis_recovery_2026-07-18.md",local:"",read:"理解为什么 96 格完成后仍不能直接改字段继续汇总，以及 opaque Merkle 封存如何限制事后自由度。",output:"写出允许修改的唯一 query schema 映射和所有禁止修改项。",verified:"恢复协议先提交；checkpoint payload 未解析；96 格未重算"},
    {id:"c-route-lock",stage:"warm-start",level:"必做",type:"项目合同",title:"PoolFire BOST Warm Start 路线锁定",url:"../document_reader.html?doc=docs%2Fpoolfire_bost_warmstart_project_lock_2026-07-23.md",local:"",read:"读清问题定义、C0/C1/C2、matched-accuracy、调用账本、停止条件和师兄待确认项。",output:"不看页面写出输入、输出、求解器、主指标和五个失败条件。",verified:"师兄已选择 C；算法与真实证据尚未验证"},
    {id:"poolfire-rho-bridge",stage:"data",level:"必做",type:"真实文件结构审计",title:"PoolFire rho 流式数据桥",url:"../document_reader.html?doc=docs%2Fpoolfire_rho_bridge_evidence_2026-07-23.md",local:"",read:"核对 trajectory/metadata SHA、ZIP64 member、NPY shape/dtype/order、full-resolution rho、READY 门和坐标冲突。",output:"解释为什么 6.43 GB 压缩包对应 9.31 GB payload、为什么原始桥不能直接 stride 抽点，以及为什么当前仍不能转换 Δn。",verified:"完整 source size/SHA、ZIP CRC、(101,80,80,200) rho、checksums 与 READY 已通过；20 项定向测试在两套 Python 环境通过；单位/光学链仍待闭合"},
    {id:"l2ws-paper",stage:"warm-start",level:"核心",type:"理论与训练目标",title:"Learning to Warm-Start Fixed-Point Optimization Algorithms",url:"https://jmlr.org/papers/v25/23-1174.html",local:"",read:"对比 fixed-point residual loss 与 solution-distance loss，理解网络如何为后续固定迭代负责。",output:"写出它的训练目标、下游迭代接口和泛化假设，并标出哪些假设不适用于 BOST 非线性逆问题。",verified:"JMLR 25(166), 2024；官方全文、PDF 与代码链接可用"},
    {id:"inverse-acoustic-warmstart",stage:"warm-start",level:"核心",type:"邻近逆问题",title:"A Neural Network Warm-Start Approach for Inverse Acoustic Scattering",url:"https://arxiv.org/abs/2212.08736",local:"",read:"重点看学习初值如何接传统反演、有限孔径/噪声测试，以及训练样本复杂度限制。",output:"列出声学逆散射与 BOST 的三点共同结构和三点不可直接迁移之处。",verified:"JCP 490, 112341 (2023)；arXiv 开放全文"},
    {id:"nows-paper",stage:"warm-start",level:"核心",type:"邻近工作",title:"NOWS: Neural Operator Warm Starts for Accelerating Iterative Solvers",url:"https://arxiv.org/abs/2511.02481",local:"",read:"提取 neural operator 初值、Krylov solver、iteration/runtime 口径与稳定性边界。",output:"列出 NOWS 与 BOST 逆问题的四个差异，避免把 warm start 本身误写为创新。",verified:"arXiv v4，2026-05-07 更新"},
    {id:"super-fidelity-paper",stage:"warm-start",level:"核心",type:"邻近工作",title:"Neural operator-based super-fidelity warm starts",url:"https://arxiv.org/abs/2312.11842",local:"",read:"重点看学习初值而非替换求解器、跨离散化与端到端成本。",output:"解释 steady forward PDE 结论为何不能直接迁移到 BOST 逆问题。",verified:"arXiv v2，2025-02-27 更新"},
    {id:"poolfire-data",stage:"data",level:"必做",type:"公开高保真 CFD",title:"REALM PoolFire",url:"https://huggingface.co/datasets/realm-bench/realm-bench-PoolFire/tree/main",local:"",read:"只先检查 metadata 与单条 trajectory；确认 rho、shape、time、units、NaN/Inf 和 split。",output:"一个不加载全数组的 header report + 一张 trajectory-level split 表。",verified:"REALM 官方 Hugging Face 数据仓库；首轨迹头为 (101,9,80,80,200) float64，但它不是现成 BOST 数据"}
  ],
  researchTracks: [
    {
      id:"warmstart-c0", rank:1, title:"C0 · Adjoint-Residual Warm Start", badge:"唯一主线的最小闭环", risk:"低-中", novelty:"创新性不是模型名字，而是 BOST 的 geometry-conditioned initialization、物理 refinement 与严格 matched-accuracy 成本证据", data:"PoolFire Δrho/Δn 使用 validation 前冻结的 reference/gauge，并按完整 trajectory/工况切分；BOS 工具输出语义确认后生成多视角观测；后续增加独立公开或组内样例", hardware:"先在 Mac 上做 16³-32³、低分辨率像素与少视角 smoke；确认 headroom 后再决定是否租 GPU", question:"一次 A^T y 或参考态 J^T residual 加几何输入生成的 x0，能否在部署可见停止规则下保持终点等价并减少物理调用和总时间？", contribution:"把 direct operator 变成可被经典求解器纠正的低成本起点，并分开报告 deployable 主账、oracle headroom、p90 与 harm rate。", next:"单条 PoolFire 轨迹审计→reference/gauge 与 BOS 常数/线性场验证→Zero/BP/PCGLS→小型 residual FNO/3D U-Net warm start。", stop:"网络推理与特征成本超过所省迭代、终点精度变差、p90/harm 变坏或 trajectory OOD 收益消失时停止。"
    },
    {
      id:"warmstart-c1", rank:2, title:"C1 · Observable Krylov Warm Start", badge:"优先机制创新", risk:"中", novelty:"把网络输出限制在由 A^T y 与少量 operator-only Krylov/basis 向量构成的可观测子空间，并显式报告 setup 与摊销成本", data:"与 C0 相同；basis 只可由 operator/训练集构造，不能看 test truth", hardware:"小 basis 可在本机；每增加一个 A/A^T 都必须证明能省回更多 refinement calls", question:"可观测子空间约束能否降低 learned x0 的重投影异常和坏尾部，同时保留加速？", contribution:"给 warm start 加入可解释、可审计的物理子空间约束；可扩展为 rig-amortized basis。", next:"先比较 q0=A^T y、q1=A^T A q0 的一/二向量版本，再决定 Lanczos 或近零空间 basis。", stop:"basis 构造成本无法摊销、简单 BP/PCGLS 已解释全部收益，或约束后不再加速时停止。"
    },
    {
      id:"warmstart-c2", rank:3, title:"C2 · Correctable Frontier Training", badge:"论文升级候选", risk:"高", novelty:"训练目标直接包含固定 1/2/4 步物理 refinement 后的误差，使网络学习最容易被求解器纠正的初值，而不只是单步最像真值", data:"只有 C0/C1 在未见 trajectory 上显示稳定 headroom 后才启动", hardware:"截断反传或 stop-gradient 小场开发；全 3D unroll 可能需要 GPU", question:"最小单步 field loss 的 x0，是否不如固定短程 refinement 后误差最小的 x0？", contribution:"连接 neural operator、iterative inverse solver 和成本-精度前沿，形成可一对一消融的机制主张。", next:"比较 field-only、field+measurement、1/2/4-step trajectory loss；所有物理调用进入账本。", stop:"训练显存/成本失控、收益仅来自更多训练预算、或 C0/C1 没有基础 headroom 时不启动。"
    },
    {
      id:"strong-baselines", rank:4, title:"强基线与公平成本账本", badge:"不是创新，但不可跳过", risk:"低", novelty:"无；作用是防止把弱基线、额外调用或较差终点包装成加速", data:"与主实验完全共用数据、forward、split、归一化和阈值", hardware:"本机即可先跑小场；完整 wall time 必须在同一设备、同一进程协议下测", question:"C0/C1/C2 是否同时胜过 Zero-CGLS、BP-CGLS、fixed-SPD PCGLS 和 direct FNO/DeepONet？", contribution:"冻结 calls、wall、memory、field/gradient、measurement、p50/p90/worst 与未达标率。", next:"先复用仓库 CGLS/PCGLS 与 fixed-warm shell，删掉旧 13F+13A^T 特征链。", stop:"若最强经典初始化达到相同或更好的前沿，论文主张应降级为负结果或工程复现。"
    },
    {
      id:"temporal-upgrade", rank:5, title:"Rig-Amortized / Temporal Warm Start", badge:"远期 4D 升级", risk:"高", novelty:"同一 rig 或相邻时刻共享 operator-only basis 与上一帧状态，贴近 TDBOST 的连续高速重建需求", data:"需要静态 C 路线通过，再加入合法时序 trajectory；不得随机拆相邻帧", hardware:"basis setup 与 break-even 帧数必须单列；全时序训练可能需要 GPU", question:"一次 rig setup 或前一帧信息能否在不损伤瞬态火焰前缘的情况下进一步降低每帧成本？", contribution:"将单帧 C 路线自然扩展到连续采集，并给出 setup amortization 与 rig mismatch fallback。", next:"只在 C0/C1 通过 PoolFire trajectory OOD 和组内静态样例后启动。", stop:"静态 warm start 不成立、无连续时序数据或时间先验抹平瞬态结构时停止。"
    }
  ],
  paperGates: [
    {id:"G0",title:"C 任务与物理语义锁定",evidence:"师兄确认目标为 warm start 加速，并确认 Δrho/Δn reference/gauge、偏折单位、forward/adjoint 和组内基线。",stop:"只确定 C 名称，但重建变量、不可观常数或同精度口径仍靠猜。"},
    {id:"G1",title:"PoolFire 无泄漏",evidence:"按完整 trajectory、火源功率/尺寸和 rig 划分；相邻帧、噪声副本和同场视角不跨 split。",stop:"随机帧切分，或用 test truth 构造 reference/归一化。"},
    {id:"G2",title:"强基线与调用公平",evidence:"Zero/BP/PCGLS、direct FNO/DeepONet 与 candidate 共用 forward、阈值和硬件；A^T y、warm projection、basis setup、搬运和网络推理均记账。",stop:"只省 solver iterations，但隐藏特征、setup 或推理成本。"},
    {id:"G3",title:"Matched-accuracy 双账主门",evidence:"主表使用 validation 冻结、部署可见的停止规则，检查终点误差等价并报 calls/wall/memory/p90；有真值时另报 oracle time-to-target headroom。",stop:"用 test truth 控制主表停止、把 oracle headroom 冒充在线耗时，或拿更差终点换速度。"},
    {id:"G4",title:"机制一对一消融",evidence:"BP、residual network、geometry、observable/Krylov constraint、trajectory loss 和 fallback 分别消融。",stop:"只有一个大模型，无法解释加速来自哪里。"},
    {id:"G5",title:"独立迁移",evidence:"至少未见 PoolFire trajectory/工况，并增加与训练 forward 独立的公开或组内 BOST 样例。",stop:"所有结果只在同一 generator 的 IID 帧上成立。"},
    {id:"G6",title:"统计与坏尾部",evidence:"至少三种子、逐 trajectory paired delta、置信区间、p90/worst/harm rate 和预定停止规则。",stop:"把 voxel/ray/frame 当独立样本或只报最好 seed。"},
    {id:"G7",title:"可复现与展示",evidence:"公开代码不含私有工具；环境、config、checksum、图表脚本、三维切片/等值面和成本表可一键重跑。",stop:"必须依赖师兄路径或手工 notebook 操作才能复现。"}
  ],
  seniorQuestions: [
    "notebook 的 refractive_index_field 应返回密度 rho、折射率 n，还是 n-1？",
    "level=2.2/3200 和 SprayFlame 分支的 2.48/10000 分别是什么常数、单位或缩放？",
    "保存的 XYdeflection 是 pixel、归一化相机坐标、偏折角还是物理长度？正负号怎样定义？",
    "这份 notebook 生成的是偏折坐标；最终 BOS 点图是否还有单独的 background warping/渲染脚本？",
    "组里当前要加速的重建基线具体是哪一个，入口脚本或函数是什么？",
    "组内是否已有与该 simulator 匹配的 forward/adjoint/VJP；若没有，认可先从线性化 A/A^T + CGLS 做 C0 吗？",
    "“同精度”主要按三维 field error、measurement/reprojection、保留相机还是下游 PIV 补偿判断？",
    "PoolFire 直接使用 REALM 公开版本，还是组里已有预处理版本、固定 split 或更适合的 CFD 数据？",
    "相机内外参、九视角角度、near/far、体场范围和 ray steps 哪些必须沿用，哪些可作为开发近似？",
    "哪些组内工具、代码、输出图和参数可公开，哪些只能留在本机私有目录？"
  ]
};
