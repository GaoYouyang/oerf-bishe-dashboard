window.OPERATOR_LEARNING_GUIDE = {
  version: "2026.07.19-n5-d4b-forensics",
  updated: "2026-07-19",
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
      label: "我能把 T、ρ、n、位移和重建场连成物理链",
      evidence: "不看资料画出 T → ρ → n → 光线偏折 → 背景位移图。",
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
      evidence: "写出 BOST 中 y → n(x,y,z) 的输入、输出、几何元数据和离散化。",
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
      formula: "G: y(s) → n(x,y,z)",
      bost: "BOST 的多视角位移/投影是输入函数，三维折射率场是输出函数。",
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
      id: "W8", phase: "BOST 算子", week: "第 8 周", title: "参考解、相消残差与 field-adjoint 前门", hours: "14-18h", depends: ["W3", "W4", "W7"],
      learn: ["RK4 步长加密与经验收缩阶", "两个近似量相减的小分母问题", "JVP/VJP dot test 与 normwise backward error", "support level set、simple root 与 transversality"],
      build: ["复算 D1/D2/D3 的参考解证据链", "复核 D4 与 D4b 的 map/topology 差异", "从取证 JSON 手算 p14 的 14,466.67× signal ratio", "任选 3 个 flips 还原 group/step/stage/ray/offset", "画 exact-bit gate 与 simple-root gate 的差异图", "向师兄确认真实 renderer 是否含 hard mask/occupancy/termination"],
      pass: ["能解释为什么精确 contraction 仍不能救回 D4b", "能区分 protocol support-set 与 forward hard branch", "能从 21 位表解释 stable h 与 changed h 的证据边界", "能写出一页不看 fresh 结果的 D4c 指标草案"],
      resources: ["n4-evaluator-audit", "n5-reference-plan", "n5-d3-reference-pack", "n5-d4-field-derivative", "n5-d4b-population-derivative", "n5-d4b-postopen-forensics", "n3-field-adjoint", "adjoint-nonlinear-ray", "he-data-contract"],
      paper: "先证明训练标签与梯度值得相信；如果 H-P1 低于实验噪声，停止 residual operator 比硬做网络更正确。"
    },
    {
      id: "W9", phase: "机制创新", week: "第 9 周", title: "mixed-scale 伴随与 transversality 证书", hours: "14-18h", depends: ["W1", "W8"],
      learn: ["dot-product conditioning 与 backward error", "level set 与隐函数定理", "simple/grazing roots", "interval Newton/Bernstein certificate 与 fail-closed"],
      build: ["在新 development cells 生成高/低 dot-signal directions", "并列实现 dot-relative、normwise 与 mixed absolute-relative 指标", "实现 exact-bit/simple-root/cell-frustum 三路消融", "冻结拒答原因、阈值来源和 fresh manifest"],
      pass: ["阈值不是照 p14 调出", "能构造 root birth/death/grazing 的反例", "simple-root certificate 不把危险事件放行", "fresh 开封前代码/config/test/hash 全冻结"],
      resources: ["n5-d4b-postopen-forensics", "n3-field-adjoint", "adjoint-nonlinear-ray", "n5-d4b-population-derivative"],
      paper: "这周只形成可被 fresh 数据否决的证书候选；不训练 DeepONet/FNO，也不把 post-open 21 flips 写成算法成功。"
    },
    {
      id: "W10", phase: "机制创新", week: "第 10 周", title: "query-camera 标定与信息价值", hours: "14-18h", depends: ["W9"],
      learn: ["reserved query 与数据泄漏边界", "一维 closed-form line search", "sensor selection/value of information"],
      build: ["先做 S-only / S+Q correction / S∪Q direct reconstruction 同总相机预算对照", "分开固定 query、随机 query 和逐样本 adaptive query", "另留 Q_audit，再扫 angle、noise、sync error 与 0/1/2/all-query 收益-成本"],
      pass: ["Q_fit 不进入初始重建或最终 Q_audit 验收", "同总相机数下击败直接重建强基线", "报告每个 split、p10/CVaR/harm rate 和真实相机成本"],
      resources: ["fair-budget-audit", "qc-red-team", "t16-evidence", "selectivenet", "operator-uq"],
      paper: "受控推理审计已触发当前 QC-SNCO checkpoint 的停止条件；训练 mask 匹配并锁定新测试集前不恢复。"
    },
    {
      id: "W11", phase: "几何泛化", week: "第 11 周", title: "可变相机布局与不规则观测", hours: "14-20h", depends: ["W8"],
      learn: ["fixed-grid FNO 的几何限制", "预白化与可变 ray-set", "data-consistent unrolling 与 correction", "leave-one-geometry-out 外层验证"],
      build: ["复跑 v3k-F 并解释 semi-convergence", "复核 CG-PDNO 的 zero-trust=fallback", "把 recurrent path 改为共享 physics base + 单次 correction", "与同总调用 SPG/PBB/CGLS 比较"],
      pass: ["source-field harm>1% 不高于 5%", "mean 与 p10 同时改善", "额外 forward 完整记账", "fresh lock 前不读新域", "不把同生成器 smoke 称为 BOST superiority"],
      resources: ["cg-pdno-research", "v3k-f-stopping", "v3h-geometry-gate", "v3i-variable-dataset", "v3j-functional-negative", "v3k-counterfactual", "v3k-b-ray-set", "v3k-c-adjoint", "v3k-d-controls", "v3k-e-pbb", "bb1988", "spg2000", "discrepancy-hanke", "gino-paper", "vidon-paper", "nio-paper"],
      paper: "CG-PDNO 三种子 development-fresh test 平均 +21.77%、最大种子 harm 0%；但同生成器 inverse crime 未解决。noise-only trust 在新模型上无增量，下一门为 Base-Correction 结构、独立 generator 和真实 held-out view。"
    },
    {
      id: "W12", phase: "真实迁移", week: "第 12 周", title: "真实 BOST 接口与 sim-to-real 诊断", hours: "12-20h", depends: ["W7", "W10"],
      learn: ["calibration/mask/units 数据契约", "synthetic bias 与 domain shift", "zero-shot、few-shot 与 per-instance refinement"],
      build: ["读取一份真实或开放 BOST 样例", "在不训练时先画观测/重投影残差", "对比 direct、few-shot 和 operator→NeRIF refinement"],
      pass: ["每个物理量单位可追溯", "不用真实 test 调超参数", "报告失败个例与数据健康检查"],
      resources: ["open-bos-dataset", "nerif-paper", "t16-evidence", "public-data-transfer", "he-data-contract"],
      paper: "这是从精致 synthetic demo 进入可投稿证据的必经关口。"
    },
    {
      id: "W13", phase: "可靠性", week: "第 13 周", title: "校准、拒答与风险-覆盖", hours: "10-14h", depends: ["W8", "W12"],
      learn: ["calibration split", "conformal coverage 与分布假设", "selective prediction/risk-coverage"],
      build: ["将 uncertainty 与真实误差做排序对照", "画 risk-coverage 曲线", "验证 fallback 是否真正降低系统误差"],
      pass: ["calibration/test 数据不混用", "同时报 Spearman/AUC/coverage 与失败样本", "UQ 失败时不宣称可部署拒答"],
      resources: ["operator-uq", "conformal-deeponet", "selectivenet"],
      paper: "UQ 是完整系统贡献，不是为了再添一张热力图。"
    },
    {
      id: "W14", phase: "论文工程", week: "第 14 周", title: "强基线、消融与统计封口", hours: "14-18h", depends: ["W10", "W12"],
      learn: ["matched capacity/compute", "multi-fidelity screen 与 survivor bias", "paired statistics 与 effect size", "ablation 与 mechanism claim 的一对一关系"],
      build: ["传统/SIRT/NeRIF/U-Net/FNO/NIO-style 对照表", "复算 v3g 短程与长程排名翻转", "预注册主指标、超参数预算与停止准则", "生成八张核心论文图和 checksum"],
      pass: ["每个创新点都有直接消融", "筛选成本与最终训练成本分开报告", "超参数预算公平", "结论不超出数据分布和统计单元"],
      resources: ["nio-paper", "neural-correction", "v3g-deeponet-capacity", "t16-readme"],
      paper: "从‘有个新模型’转为‘有一条可反驳的机制证据链’。"
    },
    {
      id: "W15", phase: "投稿决策", week: "第 15-16 周", title: "可展示系统、论文包与师兄审核", hours: "16-24h", depends: ["W14"],
      learn: ["论文叙事与证据等级", "可复现包和许可边界", "投稿、降级和停止决策"],
      build: ["三维切片/等值面/观测重投影互动展示", "paper figures + tables + appendix + code README", "一页师兄审核材料和十分钟演示"],
      pass: ["新环境一条命令跑通主表", "页面不暗示合成结果已代表真实流场", "师兄对任务定义、数据和主指标明确签字/回复"],
      resources: ["t16-evidence", "t16-readme", "paper-library-core"],
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
    {id:"n3-recovery-disclosure",stage:"audit",level:"进阶",type:"研究诚信",title:"N3 KeyError 盲态分析恢复披露",url:"../document_reader.html?doc=docs%2Fn2_pvgr_n3_blind_analysis_recovery_2026-07-18.md",local:"",read:"理解为什么 96 格完成后仍不能直接改字段继续汇总，以及 opaque Merkle 封存如何限制事后自由度。",output:"写出允许修改的唯一 query schema 映射和所有禁止修改项。",verified:"恢复协议先提交；checkpoint payload 未解析；96 格未重算"}
  ],
  researchTracks: [
    {
      id:"pvgr-residual", rank:1, title:"Certified residual BOST operator", badge:"当前主线：mixed-scale adjoint + transversality-aware support", risk:"高", novelty:"把 curved-ray residual 的低信号 backward-error 证书、移动 support 等值面的 simple-root/transversality 证书、Picard residual 与 fail-closed fallback 放进同一可证伪合同；创新不来自把 FNO 换名，也不允许拿 post-open p14 反调阈值", data:"已有 N3 96 条件、D3 32×256×2 mixed reference、D4 tiny pass、D4b 完整 32-cell fail-closed census，以及精确 contraction/21-bit 取证；仍缺独立 D4c development/fresh population、真实 mask 语义、decoder chain、8-view reconstruction、独立 generator 和 OERF flow-off noise", hardware:"D4b 在本机 CPU 用 333.070 秒完成 12,558,336 queries、峰值约 475.3 MiB；本轮 90 signature replay 与精确 contraction 约 7 秒，D4c 小规模证书仍可本机；32³-64³ 多模型多种子以后再评估 GPU", question:"能否用 mixed-scale backward-error 区分 residual 低信号与真实 adjoint inconsistency，并用 simple-root certificate 允许平滑 crossing 位移、同时拒绝 root birth/death/grazing 和危险几何事件？", contribution:"D4b 暴露局部失败；取证排除最终求和顺序、量化 14,466.67× signal suppression，并把 21 个 support-set flips 定位到 ray/stage/offset。候选贡献收缩为两类可验证导数证书 + inverse closure + small operator + fallback。", next:"独立 D4c development→冻结 mixed-scale/simple-root 规则→fresh derivative audit→decoder-chain dot/FD→6+2 view reconstruction→真实 noise units→同预算 DeepONet/FNO/FFNO。", stop:"若师兄 renderer 无 active mask 且 support 证书对 FD/optimizer 无预测价值，则停止 topology 主贡献；若 mixed-scale 规则不能 fresh 闭合、decoder 任一 context 失败、H-P1 低于实验噪声或同预算尾部不优于 P1，则停止 residual operator。"
    },
    {
      id:"correction", rank:2, title:"v3k-F 预白化停止与强数值侧门", badge:"确定性辅线；learned stop 继续关闭", risk:"中", novelty:"它主要是强基线和可部署停止问题，不单独宣称新算子", data:"v3k-F 仍是 development synthetic audit；真实 camera covariance 与 fresh blind 暂缺", hardware:"本机维护 discrepancy/SPG/call frontier，不为 learned stop 扩网络", question:"部署可得 covariance/noise proxy 能否稳定控制 semi-convergence，为残差算子提供不可逃避的强数值对手？", contribution:"discrepancy 对 noise OOD 相对 fixed Landweber 平均 +7.10%，但仍有 12.5% harm；平均改善不足以放行 learned stop。", next:"真实 flow-off covariance→pooled/worst-camera discrepancy→fresh lock→只作强数值基线。", stop:"确定性方法已解释全部 headroom 时，不训练 stopping network。"
    },
    {
      id:"baseline", rank:3, title:"240-epoch FNO + 有界 DeepONet baseline audit", badge:"epoch / cost / DeepONet budget 已锁", risk:"低-中", novelty:"低，但决定全部创新结论是否可信", data:"同一 K=6 validation protocol；策略和 checkpoint 冻结后才读取 dev2/Q_audit", hardware:"v3e：FNO 14.72M FLOPs-v1、1.49ms 推理、8.54ms 完整训练步；v3g 72-run DeepONet screen 本机 153.69s", question:"候选能否在 epoch、wall time、FLOPs 与内存多轴上形成稳定 Pareto 优势，而不是只赢一个弱 endpoint？", contribution:"validation 冠军、plateaued control、五架构成本 schema、DeepONet rank/pool/LR 有界审计、provenance hashes 与尾部/OOD 边界。", next:"冻结 rank-48 DeepONet control；停止规则稳定后补 operator warm-start 的 time-to-target。", stop:"不再用参数量或无限延长 baseline 阻塞写代码；top-3 DeepONet 补充只有师兄明确要求才一次性预注册，不能看 dev2 后扩表。"
    },
    {
      id:"sensor-design", rank:4, title:"BOST camera layout × inverse operator co-design", badge:"中期候选；依赖真实安装约束", risk:"中-高", novelty:"高，依赖真实安装约束", data:"多套可行相机布局、遮挡/标定/同步成本与真实或高保真 forward", hardware:"8-24 GB GPU；主要成本在实验几何与批量 forward", question:"在固定安装成本下，哪些视角布局能同时降低平均误差和最坏体场风险？", contribution:"block-camera VOI、风险约束布局、operator 与实验设计协同。", next:"先比较 uniform/max-gap/QR-Fisher/learned policy；K=6 max-gap 仅作假设来源，不作正结果。", stop:"布局收益在独立 geometry/真实装置消失，或安装成本不可接受时停止。"
    },
    {
      id:"cgpdno", rank:5, title:"历史支线：Base-Correction CG-PDNO", badge:"3-seed 工程信号；不再是当前主线", risk:"中高", novelty:"可能来自 variable ray geometry + calibrated covariance + 共享物理 fallback；但证据仍在 straight-ray analytic phantom", data:"已有 disjoint geometry pools 和 3 seeds，仍缺独立 generator、OpenBOST 与 OERF", hardware:"本机已完成 8³-32³ smoke；当前不继续扩模", question:"共享预白化物理 base 上的一次 correction 能否在 sparse-view BOST 同调用前沿不伤尾部？", contribution:"development-fresh test 平均 +21.77%、最大种子 harm 0%，但 joint-OOD 前驱灾难与 noise-only trust 负结果关闭了成功表述。", next:"仅作为 residual-operator 的历史对照和结构警告，不与 N3 curved-ray 证据混写。", stop:"除非 N3 residual target 与真实数据合同支持同一结构，否则不恢复扩展。"
    },
    {
      id:"qcsnco", rank:7, title:"Query-Calibrated Support-Null Correction Operator", badge:"停止当前 checkpoint；保留负结果", risk:"高", novelty:"组件邻近既有方法，且当前 direction 质量不足", data:"若恢复，需训练 mask 匹配、锁定新 fields、真实 cone-ray 与独立采集", hardware:"8-24 GB GPU；另需 Q_audit 仪器", question:"null consistency 为什么不能保证三维 field correctness？", contribution:"当前最可信贡献是机制性负结果与相机信息价值边界。", next:"不调新架构；只在 direct/warm-start 主线稳定后，决定是否做一次训练匹配复核。", stop:"v2d 在 K=4/6/8 的同预算门槛 0/3 通过，已触发停止条件。"
    },
    {
      id:"geometry", rank:8, title:"可变相机/光线几何神经算子", badge:"高风险升级", risk:"高", novelty:"高，但容易过度建模", data:"至少多套布局，必须 outer leave-one-geometry-out", hardware:"16-48 GB GPU；point/ray encoder 成本高", question:"固定网格 FNO 的失败是否主要来自观测几何变化？", contribution:"将 camera/ray metadata 与 inverse operator 显式联系，支持可变传感器。", next:"先用 metadata-FNO/padding+mask 建立强对照，再决定 GINO/VIDON/GNOT。", stop:"显式 metadata 基线已解决外推，或组内布局实际固定时停止。"
    },
    {
      id:"warmstart", rank:6, title:"Own operator warm-start NeRIF", badge:"baseline、geometry、v3d 三闸门后进入", risk:"中-高", novelty:"中-高", data:"需 NeRIF forward/loss、统一 stop rule 与至少少量真实样例", hardware:"本机 MPS 先做小场 timing；升尺度后再评 GPU", question:"通过 v3d 开发门槛的初值能否比 random/ridge/plateau FNO/DeepONet 更快达到相同 NeRIF 独立审计质量？", contribution:"把强数值反演、BOST acquisition-conditioned prior 与 per-instance continuous field 变成公平级联。", next:"只在 acquisition-conditioned 分支稳定胜 F-Adapter 与 geometry controls 后比较 random/ridge/FNO/DeepONet/own/oracle 六种初始化。", stop:"端到端总时间改善不足 30%、最终 Q_audit 变差或尾部伤害增加时不作主贡献。"
    },
    {
      id:"4d", rank:9, title:"3D inverse operator → 4D evolution/correction", badge:"远期扩展", risk:"很高", novelty:"高", data:"需同步高速多视角时序数据", hardware:"24-80 GB GPU 或明确低秩/分块设计", question:"低秩时空先验能否在不损伤瞬态前缘的情况下减少 4D BOST 的样本与计算量？", contribution:"与 He 的 TDBOST 时空线直接对齐。", next:"只在静态 3D 模型、几何接口和真实数据都过关后开始。", stop:"静态 3D 尚不稳定，或没有同步时序数据时不启动。"
    }
  ],
  paperGates: [
    {id:"G0",title:"任务锁定",evidence:"输入、输出、support/query 角色、主指标和相机成本经师兄确认。",stop:"任务仍在 3D inverse 与 4D evolution 之间摇摆。"},
    {id:"G1",title:"数据无泄漏",evidence:"phantom/family/view/noise/geometry 的 manifest 和 outer split 在训练前固定。",stop:"调参见到外层 geometry 或真实 test。"},
    {id:"G2",title:"强基线公平",evidence:"传统正则化、physics lift/SIRT、matched U-Net/CNO/FNO、NeRIF；迭代精化另含 fixed Landweber、quadratic、PBB、全调用记账 SPG 与固定/自适应 stop。",stop:"新法只和弱 endpoint、单一 optimizer protocol、更多 A/A^T 调用或 oracle stop 对照。"},
    {id:"G3",title:"机制被消融",evidence:"support-fit、null projection、query calibration 和 geometry encoder 各有一对一消融。",stop:"只有整体模型对比，不知道改善来自哪里。"},
    {id:"G4",title:"独立泛化",evidence:"至少包含 family OOD、view/noise joint OOD 和 leave-one-geometry-out。",stop:"只报 IID 或随机切分。"},
    {id:"G5",title:"真实/高保真证据",evidence:"至少一个与训练生成器独立的公开或组内数据包。",stop:"所有结论都在同一 synthetic generator 内循环。"},
    {id:"G6",title:"统计单元正确",evidence:"多种子、逐样本 paired delta、置信区间/效应量和预定停止规则。",stop:"把 voxel 当独立样本或只报最好 seed。"},
    {id:"G7",title:"可复现与展示",evidence:"环境、config、checkpoint 边界、checksums、图表脚本、三维展示和许可清单完整。",stop:"只在原机器/原 notebook 中手工跑通。"}
  ],
  seniorQuestions: [
    "组内 displacement 的单位是 pixel、归一化 detector coordinate 还是物理长度；forward observable 对应 exit angle、path-integrated deflection 还是最终 pixel displacement？",
    "同一 rig / flow-off 是否有 20 次左右 repeats，可否提供 per-pixel/ray mask、confidence 或 optical-flow covariance 来冻结实验噪声底？",
    "D2 的 narrow-aperture selected tail 已到 H8192 且观测阶约 2.2-2.5；组内会把这种数值尾差与真实光圈、pixel displacement 和 flow-off noise 怎样对齐？",
    "组内所说的算子学习，是 projection-to-volume inverse operator，还是 3D/4D field evolution operator？",
    "能否给一份可脱敏的 displacement + camera geometry + mask/confidence + baseline reconstruction 最小数据包？",
    "哪些相机必须参与重建，是否允许保留 1-2 个 query cameras 只做标定/验证？",
    "相机数量、位姿和光线几何是否会在样本之间变化？",
    "是否认可把 v3f rank-48 DeepONet 冻结为毕设 control；投稿前是否还需要一次性 top-3 long-horizon 补充？",
    "师兄最看重 field error、held-out reprojection、front location、速度还是失败率？",
    "是否有 NeRIF/TDBOST 可调用的 forward F(x) 与匹配 Jacobian-adjoint/VJP？真实 support 是否部署可得？",
    "真实 NeRIF/BOST renderer 是否有随网络参数变化的 hard mask、occupancy pruning、ray termination 或 threshold branch？support threshold 会改变 forward 值，还是只用于 ROI/可视化/证书？",
    "D4b 的 p14 精确 contraction 仍不过门且 component/residual signal ratio 为 14,466.67；能否用组内 F/Jv/Jᵀq 小例子逐段对账？NeRIF/TDBOST 的 decoder 是 voxel、MLP 还是 tensor decomposition？",
    "能否提供一个匿名最小导数 case：相机参数、1 个 field checkpoint、4 条 rays、forward output、loss cotangent、期望 Jv/Jᵀq 与单位？",
    "真实 displacement 是否提供 pixel/ray confidence、noise level 或 covariance proxy，能否只用 calibration split 估计？",
    "能否按独立 experiment case/geometry 封存一套从未查看的 blind fields？",
    "真实实验中多一台 query camera 的同步、标定和布置成本是多少？",
    "哪些数据、图和代码可公开，哪些只能留在组内？"
  ]
};
