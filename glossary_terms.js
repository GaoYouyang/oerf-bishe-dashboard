window.OERF_GLOSSARY_TERMS = [
  {
    zh: "背景纹影",
    en: "Background-Oriented Schlieren, BOS",
    category: "BOST 主线",
    aliases: ["BOS", "background-oriented schlieren", "background oriented schlieren", "background displacement"],
    explain: "用已知背景图案作为参考尺。流场中的折射率梯度会让光线发生微小偏折，最终表现为背景图像的像素位移。",
    use: "这是 BOST、PIV-BOST 和很多系统误差分析的入口。算法通常先处理 BOS 图像位移，再进入三维反演或速度补偿。",
    pitfall: "BOS 直接测到的是图像位移或偏折效应，不是速度场、温度场或密度场本身。",
    keywords: "BOS; background displacement; optical flow BOS; density gradient visualization"
  },
  {
    zh: "背景纹影层析",
    en: "Background-Oriented Schlieren Tomography, BOST / TBOS",
    category: "BOST 主线",
    aliases: ["BOST", "TBOS", "tomographic BOS", "tomographic background-oriented schlieren", "background-oriented schlieren tomography"],
    explain: "从多个视角的 BOS 位移场反演三维折射率、密度或温度场，本质是光学投影测量加三维反问题。",
    use: "这是何远哲 NeRIF、4D BOST 和 PIV-BOST 补偿共同依赖的主技术。你的题名最好保留 BOST/TBOS 或 refractive-index reconstruction。",
    pitfall: "完整 BOST 不只是神经网络，还包括相机几何、背景标定、位移估计、正则化、单位换算和重投影验证。",
    keywords: "BOST; TBOS; refractive-index tomography; density tomography; multi-view BOS"
  },
  {
    zh: "神经折射率场",
    en: "Neural Refractive Index Field, NeRIF",
    category: "神经场/AI",
    aliases: ["NeRIF", "neural refractive index field", "coordinate MLP tomography"],
    explain: "用坐标神经网络把三维坐标映射到折射率值或相关物理量，形成连续的隐式场表示。",
    use: "这是你最贴何远哲的主线。先用合成 BOST forward model 跑通 NeRIF 风格重建，再做视角、噪声和模型容量鲁棒性分析。",
    pitfall: "NeRIF 不是把图片直接扔进网络得到答案；它需要可微投影、物理 loss 和重投影一致性约束。",
    keywords: "NeRIF; neural refractive index field; implicit neural representation; coordinate network"
  },
  {
    zh: "神经偏折场",
    en: "Neural Deflection Field, NeDF",
    category: "神经场/AI",
    aliases: ["NeDF", "neural deflection field", "sparse-view TBOS"],
    explain: "另一类神经隐式表示路线，强调从稀疏视角的偏折观测学习与折射相关的场。",
    use: "适合作为 NeRIF 的方法邻居和对照项，帮助你讨论“表示折射率本身、梯度，还是偏折场”的差异。",
    pitfall: "不要把 NeDF 和 NeRIF 写成同一个东西；它们的表示变量、loss 和可解释物理量可能不同。",
    keywords: "NeDF; sparse-view reconstruction; neural deflection; neural BOS tomography"
  },
  {
    zh: "神经折射率基元",
    en: "Neural Refractive Index Primitives, NRIP",
    category: "神经场/AI",
    aliases: ["NRIP", "neural refractive index primitives", "neural primitives"],
    explain: "把折射率场拆成可学习的局部或结构化基元，用来表达火焰等复杂折射率分布。",
    use: "可作为 NeRIF 后续进阶读物。如果你要比较不同神经表示，NRIP 是自然的第三个参考点。",
    pitfall: "本科阶段不建议直接从 NRIP 开题；先跑通 NeRIF/voxel baseline，再讨论表示升级。",
    keywords: "NRIP; neural primitives; flame field reconstruction; hash encoding"
  },
  {
    zh: "折射率场",
    en: "Refractive index field, n(x,y,z)",
    category: "BOST 主线",
    aliases: ["refractive index field", "refractive index", "n(x,y,z)", "n-1"],
    explain: "空间中每一点的折射率分布，决定光线在流场中如何弯折。",
    use: "NeRIF 最直接的输出对象通常就是折射率场；后续可通过 Gladstone-Dale 关系换算密度。",
    pitfall: "折射率变化很小，数值尺度容易被忽略；画图时常显示 n-1 或归一化扰动。",
    keywords: "refractive index field; optical path; gas refractivity; refractive-index perturbation"
  },
  {
    zh: "折射率梯度",
    en: "Refractive index gradient, grad n",
    category: "BOST 主线",
    aliases: ["refractive index gradient", "gradient of refractive index", "grad n", "density gradient"],
    explain: "折射率空间变化率，是造成光线偏折和背景图像位移的核心原因。",
    use: "很多 BOS 公式从折射率梯度的路径积分出发；PIV-BOST 补偿也依赖它解释像素偏移。",
    pitfall: "从梯度恢复标量场需要积分或约束，噪声会带来漂移和边界问题。",
    keywords: "refractive index gradient; ray deflection; gradient integration; Poisson integration"
  },
  {
    zh: "Gladstone-Dale 关系",
    en: "Gladstone-Dale relation",
    category: "流体/燃烧",
    aliases: ["Gladstone-Dale", "Gladstone Dale", "refractive index density relation", "K rho"],
    explain: "把折射率和密度联系起来的经验关系，常写作 n - 1 = K rho。",
    use: "它是从 BOST 折射率场进入密度场、温度场和流体解释的桥。",
    pitfall: "K 与气体成分、波长和温度有关；燃烧多组分场不能无脑用单一常数解释所有误差。",
    keywords: "Gladstone-Dale; density from BOS; gas refractivity; refractive index density"
  },
  {
    zh: "密度场",
    en: "Density field, rho(x,y,z)",
    category: "流体/燃烧",
    aliases: ["density field", "density gradient", "rho", "volumetric density"],
    explain: "单位体积质量分布，是可压缩流、热羽流、火焰和喷流诊断中的核心物理量。",
    use: "BOST 重建折射率后常把它转成密度，再进一步解释温度、压力或流动结构。",
    pitfall: "密度不是温度；需要状态方程、压力假设和组分信息才能进一步换算。",
    keywords: "density field; gas density measurement; volumetric density; compressible flow"
  },
  {
    zh: "温度场",
    en: "Temperature field, T(x,y,z)",
    category: "流体/燃烧",
    aliases: ["temperature field", "thermometry", "gas temperature", "T(x,y,z)"],
    explain: "空间温度分布，火焰、热羽流和反应流诊断中非常重要。",
    use: "BOST 可以在一定假设下由密度推温度；PLIF、TAS、LII 等则可能直接或间接提供温度信息。",
    pitfall: "从 BOST 推温度需要压力和组分假设，不能把折射率图直接称作温度图。",
    keywords: "temperature field; gas thermometry; density temperature conversion; flame temperature"
  },
  {
    zh: "背景位移场",
    en: "Background displacement field",
    category: "BOST 主线",
    aliases: ["background displacement", "image displacement", "pixel shift", "displacement field"],
    explain: "流场存在前后背景图案在图像上的像素偏移，通常有水平和竖直两个分量。",
    use: "这是 BOS 测量的直接数据，也是 BOST forward model 和 reprojection error 的主要比较对象。",
    pitfall: "位移场不是速度场；不要把 u/v 像素位移和流体速度 u/v 混用。",
    keywords: "image displacement; BOS displacement field; pixel shift; deflection measurement"
  },
  {
    zh: "重投影误差",
    en: "Reprojection error / projection consistency",
    category: "反问题/层析",
    aliases: ["reprojection error", "projection consistency", "forward validation", "re-projection"],
    explain: "用重构出的三维场重新生成各视角观测，再与真实测得的位移或投影比较。",
    use: "本科项目里它是最容易说清楚的可信度指标：重构得好不好，至少要能解释原始观测。",
    pitfall: "重投影误差低不一定代表三维真值准确；病态反问题可能有多个场解释同一组投影。",
    keywords: "reprojection error; projection consistency; tomography validation; forward validation"
  },
  {
    zh: "前向模型",
    en: "Forward model",
    category: "反问题/层析",
    aliases: ["forward model", "forward projection", "differentiable projection", "ray integration"],
    explain: "给定三维物理场和相机几何，计算理论上应该观测到什么图像位移、投影或信号。",
    use: "你做合成数据和 NeRIF 训练时，前向模型是最重要的代码模块。",
    pitfall: "前向模型过于简化会让 demo 好看但无法迁移真实数据；需要明确小角度、直线光线或 ray tracing 假设。",
    keywords: "forward model; differentiable projection; ray tracing; BOS simulation"
  },
  {
    zh: "反问题",
    en: "Inverse problem",
    category: "反问题/层析",
    aliases: ["inverse problem", "inverse modeling", "inverse reconstruction", "computational imaging"],
    explain: "从观测反推未知物理场。BOST 从多视角位移反推三维折射率/密度，就是典型反问题。",
    use: "开题报告要把你的项目放在“光学诊断 + 反问题 + 神经表示”这个框架里。",
    pitfall: "反问题通常病态，不能只靠训练 loss 下降说明结果可靠。",
    keywords: "inverse problem; inverse modeling; optical inverse problem; computational imaging"
  },
  {
    zh: "病态性",
    en: "Ill-posedness",
    category: "反问题/层析",
    aliases: ["ill-posed", "ill-posedness", "non-uniqueness", "instability"],
    explain: "解可能不存在、不唯一，或者对噪声极端敏感，这是少视角层析的核心困难。",
    use: "你的鲁棒性分析、少视角扫描和正则化设计都可以围绕病态性展开。",
    pitfall: "不要把“网络能输出一张图”误当成“问题已经有唯一正确解”。",
    keywords: "ill-posed inverse problem; instability; non-uniqueness; sparse-view tomography"
  },
  {
    zh: "少视角重构",
    en: "Sparse-view reconstruction",
    category: "反问题/层析",
    aliases: ["sparse-view", "few-view", "limited view", "limited angle"],
    explain: "相机视角数量有限时进行三维重构，信息不足导致伪影和不稳定。",
    use: "非常适合本科切入：3/5/7/9 视角扫描、噪声扫描和 baseline 对比都能形成清晰图表。",
    pitfall: "少视角不是单纯减少输入张数；相机角度分布和缺失方向同样重要。",
    keywords: "sparse-view reconstruction; limited-view tomography; limited-angle tomography; few-view BOST"
  },
  {
    zh: "正则化",
    en: "Regularization",
    category: "反问题/层析",
    aliases: ["regularization", "Tikhonov", "total variation", "smoothness prior", "loss weighting"],
    explain: "给反问题增加额外约束，例如平滑、稀疏、低秩、物理方程或边界条件。",
    use: "传统 BOST baseline 和 NeRIF loss 都离不开正则化；它也是参数扫描的核心。",
    pitfall: "正则化太强会过平滑，太弱会噪声放大；必须用指标和图像同时判断。",
    keywords: "regularization; Tikhonov; TV; smoothness; loss weight"
  },
  {
    zh: "ART / SIRT / SART",
    en: "Algebraic reconstruction techniques",
    category: "反问题/层析",
    aliases: ["ART", "SIRT", "SART", "algebraic reconstruction", "iterative tomography"],
    explain: "把层析重构离散成线性方程组后，用迭代方式求解的经典体素方法。",
    use: "它们是 NeRIF 论文里最自然的传统 baseline。你要能解释为什么神经场可能比体素 baseline 更连续或更稳。",
    pitfall: "传统 baseline 不等于落后；它们可解释、稳定、容易做误差分析，是答辩时的参照物。",
    keywords: "ART; SIRT; SART; algebraic reconstruction; iterative reconstruction"
  },
  {
    zh: "Landweber 迭代",
    en: "Landweber iteration",
    category: "反问题/层析",
    aliases: ["Landweber", "Landweber iteration", "iterative regularization"],
    explain: "一种用梯度下降思想求解线性反问题的经典迭代方法。",
    use: "适合作为你写“传统反问题 baseline”的补充词，也可帮助理解 loss 最小化。",
    pitfall: "迭代次数本身就是正则化；跑太久可能拟合噪声。",
    keywords: "Landweber iteration; iterative regularization; inverse problem baseline"
  },
  {
    zh: "Poisson 积分",
    en: "Poisson integration",
    category: "反问题/层析",
    aliases: ["Poisson integration", "Poisson equation", "gradient integration", "displacement integration"],
    explain: "从梯度场恢复标量场时常见的数学步骤，可通过求解 Poisson 方程实现。",
    use: "BOS 位移、折射率梯度和标量折射率/密度之间的桥常会涉及它。",
    pitfall: "边界条件会显著影响结果；梯度噪声也会在积分中积累。",
    keywords: "Poisson integration; gradient integration; Poisson constrained PINN; displacement integration"
  },
  {
    zh: "坐标神经网络",
    en: "Coordinate MLP / coordinate-based neural network",
    category: "神经场/AI",
    aliases: ["coordinate MLP", "coordinate network", "coordinate-based neural network", "implicit neural representation", "INR"],
    explain: "输入坐标，输出该点物理量的神经网络，常用于连续场表示。",
    use: "NeRIF、NeDF、神经场重构都依赖这个概念；你需要会用 PyTorch 写最小版。",
    pitfall: "普通 MLP 容易学不出高频细节，通常需要 Fourier features 或其他编码。",
    keywords: "coordinate MLP; coordinate network; implicit neural representation; neural field"
  },
  {
    zh: "位置编码 / 傅里叶特征",
    en: "Positional encoding / Fourier features",
    category: "神经场/AI",
    aliases: ["positional encoding", "Fourier features", "sinusoidal encoding", "spectral bias"],
    explain: "把低维坐标映射到高频特征，帮助 MLP 表达细节结构。",
    use: "你的 NeRIF 复现可以把编码频率、层数、宽度作为第一批参数扫描。",
    pitfall: "高频编码可能增强细节，也可能更容易拟合噪声。",
    keywords: "positional encoding; Fourier features; sinusoidal features; spectral bias"
  },
  {
    zh: "哈希编码",
    en: "Hash encoding / multi-resolution hash grid",
    category: "神经场/AI",
    aliases: ["hash encoding", "multi-resolution hash grid", "instant-ngp", "hash grid"],
    explain: "用多分辨率哈希表表示空间特征，常用于加速神经场训练和提高局部细节表达。",
    use: "如果普通 NeRIF 太慢，哈希编码是后续加速实验方向。",
    pitfall: "实现复杂度更高，也可能引入额外超参数；本科初期不必硬上。",
    keywords: "hash encoding; multi-resolution hash grid; neural field acceleration; instant NGP"
  },
  {
    zh: "物理约束神经网络",
    en: "Physics-Informed Neural Network, PINN",
    category: "神经场/AI",
    aliases: ["PINN", "physics-informed neural network", "physics constrained", "physics-informed"],
    explain: "把物理方程、边界条件或守恒关系写入 loss，让网络不仅拟合数据，也符合物理。",
    use: "BOS/BOST 后续可从密度场进一步推速度、压力或温度场时，PINN 是重要方法邻居。",
    pitfall: "PINN 不自动保证正确；物理方程、尺度归一化和 loss 权重很容易让训练失败。",
    keywords: "PINN; physics-informed neural network; physics-constrained reconstruction; flow inference"
  },
  {
    zh: "数据同化",
    en: "Data assimilation",
    category: "反问题/层析",
    aliases: ["data assimilation", "EnKF", "ensemble Kalman filter", "variational assimilation"],
    explain: "把实验观测和物理/数值模型融合，得到更一致的状态估计。",
    use: "蔡老师组也有反应流仿真与数据同化方向；BOST 重构场可作为观测进入同化框架。",
    pitfall: "数据同化比单纯重构更依赖流体方程和数值模拟，本科阶段适合做 toy，不宜一开始开大题。",
    keywords: "data assimilation; ensemble Kalman filter; variational assimilation; reacting flow"
  },
  {
    zh: "张量分解",
    en: "Tensor decomposition",
    category: "神经场/AI",
    aliases: ["tensor decomposition", "CP decomposition", "Tucker decomposition", "tensor factorization"],
    explain: "把高维时空数据拆成低秩因子，减少参数量并利用时序/空间相关性。",
    use: "4D BOST 论文的关键词。你可以做低秩 X-Y-Z-T toy，而不是完整复现 TOG 系统。",
    pitfall: "低秩假设并不总成立；剧烈湍流或火焰细节可能需要更高秩或局部模型。",
    keywords: "tensor decomposition; low-rank representation; CP decomposition; Tucker decomposition; 4D reconstruction"
  },
  {
    zh: "低秩先验",
    en: "Low-rank prior",
    category: "反问题/层析",
    aliases: ["low-rank", "low-rank prior", "temporal coherence", "dynamic tomography"],
    explain: "假设数据可以由较少的基模式表示，用于稳定反演和压缩时序重构。",
    use: "是 4D BOST 子模块的可毕业切口：比较逐帧重构与低秩时序表示。",
    pitfall: "低秩可能丢掉瞬态小结构；需要用误差图说明哪些区域被牺牲。",
    keywords: "low-rank prior; temporal coherence; dynamic tomography; 4D BOST"
  },
  {
    zh: "粒子图像测速",
    en: "Particle Image Velocimetry, PIV",
    category: "PIV/速度场",
    aliases: ["PIV", "particle image velocimetry", "tracer particles"],
    explain: "通过连续图像中示踪粒子位移估计速度场，是实验流体力学常用速度测量方法。",
    use: "PIV-BOST 的核心是用 BOST 折射率场补偿 PIV 测量中的折射误差。",
    pitfall: "PIV 的像素位移经过时间间隔和尺度换算才是速度；与 BOS 背景位移不是同一个量。",
    keywords: "PIV; particle image velocimetry; velocity measurement; tracer particles"
  },
  {
    zh: "立体 PIV",
    en: "Stereo-PIV",
    category: "PIV/速度场",
    aliases: ["stereo-PIV", "stereo PIV", "stereo velocity", "stereo-velocity"],
    explain: "用两个或多个相机从不同角度拍摄粒子图像，恢复平面内三个速度分量。",
    use: "何远哲参与的 stereo-velocity PIV-BOST 后续论文涉及这个概念。",
    pitfall: "立体 PIV 对标定和折射更敏感，直接作为本科第一阶段风险偏高。",
    keywords: "stereo-PIV; stereo velocity measurement; refractive index compensation; turbulent combustion"
  },
  {
    zh: "折射率补偿",
    en: "Refractive index compensation",
    category: "PIV/速度场",
    aliases: ["refractive index compensation", "PIV-BOST", "optical distortion correction", "velocity error"],
    explain: "用折射率场估计光线路径偏折，从而修正图像位置或速度测量误差。",
    use: "这是 PIV-BOST 论文的关键词，也是你第二阶段可能从 NeRIF 升级到真实需求的方向。",
    pitfall: "补偿可以发生在图像层、坐标层或速度场层，要问清师兄组内采用哪一种。",
    keywords: "PIV-BOST; refractive index compensation; optical distortion correction; velocity error"
  },
  {
    zh: "光流",
    en: "Optical flow",
    category: "光学诊断",
    aliases: ["optical flow", "Horn-Schunck", "Lucas-Kanade", "Farneback", "RAFT"],
    explain: "估计两帧图像间每个像素的运动或位移，是 BOS 位移估计的重要方法之一。",
    use: "你可以做 BOS 位移估计 benchmark，比较传统相关法、Horn-Schunck、Lucas-Kanade 和现代光流。",
    pitfall: "光流估计的是图像亮度模式运动，不等于物理粒子速度；BOS 中还需和折射模型连接。",
    keywords: "optical flow; Horn-Schunck; Lucas-Kanade; BOS displacement estimation"
  },
  {
    zh: "互相关位移估计",
    en: "Cross-correlation displacement estimation",
    category: "光学诊断",
    aliases: ["cross-correlation", "correlation peak", "interrogation window", "subpixel"],
    explain: "把图像切成窗口，通过相关峰值估计局部位移，PIV 和 BOS 都常用。",
    use: "传统、稳健、容易解释，适合作为位移估计 baseline。",
    pitfall: "窗口大小会影响空间分辨率和噪声；强折射或大位移时可能丢峰。",
    keywords: "cross-correlation; PIV interrogation window; BOS displacement; subpixel peak"
  },
  {
    zh: "相机几何",
    en: "Camera geometry / camera calibration",
    category: "实验/数据",
    aliases: ["camera geometry", "camera calibration", "intrinsic", "extrinsic", "multi-view geometry"],
    explain: "相机内参、外参、畸变、视角和背景/测量区域位置共同决定投影关系。",
    use: "真实 BOST 数据能否跑通，很大程度取决于你是否拿到并正确理解这些参数。",
    pitfall: "没有相机几何就很难从“好看的图像”变成“可信三维物理场”。",
    keywords: "camera calibration; intrinsic parameters; extrinsic parameters; multi-view geometry; BOS calibration"
  },
  {
    zh: "标定误差",
    en: "Calibration error",
    category: "实验/数据",
    aliases: ["calibration error", "geometric uncertainty", "camera perturbation", "sensitivity analysis"],
    explain: "相机参数、背景位置、尺度转换或同步信息的误差。",
    use: "这是你可以做成可毕业小题的方向：分析标定扰动对 BOST/NeRIF 结果的影响。",
    pitfall: "真实系统误差往往比网络结构差异更致命，答辩时要主动承认和量化。",
    keywords: "calibration error; geometric uncertainty; sensitivity analysis; camera perturbation"
  },
  {
    zh: "背景图案",
    en: "Background pattern",
    category: "实验/数据",
    aliases: ["background pattern", "random dot pattern", "projected pattern", "speckle background"],
    explain: "BOS 中用于测量图像位移的纹理背景，可以是随机点、棋盘、投影图案或自然复杂背景。",
    use: "位移估计精度与背景纹理强相关；做低成本 BOS demo 时这是第一批要设计的东西。",
    pitfall: "图案太稀、太规则或对比度不足都会让相关/光流估计失效。",
    keywords: "background pattern; random dot pattern; projected pattern; BOS texture"
  },
  {
    zh: "事件相机",
    en: "Event camera",
    category: "光学诊断",
    aliases: ["event camera", "event-based", "event-based BOS", "frame-event fusion"],
    explain: "只在像素亮度变化时输出事件，具有高时间分辨率和低冗余特点。",
    use: "OERF 有 event-based BOS / imaging velocimetry 相关论文，可作为高速流动显示方法邻居。",
    pitfall: "事件数据不是普通帧图像；需要处理时间戳、极性和重建/融合策略。",
    keywords: "event camera; event-based BOS; event-based imaging velocimetry; frame-event fusion"
  },
  {
    zh: "数字微镜器件",
    en: "Digital Micromirror Device, DMD",
    category: "光学诊断",
    aliases: ["DMD", "digital micromirror device", "dynamic projection", "structured illumination"],
    explain: "可高速控制投影图案的微镜阵列，常用于结构光、动态背景或主动照明。",
    use: "OERF 的事件相机和 PTV/PIV 论文中有 DMD 相关路线，可作为实验硬件背景。",
    pitfall: "DMD 方向偏硬件和同步控制，若没有设备，不适合作为你的主线。",
    keywords: "DMD; digital micromirror device; dynamic projection; structured illumination"
  },
  {
    zh: "平面激光诱导荧光",
    en: "Planar Laser-Induced Fluorescence, PLIF",
    category: "光学诊断",
    aliases: ["PLIF", "planar laser-induced fluorescence", "laser-induced fluorescence"],
    explain: "用激光片激发特定物种或示踪剂发光，测量二维浓度、温度或火焰结构。",
    use: "OERF 极端反应流诊断的重要方向；你需要了解它和 BOST/PIV 如何互补。",
    pitfall: "PLIF 和 BOS 的成像物理完全不同，不能把一个方法的误差模型直接套到另一个上。",
    keywords: "PLIF; laser-induced fluorescence; flame diagnostics; species concentration"
  },
  {
    zh: "层析吸收光谱",
    en: "Tomographic Absorption Spectroscopy, TAS",
    category: "光学诊断",
    aliases: ["TAS", "tomographic absorption spectroscopy", "hyperspectral tomography", "absorption spectroscopy"],
    explain: "从多束光吸收信号反演温度和组分场，是蔡老师早期和长期代表方向之一。",
    use: "作为 OERF 反问题谱系的重要背景，帮助你理解“光学观测到物理场”的共性。",
    pitfall: "TAS 依赖光谱线强、吸收模型和非线性反演；不要把它简单等同于图像层析。",
    keywords: "tomographic absorption spectroscopy; hyperspectral tomography; gas diagnostics; line-of-sight absorption"
  },
  {
    zh: "体发射层析",
    en: "Volumetric Emission Tomography, VET",
    category: "光学诊断",
    aliases: ["VET", "volumetric emission tomography", "flame tomography", "chemiluminescence tomography"],
    explain: "从多个视角的发光图像反演三维发射强度场，用于火焰和反应流三维可视化。",
    use: "这是蔡组 3D flame reconstruction 的重要谱系，可作为 NeRIF 的传统三维重构背景。",
    pitfall: "VET 反演的是发射强度，不是折射率；forward model 和物理解释与 BOST 不同。",
    keywords: "volumetric emission tomography; flame tomography; chemiluminescence tomography; view registration"
  },
  {
    zh: "数字全息",
    en: "Digital holography",
    category: "光学诊断",
    aliases: ["digital holography", "holographic imaging", "particle holography"],
    explain: "记录光波干涉信息并数值重建三维粒子或相位信息。",
    use: "OERF 金属颗粒燃烧、颗粒浓度和粒径测量方向常见；适合了解三维成像的另一支。",
    pitfall: "全息重建和 BOST 层析的数学/硬件链路不同，别混写。",
    keywords: "digital holography; particle holography; 3D particle tracking; holographic imaging"
  },
  {
    zh: "激光诱导炽光",
    en: "Laser-Induced Incandescence, LII",
    category: "光学诊断",
    aliases: ["LII", "laser-induced incandescence", "soot diagnostics"],
    explain: "用激光加热碳烟颗粒，通过炽光信号推断碳烟体积分数、温度或粒径。",
    use: "OERF 4D soot diagnostics 是四维重构思想的相邻方向，但物理对象不是折射率场。",
    pitfall: "LII 的信号模型含颗粒热传递和辐射过程，门槛不低。",
    keywords: "LII; laser-induced incandescence; soot diagnostics; 4D soot measurement"
  },
  {
    zh: "反应进度变量",
    en: "Reaction progress variable",
    category: "流体/燃烧",
    aliases: ["reaction progress variable", "progress variable", "flame front", "premixed flame"],
    explain: "描述反应从未燃到已燃推进程度的标量变量，常用于预混火焰建模和实验诊断。",
    use: "何远哲参与的 NH3/CH4 turbulent premixed flames 旁支论文会涉及。",
    pitfall: "它通常依赖温度、组分或标量场定义，不是单独由 BOS 位移直接测得。",
    keywords: "reaction progress variable; turbulent premixed flame; NH3 CH4 flame; flame front"
  },
  {
    zh: "湍流预混火焰",
    en: "Turbulent premixed flame",
    category: "流体/燃烧",
    aliases: ["turbulent premixed flame", "premixed combustion", "flame turbulence"],
    explain: "燃料和氧化剂预先混合后在湍流中燃烧，火焰面被湍流拉伸、皱褶和扰动。",
    use: "OERF 燃烧诊断论文常见场景；PIV-BOST 的真实应用也可能与它相关。",
    pitfall: "如果流体/燃烧基础薄弱，不要一上来做完整机理；先把测量和重构链路弄清楚。",
    keywords: "turbulent premixed flame; flame diagnostics; combustion turbulence; flame front"
  },
  {
    zh: "射流",
    en: "Jet flow",
    category: "流体/燃烧",
    aliases: ["jet flow", "heated jet", "helium jet", "supersonic jet", "underexpanded jet"],
    explain: "流体从孔口或喷嘴喷出形成的流动，是 BOS/BOST、PIV 和湍流实验常见测试对象。",
    use: "热射流、氦气射流、超声速射流都适合作为公开数据或合成 phantom 的物理参照。",
    pitfall: "低速热羽流、欠膨胀射流、超声速射流的物理尺度和可压缩性完全不同。",
    keywords: "jet flow; heated jet; helium jet; supersonic jet; density field"
  },
  {
    zh: "浮力羽流",
    en: "Buoyant plume",
    category: "流体/燃烧",
    aliases: ["buoyant plume", "thermal plume", "natural convection", "plume"],
    explain: "由温度或密度差驱动上升的流动，如热空气、火焰上方热羽流。",
    use: "低成本 BOS demo 和开源 TBOS 数据常以热羽流为对象，安全且易观察。",
    pitfall: "羽流速度和密度结构受环境扰动影响大，重复性可能不如数值 phantom。",
    keywords: "buoyant plume; thermal plume; natural convection; BOS plume"
  },
  {
    zh: "不确定度量化",
    en: "Uncertainty quantification, UQ",
    category: "实验/数据",
    aliases: ["uncertainty quantification", "UQ", "error propagation", "confidence interval"],
    explain: "评估测量、模型和反演结果的不确定性，说明结果可信范围。",
    use: "非常适合本科项目加分：对噪声、视角、标定、网络初始化做误差条或置信区间。",
    pitfall: "只给一张漂亮重构图不够；要说明在哪些条件下会失败。",
    keywords: "uncertainty quantification; error propagation; sensitivity analysis; confidence interval"
  },
  {
    zh: "合成数据",
    en: "Synthetic data / phantom",
    category: "实验/数据",
    aliases: ["synthetic data", "phantom", "synthetic phantom", "ground truth"],
    explain: "人为构造的已知真值数据，用来测试重构算法和误差传播。",
    use: "这是你即使没有组内数据也能毕业的底盘。先用 phantom 跑通 forward/inverse/metrics。",
    pitfall: "合成数据过于理想会掩盖真实图像噪声、标定误差和缺失视角问题。",
    keywords: "synthetic phantom; simulated BOS; ground truth field; benchmark data"
  },
  {
    zh: "真实数据接口",
    en: "Real-data interface",
    category: "实验/数据",
    aliases: ["data interface", "metadata", "manifest", "calibration file", "mask"],
    explain: "把组内真实图像、标定、视角、mask、单位和元数据整理成算法可读格式。",
    use: "如果师兄给数据，你最先要交付的不是大模型，而是能稳定读取、检查、可视化和记录的数据接口。",
    pitfall: "真实数据没有统一接口时，后续所有算法实验都会变成手工修文件。",
    keywords: "data interface; metadata; calibration file; mask; reproducible pipeline"
  },
  {
    zh: "基线方法",
    en: "Baseline method",
    category: "实验/数据",
    aliases: ["baseline", "baseline method", "control experiment", "voxel reconstruction"],
    explain: "用于对比新方法的传统或简单方法，帮助判断改进是否真实。",
    use: "你的 NeRIF 项目至少需要一个 voxel/SIRT/Tikhonov 或逐帧 baseline。",
    pitfall: "不要只和自己方法的不同超参数比；答辩老师会问传统方法在哪里。",
    keywords: "baseline; ablation; control experiment; voxel reconstruction; traditional tomography"
  },
  {
    zh: "消融实验",
    en: "Ablation study",
    category: "神经场/AI",
    aliases: ["ablation", "ablation study", "parameter scan", "loss weight"],
    explain: "逐个去掉或改变模型组件，观察性能变化，用来证明每个设计是否有用。",
    use: "适合你的论文图：编码方式、loss 项、视角数量、正则化权重、噪声水平都可以消融。",
    pitfall: "消融不是随机试几个参数；每一组都要回答一个清晰问题。",
    keywords: "ablation study; parameter scan; loss weight; model capacity; robustness"
  },
  {
    zh: "视图配准",
    en: "View registration",
    category: "实验/数据",
    aliases: ["view registration", "multi-camera calibration", "camera pose", "camera registration"],
    explain: "把多相机视角准确对应到同一个三维坐标或同一个物理对象。",
    use: "多视角火焰层析、BOST、VET 都绕不开；公开数据复现实验也常需要检查它。",
    pitfall: "配准差一点，三维结果可能出现鬼影或偏移，但网络仍可能拟合出看似合理的图。",
    keywords: "view registration; multi-camera calibration; camera pose; tomographic reconstruction"
  },
  {
    zh: "实验日志",
    en: "Experiment log",
    category: "实验/数据",
    aliases: ["experiment log", "reproducibility", "configuration", "result tracking"],
    explain: "记录每次数据、代码版本、参数、结果图和失败原因的结构化日志。",
    use: "你后续想做学习日志网页，这个概念要提前嵌入：每周能复现自己的结果。",
    pitfall: "没有日志，参数扫描越多越乱，最后很难写论文方法和结果。",
    keywords: "experiment log; reproducibility; configuration; result tracking; research diary"
  },
  {
    zh: "科研智能体",
    en: "Agent for Science / scientific agent",
    category: "神经场/AI",
    aliases: ["Agent for Science", "scientific agent", "AI scientist", "science agent", "language agent"],
    explain: "把大语言模型、工具调用、代码执行、实验规划和报告生成连接成科研辅助流程。",
    use: "蔡老师主页列出 Agent for Science；对你的毕设最稳的落点是 BOST/NeRIF 参数扫描、数据质量检查和自动报告，而不是让 agent 直接替代物理反演。",
    pitfall: "agent 生成的结论必须回到真实数据、重投影误差、baseline 和师兄确认；不能把自动报告当作实验事实。",
    keywords: "Agent for Science; scientific agent; AI scientist; language agent; autonomous research"
  },
  {
    zh: "自驱动实验室",
    en: "Self-driving laboratory",
    category: "实验/数据",
    aliases: ["self-driving lab", "self-driving laboratory", "autonomous laboratory", "autonomous experimentation"],
    explain: "用自动化平台、机器学习和闭环决策来选择下一组实验条件，加速探索实验空间。",
    use: "可以借它的语言设计 BOST 参数扫描：视角数量、噪声、背景图案、loss 权重和数据质量阈值由脚本自动排布。",
    pitfall: "自驱动实验室不等于无监督乱跑实验；每个闭环目标、约束和安全边界都要明确。",
    keywords: "self-driving lab; autonomous experimentation; closed-loop experiment; scientific automation"
  },
  {
    zh: "闭环实验",
    en: "Closed-loop experimentation",
    category: "实验/数据",
    aliases: ["closed-loop", "closed-loop experimentation", "active learning", "Bayesian optimization"],
    explain: "根据上一轮实验结果自动决定下一轮参数，使有限实验预算尽量探索最有价值区域。",
    use: "本科可做成参数扫描助手：先跑粗网格，再按误差/不确定度挑最值得细扫的视角数、噪声水平或网络容量。",
    pitfall: "如果目标函数错了，闭环会很高效地优化错误目标；BOST 中必须同时看重投影误差、真值误差和物理可解释性。",
    keywords: "closed-loop experimentation; active learning; Bayesian optimization; experiment design; parameter search"
  },
  {
    zh: "工具调用智能体",
    en: "Tool-using agent",
    category: "神经场/AI",
    aliases: ["tool-using agent", "tool use", "code execution agent", "notebook agent"],
    explain: "让语言模型调用搜索、代码、绘图、仿真、文件读写等工具完成多步任务。",
    use: "适合你的本地网页后续升级：让 agent 跑 M0-M3 demo、汇总指标、生成图表和更新学习日志。",
    pitfall: "工具调用能力越强，越要限制写文件、上传、删除和使用私有 PDF 的边界；公开网页只放可公开材料。",
    keywords: "tool use; code execution; notebook agent; automated report; scientific workflow"
  },
  {
    zh: "算子学习",
    en: "Operator learning",
    category: "神经算子",
    aliases: ["operator learning", "learned operator", "solution operator", "inverse operator"],
    explain: "学习一类输入函数到输出函数的共同映射，而不是只拟合一个固定样本。对 BOST，可把多视角位移/投影函数映射到三维折射率函数。",
    use: "T16 把它落成跨 phantom、视角、噪声和分辨率共享的 BOST inverse operator；新观测原则上只需一次前向推理。",
    pitfall: "算子学习不等于普通 PDE 数值离散，也不等于给每个新样本重新训练一个 NeRIF；是否真的跨网格和跨工况泛化必须实验验证。",
    keywords: "operator learning; function space; inverse operator; solution operator; BOST reconstruction"
  },
  {
    zh: "神经算子",
    en: "Neural operator",
    category: "神经算子",
    aliases: ["neural operator", "neural operators", "function-to-function network", "mesh invariant network"],
    explain: "用共享神经网络参数近似函数空间之间的映射，常见实现包括 DeepONet、FNO、图神经算子和低秩神经算子。",
    use: "在 T16 中，它不是替代 BOST 光学物理，而是学习传统反投影/SIRT 粗解到更准确三维折射率场的跨样本残差。",
    pitfall: "论文中的 discretization invariance 是架构/极限性质，不代表有限数据和有限频谱模式下自动实现 zero-shot super-resolution。",
    keywords: "neural operator; function-to-function; discretization invariance; FNO; DeepONet"
  },
  {
    zh: "深度算子网络",
    en: "Deep Operator Network, DeepONet",
    category: "神经算子",
    aliases: ["DeepONet", "deep operator network", "branch net", "trunk net"],
    explain: "用 branch net 编码输入函数在传感器处的采样，用 trunk net 编码输出查询坐标，再组合得到目标函数在该坐标的值。",
    use: "可把固定顺序的多视角 BOST 观测送入 branch，把三维坐标送入 trunk，连续查询折射率 n(x,y,z)。",
    pitfall: "标准 DeepONet 常假设固定传感器数量和顺序；BOST 的缺失视角、可变几何和大尺寸位移场需要 mask、集合编码或先做物理提升。",
    keywords: "DeepONet; branch net; trunk net; sensor encoding; coordinate query"
  },
  {
    zh: "傅里叶神经算子",
    en: "Fourier Neural Operator, FNO",
    category: "神经算子",
    aliases: ["FNO", "Fourier neural operator", "spectral convolution", "Fourier layer"],
    explain: "在频域参数化全局卷积核，只保留有限 Fourier modes，在规则网格上高效学习函数到函数的映射。",
    use: "T16 推荐先把多视角观测通过伴随反投影或固定预算 SIRT 提升成规则三维体，再用 3D FNO 预测残差。",
    pitfall: "3D FFT 很吃显存；FNO 也可能丢失尖锐局部结构。必须从 24^3/32^3 起步，与 3D U-Net、公平插值和传统重建对照。",
    keywords: "FNO; Fourier neural operator; spectral convolution; Fourier modes; resolution transfer"
  },
  {
    zh: "物理约束神经算子",
    en: "Physics-Informed Neural Operator, PINO",
    category: "神经算子",
    aliases: ["PINO", "physics-informed neural operator", "physics constrained operator", "reprojection loss"],
    explain: "在学习一族输入输出映射时，同时加入可微物理方程或观测模型残差，使预测不仅贴标签，也满足已知物理关系。",
    use: "对 T16，第一阶段最可信的物理约束是把预测折射率场通过 BOST forward operator 重投影回训练/留出视角，并加入边界与梯度约束。",
    pitfall: "没有速度、压力、边界条件和热力学闭合时，不要为了写 PINO 强行加入 Navier-Stokes；BOST forward consistency 已经是更直接的物理验收。",
    keywords: "PINO; physics-informed neural operator; physics loss; BOST forward model; reprojection"
  },
  {
    zh: "物理提升",
    en: "Physics lift / adjoint lift",
    category: "神经算子",
    aliases: ["physics lift", "adjoint lift", "backprojection lift", "SIRT lift", "coarse reconstruction"],
    explain: "先用已知观测几何的伴随、反投影或固定预算迭代，把不规则多视角观测变成规则三维粗解，再送入场到场网络。",
    use: "T16 的 channel 0 是训练集统一标定后的 physics lift；FNO/U-Net 只学习它与真值之间的残差。",
    pitfall: "如果每个样本都用自己的真值重新缩放 lift，就发生 GT leakage；标定必须只在训练集拟合一次并锁定。",
    keywords: "physics lift; adjoint; backprojection; SIRT; BOST geometry; train-only calibration"
  },
  {
    zh: "残差算子学习",
    en: "Residual operator learning",
    category: "神经算子",
    aliases: ["residual FNO", "operator residual", "physics residual correction", "delta field"],
    explain: "不让网络从零生成完整物理场，而是预测传统粗解还缺少的修正量，再与粗解相加。",
    use: "T16 写成 n_hat = z0 + FNO(z0, geometry metadata)，使网络重点修正 sparse-view streak、噪声和系统性偏差。",
    pitfall: "残差更容易训练不代表一定更物理；如果 z0 带 geometry bias，网络可能只在训练分布里学会掩盖它。必须和 direct prediction 做消融。",
    keywords: "residual operator; residual FNO; physics lift; delta field; ablation"
  },
  {
    zh: "留出视角重投影",
    en: "Held-out-view reprojection",
    category: "反问题/数值",
    aliases: ["held-out reprojection", "test-view consistency", "unseen camera", "withheld view"],
    explain: "重建时故意不使用一部分相机视角，再把预测三维场投影到这些未见视角，与真实观测比较。",
    use: "真实 BOST 没有三维真值时，它比 train-view residual 更接近可部署的模型选择信号。",
    pitfall: "低 held-out error 仍不能保证三维场唯一正确；forward-model bias、相邻视角相关性和噪声都可能让错误场通过该检查。",
    keywords: "held-out view; reprojection; data consistency; test camera; no ground truth"
  },
  {
    zh: "工况域外测试",
    en: "Condition-level out-of-distribution test",
    category: "实验/数据",
    aliases: ["condition OOD", "view OOD", "noise OOD", "family OOD", "cell holdout"],
    explain: "把完整的视角数、噪声、几何或 phantom family 工况留在训练集之外，而不是只随机拆不同 seed。",
    use: "T16 分开报告 IID、3-view、noise 0.10、joint 和 thin-front family OOD，避免总体均值掩盖失败域。",
    pitfall: "同一时序相邻帧或同一生成器邻近参数若跨 train/test，会让 OOD 名义成立、实际仍是插值。",
    keywords: "OOD; condition cell; view count; noise; phantom family; leakage"
  },
  {
    zh: "逆算子与演化算子",
    en: "Inverse operator vs evolution operator",
    category: "神经算子",
    aliases: ["inverse operator", "evolution operator", "measurement-to-field", "history-to-future"],
    explain: "逆算子从观测恢复隐藏物理场；演化算子从当前/历史物理状态预测未来轨迹。两者的数据、物理约束和失败模式不同。",
    use: "T16 当前实现是 BOST measurement/lift 到静态 3D field 的 inverse operator；若师兄要 4D evolution，需改成历史投影/三维场到未来场。",
    pitfall: "不能因为网络都叫 FNO 就把重建和时间预测混成一个任务；train/test split、baseline 和指标必须重新定义。",
    keywords: "inverse operator; evolution operator; reconstruction; forecasting; 4D BOST"
  },
  {
    zh: "摊销推理",
    en: "Amortized inference",
    category: "神经算子",
    aliases: ["amortized inverse", "shared inference", "one-shot inference", "operator initialization"],
    explain: "用许多训练实例预先学习共享映射，把新实例原本需要反复优化的计算摊到训练阶段，新观测只需一次或少量前向/精化步骤。",
    use: "T16 可一次输出 NeRIF 初值，再用 BOST forward loss 做短时 refinement，比较总耗时、最终误差和失败率。",
    pitfall: "网络前向快不等于总成本低；必须计入 paired data 生成、训练、域偏移和必要的 test-time refinement。",
    keywords: "amortized inference; per-instance optimization; NeRIF initialization; runtime"
  },
  {
    zh: "傅里叶模态与时间秩",
    en: "Fourier modes vs temporal rank",
    category: "神经算子",
    aliases: ["n_modes", "Fourier modes", "temporal rank", "SVD rank", "tensor rank"],
    explain: "FNO 的 Fourier modes 是每层保留的空间频率数量；M3B 的 temporal SVD rank 是时序低秩近似维数；TDBOST 的张量/平面分解秩又是另一种参数。",
    use: "T16 的 `n_modes=(4,6,6)` 控制 3D spectral convolution，不能由 M3B 的 rank 3 结果直接推出。",
    pitfall: "把几个都叫 rank/mode 的超参数混用，会产生看似有论文依据、实际上无量纲对应的错误调参。",
    keywords: "FNO n_modes; Fourier frequency; temporal SVD rank; TDBOST tensor rank; hyperparameter"
  },
  {
    zh: "可靠度门控",
    en: "Reliability gate / confidence-gated skip",
    category: "神经算子",
    aliases: ["reliability gate", "confidence gate", "gated residual", "adaptive skip", "view-aware gate"],
    explain: "让模型根据观测覆盖、几何条件或数据一致性残差，决定粗重建应以多大权重直接进入最终输出。",
    use: "T16 三种子消融发现 Residual FNO 在 IID/noise 更好，而 Absolute FNO 在 3-view/joint 更好，因此可学习 alpha 并写成 n_hat = alpha*z0 + FNO(x)。",
    pitfall: "gate 若直接用真值误差或 test label 训练会泄漏；真实部署只能使用 view geometry、观测 residual、标定质量等可观测量。",
    keywords: "reliability gate; adaptive residual; view coverage; geometry confidence; BOST operator"
  },
  {
    zh: "残差输出与绝对输出",
    en: "Residual output vs absolute output",
    category: "神经算子",
    aliases: ["residual output", "absolute output", "direct field head", "skip connection ablation"],
    explain: "残差输出把网络修正量加到 physics lift；绝对输出由网络直接给出完整场。两者可以使用相同输入，但优化目标和对粗解偏差的继承方式不同。",
    use: "T16 的 absolute-output FNO 仍接收 calibrated lift，只拿掉输出端 skip；它用于检验 residual parameterization，不等同 raw projection-to-volume。",
    pitfall: "只因拿掉 skip 就把模型称为 direct measurement operator 会夸大实验；是否使用了物理提升输入必须单独说明。",
    keywords: "residual output; absolute output; skip ablation; physics lift; direct operator"
  },
  {
    zh: "优化种子区间",
    en: "Optimization-seed interval",
    category: "实验/数据",
    aliases: ["training seed", "optimization seed", "seed variance", "Student-t seed interval"],
    explain: "固定数据集，只改变网络初始化和训练随机顺序，观察模型结果对优化随机性的敏感程度。",
    use: "T16 第二 checkpoint 使用三个训练种子和 Student-t 95% 区间，检查 Residual/Absolute、reprojection loss 与 matched U-Net 排名是否稳定。",
    pitfall: "它不是对真实流场总体的置信区间，也不是数据 bootstrap；三个种子的统计功效很低，只适合稳定性筛查。",
    keywords: "optimization seed; training variance; Student-t interval; reproducibility; statistical power"
  },
  {
    zh: "支持/查询视角拆分",
    en: "Support/query view split",
    category: "神经算子",
    aliases: ["support view", "query view", "camera split", "view splitting", "withheld camera training"],
    explain: "把一组相机拆成输入模型的 support views 和只用于计算物理一致性 loss 的 query views，让网络在没有三维真值时仍能从未输入相机获得监督。",
    use: "T16 已用固定 support/query 变体训练双输出模型；v2 中 query 只监督 support-fit 无法观测的受限修正或零空间残差。",
    pitfall: "query view 必须完全不进入网络输入；相邻相机高度相关时，低 query error 仍可能高估真正的跨几何泛化。",
    keywords: "support query split; held-out camera; self-supervised tomography; view splitting; BOST"
  },
  {
    zh: "专家路由",
    en: "Expert routing / mixture-of-experts gate",
    category: "神经算子",
    aliases: ["expert routing", "mixture of experts", "MoE gate", "router", "expert selection"],
    explain: "让一个 router 根据输入条件给多个专门模型分配权重或选择其中一部分，而不是强迫一个模型覆盖所有工况。",
    use: "T16 把 trust-and-correct Residual 与 rebuild Absolute 看作两个专家；v1 的神经 router 已塌缩，v2 先用 support-view 闭式权重锚定。",
    pitfall: "router 容易塌缩到单一专家或学习数据集标签捷径；必须报告 expert usage、matched 参数和跨工况行为。",
    keywords: "mixture of experts; router collapse; residual expert; absolute expert; condition routing"
  },
  {
    zh: "事后神谕遗憾",
    en: "Oracle regret",
    category: "实验/数据",
    aliases: ["oracle regret", "per-cell oracle", "retrospective oracle", "best-expert regret"],
    explain: "先用真值事后挑出每个 seed/工况中最好的专家，再计算可部署模型相对这个理想选择多出的误差。",
    use: "T16 用 Residual/Absolute 两者较小误差作为 oracle，检查 reliability model 是否真正接近按工况选最佳分支。",
    pitfall: "oracle 使用真值，只能做 synthetic 回顾审计，不能在真实 BOST 部署时替模型选分支。",
    keywords: "oracle regret; best expert; retrospective audit; model selection; ground truth"
  },
  {
    zh: "几何感知特征反投影",
    en: "Geometry-aware feature backprojection",
    category: "反问题/数值",
    aliases: ["feature backprojection", "geometry-aware backprojection", "ray feature lifting", "2D-to-3D feature lift"],
    explain: "先用 2D 网络提取各视角特征，再依据相机投影几何把它们回投到三维位置并融合，而不是按通道简单堆图像。",
    use: "它是可变视角 BOST operator 的推荐输入层：网络接口不随相机数量变化，同时显式保留 ray coverage。",
    pitfall: "错误标定会把 2D 特征系统性投到错误位置；必须把 geometry perturbation 与 coverage map 纳入测试。",
    keywords: "geometry aware; feature backprojection; projection matrix; ray coverage; multi-view reconstruction"
  },
  {
    zh: "分支分歧",
    en: "Branch disagreement",
    category: "神经算子",
    aliases: ["branch disagreement", "expert disagreement", "ensemble disagreement", "prediction spread"],
    explain: "两个独立输出分支对同一三维场预测差异的大小，可作为模型不确定或工况偏移的候选信号。",
    use: "SC-DBNO 可读取 Residual/Absolute 在体场、梯度或重投影上的差异，但 v1 审计显示单独 branch disagreement 与 field 胜负相关性弱。",
    pitfall: "两个分支可能一起错且高度相关；低 disagreement 不等于低 error，必须验证它与真实误差的相关性和校准。",
    keywords: "branch disagreement; ensemble uncertainty; expert disagreement; failure detection"
  },
  {
    zh: "风险-覆盖率曲线",
    en: "Risk-coverage curve / selective prediction",
    category: "实验/数据",
    aliases: ["risk coverage", "selective prediction", "abstention", "coverage", "error rejection"],
    explain: "按模型不确定性从高到低拒绝一部分样本，观察保留样本比例下降时平均误差是否同步下降。",
    use: "它检验 T16 的 reliability score 是否真的能识别失败，而不只是给每个结果附一个没有含义的 confidence。",
    pitfall: "只报 AUROC 或平均不确定性不足以证明可用；还要预先定义拒答成本、覆盖率和真实部署阈值。",
    keywords: "risk coverage; selective prediction; abstention; failure detection; uncertainty calibration"
  },
  {
    zh: "视角丢弃训练",
    en: "View dropout / camera dropout",
    category: "实验/数据",
    aliases: ["view dropout", "camera dropout", "view augmentation", "random view masking"],
    explain: "训练时随机移除部分相机或改变视角组合，让模型学习在不同观测覆盖下工作。",
    use: "T16 dual v1 已在训练中加入 3-view/all-but-one support 变体，因此旧 `test_view_ood` 名称在该轮不再代表严格视角数 OOD。",
    pitfall: "随机少几张并不等于 realistic geometry OOD；还要控制最大角度缺口、相邻视角簇和系统性遮挡。",
    keywords: "view dropout; camera masking; sparse-view training; geometry augmentation"
  },
  {
    zh: "不确定性校准",
    en: "Uncertainty calibration",
    category: "神经算子",
    aliases: ["uncertainty calibration", "coverage calibration", "predictive interval", "calibrated confidence"],
    explain: "检查模型给出的置信度或预测区间是否与实际错误频率相符，例如标称 95% 区间是否约覆盖 95% 真值。",
    use: "T16 可先用 deep ensemble/branch disagreement 做低成本基线，再决定是否需要 LUNO、Bayesian NO 或 probabilistic operator。",
    pitfall: "IID 上校准良好不代表 OOD 上可靠；必须分别报告 view/noise/geometry/family 域，且区分 epistemic 与 measurement uncertainty。",
    keywords: "uncertainty calibration; predictive interval; coverage; Bayesian neural operator; OOD"
  },
  {
    zh: "support-fit 闭式融合",
    en: "Closed-form support-consistency mixture",
    category: "反问题/数值",
    aliases: ["support-fit", "support consistency weight", "closed-form mixture", "projection-fit routing"],
    explain: "在两个候选体场的连线上，直接求使已观测 support views 重投影误差最小的标量权重；线性 forward operator 下有一维最小二乘闭式解。",
    use: "T16 dual v1 中，support-fit 不读取 field truth 或额外相机，将五域 field oracle regret 降到 0.014%，成为新模型必须击败的强基线。",
    pitfall: "它只在两专家的线段上优化已观测一致性，无法恢复 support operator 零空间中的信息，也可能拟合观测噪声。",
    keywords: "support fit; closed-form routing; least squares; projection consistency; dual branch"
  },
  {
    zh: "路由可识别度",
    en: "Routing identifiability",
    category: "反问题/数值",
    aliases: ["routing identifiability", "expert distinguishability", "mixture curvature", "projection separation"],
    explain: "衡量当前观测是否能区分两个专家。在线性 support operator 中，`||A_S(x_res-x_abs)||^2` 就是融合权重目标的曲率/信息量候选。",
    use: "SC-DBNO v2 计划用它加权 query loss、校准不确定性，并在两专家投影几乎相同时拒绝输出假精确路由。",
    pitfall: "体场差异大不等于观测可区分；必须通过 forward operator 计算，并检查噪声尺度。",
    keywords: "identifiability; expert distinguishability; forward operator; uncertainty; abstention"
  },
  {
    zh: "support 零空间残差",
    en: "Support-nullspace correction",
    category: "反问题/数值",
    aliases: ["nullspace correction", "measurement nullspace", "data-consistent residual", "invisible component"],
    explain: "已观测 support views 几乎看不到的体场分量，即满足 `A_S(delta_x) ≈ 0` 的修正。它是病态少视角反演中必须依赖先验或额外观测恢复的部分。",
    use: "SC-DBNO v2 让 query cameras 主要教这个不可观测残差，同时用 support reprojection 约束不破坏已观测一致性。",
    pitfall: "网络可以在零空间中生成观测无法否定的幻觉；必须用新视角、物理边界、流态先验或真值审计。",
    keywords: "nullspace; data consistency; sparse view; invisible component; query camera"
  },
  {
    zh: "路由器塌缩",
    en: "Router collapse",
    category: "神经算子",
    aliases: ["router collapse", "gate collapse", "uniform routing", "expert collapse"],
    explain: "路由网络几乎总是输出同一权重，或总选同一专家，没有利用样本/工况差异。",
    use: "T16 dual v1 的 query router 在五域平均 Residual 权重均约 0.50，field 端点选择准确率 40.5%，是已保留的负结果。",
    pitfall: "等权可能本身是强 ensemble，因此平均误差不差不代表路由成功；必须报权重分布、选择相关、expert usage 和 oracle regret。",
    keywords: "router collapse; uniform gate; expert usage; mixture of experts; negative result"
  },
  {
    zh: "整类留出验证",
    en: "Leave-one-family-out validation (LOFO)",
    category: "实验/数据",
    aliases: ["LOFO", "leave one family out", "group holdout", "morphology holdout"],
    explain: "每次把一个完整数据族从训练中移除，只在其余族训练，再到被移除族测试；比随机拆样本更能检查模型是否依赖族标签或近重复结构。",
    use: "M3B family selector 外层整类留出一种 phantom morphology，内层只在其余 family 之间选择 ridge 正则强度。",
    pitfall: "synthetic morphology LOFO 仍不等于真实流场、相机几何或实验装置泛化；下一步还要 leave-one-geometry-out 和真实数据。",
    keywords: "LOFO; group holdout; morphology transfer; nested validation; data leakage"
  },
  {
    zh: "可观测容量选择",
    en: "Observable capacity selection",
    category: "神经算子",
    aliases: ["capacity selector", "rank selector", "adaptive rank", "model capacity control"],
    explain: "不使用未知三维真值，而依据投影一致性、奇异谱、时序导数和积分量等可测信号，为每个工况选择 rank、factor 数或网络容量。",
    use: "M3B 在 rank 1/2/3/5/8/12/full 候选中选择 temporal capacity；未来可映射到 TDBOST plane factors 或 temporal-operator width。",
    pitfall: "训练 selector 可以用 synthetic truth，但测试 fold 绝不能读取 field error、oracle rank、family 或真实 noise label。",
    keywords: "adaptive rank; capacity control; observable selector; temporal operator; BOST"
  },
  {
    zh: "满秩拒绝平滑",
    en: "Full-rank abstention",
    category: "反问题/数值",
    aliases: ["full rank fallback", "no smoothing option", "capacity abstention", "identity fallback"],
    explain: "把“不施加低秩压缩”作为合法候选；当观测较干净或动态结构复杂时，selector 可以拒绝过度平滑。",
    use: "M3B 新矩阵的 noise-free vortex/chirp 条件会选择 rank 18，避免固定 rank 3 丢失结构。",
    pitfall: "full rank 不等于真实正确，它仍继承 framewise reconstruction 的 forward-model 偏差和噪声；只是拒绝额外时序压缩。",
    keywords: "full rank; abstention; smoothing bias; fallback; temporal reconstruction"
  },
  {
    zh: "形态迁移",
    en: "Morphology transfer",
    category: "实验/数据",
    aliases: ["phantom family transfer", "flow morphology shift", "shape generalization", "structural OOD"],
    explain: "训练和测试的三维结构类型不同，例如 blobs、ring/front、expanding shell 与 jet filaments，用来检查模型是否只记住一种空间形态。",
    use: "M3B 发现 blobs-sheet 主要需要 rank 2-5，而 vortex-ring 常需要 rank 8-full，说明 morphology 是容量选择的一阶变量。",
    pitfall: "人工 phantom 之间的差异不代表真实火焰、激波或喷流的统计分布；只能作为结构压力测试。",
    keywords: "morphology transfer; phantom family; structural OOD; flow regime; generalization"
  },
  {
    zh: "验证相机幅度标定",
    en: "Query-camera amplitude calibration",
    category: "反问题/数值",
    aliases: ["query camera calibration", "query line search", "reserved view", "validation camera"],
    explain: "先用 support cameras 重建并生成零空间修正方向，再用 query measurement 闭式求修正幅度。方向保持 support consistency，query 只决定修正用多少。",
    use: "T16 v2d 已固定 Q_fit=80°、Q_audit=60°；当前 learned correction 在 K=4/6/8 均明显落后把同一个 Q 直接加入重建。",
    pitfall: "support consistency 和正 alpha 都不代表 field 正确。还要与 S∪Q direct 比较，并把 Q_audit 在选角前锁定。",
    keywords: "query camera; line search; nullspace correction; validation view; BOST"
  },
  {
    zh: "观测价值",
    en: "Value of information (VOI)",
    category: "实验/数据",
    aliases: ["value of information", "sensor utility", "camera value", "measurement value"],
    explain: "衡量增加一台相机、一个角度或一类测量后，重建误差、失败率或不确定度实际改善多少，并与硬件、同步和数据带宽成本比较。",
    use: "v2d 提示新增视角直接进入 inverse operator 的价值通常高于标定当前 null direction；K=6 max-gap 只形成 camera-layout/VOI 新假设。",
    pitfall: "在 synthetic canonical views 上选出的最佳角度不一定适合真实装置；相机位置可达性和遮挡也属于成本。",
    keywords: "value of information; sensor placement; camera selection; experimental design; Pareto"
  },
  {
    zh: "整几何留出验证",
    en: "Leave-one-geometry-out validation (LOGO)",
    category: "实验/数据",
    aliases: ["LOGO", "leave one geometry out", "geometry holdout", "camera-layout holdout"],
    explain: "把一种完整相机布局从训练和调参中移除，只在其余布局上建模，再到未见布局测试，用于审计跨几何迁移和数据泄漏。",
    use: "M3B geometry audit 对 uniform、rotated、limited-arc、dual-cluster、jittered 和 calibration-offset 六类布局做 nested LOGO。",
    pitfall: "synthetic angle layout 仍不等于真实 ray calibration、镜头畸变或曲光线；LOGO 通过不代表硬件迁移通过。",
    keywords: "leave one geometry out; LOGO; camera layout; geometry transfer; nested validation"
  },
  {
    zh: "函数型保形校准",
    en: "Functional conformal calibration",
    category: "神经算子",
    aliases: ["conformal prediction", "functional coverage", "risk-controlling quantile operator", "prediction band"],
    explain: "用独立 calibration set 对整条函数或三维场的预测区间做有限样本覆盖校准，而不只校准一个像素或一个标量。",
    use: "当 T16 有独立高分辨率数据后，可把 ensemble/UQ 输出校准为可检验的 field coverage 或 selective risk。",
    pitfall: "校准保证依赖交换性与校准分布；跨 family、geometry 或真实装置时需要重新审计，不能沿用 IID 阈值。",
    keywords: "functional conformal prediction; calibrated uncertainty; coverage; neural operator; risk control"
  },
  {
    zh: "神经修正算子",
    en: "Neural correction operator",
    category: "神经算子",
    aliases: ["correction operator", "reconstruction plus correction", "learned refinement operator", "physics lift correction"],
    explain: "先用传统或短迭代物理求解器得到粗重建，再由跨样本学习的算子修正，而不是从观测直接黑箱映射到最终场。",
    use: "它为 T16 的 physics lift、support-fit 和 learned correction 提供最近的组件级方法先例；可与 direct inverse operator 做公平对照。",
    pitfall: "EIT、MRI 或 CT 的修正算子结论不能直接迁移 BOST；粗解误差结构、forward operator 和数据一致性都不同。",
    keywords: "neural correction operator; inverse problem; physics reconstruction; learned correction; BOST"
  },
  {
    zh: "伴随算子",
    en: "Adjoint operator",
    category: "反问题/数值",
    aliases: ["adjoint", "transpose operator", "backprojection", "A transpose"],
    explain: "若 forward operator A 把体场映到观测，伴随 A* 把观测残差按内积意义反传回体场。离散实矩阵下常对应转置 A^T，但它通常不是逆算子。",
    use: "BOST 中可用伴随/反投影形成 physics lift、梯度方向或传统迭代重建的更新。",
    pitfall: "A* y 只是把信息摊回去，不等于恢复真实 x；少视角、缩放和几何权重都会影响结果。",
    keywords: "adjoint operator; transpose; backprojection; gradient; BOST forward"
  },
  {
    zh: "奇异值分解",
    en: "Singular Value Decomposition, SVD",
    category: "数学基础",
    aliases: ["SVD", "singular values", "left singular vectors", "right singular vectors"],
    explain: "把矩阵写成 A = U Sigma V^T。奇异值说明不同方向被观测放大或压低多少，V 中对应零奇异值的方向构成零空间。",
    use: "用小型 BOST forward matrix 的奇异谱判断可观测 rank、噪声放大和 exact null headroom。",
    pitfall: "小 synthetic 矩阵的 SVD 结论不自动代表真实 cone-ray 系统；大规模系统也通常不能直接做完整 SVD。",
    keywords: "SVD; singular value; rank; nullspace; inverse problem"
  },
  {
    zh: "条件数",
    en: "Condition number",
    category: "数学基础",
    aliases: ["condition number", "kappa", "ill conditioning", "sensitivity"],
    explain: "常用最大非零奇异值与最小非零奇异值之比衡量反演对观测扰动的敏感程度。比值越大，微小噪声越可能被放大。",
    use: "比较不同相机布局、视角数和标定扰动下的线性化 BOST forward 健康度。",
    pitfall: "条件数只描述局部线性与所选范数；正则化、非线性 ray bending 和先验会改变实际误差。",
    keywords: "condition number; ill-conditioned; noise amplification; singular spectrum; camera geometry"
  },
  {
    zh: "穆尔-彭若斯伪逆",
    en: "Moore-Penrose pseudoinverse",
    category: "数学基础",
    aliases: ["pseudoinverse", "Moore Penrose", "pinv", "A plus"],
    explain: "当 A 不可逆或不是方阵时，A+ 给出最小二乘意义下的最小范数解。它能拟合可观测部分，但不会凭空恢复零空间信息。",
    use: "建立小矩阵 inverse baseline、定义 range/null projector，并核对 closed-form query update。",
    pitfall: "直接对很小奇异值取倒数会放大噪声；真实反问题通常还需要截断或 Tikhonov 正则化。",
    keywords: "pseudoinverse; pinv; least squares; minimum norm; nullspace projector"
  },
  {
    zh: "拉东变换",
    en: "Radon transform",
    category: "反问题/层析",
    aliases: ["Radon", "projection transform", "sinogram", "line integral"],
    explain: "把二维函数沿不同角度的直线积分，形成角度-探测器坐标上的投影数据，是理解 CT 和简化层析 forward 的经典模型。",
    use: "先用 scikit-image phantom/sinogram toy 学视角不足、噪声和重建误差，再迁移到 BOST 的偏折/梯度观测。",
    pitfall: "BOST 观测不是普通衰减 CT 的标量线积分；折射率梯度、光路几何和可能的 ray bending 都使 forward 不同。",
    keywords: "Radon transform; sinogram; line integral; tomography; sparse view"
  },
  {
    zh: "滤波反投影",
    en: "Filtered backprojection, FBP",
    category: "反问题/层析",
    aliases: ["FBP", "filtered back projection", "ramp filter", "analytic reconstruction"],
    explain: "先在投影域做频率滤波，再把各角度投影反投影回空间的解析层析方法。它快、透明，适合作为第一条传统 baseline。",
    use: "在 2D Radon toy 中比较 FBP 与 SART，观察少视角条纹、噪声放大和正则化需求。",
    pitfall: "FBP 的标准公式依赖特定直线积分几何；不能直接把 CT 实现当成真实 BOST 求解器。",
    keywords: "FBP; filtered backprojection; ramp filter; CT reconstruction; baseline"
  },
  {
    zh: "相对 L2 误差",
    en: "Relative L2 error",
    category: "实验/数据",
    aliases: ["relative L2", "normalized error", "field error", "L2 norm"],
    explain: "常写为 ||x_hat-x||_2 / ||x||_2，用真值能量归一化总体场误差，便于跨样本比较。",
    use: "作为 synthetic BOST 体场主指标之一，并同时报告逐样本 paired delta、p10/CVaR、harm rate、梯度和积分量。",
    pitfall: "单一平均 L2 会掩盖薄前沿、局部峰值、符号错误和少数严重失败；真实数据无 field truth 时也无法直接计算。",
    keywords: "relative L2; field error; normalized RMSE; paired delta; harm rate"
  },
  {
    zh: "自动微分",
    en: "Automatic differentiation, autograd",
    category: "PyTorch/实现",
    aliases: ["autograd", "automatic differentiation", "computational graph", "backpropagation"],
    explain: "框架记录张量运算并按链式法则计算导数，使 forward residual 对网络参数和坐标的梯度可自动获得。",
    use: "训练 FNO/DeepONet，或在 NeRIF 中计算坐标场导数、ray loss 与正则项。",
    pitfall: "autograd 给出的是代码所定义计算图的导数，不保证物理公式、单位、边界或张量维度正确；仍需有限差分检查。",
    keywords: "autograd; computational graph; backward; gradient check; PyTorch"
  },
  {
    zh: "张量维度约定",
    en: "Tensor layout and shape contract",
    category: "PyTorch/实现",
    aliases: ["tensor shape", "B C D H W", "batch channel depth height width", "shape contract"],
    explain: "为每个数组明确 batch、view/channel、depth、height、width 和 coordinate 维度。3D PyTorch 常用 [B,C,D,H,W]，但观测 view 不应无说明地当 channel。",
    use: "在数据加载、physics lift、3D operator、forward projection 和 loss 之间打印并断言 shape、dtype、device、单位。",
    pitfall: "维度广播可能让代码正常运行却算错物理量；view、time、channel 和 coordinate 是最常混淆的四类轴。",
    keywords: "tensor shape; BCDHW; view channel; broadcasting; data contract"
  },
  {
    zh: "逆犯罪",
    en: "Inverse crime",
    category: "实验/数据",
    aliases: ["inverse crime", "same forward model", "matched simulator", "synthetic leakage"],
    explain: "用同一个离散 forward 生成训练/测试观测，又用它做重建与验收，会让模型在过于匹配的世界里显得异常准确。",
    use: "T16 必须用不同分辨率、ray tracer、扰动或真实装置做 independent-forward audit。",
    pitfall: "随机切分 phantom 并不能消除 inverse crime；forward 离散、几何、噪声与标定也要改变。",
    keywords: "inverse crime; independent forward; simulator mismatch; synthetic benchmark; generalization"
  },
  {
    zh: "拟合视角与审计视角",
    en: "Q_fit / Q_audit split",
    category: "实验/数据",
    aliases: ["Q fit", "Q audit", "query split", "audit camera", "held-out validation view"],
    explain: "Q_fit 可用于选择角度、求 alpha、调阈值或训练；Q_audit 必须完全不参与开发，只在最后验收重投影或失败率。",
    use: "验证 query-calibrated correction 是否对真正未见相机仍有收益，并阻止把同一 query 同时用于调参与宣称泛化。",
    pitfall: "只把 Q 从初始 reconstruction 中拿走还不够；只要它参与模型选择、角度搜索或幅度拟合，就不能再作为独立审计。",
    keywords: "Q_fit; Q_audit; query camera; held-out view; data leakage; model selection"
  },
  {
    zh: "重建预算与安装预算",
    en: "Reconstruction budget vs installed-camera budget",
    category: "实验/数据",
    aliases: ["camera budget", "reconstruction views", "installed cameras", "audit instrument"],
    explain: "重建预算 K 只数参与一次推理输出的 S 与 Q_fit；若另有完全独立的 Q_audit 相机，真实采集或安装数量是 K+1。",
    use: "v2d 在同一个 K 内公平比较 S∪Q direct 与 S+Q correction，但额外锁定一台 Q_audit 做科学验收。",
    pitfall: "不能把‘同重建预算’写成‘同硬件总成本’；相机、光窗、同步、标定和带宽都属于实验成本。",
    keywords: "reconstruction budget; installed camera; Q_audit; hardware cost; fair comparison"
  }
];
