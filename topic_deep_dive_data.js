window.OERF_TOPIC_DEEP_DIVES = [
  {
    id: "T1",
    title: "NeRIF/BOST 最小复现与鲁棒性分析",
    shortTitle: "NeRIF/BOST 鲁棒性",
    fit: "core",
    feasibility: 5,
    novelty: 3,
    risk: 2,
    verdict: "最推荐作为本科毕设主线。它能从合成数据启动，又能自然对接何远哲的 NeRIF、PIV-BOST 和 4D BOST 后续问题。",
    researchAbstract: "这条线把 BOST 看成一个带物理 forward model 的稀疏视角反问题：未知量是三维折射率场，观测量是多视角背景图像位移或等价投影。NeRIF 的关键不是简单把网络当黑箱，而是用坐标神经场表示连续折射率，并通过投影一致性、平滑约束和重投影误差来判断结果是否可信。本科版本最应该研究的是：在视角少、噪声高、场结构复杂或网络容量不足时，NeRIF-style 表示相对传统体素/正则化 baseline 的优势边界在哪里。",
    physicsPlain: "热流、火焰或喷流会让空气密度变化，密度变化又会让折射率变化。光线穿过这片区域时会发生微小偏折，背景点阵在相机里就像被挪动了一点。BOST 的工作就是反过来：从很多方向看到的这些小位移，推回三维空间里哪里折射率高、哪里折射率低。",
    scope: "合成 phantom + 简化 forward model + voxel/regularized baseline + coordinate MLP + view/noise/capacity 扫描。",
    difficulty: "中等偏稳：数学和代码量不小，但不依赖真实实验数据即可启动。",
    software: ["Python", "NumPy/SciPy", "scikit-image", "PyTorch", "Matplotlib/Plotly", "可选 ASTRA Toolbox"],
    hardware: ["普通笔记本可做 2D/小 3D", "有 GPU 可加速 MLP 和参数扫描", "不需要先搭实验台"],
    knowledge: ["BOS/BOST 基本光路", "Gladstone-Dale 折射率-密度关系", "Radon/投影反问题", "坐标神经场和 Fourier features", "误差指标与重投影验证"],
    execution: [
      "第 1 周：复现 2D phantom -> deflection/projection -> baseline inverse，确定数据结构。",
      "第 2-3 周：升到 3D stack 或小体素，加入 5/7/9 视角和噪声扫描。",
      "第 4-6 周：实现 coordinate MLP / Fourier features，对比 voxel/Tikhonov/Landweber/SART。",
      "第 7-8 周：补重投影误差、切片 RMSE、SSIM、运行时间和失败案例图。",
      "后续：如果师兄给真实数据，把 synthetic geometry 替换为组内 manifest。"
    ],
    deliverables: ["可运行 demo", "视角数-误差曲线", "噪声鲁棒性热图", "重投影验证图", "传统 baseline 对比表", "给师兄看的 1 页结论"],
    corePapers: [
      ["NeRIF", "He et al., Neural refractive index field, Physics of Fluids 2025", "何远哲主线核心；抽 forward model、network、loss、实验设置。", "https://doi.org/10.1063/5.0250899"],
      ["BOS 综述", "Raffel, Background-oriented schlieren techniques, Experiments in Fluids 2015", "给 BOS 位移、灵敏度、实验误差一个稳固入口。", "https://doi.org/10.1007/s00348-015-1927-5"],
      ["传统 BOST", "Grauer and Steinberg, UBOST, Experiments in Fluids 2020", "强 baseline；提醒你不能只和弱 toy 比。", "https://doi.org/10.1007/s00348-020-2912-1"],
      ["连续表示前史", "Direct BOST using radial basis functions, Optics Express 2022", "说明非体素连续表示在 NeRIF 前已经存在。", "https://doi.org/10.1364/OE.459872"],
      ["方法邻居", "NeDF / NRIP / Neural Fields survey", "用来解释表示变量、编码、mask 和 gradient loss 的设计空间。", "./paper_library/index.html?q=NeDF%20NRIP%20Neural%20Fields"]
    ],
    videoResources: [
      ["MIT OpenCourseWare 流体力学视频入口", "用于补流体力学直觉，不替代 BOST 论文。", "https://www.youtube.com/@mitocw"],
      ["YouTube: background oriented schlieren NASA 检索", "只作为观察 BOS 可视化效果的入口，正式引用仍以论文为准。", "https://www.youtube.com/results?search_query=background+oriented+schlieren+NASA"],
      ["OpenPIV / scientific Python 生态", "帮助理解后续 PIV-BOST 工具链。", "https://openpiv.readthedocs.io/en/latest/"]
    ],
    askSenior: ["组内认可的 baseline 是 UBOST、体素法、SIRT，还是 NeRIF 论文代码？", "真实数据最主要失败模式是视角少、噪声、几何误差、mask，还是重建速度？", "是否能给一小份脱敏 displacement/geometry/mask 样例？"],
    riskControl: ["不要承诺完整复现 NeRIF 论文级实验；先做简化闭环。", "所有结论都要绑定 synthetic 设置，避免夸成真实火焰结论。", "传统 baseline 要认真做，否则师兄会觉得只是调网络。"]
  },
  {
    id: "T2",
    title: "少视角 BOST 的神经隐式重构与不确定度",
    shortTitle: "少视角与不确定度",
    fit: "core",
    feasibility: 5,
    novelty: 4,
    risk: 2,
    verdict: "很适合作为 T1 的研究化增强版：不是只复现，而是回答“什么时候可信”。",
    researchAbstract: "少视角 BOST 的困难来自信息不足：不同三维折射率场可能生成相近的投影位移，噪声还会被反演放大。这一选题把 view count、view orientation、位移噪声、mask 缺损和模型容量作为实验变量，利用重投影误差、ensemble 方差、bootstrap、confidence weighting 或残差空间分布来构造不确定度图谱。它的价值在于为真实 OERF 数据提供判断标准：不是只问重构图好不好看，而是问哪些区域可以相信，哪些区域只是算法补出来的。",
    physicsPlain: "从一个方向看影子，很难知道三维物体的真实形状；多几个方向才稳。BOST 也是这样，视角少时三维场会变成“猜谜”。不确定度分析就是给这张猜出来的图加一个可信度标签。",
    scope: "T1 pipeline + 视角数/角度/噪声/缺失 mask 扫描 + uncertainty 或 confidence map。",
    difficulty: "中等：需要会做批量实验和统计图，但核心物理链条与 T1 共用。",
    software: ["Python", "PyTorch", "SciPy", "scikit-image", "pandas", "Plotly/Seaborn"],
    hardware: ["普通笔记本可跑小规模", "GPU 可做 ensemble 或更多参数扫描"],
    knowledge: ["反问题病态性", "误差传播", "bootstrap/ensemble 基本思想", "重投影残差", "BOS 位移置信度"],
    execution: [
      "固定一个可解释 phantom，先跑 3/5/7/9/13 视角。",
      "加入位移噪声、坏视角、mask 裁剪和 view orientation 变化。",
      "对每组实验重复多次，记录均值、方差、p95 error、重投影误差。",
      "生成 uncertainty map：可以从 ensemble 方差、残差局部强度或 confidence weighting 起步。",
      "写出“可信重构需要的最低视角/噪声条件”建议。"
    ],
    deliverables: ["少视角误差曲线", "不确定度热图", "失败案例库", "可信区域 mask", "实验设计建议表"],
    corePapers: [
      ["NeRIF", "He et al., Physics of Fluids 2025", "主模型和实验场景。", "https://doi.org/10.1063/5.0250899"],
      ["BOS UQ", "Rajendran et al., uncertainty quantification in BOS density estimation", "把 BOS 位移误差传播到密度估计。", "./paper_library/index.html?q=uncertainty%20quantification%20BOS"],
      ["WLS integration", "Uncertainty-based weighted least squares density integration for BOS", "给 confidence weighting 一个传统方法依据。", "./paper_library/index.html?q=weighted%20least%20squares%20density%20integration"],
      ["CTC spatial resolution", "Yu / Liu / Cai spatial resolution for 3D tomography", "把重构质量从视觉效果推进到空间分辨率指标。", "./paper_library/index.html?q=spatial%20resolution%20computed%20tomography%20Cai"],
      ["TV / SART baseline", "Kak & Slaney, SART, TV-min CT, Chambolle-Pock", "少视角反问题的传统语言。", "./classical_tomography_baseline_bridge.html"]
    ],
    videoResources: [
      ["MIT OpenCourseWare 流体力学频道", "补流体背景；重点看涡、扩散、边界条件直觉。", "https://www.youtube.com/@mitocw"],
      ["Steve Brunton 频道", "可看 sparse modeling / DMD / scientific ML，帮助理解不确定度和低维表示。", "https://www.youtube.com/@Eigensteve"],
      ["OCW 搜索：tomography / inverse problems", "作为补反问题课程入口。", "https://ocw.mit.edu/search/?q=tomography"]
    ],
    askSenior: ["真实 BOST 中最常见的是少视角、坏视角还是位移噪声？", "论文更需要 per-pixel uncertainty，还是只需要整体验证指标？", "师兄认可用 synthetic uncertainty 作为本科主结果吗？"],
    riskControl: ["不确定度不是随便画一张方差图，要和实验变量绑定。", "重投影误差低不等于真实场误差低，需要在 synthetic 中说明非唯一性。", "不要承诺 Bayesian deep learning 大工程，先做可解释 ensemble/残差图。"]
  },
  {
    id: "T3",
    title: "面向 PIV-BOST 的折射补偿工具",
    shortTitle: "PIV-BOST 补偿",
    fit: "core",
    feasibility: 3,
    novelty: 4,
    risk: 3,
    verdict: "上限很高、贴师兄很强，但真实数据决定天花板。建议作为第二阶段升级线，或和 T1/T5 绑定。",
    researchAbstract: "PIV 测速度依赖粒子图像位移，但火焰和热流中的折射率梯度会改变光路，使粒子在相机里的位置发生系统偏移。PIV-BOST 的核心思想是同步获取 BOST 折射率场，用它估计光路偏折并补偿 PIV 速度测量。一个本科可执行问题是拆解补偿层级：补偿应发生在 raw particle image、互相关 displacement map、calibration/disparity，还是最终 velocity vector？合成 particle image 可以先验证误差传播，真实 OERF 数据则可作为加分。",
    physicsPlain: "PIV 像是在拍小颗粒的运动，但热空气像一块不均匀的透明玻璃，会把颗粒影像“掰歪”。BOST 给出这块透明玻璃的折射率分布，所以可以反过来修正 PIV 看到的位置。",
    scope: "synthetic particle image pair + refractive distortion + OpenPIV/PIVlab baseline + compensation before/after metrics。",
    difficulty: "中等偏高：PIV 图像处理、BOST 折射补偿和标定坐标会交织在一起。",
    software: ["OpenPIV", "PIVlab", "Python/OpenCV", "NumPy/SciPy", "Matplotlib", "可选 RAFT/光流模型"],
    hardware: ["无真实数据时不需要硬件", "真实实验需要同步 PIV/BOST 图像、标定文件、触发信息", "GPU 仅在深度光流时需要"],
    knowledge: ["PIV 互相关原理", "粒子图像生成", "折射率梯度导致的像素位移", "相机标定和坐标变换", "速度误差指标"],
    execution: [
      "先做 velocity-field toy：真实速度 + 折射偏移误差 -> 观测速度。",
      "升级到 synthetic particle image pair，用 OpenPIV/PIVlab 跑互相关。",
      "插入简化 refractive distortion，比较 raw image dewarping、displacement correction 和 vector correction。",
      "扫描粒子密度、window size、折射强度、噪声和时间间隔。",
      "如有真实数据，再接入 PIV-BOST manifest 和组内标定。"
    ],
    deliverables: ["补偿前后速度误差图", "PIV 参数敏感性表", "raw/displacement/vector 三层接口图", "OpenPIV/PIVlab 可复现实验", "真实数据字段清单"],
    corePapers: [
      ["PIV-BOST", "Zheng / He et al., simultaneous PIV-BOST, Experiments in Fluids 2025", "主线核心；抽同步测量和补偿公式。", "https://doi.org/10.1007/s00348-025-04093-y"],
      ["Stereo PIV-BOST", "Zheng / He et al., stereo-velocity compensation, POCI 2026", "展望 stereo，但本科不要轻易承诺完整 stereo。", "https://doi.org/10.1016/j.proci.2026.106175"],
      ["PIV in RI fields", "Vanselow et al., PIV in refractive index fields", "说明折射率场如何污染 PIV。", "./paper_library/index.html?q=PIV%20refractive%20index%20fields"],
      ["PIV 工具", "PIVlab / OpenPIV papers", "建立互相关和开源处理 baseline。", "./paper_library/index.html?q=PIVlab%20OpenPIV"],
      ["Synthetic benchmark", "PIV/BOS synthetic image generation + MIRAGE", "无真实数据时最重要的实验设计支撑。", "./paper_library/index.html?q=PIV%20BOS%20synthetic%20MIRAGE"]
    ],
    videoResources: [
      ["OpenPIV 官方文档", "直接服务 PIV 图像处理实现。", "https://openpiv.readthedocs.io/en/latest/"],
      ["PIVlab 官网", "PIVlab 软件、教程和论文入口。", "https://www.pivlab.de/"],
      ["YouTube: PIVlab tutorial 检索", "只作为操作学习入口，引用仍以 PIVlab/OpenPIV 论文为准。", "https://www.youtube.com/results?search_query=PIVlab+tutorial+particle+image+velocimetry"]
    ],
    askSenior: ["组内 PIV-BOST 补偿发生在哪一层？", "能否给一对 raw particle image、BOST 位移/折射率场和标定文件？", "本科成果更需要算法精度、误差传播，还是数据处理工具？"],
    riskControl: ["没有 raw PIV 图像时，不要声称做了 image-layer compensation。", "stereo-PIV-BOST 先作为接口展望，别变成主承诺。", "合成粒子图像要保留参数表，否则结果不可解释。"]
  },
  {
    id: "T4",
    title: "4D BOST 跨形态无真值时序容量选择",
    shortTitle: "4D BOST 容量 Selector",
    fit: "core",
    feasibility: 3,
    novelty: 5,
    risk: 5,
    verdict: "已有 864 格可重复底盘，已不只是概念挑战线；真正风险转为 geometry/真实数据迁移与 UQ。适合由师兄确认后做独立毕设，或并入 T16 的时序容量控制。",
    researchAbstract: "4D BOST 处理随时间变化的三维折射率场，固定低秩先验会在不同流场形态、噪声和视角下发生容量失配。本题不复现 ACM TOG 全系统，而研究测试时不看 field truth 的可观测容量选择：在四类 3D morphology、三类 dynamics、四档 noise、三档 views 和六个配对 seed 上，利用奇异谱、support 重投影一致性与时序物理统计选择 rank 或回退 full rank，并以 nested leave-one-family-out 检验 synthetic transfer。下一步升级为 geometry transfer、uncertainty/abstention 和真实 4D BOST 验证。",
    physicsPlain: "三维火焰或喷流不是静止雕塑，而是一段电影。如果每帧单独猜三维结构，画面会抖、成本也高。低秩时序先验相当于说：这段电影虽然复杂，但主要变化模式可能只有几个。",
    scope: "4D multi-morphology phantom + sparse projection + multi-rank candidates + observable selector + nested LOFO + UQ/拒答升级。",
    difficulty: "中高：synthetic selector 已跑通；难点是避免真值泄漏、处理 geometry shift，并接入真实 BOST 表示容量。",
    software: ["Python", "NumPy/SciPy", "tensorly 可选", "PyTorch 可选", "Plotly/Matplotlib"],
    hardware: ["小 phantom 可在笔记本运行", "真实 4D 或大体素需要 GPU/大内存", "无需实验硬件即可做 toy"],
    knowledge: ["低秩矩阵/张量分解", "动态流场基本概念", "temporal smoothness", "DMD/主模态", "4D 数据可视化"],
    execution: [
      "保留 blobs-sheet、vortex-ring、expanding-shell、jet-filaments 四类 4D stress test，并让师兄替换不合理形态。",
      "从同一 framewise 重建共享奇异分解，生成 rank 1/2/3/5/8/12/full 候选。",
      "只用 spectrum、support residual、时序导数、积分量和可选 held-out view 构造测试时特征。",
      "用 nested leave-one-family-out 训练与评价 selector，报告 mean/p95/worst regret、1%-oracle coverage 和失败格。",
      "接真实 geometry 后升级 leave-one-geometry-out、UQ/拒答，并与 temporal operator 或 TDBOST capacity 对齐。"
    ],
    deliverables: ["864 格/6,048 候选 benchmark", "跨形态 oracle map", "无真值 selector 与特征消融", "UQ/拒答和 failure-cell 展示", "真实 geometry/数据迁移报告"],
    corePapers: [
      ["4D BOST", "He et al., Tensor Decomposition-Based Four-dimensional BOST, ACM TOG 2026", "核心论文；只拆 rank/时序子问题。", "https://doi.org/10.1145/3809488"],
      ["Time-resolved BOST", "Molnar and Grauer, Algorithm for Time-Resolved BOST, AIAA 2025", "外部 time-resolved BOST 参照。", "./paper_library/index.html?q=Time-Resolved%20Background-Oriented%20Schlieren%20Tomography"],
      ["OERF 深度学习体层析", "Huang / Cai limited-projection volumetric tomography", "蔡组少投影、时序预测前史。", "./paper_library/index.html?q=limited-projection%20volumetric%20tomography%20Cai"],
      ["低秩表示邻居", "TensoRF / Tensor4D / K-Planes / HexPlane", "只借表示和 rank-memory 语言，不当 BOST baseline。", "./paper_library/index.html?q=TensoRF%20Tensor4D%20K-Planes%20HexPlane"],
      ["DMD", "Schmid / Tu / Jovanovic DMD and sparsity-promoting DMD", "用于解释时序模态和主频指标。", "./paper_library/index.html?q=dynamic%20mode%20decomposition"]
    ],
    videoResources: [
      ["Steve Brunton YouTube 频道", "DMD、低维动力学和 scientific ML 的高相关视频入口。", "https://www.youtube.com/@Eigensteve"],
      ["Steve Brunton 网站", "配套代码、书和讲义入口。", "https://www.eigensteve.com/"],
      ["YouTube: dynamic mode decomposition 检索", "用于快速补 DMD 直觉。", "https://www.youtube.com/results?search_query=Steve+Brunton+dynamic+mode+decomposition"]
    ],
    askSenior: ["TDBOST 中哪一个 rank/plane factor/width 是组内真正需要按 case 调的容量？", "能否给 18-30 帧的小样例、camera geometry、support/test view 和必须保真的物理量？", "下一轮优先 geometry transfer、UQ/拒答、bias/sync/exposure blur，还是直接接真实数据？"],
    riskControl: ["不要把 synthetic LOFO 写成真实流场泛化。", "不要把 held-out residual 单独当 field quality；正式实验必须保留 full-rank/NeRIF 回退。", "创新点不能写成 ridge 选 rank，而应是 BOST 可观测证据、容量控制和可靠性机制。"]
  },
  {
    id: "T5",
    title: "BOS 位移估计算法 benchmark",
    shortTitle: "BOS 位移 benchmark",
    fit: "high",
    feasibility: 5,
    novelty: 3,
    risk: 1,
    verdict: "很稳的工具型题。它不抢 NeRIF 主线，但能实打实服务 BOST/PIV-BOST 的输入质量。",
    researchAbstract: "BOST 重构的第一步通常不是三维反演，而是从参考/扰动背景图像中估计像素位移。相关法、Lucas-Kanade、Horn-Schunck、Farneback、wavelet optical flow、RAFT-style neural flow 和 registration network 在噪声、纹理、位移幅值、局部遮挡和实时性上各有边界。本科可以建立一个 synthetic BOS image benchmark，比较位移 endpoint error、bad-pixel ratio、置信度图、运行时间，并把误差传播到重投影或折射率重构。",
    physicsPlain: "BOS 的观测量是背景图案移动了多少像素。位移估计就像给两张几乎一样的图片找每个小点挪到哪里；这一步错了，后面的三维重构再高级也会被喂脏数据。",
    scope: "synthetic BOS pairs + 多种位移算法 + EPE/bad mask/runtime + 下游重构误差传播。",
    difficulty: "低到中：工程工作多，但可控、可展示、可复现。",
    software: ["OpenCV optical flow", "scikit-image registration", "OpenPIV/PIVlab 可选", "PyTorch 可选", "NumPy/Matplotlib"],
    hardware: ["普通笔记本足够", "深度光流可选 GPU", "无实验硬件依赖"],
    knowledge: ["图像配准", "光流基本假设", "互相关窗口", "BOS 背景图案设计", "误差传播"],
    execution: [
      "生成带已知位移 ground truth 的随机点背景。",
      "加入 blur、噪声、照度变化、位移梯度、遮挡和纹理稀疏。",
      "比较 Farneback、Lucas-Kanade、Horn-Schunck、互相关、可选 RAFT-lite。",
      "输出 EPE、bad-pixel ratio、runtime 和 confidence map。",
      "把位移误差输入 T1/T2 的简化重构，看下游影响。"
    ],
    deliverables: ["公开 benchmark 表", "算法推荐矩阵", "坏区域 mask", "实时/离线耗时对比", "下游重构误差传播图"],
    corePapers: [
      ["BOS 位移对比", "Comparison of displacement estimation techniques for BOS of high-speed turbulent flows", "最直接的 benchmark 参照。", "./paper_library/index.html?q=displacement%20estimation%20BOS"],
      ["基础光流", "Horn-Schunck / Lucas-Kanade / Farneback", "建立方法族谱，不要只调库。", "./paper_library/index.html?q=Horn%20Schunck%20Lucas%20Kanade%20Farneback"],
      ["深度光流", "FlowNet / PWC-Net / RAFT", "作为现代 neural optical flow 对照。", "./paper_library/index.html?q=FlowNet%20PWC-Net%20RAFT"],
      ["蔡组相关", "Wavelet optical-flow BOS thermal convection", "贴 OERF/蔡组 optical-flow BOS 语言。", "./paper_library/index.html?q=wavelet%20optical%20flow%20BOS"],
      ["新近注册网络", "Unsupervised registration network for BOS displacement extraction", "说明 registration/AI 前处理方向。", "./paper_library/index.html?q=unsupervised%20registration%20BOS"],
      ["CNN 去噪", "Denoising algorithm for BOS images based on CNNs", "补噪声预处理邻居；适合和滤波、互相关、光流置信度一起比较。", "./paper_library/index.html?q=CNN%20denoising%20BOS"],
      ["合成测速", "Time-resolved BOS velocimetry of synthetic raytraced images", "用 ray-traced synthetic images 给位移/速度 benchmark 增加时间维和 ground-truth 对照。", "./paper_library/index.html?q=synthetic%20raytraced%20BOS%20velocimetry"],
      ["focusing schlieren 合成", "Schlieren image velocimetry using synthetic raytraced focusing-schlieren images", "把 focusing-schlieren 光路和已知 ground truth 结合，可作为 F 路线位移/速度估计的动态合成基准。", "./paper_library/index.html?q=raytraced%20focusing%20schlieren%20velocimetry"],
      ["条纹投影定量 schlieren", "Quantitative Schlieren imaging based on fringe projection", "把位移/相位/条纹投影测量接到定量密度或折射率梯度估计，适合给 F 路线增加非随机点背景的 phase-processing 对照。", "./paper_library/index.html?q=Quantitative%20Schlieren%20fringe%20projection"],
      ["喷雾/燃料蒸发 BOS", "Injected fuel vaporization visualized by BOS", "把位移估计 benchmark 接到真实工程热/喷雾场景：纹理、烟雾、折射率梯度和可视化边界比干净 toy 更接近实验噪声。", "./paper_library/index.html?q=fuel%20vaporization%20BOS"]
    ],
    videoResources: [
      ["OpenCV optical flow 文档", "直接服务算法实现。", "https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html"],
      ["OpenPIV 文档", "互相关/PIV 位移估计的工程入口。", "https://openpiv.readthedocs.io/en/latest/"],
      ["YouTube: optical flow tutorial 检索", "用于补 CV 直觉，需以论文/文档为准。", "https://www.youtube.com/results?search_query=optical+flow+tutorial+opencv"]
    ],
    askSenior: ["组里当前 BOS 位移估计用互相关、光流还是自写方法？", "最痛的是噪声、速度、强梯度、背景图案，还是实时性？", "师兄是否需要一个 view health report 来标坏视角？"],
    riskControl: ["不要做成纯 CV benchmark；一定要接 BOST/PIV-BOST 下游误差。", "深度光流不是必须，传统方法做好更稳。", "位移 ground truth 要可解释，否则评估会虚。"]
  },
  {
    id: "T6",
    title: "相机几何与标定误差对 BOST 的影响",
    shortTitle: "几何标定误差",
    fit: "high",
    feasibility: 4,
    novelty: 4,
    risk: 3,
    verdict: "适合物理本科把实验几何讲清楚，也很像课题组真实需求；最好作为 T1/T2 的真实数据质量控制副线。",
    researchAbstract: "BOST 的 forward model 依赖相机位置、视角方向、背景板距离、ROI、mask 和光线模型。标定误差会让同一三维场被错误投影，从而产生系统性重构偏差。本选题可以在 synthetic geometry 中扰动内参、外参、视角角度、背景距离和 mask，建立敏感性矩阵；进阶可加入 view registration、depth-of-field / cone-ray、directional rays 或可微几何修正。",
    physicsPlain: "三维重构像用很多把尺子量同一个物体。尺子的角度、位置或零点错了，算法会努力拼出一个看似合理但其实歪掉的三维场。",
    scope: "相机/视角/背景几何扰动 + 重构误差敏感性 + 标定建议 + manifest 字段检查。",
    difficulty: "中等偏高：需要把几何变量定义清楚，但不一定需要真实设备。",
    software: ["Python", "NumPy", "OpenCV calibration 可选", "scipy.spatial", "Plotly 3D", "PyTorch 可选"],
    hardware: ["合成数据可启动", "真实路线需要相机几何和标定板/背景板参数", "不必自己搭完整 BOST"],
    knowledge: ["相机模型", "坐标变换", "ray tracing / projection", "BOST view geometry", "敏感性分析"],
    execution: [
      "定义简化多视角几何和标准 phantom。",
      "逐个扰动相机角度、距离、背景位置、焦距、ROI/mask。",
      "记录 field error、reprojection residual、resolution proxy。",
      "做 sensitivity heatmap 和 top-error-source 排名。",
      "整理成真实数据 manifest 校验规则。"
    ],
    deliverables: ["几何误差敏感性矩阵", "view health report 原型", "manifest 模板", "标定参数优先级", "给师兄的数据字段清单"],
    corePapers: [
      ["BOST 系统", "Endoscopic BOST / single-camera multi-view OERF papers", "理解九视角/内窥几何。", "./paper_library/index.html?q=endoscopic%20BOST%20Cai"],
      ["景深模型", "Depth-of-field effects in BOS", "提醒 pinhole/thin-ray 不是唯一模型。", "./paper_library/index.html?q=depth-of-field%20BOS"],
      ["方向光线", "Directional rays BOS spatial resolution", "补 spatial resolution 和光线方向约束。", "./paper_library/index.html?q=directional%20rays%20BOS"],
      ["视角/分辨率", "Yu / Liu / Cai spatial resolution and view arrangement", "蔡组 tomography 指标前史。", "./paper_library/index.html?q=view%20arrangement%20spatial%20resolution%20Cai"],
      ["BOS 设计", "Practical aspects of designing BOS experiments", "实验参数建议。", "./paper_library/index.html?q=practical%20BOS%20design"],
      ["自对准 Schlieren", "Plenoptic / automatic self-aligned focusing schlieren", "补真实光路自动对准、光场/聚焦 schlieren 和实验系统稳定性约束。", "./paper_library/index.html?q=self-aligned%20focusing%20schlieren"],
      ["振动修正", "BOS vibration vulnerability correction", "把相机/平台振动写入 view health report，而不是只怪重构算法。", "./paper_library/index.html?q=vibration%20vulnerability%20BOS"],
      ["Rayleigh 密度验证", "BOS and laser Rayleigh scattering complementary density visualization", "用独立密度测量校准/验证 BOS 结果，适合把 M3C 从重投影误差推进到外部物理验证。", "./paper_library/index.html?q=Rayleigh%20scattering%20BOS%20density%20visualization"],
      ["楔形棱镜标定", "BOS calibration based on light deflection angle of a wedge prism", "给几何/角度标定提供一个可解释物理量：已知棱镜偏折角可用来检查像素位移到光线偏折角的标定。", "./paper_library/index.html?q=wedge%20prism%20BOS%20calibration"]
    ],
    videoResources: [
      ["OpenCV camera calibration 文档", "直接服务标定误差理解。", "https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html"],
      ["YouTube: camera calibration OpenCV 检索", "只作为操作补充，正式依据仍用文档/论文。", "https://www.youtube.com/results?search_query=OpenCV+camera+calibration+tutorial"],
      ["MIT OCW / computational imaging 检索", "补投影几何和成像模型。", "https://ocw.mit.edu/search/?q=computational%20imaging"]
    ],
    askSenior: ["真实系统中最不稳定的是相机外参、背景板、ROI、mask 还是同步？", "组内是否有 view geometry 文件格式规范？", "本科做 view health report 是否比新模型更有价值？"],
    riskControl: ["几何变量不要一次放太多，先 one-factor-at-a-time。", "reprojection residual 低可能掩盖真实几何偏差，要保留 synthetic ground truth。", "可微标定只作进阶，不作为保底。"]
  },
  {
    id: "T7",
    title: "火焰前沿 U-Net 检测和弱监督标注",
    shortTitle: "火焰前沿检测",
    fit: "medium",
    feasibility: 4,
    novelty: 3,
    risk: 2,
    verdict: "可毕业但偏离何远哲 BOST 主线。只有当师兄/课题组确实需要火焰图像标注工具时才上升优先级。",
    researchAbstract: "火焰前沿检测把燃烧图像中的 flame front、reaction zone 或亮度边界分割出来，可服务火焰形态、曲率、传播速度和三维重构后处理。弱监督方向可以减少人工标注量，通过少量标注、伪标签、边界损失或时序一致性提升跨工况泛化。它和 OERF 的火焰层析/计算流动可视化有关，但和 He 的 BOST/NeRIF 主线只在“重构后物理结构提取”层面相连。",
    physicsPlain: "火焰前沿就是燃烧反应最活跃的边界区域。检测它类似在图像里把火焰轮廓和背景分开，但真正难点是火焰会抖动、亮度会变、不同工况形态差很多。",
    scope: "火焰图像/视频 + 少量标注 + U-Net/轻量分割 + 边界误差/IoU + 泛化实验。",
    difficulty: "中等：深度学习工具成熟，但数据和标注决定质量。",
    software: ["Python", "PyTorch", "OpenCV", "labelme/CVAT 可选", "scikit-image", "Matplotlib"],
    hardware: ["小模型普通 GPU 最好", "CPU 也可做小数据", "需要图像数据或公开样例"],
    knowledge: ["燃烧火焰基本结构", "图像分割", "U-Net", "弱监督/伪标签", "IoU 和边界误差"],
    execution: [
      "确认是否有组内火焰图像和标注需求。",
      "建立最小标注规范：前景、边界、不确定区域。",
      "训练 U-Net/轻量模型，对比阈值法/边缘法 baseline。",
      "做跨工况泛化和失败案例分析。",
      "如能对接 BOST/CTC，则把前沿作为重构后物理量。"
    ],
    deliverables: ["标注规范", "分割模型", "IoU/边界误差表", "失败案例库", "可视化工具"],
    corePapers: [
      ["火焰分割", "Deep learning-based image segmentation for instantaneous flame front extraction", "最直接的分割入口。", "./paper_library/index.html?q=flame%20front%20segmentation"],
      ["OERF 火焰三维", "3D flame surface measurements via scanning and optical imaging", "把分割结果接到火焰形态物理量。", "./paper_library/index.html?q=3D%20flame%20surface%20Cai"],
      ["OERF 深度学习火焰演化", "Online in situ prediction of 3-D flame evolution from 2-D projections", "课题组 AI for flame tomography 前史；正式 DOI 经 Cambridge/Crossref 核为 .545。", "https://doi.org/10.1017/jfm.2019.545"],
      ["U-Net 基础", "U-Net biomedical image segmentation", "方法底座。", "https://arxiv.org/abs/1505.04597"]
    ],
    videoResources: [
      ["PyTorch 官方 tutorials", "图像分割实现入口。", "https://pytorch.org/tutorials/"],
      ["YouTube: U-Net image segmentation PyTorch 检索", "只作实现参考，论文写作不引用视频。", "https://www.youtube.com/results?search_query=U-Net+image+segmentation+PyTorch+tutorial"],
      ["MIT OCW combustion/fluid search", "补燃烧和流体背景。", "https://ocw.mit.edu/search/?q=combustion"]
    ],
    askSenior: ["POCI/火焰图像方向是否需要后续标注或泛化实验？", "这条线是否仍由何远哲带，还是会偏离主线？", "检测结果最终要服务 BOST、CTC，还是单独图像分析？"],
    riskControl: ["没有数据就不要选。", "不要把普通分割包装成 BOST 创新。", "需要明确物理输出，不只是 IoU。"]
  },
  {
    id: "T8",
    title: "数字全息颗粒 3D tracking 小工具",
    shortTitle: "数字全息颗粒追踪",
    fit: "medium",
    feasibility: 3,
    novelty: 3,
    risk: 3,
    verdict: "贴 OERF 多相反应流，但不贴 He/BOST 主线。除非师兄明确给颗粒/全息数据，否则作为备选。",
    researchAbstract: "数字全息可以从干涉图中恢复颗粒的三维位置、尺寸、速度甚至温度信息，适合金属颗粒燃烧、铝化推进剂和多相流诊断。本科可拆成图像预处理、颗粒检测、3D 定位、轨迹连接和质量控制工具。它的研究价值来自真实实验数据处理，但如果没有组内全息图像，很难只靠公开材料做出有说服力成果。",
    physicsPlain: "全息不是直接拍颗粒照片，而是记录光波被颗粒散射后的干涉纹。通过重建光场，可以判断颗粒在三维空间里的位置，就像从一张特殊的影子图推回小颗粒在哪。",
    scope: "hologram preprocessing + particle detection + depth localization + trajectory linking + visualization。",
    difficulty: "中等偏高：图像处理可做，但光学全息原理和真实数据质量很关键。",
    software: ["Python", "OpenCV", "scikit-image", "trackpy 可选", "NumPy/SciPy", "Napari/Plotly"],
    hardware: ["没有真实全息数据时无法充分验证", "不建议本科自行搭全息系统", "需要存储较多图像序列"],
    knowledge: ["数字全息基本原理", "傅里叶光学入门", "颗粒检测", "轨迹连接", "燃烧颗粒物理背景"],
    execution: [
      "先问清是否有全息图像和处理需求。",
      "读 OERF 数字全息综述和铁颗粒论文，整理数据处理链。",
      "做图像预处理和候选颗粒检测。",
      "实现简单 3D depth scoring 或接已有重建结果做 tracking。",
      "输出轨迹、速度、尺寸分布和质量报告。"
    ],
    deliverables: ["颗粒检测工具", "3D 轨迹可视化", "尺寸/速度统计", "数据质量报告", "失败帧索引"],
    corePapers: [
      ["OERF 综述", "Recent advances and applications of digital holography in multiphase reactive/nonreactive flows", "全息方向入口。", "./paper_library/index.html?q=digital%20holography%20reactive%20flows"],
      ["铁颗粒", "Burning iron particles / micro-explosion / concentration OERF papers", "课题组真实应用背景。", "./paper_library/index.html?q=iron%20particle%20holography%20Cai"],
      ["tracking", "Robust 3D tracking of dynamic reacting particles", "如果做追踪，这是最贴的方向。", "./paper_library/index.html?q=Robust%203D%20tracking%20reacting%20particles"],
      ["图像处理工具", "scikit-image paper", "本科实现的稳妥工具底座。", "./paper_library/index.html?q=scikit-image"]
    ],
    videoResources: [
      ["YouTube: digital holography tutorial 检索", "只作为光学直觉入口，正式路线仍看 OERF 论文。", "https://www.youtube.com/results?search_query=digital+holography+tutorial"],
      ["scikit-image examples", "图像处理实现参考。", "https://scikit-image.org/docs/stable/auto_examples/"],
      ["trackpy 文档", "颗粒轨迹连接工具入口。", "http://soft-matter.github.io/trackpy/stable/"]
    ],
    askSenior: ["组里是否有颗粒/全息数据需要本科生处理？", "这条线是否仍由何远哲带？", "最终物理量是位置、速度、尺寸、温度，还是轨迹质量？"],
    riskControl: ["没有数据就不要作为主线。", "不要把普通 tracking 工具写成全息重建创新。", "如果转向这条线，需要重新补全息光学基础。"]
  },
  {
    id: "T9",
    title: "BOST/NeRIF 论文图自动生成工作台",
    shortTitle: "论文图工作台",
    fit: "high",
    feasibility: 5,
    novelty: 2,
    risk: 1,
    verdict: "最稳的工程交付，可以和 T1/T2/T3/T4 任意主线绑定。单独作为毕设时新意略弱，但组内价值可能很高。",
    researchAbstract: "BOST/NeRIF 实验会产生大量切片、投影、重投影、误差图、参数扫描表和失败案例。如果每次手工出图，复现性差且难以比较。这个选题把数据 manifest、指标计算、统一图表模板、HTML 报告和实验日志接成工作台，让每次实验都能自动生成论文级图和师兄审阅页。它的研究点可以落在可复现实验管理和质量控制，而不是声称提出新模型。",
    physicsPlain: "算法实验不是只跑出一张好看的图，还要知道这张图来自哪个参数、哪个数据、哪个版本。工作台像一个实验记录和出图机器，保证每次结果都能追溯。",
    scope: "manifest + metrics + visualization + report + experiment log，服务 BOST/NeRIF/PIV-BOST。",
    difficulty: "低到中：工程量大但风险小，非常适合长期积累。",
    software: ["Python", "pandas", "Matplotlib/Plotly", "Jinja2 可选", "HTML/CSS", "JSON/CSV"],
    hardware: ["无特殊硬件", "可在本地跑", "适合接入未来 GPU 实验输出"],
    knowledge: ["实验管理", "指标定义", "可视化设计", "文件组织", "基础前端"],
    execution: [
      "定义 BOST/NeRIF 结果目录和 manifest 字段。",
      "实现统一指标读取：RMSE、SSIM、重投影误差、runtime。",
      "生成切片、投影、残差、参数曲线和失败案例图。",
      "生成单次实验 HTML 报告和总索引。",
      "把 T1/T2/T3/T4 demo 全部接入工作台。"
    ],
    deliverables: ["统一结果目录", "自动出图脚本", "HTML 实验报告", "参数扫描索引", "师兄审阅入口"],
    corePapers: [
      ["NeRIF", "He et al. 2025", "工作台服务的主模型。", "https://doi.org/10.1063/5.0250899"],
      ["Computational Flow Visualization", "Huang et al., Cell Reports Physical Science 2024", "开题定位：光学+计算揭示 hidden properties。", "https://doi.org/10.1016/j.xcrp.2024.102282"],
      ["Open BOS dataset", "Open-source BOS tomography dataset", "可用来测试数据接口。", "./paper_library/index.html?q=open-source%20BOS%20tomography%20dataset"],
      ["ScienceAgentBench / AI Scientist", "Agent for Science references", "如果加入自动实验代理，只作为辅助层。", "./paper_library/index.html?q=ScienceAgentBench%20AI%20Scientist"]
    ],
    videoResources: [
      ["Plotly Python 文档", "交互式图和 HTML 报告。", "https://plotly.com/python/"],
      ["Matplotlib examples", "论文静态图风格。", "https://matplotlib.org/stable/gallery/index.html"],
      ["YouTube: scientific Python visualization 检索", "只作实现补充。", "https://www.youtube.com/results?search_query=scientific+python+visualization+matplotlib+plotly"]
    ],
    askSenior: ["师兄最常用的结果格式是什么？", "论文图更偏切片、投影、残差、曲线还是动画？", "这个工具是否能服务组里其他学生？"],
    riskControl: ["单独作为论文题时新意不足，要绑定 T1/T2/T3 的科学问题。", "不要做花哨 UI，优先保证数据可追溯。", "指标定义必须和师兄确认。"]
  },
  {
    id: "T10",
    title: "微型光谱仪重构算法复现",
    shortTitle: "微型光谱仪重构",
    fit: "low",
    feasibility: 3,
    novelty: 3,
    risk: 3,
    verdict: "贴蔡老师另一条强方向，但不贴何远哲当前 BOST 主线。只有导师明确希望转向光谱仪时再考虑。",
    researchAbstract: "微型计算光谱仪通过编码器件、响应矩阵和重构算法把小型硬件测到的混合信号还原成光谱。算法上涉及校准矩阵、正则化反演、压缩感知、神经网络谱恢复和噪声鲁棒性。它与 BOST 共享“硬件编码 + 计算重构 + 病态反问题”的语言，但物理对象和实验系统完全不同。",
    physicsPlain: "传统光谱仪用棱镜/光栅把颜色分开；微型光谱仪把硬件做得很小，让不同波长在探测器上留下不同混合响应，再用算法解出原来的光谱。",
    scope: "response matrix toy + regularized inverse + neural reconstruction + noise/condition-number scan。",
    difficulty: "中等：算法可做，但需要转课题方向和数据来源。",
    software: ["Python", "NumPy/SciPy", "scikit-learn", "PyTorch 可选", "Matplotlib"],
    hardware: ["无真实光谱仪数据可用 toy", "真实方向需要校准数据/响应矩阵", "不需要 GPU"],
    knowledge: ["光谱基础", "线性反问题", "正则化", "压缩感知", "神经网络回归"],
    execution: [
      "读蔡老师微型光谱仪代表论文，确认是否转向。",
      "构造 response matrix 和 synthetic spectra。",
      "比较 least squares、Tikhonov、LASSO、NN reconstruction。",
      "扫描噪声、谱线间距、矩阵条件数和校准误差。",
      "如有真实数据，做响应矩阵校准和测试集评估。"
    ],
    deliverables: ["谱恢复 toy", "算法对比表", "噪声鲁棒性曲线", "校准误差分析", "转向决策说明"],
    corePapers: [
      ["Science Review", "Miniaturization of optical spectrometers, Science 2021", "蔡老师光谱仪方向代表入口。", "./paper_library/index.html?q=Miniaturization%20of%20optical%20spectrometers"],
      ["vdW spectrometer", "Miniaturized spectrometers with a tunable van der Waals junction, Science 2022", "硬件方向代表。", "./paper_library/index.html?q=van%20der%20Waals%20spectrometer"],
      ["Reconstructive spectrometers", "Reconstructive spectrometers: hardware miniaturization and computational reconstruction", "算法综述入口。", "./paper_library/index.html?q=Reconstructive%20spectrometers"],
      ["CTIS OERF", "Super-resolution computed tomography imaging spectrometry", "OERF 计算光谱成像旁支。", "./paper_library/index.html?q=CTIS%20Cai"]
    ],
    videoResources: [
      ["YouTube: computational spectroscopy 检索", "只作背景；具体算法以论文为准。", "https://www.youtube.com/results?search_query=computational+spectroscopy+reconstruction"],
      ["MIT OCW optics search", "补基础光学。", "https://ocw.mit.edu/search/?q=optics"],
      ["scikit-learn linear models", "正则化回归实现入口。", "https://scikit-learn.org/stable/modules/linear_model.html"]
    ],
    askSenior: ["导师是否希望你离开 BOST 主线转光谱仪？", "是否有响应矩阵和真实光谱数据？", "何远哲是否会继续带这条线？"],
    riskControl: ["不要同时主攻 BOST 和光谱仪两条完全不同主线。", "没有数据时只能做算法 toy，论文价值有限。", "它更适合作为课题组全景背景。"]
  },
  {
    id: "T11",
    title: "Agent for Science 辅助 BOST 参数扫描",
    shortTitle: "科研智能体辅助",
    fit: "medium",
    feasibility: 4,
    novelty: 3,
    risk: 2,
    verdict: "可以作为工作流增强，不宜替代物理/算法主线。最适合和 T1/T2/T9 组合。",
    researchAbstract: "科研智能体在这个场景里不应被写成“自动发现新物理”，而应作为 BOST/NeRIF 实验助手：生成配置、排队运行参数扫描、检查输出缺失、汇总指标、生成失败报告和学习日志。它的价值在于降低重复实验成本，帮助你和师兄快速定位哪些参数组合值得精读。",
    physicsPlain: "这个题不是让 AI 猜流场，而是让 AI 帮你做实验助理：把该跑的参数表列好，跑完检查图和指标，发现异常提醒你。",
    scope: "local agent/script + config generation + batch run + metric collection + report + human approval。",
    difficulty: "中等：软件工程为主，物理创新来自它服务的 BOST 实验。",
    software: ["Python", "JSON/YAML", "pandas", "Jinja2/HTML", "可选 LLM API", "Git"],
    hardware: ["无特殊硬件", "可接本地或远程 GPU", "需要实验脚本可重复运行"],
    knowledge: ["实验设计", "参数扫描", "可复现工作流", "基本自动化", "科研记录"],
    execution: [
      "选定 T1/T2 demo 作为被管理实验。",
      "定义 config schema 和实验 registry。",
      "实现批量运行、失败重跑、指标收集。",
      "生成自动报告和学习日志草稿。",
      "加入人工确认点，避免 agent 自动改动核心结论。"
    ],
    deliverables: ["参数扫描代理", "实验日志网页", "失败重跑机制", "结果索引", "师兄周报自动草稿"],
    corePapers: [
      ["ScienceAgentBench", "Toward rigorous assessment of language agents for data-driven scientific discovery", "给科研智能体评价边界。", "./paper_library/index.html?q=ScienceAgentBench"],
      ["AI Scientist", "The AI Scientist", "只作为背景，不照搬自动论文生成。", "./paper_library/index.html?q=AI%20Scientist"],
      ["Self-driving labs", "Science acceleration and accessibility with self-driving labs", "闭环实验背景。", "./paper_library/index.html?q=self-driving%20labs"],
      ["BOST 主线", "NeRIF / PIV-BOST / 4D BOST", "agent 必须服务这些实验。", "./he_bost_citation_spine.html"]
    ],
    videoResources: [
      ["Steve Brunton scientific ML 入口", "补科学计算自动化和数据驱动建模直觉。", "https://www.youtube.com/@Eigensteve"],
      ["GitHub Actions 文档", "自动报告/静态页面部署可参考。", "https://docs.github.com/en/actions"],
      ["YouTube: experiment tracking Python 检索", "只作工程实现参考。", "https://www.youtube.com/results?search_query=experiment+tracking+python+mlflow"]
    ],
    askSenior: ["组内是否接受 AI/自动化作为辅助成果？", "师兄最想自动化的是跑参数、查失败、出图还是写周报？", "哪些步骤必须人工确认？"],
    riskControl: ["不要把 agent 写成主创新点，主创新仍是 BOST 实验问题。", "不要自动生成未经核验的论文结论。", "先做脚本化工作流，再谈 LLM。"]
  },
  {
    id: "T12",
    title: "低成本 BOS/BOST 教学演示",
    shortTitle: "低成本 BOS 演示",
    fit: "high",
    feasibility: 4,
    novelty: 2,
    risk: 2,
    verdict: "非常适合入门和展示，但作为正式毕设需要绑定 T1/T5 的算法分析，否则容易变成教学装置。",
    researchAbstract: "低成本 BOS 用相机、背景图案、热羽流/蜡烛/热风和简单图像处理展示折射率梯度导致的背景位移。它可以作为 BOST/NeRIF 的数据采集前置实验，让你理解背景图案、距离、焦距、热扰动、噪声和安全限制。若要成为毕设，需要把它扩展成图案设计、位移估计、误差分析或低成本数据质量报告，而不只是拍到可视化效果。",
    physicsPlain: "热空气密度低、折射率不同，光线穿过时会弯一点。把带点阵的背景放在后面，用相机拍，就能看到背景图案被热羽流扭曲。",
    scope: "low-cost BOS setup + background design + displacement extraction + safety/data protocol + optional neural toy。",
    difficulty: "低到中：实验直观，但定量和安全规范要认真。",
    software: ["Python/OpenCV", "ImageJ 可选", "OpenPIV 可选", "Matplotlib", "手机/相机采集软件"],
    hardware: ["相机或手机", "点阵/棋盘背景", "热风/小热源/安全火源", "三脚架和稳定光照"],
    knowledge: ["BOS 光路", "背景图案设计", "图像配准/光流", "实验安全", "误差来源"],
    execution: [
      "先用手机/相机拍热羽流 BOS，确认可见位移。",
      "比较随机点、棋盘、投影背景和自然背景。",
      "用 T5 的位移算法做定量位移图。",
      "扫描背景距离、焦距、热源强度和光照。",
      "把采集数据接到 T1 的 synthetic/real-like pipeline。"
    ],
    deliverables: ["演示视频/图", "采集协议", "背景图案对比", "位移图", "安全说明", "可给新人的教学页"],
    corePapers: [
      ["BOS 设计", "Practical aspects of designing BOS experiments", "低成本实验参数入口。", "./paper_library/index.html?q=practical%20BOS%20design"],
      ["Smartphone BOS", "Mobile visualization of density fields using smartphone BOS", "低成本硬件样板。", "./paper_library/index.html?q=smartphone%20BOS"],
      ["Pocket/Single-ended Schlieren", "Pocket Schlieren / single-ended projected virtual source", "低成本/现场可视化邻居。", "./paper_library/index.html?q=pocket%20schlieren%20single-ended"],
      ["MIRAGE", "Synthetic BOS/schlieren image generation via MIRAGE", "连接真实演示与仿真 benchmark。", "./paper_library/index.html?q=MIRAGE"],
      ["燃料蒸发 BOS", "Injected fuel vaporization using BOS", "把低成本热羽流演示推广到工程喷雾/蒸发场景：重点看背景布置、折射率梯度、成像距离和可视化边界。", "./paper_library/index.html?q=Visualization%20Injected%20Fuel%20Vaporization%20BOS"],
      ["等离子体热流 BOS", "BOS thermal characterization of plasma-actuator induced flow", "适合解释 BOS 为什么能在电磁干扰或不方便接触测温的场景中做热场可视化；也给低成本热羽流 demo 提供安全边界语言。", "./paper_library/index.html?q=plasma%20actuator%20BOS%20thermal%20characterization"],
      ["SAFS 光路设计", "Design and validation of Self-Aligned Focusing Schlieren optical systems", "Research Square CC BY 预印本已缓存；适合把 alignment、field of view、sensitivity、calibration 和验证流程写成低成本/受限空间装置变量。", "./paper_library/index.html?q=Self-Aligned%20Focusing%20Schlieren%20optical%20systems"],
      ["高速 SAFS 测速", "Application of high-speed self-aligned focusing schlieren system for supersonic flow velocimetry", "把低成本/自对准光路和速度估计联系起来；适合说明光路不是装饰，曝光、帧率和定量测速会共同决定数据能不能用于算法。", "./paper_library/index.html?q=high-speed%20self-aligned%20focusing%20schlieren%20velocimetry"],
      ["自动对准", "Automatic self-aligned focusing schlieren", "提醒低成本演示也要面对对准稳定性，不只是拍到漂亮图。", "./paper_library/index.html?q=automatic%20self-aligned%20focusing%20schlieren"],
      ["脉冲照明", "High-speed SAFS and BOS with pulsed laser illumination", "补低成本/高速演示的照明、曝光、同步和安全边界；只作官方 DOI 入口，不缓存未确认许可 PDF。", "./paper_library/index.html?q=pulsed%20laser%20illumination%20SAFS%20BOS"],
      ["knife-edge-free", "Knife-edge-free self-aligned colour schlieren imaging", "给本科低成本演示一个更安全、更容易解释的彩色 schlieren 邻居；当前只作开放入口，不缓存未确认许可 PDF。", "./paper_library/index.html?q=knife-edge-free%20self-aligned%20colour%20schlieren"],
      ["彩色/全景/窄场 Schlieren", "Color Panoramic and Narrow-Field Schlieren Observations of Heterogeneous Fluid Flows", "补低成本/教学演示之外的真实 Schlieren 光路能力：同一类密度梯度可通过不同视场、色彩和灵敏度策略呈现。", "./paper_library/index.html?q=Color%20Panoramic%20Narrow-Field%20Schlieren"],
      ["微尺度参数", "Key parameters in schlieren systems for microscale flow-field measurements", "把焦距、刀口/光阑、放大率、灵敏度等系统参数写成可扫描变量。", "./paper_library/index.html?q=microscale%20schlieren%20key%20parameters"]
    ],
    videoResources: [
      ["YouTube: background oriented schlieren 检索", "用于观察不同演示装置，需筛选官方/高校来源。", "https://www.youtube.com/results?search_query=background+oriented+schlieren+experiment"],
      ["YouTube: schlieren photography setup 检索", "仅作可视化装置直觉。", "https://www.youtube.com/results?search_query=schlieren+photography+setup"],
      ["OpenCV optical flow 文档", "把演示视频转成位移图。", "https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html"]
    ],
    askSenior: ["实验室是否允许本科生搭低成本热羽流/BOS 装置？", "安全边界和空间限制是什么？", "师兄是否认可它作为算法主线的数据演示章节？"],
    riskControl: ["不要把演示当完整 BOST。", "明火/热源必须遵守实验室安全。", "正式论文要有定量指标和算法连接。"]
  },
  {
    id: "T13",
    title: "基于开源 BOS/TBOS 数据的预演 benchmark",
    shortTitle: "开源数据预演",
    fit: "high",
    feasibility: 4,
    novelty: 3,
    risk: 2,
    verdict: "真实组内数据未到位时的最佳退路。它能让你先把 loader、manifest、指标和图表做完整。",
    researchAbstract: "开源 BOS/TBOS 数据可以让本科生在不等待组内权限的情况下预演整个 pipeline：下载数据、解析相机/视角/标定、提取位移、运行 baseline、计算重投影和误差指标、生成报告。它的关键不是追求 SOTA，而是建立可迁移接口，一旦何远哲给 OERF 样例数据，只需要替换 manifest 和 loader。",
    physicsPlain: "如果暂时拿不到实验室数据，就先用别人公开的数据练完整流程。这样等师兄给数据时，你不是从零开始，而是已经有一套能读、能跑、能出图的工具。",
    scope: "open dataset loader + manifest + baseline run + visualization + migration checklist。",
    difficulty: "中等：数据格式可能乱，但风险可控。",
    software: ["Python", "requests/zip tools", "NumPy", "OpenCV", "pandas", "Plotly/Matplotlib"],
    hardware: ["本地磁盘空间", "小数据无需 GPU", "大数据需筛选 subset"],
    knowledge: ["数据 manifest", "BOS/TBOS 数据结构", "标定字段", "位移/重构指标", "可复现数据处理"],
    execution: [
      "选择 1-2 个开源 BOS/TBOS 数据集，不贪多。",
      "写下载/校验/子集选择脚本。",
      "统一成 OERF-style manifest：views、frames、mask、geometry、units。",
      "运行位移估计或读取已有 deflection。",
      "生成和 T1/T5/T9 兼容的报告。"
    ],
    deliverables: ["open-data loader", "subset manifest", "复现实验报告", "迁移到 OERF 数据清单", "数据质量表"],
    corePapers: [
      ["Open BOS dataset", "Open-source BOS tomography dataset of high-speed flow over a flight body", "最直接的开放 BOST 数据入口。", "./paper_library/index.html?q=open-source%20BOS%20tomography%20dataset"],
      ["Helium jet dataset", "Schlieren and BOS velocimetry dataset", "PIV/BOS 图像处理练习。", "./paper_library/index.html?q=helium%20jet%20BOS%20velocimetry"],
      ["NASA TomoBOS", "Development of Tomographic BOS capability at NASA Langley", "大型风洞 TBOS 工程参照。", "./paper_library/index.html?q=NASA%20TomoBOS"],
      ["MIRAGE / synthetic", "MIRAGE and PIV/BOS synthetic image generation", "数据不足时的合成补充。", "./paper_library/index.html?q=MIRAGE%20synthetic%20BOS"],
      ["AIAA synthetic", "Time-resolved BOS velocimetry of synthetic raytraced images", "没有真实开源多视角数据时，可把 raytraced synthetic sequence 当成预演 benchmark。", "./paper_library/index.html?q=Time%20Resolved%20BOS%20Velocimetry%20Synthetic%20Raytraced"]
    ],
    videoResources: [
      ["OpenPIV 文档", "处理图像数据入口。", "https://openpiv.readthedocs.io/en/latest/"],
      ["Zenodo", "很多流体数据集发布在这里，可检索 BOS/PIV。", "https://zenodo.org/search?q=background%20oriented%20schlieren"],
      ["YouTube: NASA BOS wind tunnel 检索", "只作工程场景观察。", "https://www.youtube.com/results?search_query=NASA+background+oriented+schlieren+wind+tunnel"]
    ],
    askSenior: ["如果组内数据暂时不能给，师兄是否认可先用 open BOS 数据预演？", "OERF 数据 manifest 应该包含哪些字段？", "开源数据结果能否作为开题/中期的保底图？"],
    riskControl: ["不要同时下载太多大数据集，先做小 subset。", "开源数据场景和 OERF 火焰不同，结论要写清边界。", "必须保留迁移清单，否则只是复现别人的数据。"]
  },
  {
    id: "T14",
    title: "NeRIF vs NeDF 表示方式对比",
    shortTitle: "神经表示方式对比",
    fit: "core",
    feasibility: 4,
    novelty: 4,
    risk: 3,
    verdict: "很有论文感，适合有一定 PyTorch/GPU 基础后做。建议作为 T1 的方法消融章节。",
    researchAbstract: "BOST neural field 可以选择不同的被表示变量：直接表示折射率 n(x,y,z)，表示折射率梯度或 deflection field，表示低维 primitive/hash-grid，或直接监督投影位移。不同表示会改变物理约束、梯度计算、噪声敏感性和可解释性。本科可在同一 synthetic forward model 下比较 plain MLP、Fourier features、SIREN、hash/grid-like encoding，以及 NeRIF/NeDF/NRIP 风格变量，输出少视角和噪声下的稳定性边界。",
    physicsPlain: "同一个流场可以用不同方式描述：可以直接描述每个点的折射率，也可以描述折射率变化的方向，或者描述光线最后偏了多少。选哪种描述，会影响算法好不好学、结果好不好解释。",
    scope: "same phantom/same views + different neural representations + encoding/loss ablation + stability metrics。",
    difficulty: "中等偏高：需要写清楚变量定义和 loss，避免只做网络调参。",
    software: ["PyTorch", "NumPy", "tiny-cuda-nn 可选", "Matplotlib", "Weights/CSV logging"],
    hardware: ["小模型可 CPU/GPU", "hash/grid 或大扫描最好 GPU", "无需实验硬件"],
    knowledge: ["Neural fields", "Fourier features", "SIREN", "hash encoding", "自动微分", "物理约束 loss"],
    execution: [
      "固定 T1 的 synthetic BOST forward model。",
      "实现 plain MLP、Fourier features、SIREN-style activation。",
      "设计三类输出变量：n、grad n / deflection proxy、primitive/grid-like。",
      "统一训练预算，扫描视角数和噪声。",
      "报告精度、速度、稳定性、物理可解释性和失败模式。"
    ],
    deliverables: ["表示方式对比表", "encoding 消融", "loss 设计图", "速度/显存曲线", "少视角稳定性结论"],
    corePapers: [
      ["NeRIF", "He et al. 2025", "直接表示折射率场主线。", "https://doi.org/10.1063/5.0250899"],
      ["NeDF", "Neural deflection fields for sparse-view TBOS", "表示 deflection/gradient 邻居。", "./paper_library/index.html?q=NeDF"],
      ["NRIP", "Neural Refractive Index Primitives", "hash encoding、mask、gradient loss 邻居。", "./paper_library/index.html?q=Neural%20Refractive%20Index%20Primitives"],
      ["Fourier/SIREN/hash", "Fourier Features / SIREN / Instant-NGP / Neural Fields survey", "神经表示基础。", "./paper_library/index.html?q=Fourier%20Features%20SIREN%20Instant-NGP%20Neural%20Fields"],
      ["Single-view RI tomography", "Single View Refractive Index Tomography with Neural Fields", "curved-ray neural RI field 方法邻居。", "./paper_library/index.html?q=Single%20View%20Refractive%20Index%20Tomography"]
    ],
    videoResources: [
      ["YouTube: neural fields / NeRF tutorial 检索", "补坐标神经场直觉。", "https://www.youtube.com/results?search_query=neural+fields+NeRF+tutorial"],
      ["NVIDIA Instant-NGP project", "hash/grid encoding 的直观入口。", "https://nvlabs.github.io/instant-ngp/"],
      ["PyTorch tutorials", "实现坐标 MLP。", "https://pytorch.org/tutorials/"]
    ],
    askSenior: ["师兄更关心 n、grad n、deflection 还是最终 PIV 速度补偿？", "NeRIF 代码是否已有固定 encoding/loss？", "NRIP/NeDF 是否需要写成正式对比，还是 related work 即可？"],
    riskControl: ["不要承诺完整复现 NeDF/NRIP；只比较表示思想。", "训练预算必须统一，否则比较不公平。", "表示方式对比要回到物理可解释性，不只是 loss 低。"]
  },
  {
    id: "T15",
    title: "BOST 时间不同步与投影补全误差图谱",
    shortTitle: "同步/缺失视角误差",
    fit: "core",
    feasibility: 5,
    novelty: 4,
    risk: 2,
    verdict: "非常适合作为 T1 的真实实验增强线：贴 4D、贴系统误差、也能用合成数据先做。",
    researchAbstract: "真实多视角 BOST/4D BOST 的问题不止少视角和噪声，还包括相机触发时间差、坏视角、缺失帧、局部遮挡、projection completion 引入的伪结构，以及不同视角看到的不是同一瞬间流场。本题在 synthetic dynamic phantom 上加入 time offset、missing view、bad-view noise 和 projection synthesis baseline，建立误差图谱和容忍范围。它能帮助回答组内真实数据质量问题：这组多视角数据到底还能不能可信重构？",
    physicsPlain: "多相机拍高速火焰时，如果某个相机晚了一点点，它看到的火焰已经变了。把这些不同时间的照片硬拼成一个三维场，就会出错。这个题就是量化“晚多少会坏、缺几个视角还能用”。",
    scope: "dynamic phantom + time offset + missing/bad views + projection completion + error tolerance map。",
    difficulty: "中等偏稳：可以从 T1/T4 合成系统扩展，真实意义强。",
    software: ["Python", "NumPy/SciPy", "PyTorch 可选", "tensorly 可选", "Matplotlib/Plotly"],
    hardware: ["合成数据普通电脑可启动", "4D 大扫描最好 GPU", "无需真实实验硬件"],
    knowledge: ["多视角同步", "动态流场", "缺失数据补全", "低秩/插值 baseline", "重投影误差"],
    execution: [
      "构造 moving/oscillating 3D phantom。",
      "模拟每个视角不同 time offset。",
      "加入 missing view、bad view、noise、mask 裁剪。",
      "比较直接重构、简单插值、低秩补全、轻量 projection completion。",
      "输出 time offset / missing view / noise 三维误差热图和容忍范围。"
    ],
    deliverables: ["同步误差容忍图", "缺失视角热图", "projection completion baseline", "自动质量报告", "真实数据风险清单"],
    corePapers: [
      ["BOST time asynchrony", "Gao et al., Fire 2023", "直接支撑时间不同步问题。", "./paper_library/index.html?q=time%20asynchrony%20BOST"],
      ["4D BOST", "He et al., ACM TOG 2026", "把时间维度和真实高速场联系起来。", "https://doi.org/10.1145/3809488"],
      ["Projection completion", "PENTAGON / PIPEN / NVRT / Fast-NFRT / FluidNeRF", "投影补全和少视角神经重构邻居。", "./paper_library/index.html?q=PENTAGON%20PIPEN%20FluidNeRF"],
      ["Dynamic backgrounds / missing views", "Dynamic backgrounds BOS and BOS UQ papers", "系统误差和位移质量控制。", "./paper_library/index.html?q=dynamic%20backgrounds%20BOS%20uncertainty"],
      ["Low-rank / DMD", "DMD / Robust PCA / dynamic tomography", "低秩时序和缺失数据补全语言。", "./paper_library/index.html?q=Robust%20PCA%20DMD%20dynamic%20tomography"],
      ["振动/坏视角", "BOS vibration vulnerability correction", "把机械振动、视角漂移和 registration 稳定化纳入真实数据质量报告。", "./paper_library/index.html?q=vibration%20vulnerability%20BOS"],
      ["高速诊断", "Detonation-exhaust BOS velocity and density diagnosis", "提醒高速/强瞬态场里同步、位移和 derived-field 误差会被放大。", "./paper_library/index.html?q=detonation%20exhaust%20BOS%20velocity%20density"],
      ["极端流 Schlieren/TAS", "Impinged shock LAS + high-speed schlieren", "把 T15 的同步/高速误差问题接回 OERF 极端流光学诊断背景：高帧率 Schlieren 与吸收谱联合测量时，触发、时间基准和系统响应都必须说清楚。", "./paper_library/index.html?q=impinged%20shock%20laser%20absorption%20high-speed%20schlieren"],
      ["工程验证邻居", "Helium coolant-channel BOS validation", "把缺失/坏视角和系统误差讨论落到工程通道流验证场景：simulation-validation 论文更重视边界条件、实验可比性和模型验证。", "./paper_library/index.html?q=helium%20coolant%20baffled%20BOS%20validation"],
      ["高速 BOS 特征提取", "Wavepacket extraction from high-speed BOS measurements of a heated jet", "把 T15 从“同步/坏视角”推进到“高速 BOS 电影能提取什么相干结构”：适合讨论 time-resolved data、wavepacket、feature extraction 和 aeroacoustic flow visualization。", "./paper_library/index.html?q=wavepacket%20high-speed%20BOS%20dual-stream%20jet"],
      ["脉冲照明与同步", "High-speed SAFS and BOS with pulsed laser illumination", "补高速可视化系统的 illumination / exposure / trigger 约束，提醒算法结果还受光源、快门和同步链路影响。", "./paper_library/index.html?q=High-Speed%20Self-Aligned%20Focusing%20Schlieren%20Background-Oriented%20Schlieren%20Pulsed%20Laser"],
      ["倾斜平面成像", "Scheimpflug-integrated focusing schlieren for tilted planes in 3D high-speed flows", "把同步/缺失视角问题扩展到成像平面选择：不是所有视角都在同一焦平面，tilted-plane imaging 会影响时序数据的可解释性。", "./paper_library/index.html?q=Scheimpflug%20integrated%20focusing%20schlieren"],
      ["高速 SAFS 测速", "High-speed self-aligned focusing schlieren for supersonic flow velocimetry", "用真实高速测速应用提醒 T15：时间分辨率、光路对准和速度提取算法必须一起写进数据健康报告。", "./paper_library/index.html?q=supersonic%20flow%20velocimetry%20SAFS"],
      ["光路/灵敏度约束", "SAFS / digital focusing / weak-RI-gradient schlieren system papers", "把 T15 的坏视角、缺失视角和同步误差扩展到光路设计、灵敏度、受限空间、振动和系统参数层面。", "./paper_library/index.html?q=SAFS%20focusing%20schlieren%20weak%20refractive%20index%20gradient"]
    ],
    videoResources: [
      ["Steve Brunton DMD 视频入口", "理解动态模态和低维时序。", "https://www.youtube.com/@Eigensteve"],
      ["YouTube: multi camera synchronization 检索", "只作实验同步直觉，不作为论文依据。", "https://www.youtube.com/results?search_query=multi+camera+synchronization+high+speed+imaging"],
      ["MIT OCW fluid mechanics channel", "补动态流场基础。", "https://www.youtube.com/@mitocw"]
    ],
    askSenior: ["九视角/4D BOST 中是否存在触发时间差、坏视角或缺帧痛点？", "组里现在怎么处理缺失视角：丢帧、插值、mask，还是人工筛？", "师兄更需要补全算法，还是判断数据能不能信的质量报告？"],
    riskControl: ["projection completion 不要做成大网络，先做可解释 baseline。", "time offset 要和流场时间尺度绑定，不要只扫抽象数字。", "真实数据结论必须等师兄确认同步记录。"]
  },
  {
    id: "T16",
    title: "面向少视角 BOST 的自有几何条件化算子与 NeRIF 物理精化",
    shortTitle: "Own operator → NeRIF",
    fit: "core",
    feasibility: 4,
    novelty: 5,
    risk: 3,
    verdict: "这是当前最值得与何远哲核准的研究池。v3k-D 已把 FNO/ridge 初值、global/geometry/spectral step、逐场二次最速下降和实际 A/A^T 调用数放进同一 validation-only 协议。FNO-start quadratic 的 validation relative L2 为 0.143855，但只比 128-call geometry Landweber 好 1.24%，且 noise-OOD 反向。下一道门不是立即训练新网络，而是 projected BB + fresh case-level lock；learned conditional scalar、确认性 superiority、真实 NeRIF refinement 继续关闭。",
    researchAbstract: "普通 NeRIF 针对一个观测场做 per-instance optimization；共享算子从多个 phantom 学摊销逆映射。既有负结果说明弱 baseline、短训练或没有使用 acquisition geometry 都会制造虚假创新。v3k-C/D 进一步表明，伴随残差与经典自适应步长可以在不增加学习参数时吃掉大部分可解释收益。真正可投稿的问题因而被压缩为：在相同 forward/adjoint 调用预算下，学习到的有界步长/预条件器能否稳定超过 projected BB、quadratic line-search、ridge-start 和 FNO-start，并在 noise、family、joint OOD 与 worst-layout 尾部同时获益，随后是否真的缩短 NeRIF 的 time-to-quality？",
    physicsPlain: "先把少视角反问题中相机确实看得到的部分用 ridge/CGLS 稳定解出来，再让小网络只补训练样本中反复出现、数值法容易抹平的结构。网络结果不直接当最终答案，而是作为 NeRIF 的连续场起点；NeRIF 再用真实光线前向模型逐场校正。",
    scope: "synthetic/open/real paired data + fixed camera-budget protocol + ridge/CGLS/TV/RBF + matched 3D U-Net/FNO + geometry-aware camera-set encoder + random/ridge/operator/oracle NeRIF initialization + time-to-Q_audit-quality + OOD/geometry/real-data audit；dual/null 路线作为负结果附录。",
    difficulty: "中等偏高：模型代码比完整 NeRIF/TDBOST 短，但批量三维数据、显存、跨分布划分和公平 baseline 决定结果可信度。",
    software: ["Python", "PyTorch", "neuraloperator", "DeepXDE 可选", "NumPy/SciPy", "scikit-image/ASTRA 可选", "pandas", "Matplotlib/Plotly"],
    hardware: ["2D/O0 tutorial 可在普通笔记本运行", "24^3-32^3 的 3D FNO 建议使用 NVIDIA GPU", "先实测显存再决定 48^3/64^3，不承诺 200^3", "不需要先搭实验台，真实数据是升级项"],
    knowledge: ["函数空间与算子学习", "DeepONet branch/trunk", "Fourier Neural Operator 与 FFT", "BOST forward/adjoint operator", "反问题与正则化", "PyTorch 3D tensor", "OOD generalization 与 data leakage", "重投影/held-out view 评价"],
    execution: [
      "已完成 checkpoint：168 个 8x16x16 paired volumes、train-only global calibration、physics lift、3D U-Net、官方 residual FNO、五类 test domain 和逐样本多指标。",
      "已完成三种子消融：residual/absolute output、with/without reprojection、43k FNO vs 49k U-Net；发现 output parameterization 随 view 工况翻转。",
      "已完成 18-run reliability 筛查：fixed/learned metadata/observed-residual gates 没有统一 Residual/Absolute 翻转，关键统计整轮复跑一致。",
      "已完成 3-seed dual v1：128 个固定 support/query 变体、共享双输出、uniform/support-fit/query 三种融合和 feature-alignment 审计；确认 support-fit 强基线与 query-router collapse。",
      "已完成独立专家 v2a：86,823 参数、分支差异提升、五域 support-fit、独立 checksum 与容量边界审计。",
      "已完成零空间上界与 v2b：oracle null headroom 为 38.626%，matched learned null 总体 -0.126%，证明硬一致性稳定但方向学习不足。",
      "已完成 adaptive-query v2c pilot：逐样本从候选角中选 query 后闭式求 alpha，单 query +0.746%，all-query +0.956%，support leakage 2.26e-9；这不是预先固定一台相机的部署结论。",
      "已完成 controlled-budget v2d：锁定 Q_fit/Q_audit，对照 S-only、S∪Q direct、learned correction 与 numerical update；当前 checkpoint 在 K=4/6/8 为 0/3 通过。",
      "已完成 training-matched v3a：256 source fields、96 independent test fields、5,184 sample rows、1,728 field clusters；ridge-FNO 在 K=4/6/8 为 3/3 development gates 通过。",
      "已完成自有算法 v3b：同输入 U-Net/FNO/DeepONet/ray-set benchmark、4,320 sample rows、1,440 field clusters；K=6 通过，K=4/8 暴露效应与尾部风险。",
      "已完成 v3d 96-epoch FNO plateau audit：288 history rows、21 seed-checkpoints、2,688 per-seed field rows；末个 block mean/max-seed validation 改善 2.67%/2.83%，未过 plateau。",
      "已完成 v3d 240-epoch optimizer audit：2,016 history rows、171 checkpoints、1,536 dev2 rows；FNO 固定 epoch 冠军已锁。",
      "已完成 v3e 五架构成本账本：15 fresh MPS workers、参数/FLOPs-v1/时间/内存与 FNO time-to-target；参数少不再视为低成本证据。",
      "已完成 v3j-v3k-B 几何机制淘汰：global modulation 不可辨识，局部 ray values 有用但 angle pairing 有害；停止 attention 扩参并转向伴随残差。",
      "已完成 v3k-C adjoint 闸门：28 布局通过 identity/finite-difference 检查，validation-only 选中的 geometry-normalized projected Landweber 成为必须击败的数值基线。",
      "已完成 v3k-D 强数值对照：raw/feasible ridge、global/geometry/lookup Landweber 与逐场 quadratic control；选择承诺在读取 test 前写盘，0/128/193-call 前沿与逐步目标审计均可复核。",
      "当前 7 天动作：实现 projected BB；冻结 safeguard、restart、投影和调用计数；只在 validation 选择，开发域只报告，不训练 learned scalar。",
      "当前第 1-2 周：让师兄填写 geometry/data manifest；从 v3e 复算成本，并冻结所有候选的 60/120/180/240 matched learning-curve schema。",
      "第 3-4 周仅在 geometry Go 后：先写 LoRA/F-Adapter/acquisition-conditioned 功能 pilot；共同填满 error–compute frontier 后才比较 geometry controls。",
      "第 5-6 周：建立独立 24^3/32^3 forward data，并加入 angular-layout、遮挡、标定扰动和 thin-front family OOD。",
      "第 7-8 周：把 synthetic reliability features 映射到组内 displacement confidence、ray coverage 与 geometry；无法获取时明确停在 closure。",
      "第 9-10 周：接入组内样例；用 held-out camera、边界、积分量、risk-coverage 和专家判断做无真值 failure audit。",
      "第 11 周：比较 random/physics/operator 三种 NeRIF 初始化的总 wall time、最终 residual 与失败率。",
      "第 12 周：形成 8 张论文图、模型/数据卡、负结果边界和可复跑环境。"
    ],
    deliverables: ["已完成 168-sample BOST operator dataset", "已完成 lift/U-Net/FNO 对照", "已完成三种子 residual/loss/capacity 消融", "已完成 18-run simple-gate 负对照", "已完成 3-seed shared dual v1 与 query-router collapse", "已完成 independent dual v2a 与 support-fit 强基线", "已完成 exact nullspace headroom audit", "已完成 matched free/null learned corrector 负对照", "已完成 adaptive-query v2c pilot", "已完成 controlled-budget v2d 的 19,008 行结果与 checksum validator", "已完成 training-mask-matched v3a 的 5,184 行结果、96-field 风险统计与 28 tests", "已完成自有 ray-set v3b 的 4,320 行结果、同信息 U-Net/FNO/DeepONet 对照与独立 validator", "已完成 v3c 独立 dev2、zero-init adapter、continued-FNO 对照、成本审计与 checksum validator", "已触发停止当前 frozen per-view adapter；blind final 仍封存", "已完成 F-Adapter/R2-FFNO/MG-TFNO/GINO/FourierFT 最近邻查重与 v3d 创新门槛", "已完成 v3d FNO 24→96 epoch validation-only 历史审计", "已完成 v3d 三 optimizer protocols 到 240 epochs、2,016 history rows、171 checkpoints、1,536 dev2 rows、配对统计与 validator", "已锁定 carry-Adam/restart-cosine 开发冠军与 long-cosine plateau control", "已完成 v3e 五架构 15-worker 参数/FLOPs-v1/MPS 时间/内存与 time-to-target 账本", "已完成 geometry/data Go-No-Go manifest", "已完成六几何 nested-LOGO/UQ 审计", "已完成 v3k-C adjoint identity/gradient 与 projected Landweber 数值闸门", "已完成 v3k-D 强数值基线、0/128/193-call 账本、selection commit、checksum 与独立 validator", "待实现 projected BB 并冻结 safeguard/restart 协议", "待建立 fresh case-level blind lock", "待师兄确认真实 F/J^T/VJP 或可微 forward loss", "待完成独立 nonlinear/cone-ray forward", "待补 CGLS/TV/RBF/GRU-BOST", "待完成 NeRIF warm-start", "待接组内 geometry/data"],
    corePapers: [
      ["DeepONet 根论文", "Lu et al., Learning nonlinear operators via DeepONet, Nature Machine Intelligence 2021", "抽 branch/trunk、sensor sampling、function-to-function mapping 和 generalization error。", "https://doi.org/10.1038/s42256-021-00302-5"],
      ["FNO 根论文", "Li et al., Fourier Neural Operator for Parametric Partial Differential Equations, ICLR 2021", "抽 spectral convolution、resolution transfer 和 Navier-Stokes benchmark；不要把论文的 zero-shot 结论直接搬到 BOST。", "https://iclr.cc/virtual/2021/poster/3281"],
      ["v3d 最近邻", "Zhang et al., F-Adapter, NeurIPS 2025", "frequency-adaptive PEFT 已被提出；复现 LoRA/vanilla/F-Adapter 比较，并把它设为 acquisition-conditioned 方法必须击败的最近邻。", "https://proceedings.neurips.cc/paper_files/paper/2025/hash/a12e362d89d4e0b40760f839f91550ee-Abstract-Conference.html"],
      ["低秩谱核先例", "Chou and Chen, Reduced-rank Factorized FNO, ACML 2025", "低秩 spectral kernels 与 rank saturation 已有先例；rank 必须 validation-select，不能当原创组件。", "https://proceedings.mlr.press/v304/chou26a.html"],
      ["张量化谱核先例", "Kossaifi et al., Multi-Grid Tensorized FNO, TMLR 2024", "参数空间 tensor factorization 已覆盖；本项目必须把采集几何条件化和参数压缩严格分开。", "https://openreview.net/forum?id=oFqHIkw8sd"],
      ["geometry 最近邻", "Li et al., Geometry-Informed Neural Operator, NeurIPS 2023", "geometry-informed 概念已存在，但其 geometry 是物理域/点云；本项目需明确研究相机/光线采集算子。", "https://proceedings.neurips.cc/paper_files/paper/2023/hash/70518ea42831f02afc3a2828993935ad-Abstract-Conference.html"],
      ["神经算子总论", "Kovachki et al., Neural Operator: Learning Maps Between Function Spaces, JMLR 2023", "建立 function/operator/discretization-invariance 的严格语言，按综述选 baseline。", "https://www.jmlr.org/papers/v24/21-1524.html"],
      ["逆算子", "Molinaro et al., Neural Inverse Operators for Solving PDE Inverse Problems, ICML 2023", "最直接的 inverse-operator 方法论，解释为什么逆问题不只是普通 supervised regression。", "https://openreview.net/forum?id=S4fEjmWg4X"],
      ["物理约束算子", "Li et al., Physics-Informed Neural Operator, ICLR 2022", "借鉴 data loss + physics loss；在 T16 第一阶段把 physics 明确定义为 BOST forward projection，而不是先上 Navier-Stokes。", "https://openreview.net/forum?id=dtYnHcmQKeM"],
      ["稀疏场重建", "Zhao et al., RecFNO, International Journal of Thermal Sciences 2024", "最接近本科主模型：比较 sparse observation embedding、FNO、分辨率迁移和 POD/CNN baseline。", "https://doi.org/10.1016/j.ijthermalsci.2023.108619"],
      ["实验流体邻居", "Zhang et al., Operator learning for reconstructing flow fields from sparse measurements, JCP 2025", "含 Schlieren 与 3D turbulent jet 案例；抽 sensor dropout、噪声、实验数据和 full-field reconstruction 评价。", "https://doi.org/10.1016/j.jcp.2025.114148"],
      ["少标签逆算子", "Cho and Son, Physics-Informed Deep Inverse Operator Networks, ICLR 2025", "拿到真实观测和可信 forward、但缺 paired 3D truth 时再研究；先检查其稳定性假设能否迁到 BOST。", "https://openreview.net/forum?id=0FxnSZJPmh"],
      ["几何算子", "Li et al., Geo-FNO, JMLR 2023; GINO, NeurIPS 2023", "真实 ray/mesh 无法可靠提升到规则网格时的进阶方案；第一版不引入。", "https://jmlr.org/papers/v24/23-0064.html"],
      ["高分辨率算子", "Kossaifi et al., Multi-Grid Tensorized FNO, TMLR 2024", "24^3/32^3 已证明有价值且显存成为瓶颈后，再读 domain/parameter decomposition。", "https://research.nvidia.com/labs/lpr/publication/kossaifi2023multi/"],
      ["OERF 数据驱动前史", "Huang et al., Limited-projection volumetric tomography via deep learning, AST 2020", "蔡组 projection-to-volume 前史；比较 limited projections、ART baseline、数据生成和 inference speed。", "https://doi.org/10.1016/j.ast.2020.106123"],
      ["OERF 时序前史", "Huang, Liu and Cai, Online prediction of 3-D flame evolution, JFM 2019", "概念上连接 projection history 与 3D evolution；它不是 neural-operator 论文，引用时要如实称为课题组数据驱动前史。", "https://doi.org/10.1017/jfm.2019.545"],
      ["何远哲主线", "He et al., Neural refractive index field, Physics of Fluids 2025", "提供 BOST forward model、折射率输出、边界和重投影评价，作为 T16 的物理定义而不是算子架构来源。", "https://doi.org/10.1063/5.0250899"]
      ,["BOST 直接学习前史", "Bo et al., Background-oriented Schlieren tomography using gated recurrent unit, Optics Express 2023", "强制回答 ridge-FNO 相对既有 projection-to-volume network 的新增价值。", "https://doi.org/10.1364/OE.505992"]
      ,["最新强竞品", "Lu et al., Neural refractive index primitives, Combustion and Flame 2026", "抽 hash/Fourier、discrete gradient、3D mask、真实噪声饱和失败与 held-out view。", "https://doi.org/10.1016/j.combustflame.2026.115082"]
      ,["INR 初始化", "Tancik et al., Learned Initializations for Coordinate-Based Neural Representations, CVPR 2021", "用 time-to-quality 与 partial-observation 语言设计 NeRIF warm-start 对照。", "https://openaccess.thecvf.com/content/CVPR2021/html/Tancik_Learned_Initializations_for_Optimizing_Coordinate-Based_Neural_Representations_CVPR_2021_paper.html"]
      ,["几何感知反投影", "Liu et al., Geometry-Aware Attenuation Learning, IEEE TMI 2025", "2D encoder、geometry feature backprojection 和 3D decoder 是可变视角 BOST 输入层的直接结构参考。", "https://doi.org/10.1109/TMI.2024.3473970"]
      ,["跨视角聚合", "Lin et al., C2RV, CVPR 2024", "证明不同 view 不应等权；提取 scale-view cross-attention、跨区域一致性和少视角消融。", "https://openaccess.thecvf.com/content/CVPR2024/html/Lin_C2RV_Cross-Regional_and_Cross-View_Learning_for_Sparse-View_CBCT_Reconstruction_CVPR_2024_paper.html"]
      ,["自监督拆分", "Hendriksen et al., Noise2Inverse, IEEE TCI 2020; Sechaud et al., Equivariant Splitting, ICLR 2026", "为 support/query cameras、无完整三维真值训练和 incomplete forward model 提供方法依据。", "https://openreview.net/forum?id=upMIVpe467"]
      ,["实验流场无真值", "Mo and Magri, 3D turbulent flow reconstruction, DCE 2026", "用未见传感平面与 physics loss 做模型选择，直接服务真实 BOST 无 field GT 的验收设计。", "https://doi.org/10.1017/dce.2026.10038"]
      ,["可变传感器", "Prasthofer et al., VIDON, 2022", "相机数与位置跨样本变化时使用 permutation-invariant sensor encoder；固定几何不先上。", "https://arxiv.org/abs/2205.11404"]
      ,["operator-INR 桥", "Serrano et al., CORAL, NeurIPS 2023", "把任意采样几何与 coordinate neural field 连接起来，服务 operator warm-start NeRIF。", "https://proceedings.neurips.cc/paper_files/paper/2023/hash/df54302388bbc145aacaa1a54a4a5933-Abstract-Conference.html"]
      ,["粗解加修正", "Bhat et al., Neural Correction Operator, JCP 2026", "有限物理重建加 learned correction 的最近组件级先例；必须保留 EIT/BOST forward 边界。", "https://arxiv.org/abs/2507.18875"]
      ,["局部高频算子", "Liu-Schiaffini et al., Localized Kernels, ICML 2024", "thin-front OOD 被独立确认后比较 global FNO 与 local/global matched model。", "https://arxiv.org/abs/2402.16845"]
      ,["函数型校准", "Ma et al., Calibrated UQ for Operator Learning, TMLR 2024", "在独立 calibration set 上建立 field coverage 与 selective risk，不能在 test 上调阈值。", "https://openreview.net/forum?id=cGpegxy12T"]
    ],
    videoResources: [
      ["v3k-D 强数值基线实验室", "从零理解 residual、adjoint、投影、固定步、逐场二次步长、调用成本与 projected BB 下一闸门。", "./strong_numerical_controls_lab.html"],
      ["v3k-D 师兄 brief", "核对 selection-before-test、0/128/193-call 前沿、noise-OOD 反例与 learned scalar 阻断条件。", "./document_reader.html?doc=v3k_d_strong_numerical_controls_brief.md"],
      ["保姆式独立学习主页", "从零基础检查、16 周解锁路线、角色化论文到投稿门槛；每天只给一个当前动作。", "./operator-learning/"],
      ["v2d 同重建预算判决", "查看 direct/correction/numerical 的均值、cluster CI、p10、harm rate 与研究转向。", "./operator_fair_budget_dashboard.html"],
      ["v3a Ridge-FNO → NeRIF 执行主页", "查看 3/3 development gates、未来 7 天任务、12 周闸门、角色化论文和师兄十问。", "./ridge_fno_nerif_roadmap.html"],
      ["v3a 师兄简报", "一页说明强数值起点机制、精确结果、不能声称的内容与 M0 NeRIF warm-start。", "./document_reader.html?doc=ridge_fno_nerif_review_brief.md"],
      ["v3b-v3e 自有算法竞技场", "动态查看 v3c 负结果、96/240-epoch optimizer gates、五架构成本账本与逐域边界。", "./own_algorithm_lab.html"],
      ["v3e compute accounting brief", "查看五架构成本、PEFT 反例、FNO time-to-target 和下一轮 matched learning-curve 合同。", "./document_reader.html?doc=v3e_compute_accounting_brief.md"],
      ["v3d optimizer protocol brief", "查看 FNO epoch 冠军、plateaued control、配对尾部风险和 time-to-target 来源。", "./document_reader.html?doc=v3d_fno_optimizer_protocol_brief.md"],
      ["v3d FNO plateau brief", "查看 validation-only 规则、24→96 轨迹、为什么 +16.54% dev2 只作诊断以及当前 adapter 阻断。", "./document_reader.html?doc=v3d_fno_validation_plateau_brief.md"],
      ["v3d geometry/data manifest", "发给师兄勾选相机几何是否变化、最小数据合同、坐标约定、audit role 与 Go/No-Go。", "./document_reader.html?doc=v3d_geometry_data_manifest.md"],
      ["v3c K=6 负结果师兄 brief", "审核 adapter vs continued/base 的配对统计、计算成本、停止决定和 PEFT 强对照下一步。", "./document_reader.html?doc=v3c_k6_dev2_negative_result_brief.md"],
      ["v3d 查重与创新门槛", "逐项对照 F-Adapter/R2-FFNO/MG-TFNO/GINO，查看六个对手、三个 geometry 消融和十天执行顺序。", "./document_reader.html?doc=v3d_prior_art_and_novelty_gate.md"],
      ["v2d 师兄简报", "一页说明 0/3 通过、PILOT_ONLY 边界、K=6 max-gap 线索和下一步。", "./document_reader.html?doc=fair_camera_budget_review_brief.md"],
      ["QC-SNCO 红队审计", "先看为什么 +0.746% 仍不能称为投稿级闭环，以及下一轮必须补哪些公平对照。", "./document_reader.html?doc=qc_snco_red_team_audit.md"],
      ["T16 专项执行页", "先看推荐模型、loss、dataset split、12 周路线和失败条件。", "./operator_learning_bost_bridge.html"],
      ["T16 几何 / 零空间证据总页", "集中查看 independent dual、nullspace 上界、learned-null 负结果、query-calibrated v2c、M3B geometry/UQ 和投稿门槛。", "./operator_nullspace_evidence_dashboard.html"],
      ["算子 × 三维创新实验室", "查看 gate/dual/range-null 全部历史证据、54 篇主线文献和论文成果梯度。", "./operator_3d_innovation_lab.html"],
      ["T16 已运行代码", "打开数据、模型、训练、结果与复跑命令；不是伪代码。", "./document_reader.html?doc=demo_t16_operator%2FREADME.md"],
      ["师兄审核 brief", "把主结果、反例、不能声称的内容和七个决策压缩成会前材料。", "./document_reader.html?doc=t16_operator_smoke_review_brief.md"],
      ["三种子消融 brief", "核对 residual/absolute、物理损失、匹配容量 U-Net 和 reliability-gate 研究判断。", "./document_reader.html?doc=t16_operator_ablation_review_brief.md"],
      ["reliability + dual brief", "核对 18-run gate 负对照、3-seed dual v1、support-fit 闭式基线、router collapse 与 v2 推导。", "./document_reader.html?doc=t16_reliability_gate_review_brief.md"],
      ["12 周学习路径", "从逆问题、PyTorch 3D、FNO/DeepONet 到 OERF geometry 对接的过关产物。", "./document_reader.html?doc=t16_operator_learning_path.md"],
      ["实验规格文档", "数组形状、baseline、指标和停止条件的可复制任务书。", "./document_reader.html?doc=operator_learning_bost_experiment_spec.md"],
      ["ETH 官方课程", "Operator Learning、DeepONet、Neural Operators、FNO/CNO 四讲，带 slides 和 tutorial code。", "https://camlab.ethz.ch/teaching/deep-learning-in-scientific-computing-2023.html"],
      ["DeepONet 官方讲座", "MIT CBMM 收录 George Karniadakis 的 DeepONet 讲座和可检索 transcript。", "https://cbmm.mit.edu/video/deeponet-learning-nonlinear-operators-based-universal-approximation-theorem-operators"],
      ["ICLR 2021 FNO 页面", "官方 poster 页面含论文、slides 和会议信息。", "https://iclr.cc/virtual/2021/poster/3281"],
      ["NeuralOperator 官方教程", "PyTorch 官方生态库的 FNO/TFNO/GINO 用户指南，先跑小 Darcy/Burgers 示例。", "https://neuraloperator.github.io/dev/user_guide/index.html"],
      ["DeepXDE operator demos", "DeepONet 与 physics-informed DeepONet 的官方可运行示例。", "https://deepxde.readthedocs.io/en/latest/demos/operator.html"]
    ],
    askSenior: ["是否认可 projected BB 为 learned conditional scalar 的前置强基线，并按实际 A/A^T 调用数比较？", "能否封存从未用于 v3a-v3k-D 决策的 fresh experiment-case lock？", "组内是否能提供真实 F/J^T/VJP，或至少可微 forward loss 与 adjoint consistency 检查办法？", "组内所说的算子学习是 projection-to-volume inverse operator，还是 3D/4D flow evolution operator？", "组内相机/光线 geometry 是否跨样本变化，足以支持条件化方法？", "若 geometry 固定，未来是否需要不同 K、missing camera、标定漂移或跨装置？", "能否先提供一个脱敏样例的 displacement + geometry + mask + calibration + 单位说明？", "输入应是 raw image、displacement、projection/sinogram，还是 CGLS/SIRT/adjoint 初值？", "真实数据无 field GT 时，held-out reprojection、front location、积分量或 PIV correction 哪个最可信？", "是否接受 operator 输出粗场后只优化 zero-init NeRIF residual，并把总 time-to-quality 作为主终点？"],
    riskControl: ["不要把 neural field、PINN、FNO 和 DeepONet 混成同一个概念。", "v3k-D 的 quadratic 相对 geometry Landweber 只有约 1.24% validation 增益，却多 65 次 forward 调用；不能只报绝对误差冠军。", "noise-OOD 上 quadratic 相对 geometry Landweber 反向且高伤害，任何 learned step 必须同时报告均值、p10、harm rate 与 worst-layout。", "learned scalar 在 projected BB、fresh lock 与真实 adjoint 可用性确认前保持关闭。", "test/Q_audit 只能解释冻结结果，不能反向选择策略、epoch 或模型。", "F-Adapter 已覆盖 Fourier operator 的 frequency-adaptive PEFT，R2-FFNO/MG-TFNO 已覆盖低秩/张量化谱核；这些组件本身不能声称原创。", "operator 初始化 INR 本身已有先例，创新必须来自 BOST acquisition geometry、强数值分解、相机预算和可靠性证据。", "正确 geometry 必须胜 shuffled、constant 与 same-rank static controls，否则 geometry claim 失败。", "训练/测试必须按 phantom family、case 与 geometry 隔离，防止同分布泄漏。", "FNO 的分辨率不变性必须在 BOST 上实测，不能照抄根论文结论。", "CGLS/TV/RBF、matched U-Net、GRU-BOST、projected BB 和 per-instance NeRIF 都要有公平预算的 baseline。", "FBP-lift operator、simple gate、query router、QC-SNCO 与 frozen per-view adapter 都已触发停止条件，必须作为负结果保留。", "统计单位是独立体场，model seeds 只用于重复性，不能把 per-seed rows 当独立物理样本。", "网络推理快不等于总成本低，必须计入 ridge/CGLS、operator、NeRIF refinement、训练与摊销盈亏点。", "真实数据没有 ground truth 时，必须依靠独立 Q_audit、物理边界、risk-coverage 和组内参考结果，不能只看切片漂亮。"]
  }
];
