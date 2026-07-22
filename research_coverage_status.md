# OERF 毕设调研覆盖度状态

生成日期：2026-07-15

用途：记录当前网页和论文库已经覆盖到哪里、哪里仍需继续挖、哪些方向只作背景不建议转主线。这个文件适合给何远哲师兄看，也适合自己后续继续增量更新时避免重复劳动。

## 当前状态

- 论文库：609 篇核心论文与阅读入口。
- 缓存开放 PDF：192 篇，均已通过 PDF 解析、哈希/来源或首页检查。
- PDF 术语索引：192 篇公开缓存 PDF 已用本地文本抽取生成页码级术语索引；191 篇命中术语，总计 53,910 次术语出现，并为 192 篇公开缓存 PDF 生成单篇“术语导读/页码地图”页，用于初学者按页定位 BOS/BOST/NeRIF/PINN/PIV/light-field/projected-background/high-speed SAFS/physics-based refractive tomography/laser-induced incandescence 等概念，支持跨论文术语反查，不复制论文正文。
- 技术术语索引：主页已独立维护 62 个术语条目，覆盖 BOST 主线、光学诊断、反问题/层析、神经场/AI、神经算子、PIV/速度场、流体/燃烧、实验/数据，并新增算子学习、神经算子、DeepONet、FNO、PINO 五个 T16 入口。
- 补课教材库：`learning_library/` 已整理 20 条开放课程/讲义/官方文档/商业教材边界，服务流体力学、燃烧、光学、反问题、计算机视觉、PyTorch 和 BOST/NeRIF 论文阅读。
- 本地私有全文队列：48 篇，包含 7 篇 P0、18 篇 P1、18 篇 P2、5 篇 P3；只用于 WebVPN/图书馆/师兄合法路径下的个人精读，不上传公开 GitHub Pages。
- 私有 PDF 入库：本机 `private_library/tools/sync_private_library.py` 已支持从 `private_library/inbox/` 或系统下载目录最近文件中按 DOI/题名匹配 PDF，生成本地私有索引和状态报告；公开库只同步状态，不同步受限 PDF 本体。
- 主线定位：何远哲 NeRIF / PIV-BOST / 4D BOST。
- 当前最稳毕业题：少视角 BOST 的神经隐式折射率场重构与鲁棒性分析。
- 当前最有价值增强副线：M3C 系统误差/真实数据质量控制，以及 F 路线 BOS/PIV-BOST 位移估计、光流 benchmark 与 input health report。
- 当前新增重点候选：T16 面向少视角 BOST 的强数值锚点、FNO 固定 epoch baseline 与后置 NeRIF 物理精化。240-epoch audit 已锁定 FNO 内部冠军与 plateaued control；v3e 五架构成本 schema 已完成，下一步是 geometry Go/No-Go 和 matched error–compute curves。
- 用户最终目标已进一步锁定为开发自有算法并公平击败 U-Net/FNO/DeepONet 等现有模型。v3b provisional ray-set 候选的历史结果只有 K=6 过开发门槛，v3c 与 v3d 已进一步证明其中相当部分收益来自 FNO 训练不足；它现在只作为负结果与方法学演化记录，不再是当前候选。
- v3c 工程协议闸门已通过：zero-init adapter 初始输出与冻结 FNO 完全一致，4,988 个新参数在第一次反向中可学习，基础 FNO 参数无漂移；328 个 dev2 fields 与 v3b sample seeds 零重叠。blind final 仅预注册、未生成种子，所以当前仍不允许确认性 superiority 表述。
- v3c K=6 dev2 三种子性能对照已完成：frozen per-view adapter 相对同 checkpoint continued FNO 为 -4.965% [95% CI -5.569%, -4.397%]，四域和三种子全负，训练/推理成本约为 7.18×/6.46×，因此停止当前 adapter。adapter 相对 base FNO 仍有 +1.425%，但 continued FNO 相对 base 为 +5.846%，暴露 24-epoch FNO 尚未饱和。
- v3d 最近邻查重已完成：F-Adapter 已覆盖 Fourier operator 的 frequency-adaptive PEFT，R2-FFNO/MG-TFNO 已覆盖低秩或张量化谱核，GINO 已使用 geometry-informed operator 概念。因此下一轮不能把低秩/frequency adapter 本身当创新，只保留 acquisition-geometry-conditioned F-Adapter 假设，并强制加入 F-Adapter、LoRA、vanilla adapter、last-block、geometry-shuffled/constant/static controls；blind final 仍关闭。
- 红队复核后 validation 改为每个三维场等权，batch contract 绑定实际 train indices。v3d-v3g 已锁定 240-epoch FNO 与 rank-48 DeepONet control。v3h 证明旧 K=6 固定布局不可辨识；v3i 生成 328 场 × 28 布局的平衡 one-field/one-geometry 私有数据。v3j 完成 4 adapter × 3 seeds × 24 epochs：static adapter 相对 locked FNO 改善 +3.6336%，但 correct geometry 相对 static/K-cardinality/shuffled 为 -0.00019%/-0.00008%/-0.01628%，均仅 1/3 种子正向且 CI 跨 0。同模型 swap 显示 embedding 会变而 correction 只变 0.1785%。因此停止当前全局 geometry modulation 扩参，v3k 先做同场多布局反事实监督；真实 geometry、superiority 与 blind final 继续阻断。
- v3k-A 已用等曝光双臂完成数据不足/结构失效拆分：M1 每场固定一布局重复 4 次，M4 每场使用 4 个不同布局；两臂各 160 个独立训练场、640 行、每布局 40 行、24 epochs、三种子和 1,023 个可训练参数。M4 correct 相对 shuffled validation 为 +0.0675% [95% field-cluster CI +0.0410%, +0.0952%]、3/3 seeds 正向，M4-M1 interaction 为 +0.0349% [+0.0126%, +0.0580%]；但两者上界均低于预注册 +0.25%。M4 descriptor swap 已令 correction 改变 8.1399%，最终场收益仅 +0.0130% [-0.0138%, +0.0391%]。结论是 one-field/one-geometry 确有小影响，但全局 geometry-to-channel modulation 的空间修正未对准场误差；停止 width/capacity search，下一步只做体素级 per-view ray-set 条件化。24 runs、576 history rows、20,160 sample rows、960 swaps、24 私有 checkpoints 和 base drift 0 已由独立 validator 对账；superiority、真实 geometry 与 blind final 继续关闭。
- v3k-B 已完成 915 参数 voxel-local ray-set 的 12-run M4 机制审计。预运行曾发现 whole-layout derangement 会偷带 base 未使用的新相机，第一种子后立即停止；正式 control 固定相同 mask、ray values、angle multiset 和预算，只循环错配 ray-angle correspondence。Correct ray-set 相对 geometry-only 为 +1.4445% [95% field-cluster CI +0.5149%, +2.3801%]、3/3 seeds 正向，证明局部观测值有用；但相对 pairing-shuffle 为 -0.1967% [-0.4023%, -0.0075%]，同模型 swap 令 correction 改变 19.0683% 而场收益为 -0.3586% [-0.6490%, -0.1108%]、3/3 seeds 负向，joint OOD 也为 -0.2867%。结论是 scalar sin/cos pairing 会驱动稳定但错误方向的修正；停止 attention width/head 搜索，v3k-C 先做数值 Landweber 与 `A_i^T(y_i-A_i x0)` per-view adjoint-residual。12 runs、288 history、10,080 sample rows、480 swaps、168 个无固定点 pairing 和 base drift 0 已独立复算；superiority 与 blind final 继续关闭。
- v3k-C 已完成 zero-new-learning adjoint/Landweber 数值闸门。红队先发现 `soft_window*ReLU(x)` 非幂等、每轮重复施加 oracle taper，以及 validation screen 在 selection commit 前计算 Q_audit；正式版已改为非负二值硬支撑投影，并让 selection function/screen/commit 完全不含 audit 或 reprojection 指标。28 布局的 adjoint identity 最大误差 1.94e-15，finite-difference gradient 最大误差 2.44e-8；40 个 validation fields × 4 layouts 的 112-grid 选中 standard `beta=1.9,T=64`。它相对 feasible FNO 的 validation 场级收益为 +44.6024% [95% CI +40.1023%, +49.1099%]，四个 development test domains 的最小平均收益为 +22.4001%；但最差 noise-OOD layout 的 >1% harm rate 为 9.375%，最差单场 -17.5799%。独立 validator 已复算 28 checks、112 screen rows、672 field-layout pairs、2,688 sample rows、20 layout cells 和 11 checksums；checkpoint drift 0。结论只是 tuned Landweber 成为必须击败的强基线；先补 global-step、closed-form line-search、lookup 和 ridge-start，再只授权 conditional scalar development prototype。confirmatory scalar、voxel preconditioner、per-view set、superiority 与 blind final 全部关闭；现有 development domains 已被前序实验查看，v3i 坐标通道命名对调也必须在新数据/重训合同中修正。
- v3k-D 已补齐 FNO/ridge 双起点、global absolute step、geometry-normalized step、投影前二次闭式步长与 validation-tuned spectral lookup。40 个 validation fields × 4 layouts 的 474-row screen 选出 FNO quadratic 为绝对精度冠军：field L2 0.143855，相对 feasible FNO +45.5258%；但其每样本需 129 A + 64 A^T，128-call 前沿仍是 FNO geometry Landweber（0.147083）。quadratic 相对后者仅 +1.2443% [95% CI +0.5282%, +1.9583%]，17.5% validation fields 伤害超过 1%，noise OOD 平均反而 -0.9128%；所以不能称同预算胜利。低/高谱区 lookup 都选回 `beta=1.9,T=64`，没有条件化价值；validation-truth oracle 仍有 1.5866% 平均 headroom、67.5% fields 的最好规则不是单一冠军，提示“起点 × 噪声 × 迭代阶段”可能有信号。独立 validator 已复算 672 pairs、8,064 sample rows、60 summaries、135 pairwise cells、10 quadratic audits、60 call-ledger rows 与 12 checksums；checkpoint drift 0。learned scalar development 现改为关闭，下一步先补 projected BB1/BB2、matched-call stopping 与 fresh lock。
- `v3d_geometry_data_manifest.md` 已建立组内接口：若相机/光线 geometry 完全固定，且课题不涉及不同 K、缺失相机、标定漂移或跨装置迁移，则 acquisition-conditioned 路线直接 No-Go，保底转向 plateau-FNO/F-Adapter benchmark、v3c 负结果和 NeRIF warm-start。
- 官方项目对齐：蔡老师中文主页列出 2025-2029 先进航空发动机非接触式测量/数据同化项目、2026-2030 数据融合实验流体力学创新研究群体，以及 2020-2023 单相机内窥 BOST 火焰折射率/密度/温度同步测量项目；你的题目应表述为服务“非接触测量 + BOST/NeRIF 重构 + 数据融合实验流体力学接口”的工具型研究。
- 何远哲作者核验：2026-07-09 ORCID public API 返回 6 组作品记录，确认 4D BOST、NeRIF、2025 PIV-BOST、iron particle heat exchange 和 NeRIF arXiv/preprint；2026 stereo PIV-BOST 与 NH3/CH4 reaction-progress 仍按 Semantic Scholar/Crossref/出版社线索处理，正式沟通时请师兄确认。

## 方向覆盖分层

| 层级 | 已覆盖内容 | 覆盖度判断 | 对毕设的处理 |
| --- | --- | --- | --- |
| P0 何远哲主线 | NeRIF、PIV-BOST、4D BOST、stereo-PIV-BOST、何远哲作者审计 | 已足够支撑开题 | 必读，决定题目和第一阶段 demo |
| P0/P1 神经算子 × 三维重建 | DeepONet、FNO、Neural Operator、PINO、NIO、RecFNO、Energy Transformer sparse-flow reconstruction、DeepONet-EIT，以及蔡组 2018-2020 projection-to-field/volume 前史 | 已形成 T16 开题脊柱 | 用于 measurement-to-volume 模型设计；蔡组前史不冒充正式 neural-operator 论文，必须保留 BOST forward physics 和 OOD 验证 |
| P1 BOST/BOS/PIV-BOST 方法邻居 | UBOST、RBF-BOST、GRU-BOST、pyramid BOST、hybrid BOST refinement、evolutionary BOST、BOS 综述、BOS 老基础、axisymmetric RI reconstruction、hot-cylinder refractive-index BOS、PyAbel / Abel transform validation toolchain、classical tomography / inverse baseline、ART/SART/FBP/TV/LSQR/ASTRA/scikit-image、Horn-Schunck / Lucas-Kanade / Farneback / Brox / Barron / FlowNet / PWC-Net / RAFT 光流位移估计根文献、localized gradient-index reconstruction、depth-of-field cone-ray model、reference-free/projection BOS、Mach 10 / 11-ft transonic TomoBOS system preparation、large-scale/high-speed BOS、buoyant plume TBOS、overexpanded supersonic jet TBOS、laser-speckle heated-jet TBOS、serpentine-nozzle TBOS、NASA optical-flow / cross-correlation BOS、retroreflective / natural-background / flight BOS、ground-test / hypersonic-inlet / rocket-engine BOS、vector tomography BOS、shock-wave / supersonic-flow TBOS、BOS vs interferometry benchmark、onboard consumer-grade BOS、Pocket Schlieren、forward-projected BOS、EBOS 多背景平均、dynamic backgrounds、event-based / event-triggered BOS、adaptive event-frame schlieren、real-time / Fourier BOS、optical-flow underwater shock BOS、free-field gas-detonation BOS、helicopter BOS velocimetry、super-temporal schlieren reconstruction、characteristics-based quantitative schlieren、DOE-BOS / hidden-grid / fringe-demodulation phase processing、lean H2-air flame tomoBOS limitations、Tomo-PIV/Tomo-BOS transonic jet、supersonic crossflow BOS、实验设计、位移/光流/CNN/speckle sensitivity、不确定度、PIV refractive-index error、PIV classical operating rules、PIV local uncertainty、PIV+BOS simultaneous velocity-density measurement、thermal-plume / shock-wave PIV optical distortion、stereo-PIV distortion compensation、non-parametric 3D disparity calibration、physics-informed BOS velocity/pressure reconstruction、single-view dynamic airflow sensing、neural optical flow / RAFT-PIV、PIV optical distortion correction、PIVlab/OpenPIV/piv-image-generator/SynthPix、NeDF/NRIP、FluidNeRF、NeuroFluid、Fourier Features / SIREN / NeRF / Instant-NGP / DeepSDF / Occupancy Networks / Neural Fields survey、TensoRF / Tensor4D / K-Planes / HexPlane / TiNeuVox / D-TensoRF、DMD / dynamic tomography / Robust PCA / low-rank+sparse temporal metrics、Single View Refractive Index Tomography with Neural Fields、plenoptic / light-field BOS、TIMes emission + refractive-index multi-modal tomography、强折射火箭羽流和火焰 HBOS 案例 | 更充分 | 用于 baseline、误差图谱、PIV-BOST 前史、数据质量控制、图像层/标定层/速度矢量层补偿、真实相机/景深/curved-ray forward-model 检查、高速/低成本传感器旁支、速度诊断/时间重建、条纹相位/背景编码、多背景降噪、事件相机融合、多模态同步测量、传统 FBP/SART/TV/LSQR 对照、optical-flow / displacement benchmark、neural field encoding 基础、NRIP-style hash encoding / 3D mask / gradient-loss 消融、4D tensor/plane-factor 表示法、DMD/low-rank+sparse 时序指标和 related work；ART/SART/FBP/TV/LSQR/ASTRA/scikit-image 只用于传统层析 baseline 和反问题语言，不等同 BOST 光学物理模型；Horn/LK/Farneback/FlowNet/PWC/RAFT 只用于图像位移估计和 PIV-BOST image-layer benchmark，不等同折射率场重构；Fourier/SIREN/NeRF/Instant-NGP/DeepSDF/Occupancy/Neural Fields、NRIP、FluidNeRF/NeuroFluid/TensoRF/DMD 系列只用于解释表示法和评估指标，不等同 NeRIF/4D BOST 物理模型 |
| P1 OERF CTC/BOST/高速体诊断前史 | 2013 CTC validation、2013 practical tomography、Frontiers CTC/FCT survey、2017 restricted-FOV CTC、spatial resolution CTC、high-spatial-resolution CTC、4 kHz CTC、annular-chamber CTC、SVLIF 高速体诊断、absorption-corrected CTC、endoscopic BOST、limited-projection VT、transfer/semi-supervised VT | 很有价值，已补齐关键根论文、开放综述、真实燃烧器几何和 AI 过渡节点 | 用于说明真实三维层析长期受 view/geometry/resolution/time-resolution/label scarcity 限制 |
| P1 OERF TAS/NTAS/非线性层析 | PECS TAS 综述、2008-2018 TAS root sweep、hyperspectral tomography、regularization parameter、POD 低维先验、controlled validation、velocity uncertainty、WMS/MUMAS、spectral-line selection、nonlinear tomography、pressure imaging、beam optimization、TAS inversion benchmark、NTAS regularization、deep learning、POD-CNN、dictionary learning、unstructured mesh LAT | 更充分 | 不转主线，只借反问题语言、正则化/低维先验、view/spectral selection、multi-parameter forward model 和 learned prior |
| P2 计算光谱/CTIS | Science/Nature 光谱仪、eLight 综述、CTIS superiorization、SRCTIS、CTIS-GAN、memristor spectrometry、二维材料可调红外探测器 / intelligent optoelectronics、LN metasurface computational spectrometer、Q-BIC metasurface diffusion-model inverse design、van der Waals junction spectrometer | 足够作课题组图谱 | 解释“硬件编码 + 计算重构”范式，不建议替代 BOST |
| P2 颗粒/全息/碳烟/火焰前沿 | 金属颗粒诊断、铁颗粒相变/燃烧建模、铁颗粒微爆、氨共燃、数字全息 DIH 粒径/三维位置/速度、聚类粒子检测、全息颗粒追踪、4D LII、U-Net/flame-front segmentation 邻居 | 更充分，足够作备选方向 | 只有师兄明确转方向时再深入 |
| P2 等离子体/材料 | Nature plasma、Nature Communications、Nature Synthesis；Nature / Nature Synthesis 两篇已用 DOE/OSTI accepted manuscript 补本地可读 PDF | 只需背景 | 不建议本科主线转入 |
| P2 LES/数据同化 | EnKF flamelet data assimilation、LDM reacting-flow modeling；`113171` DOI 已从 LES/flash-boiling 线索移出并纠正为铁颗粒相变/燃烧论文 | 初步覆盖 | 写未来展望：重构场作为 CFD/低维模型观测 |
| P3 light-field / plenoptic | plenoptic assessment、单相机 kHz 火焰体成像、single-camera endoscopic 参数研究、标定模型、10 kHz endoscopic LIF、data-driven kHz 3D flame sequence、plenoptic BOS 根论文、Sandia/OSTI plenoptic camera BOS、high-magnification long-working-distance plenoptic BOS 和 2024 Coded Optical Imaging 书章入口 | 已整理为背景桥梁，并补入 3 篇小型 OSTI PDF | 用于解释少相机/硬件编码、light-field refocusing、真实几何误差和 OERF 计算成像背景，不替代 He 主线 |

## 当前最值得继续挖的缺口

1. **OERF light-field / plenoptic imaging**  
   已新增 `light_field_single_camera_bridge.md`，把 2019 JOSA A plenoptic imaging、2019/2020 single-camera endoscopic tomography 和 10 kHz endoscopic LIF 整理成 BOST/NeRIF 背景桥梁。下一步只有在师兄明确提光场相机、内窥系统或单相机多输入端时，才继续扩系统文献。

2. **真实 BOST 数据接口与质量控制**  
   已补 Open BOS manifest/subset、BOS uncertainty、depth-of-field cone-ray model、localized gradient-index BOS、NIRT flow diagnostics、directional rays BOS、optical-system optimization、telecentric single-camera TBOS、蔡组 CTC spatial-resolution 和 view registration / endoscopic calibration 前史。本轮新增 `bost_system_error_bridge.html`，把 view geometry、mask/ROI、displacement confidence、spatial resolution、calibration/forward-model error 和数据权限整理成 M3C 质量控制路线；最新又补入 detonation-exhaust BOS velocity/density diagnosis、3D quantitative Schlieren imaging、plenoptic/automatic self-aligned focusing schlieren、microscale schlieren key-parameter analysis、BOS vibration correction 和 CNN BOS denoising，用于把真实数据健康报告从“算法误差”扩到“光路、振动、前处理、系统参数”四类。继续推进时应先问何远哲：组内真实数据最痛的是哪一层误差，以及哪些图可以写进本科论文。

3. **PIV-BOST / Stereo PIV-BOST 原始图像层补偿**
   已补 PIV refractive-index error、PIV operating rules、instantaneous local uncertainty、neural optical flow for PIV、RAFT-PIV 数据入口、Elsinga BOS-correct-PIV 老根论文、Vanselow stereo-PIV 折射误差、Zha image-layer/vector-layer 去畸变、Gao/Czarske Hartmann-Shack + CNN 畸变校正，以及 PIVlab/OpenPIV/piv-image-generator/SynthPix 工具链。本轮进一步补了 Horn-Schunck、Lucas-Kanade、Farneback、Brox、Barron、FlowNet、FlowNet2、PWC-Net、RAFT 这条光流/位移估计基础链，新增 `optical_flow_displacement_bridge.html`；最新又新增 `stereo_piv_bost_bridge.html`，把 2025 simultaneous PIV-BOST 与 2026 stereo-velocity PIV-BOST 拆成 raw image、calibration/disparity、velocity-vector 三层补偿路线，并补入 time-resolved synthetic raytraced BOS velocimetry 与 CNN BOS denoising 作为 T5 image-pair / displacement benchmark 的动态和去噪扩展。下一步如果师兄确认需要 PIV-BOST，应问清组内补偿发生在 image layer、calibration/disparity layer、correlation layer 还是 velocity-vector layer；如果暂时不给真实数据，可先做 OpenPIV/Farneback/RAFT-PIV 的合成图像 benchmark，或做 synthetic stereo-PIV distortion toy。

4. **4D BOST 的正式全文、公开代码与数据边界**
   ACM 官方 HTML/eReader 已核验可读 19 页、OA/CC BY 4.0，raw PDF 尚未本地缓存；`tdbost_reproducibility_audit.html` 已抽取 MM plane-pairs、3x128 decoder、L=3 encoding、ray/path double integration、TV+L1/boundary loss、6x200 DC、9-view synthetic、500-frame flame、100G0 消融和 13.08 GB mixed-precision 显存结果。Hyz617/TDBOST 仍只有结构参考价值，根目录无 LICENSE，并存在 `cuda:3`、Linux `.so`、绝对路径、依赖冲突及 fuel split 重叠；正文与公开 config 的 rank/decoder/test split 差异已单列。M3B 已从 low-rank toy 推进到六轴 OFAT 和 rank×noise×views×dynamics 交叉实验，用于测 temporal consistency、固定-rank regret、operating map、field/held-out/mass trade-off 和 forward-model failure signature。下一步优先问师兄版本差异、真实 rank 选择口径，以及可否拿到一小段真实 4D 数据。

## T16 算子学习新增判断

- 优先做 inverse operator：BOST displacement/projection + geometry 到三维 refractive-index field；它最贴何远哲三维重建。
- 当前开发模型锁定为 validation-tuned ridge + residual 3D FNO；FBP-lift FNO 已在三个预算给出负对照。DeepONet 暂不扩展，temporal/evolution operator 放到 NeRIF warm-start 和真实 geometry 通过后。
- 物理约束先用 BOST forward/reprojection、boundary、gradient，不在缺少速度/压力/边界条件时强行使用 Navier-Stokes loss。
- 评价必须包含 unseen phantom family、完整 view/noise condition-cell OOD、失败格、held-out view、field/积分量分叉、U-Net/SIRT baseline 和速度/显存 Pareto；M3B 合成噪声乘子须在真实数据上重标定。
- M3B temporal SVD rank 不等于 FNO Fourier modes、latent width 或 neural-operator rank；它只为静态 T16 提供实验设计反例，并在未来 temporal operator 阶段作为 baseline。
- 详细入口：`ridge_fno_nerif_roadmap.html`、`ridge_fno_nerif_review_brief.md`、`operator_learning_bost_bridge.html`、`topic_deep_dive.html?id=T16`。

### 2026-07-10 v2a-v2c 证据收束

- 独立 Residual/Absolute experts 共 86,823 参数，平均 branch disagreement 较共享 v1 提高约 59%；support-fit 在 79.17% 样本-种子单元达到或优于最佳端点。
- truth-oracle support-null audit 给出 38.626% 平均 field headroom，且剩余误差中 59.459% 能量位于 declared support nullspace；该上界使用 field truth，不可部署。
- matched 13,105 参数 free/null corrector 证明 hard null projection 把最大 support leakage 压到 2.59e-8 并稳定训练，但总体 field improvement 仍为 -0.126%。
- adaptive-query v2c 按每个样本从多个候选角中选 query，再闭式求修正幅度：+0.746% 总体 field improvement，all-query 为 +0.956%。底层是 88 个独立测试体场 × 3 个 model seeds；15 个 seed-domain 汇总均值全正不能替代逐体场统计，最大 support leakage 为 2.26e-9。
- v2d 已补齐 controlled inference 层的 `S-only / S∪Q direct / S+Q correction`、numeric query-null update、fixed/random/max-gap/adaptive query 与预锁定 Q_audit。当前 checkpoint 路径 K=4/6/8 为 0/3 通过，QC-SNCO 停止扩模型；其提出的 training-mask-matched direct inverse operator 已由下述 v3a 完成，剩余工作是 locked final fields、operator warm-start NeRIF 与独立/真实 forward。
- 独立学习入口：`operator-learning/index.html`；红队审计：`qc_snco_red_team_audit.md`；证据入口：`operator_nullspace_evidence_dashboard.html`；师兄 brief：`t16_geometry_nullspace_review_brief.md`。

### 2026-07-11 v3a direct operator 证据与主线切换

- 每个 K=4/6/8 预算独立训练，固定 60° Q_audit 不进入输入、训练、early stop 或模型选择；96 个独立测试场、3 个模型种子、5,184 条样本结果和 20,000 次四域等权 bootstrap 已由独立 validator 对账。
- ridge-FNO 相对 validation-locked ridge 的平均 field 改善为 21.54% / 19.68% / 16.91%，p10 为 4.27% / 3.19% / 0.69%，harm>1% 为 0% / 0% / 4.17%，所以三个预算都通过开发门槛。
- FBP-lift residual FNO 为 -2.16% / -16.10% / -21.53%，证明 operator 不能替代一个合格的反演底座；当前研究命题应写成“数值可观测分量 + learned residual + per-instance physics refinement”。
- 下一 M0 只做 random/ridge/ridge-FNO/oracle 四种 NeRIF 初始化，在完全相同网络、optimizer、K、stop rule 和硬件下比较 time-to-Q_audit-quality、最终质量非劣、失败率、p10 与 harm。开发门槛建议先用总时间减少 30%、最终 Q_audit 不劣 1%、p10≥-5%、harm≤10%。
- 论文创新不能只写“FNO 初始化 NeRIF”；Tancik learned initialization、STRAINER、CORAL/O-INR 已提供相邻先例。需要 BOST-specific geometry、强数值/残差分解、固定相机预算、NeRIF forward refinement、独立 nonlinear/cone-ray stress test 和真实 repeated acquisitions 才能形成较强贡献。

## 不建议继续扩大的方向

### 2026-07-17 DG-WPCGLS fresh multi-seed NO-GO

- 已从“需要真实 covariance”推进到“covariance 是否改变三维 inverse”：
  显式 detector whitener 与 BOST wrapper 保持严格 `4F+4A^T`。
- 真实 PSU detector geometry、8 类 analytic reaction morphologies、16 个
  fresh calibration/field/noise replicates 上，DG-CovGate 相对
  component-IID 的 mean field gain 为 `+1.178%`，95% CI
  `[+0.786%, +1.571%]`。
- 预注册尾部门失败：p10 `-1.029%`，>1% harm `10.94%`；annular kernel
  平均约 `-2.04%`。Oracle covariance 出现同样尾部，因此不能只继续增加
  flow-off repeats 或模型容量。
- 当前最窄研究问题改为：covariance 改变 normal spectrum 后，怎样联合选择
  fixed SPD preconditioner、whitening tempering 与 early stopping，并用
  deployment-visible signal 控制 reacting-front tail risk。
- 本机 16 种子运行约 `14.5 s`；当前网络和 Mac 算力均不是短板。真实
  flow-off repeats 与独立 session 仍是最终迁移瓶颈。

### 2026-07-17 deterministic factor-PDHG Gate B NO-GO

- Gate A 在冻结 view-local/single-scale fixture 上完成 13/13 mechanics
  attestation 后，V4 Gate B 首次真正运行 scalar、view-block、voxel-factor
  A-only PDHG 与 same-run exact-K graph-PCGLS。
- 两 replicate × 八类解析反应场 × 四 checkpoint × 四方法形成 256 条正式方法行；
  独立 validator 重算 4,048 项 checksum、调用账本、配对关系与门禁并通过。
- A-only 支持域必须拆成 2,322 个 data-coupled voxels 与 422 个 A-null support
  voxels；旧 2,744 来自 A+D connectivity，不能拿来声明数据可辨识性。
- K=32 voxel-factor 相对 scalar mean gain 仅 `+1.321%`，相对 view-block
  `+1.242%`，与 graph-PCGLS 的 mean error gap 为 `133.439%`；front-F1
  `0.1366`，graph 为 `0.7443`。八门过五门，正式判 NO-GO。
- 关闭 static factor-PDHG、TV/Huber、warm start 与 FM-CG-PDNO learned
  proximal smoke；D0 只查 exact-|A| / factor tightness 与长时轨迹，不回调性能门。
- 下一论文方向按组内数据二选一：有 aperture/focus/phantom/paired renderer
  优先 RayKernel-DCO；有 timestamp/exposure/dropout 连续序列优先 TRAIL-4D。
  这两条均需另写基线、数据角色和 fresh/real 停止门。

- 不再大规模扩 Nature/Science 光谱仪硬件论文：它们能证明课题组很强，但不帮助何远哲 BOST 定题。
- 不再把等离子体/材料合成作为主线：与用户物理基础和何远哲带教方向距离较远。
- 不再追求“所有燃烧论文都收录”：对本科毕设会稀释主线。

## 下一次和何远哲最该问的 10 个问题

1. 主线是否选 NeRIF/BOST 鲁棒性，而不是 PIV-BOST 或完整 4D BOST？
2. 真实 BOST 数据的主要痛点是 view count、noise、geometry、mask、bad view、还是 runtime？
3. 是否能给一小份九视角样例：flow-off、flow-on、mask、位移场、视角几何、参考切片？
4. baseline 应对齐组内代码、UBOST、传统体素法，还是 NeRIF 论文 baseline？
5. 是否有 spatial resolution / MTF / phantom 评价口径？
6. mask/ROI 如何进入 loss 和指标？
7. PIV-BOST 若做本科题，补偿应在图像层、互相关层，还是速度场层？
8. 4D BOST 的六轴、交叉实验与跨形态无真值 selector 已完成后，师兄更希望继续做 leave-one-geometry-out、UQ/拒答、bias/sampling 阈值，还是接真实数据？
9. 哪些真实图可以写进本科论文，哪些只能内部展示？
10. 第一周交付什么最有用：data loader、view-quality report、baseline、还是重投影验证？

## 目前网页里最该给师兄看的入口

- `he_review_feedback_pack.md`：最快速的审核入口。
- `he_review_portal.html`：最适合直接发给何远哲师兄的 5 分钟 HTML 审核入口。
- `he_meeting_decision_pack.md`：15-30 分钟会前展示顺序。
- `top_topic_proposal_cards.md`：A/B/C/E/F 与 T16 选题卡。
- `topic_deep_dive.html`：16 个可点击毕设选题详情档案。每个题目都包含研究方向摘要、通俗物理解释、执行难度、软件/硬件/知识基础、核心文献、视频/课程/软件入口、要问何远哲的问题和风险控制；主页“可选毕设题库”卡片已接入对应 T1-T16 详情。
- `oerf_latest_direction_triage.md`：把蔡老师主页最新论文分成主线升级、方法邻居和应用背景，防止选题发散。
- `oerf_source_gap_scan.md`：2026-07-10 已重新拉取 OERF `publications.ts` / `research.ts` / `members.ts`，确认官网源码仍为 2026-05-11 提交；本地库现已扩到 609 篇入口、192 个公开缓存 PDF，JFM、PECS VET、RIVR、Science vDW spectrometer 等代表作已按正式 DOI 覆盖。该页专门解释为什么不能机械照抄 OERF 源码 DOI。
- 本轮新增：Experiments in Fluids 2026 physics-informed rotating 3D velocimetry 官方开放 PDF，并继续保留 Physics of Fluids 非圆孔冲击射流 time-resolved tomo-PIV / entrainment 两篇入口、Advanced Functional Materials 2026 二维材料可调红外探测器综述，以及 4 篇 EBIV/PIV/BOS derived-field 支撑文献；AFM、EBIV、in-line PIV 和 PINN/R3DV 的开放 PDF 已缓存。它们只补流体力学/tomo-PIV、computational spectrometry/intelligent optoelectronics、事件相机测速、PIV 图像层、derived-field 和 AI flow visualization 支撑旁支，不改变 He/NeRIF/PIV-BOST/4D BOST 主线。
- `computational_flow_visualization_bridge.html`：把 computational flow visualization / hidden properties 语言转成 NeRIF、PIV-BOST、4D BOST、M3C 和 Agent for Science 辅助层的开题站位。
- `agent_for_science_bost_bridge.html`：把蔡老师主页 Agent for Science 方向压成 BOST/NeRIF 工作流辅助层，明确可做参数扫描、数据健康报告、学习日志和会前报告，不替代何远哲主线物理反演。
- `stereo_piv_bost_bridge.html`：把 2025 PIV-BOST 和 2026 stereo-velocity PIV-BOST 转成 B 路线升级、数据请求、三层补偿和本科降级 demo。
- `light_field_single_camera_bridge.md`：解释光场/单相机内窥 tomography 为什么是 BOST/NeRIF 的背景桥梁，而不是替代主线。
- `bost_system_error_bridge.html`：解释真实 BOST/PIV-BOST 数据接入前应如何检查 view geometry、mask/ROI、位移置信度、空间分辨率、标定和权限字段。
- `tdbost_decision_bridge.md`：解释 4D BOST 为什么应收缩成低秩时序/数据接口子问题，而不是本科完整复现。
- `data_request_checklist.md`：向师兄要数据的字段级清单。
- `paper_library/index.html`：609 篇论文/PDF 跳转库，其中 192 篇已缓存并通过解析校验的开放 PDF；神经算子/逆重建新增条目采用 arXiv、会议或出版社链接，不增加公开站二进制体积。每篇条目都有详情页，每篇公开缓存 PDF 都拥有单篇术语导读/页码地图页。
- `nrip_reproducibility_audit.html`：新增 NRIP 专项页，核对 Combustion and Flame 正式 DOI、arXiv、作者 GitHub 公开状态、483 个文件 blob / 约 2.57 GB 仓库体量、无 LICENSE 边界、README 与代码默认参数差异，并给出 NeRIF-vs-NRIP clean-room 消融矩阵、Mac/CUDA 路线和 7 个师兄问题。
- `tdbost_reproducibility_audit.html`：新增 4D BOST / TDBOST 专项渲染页，结合 ACM TOG 正式 HTML/eReader 全文、GitHub API 仓库健康检查和本地 M3B 确定性重跑。页面呈现方法/训练参数、synthetic/真实火焰结果、DC 消融、显存表、正文-vs-code 配置差异，并明确列出 no LICENSE/release、`cuda:3`、Linux projection `.so`、`/home/PUBLIC_USER_REDACTED/...` import-time 路径、冲突 requirements 和 fuel train/test 重叠。rank 3 mean relative L2 为 0.3471、temporal smoothness 为 0.1767，同时说明该指标定义不同于论文 Eq. 24，且低秩去抖无法自动修正共同 forward-model 系统偏差。页面现已接入六轴筛查与交叉验证，不声称完整复现 TOG。
- `m3b_six_axis_dashboard.html`：把 M3B 从单 seed rank toy 升级为 8 个配对种子、448 条方法结果的 OFAT 六轴筛查。新增全序列统一仿射对齐、论文式 squared L2、时间一/二阶误差、dynamics-energy ratio、mass trace 与 held-out deflection；rank 3 默认场 L2 改善 3.90%、时间梯度改善 44.68%、held-out 改善 9.23%，但 mass trace 恶化 0.94%、noise=0 时场误差恶化 0.56%。页面明确低秩是有条件的 denoiser，不是系统偏差校正器。
- `m3b_interaction_dashboard.html`：把 OFAT 线索推进为 rank×noise×views×dynamics 平衡交叉实验，覆盖 80 个环境、8 个配对种子、3,840 条方法记录和 3,200 组配对比较。逐格最优 rank 2/3/5/8 分别出现 27/20/24/9 次；rank 3 以 0.79% mean regret、2.21% p95 regret 成为最稳固定默认值，但有 19/80 个 field 负收益环境。116/400 个环境×rank 单元出现 field 改善而 mass trace 恶化。下一步收束为无真值 rank selector、bias magnitude、sampling rate/exposure blur 和真实数据 failure-signature 验证。
- `m3b_family_selector_dashboard.html`：把单 phantom oracle rank 推进为 4 morphology×3 dynamics×4 noise×3 views×6 seeds 的 nested-LOFO 可观测容量选择，覆盖 144 个环境、864 个观测格、6,048 个候选和 4,320 条特征消融决策。固定 rank 3 的 mean/p95 regret 为 1.561%/5.635%；无 held-out 多特征 selector 为 0.252%/1.267%，带 held-out 为 0.210%/1.054%。直接最小化 held-out residual 反而达到 2.423%/9.467%；capacity/spectrum、support、temporal 单组不如组合。独立 validator 检查 row count、unique key、outer-family exclusion、forbidden-feature absence 和 oracle/report 对账。下一步收束为 geometry transfer、UQ/拒答、真实数据与系统偏差验证。
- `demo_m3b/results/geometry_uq/`：六类 synthetic angular layout 的 nested-LOGO/UQ 已完成，覆盖 1,728 observation cells 与 12,096 candidates/selectors。无 held-out selector mean/p95 regret 为 0.273%/1.196%，93.81% 单元在 oracle 1% 内；combined UQ 失效，prediction std 的 high-risk AUC 约 0.697，但 full-rank fallback 会恶化 system risk。geometry 不是唯一主风险，morphology/dynamics/noise 更主导剩余失败。
- `he_yuanzhe_sync_dashboard.html`：新增何远哲同步主线总页，把论文矩阵、路线排序、三个月预研、1 年毕设路线、WebVPN 私有库边界和 10 个师兄问题合并到一个可分享入口。
- `he_bost_citation_spine.html`：新增 He/BOST 引用脊柱页，把 BOS sensitivity、UBOST、Direct BOST-RBF、NeRIF、NeDF/NRIP、PIV-BOST 和 4D BOST 压成一条可解释的引用链，明确 A 路线为本科主线、B/C 为数据驱动升级线。
- `he_bost_14day_reading_plan.html`：新增 14 天 He/BOST 精读执行包，把 BOS 物理观测链、传统 BOST baseline、NeRIF/NeDF/NRIP、PIV-BOST/4D BOST 升级和会前 proposal 拆成每日公式、图表、demo/动作和师兄问题。
- `first_month_experiment_board.html`：新增第一月实验执行板，把 M0-M3C、E4、数据 manifest、论文图表和师兄确认项组织成每周可交付任务，衔接 14 天精读后的实际开工。
- `code_data_reuse_dashboard.html`：公开代码与数据复用执行页，把 GitHub license、DataCite rights、Figshare/Zenodo 体量和第一周可执行动作压缩成 A-data / B-image / C-structure 三条保底路线。2026-07-09 复核后，TDBOST/NRIP/event_based_bos/TRON-BOS 等无明确 LICENSE 的仓库仍只作结构阅读；helium jet BOS 和 TRON-BOS sample data 已由 DataCite/Zenodo 确认为 CC BY 4.0，但 GB 级数据只放入口和本地 benchmark，不进入 Pages。
- `xmu_vpn_private_library_protocol.html`：新增 XMU WebVPN 与私有论文库协议。公开 GitHub Pages 只放 DOI、出版社入口和开放 PDF；通过 WebVPN/图书馆订阅访问到的 PDF 只放本地 `private_library/`，不上传公开仓库，即使仓库保持 private 也不把订阅 PDF 推到 GitHub 或 Pages。XMUM 场景优先使用资源港数据库导航的首字母筛选和“直达”入口，不走统一身份认证/CARSI；遇到验证码或二次认证就记录为需手动协助。当前私有全文队列为 48 篇，新增 heterodyne BOST reactive flows、axisymmetric micro-rocket BOS、hydrogen jet schlieren/BOS、helium coolant BOS velocimetry 和 BOS-derived velocity assimilation 等 P2/P3 旁支全文目标；本机入库工具已能处理 inbox 和系统下载目录最近文件。
- `thesis_pathway_matrix.html`：当前路线编号已统一为 A/B/C/D/F/G/H/I。F 是 BOS/PIV-BOST 位移估计与质量控制；G 是 Agent for Science 辅助 BOST 参数扫描；H 是事件相机高速可视化；I 是金属颗粒/单相机多物理量图像诊断。F 是最可立即落地的工具线，H/I 仍是旁支，不替代 He/NeRIF/PIV-BOST 主轴。
- `reproducibility_audit.html`：NeRIF / PIV-BOST / 4D BOST 主线可复现性审计，适合解释哪些能读、能跑、能公开，哪些必须等组内数据或许可。
- `pdf_cache_audit.html`：公开分享用的 PDF 可读性与缓存边界审计，解释哪些能直接缓存、哪些只能跳转。
- `source_audit.md`：DOI、题名、PDF 缓存边界和纠错记录。
