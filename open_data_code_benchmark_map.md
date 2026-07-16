# 公开数据与开源代码 Benchmark 路线图

用途：如果 OERF 真实 BOST / PIV-BOST / 4D BOST 数据暂时不能给，这份路线图用于保证毕业设计仍然能推进。目标不是偏离何远哲主线，而是先用公开数据和开源代码把 data loader、baseline、metrics、visualization 和 report 做完整，再迁移回组内数据。

最后核验：2026-07-10。来源包括 arXiv / Springer / Penn State Data Commons / GitHub / MathWorks / OpenPIV / Zenodo / DataCite / Optica / Crossref / Unpaywall / OpenAlex 官方页面。正式论文引用仍按 `references.bib` 和 `source_audit.md`。代码和数据复用边界另见 `code_data_reuse_audit.md`。

---

## 1. 总判断

你的公开数据策略应分七层：

1. **主 benchmark**：Open-source BOS tomography dataset of high-speed flow over a flight body。最接近“多视角 BOS/TBOS + 有标定 + 有重构结果 + 可做有限视角子采样”的需求。
2. **4D BOST 代码参考**：Hyz617/TDBOST。它直接对应 tensor decomposition 4D BOS/BOST reconstruction，可作为阅读代码结构、配置、dataloader 和 run loop 的参考。
3. **合成图像和误差仿真**：Rajendran/Vlachos 的 PIV/BOS ray-tracing synthetic image generation 与 photon GPL-3.0 代码。用于在没有 OERF 真实数据时生成实验感更强的 BOS/PIV 图像，做误差、畸变和位移估计 benchmark。
4. **图像处理练手**：OpenPIV / OpenPIV-BOS / OpenPIV Python / dot tracking / BOS uncertainty。用于学习 PIV/BOS 位移估计、互相关、图像 pair 管理、密度积分和参数扫描。
5. **光流/位移估计基础**：Horn-Schunck、Lucas-Kanade、Farneback、Brox、Barron、FlowNet、FlowNet2、PWC-Net、RAFT。用于把 PIV-BOST 图像层 benchmark 从“会跑一个方法”升级成 classical/local/global/dense/deep optical flow 的可解释对照。
6. **传统层析 baseline**：scikit-image Radon/SART、ASTRA Toolbox、ART/SART、TV minimization、LSQR 和 Kak & Slaney 教材。用于保证 NeRIF/BOST 不是只和“自己写的弱 baseline”比较，而是能对齐 FBP/SART/regularized least-squares/TV 这些反问题标准语言。
7. **方法与几何参考**：TRON-BOS、NeDF、NRIP、DoF-BOS cone-ray model、localized gradient-index BOS、NIRT flow diagnostics、Fourier Features / SIREN / NeRF / Instant-NGP / DeepSDF / Occupancy Networks / Neural Fields survey、FluidNeRF、NeuroFluid、U-Net refractive-gradient reconstruction、PINN-BOS、FlameRF、NDFRT 和 sparse PINN flame reconstruction。用于理解外部 TBOS / neural field / density tomography / 真实相机 forward model / 4D low-rank flow reconstruction 方法，但不要一开始把它们变成主线。

保底题目可以写成：

> 面向少视角 BOST 的神经隐式折射率场重构与合成/开源数据验证。

如果之后何远哲给真实数据，这套公开 benchmark pipeline 直接迁移到 OERF 数据接口。

许可判断：核心 BOST/NeRIF 代码尽量自写；MIT 项目如 PyAbel / RTNF 可以作为轻量依赖或实现参考；GPL 项目如 OpenPIV Python、photon、piv-image-generator 和 pressure-osmosis 适合作本地实验依赖或对照，不建议直接混入最终毕业代码；无 LICENSE 项目如 TDBOST、NeuroFluid、TRON-BOS 和 event_based_bos 只读结构、接口和实验设计，不复制代码。

---

## 2. 公开资源分层表

| 资源 | 类型 | 已核验入口 | 对毕设作用 | 风险/边界 |
| --- | --- | --- | --- | --- |
| Open-source BOS tomography dataset of high-speed flow over a flight body | 主 benchmark 数据集 + 代码线索 | arXiv `2508.17120`；DOI `10.1007/s00348-026-04189-z`；Data Commons DOI `10.26208/1VE2-5C19`；Penn State Data Commons README 已缓存为 `data_templates/open_bos_zip_file_content.txt`；详见 `open_bos_dataset_access_protocol.md` | 70 视角高速流 BOS；已核 12 个 zip 约 51.66 GB。新增自动解析：13 个 `Angle_*` calibration 角度 x 7 相机 = 91 个标定文件；10 个 `ROT_***` flow groups x 7 相机 = 70 个 REF/DEF image-pair views；DataCite 摘要写有 full tomography codebase，清单含 `pyscripts/NIRT.py`、`train.py`、`network.py`、MATLAB `scripts/` 和 `tools/`；已生成 `open_bos_index_summary.json`、`open_bos_view_manifest.csv`、`open_bos_view_plan.md`、`open_bos_subset_plans.json/md` 和 `figures/open_bos_view_grid.svg` | 流场是高速飞行体，不是火焰；不要全量下载进网页仓库；DataCite `rightsList` 为空，代码只读结构、不复制；ROT-to-Angle 几何映射不能从目录名硬猜；物理场和 OERF 火焰 BOST 不同，但有限视角重构问题相近 |
| Convection Over Coffee | BOS dataset 论文线索 / 教学案例 | AIAA SCITECH 2026 DOI `10.2514/6.2026-1499`，Crossref 已核 | 适合补 BOS dataset framing、热对流/coffee-steam 可视化和教学型 benchmark 语言；可作为“低门槛可视化数据集”线索继续追踪 | 本轮未找到独立 DataCite/Zenodo/OSF/PSU Data Commons 下载入口；不能把它当成已可运行的 open benchmark，也不要和上面的 70-view Penn State Open BOS tomography dataset 混淆 |
| Hyz617/TDBOST | 4D BOS/BOST tensor-decomposition 公开代码线索 | GitHub `Hyz617/TDBOST`；临时 clone 核到 `All_util/`、`TDmodel/`、`configs/`、`data/`、`dataloader/`、`render/`、`projget.py`、`run.py`、`proj.cpython...so`；README 写 4D BOS tomography 与 Google Drive spray case | 最贴何远哲 4D BOST 的公开代码参考；可学习 tensor rank、配置结构、projection/data generation、sample deflection data、spray case loader 和运行入口 | 仓库没有 `LICENSE` 文件；README 的 “open-source” 描述不等于开源许可；不能直接复制代码进毕业代码或网页；Google Drive spray case 不随仓库提供，需要问师兄这是否可作为参考 |
| TensoRF / Tensor4D / K-Planes / HexPlane / TiNeuVox / D-TensoRF | 4D tensor / plane-factor 表示根文献 | DOI `10.1007/978-3-031-19824-3_20`、`10.1109/CVPR52729.2023.01596`、`10.1109/CVPR52729.2023.01201`、`10.1109/CVPR52729.2023.00021`、`10.1145/3550469.3555383`；D-TensoRF 为 arXiv `2212.02375` | 给 4D BOST 的低秩时空表示、plane-pair decomposition、rank-memory-time 表、temporal consistency 和显存/速度权衡提供方法语言；6 篇 arXiv PDF 已缓存 | 这些是 radiance/dynamic-scene rendering 文献，不是折射率场 BOST baseline；只用于解释表示法和实验变量，不直接比较数值结果 |
| DMD / dynamic tomography / low-rank sparse metrics | 4D 时序评价和动态反问题数学工具 | Schmid DMD DOI `10.1017/S0022112010001217`；Tu DMD DOI `10.3934/jcd.2014.1.391`；sparsity-promoting DMD DOI `10.1063/1.4863670`；dynamic X-ray motion model DOI `10.1088/1361-6420/aa99cf`；DRKF DOI `10.1109/TCI.2019.2896527`；dynamic MRI DOI `10.1002/mrm.25240`；Robust PCA DOI `10.1145/1970392.1970395` | 给 M3B 增加 temporal mode、dominant frequency、low-rank residual、sparse transient、Kalman/state-space prior 和 frame-count/rank/noise 评价语言；其中 5 篇 arXiv PDF 已缓存 | 不是 BOST 论文，不作为重构 baseline；只用作“4D BOST 评估指标和低秩时序先验”的数学背景 |
| FlameRF / NDFRT / sparse neural flame reconstruction | 4D/时序神经流场方法邻居 | DOI `10.1016/j.egyai.2026.100758`、`10.1016/j.engappai.2025.112288`、`10.1016/j.combustflame.2025.114454`、`10.1016/j.combustflame.2023.113275` | 给 4D BOST 的低秩时空表示、projection supervision、temporal interpolation、sparse measurement reconstruction 提供 related work 和 demo 变量 | 多数是 flame chemiluminescence/temperature/velocity reconstruction，不是 BOST 折射率场；只做方法参考，不应当成可直接比较的 baseline |
| FluidNeRF / NeuroFluid neural-field bridge | neural radiance field / differentiable rendering 方法邻居 | AIAA FluidNeRF DOI `10.2514/6.2023-0412`；Auburn ETD `10415/9186` 在线 PDF；NeuroFluid PMLR 162:7919-7929 / arXiv `2203.01762` PDF 已缓存 | 帮助区分 NeRF-style emission/radiance-field tomography、particle-driven differentiable rendering 和 NeRIF 的 refractive-index / gradient forward model；可写成 related work 的一小段 | 不建议转成毕业主线；FluidNeRF dissertation 56.6 MB 只放在线入口不缓存；NeuroFluid 代码 license 仍需复核，不能直接搬代码 |
| Neural field / encoding foundation | 坐标神经场和编码基础文献 | Fourier Features arXiv `2006.10739` / NeurIPS；SIREN arXiv `2006.09661` / NeurIPS；NeRF DOI `10.1007/978-3-030-58452-8_24`；Instant-NGP DOI `10.1145/3528223.3530127`；DeepSDF DOI `10.1109/CVPR.2019.00025`；Occupancy Networks arXiv `1812.03828` / CVF；Neural Fields survey DOI `10.1111/cgf.14505` | 给 NeRIF/NRIP/NIRT 的 coordinate MLP、Fourier/positional/hash encoding、连续标量场、导数监督和 neural field 术语提供根文献；7 篇开放 PDF 已缓存 | 这是方法语言和编码基础，不是 BOS/TBOS 数据源，也不是 BOST baseline；阅读时只抽取表达能力、频谱偏置、编码速度和连续场表示，不比较渲染指标 |
| PIV/BOS synthetic image generation / photon | 合成图像与 ray-tracing 代码 | 论文 DOI `10.1088/1361-6501/ab1ca8`；arXiv `1812.05902`；OSTI `1598798`；GitHub `lalitkrajendran/photon` GPL-3.0 | 生成包含密度/折射率梯度、光学畸变和相机成像效果的 PIV/BOS 合成图像；非常适合无组内数据时建立可复现误差 benchmark | CUDA/C++/Python 依赖重，本科阶段不宜完整移植；可先读接口和思想，用简化 2D/3D ray model 替代 |
| MIRAGE BOS/schlieren synthetic image generation | MATLAB ray-tracing synthetic BOS/schlieren 工具链 | DOI `10.1088/1361-6501/ae4f0a`；IOP PDF 已核 CC BY 4.0 并缓存；文章题名含 MATLAB implementation of ray tracing for variable density Gradient Environments | 比 photon 更适合作为本科可读的 synthetic benchmark 参照：梳理密度/折射率场、光线追迹、BOS/schlieren image generation 和 validation cases；可把 M0/M3A 的 simple forward model 升级为 ray-tracing-like 接口 | 本轮只缓存论文 PDF；Purdue GitHub Enterprise 代码入口访问稳定性和许可仍需单独确认，不直接复制代码。MIRAGE/photon 都只能支撑 synthetic benchmark，不能代替 OERF 真实火焰结论 |
| Time-resolved synthetic raytraced BOS velocimetry | 时间序列合成 BOS / velocity benchmark | AIAA DOI `10.2514/6.2026-2293`；无明确开放 PDF 许可，只放 DOI/出版社入口 | 给 T5/T13 一个更接近动态数据的 synthetic 参照：不仅生成单帧位移，还能把 low-speed mixing flow、time-resolved image sequence 和 velocity estimate 放进 benchmark 语言 | AIAA PDF 受控，不缓存；这只是合成测速支撑，不等价于 OERF 真实 BOST 或何远哲 4D BOST |
| Single-ended projected virtual-source schlieren | 低成本/现场 schlieren 演示 | Optica Open DOI `10.1364/opticaopen.32226972`；figshare PDF 已缓存 | 用投影虚拟点光源和普通墙面减少对对侧精密光学元件的依赖；适合低成本演示、现场约束和“为什么真实实验光路麻烦”的背景 | 这是 schlieren 演示，不是 BOST/NeRIF 重构 baseline；可作为科普/实验约束旁支，不建议转成毕业主线 |
| Unsupervised BOS registration network | 位移提取 / registration 前处理方法 | Measurement DOI `10.1016/j.measurement.2026.121572`；SSRN preprint DOI `10.2139/ssrn.5797779` 已核但 PDF 返回 403 | 给 F 路线/数据健康报告增加一个学习型 displacement extraction 邻居，可和互相关、Farneback/DIS/RAFT-lite 做 benchmark 变量 | Elsevier/SSRN 当前不缓存 PDF；该条只作前处理 related work，不等于 NeRIF/BOST 三维重构 |
| BOS/Schlieren system-error support batch | 真实输入质量、光路稳定性和前处理边界 | DOI `10.1063/5.0299618`、`10.1016/bs.po.2025.02.001`、`10.1088/1361-6501/ae46c7`、`10.1364/OL.576200`、`10.1088/1361-6501/ae44b9`、`10.5139/jksas.2025.53.1.1`、`10.1117/12.3116412`；均只放官方入口 | 支撑 T6/T12/T15 的真实系统误差：detonation-exhaust velocity/density BOS、3D quantitative schlieren、plenoptic/automatic self-aligned focusing schlieren、microscale 参数敏感性、vibration correction、CNN denoising | 不缓存受控 PDF；这些条目用于定义 health report 字段和 related work，不替代 NeRIF/PIV-BOST/4D BOST 主线 |
| Focusing / self-aligned / rainbow schlieren optical-system batch | 低成本演示、光路约束、灵敏度和 focusing-schlieren synthetic benchmark | DOI `10.21203/rs.3.rs-9473921/v1`、`10.2514/6.2026-0011`、`10.2514/6.2026-0010`、`10.2514/6.2025-0238`、`10.1007/s00348-024-03951-5`、`10.1007/s10494-025-00687-y`、`10.1364/OL.511274`、`10.1364/AO.533087`、`10.1117/12.3066003`、`10.1364/AO.579510`、arXiv `2504.05433`；Research Square SAFS CC BY 4.0 PDF 已缓存 | 给 T12/T15/M3C 增加可执行变量：self-alignment、field-of-view、sensitivity、weak refractive-index gradient、turbomachinery/cascade time-resolved sequence、ray-traced focusing-schlieren image velocimetry 和 knife-edge-free colour schlieren；可把低成本可视化与 F 路线位移/速度 benchmark 接起来 | 只有 Research Square SAFS 预印本已确认可缓存；AIAA/Springer/Optica/SPIE/arXiv 其余条目只放官方入口。focusing schlieren 是光路/可视化/benchmark 支撑，不替代多视角 BOST/NeRIF 重构 |
| High-speed BOS / pulsed-illumination support batch | 高速 BOS 电影、相干结构提取、脉冲照明与同步链路 | DOI `10.2514/6.2026-3492`、`10.1364/LACSEA.2024.LTu3E.2`；均只放官方入口 | 给 T15/M3C 增加 time-resolved BOS feature extraction、wavepacket、pulsed-laser illumination、exposure/trigger 和 SAFS+BOS 对照语言 | AIAA PDF route 返回 Cloudflare challenge，HAL document route 返回 cookie-verification HTML；Optica viewmedia route 返回 202 HTML/TDM reservation。只放官方 DOI/入口，不缓存 PDF |
| BOS density validation / calibration / applied visualization batch | 标定、外部密度验证、相位/条纹处理和低成本/高速应用约束 | DOI `10.1007/s00348-024-03772-6`、`10.1088/1742-6596/2509/1/012023`、`10.3390/en17194867`、`10.3390/en16010540`、`10.3390/aerospace11080603`、`10.1364/AO.509665`、`10.1364/OL.605166`、`10.1364/OL.428011`、`10.1051/epjconf/202226401034`、`10.24425/mms.2024.148534`、`10.1364/AO.553974`；MDPI/EPJ 开放 PDF 已缓存，PAN/Optica/Springer/IOP 只放官方入口 | 给 T6/T12/T15/M3C 增加三类可执行检查：Rayleigh scattering 或 compact gas-jet density retrieval 作为外部验证语言；wedge prism / fringe projection / hidden grid 作为标定和 phase-processing 对照；fuel vaporization / plasma actuator / high-speed SAFS 作为工程场景约束 | Springer/IOP/Optica 自动 PDF 请求返回 HTML、验证页或 TDM wrapper；PAN PDF 可打开但未确认开放再分发许可，因此不缓存。该批用于 health report 和 related work，不替代 NeRIF/PIV-BOST/4D BOST |
| BOS dot tracking / uncertainty / WLS pipeline | 位移估计与不确定度工具链 | `10.1007/s00348-019-2793-3`、`10.1088/1361-6501/ab60c8`、`10.1007/s00348-020-02978-8`、`10.1007/s00348-020-03071-w`；Vlachos group code/data pages | 建立 displacement quality、uncertainty map、density integration 和 confidence weighting 的方法背景，可升级 T15 误差图谱 | 部分代码在 Purdue GitHub Enterprise/旧链接，访问稳定性不如 GitHub；正式使用前需逐个确认 license/可访问性 |
| Optical flow / displacement foundation | PIV-BOST / BOS image-layer baseline | Horn-Schunck DOI `10.1016/0004-3702(81)90024-2`；Lucas-Kanade DBLP `conf/ijcai/LucasK81`；Farneback DOI `10.1007/3-540-45103-X_50`；Brox DOI `10.1007/978-3-540-24673-2_3`；Barron DOI `10.1007/BF01420984`；FlowNet DOI `10.1109/ICCV.2015.316`；FlowNet2 DOI `10.1109/CVPR.2017.179`；PWC-Net DOI `10.1109/CVPR.2018.00931`；RAFT DOI `10.1007/978-3-030-58536-5_24` | 先做 D0 合成位移 benchmark：OpenPIV 互相关、Lucas-Kanade/Farneback、RAFT/RAFT-PIV；报告 endpoint error、bad mask、confidence 和速度误差，再决定是否接 PIV-BOST image dewarping | 光流只估二维图像位移或速度代理，不直接重构三维折射率场；Brox/Barron 作者 PDF 不本地缓存，Horn/Lucas 只放 DOI/DBLP；深度光流若无 GPU，先读方法和用轻量 baseline |
| VT-BOS pressure imaging | BOST 物理量延展 / 压力场时空测量 | Ultrasonics DOI `10.1016/j.ultras.2025.107614`；arXiv `2410.23652` PDF 已缓存 | 说明 BOS/TBOS 不只输出折射率/密度，也可服务 pressure field 和 hydrophone calibration；适合作为 related work 与“hidden physical properties”例子 | 应用场景是 MHz ultrasound / hydrophone calibration，不是 OERF 火焰；只作物理量延展和误差链背景 |
| Pyramid BOST / realtime GPU optical-flow BOS | 重构 refine 与前处理加速 | DOI `10.1007/s00348-025-04153-3`、`10.1007/s00348-026-04277-0`；Research Square 预印本 PDF 已缓存 | 如果不做完整 NeRIF，也可以做“位移场质量 + coarse-to-fine refinement + GPU 光流”小工具，贴真实实验需求 | pyramid BOST 是正式出版社入口但非开放缓存；GPU BOS 缓存的是预印本，正式引用要用 Springer DOI |
| BOS design + AI displacement | 实验设计与位移/形变估计 | DOI `10.1007/s00348-023-03602-1`、`10.1007/s00348-023-03618-7`、`10.1007/s00348-025-04058-1`；开放 PDF 已缓存 | 做一个 preprocessing report：sensitivity/blur/resolution 参数、位移估计方法、噪声/置信度、PIV image warping 的可视化 | 这些是外部方法邻居，不替代何远哲 NeRIF/PIV-BOST/4D BOST；用于准备和师兄对齐数据接口 |
| Case Western wOFA / BOS Processing Software | BOS 位移估计 GUI / 前处理工具 | AIAA DOI `10.2514/1.J060218`；OSF DOI `10.31219/osf.io/m6g4u`；Case Western FPI downloads page | 适合把 M3A 从简单位移 toy 升级到真实 BOS image-pair preprocessing：wOFA、ILS/互相关对照、背景图案选择、位移分辨率和敏感度比较 | OSF PDF metadata 明确 `No license`，本站不缓存；FPI 页面写软件可 free use 并要求引用 Schmidt and Woike (2021)，但未给出可复制/再分发的开源许可证，先作为外部工具和方法参考 |
| NASA optical-flow / ground-test BOS engineering | 真实系统设计与位移算法 QA | 12 篇 NASA/NTRS 官方 PDF 已缓存：`20180002139`、`20190000199`、`20190002831`、`20205001759`、`20170000930`、`20140011477`、`20150004105`、`20170010206`、`20170001727`、`20170011066`、`20170005509`、`20250010967` | 给 M3A/M3C 提供真实实验边界：互相关 vs optical flow、反光/自然/飞行背景、风洞/火箭/高超进气道场景、实时预览、背景图案制作和 view-quality QA | 这些是工程支撑，不是何远哲 NeRIF/PIV-BOST/4D BOST 的核心 baseline；本科只抽取系统误差、图像质量和实验设计 checklist |
| PyAbel / Abel transform validation toolchain | 轴对称 BOS / interferometry benchmark 工具 | RSI DOI `10.1063/1.5092635`；arXiv `1902.09007` PDF 已缓存；GitHub `PyAbel/PyAbel` metadata 显示 MIT license | 可做一个小 benchmark：从 axisymmetric refractive-index / density phantom 生成投影，用 PyAbel 反演，再比较 BOS vs interferometry / noise / smoothing 的误差 | 只适用于轴对称或 Abel 假设成立的验证/教学场景；不替代 3D/4D BOST 或 NeRIF，但能让 validation 章节有可运行基础 |
| Traditional TBOS arXiv baselines | 真实实验 baseline / view-count 对照 | arXiv `2503.17705` buoyant plume TBOS PDF 已缓存；arXiv `2311.10332` overexpanded supersonic jet TBOS PDF 已缓存 | 可做一个 M3C 对照表：8-camera plume vs 4-camera jet、cross-correlation/Poisson/SART、FBP/SART、camera parameter sensitivity、NPR / puffing / shock-spacing 评价 | 不是 OERF 数据，也不是 neural field；用于补传统 TBOS baseline 和真实实验误差语言 |
| Physics-informed indoor airflow BOST | single-view projector-camera / PINN-regularized BOST 邻居 | arXiv `2509.14442` PDF 已缓存；MERL technical-report PDF 为额外官方来源 | 给 M3C 提供 single-view ill-posedness、projector-camera geometry、improved ray tracing、physics-based rendering loss 和 PINN regularization 的对照材料 | 应用是室内浮力流，不是反应流；适合做 forward-model / ill-posedness / regularization 对照，不建议替代何远哲 NeRIF 主线 |
| Swin / complex-background BOS / U-Net refractive-gradient | AI 前处理与 supervised neural reconstruction | DOI `10.1364/JOSAA.487192`、`10.1364/OE.557791`、`10.1117/1.OE.64.9.094107`；TU Graz data/code DOI `10.3217/vyf5w-yqf81` | 把 M3A 从“一个位移估计器”扩成方法族对照：optical flow、CNN、transformer、复杂背景鲁棒性和 U-Net 折射率梯度重构 | Optica/SPIE PDF 不缓存；TU Graz zip 约 1.8 GB，只放入口。它们是外部邻居，不是何远哲主线 |
| DoF-BOS / localized gradient-index / NIRT flow diagnostics | M3C 真实系统误差与 neural tomography sanity check | DOI `10.2514/1.J064095`、`10.1364/AO.58.007795`、`10.1088/1361-6501/ad296a`；DoF-BOS arXiv PDF 已缓存，Optica/IOP 只放官方入口 | 把 M3C 从缺失视角扩展到真实相机 forward model：有限光圈、f-number、景深 blur、localized gradient-index 和 coordinate neural representation | 不承诺复现完整 cone-ray/NIRT；本科阶段先做简化 blur/shift/gradient-field toy 和 checklist |
| AB-PoissonPINN BOS displacement integration | BOS 前处理 / 位移积分方法邻居 | JOSA A DOI `10.1364/JOSAA.581983`；Optica 官方入口；Unpaywall/OpenAlex 标 closed | 适合做一个小而硬的 preprocessing/robustness report：Poisson 约束、adaptive loss weights、噪声条件下的位移场积分和 confidence map | 无开放 PDF，不缓存；只作为方法邻居和可复现实验灵感，不能替代 NeRIF/BOST 主线 |
| PIV operating rules / uncertainty / RAFT-PIV | PIV-BOST 图像层 benchmark | DOI `10.1016/j.optlaseng.2020.106185`、`10.1007/s00348-012-1341-1`、`10.1038/s42256-021-00369-0`、Zenodo `10.5281/zenodo.4432495` | 用传统 PIV 规则、per-vector uncertainty 和 RAFT-PIV 数据/代码设计 OpenPIV vs neural optical flow 对照 | Nature/ScienceDirect/Springer PDF 不缓存；先用链接和数据格式做路线图 |
| PIVlab / OpenPIV / piv-image-generator / SynthPix | PIV-BOST image-pair benchmark | DOI `10.5334/jors.334`、`10.1016/j.softx.2020.100585`、`10.3390/fluids8110285`、`10.1016/j.softx.2020.100537`、`10.1016/j.softx.2026.102642` | 生成 synthetic particle image pairs，跑传统互相关 baseline，再叠加 refractive distortion / dewarping，比较 raw image vs velocity-field correction | PIVlab 2021、OpenPIV-Python CPU 2023 和 SynthPix PDF 已缓存；ScienceDirect 自动 PDF 受限时只放官方页 |
| OpenPIV/bos | BOS 图像处理代码 | GitHub `OpenPIV/bos`；主体 LICENSE 为 MIT，但 `imwarp.m` 带有研究用途/非商业限制文本 | 学习 BOS 位移估计、Gladstone-Dale、stratified liquid case 的处理流程 | MATLAB 代码；不是 3D NeRIF，也不等于 OERF 九视角系统；若复制代码，需保留 license 并避开受限 `imwarp.m` |
| OpenPIV Python | PIV 图像分析工具 | GitHub `OpenPIV/openpiv-python`；OpenPIV docs；Fluids 2023 PDF 已缓存 | PIV-BOST 路线中处理粒子图像 pair、互相关窗口、速度矢量场 | 服务 PIV，不直接做 BOST tomography |
| PIV in refractive-index fields | PIV-BOST 图像层误差前史 | DOI `10.1007/s00348-019-2795-1`；Springer open PDF 已缓存 | 量化 particle-position error、systematic/random velocity error 和 ray-tracing verification；可把 M3A 从 velocity toy 升级到 image-layer bias toy | 外部基础论文，不是 OERF 作者线；不要替代何远哲 PIV-BOST 主线 |
| PIV optical distortion / calibration correction | PIV-BOST 图像层与标定层补偿 | DOI `10.1186/s41476-018-0073-0`、`10.18409/ispiv.v1i1.116`、`10.1063/5.0233759`、`10.1007/s00348-015-2004-9`、`10.1088/0957-0233/8/12/008` | 做一个小型 image-layer benchmark：先给 synthetic particle image 加空间畸变/模糊，再比较 sharpness metric、自定义 dewarp、3D disparity calibration 思路和 velocity-field correction | JEOS sharpness metrics 与 ISPIV 3D disparity calibration PDF 已缓存；AIP/IOP/Springer 条目只放 DOI/出版社入口 |
| TRON-BOS | TBOS / density tomography 代码和 MATLAB File Exchange | GitHub `3dfernando/tron-bos`；MathWorks File Exchange；论文 DOI `10.1007/s00348-025-04052-7`；GitHub API 未识别到仓库 LICENSE；其依赖 `3dfernando/pressure-osmosis` / OSMODI 为 GPLv3+ | 学习单相机/旋转 nozzle / telecentric BOS 的 density tomography pipeline | 需要 MATLAB、OSMODI solver 和 CUDA/CPU 编译；几何与 OERF 九视角 BOST 不同；只读结构和思想，复制/改代码前需确认许可 |
| TRON-BOS sample data | TRON-BOS 原始/样例数据 | Zenodo DOI `10.5281/zenodo.18118110`；license `CC BY 4.0`；`DaVis.zip` 约 9.1 GB、`Delta.mat` 约 1.38 GB、`Sinograms.mat` 约 1.28 GB，另有小型 `Inverse_Radon_DEMO.m`、`OSMODI.mexw64`、`vtkwrite.m` 和 STL | 如果想试 TRON-BOS 而不是只读代码，可用样例数据摸清 MATLAB pipeline；优先读小脚本和 metadata，暂不下载 GB 级数据 | 不是火焰，不是九视角；更适合做工程组织参考；大文件不要放进 GitHub Pages |
| Schlieren/BOS helium jet velocimetry | 2D BOS/Schlieren velocimetry 数据 + demo | Zenodo `10.5281/zenodo.6136052`；license `CC BY 4.0`；四个 run zip 约 959 MB、906 MB、1067 MB、855 MB；GitHub `alexlib/openpiv_bos_velocimetry` | 练 OpenPIV/PIVPy、速度场后处理和统计图；适合 PIV-BOST 前置训练 | 是 helium jet 2D/velocimetry，不是三维 BOST；不要把 GB 级 zip 放进 GitHub Pages |
| PIVMat | PIV/BOS 后处理工具箱 | MathWorks File Exchange `10902-pivmat-4-22` | 若组内给 DaVis/PIVlab/OpenPIV 输出，可用于读矢量场、滤波、统计和可视化 | MATLAB 工具；本身不做 PIV 求解 |
| Event-based / event-triggered BOS | 事件相机 BOS 代码、数据和 OERF 旁支 | GitHub `tub-rip/event_based_bos`；TPAMI DOI `10.1109/TPAMI.2023.3328188`；DepositOnce dataset DOI `10.14279/depositonce-19492`；蔡组 event-triggered BOS DOI `10.1364/OL.515700`、`10.1088/1361-6501/ad6172`、`10.1364/OL.545691` | 了解现代光流/事件数据工程组织、HDF5/config 结构、kFPS 级 BOS/Schlieren 采集和 event-frame fusion；说明 OERF 的 BOS 方向还有高速/低成本传感器旁支 | GitHub API 未识别到 LICENSE，README 指向 `LICENSE` 但根目录未返回该文件；只作结构阅读，复制代码前需确认许可。旁支，不建议作为本科主线 |
| TU Graz heterodyne BOS code/data split | 火焰热声振荡 heterodyne BOS 代码和数据入口 | DOI `10.3217/7k9f0-7wb03`；CC BY 4.0；`Python_Codes.zip` 约 23.7 MB、`Data_Sets.zip` 约 1.8 GB | 比 4.4 GB 合包更适合先研究代码流程，学习 Fourier/phase processing、heterodyne BOS 和 thermoacoustic oscillation 数据组织 | 仍是外部数据，不是 OERF/何远哲数据；代码包可先读，数据包不要放进 GitHub Pages |
| NeDF | sparse-view TBOS neural field 方法 | arXiv `2409.19971` | 和 NeRIF 做表示方式对照：折射率场 vs deflection field | 方法竞品，不是数据源 |
| NRIP | neural refractive index primitives 方法 | Combustion and Flame DOI `10.1016/j.combustflame.2026.115082`；arXiv `2605.11454` PDF 已缓存；GitHub `Weihu22/Neural-Implicit-Reconstruction-for-BOS` 公开可访问；专项页 `nrip_reproducibility_audit.html` | 用统一 forward model 设计 hash/Fourier、2D/3D mask、automatic/discrete gradient、5/7/9-view、noise 和 ray-budget clean-room 消融；作者仓库只用于确认输入输出和工程模块 | GitHub API 显示 no license；483 个 blob / 约 2.57 GB，含大量构建产物；README 示例参数与 argparse defaults 不完全一致。只读结构，不复制、不镜像、不把默认命令写成复现结论 |
| Single-view refractive-index tomography with neural fields | 计算视觉 neural refractive-index tomography 代码/论文 | CVPR DOI `10.1109/CVPR52733.2024.02396`；project page `https://imaging.cms.caltech.edu/rtnf/`；GitHub `brandonyzhao/rtnf` MIT License | 学 curved-ray tracing through neural fields、single-view ambiguity、source-prior 设计，用作 NeRIF forward-model 概念邻居 | 不是 BOS/燃烧论文，不应作为 BOST baseline；只做物理 forward model 与代码组织参考 |
| PINN-BOS / physics-informed BOS | 方法邻居 | JFM DOI `10.1017/jfm.2021.135` | 长期展望：从 BOS 到密度、速度、压力和物理约束 | 本科阶段不建议主攻 N-S/PINN |
| UAV weak-flow high-speed BOS | 低成本/大视场弱流 demo | Sensors DOI `10.3390/s22010043`；MDPI CC BY PDF 已缓存 | 可借鉴 fine background pattern、frequency-domain PIV correlation、Poisson density calculation、time-series baseline stacking，用于公开/低成本输入质量演示 | UAV rotor / weak airflow 不是 OERF 火焰；不要把低成本 demo 写成主毕业题，只作无真实数据时的工程训练 |
| Classical tomography / inverse baseline | 传统反问题 baseline 和可运行工具链 | ART DOI `10.1016/0022-5193(70)90109-8`；SART DOI `10.1177/016173468400600107`；Kak & Slaney 在线教材；TV-min CT DOI `10.1118/1.3560887` / arXiv PDF 已缓存；Chambolle-Pock DOI `10.1007/s10851-010-0251-1`；ASTRA DOI `10.1016/j.ultramic.2015.05.002`；scikit-image DOI `10.7717/peerj.453` / PDF 已缓存；LSQR DOI `10.1145/355984.355989` | 为 NeRIF/BOST demo 设置 FBP/SART、regularized least squares、TV 和 ASTRA/scikit-image baseline；适合 Week 4-6 先做 2D/3D phantom、sinogram、view-count/noise scan，再迁移到 BOST 位移/偏折投影 | 医学/电子层析的 Radon forward model 不是 BOST 折射率光学模型；只能作为数学 baseline 和工具链，不应写成 BOST 物理替代 |
| scikit-image Radon example | 2D tomography toy | scikit-image 官方示例 | 训练 Week 4 的 2D projection / sinogram / FBP / SART 直觉 | 只是 CT toy，不是 BOS 真实光路 |

---

## 3. 主 benchmark：Open-source BOS tomography dataset

### 公开信息

- 论文题名：Open-source BOS tomography dataset of high-speed flow over a flight body。
- 期刊：Experiments in Fluids, 2026。
- DOI：`10.1007/s00348-026-04189-z`。
- arXiv：`https://arxiv.org/abs/2508.17120`。
- Data / code DOI：`10.26208/1VE2-5C19`。
- Penn State Data Commons 目录含 12 个 zip，总量约 51.66 GB，不能放入这个 GitHub Pages 仓库，只能作为外部 benchmark 下载。
- Data Commons README 显示数据包包含 `batch/`、`cad/`、`cal_data/`、`cal_pkg/`、raw/processed images、`.mat` 标定文件等；arXiv 正文说明数据集含 70 views，并提供 NIRT、total variation regularization、data assimilation、limited-data 9-view reconstruction 等分析线索。

### 为什么最适合作为保底

- 有多视角 BOS/TBOS 数据，不是单张示意图。
- 有 calibration、flow-off/flow-on、mask、deflection estimates、3D reconstructions 和 NIRT code，可训练真实 data loader。
- 可做 view subsampling：70 views -> 9 views / 7 views / 5 views。
- 有公开论文可引用，不怕开题时被问“没有真实数据怎么办”。

### 下载边界

- 不要把 51.66 GB 数据集提交到 GitHub；只在本地或移动硬盘保留。
- 先下载 README 和最小一包，确认目录结构，再决定是否全量下载。
- 先写 `open_bos_manifest.json`、`open_bos_view_manifest.csv` 和 `load_open_bos.py`，不要一上来复现完整 NIRT。
- 开题中只承诺“公开 benchmark 预演”，不要把高速飞行体结果写成火焰物理结论。

### 和 OERF 数据的相同点

- 都是 BOS/TBOS 类型的光学位移观测。
- 都涉及有限视角重构、标定、mask、位移场和三维场。
- 都需要 data manifest、view geometry、reprojection validation。

### 和 OERF 数据的不同点

- Open dataset 是高速流/flight body；OERF 可能是火焰/反应流。
- 物理转换链不同：高速流可能关注 density/shock；火焰 BOST 更强调 refractive index/density/temperature。
- 相机/视角系统不同：公开数据的 70 views 不等于 OERF 单相机九输入端光纤束。

结论：它适合做 pipeline benchmark，不适合替代 OERF 真实物理结论。

---

## 3.1 M3C：系统误差与视角鲁棒性小 benchmark

新增 OERF CTC / endoscopic bridge 文献和 `bost_system_error_bridge.html` 后，M3C 可以被定义得更具体：不追求完整复现 NeRIF 或 4D BOST，而是做一个能服务真实实验数据接入的误差工具。

| 子问题 | 对应文献 | 可做实验 | 为什么适合本科 |
| --- | --- | --- | --- |
| 受限视场 / 坏视角 | restricted-FOV CTC, DOI `10.1364/AO.56.007107` | 随机或按几何遮挡移除视角，画 error vs missing views | 不依赖真实燃烧机理，直接服务 BOST 数据质量检查 |
| 空间分辨率指标 | spatial resolution CTC, DOI `10.1364/OE.25.024093` | 用 phantom / edge / blob 测重建分辨率，而不只报 L2 | 能写出比“图像更像”更扎实的指标 |
| 早期 CTC 实验验证 | CTC validation / practical tomography, DOI `10.1364/OE.21.007050`、`10.1364/AO.52.008106` | 比较 random view orientation 和 restricted orientation，记录 resolution/accuracy/runtime | 把 M3C 从“AI toy”拉回真实三维火焰层析指标 |
| 吸收/折射系统误差 | absorption-corrected CTC, DOI `10.1088/1361-6501/ab01c1` | 给 forward projection 加简化 bias，比较补偿前后 | 和 PIV-BOST 的误差补偿思路相通 |
| 视角选择 | camera arrangement optimization, DOI `10.1364/JOSAB.385291` | 从 70-view open BOS 子采样到 9/7/5 view，比较均匀选角和相关性选角 | 可能成为师兄真会用的小工具 |
| 光路/投影相关性 | absorption beam optimization, DOI `10.1364/OE.25.005982` | 把 TAS 的 weight-matrix row orthogonality 改写成 BOST view-subset score | 给 view selection 一个更扎实的反问题指标来源 |
| 投影模型选择 | imaging model assessment, DOI `10.1016/j.measurement.2022.112174` | 在同一 phantom 上比较 parallel-ray / pinhole / blur kernel / simplified VSF 的误差和速度 | 能把“forward model 太简化”变成可测指标，而不是口头风险 |
| 有限光圈/景深 | depth-of-field BOS cone-ray model, DOI `10.2514/1.J064095` | 用同一 refractive-index phantom 生成 thin-ray/pinhole 位移和带 blur/cone-ray proxy 的位移，比较重构偏差随 f-number / blur radius 的变化 | 很贴真实相机和背景纹影系统，比单纯调网络层数更可能对组内数据有用 |
| 局部梯度折射率场 | localized gradient-index BOS, DOI `10.1364/AO.58.007795` | 把 displacement / deflection angle 转成 localized gradient-index proxy，再比较从 `grad n` 积分到 `n` 的误差 | 帮你弄清 NeRIF 到底在拟合折射率、折射率梯度，还是位移场一致性 |
| 通用 neural tomography 表示 | NIRT for flow diagnostics, DOI `10.1088/1361-6501/ad296a` | 不复现全文，只做 related-work 表：voxel/SART、coordinate MLP、NeRIF、NILAT/NIRT 的观测模型和损失差异 | 能解释 NeRIF 的创新不是“用了 MLP”，而是把神经隐式场落到 BOS refractive-index physics |
| 透明窗口/折射边界 | transparent plates tomography, DOI `10.1016/j.optlaseng.2023.107699` | 给 forward projection 加一个简化 plate-induced shift，比较校正前后重构偏差 | 贴近真实燃烧室/光窗实验，不需要完整 CFD |
| 网格离散 | unstructured-mesh laser absorption tomography, DOI `10.1088/1361-6501/ad068f` | 用同一流场在 regular voxel 和 boundary-aware mesh 上离散，比较边界附近误差 | 能把 NeRIF/体素/网格三种表示的优缺点讲清楚 |
| mask / ROI | finite-element basis and adjustable mask, DOI `10.1364/OE.443643` | 在 phantom 外加不同 mask/ROI，比较 artifact、速度和边界误差 | 真实实验常有有效视场和遮挡，mask 处理比换模型更实际 |
| learned prior | TAS dictionary learning / NTAS deep learning, DOI `10.1364/OE.440709`、`10.1016/j.jqsrt.2018.07.011` | 只写 related-work 对照，不在本科阶段复现模型 | 解释 NeRIF 不是孤立 AI，而是 OERF learned-prior 谱系的一环 |
| 速度接口 | single-camera 3D velocimetry, DOI `10.1364/AO.58.001363` | 只做数据结构和误差指标对照，不承诺复现速度场 | 能自然连接到 PIV-BOST 速度补偿线 |

最保守交付：一个 `view_mask_experiment.csv`，三张图 `missing_view_error.png`、`resolution_phantom.png`、`view_selection_compare.png`，再加一页“哪些结论能迁移到 OERF，哪些不能”。

---

## 4. Open benchmark 的 30 天执行路线

| 时间 | 目标 | 交付 |
| --- | --- | --- |
| Day 1-3 | 下载/阅读 dataset README，不急着跑完整重构 | `open_bos_manifest.json`、`open_bos_index_summary.json`、`open_bos_view_manifest.csv`、`open_bos_subset_plans.json`、目录树截图 |
| Day 4-6 | 写 data loader，只读取 70 行 view manifest、5/7/9-view subset、calibration hint、mask 或一小份图像 | `load_open_bos.py`、`sample_views.png`、`view_grid.svg` |
| Day 7-10 | 做 70-view 到 9-view / 5-view 子采样 | `view_subset_table.csv`、`view_geometry_plot.png` |
| Day 11-14 | 接入已有 M1/M2 metrics，不追求完整 NIRT | `metrics_open_bos_baseline.csv` |
| Day 15-18 | 做 deflection / reconstruction 可视化 | `deflection_preview.png`、`reconstruction_slice.png` |
| Day 19-22 | 对比 synthetic pipeline 和 open BOS pipeline 的接口差异 | `interface_gap_table.md` |
| Day 23-26 | 写自动报告 | `open_bos_report.html` 或 `summary.md` |
| Day 27-30 | 写给何远哲看的迁移说明 | “哪些模块可迁移到 OERF，哪些必须等组内几何/单位” |

---

## 5. 开源代码怎么用

### PIV/BOS synthetic image generation / photon

用法定位：

- 这是无组内数据时最值得先读的一条工具链：用 ray tracing 从给定密度/折射率梯度场生成实验感更强的 BOS/PIV 图像。
- 2026 MIRAGE 论文是同一工具层的更近邻补充：它把 BOS/schlieren synthetic image generation 写成 MATLAB ray-tracing workflow，适合本科阶段先读公式、接口和验证案例，再决定是否做简化复现。
- 2026 AIAA time-resolved synthetic raytraced BOS velocimetry 可作为更接近动态 benchmark 的补充：它提醒 T5/T13 不只比较静态位移，还要预留 time sequence、velocity estimate 和 ground-truth flow variables。
- 对你的 M0/M1/M2 很有价值：现在 demo 里 forward model 较简化，后续可把它升级成“更接近真实相机/背景/粒子图像”的 synthetic benchmark。
- 对 PIV-BOST 很有价值：同一套思想可以生成 PIV 粒子图像和 BOS 背景图像，研究折射率梯度如何污染速度测量。

不建议：

- 不要一开始完整移植 photon 的 CUDA/C++ 依赖；本科阶段可以先把它当作方法和接口参考。
- 不要把 synthetic ray tracing 结果写成 OERF 真实火焰结论；它的价值是误差分析和 pipeline 预演。

一周任务：

- 读 `10.1088/1361-6501/ab1ca8` 的 input/output 定义。
- 读 `10.1088/1361-6501/ae4f0a` 的 MIRAGE workflow，提取哪些变量可以转成简化 Python/MATLAB toy。
- 读 `10.2514/6.2026-2293` 的题名、摘要和官方入口，确认它能给 T5/T13 提供哪些 time-resolved synthetic benchmark 变量。
- 整理 density field -> ray tracing -> BOS/PIV images -> displacement/velocity -> uncertainty 的数据流图。
- 把 M0/M3A 的 synthetic generator 改成可插拔接口：simple forward model / ray-tracing-like forward model 两档。

### OpenPIV / OpenPIV-BOS

用法定位：

- PIV-BOST 路线中用于粒子图像 pair 的互相关和速度场估计。
- BOS 图像处理练手中用于背景图案位移估计。
- Vanselow et al. 2019 的 refractive-index-field PIV 论文可作为 M3A 的图像层误差前史：先做 particle-position error，再做 OpenPIV cross-correlation，再回到速度场补偿。
- Teich et al. 2018、Michaelis et al. 2021、Bao & Lithgow-Bertelloni 2024、Elsinga & Orlicz 2015 可以把 M3A 拓展成四层误差图：image sharpness / calibration disparity / thermal-plume distortion / shock-gradient particle imaging error。
- Masker et al. 2025 的 neural optical flow for PIV 可作为进阶对照：用连续神经场 + differentiable image warping 估计 PIV 位移，但本科阶段不应跳过 OpenPIV/互相关 baseline。
- Horn-Schunck / Lucas-Kanade / Farneback / Brox / Barron / FlowNet / PWC-Net / RAFT 这一批用于把“位移估计方法族”讲清楚；新增 `optical_flow_displacement_bridge.html` 可作为 PIV-BOST image-layer benchmark 的阅读入口。
- RAFT-PIV 提供 Nature Machine Intelligence 论文、Zenodo 数据和 Code Ocean 代码入口；如果要做深度光流对照，先确认许可、数据体量和本机/GPU 能否处理。
- PIVlab 2021 的 enhanced algorithms、OpenPIV-MATLAB/Python、piv-image-generator 和 SynthPix 给 M3A 一个更稳的落地路线：先生成可控粒子图像，再用互相关得到位移/速度，最后插入折射畸变与 BOST 补偿。
- Elsinga 2005、Zha 2016、Gao/Czarske 2021/2024 这一组用于决定补偿层级：raw image dewarping / raw particle image dewarping、correlation/displacement correction、还是 exported velocity-field correction。

不建议：

- 不要指望 OpenPIV 直接完成 NeRIF 或 3D BOST。
- 不要把 OpenPIV-BOS 的 stratified liquid case 直接写成 OERF 火焰 BOST 结果。

一周任务：

- 读 OpenPIV Python tutorial。
- 处理一对 synthetic particle image，先不加折射误差。
- 加入一个简化 refractive particle-position shift，再比较 OpenPIV 输出的速度误差。
- 输出 velocity quiver、outlier removal 前后对比、position-error 到 velocity-error 的参数表。

### Pyramid BOST / realtime GPU optical-flow BOS

用法定位：

- 如果何远哲暂时不给完整 4D BOST 数据，这两类文献可以转成前处理/误差工具：先把位移估计、实时光流、coarse-to-fine refine 做扎实。
- Pyramid BOST 提供多分辨率重构、image warping、projection matrix correction 的叙事；适合解释高梯度场为什么需要逐层修正。
- Realtime GPU BOS 提供 OpenCV/GPU optical-flow 工程化思路；适合做一个小而实用的前处理 benchmark。

不建议：

- 不要把 GPU 光流前处理包装成完整 BOST 重构。
- 不要把 pyramid BOST 直接说成 NeRIF baseline；它是传统/混合 refinement 方法邻居。

一周任务：

- 给 `demo_m2/` 补一张“位移估计误差 -> 重构误差”的流程图。
- 如果有时间，做一个 CPU optical-flow vs simplified GPU/OpenCV setting 的速度/误差表。
- 问何远哲：组里更缺前处理工具、重构 refinement，还是完整 neural reconstruction demo？

### TDBOST

用法定位：

- 这是当前最贴 4D BOST / tensor decomposition 的公开代码线索。
- README 说明其目标是用 tensor decomposition 重建 4D Background Oriented Schlieren tomography；临时 clone 显示入口包括 `projget.py`、`run.py`、`configs/`、`TDmodel/`、`dataloader/`、`render/`、`All_util/`、`data/` 和编译好的 `proj.cpython...so`。
- `data/` 中有小样例和 `sample_deflection.png`，但 README 仍说明 spray case 原始数据需从 Google Drive 另取；不要把仓库里的小样例误当完整实验数据。
- 适合拿来学习项目结构、配置组织、数据生成/读取、tensor rank 和渲染结果，不应直接替代你自己的本科 demo。

不建议：

- 仓库没有 `LICENSE` 文件；README 的 “open-source” 说法不是法律意义上的可复用许可，不要直接复制代码进毕业项目或公开网页。
- 原始 spray case 数据在 Google Drive，不随仓库提供，不能假设数据能长期稳定下载。
- 先不要承诺完整复现 TDBOST；它更适合作为 M3B/M3C 的代码结构参考和会前问题来源。

一周任务：

- 只读 README、`configs/`、`run.py`、`projget.py` 和 dataloader 的输入输出。
- 整理一张 “TDBOST 模块 -> 我的 M3B/M3C 模块” 对照表。
- 问何远哲：这个仓库是否就是可参考的公开实现，是否允许借鉴结构，哪些数据/模块不能公开。

### FlameRF / NDFRT / sparse neural flame reconstruction

用法定位：

- 它们不是 OERF / 何远哲论文，但和 4D BOST 的“时空低秩表示、projection supervision、稀疏测量重构、temporal interpolation”高度相邻。
- 适合放在 related work 的最后一小段：说明你的 4D BOST 子问题不是凭空出现，而是和 TensoRF、Tensor4D、K-Planes、HexPlane、NDFRT、PINN sparse reconstruction 同处一个“计算重构流场”的方法族。
- 对 `demo_m3b/` 的价值是变量设计：rank、时间插值、噪声、投影数量、逐帧误差和时序平滑误差。

不建议：

- 不要把 FlameRF/NDFRT 写成 BOST baseline；它们观测量和物理模型不同。
- 不要在开题时承诺复现 TensoRF、K-Planes、HexPlane 或 PINN flame reconstruction。它们更适合支撑“我为什么扫 rank / temporal interpolation / sparse measurements”。

一周任务：

- 只读 abstract、method figure 和实验变量表。
- 把 M3B 的指标从单帧 RMSE 扩成 temporal consistency / interpolation RMSE / rank-memory-time 表。
- 问何远哲：4D BOST related work 里是否需要纳入 TensoRF/NDFRT 类方法，还是只写 BOST/tomography 文献即可。

### DMD / dynamic tomography / low-rank sparse metrics

用法定位：

- 这批不是公开数据，也不是 4D BOST 的竞品代码，而是 M3B 评估层的数学工具。
- DMD 适合从重构出的 `volume[t]` 或某个 slice/ROI 时间序列里提主模态、主频和模态能量，避免只看逐帧 L2。
- Dynamic X-ray tomography 的 motion model / DRKF 说明动态层析可以写成状态估计问题，适合给 frame-count、rank、temporal prior 扫描找理论背景。
- Robust PCA 和 low-rank+sparse dynamic MRI 适合把慢变背景、coherent structure 和 sparse transient 分开，给 4D BOST 的低秩张量先验加一个更清晰的误差解释。

不建议：

- 不要把 DMD/RPCA 结果写成 BOST 重构结果；它们是重构后的诊断和评估。
- 不要承诺复现医学动态 CT/MRI 算法；本科只做 metric/probe 层即可。

一周任务：

- 在 M3B synthetic phantom 上保存 `reconstruction[t, z, y, x]` 和 `truth[t, z, y, x]`。
- 对一个固定 slice 或 ROI 做 DMD，输出 top-5 mode energy、dominant frequency 和重构 residual。
- 做 low-rank+sparse 分解或简化 SVD residual 表，比较 rank、noise、frame-count 对 temporal smoothness 和 sparse residual ratio 的影响。
- 问何远哲：4D BOST 的真实评价是否需要 DMD/主频/模态能量，还是只需要重投影误差、逐帧误差和 temporal smoothness。

### FluidNeRF / NeuroFluid neural-field bridge

用法定位：

- FluidNeRF 是 NeRIF 正文会引用的 neural radiance field flow-diagnostics 邻居；AIAA 正式论文没有开放 PDF，但 Auburn ETD 有完整博士论文入口。
- NeuroFluid 是 ICML/PMLR 开放论文，使用 particle-driven neural renderer 和 state transition model 做 fluid dynamics grounding；它更偏计算机视觉/物理推理，不是 BOST。
- 这组材料只用于 related work：说明 NeRF/NeuroFluid/FluidNeRF 主要是 radiance/emission/rendering 或 particle dynamics，而 NeRIF 的关键是折射率场、折射率梯度和 BOS projection consistency。

不建议：

- 不要把 NeuroFluid 写成 OERF 或何远哲论文。
- 不要把 PhysNeRF/particle renderer 当成 BOST forward model；BOST 的观测链条是折射率梯度导致背景位移。
- 不要直接使用 NeuroFluid 代码；本轮 GitHub license 查询遇到 API 限流，正式复用前必须另核 license 和依赖。

一周任务：

- 只读 NeuroFluid 的 method figure 和 FluidNeRF dissertation 的 abstract / method overview。
- 做一张表：NeRF / FluidNeRF / NeuroFluid / NeRIF 的观测量、forward model、网络输入输出和适合的实验数据。
- 问何远哲：开题 related work 是否需要保留 NeRF 类邻居，还是聚焦 BOST/CT/BOS 文献即可。

### TU Graz U-Net refractive-gradient data/code

用法定位：

- 这是 SPIE Optical Engineering 2025 论文 `10.1117/1.OE.64.9.094107` 对应的数据/代码 DOI `10.3217/vyf5w-yqf81`。
- DataCite 记录显示 license 为 CC BY 4.0，`repository.zip` 约 1.8 GB。
- 数据结构很实用：`gradient test` 中有 100 对 numpy array 样例；`Neural Network code` 中有 `main.py`、`Evaluate_model.py` 和预训练 PyTorch 模型 `net_model.pt`。
- 它适合做 U-Net supervised reconstruction 的 loader、训练/评估脚本阅读和指标复现，不适合直接全量放进 GitHub Pages。

不建议：

- 不要把它写成 OERF/何远哲数据或 baseline。
- 不要在开题时承诺完整复现论文，只承诺读懂结构、跑通小样例和形成“数据接口模板”。

一周任务：

- 下载前先只记录文件清单、大小、license 和 DOI。
- 如果师兄认可，再本地下载 zip，抽 `gradient test` 的 100 对样例跑一次 shape/statistics/visualization。
- 把 `main.py` / `Evaluate_model.py` 的输入输出字段翻译成 `data_templates/` 里的通用 schema。

### TRON-BOS

用法定位：

- 学习 telecentric / rotating nozzle / parallel-ray density tomography 的工程组织。
- 参考 MATLAB 项目如何组织 reconstruction、solver、sample data 和 publications。
- 官方 README 明确 workflow：采集 speckle BOS 图像、用 PIV/位移代码得到 vector displacement、用 OSMODI 梯度反演得到 line-integrated density、逐层 FBP、堆叠成体密度场。
- Zenodo `10.5281/zenodo.18118110` 已核到 CC BY 4.0 和真实文件清单：`DaVis.zip` 约 9.1 GB、`Delta.mat` 约 1.38 GB、`Sinograms.mat` 约 1.28 GB；小文件包括 `Inverse_Radon_DEMO.m`、`OSMODI.mexw64`、`vtkwrite.m` 和 4-nozzle rocket STL。
- GitHub `3dfernando/tron-bos` 没有被 API 识别到 LICENSE；其依赖的 `3dfernando/pressure-osmosis` / OSMODI 声明主体为 GPLv3 or later，且 MATLAB GPU 版本依赖 CUDA/mexcuda。

不建议：

- 不要把 TRON-BOS 当成 OERF 九视角 BOST 的直接 baseline。
- 不要在没有 MATLAB/OSMODI 配置的情况下把毕业设计押在它上面。
- 不要把 GB 级 Zenodo 数据塞进网页仓库；只保留 DOI 和下载入口。
- 不要在没有 license 确认时复制 TRON-BOS 代码到自己的仓库；可以先读结构、复现实验流程图或写独立 loader。

一周任务：

- 只读 README 和论文摘要。
- 记录它的数据结构、solver 依赖、输出物理量。
- 判断哪些概念能迁移到你的 `data_templates/`。

### Event-based / event-triggered BOS / helium jet velocimetry / PIVMat

用法定位：

- `event_based_bos`：看现代光流/事件相机项目如何组织 config、dataset、HDF5 和评估；仅作旁支启发。
- 蔡组 event-triggered / event-frame schlieren 系列：`10.1364/OL.515700` 用事件触发相机 + 脉冲激光散斑投影做 heated jet kFPS 可视化；`10.1088/1361-6501/ad6172` 加入 DMD 动态投影；`10.1364/OL.545691` 融合 frame camera 和 event camera 生成自适应时间分辨率 schlieren 视频。
- `openpiv_bos_velocimetry` + Zenodo helium jet 数据：练从图像到速度/统计结果的完整小流程，适合 PIV-BOST 的前置训练。
- `PIVMat`：若组内导出 DaVis/PIVlab/OpenPIV 矢量场，提供 MATLAB 后处理、滤波、统计和可视化能力。

不建议：

- 不要把 event camera BOS 写成你的主线，除非何远哲明确要求。
- 不要把 event-triggered BOS 误写成 NeRIF/PIV-BOST 的 baseline；它更像高速采集和计算成像传感器旁支。
- 不要把 helium jet 的 2D velocimetry 结果包装成 BOST tomography。
- 不要在没有 MATLAB 环境时把 PIVMat 作为唯一工具链。

### TU Graz heterodyne BOS code/data split

用法定位：

- `10.3217/7k9f0-7wb03` 对应 Tasmany et al. 2024 heterodyne BOS 火焰热声振荡测量的数据和 Python 代码。
- 文件清单比合包友好：`Python_Codes.zip` 约 23.7 MB，可先研究；`Data_Sets.zip` 约 1.8 GB，等需要跑数据时再下。
- 它和 `10.1007/s00348-024-03890-1` 论文、`10.3217/nzz9b-rn487` 大数据入口，以及 U-Net RI-gradient 数据 `10.3217/vyf5w-yqf81` 组成同一 TU Graz thermoacoustic / HBOS 方法线。

不建议：

- 不要把代码复制进毕业项目；先看 license、引用方式和输入输出。
- 不要把 heterodyne BOS 频域处理直接写成 NeRIF baseline；它更像可复现实验和数据工程参考。

### NeDF / NRIP / PINN-BOS

用法定位：

- 作为方法邻居和相关工作。
- 用于回答开题问题：“你的 NeRIF 和外部 sparse-view/neural BOS 方法有什么关系？”
- NRIP 公开仓库可读结构：`MATLAB/` 负责 phantom / test data，`PYTHON/NIR-BOS` 负责训练；它能帮助设计自己的 loader、encoding 和 loss ablation，但不能在无 license 情况下复制代码。具体文件树、配置冲突、Mac/CUDA 门槛与第一月独立实现路线见 `nrip_reproducibility_audit.html`。

不建议：

- 不要开局同时复现 NeRIF、NeDF、NRIP 和 PINN-BOS。
- 本科主线只需要选一个最简单的对照：体素 baseline + NeRIF-style coordinate field。

---

## 6. 公开数据到 OERF 数据的接口映射

| 接口字段 | Open-source BOS dataset | OERF BOST 需要问何远哲 |
| --- | --- | --- |
| `dataset_id` | open_bos_flight_body_2026 | 组内样例编号是否可公开 |
| `modality` | BOS/TBOS | BOST / PIV-BOST / 4D BOST |
| `views` | 70 views，可子采样 | 九视角编号、角度或内窥通道映射 |
| `raw.flow_off` | reference/background images | flow-off 背景图是否可给 |
| `raw.flow_on` | disturbed/flow-on images | flow-on 火焰/热流图是否可给 |
| `calibration` | cal_data / geometry | 内参、外参、volume bounds、裁剪规则 |
| `mask` | masks 或视场区域 | 火焰区域/背景有效区 mask |
| `deflection` | deflection estimates | 位移场单位是 pixel 还是 mm |
| `reference_reconstruction` | NIRT / reconstruction results | 组内 baseline / NeRIF 切片能否给 |
| `permissions` | public | 本科论文是否能展示真实图像 |

---

## 7. 应该问何远哲的公开数据问题

1. 如果组内数据暂时不能给，是否认可我先用 open-source BOS tomography dataset 做 pipeline benchmark？
2. OERF 九视角数据和 open BOS 70-view 数据最大的接口差异是什么：view geometry、mask、位移单位，还是物理场转换？
3. 开题报告里是否可以把 open dataset 作为“无真实数据时的保底路线”写进去？
4. 若我先做 OpenPIV / OpenPIV-BOS 位移估计练习，组内真实位移估计更接近 cross-correlation、optical flow 还是 DeepFlow？
5. 如果我用 open dataset 生成公开图，真实 OERF 数据只内部验证，这样是否适合本科论文公开要求？

---

## 8. 本地文件配套

- `data_templates/open_bos_manifest.json`：公开 BOS benchmark manifest 草稿。
- `data_templates/open_bos_index_summary.json`：Data Commons 官方清单解析结果，区分 13 x 7 calibration 文件和 10 x 7 = 70 flow views。
- `data_templates/open_bos_view_manifest.csv`：70 行轻量 view manifest，不含大图像，只列路径和几何映射待确认项。
- `data_templates/open_bos_view_plan.md`：Open BOS 视角预演说明，可直接给师兄看。
- `data_templates/open_bos_subset_plans.json` / `.md`：5/7/9/13/21/70 views 的 deterministic limited-view 预案。
- `data_templates/bost_sample_manifest.json`：OERF BOST 样例 manifest。
- `data_templates/piv_bost_manifest.json`：OERF PIV-BOST 样例 manifest。
- `nerif_bost_implementation_blueprint.md`：数据接口和代码模块约定。
- `core_paper_reading_cards.md`：Open-source BOS dataset 精读卡。
- `opening_reference_shortlist.md`：开题参考文献中 open dataset 的引用位置。

---

## 9. 入口

- Open-source BOS dataset arXiv: https://arxiv.org/abs/2508.17120
- Open-source BOS dataset DOI: https://doi.org/10.1007/s00348-026-04189-z
- Open-source BOS data/code DOI: https://doi.org/10.26208/1VE2-5C19
- Penn State Data Commons directory: https://www.datacommons.psu.edu/download/engineering/molnar-et-al-open-source-bos-tomography-dataset-of-high-speed-flow-over-a-flight-body-2025/
- Penn State Data Commons README: https://www.datacommons.psu.edu/download/engineering/molnar-et-al-open-source-bos-tomography-dataset-of-high-speed-flow-over-a-flight-body-2025/000-readme-zip-file-content.txt
- TDBOST GitHub: https://github.com/Hyz617/TDBOST
- OpenPIV Python: https://github.com/OpenPIV/openpiv-python
- OpenPIV docs: https://openpiv.readthedocs.io/
- OpenPIV BOS: https://github.com/OpenPIV/bos
- TRON-BOS GitHub: https://github.com/3dfernando/tron-bos
- TRON-BOS MathWorks File Exchange: https://www.mathworks.com/matlabcentral/fileexchange/182964-tron-bos-volumetric-density-measurements-in-jets
- TRON-BOS sample data: https://doi.org/10.5281/zenodo.18118110
- Event-based BOS GitHub: https://github.com/tub-rip/event_based_bos
- Event-based BOS data DOI: https://doi.org/10.14279/depositonce-19492
- Event-based BOS arXiv: https://arxiv.org/abs/2311.00434
- Cai event-triggered BOS heated jet: https://doi.org/10.1364/OL.515700
- Cai event-triggered BOS with DMD: https://doi.org/10.1088/1361-6501/ad6172
- Cai adaptive event-frame schlieren: https://doi.org/10.1364/OL.545691
- Helium jet BOS velocimetry data: https://doi.org/10.5281/zenodo.6136052
- OpenPIV/PIVPy BOS velocimetry demo: https://github.com/alexlib/openpiv_bos_velocimetry
- PIVMat: https://www.mathworks.com/matlabcentral/fileexchange/10902-pivmat-4-22
- NeDF arXiv: https://arxiv.org/abs/2409.19971
- NRIP arXiv: https://arxiv.org/abs/2605.11454
- NRIP DOI: https://doi.org/10.1016/j.combustflame.2026.115082
- Single-view refractive-index tomography project: https://imaging.cms.caltech.edu/rtnf/
- Single-view refractive-index tomography code: https://github.com/brandonyzhao/rtnf
