# TDBOST 模块到本科毕设 M3B/M3C 的安全映射

生成日期：2026-07-10

渲染版入口：`tdbost_reproducibility_audit.html`；M3B 统计入口：`m3b_six_axis_dashboard.html`。原表保留字段级模块映射，两个渲染页分别负责论文/代码对账与六轴实验结果。

用途：把公开仓库 `Hyz617/TDBOST` 的结构转成可向何远哲确认的学习清单。这个文件只做结构级阅读和方案映射，不复制仓库代码，不把无许可代码放进毕业设计仓库。

## 0. 已核验事实

| 项目 | 结论 |
| --- | --- |
| 4D BOST 正式论文 | `Tensor Decomposition-Based Four-dimensional Background-Oriented Schlieren Tomography for High-Speed, High-Fidelity Flow Field Reconstruction`，ACM TOG 45(5), 1-19，DOI `10.1145/3809488` |
| 正式日期 | Crossref 显示 online date 为 2026-06-29，print date 为 2026-10-31 |
| 正式全文 | ACM 标注 OA / CC BY 4.0；HTML 与 19 页 eReader 已核可读。raw PDF 未本地缓存；正式方法/结果摘要集中在 `tdbost_reproducibility_audit.html` |
| 公开代码入口 | `https://github.com/Hyz617/TDBOST` |
| 仓库许可 | 临时 clone 未见 `LICENSE` 文件；README 里的 “open-source” 不能当作可复制许可 |
| 2026-07-10 运行健康 | main tree `3393ca7`，59 blobs / 2,287,454 bytes；`cuda:3`、Linux x86-64 projection `.so`、`/home/PUBLIC_USER_REDACTED/...` import-time 路径和冲突 requirements 阻止直接跨机运行；fuel test 18 路径全部重叠 train |
| 正文/代码版本疑点 | 正文 40/40/40 vs 默认 R=30/F=20；Appendix 3x128 decoder vs config width 200；正文 held-out test view vs fuel path overlap。须由何远哲解释，不自行拼成一个配置 |
| 数据边界 | 仓库有小样例数据与 `sample_deflection.png`，但 README 仍说明 spray case 原始数据需从 Google Drive 下载 |
| 本科策略 | 只读结构、配置、数据接口和指标设计；自己写最小 M3B/M3C toy，不复刻完整 TOG / TDBOST |

## 1. 模块映射表

| TDBOST 公开模块 | 可以学习什么 | 你的安全对应产物 | 需要问何远哲 |
| --- | --- | --- | --- |
| `README.md` | 任务定义、依赖安装、数据准备、运行入口 | 写一页自己的 `M3B/M3C README`，说明输入、输出、运行命令和不可承诺范围 | 公开 TDBOST 是否可作为结构参考？是否和组内 4D BOST 代码同源或相似？ |
| `configs/config.py` | tensor rank、学习率、路径、分辨率等配置集中管理；公开默认与正文部分参数不一致 | 自己写 `config.yaml` 或 Python dataclass，记录 frame count、rank、view count、noise level、bias、loss weights 和版本来源 | 正文/仓库的 rank、F、decoder width 和 iteration 哪套对应最终实验？ |
| `projget.py` | 生成/处理投影数据的入口 | 自己写简化投影生成器：3D+T phantom -> sparse-view deflection/projection，先服务 M3B toy | 组内真实数据给的是 raw background images、deflection maps，还是已经重建的 projection tensor？ |
| `proj.cpython...so` 与 `proj.pyi` | 投影算子可能被编译加速，说明 forward model 是性能瓶颈 | 本科阶段不用复制 `.so`；先用清晰 Python/NumPy/PyTorch 版 projection sanity check | 是否需要关注速度优化，还是先把指标和接口做稳？ |
| `TDmodel/CPmodel.py` | 低秩 CP/tensor 表示思想 | 自己做低秩 SVD/CP 风格 toy，只报告 rank-error-runtime 曲线 | 4D BOST 子问题应优先做 rank selection、memory profiling，还是 temporal metric？ |
| `TDmodel/MMmodel.py` | 对应正文三组 MM plane pairs 的公开结构线索，但无 LICENSE，且不能仅凭文件名确认逐行同源 | 作为 related architecture，不直接复现；自写小型 factor baseline，并对照 rank/F/memory/quality | 师兄是否希望比较 CP/MM/Tucker，公开仓库和正式正文允许参考到什么层级？ |
| TensoRF / Tensor4D / K-Planes / HexPlane | 低秩 tensor、spacetime plane factors、dynamic-scene 表示法 | 只提取实验变量：rank、plane factor、memory、runtime、temporal consistency；不复制代码、不把 radiance rendering 当成 BOST 物理 baseline | 4D BOST related work 是否需要写这些 CV/graphics 表示法，还是只保留 BOST/tomography 文献？ |
| DMD / Robust PCA / dynamic tomography | 动态模态、低秩+稀疏分解、时序状态估计和 motion model | 自己写 `temporal_metrics.py`：DMD mode energy、low-rank residual、sparse residual ratio、centroid trajectory error、rank/noise/frame-count 表 | 组里判断 4D BOST 好坏时，是否需要 DMD/主频/模态能量这类时序诊断，还是只看逐帧重构误差和投影残差？ |
| `TDmodel/TD_Base.py` | 模型基类、forward 接口、公共参数 | 自己定义最小 `ReconstructionModel` 接口：`forward(coords, t)`、`project(view)`、`metrics()` | 真实组内代码是否有统一 model/data 接口？ |
| `TDmodel/mlp.py` | MLP 作为局部/残差表达 | M2/M3B 里保留一个小 MLP residual baseline，但不承诺完整神经场复现 | NeRIF 和 4D BOST 在组内是否共享 MLP/encoding 组件？ |
| `dataloader/BOS_dataset.py` | 数据集对象如何组织 views、frames、deflections | 自己写数据接口模板：`case_id/view_id/frame_id/image_path/deflection_path/mask/calibration` | 师兄能否给最小样例数据和字段说明？ |
| `dataloader/ray_utils.py` | ray / camera geometry helper | 做 M3C forward-model checklist：坐标系、相机参数、mask、ray validity | 真实 BOST 的相机模型是 pinhole、telecentric、endoscopic，还是自定义标定？ |
| `dataloader/data_preprocess.py` | 数据预处理、shape、归一化 | 做数据质量报告：缺帧、坏视角、mask 覆盖、位移范围、单位和坐标轴 | 组内最常见的数据错误是什么？ |
| `render/render.py` | 可视化重建结果 | 输出固定图表：slice montage、time trajectory、rank scan、reprojection error、temporal smoothness | 师兄最想看哪些图判断“有用”？ |
| `render/util/metric.py` | 指标组织；正文 Eq. 24 L2 为平方范数比，而 M3B 目前用非平方范数比 | 定义并明确命名两套 L2，另加 SSIM/PSNR、temporal smoothness、centroid、frequency、runtime/memory 与 held-out reprojection | 组里最终报告沿用哪套 L2；主频/event retention 是否需要进入评价？ |
| `All_util/trainer.py` | 训练循环组织 | 写自己的短训练循环和日志，不借用代码；重点保留 seed、config、metric CSV | 组内更看重训练可复现，还是数据报告自动化？ |
| `data/bosdata_*` 与 `sample_deflection.png` | 小样例可帮助理解输入格式 | 只用于结构阅读，不把它当作完整 benchmark；正式 benchmark 用 open BOS dataset 或组内数据 | 是否允许用公开小样例截图给师兄讨论？ |

## 2. 你的 M3B/M3C 安全路线

1. **M3B low-rank temporal toy 已完成第一轮**
   已完成 8 种子六轴 OFAT、rank×noise×views×dynamics 交叉实验，以及 4 morphology×3 dynamics×4 noise×3 views×6 seeds 的 nested-LOFO 无真值容量选择。正式包包含 864 个观测格、6,048 个候选、4,320 条特征消融决策、full-rank 拒绝平滑和独立 validator。下一步不再重复 synthetic rank scan，而是 geometry transfer、UQ/拒答或真实数据迁移。

2. **M3C data/interface tool**
   用统一 manifest 描述 view/frame/mask/calibration/deflection，输出数据质量报告。这个比“完整复现 TDBOST”更像本科生能给组里留下的工具。

3. **只读 TDBOST，不复制代码**
   读取 `README`、`configs`、`dataloader`、`render` 的结构，转成自己的接口命名和检查项。任何代码级借鉴都要先问师兄和许可边界。

## 3. 会前可以直接问师兄的 8 个问题

1. 这个公开 TDBOST 仓库是否就是组里允许参考的实现？还是只适合了解思想？
2. 我能否只参考模块划分，不复制代码？
3. 六轴、交叉 rank 与跨形态无真值 selector 已完成后，您更希望本科生做 leave-one-geometry-out、UQ/拒答、runtime/memory profiling、bias/sampling 阈值，还是直接接真实数据？
4. 真实 4D 数据给到学生时，最小字段是什么：raw images、deflection maps、camera geometry、mask、reference reconstruction？
5. 当前 4D BOST 最大痛点是速度、显存、坏视角、同步误差，还是可视化/报告？
6. 如果没有 4D 真实数据，M3B synthetic toy 是否足够作为开题预研？
7. NeRIF 与 4D BOST 是否共享前处理接口？我能否先做一个统一 loader？
8. 论文里是否可以引用 TDBOST GitHub，还是只引用 ACM TOG 正文？

## 4. 写进开题的边界句

> 4D BOST 方向将作为挑战性拓展。本科阶段不承诺复现 ACM TOG 完整方法，而是围绕低秩时序先验、跨形态可观测容量选择、数据接口和时序一致性指标构建可复现 benchmark；公开 TDBOST 仓库仅作为结构阅读和问题拆解依据，不复制无明确许可的代码。
