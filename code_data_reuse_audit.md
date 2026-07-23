# BOST / NeRIF 公开代码与数据复用审计

生成日期：2026-07-08；最近复核：2026-07-10

用途：把公开代码、数据集和项目页从“能看到”进一步分成“能直接复用、只能结构参考、数据可下载但不能进仓库、需要问师兄/作者”的几类。这个文件面向本科毕设落地，不替代各仓库/数据平台的正式 license 文本。

## 1. 总判断

最适合 Gao Youyang 现在使用的公开资源不是“越多越好”，而是按许可和落地成本分层：

1. **可直接写入独立本科代码的轻量方法参考**：PyAbel、RTNF。二者 GitHub API 显示 MIT license，适合学习实现方式，但仍应自己写 BOST/NeRIF 主线代码。
2. **可用但会影响代码许可的工具链**：OpenPIV Python、photon、piv-image-generator、pressure-osmosis/OSMODI。这些多为 GPL 或 GPLv2+/GPLv3+，若直接复制或深度集成，毕业代码仓库的开源许可要相容。
3. **数据可引用、可下载，但不放进 GitHub Pages**：TU Graz、Zenodo、TU Berlin、TRON-BOS sample data、Open BOS Data Commons。多数数据体量从数百 MB 到十几 GB，只放 DOI/入口和 manifest。
4. **只能结构参考，不能直接搬代码**：Hyz617/TDBOST、Weihu22/Neural-Implicit-Reconstruction-for-BOS、NeuroFluid、event_based_bos、TRON-BOS、openpiv_bos_velocimetry、OpenPIV MATLAB 当前 GitHub API 未识别到明确 license 或根目录无 license。
5. **对何远哲方向最有价值的落地组合**：Open BOS dataset + 自写 loader/metrics + PyAbel/RTNF 思路 + OpenPIV Python 互相关 baseline + TDBOST 结构阅读。这样既贴 He/BOST，又能避开大规模许可风险。

## 2. GitHub 仓库复用边界

| 资源 | 2026-07-10 核验状态 | 建议用法 | 不建议 |
| --- | --- | --- | --- |
| `PyAbel/PyAbel` | GitHub API 显示 MIT License；仓库含 `abel/`、`examples/`、`doc/` | 可作为轴对称 Abel/inverse Abel validation 工具；适合 M3C 小 benchmark | 不要把 Abel 反演写成 3D BOST/NeRIF 主方法 |
| `brandonyzhao/rtnf` | GitHub API 显示 MIT License；CVPR single-view refractive-index tomography 项目代码 | 可学习 curved-ray tracing through neural fields、训练入口和项目组织 | 不是 BOS/反应流实验，不应作为 BOST baseline |
| `lalitkrajendran/photon` | GitHub API 显示 GPL-3.0；仓库很大，含 CUDA / Python / sample data | 读 ray-tracing synthetic PIV/BOS 思路；可作为生成器接口设计参考 | 不要完整移植；若复制/集成代码，毕业代码许可需兼容 GPL |
| `OpenPIV/openpiv-python` | GitHub API 显示 GPL-3.0；活跃度较高 | 可作为 PIV-BOST 图像层互相关 baseline；优先通过依赖调用，不直接复制源码 | 不要把 OpenPIV 当作 NeRIF/BOST 层析算法 |
| `ElsevierSoftwareX/SOFTX_2020_33` | `piv-image-generator`；LICENSE 写 GPL v2 or later；README 指向原 GitLab | 可生成 synthetic PIV image pairs，用于 M3A 从 vector toy 升级到 image-pair toy | 若复制代码，必须按 GPLv2+ 处理；不要误写成无许可限制 |
| `OpenPIV/bos` | GitHub API license 为 `NOASSERTION`；根 `LICENSE` 为 MIT，但 `imwarp.m` 明确非商业、研究用途 | 可以学习 BOS 位移估计、Gladstone-Dale、stratified-liquid pipeline；复制前拆掉或替换 `imwarp.m` | 不要整仓搬进毕业仓库；不要忽略 `imwarp.m` 的非商业限制 |
| `Hyz617/TDBOST` | 2026-07-10 API：main tree `3393ca7`，59 blobs / 2,287,454 bytes，no license、无 release；含 configs/dataloader/TDmodel/render/run/projection。运行入口硬编码 `cuda:3`，Linux x86-64 `.so`、`/home/PUBLIC_USER_REDACTED/...` import-time 路径与冲突 requirements 使公开快照不能直接一键复现；fuel test 18 路径全部重叠 train，spray 162/18 split 才不重叠 | 只读结构：rank、config、data loader、projection、render、run loop；先做 manifest overlap/view-time coverage 审计和 clean-room M3B；详见 `tdbost_reproducibility_audit.html` | 不直接复制代码/数据，不把 README 的 clone-install-run 当已验证流程；运行和代码级复用先问何远哲与许可 |
| `Weihu22/Neural-Implicit-Reconstruction-for-BOS` | NRIP 论文作者仓库；GitHub API 显示 public / no license，最近 pushed 2026-05-11。递归 tree 核到 483 个 blob、约 2.57 GB：`PYTHON/NIR-BOS` 91 个文件约 93 MB，`MATLAB` 149 个文件约 127 MB，`C++` 198 个文件约 2.35 GB；根目录含 `C++/`、`Common/`、`MATLAB/`、`PYTHON/`、`readme.txt`，但唯一命中 `license.txt` 的是第三方子目录文件，不是根许可 | 先读 `nrip_reproducibility_audit.html`；可读 MATLAB phantom/data generation、Python NIR-BOS 训练、hash/Fourier encoding、2D/3D mask、discrete/auto-diff gradient 和 held-out-view 组织，再独立实现小规模消融 | 不要镜像仓库或直接复制代码；README 示例用 10k iters / 2e-2 / 256 rays / 256 steps，而 argparse defaults 是 30k / 1e-2 / 4096 / 1024，不能把默认运行结果冒充论文复现；CUDA/MATLAB/C++ 全栈也不应作为 Mac 上开题 gate |
| `syguan96/NeuroFluid` | GitHub API 显示 no license；仓库含 configs、data_generation、models、trainer | 用于理解 neural radiance field / fluid dynamics grounding 的工程结构 | 不要复用代码；它不是 NeRIF/BOST baseline |
| `tub-rip/event_based_bos` | GitHub API 显示 no license；README 链接 paper/video/dataset；数据 DOI 为 TU Berlin DepositOnce | 可学习 event/BOS 数据结构、HDF5/config/run loop；数据本身 CC BY 4.0 | 不建议作为本科主线；代码复制前必须问许可 |
| `3dfernando/tron-bos` | GitHub API 显示 no license；根目录含 MATLAB demo、OSMODI mex、VTK writer | 读单相机/旋转 nozzle/telecentric BOS tomography 的 pipeline | 不要直接复制；依赖 license 更复杂 |
| `3dfernando/pressure-osmosis` | GitHub API 显示 `NOASSERTION`；`license.txt` 写主体 GPLv3 or later，含 MIT 派生文件 | 如果研究 pressure-from-PIV / OSMODI，可作为 GPL 工具链了解 | 不要把 OSMODI 依赖默默并入非 GPL 仓库 |
| `OpenPIV/openpiv-matlab` | GitHub API 未识别 license；README 指向 Figshare / OpenPIV 项目 | 适合对照 PIVlab/OpenPIV 工作流和 MATLAB 后处理 | 复制代码前需另核 Figshare / 文件级许可 |
| `alexlib/openpiv_bos_velocimetry` | GitHub API 显示 no license；README 指向 helium jet paper 和 Zenodo 数据 | 可读 notebooks，学习 OpenPIV/PIVPy BOS velocimetry demo | 不要复制 notebook 代码到公开仓库，除非作者补 license |

## 3. 数据集与文件体量边界

| 数据 / DOI | 核验结果 | 可做什么 | 边界 |
| --- | --- | --- | --- |
| Open-source BOS tomography dataset, DOI `10.26208/1VE2-5C19` | DataCite 显示 Penn State Data Commons dataset，resource type 为 Dataset；摘要写有 full tomography codebase；rightsList 当前为空；本地已解析 12 个 zip 约 51.66 GB，其中 `pyscripts/` 12 个文件含 `NIRT.py`、`train.py`、`network.py`，另有 MATLAB `scripts/` 和 `tools/` | 主 benchmark：70-view BOS/TBOS、5/7/9-view 子采样、loader/manifest/可视化；可读 NIRT 代码结构，学习变量命名和 pipeline | rightsList 空，不把数据或代码镜像进 GitHub；先只下载 README/小子集；代码只读结构、不复制 |
| TU Graz U-Net RI-gradient data/code, DOI `10.3217/vyf5w-yqf81` | DataCite 显示 CC BY 4.0；repository.zip 约 1.8 GB，含 100 对 numpy 样例、`main.py`、`Evaluate_model.py`、`net_model.pt` | 可做 supervised U-Net refractive-gradient reconstruction 的 loader 与指标复现 | 数据大，不进网页仓库；代码复用仍需保留引用和 license |
| TU Graz neural flame-field scripts, DOI `10.3217/e0th6-few77` | DataCite 显示 CC BY 4.0；约 769 MB | 学 local flame-field neural reconstruction 的数据组织 | 方法邻居，不是 BOST 主线 |
| TU Graz HBOS reactive-flow data, DOI `10.3217/nzz9b-rn487` | DataCite 显示 CC BY 4.0；约 4.4 GB | 了解 heterodyne BOS reactive-flow 数据组织 | 数据太大，只放入口 |
| TU Graz HBOS code/data split, DOI `10.3217/7k9f0-7wb03` | DataCite 显示 CC BY 4.0；`Python_Codes.zip` 约 23.7 MB，`Data_Sets.zip` 约 1.8 GB | 比 4.4 GB 合包更适合先读 Fourier/phase processing 代码 | 仍是外部火焰 HBOS，不替代 OERF 数据 |
| TU Berlin event-based BOS dataset, DOI `10.14279/depositonce-19492` | DataCite 显示 CC BY 4.0 | 可学习 event-camera BOS 数据组织 | 代码仓库无明确 license，数据和代码分开处理 |
| Helium jet BOS velocimetry, DOI `10.5281/zenodo.6136052` | Zenodo / DataCite 均显示 CC BY 4.0；5 个文件约 3.53 GiB / 3.79 GB，其中 4 个 run zip 约 0.80-0.99 GiB | 练 OpenPIV/PIVPy、BOS image-pair 和 velocity statistics | 2D helium jet，不是 3D BOST |
| TRON-BOS sample data, DOI `10.5281/zenodo.18118110` | Zenodo / DataCite 均显示 CC BY 4.0；13 个文件约 15.19 GiB / 16.31 GB，含 `DaVis.zip`、`Delta.mat`、`Sinograms.mat` 和小 MATLAB 文件 | 若想跑 telecentric BOS tomography pipeline，可先读小脚本和 metadata | 主数据太大；TRON-BOS 代码 license 不清 |
| RAFT-PIV dataset, DOI `10.5281/zenodo.4432495` / `10.5281/zenodo.4432496` | Zenodo license `cc-by-4.0`；单 zip 约 12.2 GB | 深度 PIV / optical flow 数据格式参考 | 大且偏 PIV，不要作为 BOST 主线依赖 |

## 3.1 2026-07-10 补充复核

- He 作者谱系：Semantic Scholar author `2322747160` 当前返回 `paperCount = 8`；本地论文库已覆盖 6 条 OERF/流体燃烧相关条目，未入库的 2 条为医学影像方向索引噪声，不进入 BOST/NeRIF 主线。
- GitHub 代码许可：`Hyz617/TDBOST`、`Weihu22/Neural-Implicit-Reconstruction-for-BOS`、`tub-rip/event_based_bos`、`3dfernando/tron-bos`、`alexlib/openpiv_bos_velocimetry` 仍未见可直接复用的明确 LICENSE；继续只读结构、不复制代码。NRIP 仓库已确认公开可访问，不再写“链接不可达”，但公开访问并没有补齐许可。
- 可直接参考的轻量开源代码：`PyAbel/PyAbel` 和 `brandonyzhao/rtnf` 仍为 MIT；`OpenPIV/openpiv-python` 为 GPL-3.0，适合作为外部依赖或工具学习，不适合复制源码进非 GPL 毕设仓库。
- DataCite/Zenodo 数据许可：helium jet BOS velocimetry DOI `10.5281/zenodo.6136052` 和 TRON-BOS sample data DOI `10.5281/zenodo.18118110` 均显示 Creative Commons Attribution 4.0 International / `cc-by-4.0`。二者可作为本地 benchmark 数据入口，但文件分别约 3.53 GiB 和 15.19 GiB，不放进 GitHub Pages。

## 4. 最推荐的本科落地顺序

### 第 1 档：立刻可做

1. **自写 BOST/NeRIF toy 代码**：保留当前 M0-M3B 的自写路线，避免许可污染。
2. **Open BOS manifest / subset 计划**：只用已生成的 `open_bos_view_manifest.csv`、`open_bos_subset_plans.md`，先不下载 51.66 GB 全量数据。
3. **PyAbel validation mini-benchmark**：用 MIT 工具或自己写简化 Abel transform，做 axisymmetric phantom sanity check。
4. **OpenPIV Python 作为外部依赖**：若做 PIV-BOST image-pair toy，可以在环境中安装 OpenPIV，而不是复制源码。

### 第 2 档：一周内可调研

1. **TDBOST 结构阅读**：写 “TDBOST module -> 我的 M3B module” 表，不复制代码。
2. **RTNF curved-ray toy**：只借鉴 MIT 项目中的组织思路，做一个极简 curved-ray / refractive field forward-model sanity check。
3. **piv-image-generator / SynthPix 对照**：前者 GPLv2+，后者当前只确认 arXiv 论文；先写需求清单，不急着集成。

### 第 3 档：等师兄确认后再动

1. **组内 TDBOST / 4D BOST 代码是否可看**：公开 TDBOST 无 LICENSE，必须问。
2. **真实 BOST/PIV-BOST 数据能否公开展示**：即使能内部读，也未必能放 GitHub Pages。
3. **是否允许使用 GPL 依赖**：如果最终代码要公开，GPL 依赖会影响整体许可策略。

## 5. 给何远哲的新增问题

1. 公开 `Hyz617/TDBOST` 是否和组内 4D BOST 工作有直接关系？我能否只借鉴模块结构和配置命名？
2. 如果我做 PIV-BOST image-pair toy，师兄更希望我使用 OpenPIV Python、PIVlab，还是自己写最小互相关 baseline？
3. 毕设代码未来是否需要公开？如果需要公开，是否能接受 GPL 依赖，还是必须尽量 MIT/BSD/自写？
4. Open BOS 70-view dataset 是否适合作为外部 benchmark？如果不适合，组内有没有更接近火焰 BOST 的匿名小样例？
5. 对师兄最有用的不是“复现别人代码”，而是 loader、quality report、参数扫描和图表自动化，对吗？

## 6. 结论

最稳策略是：**核心 BOST/NeRIF 代码自写，公开代码只作参考或依赖；公开数据只作外部 benchmark，不塞进网页仓库；任何无 LICENSE 仓库只读结构，不复制代码。**

这样做的好处是，你的毕业设计既能持续推进，又不会被版权、许可、数据体量和组内保密边界卡住。
