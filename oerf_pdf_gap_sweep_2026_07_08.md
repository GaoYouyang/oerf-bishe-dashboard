# OERF / He 重点论文 PDF 缺口复核

生成日期：2026-07-08

> **2026-07-10 状态更新：** 4D BOST 已从“只有 DOI/摘要”推进为“ACM 官方 HTML 全文与 19 页 eReader 已核验可读”，ACM 标注 Open access / CC BY 4.0；但无会话 raw PDF 下载仍返回 403，所以本地公开 PDF 数量中仍不计这一篇。最新方法、实验、显存、代码差异和 M3B 结果见 `tdbost_reproducibility_audit.html`。下文保留 7 月 8 日访问测试作为历史记录，凡与本提示冲突者以本提示为准。

用途：继续执行“能下载的开放 PDF 都尽量缓存”的要求，同时把不能缓存的核心论文边界说清楚。这个文件只记录公开网页、出版社/Crossref 元数据和命令行访问结果；正式精读全文应通过学校网络、出版社页面、图书馆或师兄提供的合法版本。

## 总判断

本轮优先复核何远哲主线、OERF BOST 前史、CTC/内窥层析、4D LII、颗粒/全息旁支，以及后续追加的 PIV-BOST 图像层/标定层补偿邻居中“重要但未缓存”的条目。结果是：

1. **何远哲主线仍没有新的本地可公开缓存 PDF 文件**；但 4D BOST 已有 ACM HTML/eReader 正式全文直读，不再是摘要层材料。PIV-BOST 和 stereo-PIV-BOST 仍主要依赖出版社/学校/师兄合法入口。
2. **PIV-BOST 补偿前史追加发现 2 个可稳定缓存 PDF**：JEOS/EDP Sciences 的 adaptive PIV sharpness metrics，以及 ISPIV 2021 的 non-parametric 3D disparity calibration。
3. 多个 Springer 条目命令行请求 `content/pdf/...pdf` 会回到 HTML article page，并显示订阅/购买入口；不缓存。
4. Proceedings of the Combustion Institute 的部分 ScienceDirect 页面显示 open access / Creative Commons，但命令行 PDF 请求返回 403 HTML；暂不复制 PDF。
5. IOP 条目虽然 Crossref 有 PDF link，但 license 多为 IOP standard 或 TDM/syndication 链接，不作为公开 GitHub Pages 缓存依据。
6. ResearchGate 能看到若干 full-text snippets，但不作为可缓存 PDF 来源。

## 逐条复核

| 条目 | DOI / 页面 | 当前状态 | 对网页处理 |
| --- | --- | --- | --- |
| Volumetric imaging of flame refractive index, density, and temperature using BOST | `10.1007/s11431-020-1663-5` | Springer 页面显示订阅/购买入口；命令行 PDF 请求返回 HTML article page，不是 PDF | 只放出版社入口；作为 OERF BOST 前史必读，但不缓存 |
| Simultaneous PIV-BOST refractive-index compensation | `10.1007/s00348-025-04093-y` | Springer PDF 请求返回 HTML；页面无可缓存开放 PDF；Springer 页面显示数据可用性为 no datasets generated/analyzed | 只放 Springer/DOI；请师兄或学校网络解决全文 |
| Tensor-decomposition 4D BOST | `10.1145/3809488` | 7 月 10 日已核 ACM HTML/eReader 可读 19 页、OA/CC BY 4.0，官方 supplement video 可访问；无会话 raw PDF 仍 403。公开 TDBOST 仓库无 LICENSE | 提供 ACM HTML/eReader 和本地结构化精读页；不冒充本地 PDF，代码只作 clean-room 结构参考 |
| UBOST | `10.1007/s00348-020-2912-1` | Springer 页面为 subscription preview，PDF 购买入口；不开放缓存 | 只放 Springer 入口；作为传统 baseline 和方法对照 |
| Tomographic reconstruction of an azimuthally forced flame in an annular chamber | `10.1016/j.proci.2022.08.051` | ScienceDirect 页面显示 open access / Creative Commons；但 PDF 自动下载入口返回访问控制/403 | 只放 ScienceDirect 页面；可在线阅读，不把 PDF 复制进仓库 |
| Tomographic single-shot time-resolved LII | `10.1016/j.proci.2024.105262` | ScienceDirect 页面显示 open access / Creative Commons；Crossref 有 CC BY VOR 许可记录；命令行 PDF 请求仍返回 403 HTML | 只放 ScienceDirect 页面；作为 4D LII 前史，不缓存 PDF |
| 2026 stereo PIV-BOST | `10.1016/j.proci.2026.106175` | ScienceDirect / Crossref 可核题名和期刊；未发现开放 PDF | 只作 PIV-BOST 展望，不缓存 |
| Iron particle ammonia micro-explosion | `10.1016/j.proci.2026.106209` | Crossref 只显示 Elsevier TDM license，未发现开放 PDF | 只放出版社入口 |
| Combustion diagnostics of metal particles: a review | `10.1088/1361-6501/acb076` | Crossref 有 IOP PDF links，但 license 为 IOP standard / TDM；未确认开放缓存许可 | 只放 IOP 页面 |
| Clustering-based particle detection for digital holography | `10.1088/1361-6501/abd7aa` | Crossref 有 IOP PDF links，但 license 为 IOP standard / TDM；未确认开放缓存许可 | 只放 IOP 页面 |
| Correction procedure for fiber-bundle tomographic optical setup | `10.1364/AO.507266` | Optica 页面可读摘要和图表；PDF article 入口不等于开放可缓存 PDF | 只放 Optica 页面 |
| Adaptive particle image velocimetry based on sharpness metrics | `10.1186/s41476-018-0073-0` | EDP Sciences PDF 返回真实 `%PDF`，页面标 Open Access / CC BY 4.0 | 已缓存 `adaptive_piv_sharpness_metrics_gao_2018_edp.pdf` 和首页缩略图 |
| Calibration correction of arbitrary optical distortions by non-parametric 3D disparity field for planar and volumetric PIV/LPT | `10.18409/ispiv.v1i1.116` | ISPIV 2021 repository PDF 返回真实 `%PDF` | 已缓存 `calibration_correction_optical_distortions_3d_disparity_amaral_2021_ispiv.pdf` 和首页缩略图 |
| Self-correction of the optical distortion effect of thermal plumes in particle image velocimetry | `10.1063/5.0233759` | Crossref/AIP 元数据可核；AIP direct PDF 返回 403 | 只放 DOI/出版社入口，不缓存 |
| Simultaneous particle image velocimetry and synthetic schlieren measurements of an erupting thermal plume | `10.1088/0957-0233/20/12/125402` | IOP 页面可核；未确认 redistribution-safe open PDF | 只放 IOP 页面 |
| Distortion compensation for generalized stereoscopic PIV | `10.1088/0957-0233/8/12/008` | Crossref/IOP 元数据可核；自动访问 IOP 页面返回 403 | 只放 DOI 入口 |
| Particle imaging through planar shock waves and associated velocimetry errors | `10.1007/s00348-015-2004-9` | Springer 页面可核；PDF 请求返回 HTML page，不是 PDF | 只放 Springer 页面 |

## 对毕业设计的含义

- 本科毕设不能建立在“我已经能公开下载所有核心全文”这个前提上；4D BOST 已可用 ACM 正式全文精读，PIV-BOST 等其余受限主线仍需学校资源或师兄合法版本。
- 公开网页的价值在于：把能直接读的开放 PDF 缓存好，把不能缓存的边界解释清楚，并保留 DOI/出版社入口。
- 对何远哲沟通时，可以直接问：“PIV-BOST 和 Volumetric BOST 前史是否有组内允许我阅读的版本？4D BOST 正文我已读，公开仓库与正文的 decoder/rank/test split 差异应以哪个版本为准？”
- 如果短期拿不到这些全文，仍可先用已缓存的 NeRIF arXiv、NeDF/NRIP、BOS review、NASA/NTRS 工程资料、open BOS dataset、TBOS arXiv baselines，以及本轮新增的 PIV 图像层/标定层开放 PDF 建立代码与实验框架。

## 下一步

1. 继续监控 PIV-BOST、stereo PIV-BOST 和 U-Net flame-front accepted 条目是否出现作者开放 PDF 或正式开放数据页；4D BOST 只需监控稳定 raw PDF/data release，不再重复检索摘要全文。
2. 若师兄确认可提供组内阅读版本，只用于个人学习和内部笔记，不上传公开 GitHub Pages。
3. 若发现机构库 PDF，先核对页面 license；Taverne、ResearchGate、出版社 TDM 或访问控制 PDF 不缓存。
