# 何远哲作者谱系审计

更新时间：2026-07-09

用途：防止把同名作者、索引噪声或旁支论文误塞进 BOST / NeRIF 毕设主线。正式写开题时，何远哲主线只放 NeRIF、PIV-BOST、4D BOST；旁支论文用于理解他参与的真实燃烧实验和 OERF 数据链路。

## 核验来源

| 来源 | 结论 | 用法 |
| --- | --- | --- |
| Google Scholar `jCjKaCMAAAAJ` | 检索页显示何远哲个人页，主线条目包括 NeRIF、PIV-BOST、4D BOST | 查新和排序入口；正式引用仍回到 DOI/出版社 |
| ORCID `0009-0007-7393-1032` | 2026-07-09 用 ORCID public API 复核，Works 返回 6 组记录：4D BOST、NeRIF 期刊版、2025 simultaneous PIV-BOST、iron particle heat exchange，以及 NeRIF 的 arXiv/preprint 索引项；暂未列出 2026 stereo PIV-BOST 或 NH3/CH4 reaction-progress 条目 | 作为确认何远哲本人主线论文的高可信入口；未出现在 ORCID 的新条目必须再用 Crossref/出版社和作者链确认 |
| Semantic Scholar author `2322747160` | 2026-07-09 复核仍返回 8 条，其中 6 条与 OERF/流体燃烧方向相关，2 条医学影像方向疑似同名或索引混入；该页额外给出 2026 stereo PIV-BOST 和 NH3/CH4 reaction-progress 线索；NeRIF 条目在该索引中仍显示 arXiv DOI | 只做发现线索，必须再用 DOI/出版社核验；NeRIF 正式引用使用 Physics of Fluids DOI `10.1063/5.0250899` |
| ACM profile / ACM DOI `10.1145/3809488` | 确认 4D BOST 为 ACM Transactions on Graphics 45(5), 1-19；ACM 标注 Open access / CC BY 4.0 | 4D BOST 正式引用与全文入口 |
| 4D BOST 全文/代码复核 | 2026-07-10 已核验 ACM 官方 HTML 全文、19 页 eReader（约 29.4 MB）和官方 supplement video；无会话 raw PDF 仍返回 403，故没有本地缓存文件。Hyz617/TDBOST 公开仓库根目录无 LICENSE，且存在硬编码 GPU/路径、Linux `.so`、依赖冲突和样例 split 语义问题 | 网页提供 HTML/eReader 直读和自己的结构化精读页；不缓存 ResearchGate，不把 TDBOST 当可复制代码，只作为 clean-room 模块/问题参考 |
| Springer BOS collection / Experiments in Fluids page | 确认 PIV-BOST 位于 Background Oriented Schlieren collection，article 164 | PIV-BOST 正式引用和专题背景入口 |
| 出版社 / Crossref | 核 DOI、题名、期刊、卷号、作者链 | 正式引用优先依据 |

补充复核：2026-07-09 再次用 Semantic Scholar Graph API 读取作者 `2322747160`，返回 `paperCount = 8`。与本地论文库对照后，6 条 OERF/流体燃烧相关论文均已覆盖：4D BOST、stereo PIV-BOST、NH3/CH4 reaction-progress、iron particle heat exchange、2025 PIV-BOST 和 NeRIF arXiv/期刊线索；未入库的 2 条为医学影像方向，视为同名或索引混入。

2026-07-09 08:01 CST 再次调用 ORCID public API `0009-0007-7393-1032`，仍返回 6 组 works：4D BOST、NeRIF 期刊版、2025 simultaneous PIV-BOST、iron-particle heat exchange，以及两组 NeRIF arXiv/preprint 索引。没有新增未入库的 He/BOST 主线条目；stereo PIV-BOST 和 NH3/CH4 reaction-progress 仍按 Crossref/出版社核验条目处理，不写成 ORCID 直接列出。

## 已确认与毕设相关的何远哲/OERF论文

| 层级 | 论文 | DOI | 当前定位 |
| --- | --- | --- | --- |
| P0 主线 | Neural refractive index field: Unlocking the potential of background-oriented schlieren tomography in volumetric flow visualization | `10.1063/5.0250899` | NeRIF 主论文，毕业设计最稳入口 |
| P0 主线 | Instantaneous refractive index compensation on the velocity measurement using simultaneous PIV-BOST | `10.1007/s00348-025-04093-y` | PIV-BOST 二阶段升级线 |
| P0/P1 主线 | Tensor Decomposition-Based Four-dimensional Background-Oriented Schlieren Tomography for High-Speed, High-Fidelity Flow Field Reconstruction | `10.1145/3809488` | 4D BOST 挑战线；Crossref 核到 ACM TOG 45(5), 1-19，online 2026-06-29、print 2026-10-31，适合拆成低秩/时序 toy |
| P1 展望 | Instantaneous refractive index compensation on stereo-velocity measurement in turbulent combustion | `10.1016/j.proci.2026.106175` | PIV-BOST 的 stereo-velocity 后续，不建议本科开局硬做 |
| P2 旁支 | Investigation on heat exchange of iron particle combustion based on simultaneous multi-physics field measurements | `10.1016/j.combustflame.2025.114309` | 何远哲参与的多物理测量旁支，说明真实实验链路 |
| P2 旁支 | Experimental Closure of Mean Reaction-Progress Balance in NH3/CH4 Turbulent Premixed Flames | `10.1007/s10494-026-00737-z` | 反应进度/氨甲烷湍流火焰旁支，支撑 OERF 反应流背景 |

说明：上表中 P1 stereo PIV-BOST 和 P2 NH3/CH4 reaction-progress 条目当前不是由 ORCID 直接确认，而是来自 Semantic Scholar 作者页线索，并已用 Crossref/出版社 DOI 元数据核验题名、期刊和 DOI。写开题时可以保留，但不要写成“ORCID 已列出全部旁支论文”。

## 索引噪声与排除

Semantic Scholar 同一作者页还返回了医学影像或心理方向条目，例如 `Unisyn: A Generative Foundation Model for Universal Medical Image Synthesis Across MRI, CT and PET`、`Preoperative Grading of Brain Gliomas Using 3D-ResNet18 Based on Multimodal MRI and Attention Mechanism`。这些条目与 OERF、SJTU 机械/反应流诊断链路不一致，且不在 ORCID 何远哲作品中出现；当前视为同名或索引混入，不进入毕设文献库。

## 4D BOST 开放材料边界

当前 4D BOST 的正式引用入口是 ACM DOI `10.1145/3809488`。ACM 官方 HTML 全文与 eReader 已在浏览器核验可读，正文为 19 页，ACM 标注 Open access / CC BY 4.0；官方动态 supplement video 也已找到。无会话 raw PDF 下载仍受 Cloudflare/CDN 403 限制，因此公开网页提供 HTML/eReader/视频直达和本地结构化精读，不声称已有本地 PDF。ResearchGate 仍不作为缓存来源。

Hyz617/TDBOST 是当前最贴近 tensor-decomposition 4D BOS/BOST 的公开代码线索，但仓库未见 `LICENSE` 文件，README 中的 open-source 说法不能直接等价为可复制许可。对本科毕设来说，它适合用来拆解 `configs`、`dataloader`、`TDmodel`、`render`、projection 和训练 loop，并对照正文发现 decoder width、rank/config 和 test split 的版本疑点；不适合未经确认直接搬代码。仓库运行边界与本地 M3B rank/bias 实验已经集中渲染在 `tdbost_reproducibility_audit.html`。

## 写开题时怎么用

- 主线只引用 NeRIF、PIV-BOST、4D BOST，最多加 stereo PIV-BOST 作为展望。
- iron particle 和 NH3/CH4 论文只用于说明何远哲/OERF 数据来自真实燃烧实验，不要把它们写成 BOST 算法主线。
- 若师兄明确要求转向燃烧机理或多物理实验，再把旁支论文升为正式阅读材料。
