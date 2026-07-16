# OERF 官网源码缺口与纠错扫描

生成日期：2026-07-10

用途：把 OERF 官网源码中的代表论文列表，与本地论文库 `paper_library/papers.json` 做交叉检查。结论用于网页维护、开题引用和给师兄解释“为什么没有机械照抄官网 DOI”，不替代出版社、DOI/Crossref 或论文 PDF 的最终核验。

## 扫描来源

- OERF GitHub 仓库：`https://github.com/laserdiagnostics/oerf-lab-website`
- OERF `publications.ts`：`src/data/publications.ts`
- OERF `research.ts`：`src/data/research.ts`
- OERF `members.ts`：`src/data/members.ts`
- 本地论文库：`paper_library/papers.json`

本轮使用 GitHub raw/API 拉取源码。OERF 仓库 `main` 最新提交仍为 `692a14a99ec1d550a1122b5f010d000289049424`，提交时间 `2026-05-11T04:43:45Z`，提交信息为修复 GitHub Pages 子路径部署。

## 总判断

1. OERF `publications.ts` 当前解析出 20 条代表论文 DOI。
2. 本地论文库当前规模为 561 篇，其中 192 篇已缓存可公开再分发或开放阅读的 PDF，并已生成 192 个单篇术语导读/页码地图页。
3. 本轮没有发现需要按官网代表作机械新增的核心缺口；真正的问题是官网源码中多篇 DOI/题名仍带早期占位、年份错配或跨条目错配。
4. 公开网页继续保留 DOI、出版社入口、开放 PDF 和纠错说明；通过学校 VPN 或订阅访问到的受限 PDF 只进入本机 `private_library/`，不上传 GitHub Pages。
5. Gao Youyang 当前按师兄建议把工作重心转到“三维重建 + 算子学习”；NeRIF / PIV-BOST / 4D BOST 仍是何远哲带教方向与 forward-physics 谱系，OERF 官网代表论文用于说明课题组全貌和方法来源。

## 本轮结构化扫描结果

| 项目 | 数量 | 处理 |
| --- | ---: | --- |
| OERF `publications.ts` 代表论文 DOI | 20 | 作为课题组公开代表作线索 |
| 本地库已覆盖或按正式 DOI/题名纠正后覆盖 | 15 | 保留在论文库、方向索引或审计页中 |
| 源码 DOI/题名仍不稳定或与正式元数据冲突 | 5 | 暂不按源码机械入库，等待 OERF/出版社更新 |
| 本地论文库当前规模 | 561 | 其中 192 篇已缓存开放 PDF；192 篇公开 PDF 已生成页码级术语索引和单篇术语导读/页码地图页 |

## 2026-07-10 02:50 复核补充

本轮再次拉取 `laserdiagnostics/oerf-lab-website` 的 `publications.ts`、`research.ts` 与 `members.ts`，GitHub `main` 仍指向 `692a14a99ec1d550a1122b5f010d000289049424`，提交时间仍为 `2026-05-11T04:43:45Z`；因此 OERF 官网源码本身没有新提交。本地库已增长到 561 篇论文入口、192 个公开缓存 PDF、192 个单篇术语导读页。新增 link-only 条目集中在 T16 神经算子谱系，包括 PI-DION、Voronoi sparse-field reconstruction、Geo-FNO、GINO、MG-TFNO 和 2026 sparse thermal TFNO；它们扩展模型/几何/显存方法邻居，不改变官网代表作 DOI 对账。

### 本轮重新核到的关键对照

| OERF 源码 DOI / 题名线索 | Crossref / 出版社复核 | 本地库处理 |
| --- | --- | --- |
| JFM flame deep learning: 源码 DOI `10.1017/jfm.2019.544` | Crossref 显示 `.544` 对应 breaking surface waves；正式 flame deep-learning 论文为 `10.1017/jfm.2019.545`。 | 已用 `jfm-flame-dl-huang-2019` 收录 Cambridge 页面和正式 DOI。 |
| PECS VET review: 源码 DOI `10.1016/j.pecs.2023.101123` | Crossref 404；正式 DOI 为 `10.1016/j.pecs.2022.101024`。 | 已用 `vet-grauer-cai-2023` 收录 ScienceDirect 页面和正式 DOI。 |
| RIVR flame tomography: 源码 DOI `10.1016/j.combustflame.2020.08.012` | Crossref 显示 `.012` 是 shock-induced detonation 论文；RIVR 正式 DOI 为 `10.1016/j.combustflame.2020.08.025`。 | 已用 `rivrt-flame-liu-2020` 收录，不把 `.012` 误作为蔡组 RIVR 引用。 |
| Multi-kHz three-dimensional flame imaging: 源码 DOI `10.1016/j.combustflame.2023.112731` | Crossref 显示该 DOI 是 2,3-dimethyl-2-butene oxidation kinetics；不是 OERF 源码题名。 | 本地用已核的 kHz / single-camera / endoscopic / fiber-based high-speed 3D diagnostics 条目承接方向，不按源码 DOI 引用。 |
| Science 2022 vDW junction spectrometer: 源码 DOI `10.1126/science.abl8731` | Science 正式页为 `10.1126/science.add8544`。 | 已用 `vdw-spectrometer-yoon-2022` 收录正式入口。 |

### 对当前网页维护的影响

- 本轮没有发现需要新增的 OERF 官网代表论文条目；缺口主要是源码 DOI 的可信度边界，而不是本地库漏收。
- 首页、`source_audit.md` 和开题材料应继续写明：OERF 官网源码用于确认方向和代表作线索，正式引用一律以 Crossref、出版社页或开放仓储为准。
- 因 GitHub Pages 本轮部署包已接近 1 GB 以上，后续新增 OERF 全方向材料优先采用 DOI/出版社入口和摘要级阅读建议；只有小型、明确开放许可且对 He/BOST 主线有明显价值的 PDF 才继续进入公开缓存。

## 2026-07-09 14:30 复核补充

本轮重新拉取 `laserdiagnostics/oerf-lab-website` 的 `publications.ts`、`research.ts` 与 `members.ts`，GitHub `main` 最新提交仍为 `692a14a99ec1d550a1122b5f010d000289049424`，提交时间 `2026-05-11T04:43:45Z`。因此源码没有新更新；本轮复核的价值在于用当前更完整的本地论文库重新对账。

### 源码 DOI 与 Crossref 状态

| 源码 DOI | Crossref 解析 | 本地正式处理 |
| --- | --- | --- |
| `10.1038/s41928-025-01355-x` | 404 | 本地已按正式 DOI `10.1038/s41928-026-01571-x` 收录 Nature Electronics 2026 memristor spectrometry。 |
| `10.1038/s44160-025-00755-x` | 404 | 本地已按正式 DOI `10.1038/s44160-025-00914-4` 收录 Nature Synthesis 2025 electrified vapour deposition。 |
| `10.1016/j.combustflame.2025.114780` | 404 | 本地已按正式 DOI `10.1016/j.combustflame.2026.114780` 收录 iron particle ammonia co-firing。 |
| `10.1038/s44286-025-00001-x` | 404 | 按题名反查只稳定命中 Nature Synthesis `10.1038/s44160-025-00914-4`；暂不作为 Nature Chemical Engineering 独立条目引用。 |
| `10.1038/s41467-025-00001-x` | 404 | 本地已收录 2025 Nature Communications 正式条目 `10.1038/s41467-025-66403-6`，题名为 `Enhanced production of active species and NH3 using non-equilibrium ferroelectric barrier discharge`。 |
| `10.1038/s41566-024-01456-x` | 404 | 本地已按正式 DOI `10.1038/s41566-024-01470-7` 收录 Nature Photonics 2024 `Miniaturization of optical spectrometers: simple yet powerful`。 |
| `10.1038/s41467-024-47466-x` | 404 | 本地已按正式 DOI `10.1038/s41467-024-47230-7` 收录 2024 ferroelectric electrode / afterglow Nature Communications。 |
| `10.1038/s41586-023-06689-8` | 404 | 本地已按正式 DOI `10.1038/s41586-023-06694-1` 收录 Nature 2023 atmospheric-pressure plasma。 |
| `10.1016/j.pecs.2023.101123` | 404 | 本地已按正式 DOI `10.1016/j.pecs.2022.101024` 收录 PECS 2023 VET review。 |
| `10.1016/j.combustflame.2023.112731` | 可解析，但 Crossref 题名为 2,3-dimethyl-2-butene oxidation kinetics，与 OERF 源码题名 `Multi-kHz three-dimensional flame imaging` 不符。 | 不按源码题名引用；本地用已核的 kHz / single-camera / endoscopic flame tomography 条目承接高速三维火焰诊断谱系。 |
| `10.1126/science.abl8731` | 404 | 本地已按正式 DOI `10.1126/science.add8544` 收录 Science 2022 tunable van der Waals junction spectrometer。 |
| `10.1016/j.combustflame.2020.08.012` | 可解析，但题名为 shock-induced detonation of heterogeneous energetic solid fuels，与源码 RIVR flame tomography 不符。 | 本地已按正式 DOI `10.1016/j.combustflame.2020.08.025` 收录 RIVR / view registration flame tomography。 |
| `10.1017/jfm.2019.544` | 可解析，但题名为 breaking surface waves Lagrangian transport，与源码 deep-learning flame evolution 不符。 | 本地已按正式 DOI `10.1017/jfm.2019.545` 收录 JFM 2019 online 3D flame evolution prediction。 |

### 对引用策略的影响

- `publications.ts` 仍适合当作 OERF 方向图谱来源，但不适合直接当作 BibTeX / 开题报告 DOI 来源。
- 论文库和开题材料继续以 Crossref、出版社页、arXiv、机构库和 DOI 解析结果为准。
- 给何远哲师兄看时，可以直接说明：网页没有机械照抄官网 DOI，是因为官网源码中有若干占位 DOI、年份 DOI 和题名错配；本地库已按正式元数据纠正。
- 本轮不需要新增新的 OERF 代表作条目；真正要新增的是对账说明和引用边界。

## 5 个不稳定条目的处理

| 源码条目 | 本轮核验 | 当前处理 |
| --- | --- | --- |
| Nature Chemical Engineering 2025 `10.1038/s44286-025-00001-x` | DOI 看起来像占位；Crossref 按题名反查只稳定命中 Nature Synthesis 2025 `10.1038/s44160-025-00914-4`。源码期刊名与正式元数据不一致。 | 暂不按 Nature Chemical Engineering 入库；本地保留 Nature Synthesis EVD 正式条目和 OSTI/DOE accepted manuscript 入口。 |
| Nature Communications 2025 `10.1038/s41467-025-00001-x` | DOI 看起来像占位；源码题名混合了 2024 ferroelectric afterglow 与 2025 NH3 active species 两条线索。Crossref 按题名稳定命中 2024 Nature Communications `10.1038/s41467-024-47230-7`；本地库另有 2025 NH3 正式条目 `10.1038/s41467-025-66403-6`。 | 本地保留两条正式 Nature Communications 记录；不使用源码占位 DOI。 |
| Combustion and Flame 2025 `10.1016/j.combustflame.2025.114309` | 源码题名为 `Three-dimensional reconstruction of turbulent flame dynamics...`，但 DOI/Crossref 对应 `Investigation on heat exchange of iron particle combustion...`。题名反查未稳定命中蔡组三维火焰动态论文。 | 本地保留真实 DOI 的铁颗粒 heat-exchange 条目；源码题名暂不作为正式引用。 |
| Combustion and Flame 2024 `10.1016/j.combustflame.2023.113171` | 源码题名为 `Large eddy simulation of flash boiling spray...`，但 DOI/Crossref 对应 `Phase change and combustion of iron particles...`。按题名反查未稳定命中 Weiwei Cai 相关正式 DOI。 | 本地保留真实 DOI 的铁颗粒相变/燃烧条目；flash-boiling/LES 只保留为待核线索。 |
| Combustion and Flame 2023 `10.1016/j.combustflame.2023.112731` | Crossref/出版社题名与源码写的 `Multi-kHz three-dimensional flame imaging` 不一致，实际是 2,3-dimethyl-2-butene 氧化动力学论文。按题名反查更接近 OERF 的 `kHz-rate volumetric flame imaging using a single camera`、10-kHz endoscopic VLIF 等相邻条目，但没有同题名正式 DOI。 | 不按源码题名引用；本地用已核的 kHz single-camera / 10-kHz endoscopic VLIF / Zheng 2023 3D flame surface measurements 承接 OERF 高速三维火焰测量线索。 |

## 代表性纠错

| OERF 源码条目 | 源码 DOI / 题名问题 | 本地处理 |
| --- | --- | --- |
| Nature Electronics 2026 memristor spectrometry | 源码 DOI `10.1038/s41928-025-01355-x` | 本地按正式 Nature/Crossref 线索改为 `10.1038/s41928-026-01571-x` |
| Nature Synthesis electrified vapour deposition | 源码 DOI `10.1038/s44160-025-00755-x` | 本地按正式 Nature 页面改为 `10.1038/s44160-025-00914-4` |
| Combustion and Flame 2026 iron/ammonia co-firing | 源码 DOI 年份写作 2025 | 本地按 Crossref 改为 `10.1016/j.combustflame.2026.114780` |
| Combustion and Flame 2025 iron particle heat exchange | 源码题名曾写成 turbulent flame dynamics | 本地按 Crossref 识别为 `Investigation on heat exchange of iron particle combustion...`, DOI `10.1016/j.combustflame.2025.114309` |
| Combustion and Flame 2025 iron micro-explosion | 源码 DOI 与题名不匹配 | 本地按正式题名改为 `10.1016/j.combustflame.2025.113974` |
| Nature Photonics 2024 spectrometer perspective | 源码题名和 DOI 与正式页不一致 | 本地按正式题名 `Miniaturization of optical spectrometers: simple yet powerful`，DOI `10.1038/s41566-024-01470-7` |
| Nature Communications 2024 ferroelectric plasma | 源码 DOI `10.1038/s41467-024-47466-x` | 本地按 Nature 页面改为 `10.1038/s41467-024-47230-7` |
| Nature 2023 atmospheric-pressure plasma | 源码 DOI `10.1038/s41586-023-06689-8` | 本地按 Nature 页面改为 `10.1038/s41586-023-06694-1` |
| PECS 2023 volumetric emission tomography | 源码 DOI `10.1016/j.pecs.2023.101123` | 本地按出版社/Crossref 改为 `10.1016/j.pecs.2022.101024` |
| Science 2022 tunable van der Waals junction | 源码 DOI `10.1126/science.abl8731` | 本地按 Science 页面改为 `10.1126/science.add8544` |
| Combustion and Flame 2020 RIVR | 源码 DOI `10.1016/j.combustflame.2020.08.012` | 本地按正式页改为 `10.1016/j.combustflame.2020.08.025` |
| JFM 2019 flame deep learning | 源码 DOI `10.1017/jfm.2019.544` | 本地按 Cambridge/Crossref 改为 `10.1017/jfm.2019.545` |

## 对毕设网页的影响

- OERF 官网源码适合用来组织方向：BOST/NeRIF、computational imaging、spectrometry、plasma、metal particles、LES/data assimilation。
- 参考文献、开题报告和给师兄看的材料不能直接照抄源码 DOI。
- `source_audit.md` 是正式写作前必须打开的文件；它记录了 DOI、题名、PDF 缓存边界和纠错记录。
- `private_library/` 只服务本机学习，不上传公开仓库；公开页面只显示 DOI、出版社入口、开放 PDF 或“需要 WebVPN/手动下载”的状态。

## 下一步

1. 每次 OERF 官网更新后，重新拉取 `publications.ts` 做一次轻量差异扫描。
2. 新增论文先按题名查 Crossref、出版社页和开放仓储，再决定是否加入论文库。
3. 能确认开放许可且可稳定下载的 PDF 才缓存到 `paper_library/pdfs/`；否则只放出版社入口或进入本地私有清单。
