# 算子学习 × 三维重建来源审计

审计日期：2026-07-10
规则：优先期刊/会议官方页、OpenReview accepted page、CVF/NeurIPS/JMLR/Cambridge 正式页与作者官方仓库。arXiv 只在无更高层级开放全文或需要 open PDF 时使用。所有跨模态迁移结论均标为推断。

## 已核验核心来源

| 主题 | 正式/主来源 | 核验内容 | 在项目中的边界 |
| --- | --- | --- | --- |
| NeRIF | https://doi.org/10.1063/5.0250899 | Physics of Fluids 2025、题名与 DOI | 定义 BOST forward/neural field；不等于 operator。 |
| 4D BOST | https://doi.org/10.1145/3809488 | ACM TOG 2026、OA HTML/eReader | 定义 OERF 4D 主线；本项目未完整复现。 |
| PIV-BOST | https://doi.org/10.1007/s00348-025-04093-y | Experiments in Fluids 2025 | 下游物理价值；公开站不缓存订阅 PDF。 |
| DeepONet | https://doi.org/10.1038/s42256-021-00302-5 | Nature Machine Intelligence 2021 | 根架构，不证明 BOST 有效。 |
| FNO | https://iclr.cc/virtual/2021/poster/3281 | ICLR 2021 official poster/paper | 根架构；resolution transfer 必须在 BOST 重新测。 |
| Neural Operator | https://www.jmlr.org/papers/v24/21-1524.html | JMLR 2023 full paper | 统一术语与函数空间边界。 |
| U-NO | https://openreview.net/forum?id=j3oQF9coJd | TMLR published page | 多尺度候选；非当前首个模型。 |
| F-FNO | https://openreview.net/forum?id=tmIiMPl4IPa | ICLR 2023 accepted page | efficiency/residual 参考。 |
| CNO | https://proceedings.neurips.cc/paper_files/paper/2023/hash/f3c1951b34f7f55ffaecada7fde6bd5a-Abstract-Conference.html | NeurIPS 2023 proceedings | matched local/operator baseline。 |
| MG-TFNO | https://openreview.net/forum?id=AWiDlO63bH | TMLR 2024 published PDF | 高分辨率升级，不是第一阶段。 |
| Riesz NO | https://openreview.net/forum?id=Vjw7q1quNt | ICLR 2026 accepted poster | 局部导数候选；BOST 价值是推断。 |
| NIO | https://openreview.net/forum?id=S4fEjmWg4X | ICML 2023 accepted poster | inverse operator 方法论。 |
| Ultrasound NO | https://openreview.net/forum?id=tSokLyjvW5 | MIDL 2023 short paper | tomography 类比；声学与 BOST physics 不同。 |
| Learned Primal-Dual | https://pubmed.ncbi.nlm.nih.gov/29870362/ | IEEE TMI 2018, DOI 10.1109/TMI.2018.2799231 | forward-aware unrolling baseline。 |
| MoDL | https://pmc.ncbi.nlm.nih.gov/articles/PMC6760673/ | IEEE TMI 2019, DOI 10.1109/TMI.2018.2865356；NIH manuscript 和官方代码 | 学习先验 + 数值 data-consistency block；MRI 结论不可直接搬到 BOST。 |
| Separable least squares | https://doi.org/10.1137/0710036 | SIAM JNA 1973, 10(2), 413--432 | support-fit 的数值原理背景；证明该消元步骤本身不是神经创新。 |
| Deep Null Space Learning | https://doi.org/10.1088/1361-6420/aaf14a | Inverse Problems 2019；开放 arXiv 1806.06137 交叉核验 | `Id-A^+A` 零空间修正与 data consistency 的数学前例。 |
| Siamese Cooperative Learning | https://doi.org/10.1109/TPAMI.2024.3359087 | IEEE TPAMI 2024, 46(7), 4866--4879；PubMed/DBLP/作者 manuscript 交叉核验 | incomplete measurements 下 range/null-space 双网络自监督；不是 BOST 验证。 |
| Geometry-aware attenuation | https://doi.org/10.1109/TMI.2024.3473970 | IEEE TMI early access 2024/final volume 2025；官方 repo 交叉核验 | 2D feature backprojection 结构可借，CT 结论不可搬。 |
| C2RV | https://openaccess.thecvf.com/content/CVPR2024/html/Lin_C2RV_Cross-Regional_and_Cross-View_Learning_for_Sparse-View_CBCT_Reconstruction_CVPR_2024_paper.html | CVPR 2024 official open paper | 不同 view 不等权是结构启发。 |
| Dynamic CT INR | https://openaccess.thecvf.com/content/ICCV2021/html/Reed_Dynamic_CT_Reconstruction_From_Limited_Views_With_Implicit_Neural_Representations_ICCV_2021_paper.html | ICCV 2021 official open paper/code | 4D self-supervised 邻域。 |
| Noise2Inverse | https://doi.org/10.1109/TCI.2020.3019647 | IEEE TCI 2020；CWI final PDF | view split 依据，但独立噪声假设需重审。 |
| Equivariant Splitting | https://openreview.net/forum?id=upMIVpe467 | ICLR 2026 accepted poster | incomplete forward model 自监督；BOST symmetry 需另证。 |
| Spatial MoE | https://proceedings.neurips.cc/paper_files/paper/2022/hash/4c5e2bcbf21bdf40d75fddad0bd43dc9-Abstract-Conference.html | NeurIPS 2022 proceedings | routing/collapse 工具，不证明双分支有效。 |
| PNO | https://openreview.net/forum?id=gangoPXSRw | TMLR accepted 2025 | function-level UQ 高阶方向。 |
| LUNO | https://openreview.net/forum?id=4Z04wVQ9FY | ICML 2025 spotlight | trained operator 后验近似。 |
| Bayesian NO | https://openreview.net/forum?id=6WvIkYsMA8 | TMLR accepted 2025 | uncertainty-based failure detection。 |
| 3D flow no-GT | https://doi.org/10.1017/dce.2026.10038 | Data-Centric Engineering 2026 OA full text | 实验式无真值验收邻域；传感平面不等于 BOS camera。 |

## 当前原创性判断的证据等级

### 已观察事实

1. T16 合成 closure 中 Residual/Absolute 的优胜工况发生稳定翻转。
2. fixed view gate、metadata gate 与 observed-residual channel 没有解决该翻转。
3. observed lift residual 能区分 IID 与 view/joint OOD，但直接平铺入 3D trunk 会退化。
4. 共享双输出 v1 的等权融合优于两个端点的平均值，说明可用互补存在；从零学习的 query router 权重近 0.50，不能称为成功路由。
5. 在当前线性合成 forward operator 下，support-view 闭式最小二乘融合的五域 field oracle regret 为 0.014%，是后续学习路由必须击败的强基线。

### 来源支持的结构组件

1. geometry-aware feature backprojection：IEEE TMI geometry-aware attenuation、C2RV。
2. support/query 或 measurement splitting：Noise2Inverse、Equivariant Splitting。
3. expert routing/collapse control：Spatial MoE、Neural Experts、MoE-POT。
4. operator uncertainty：PNO、LUNO、Approximate Bayesian NO。
5. operator + per-instance field：FACT/NAF 与本项目的 NeRIF warm-start 推断。

### 尚未找到直接先例，因此只能称“候选缺口”

截至本轮核验，尚未找到一篇成熟论文同时满足：BOST displacement-to-volume、Residual/Absolute 双专家、support-view 闭式融合、query-camera 零空间修正、physics-lift reliability flip 和 operator-to-NeRIF refinement。这个“组合缺口”不是原创性证明；闭式一维最小二乘本身也是标准数值工具。正式投稿前仍需由师兄/导师做领域检索，并补 Google Scholar/Web of Science/Scopus 的引文追踪。

## 不应写进论文的越界句

- “首次将神经算子用于三维流场重建。”已有 RecFNO、Energy Transformer、ultrasound tomography 等邻域。
- “query-view loss 保证三维真值正确。”病态逆问题与 forward bias 均可让错误场通过投影检查。
- “FNO 天然跨分辨率。”这是待测性质，不是本项目既得结果。
- “双分支一定优于单分支。”当前只运行了几乎全共享参数的小型 v1，还没有独立专家、等容量 ensemble 与真实数据对照。
- “support-fit 就是新算法。”当前只能称它是有效的标准数值基线；论文贡献必须来自与 BOST 病态性、零空间学习、变几何或真实验验证的实质结合。
- “真实 BOST 不需要真值。”可以无完整 field GT 训练/验收，但仍需要独立相机、物理边界、参考结果或专家审查。
