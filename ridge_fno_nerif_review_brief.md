# Ridge → residual FNO → NeRIF：v3a 师兄审阅简报

更新时间：2026-07-11
状态：`DEVELOPMENT_PILOT_RIDGE_FNO_3_OF_3_GATES`，不是投稿结论

## 一句话判断

不要继续把 FBP-style lift 直接喂给更大的网络。当前最值得推进的工作假设是：

> 先用同相机预算下、验证集锁定的 ridge/CGLS 获得可观测稳定分量，再让小型 residual FNO 学习跨样本先验修正，最后把结果作为 NeRIF 的场初始化，只用原 K 个重建相机做逐实例物理细化。

工作式写成：

\[
x_{\mathrm{ridge}}=A_S^\top(A_SA_S^\top+\lambda I)^{-1}y_S,
\qquad
x_0=x_{\mathrm{ridge}}+F_\theta(x_{\mathrm{ridge}},m,g),
\]

\[
n_\phi(\mathbf p)=\operatorname{Interp}(x_0)(\mathbf p)+r_\phi(\mathbf p),
\qquad r_{\phi_0}=0.
\]

这里的创新候选不是“用了 FNO”，而是强数值可观测分量、相机几何条件、零初始化 NeRIF 残差、固定相机预算和独立审计证据如何组成一个可靠系统。

## v3a 做了什么

- 每个三维场分别生成 K=4/6/8 三档固定重建掩码；固定 80° 相机进入重建，60° 相机在所有训练、早停、ridge 选择和模型选择前锁定为 `Q_audit`。
- 每个 K 独立训练模型；训练和评估相机身份完全匹配。
- 输入为 15 通道：K-view lift、support、view fraction、9 个相机身份通道和 z/y/x 坐标；不把仿真的真实噪声等级喂给网络。
- 新噪声按当前 K-view RMS 重新生成；test truth、未选相机和 `Q_audit` 不改变模型输入。
- 传统对照为 train-only affine FBP lift，以及只在 validation 上选择 λ 的非负 ridge。
- 256 个源体场，其中 96 个独立测试体场，覆盖 IID、noise OOD、thin-front family OOD 和 joint OOD。
- 3 个模型种子；先在同一体场内折叠种子，再做 20,000 次四域等权分层 bootstrap。
- 共 5,184 条 sample-method 结果、1,728 条 field cluster 结果；28 项相关测试与 checksum validator 通过。

## 核心结果

### Ridge-initialized residual FNO 相对 validation-locked ridge

| 重建预算 | 平均体场改善 | 95% cluster CI | p10 | 伤害率 >1% | 独立审计残差改善 | 判决 |
|---:|---:|---:|---:|---:|---:|---|
| K=4 | +21.54% | [+19.11%, +23.97%] | +4.27% | 0.00% | +14.63% | 通过 development gate |
| K=6 | +19.68% | [+17.22%, +22.09%] | +3.19% | 0.00% | +11.26% | 通过 development gate |
| K=8 | +16.91% | [+14.61%, +19.21%] | +0.69% | 4.17% | +10.01% | 通过 development gate |

门槛预先设为：均值改善的 95% CI 下界 >5%，p10 ≥0，`>1%` 伤害率 ≤5%。

最弱的域是未见 thin-front family：K=4/6/8 平均改善为 +5.93% / +5.74% / +4.08%。因此“跨 family 已解决”不能写；K=8 的 OOD 安全边际已经很薄。

### 机制消融

- FBP-lift residual U-Net：K=4/6/8 为 −0.68% / −19.03% / −21.43%，0/3 通过。
- FBP-lift residual FNO：K=4/6/8 为 −2.16% / −16.10% / −21.53%，0/3 通过。
- Ridge-initialized residual U-Net：均值为正，但只有 K=4 通过；K=6/8 的 p10 和伤害率失败。
- Ridge-initialized residual FNO：3/3 通过，是当前唯一可进入下一阶段的候选。

这支持一个更具体的假设：网络不应从弱 lift 重新猜三维场；强数值解先提供可观测成分后，FNO 才可能稳定学习低幅跨样本修正。它仍只是当前 surrogate 中的机制证据。

## 为什么还不能叫论文结果

1. Forward 仍是 `8×16×16` 线性 slice-stack，不是 NeRIF/真实 OERF 的 cone-ray、有限孔径、背景 warping 与 optical-flow 链。
2. 96 个测试场已被本轮查看，只能算 development set；确认性结论需要全新 manifest 和一次性锁定测试。
3. 相机布局固定，没有 geometry drift、缺失相机、标定误差或可变相机集合。
4. Ridge 只是 Tikhonov 风格强基线；还没有 CGLS/TV、RBF direct、GRU-BOST、NeRIF、NeDF/NRIP 概念对照。
5. 当前证明的是体场误差和 synthetic `Q_audit` 同时改善，不是 NeRIF 总墙钟时间缩短。
6. 网络训练成本尚未和逐帧摊销盈亏点一起报告。

## 下一阶段只做 M0

第一版不生成 NeRIF 全部权重，不做 hypernetwork：

1. 用 ridge-FNO 输出离散粗场 `x0`。
2. 把 `x0` 插值为连续初始场。
3. NeRIF 写成 `Interp(x0) + zero-init residual MLP`。
4. 只用 K 个重建相机优化 residual MLP；`Q_audit` 永远不参与 early stop 或选迭代数。
5. 对照 random NeRIF、ridge→NeRIF、ridge-FNO→NeRIF 和 oracle low-pass→NeRIF。

主指标改为：达到预锁 validation 阈值的总墙钟时间。建议 development go/no-go：中位 time-to-threshold 至少降低 30%，最终 `Q_audit` 非劣 1%，p10 不低于 −5%，`>1%` 伤害率不超过 10%。真正投稿前再把时间目标提高到 2×。

## 必须读的角色化文献

| 角色 | 文献 | 本项目提取内容 | 不能据此声称 |
|---|---|---|---|
| OERF 主方法 | [He et al., NeRIF, Physics of Fluids 2025](https://doi.org/10.1063/5.0250899) | 折射率连续场、BOST forward/loss、实验几何 | 没有跨样本算子或 learned initialization |
| 稀疏视角强邻居 | [Li et al., NeDF, Physics of Fluids 2024](https://doi.org/10.1063/5.0241191) | gradient field、nonlinear ray synthetic、5–180 camera study | 不是预训练 direct operator |
| 最新强竞品 | [Lu et al., NRIP, Combustion and Flame 2026](https://doi.org/10.1016/j.combustflame.2026.115082) | hash/Fourier、discrete gradient、3D mask、真实噪声失败 | hash encoding 不天然可靠；作者仓库无明确复用许可 |
| BOST 直接学习前史 | [Bo et al., GRU-BOST, Optics Express 2023](https://doi.org/10.1364/OE.505992) | projection-to-volume 网络、推理时间、留出相机 | 固定有序 GRU 不等于 neural operator |
| 逆算子定义 | [Molinaro et al., Neural Inverse Operators, ICML 2023](https://proceedings.mlr.press/v202/molinaro23a.html) | inverse operator 的问题结构和强 baseline 习惯 | 其 operator-to-function 输入不等同 BOST stack |
| INR 初始化直接先例 | [Tancik et al., Learned Initializations, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Tancik_Learned_Initializations_for_Optimizing_Coordinate-Based_Neural_Representations_CVPR_2021_paper.html) | learned init、time-to-quality、CT/3D inverse tasks | 不是 measurement-conditioned BOST 初始化 |
| 可迁移 INR 特征 | [Vyas et al., STRAINER, NeurIPS 2024](https://research.google/pubs/learning-transferable-features-for-implicit-neural-representations/) | 共享前层作为 per-instance INR 初始化 | 早期 10 dB 收益不能外推到 NeRIF |
| 可变相机集合 | [Prasthofer et al., VIDON, arXiv 2022](https://arxiv.org/abs/2205.11404) | 数量/位置可变、排列不变 sensor aggregation | 只有 PDE 数值实验，不是 BOST 证据 |
| operator-neural field 桥 | [Serrano et al., CORAL, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/df54302388bbc145aacaa1a54a4a5933-Abstract-Conference.html) | arbitrary geometry、coordinate field modulation | 不是投影反演或 warm-start 论文 |

## 12 周论文闸门

| 周 | 唯一交付 | 继续条件 |
|---:|---|---|
| 1 | 与何远哲冻结输入、输出、K、geometry、数据和主指标 | 拿到最小 forward/loss/data contract |
| 2 | 原始 NeRIF 或组内 compact INR 的 timing-to-quality 曲线 | 能稳定复跑同一场 |
| 3 | 独立 cone-ray/非线性 forward 与单位测试 | 避免 inverse crime |
| 4 | CGLS/TV、RBF/ridge、NeRIF 强基线和 final manifest | 不允许只和弱 FBP 比 |
| 5 | ridge-FNO 在新 synthetic 上一次性确认 | 三预算至少 2/3 风险门槛通过 |
| 6 | geometry/noise/family OOD 与近零空间 thin-front 反例 | 尾部失败可解释、可回退 |
| 7 | M0 `ridge-FNO field + zero-init NeRIF residual` | 总时间有稳定下降趋势 |
| 8 | random/ridge/operator/oracle 四初始化对照 | 最终 `Q_audit` 不劣 |
| 9 | 接组内真实 displacement、calibration、mask | 不读 `Q_audit` 调参 |
| 10 | 独立工况、重复采集和物理 trace | 真实证据不是连续帧伪样本 |
| 11 | 三种子、cluster bootstrap、成本与失败图 | 预注册门槛决定继续或降级 |
| 12 | 八张核心图、论文初稿和复现包 | 不再增加模型家族 |

## 需要何远哲先回答的十个问题

1. “算子学习”指 `displacement→3D field`，还是 `3D history→future field`？
2. 输出最终锁 refractive index、density、temperature 还是 gradient field？
3. 能否提供 NeRIF forward、loss、calibration、mask 与一例可复现数据？
4. 组内更接受 operator 初始化体场，还是生成/调制 NeRIF 参数？
5. Paired CFD/phantom 有多少独立工况？相邻帧和同一 CFD case 如何分组？
6. 真实数据是否包含 raw image、displacement、内外参、ray table 和重复采集？
7. 相机布局是否固定？论文是否必须支持角度变化或缺失相机？
8. 无 3D ground truth 时，最认可 held-out camera、PLIF/front、PIV compensation 还是积分量？
9. 是否同意把 CGLS/TV、RBF direct、GRU-BOST、NeRIF 与 NRIP/NeDF 设为强对照或相关工作？
10. 是否同意先批准两周可行性审计，并在第 2/6/8/10 周做明确继续或降级决定？

## 最终建议

可以继续，而且现在比“直接做一个 FNO”清楚得多。但下一步必须是把 ridge-FNO 当作 **NeRIF 初始化候选**，不是宣布一个新三维重建模型已经成立。若它在独立 forward、全新 fields 或 NeRIF 总时间上失败，应立即保留为严谨 benchmark，并降级到“强数值重建 + neural prior 的 OOD 边界”本科论文，不再堆网络。
