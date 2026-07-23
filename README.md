# OERF 毕业设计工作台

这是 Gao Youyang 面向上海交通大学 OERF Lab / 何远哲 BOST-NeRIF 方向的毕业设计前期调研工作台。

## 文件说明

- `index.html`：主网页，包含研究地图、全方向雷达、项目线索、何远哲主线、方法邻居、选题库、路线图、执行包、文献索引、精读卡、开题包和沟通模板。
- `references.bib`：核心 BibTeX 引用库，优先收录 BOST / NeRIF / PIV-BOST / 4D BOST / 竞品方法 / 背景综述。
- `paper_library/`：论文 PDF 跳转库，包含 `index.html`、`papers.json`、开放可缓存 PDF、第一页预览图和生成脚本。
- `pdf_cache_audit.html`：PDF 可读性与缓存边界审计，解释哪些论文可本地缓存、哪些只能放 DOI/出版社/开放入口。
- `opening_reference_shortlist.md`：开题参考文献短表与引用位置，包含 15 篇推荐文献、引用句模板和 GB/T 风格草稿。
- `one_page_proposal.md`：一页开题提案，可直接给导师或师兄看。
- `advisor_three_page_brief.md`：给导师/师兄看的三页沟通 brief，把研究站位、已有 demo、数据需求和一年路线压缩到 8-10 分钟。
- `he_review_portal.html`：给何远哲师兄看的 5 分钟 HTML 审核入口，收束 A/B/C/E/F/T16 路线、需要确认的问题和关键深挖入口。
- `he_review_feedback_pack.md`：给何远哲师兄的快速审核包，列出希望师兄判断的 5 件事、A/B/C/E 定题路线和可直接回复的反馈格式。
- `topic_execution_playbook.html`：毕设选题执行手册，把 A/B/C/F/D 路线转成最小毕业版、升级版、数据需求、代码需求、两周开工、补课路径和师兄沟通话术。
- `top_topic_proposal_cards.md`：NeRIF、PIV-BOST、4D BOST、自动报告、系统误差、位移质量控制和 T16 神经算子 proposal 卡，便于会前让师兄选择或修改。
- `operator_learning_bost_bridge.html`：T16 算子学习 × BOST 三维重建专项页，包含 Residual FNO/DeepONet/Temporal 三路线交互、核心文献、OERF 方法前史、12 周计划与师兄问题。
- `operator-learning/index.html`：独立的保姆式学习主页，包含基础诊断、16 周依赖解锁、角色化论文/代码资源、研究路线决策、投稿证据门槛和可复制师兄简报。
- `tdbost-code-guide/`：Hyz617/TDBOST 零基础中文代码导读，包含四站阅读路线、23 个文件的逐语句注释浏览、文件内搜索、数据契约和运行边界审计；公开导出不包含原始数据、编译二进制或私有材料。
- `operator_nullspace_evidence_dashboard.html`：T16 假设审计入口，串联 shared/independent dual、exact null 上界、matched learned-null 负结果、adaptive-query v2c、M3B 六几何 LOGO/UQ、致命缺口与师兄决策。
- `qc_snco_red_team_audit.md`：QC-SNCO 红队审计，纠正统计单元与 query 口径，定义同相机预算、Q_fit/Q_audit、non-learning update、独立 forward 和真实数据恢复门槛。
- `operator_fair_budget_dashboard.html`：T16 v2d 同重建预算判决页，比较 `S-only`、`S∪Q direct`、learned correction 与 numerical query-null update，并显示 cluster CI、p10、CVaR 和 harm rate。
- `ridge_fno_nerif_roadmap.html`：T16 v3a 独立执行主页，把 training-mask-matched direct operator 的三预算证据、ridge-FNO 机制判断、12 周 NeRIF warm-start 路线、论文门槛和师兄审核问题收进同一页面。
- `ridge_fno_nerif_review_brief.md`：可直接交给何远哲师兄的 v3a 审核 brief，分开写已证实结果、不能越界的结论、下一阶段 M0 实验和需要组内确认的接口。
- `own_algorithm_lab.html`：T16 v3b-v3k-C 自有算法开发与公平基线竞技场，动态展示 FNO/DeepONet 基线审计、28 布局可辨识性、v3k-A/B 机制负结果，以及 v3k-C 伴随/梯度校验、validation-only Landweber 选择、四域审计、worst-layout 尾部与 conditional-step 学习路线。
- `own_algorithm_review_brief.md`：给何远哲审核的自有算法总 brief，记录 v3b 三种子 1/3 开发门槛、v3c 独立 dev2 负结果、停止当前 adapter 与 v3d 查重后的路线。
- `demo_t16_operator/run_v3c_protocol_gate.py`：v3c 零初始适配器与数据协议闸门，验证初始输出严格等于冻结 FNO、首次反向可学习、328 个 dev2 fields 与 v3b 种子零重叠，并将 blind final 未封存保留为硬阻断。
- `demo_t16_operator/run_v3c_k6_dev2_pilot.py`：K=6 同 checkpoint 追加训练对照，用三种子比较 base FNO、continued FNO 和 frozen zero-init adapter；支持 analysis-only 重算、逐场配对统计、成本审计和 checksum validator。
- `v3c_k6_dev2_negative_result_brief.md`：给何远哲师兄的 v3c 负结果简报，说明当前 adapter 为何应停止、24-epoch FNO 为何尚未饱和，以及下一轮为什么必须先补 PEFT 强基线。
- `v3d_prior_art_and_novelty_gate.md`：F-Adapter、R2-FFNO、MG-TFNO、GINO 与 FourierFT 的最近邻查重，定义 acquisition-geometry-conditioned frequency adapter 的六个对手、三个机制消融、开发门槛和十天执行顺序。
- `v3d_fno_validation_plateau_brief.md`：三种子 K=6 FNO 从 24 延长到 96 epochs 的 validation-only plateau 审计；当前仍未 plateau，因此阻断所有新 adapter superiority 比较。
- `v3d_fno_optimizer_protocol_brief.md`：三种子、三 optimizer/scheduler 策略到 240 epochs 的审计；锁定 FNO 固定 epoch 冠军与 plateaued control，并把功能 pilot、跨架构算力公平和 superiority 拆成不同门槛。
- `v3e_compute_accounting_brief.md`：五架构参数/FLOPs-v1/MPS 时间与内存、FNO time-to-target、PEFT 成本反例和下一轮 matched learning-curve 合同。
- `v3f_deeponet_fno_frontier_brief.md`：四学习率 DeepONet screen、三优化协议 24→240 epoch 曲线、与 FNO 的误差—时间前沿、冻结后 dev2 统计，以及“FNO 主干 + acquisition-conditioning branch”的下一版自研模型决策。
- `v3g_deeponet_capacity_audit_brief.md`：8 架构 × 3 学习率 × 3 种子的有界 DeepONet screen、1.5× 参数上限、短程冠军的 240-epoch 排名翻转、冻结后 dev2 负结果、baseline 预算停止点与 GC-SRO controls。
- `v3h_gc_sro_geometry_gate_brief.md`：证明旧 K=6 数据只有一个布局，枚举 28 种合法布局，定量审计布局对 ridge 误差与算子条件的影响，并锁定 GC-SRO 零初始化/冻结基线/置换不变工程闸门。
- `v3i_variable_geometry_dataset_brief.md`：328 场、28 布局的 one-field/one-geometry 平衡数据合同，明确 42 通道、共享噪声、Q_audit 不泄漏、私有 NPZ 边界和下一轮五组功能对照。
- `v3j_gc_sro_functional_negative_result_brief.md`：12 次等参数训练否定当前全局几何调制机制；分开 static adapter `+3.63%` 与 geometry 零增益，定位 embedding → spatial correction 衰减，并给出 v3k 同场多布局反事实监督决策树。
- `v3k_a_counterfactual_supervision_brief.md`：24 次等曝光训练比较 M1 单布局重复与 M4 同场四布局；correct-vs-shuffled 恢复到 `+0.0675%` 但低于 `+0.25%` 门槛，swap 显示 correction 改变 `8.14%` 而最终场收益近零，因此停止全局调制扩参并转体素级 ray-set 分支。
- `v3k_b_voxel_ray_set_negative_result_brief.md`：12 次 M4 匹配训练表明局部 ray 相对 geometry-only 有 `+1.44%` 收益，但正确 ray-angle 配对相对同信息 pairing-shuffle 为 `-0.20%`；停止 attention 扩参并转向 adjoint-residual / Landweber 物理方向。
- `v3k_c_adjoint_landweber_gate_brief.md`：修复 soft-support 非幂等混杂与 selection Q_audit 泄漏后，28 布局伴随/梯度校验通过；projected Landweber 在 validation 相对 feasible FNO 改善 `+44.60%`，四个 development test domains 的最小平均收益为 `+22.40%`。同时锁定 global-step/line-search/ridge-start/lookup 强对手、worst-layout 尾部和 conditional scalar 四周路线；不作新算法或 superiority 主张。
- `v3d_geometry_data_manifest.md`：可直接给何远哲勾选的 geometry/data 最小合同、坐标约定、数据角色与 Go/No-Go 判决。
- `demo_t16_operator/run_v3d_fno_saturation_audit.py`：每 12 epochs 审计 validation 改善、保留不劣 checkpoint，并在停止点冻结后才计算 dev2 field/Q_audit diagnostics；支持 analysis-only 重算和 checksum validator。
- `fair_camera_budget_review_brief.md`：v2d 给师兄的一页审核简报，说明 0/3 门槛、训练 mask 不匹配、K=6 max-gap 线索和主线转向。
- `operator_3d_innovation_lab.html`：算子学习 × 三维重建创新实验室，集中展示 30 次训练、support-fit/range-null/query calibration、54 篇角色化主线文献和可证伪实验矩阵。
- `view_reliability_dual_branch_operator_proposal.md`：support-consistent 双分支神经算子完整研究提案，包含原型负结果、闭式物理锚点、identifiability、nullspace correction、support/query 拆分、baseline、消融和停止条件。
- `operator_3d_reconstruction_literature_atlas.md`：算子学习、几何感知逆成像、少视角 3D/4D、自监督、专家路由/UQ 和稀疏流场重建文献图谱。
- `operator_3d_source_audit_2026_07_10.md`：本轮核心来源的正式页面、证据等级、跨模态迁移边界与不应写进论文的越界句。
- `t16_reliability_gate_review_brief.md`：第一轮 fixed/learned gate 筛查的师兄简报，明确负结果、oracle regret、可复现性与下一模型依据。
- `operator_learning_bost_experiment_spec.md`：T16 可执行实验规格，定义数组、physics lift、loss、phantom-family/OOD split、baseline、图表、显存降级与停止条件；已接入 M3B 的 view/noise condition-cell、失败格和 field/held-out/积分量分叉证据。
- `cai_oerf_research_genealogy.md`：蔡伟伟 / OERF 研究谱系，把 BOST、光谱层析、计算光谱、等离子体、数据同化等方向翻译成毕设优先级。
- `cai_project_alignment_map.md`：蔡伟伟公开项目语言到本科毕设题目的对齐地图，把导师主页、OERF 方向和何远哲主线翻译成可沟通题目、开题段落和师兄问题。
- `oerf_direction_backlog.md`：OERF 全方向本科毕设研究问题池，把 BOST、PIV-BOST、4D、光谱、全息、CFD、Agent 等方向拆成可选/备选问题。
- `oerf_publication_direction_index.md`：OERF 公开论文-方向-毕设切入索引，按主线、方法背景、旁支和待核引用整理代表作。
- `oerf_latest_direction_triage.md`：OERF 最新论文方向分流表，说明 4D LII、U-Net 火焰前沿、全息追踪和金属颗粒燃烧如何作为背景或备选，而不冲散 He/BOST 主线。
- `computational_flow_visualization_bridge.html`：计算流动显示桥梁，把 Cell Reports Physical Science 2024 的 hidden properties 语言转成 BOST/NeRIF/PIV-BOST/4D BOST/M3C/Agent for Science 的开题站位。
- `agent_for_science_bost_bridge.html`：Agent for Science 辅助 BOST pipeline，把科研智能体落成参数扫描、数据质量检查、学习日志和会前报告辅助层。
- `stereo_piv_bost_bridge.html`：Stereo PIV-BOST 桥梁，把 2025 simultaneous PIV-BOST 与 2026 stereo-velocity PIV-BOST 转成 raw image、calibration/disparity、velocity-vector 三层补偿路线和本科降级方案。
- `light_field_single_camera_bridge.md`：光场 / 单相机内窥 tomography 桥梁说明，解释它和 BOST/NeRIF 的关系、可借鉴的视角/标定/硬件编码问题，以及为什么不建议替代何远哲主线。
- `bost_system_error_bridge.html`：BOST 真实数据质量与系统误差桥梁，把 view geometry、mask/ROI、displacement confidence、spatial resolution、calibration 和权限字段整理成 M3C 质量控制路线。
- `opening_report_draft.md`：本科毕业设计开题报告正文初稿，可按学院模板调整。
- `opening_ppt_outline.md`：10 页开题 PPT 逐页大纲，包含每页标题、图、讲法和答辩问题。
- `defense_qna_bank.md`：开题答辩问答库，覆盖创新性、物理模型、数据、PIV-BOST、4D BOST 和风险边界。
- `thesis_blueprint.md`：毕业论文蓝图，包含 6 章目录、图表清单、结果叙事和诚实边界。
- `figures/`：原创 PPT / 论文图示素材和生成脚本，包括 BOST 物理链条、NeRIF pipeline、数据接口图和选题决策树。
- `he_meeting_decision_pack.md`：何远哲会前决策包，把 M0-M3B demo 压缩成 15 分钟汇报、A/B/C 主线与 E 副线定题路径和会后 7 天动作。
- `meeting_notes_template.md`：何远哲沟通纪要模板，用于记录数据、baseline、限制和下一周交付。
- `starter_code_spec.md`：BOST / NeRIF 第一周最小代码闭环规格。
- `nerif_bost_implementation_blueprint.md`：NeRIF / BOST 毕设实施蓝图，包含仓库结构、数组形状、模块接口、指标、实验命名和 30 天 coding plan。
- `experiment_matrix_and_figure_plan.md`：实验矩阵与论文图表计划，把 M0-M3B、open BOS、真实数据迁移和待做实验对应到论文图表与论点。
- `experiment_templates/`：BOST/NeRIF 实验配置模板，用于统一 dataset、geometry、method、metrics 和 output。
- `minimum_demo_protocol.md`：BOST / NeRIF 最小 demo 实验协议，从 2D toy 到 3D synthetic、PIV/4D 升级。
- `weekly_checkpoint_board.md`：12 周推进验收板，把预研拆成每周图、代码、问题和交付。
- `demo_m0/`：已跑通的 2D BOST / coordinate-field toy demo，包含脚本、结果总图、视角数曲线和指标 CSV。
- `demo_m1/`：已跑通的 3D-stack sparse-view BOST toy demo，包含三维体切片、视角数曲线、指标卡和 CSV。
- `demo_m2/`：已跑通的鲁棒性扫描 demo，系统比较视角数、噪声和 coordinate regularizer 容量对误差的影响。
- `demo_m3a/`：已跑通的 PIV-BOST 速度补偿 toy demo，用向量场误差传播说明折射率补偿的基本价值和残余风险。
- `demo_m3b/`：已跑通的 4D BOST low-rank temporal toy demo，用时序三维 phantom 比较逐帧重构和低秩时序先验。
- `he_paper_deep_dive.md`：何远哲主线论文深挖，把 NeRIF / PIV-BOST / 4D BOST 转成可复现实验和开题章节。
- `he_author_audit.md`：何远哲作者谱系审计，区分主线论文、OERF 旁支和同名索引噪声。
- `paper_to_demo_map.md`：核心论文到 M0-M3B demo、论文章节、师兄问题和可交付产物的映射表。
- `core_paper_reading_cards.md`：核心论文精读卡片集，把每篇论文拆成必须抽取的公式、图、参数、demo 改动和师兄问题。
- `foundation_bridge.md`：物理本科到 OERF / BOST-NeRIF 的基础补课路线。
- `undergrad_foundation_bootcamp.md`：物理本科到 BOST/NeRIF 的 12 周基础训练营，把流体、燃烧、BOST、反问题、PIV、PyTorch 和神经场转成每周任务。
- `reading_log_template.md`：论文阅读追踪表，防止阅读无法转化成公式、实验和问题。
- `data_request_checklist.md`：向何远哲请求 BOST / PIV-BOST / 4D BOST 数据和代码的清单。
- `open_data_code_benchmark_map.md`：公开数据与开源代码 benchmark 路线图，用于无组内数据时的 open-source BOS/TBOS 预演。
- `code_data_reuse_audit.md`：BOST / NeRIF 公开代码与数据复用审计，区分 MIT/GPL/no-license 仓库、可用公开数据集和不能放入网页仓库的大文件。
- `open_bos_dataset_access_protocol.md`：Open-source BOS tomography dataset 的下载体量、目录结构、最小子集、70-view manifest 和 OERF 数据迁移协议。
- `tdbost_module_mapping.md`：TDBOST 公开仓库到 M3B/M3C 的结构映射，明确无 LICENSE 边界和安全借鉴方式。
- `tdbost_decision_bridge.md`：4D BOST / TDBOST 决策桥梁，把 TOG 主线拆成本科可做子问题、风险边界和师兄问法。
- `tdbost_reproducibility_audit.html`：4D BOST / TDBOST 可复现性渲染页，基于 ACM 正式 HTML/eReader 全文整理方法、实验、显存、消融与 paper-vs-code 差异，并展示 M3B 场切片、rank scan、时序轨迹、交互指标和下一轮系统误差实验矩阵。
- `m3b_six_axis_dashboard.html`：M3B 六轴多种子实验页，展示 448 条 rank/noise/frame/view/bias/dynamics 配对结果、95% CI、三张统计图、失败边界和原始 CSV/JSON/脚本入口。
- `m3b_interaction_dashboard.html`：M3B 交叉验证页，展示 rank × noise × views × dynamics 的 80 个环境、8 个配对种子、3,200 组配对结果、固定-rank regret、rank 3 operating map 与 field/mass 物理代价。
- `m3b_interaction_brief.md`：M3B 交叉实验一页沟通 brief，供何远哲快速判断自适应 rank 选择、真实数据和下一轮验证优先级。
- `m3b_family_selector_dashboard.html`：M3B 跨形态无真值容量选择页，展示 4 morphology × 3 dynamics × 4 noise × 3 views × 6 seeds、864 个观测格、nested-LOFO selector、full-rank 拒绝平滑、六张图和特征消融。
- `m3b_family_selector_brief.md`：最新 M3B 一页审核 brief，供何远哲判断 observable capacity selection、held-out camera、geometry transfer、UQ/拒答和真实 4D 数据优先级。
- `demo_m3b/run_m3b_family_selector.py`：可断点续跑和从 raw CSV 快速再分析的正式跨形态实验脚本；`validate_family_selector_results.py` 独立核验 864 格/6,048 候选、LOFO family exclusion 与 forbidden-feature absence。
- `topic_deep_dive.html?id=T16`：第 16 个毕设方向档案，把神经算子与何远哲 BOST 三维重建主线接起来。
- `data_templates/`：真实 BOST / PIV-BOST 数据 manifest 模板、Open BOS 70-view 索引摘要、推荐目录和校验脚本。
- `topic_decision_matrix.md`：备选毕设题目的评分和取舍矩阵。
- `source_audit.md`：来源与 DOI 审计记录，集中保存题名/DOI 冲突、PDF 缓存边界和正式引用优先级。
- `oerf_pdf_gap_sweep_2026_07_08.md`：OERF / He 高价值论文 PDF 缺口复核，说明哪些核心条目只能放 DOI/出版社入口，不能复制 PDF 到公开网页。
- `assets/backgrounds/`：本地生成的折射率场/层析光线低透明度背景图，用于主页视觉氛围，不使用人物头像。
- `qa/`：本地视觉检查临时目录。旧版含人物头像的首屏预览图已删除；后续若重新生成 QA 图，只保留无人物头像版本。

## 推荐使用顺序

1. 先看 `总判断`，确认主线为什么是 BOST / NeRIF / PIV-BOST / 4D BOST。
2. 看 `全方向雷达` 和 `oerf_direction_backlog.md`，理解 OERF 所有公开方向里哪些能做、哪些只适合作背景。
3. 看 `项目线索` 和 `cai_project_alignment_map.md`，理解为什么这个方向贴合蔡老师公开项目，以及该如何对导师/师兄分别表达。
4. 看 `何远哲同步线` 和 `方法邻居`，弄清楚 NeRIF、NeDF、PIV-BOST、4D BOST 的关系。
5. 读 `paper_library/index.html`、`pdf_cache_audit.html`、`he_paper_deep_dive.md`、`bost_system_error_bridge.html`、`paper_to_demo_map.md`、`core_paper_reading_cards.md` 和 `oerf_publication_direction_index.md`，把 NeRIF / PIV-BOST / 4D BOST 与 OERF 公开代表作、真实数据质量控制、可执行 demo 和 PDF/代码公开边界对上。
6. 按 `foundation_bridge.md` 和 `undergrad_foundation_bootcamp.md` 补最小流体、燃烧、光学诊断、反问题和 PyTorch 基础。
7. 用 `选题库`、评分模型、`topic_decision_matrix.md` 和 `topic_execution_playbook.html` 筛出 2-3 个备选题，并把每个题目拆成最小毕业版、升级版和放弃规则。
8. 先打开 `demo_m0/results/m0_summary.png` 看已跑通的 M0 demo，再看 `demo_m1/results/m1_volume_summary.png` 和 `demo_m2/results/m2_improvement_heatmap.png` 理解少视角 3D 栈重建的边界；最后看 `demo_m3a/results/m3a_compensation_summary.png` 和 `demo_m3b/results/m3b_4d_summary.png` 理解 PIV-BOST 与 4D BOST 两条升级线。
9. 用 `nerif_bost_implementation_blueprint.md`、`experiment_matrix_and_figure_plan.md`、`starter_code_spec.md` 和 `experiment_templates/` 规划第一版代码仓库、模块接口、实验矩阵和论文图表。
10. 用 `weekly_checkpoint_board.md` 规划 12 周，每周留下一个图、一个文件、一个问题。
11. 用 `reading_log_template.md` 管理论文阅读，用 `open_data_code_benchmark_map.md`、`code_data_reuse_audit.md`、`open_bos_dataset_access_protocol.md`、`tdbost_module_mapping.md`、`data_request_checklist.md` 和 `data_templates/` 准备开源数据预演、4D BOST 结构拆解、代码/数据许可边界与向何远哲要数据。
12. 找何远哲沟通前，优先发 `he_review_portal.html`；自己准备时再读 `he_review_feedback_pack.md`、`advisor_three_page_brief.md`、`top_topic_proposal_cards.md` 和 `he_meeting_decision_pack.md`，再复制 `师兄沟通`里的消息草稿和会后纪要模板。
13. 若按师兄建议主攻“三维重建 + 算子学习”，先打开 `operator_nullspace_evidence_dashboard.html`，再读 `t16_geometry_nullspace_review_brief.md` 与 `view_reliability_dual_branch_operator_proposal.md`；不要从模型名字开始读。
14. 如果需要向蔡老师解释自己为什么贴 OERF，先看 `cai_oerf_research_genealogy.md` 和 `cai_project_alignment_map.md`，把主线、支线、项目语言和题目边界讲清楚。
15. 做开题汇报时，用 `opening_report_draft.md` 写正文，用 `opening_reference_shortlist.md` 整理参考文献，用 `opening_ppt_outline.md` 组织 10 页 PPT，用 `defense_qna_bank.md` 准备问答。
16. 用 `figures/` 里的原创图示快速搭建 PPT 和论文第 1-3 章。
17. 开始写论文时，用 `thesis_blueprint.md` 先写第 1-2 章和图表清单。
17. 需要发材料时，优先使用 `advisor_three_page_brief.md` 或 `one_page_proposal.md`，不要直接把完整网页丢给对方。

## 当前首选题目

建议首选：

> 面向少视角背景纹影层析的神经隐式折射率场重构与鲁棒性分析

保底成果：

- 合成 BOST 数据生成。
- 传统体素/正则化 baseline。
- 简化 NeRIF 神经场重构。
- 视角数、噪声、编码、采样策略的参数扫描。
- 重投影验证和误差图。

冲刺成果：

- 接入 OERF 真实九视角 BOST 数据。
- 增加时间不同步、缺失视角、坏视角和投影补全的误差图谱。
- 做 PIV-BOST 折射率补偿误差传播。
- 做 4D BOST 低秩时序 toy 子问题。

## 当前可运行 demo

M0 demo 已经跑通：

```bash
cd oerf-bishe-dashboard
python3 demo_m0/run_m0_2d_bost.py
```

输出文件：

- `demo_m0/results/m0_summary.png`
- `demo_m0/results/view_count_curve.png`
- `demo_m0/results/metrics.csv`
- `demo_m0/results/view_count_metrics.csv`

当前 9 视角结果：baseline relative L2 约 0.113，coordinate-field inverse relative L2 约 0.108。这个 demo 只能证明最小闭环已经跑通，不代表完整 NeRIF 复现。

M1 demo 也已经跑通：

```bash
cd oerf-bishe-dashboard
python3 demo_m1/run_m1_3d_stack_bost.py
```

输出文件：

- `demo_m1/results/m1_volume_summary.png`
- `demo_m1/results/m1_metrics_card.png`
- `demo_m1/results/m1_view_count_curve.png`
- `demo_m1/results/metrics.csv`
- `demo_m1/results/view_count_metrics.csv`

当前 5 视角 sparse-view 结果：baseline relative L2 约 0.257，3D coordinate-regularized stack relative L2 约 0.234。视角数扫描显示：坐标正则化在 3-5 视角有帮助，但在干净的 7/9/13 视角设置下传统栈重建更强。这是一个很适合开题的诚实结论：隐式表示的价值要放在少视角、噪声、mask、真实几何和数据融合约束里讨论。

M2 demo 已经跑通：

```bash
cd oerf-bishe-dashboard
python3 demo_m2/run_m2_robustness_scan.py
```

输出文件：

- `demo_m2/results/m2_noise_view_scan.png`
- `demo_m2/results/m2_improvement_heatmap.png`
- `demo_m2/results/m2_capacity_scan.png`
- `demo_m2/results/m2_metrics_grid.csv`
- `demo_m2/results/m2_capacity_metrics.csv`

当前扫描结果：3/5 视角的 8 个噪声设置里 coordinate regularizer 全部胜出；7/9 视角的 8 个设置里传统 stack baseline 全部胜出。容量扫描显示，Fourier feature 太少会过度平滑，容量提高到约 120 以后才开始超过 5 视角 noisy baseline。

M3A demo 已经跑通：

```bash
cd oerf-bishe-dashboard
python3 demo_m3a/run_m3a_piv_bost_compensation.py
```

输出文件：

- `demo_m3a/results/m3a_compensation_summary.png`
- `demo_m3a/results/m3a_error_profile.png`
- `demo_m3a/results/metrics.csv`

当前 PIV-BOST toy 结果：观测 PIV 速度 RMSE 约 0.0101，BOST-style compensation 后 RMSE 约 0.0067；95 分位误差从约 0.0240 降到约 0.0072，但局部最大误差仍可能因折射位移估计噪声和端点近似而残留。

M3B demo 已经跑通：

```bash
cd oerf-bishe-dashboard
python3 demo_m3b/run_m3b_4d_lowrank_bost.py
python3 demo_m3b/run_m3b_six_axis_sweep.py
python3 demo_m3b/run_m3b_interaction_sweep.py
```

输出文件：

- `demo_m3b/results/m3b_4d_summary.png`
- `demo_m3b/results/m3b_rank_scan.png`
- `demo_m3b/results/m3b_temporal_trace.png`
- `demo_m3b/results/metrics.csv`
- `demo_m3b/results/rank_metrics.csv`
- `demo_m3b/results/m3b_six_axis_overview.png`
- `demo_m3b/results/m3b_rank_seed_stability.png`
- `demo_m3b/results/m3b_bias_dynamics_diagnostic.png`
- `demo_m3b/results/six_axis_raw.csv`
- `demo_m3b/results/six_axis_summary.csv`
- `demo_m3b/results/six_axis_paired_improvements.csv`
- `demo_m3b/results/six_axis_report.json`
- `demo_m3b/results/m3b_interaction_heatmaps.png`
- `demo_m3b/results/m3b_rank_selection_stability.png`
- `demo_m3b/results/m3b_interaction_tradeoffs.png`
- `demo_m3b/results/m3b_rank3_operating_map.png`
- `demo_m3b/results/interaction_raw.csv`
- `demo_m3b/results/interaction_paired.csv`
- `demo_m3b/results/interaction_cell_summary.csv`
- `demo_m3b/results/interaction_rank_selection.csv`
- `demo_m3b/results/interaction_report.json`

当前 4D low-rank toy 结果：逐帧 baseline mean relative L2 约 0.366，low-rank rank 3 后约 0.347；temporal smoothness 从约 0.279 降到约 0.177；centroid trajectory RMSE 从约 0.0401 降到约 0.0381。它说明低秩时序先验能降低逐帧抖动，但不能自动修正系统性几何/重建偏差。

正式 8 种子六轴实验进一步确认：rank 3 仍是全局 norm-ratio L2 最优点；默认场 L2 改善 3.90%、论文式 squared L2 改善 7.65%、时间一阶/二阶误差改善 44.68%/73.19%、held-out deflection 改善 9.23%，但 mass trace 恶化 0.94%，且无噪声时场 L2 恶化 0.56%。这使 M3B 从单次演示升级为可向师兄讨论的适用域/失败边界实验。

进一步完成的 rank × noise × views × dynamics 交叉实验覆盖 80 个环境、8 个配对种子、3,840 条方法记录和 3,200 组配对比较。逐格最优 rank 分布为 rank 2/3/5/8 = 27/20/24/9；rank 3 虽只在 20 格最优，却以 0.79% mean regret 和 2.21% p95 regret 成为最稳固定默认值。它在 61/80 格改善 field、19/80 格恶化 field；116/400 个环境×rank 单元出现 field 改善但 mass trace 恶化。该结果支持“按可观测量自适应选 rank”的本科子问题，但仍只代表单一 synthetic phantom 族，不代表真实 TDBOST 泛化。

## 当前图示素材

可直接用于开题 PPT 或论文草稿：

- `figures/bost_physical_chain.png`
- `figures/oerf_position_map.png`
- `figures/nerif_pipeline.png`
- `figures/data_interface_checklist.png`
- `figures/topic_decision_tree.png`
- `figures/three_month_roadmap.png`

需要修改图中文字或样式时，运行：

```bash
cd oerf-bishe-dashboard
python3 figures/generate_figures.py
```

## 当前真实数据接口模板

拿到组内样例数据后，优先按 `data_templates/` 整理：

- `data_templates/bost_sample_manifest.json`
- `data_templates/piv_bost_manifest.json`
- `data_templates/open_bos_manifest.json`
- `data_templates/open_bos_index_summary.json`
- `data_templates/open_bos_view_manifest.csv`
- `data_templates/open_bos_view_plan.md`
- `data_templates/open_bos_subset_plans.json`
- `data_templates/open_bos_subset_plans.md`
- `data_templates/validate_manifest.py`

只检查字段、不要求文件已存在：

```bash
cd oerf-bishe-dashboard
python3 data_templates/validate_manifest.py data_templates/bost_sample_manifest.json --allow-missing
```

真实文件放好后，去掉 `--allow-missing` 检查缺失路径。

无组内数据时，先用 Open BOS 公开数据做轻量预演：

```bash
cd oerf-bishe-dashboard
python3 data_templates/summarize_open_bos_index.py
```

当前解析结果显示：Open BOS 的 calibration 是 13 个 `Angle_*` 角度 x 7 台相机；论文 reported 70 views 对应 10 个 `ROT_***` flow groups x 7 台相机。`data_templates/open_bos_view_manifest.csv` 已把 70 行 REF/DEF image-pair、CC/HSOF/WOF40/mask 路径提示列好，但 ROT-to-Angle 几何映射需要从论文或脚本确认，不能只从目录名推断。

生成 5/7/9/13/21/70 views limited-view 预案：

```bash
cd oerf-bishe-dashboard
python3 data_templates/plan_open_bos_subsets.py
```

## 最新主线：v5 有限孔径算子失配

- `demo_t16_operator/v5a_blind_aperture_prelock_protocol.md`：首开前冻结的阈值、调用预算、门槛与源码哈希。
- `demo_t16_operator/run_v5a_blind_aperture_calibration.py`：明确 `A_truth != A_reconstruction` 的有限孔径首开 runner。
- `demo_t16_operator/validate_v5a_blind_aperture_calibration.py`：独立复算 16,103 项证据一致性检查。
- `v5a_blind_aperture_first_open_review.md`：讲清楚为什么平均 +1.753% 仍判失败，以及 morphology/operator confounding。
- `demo_t16_operator/v5b_rig_shared_profile_calibration_protocol.md`：下一候选的数学、数据、基线、统计与 Go/No-Go。

v5a 已封存为负结果：五项预注册门槛只过两项，p10 为 -17.784%，总体 harm 25%，接受条件 harm 37.5%。这不是“差一点成功”，而是 residual-margin 风险分数未能跨未见反应场泛化。v5b 不允许使用 v5a lock 调阈值，必须新建配对析因 development 和新的 confirmatory lock。

## 维护规则

- 新增论文时，先确认 DOI / arXiv / 出版社页面，再写入 `references.bib`。
- OERF 网站源码中的论文条目如果和出版社元数据冲突，正式引用以出版社/DOI 为准。
- 遇到 DOI 或题名冲突时，先写入 `source_audit.md`，确认后再决定是否进入 `references.bib`。
- 不要把光谱仪、金属颗粒、CFD 旁支混进主线，除非导师或何远哲明确要求转向。
- 每次和师兄沟通后，把数据权限、baseline、评价指标和下一周交付写进纪要。
- 每读一篇核心论文，都在 `reading_log_template.md` 里转成一个公式、图表或最小实验。
- 每补一块基础课，都要在 `foundation_bridge.md` 的链条里对应到 BOST 变量或代码产物。
- 每周至少按 `weekly_checkpoint_board.md` 产出一个可展示图，否则说明预研没有沉淀成毕设成果。
- 不要等实验全部完成再写论文；按 `thesis_blueprint.md` 先写背景、物理模型和方法边界。

## 验证记录

- HTML 解析通过。
- 导航锚点、本地链接和图片资源检查通过。
- 脚本语法检查通过。
- M0/M1/M2/M3A/M3B demo、图示脚本和 manifest 校验脚本均可运行。
- M2 robustness scan 可运行，并生成 noise-view scan、improvement heatmap 和 capacity scan。
- 旧 QuickLook 首屏预览图已删除，避免继续显示早期四位成员头像；主页背景和 M1 结果图资源均存在。
- 应用内浏览器出于安全策略拒绝直接打开本地 file 页面，因此未做交互式浏览器 QA。
