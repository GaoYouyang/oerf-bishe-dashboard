# Factor-PDHG Gate B 严格 NO-GO：局部静态 metric 不足以替代 Krylov 耦合

日期：2026-07-17
正式状态：`GATE_B_E2_MECHANISM_NO_GO`
源码提交：`204bbe8423f4c08d023810d30d48b9ca3a049090`
独立复核：`PASS_INDEPENDENT_GATE_B_RECOMPUTATION`，4,048 项检查

## 1. 一句话结论

真正的 A-only voxel-factor metric 在两个 replicate、八类解析反应场上给出了小而
一致的正信号，但只有 **1.321%** mean field-error reduction，远低于预先冻结的
25% 机制门；它与同 `K=32` graph-PCGLS 的平均误差差距仍有 **133.439%**。
因此静态 factor-PDHG 路线关闭，不进入 TV/Huber、warm start 或 learned proximal。

这不是“PDHG 一般无效”，也不是“神经算子无效”。它只证伪了当前真实 PSU
geometry、oracle-scale opened synthetic 数据、零初始化、A-only、固定因子和
`K <= 32` 的这一条窄路线。

## 2. 到底比较了什么

正式样本为两个 opened replicate × 八类解析反应场：plume、wavy front、thin
front、double front、annular kernel、oblique shock、vortex pair 与 multi-plume。
所有方法看到相同的真实 PSU detector geometry 与同一组合成相关噪声。

四条轨迹在 `K = 4/8/16/32` 比较：

1. scalar A-only PDHG；
2. view-block A-only PDHG；
3. voxel-factor A-only PDHG；
4. 同次运行、单样本、exact-K graph-PCGLS。

前三条 PDHG 每个 checkpoint 严格使用 `K F + K A^T`，没有 TV 调用、提前停止或
`best <= K` 选择。三种 factor 方法共享同一个 A-only 活动 gauge：2,744 个 support
voxels 中 2,322 个与数据耦合，422 个处于 A-null support 并以零嵌回。活动索引
SHA-256 为 `57cc5748864d0bb3bffe0f971b5625e3f40a1dd87fed2f10a6166514e406d0f5`。

graph-PCGLS 保留其声明的 full-support Sobolev 表示，能够向 A-null support 外推；
这一不对称会让 factor 对 graph 的 full-field gap 更难看，因此明确披露，不把结果
写成普遍的算法排名。

## 3. 三次基础设施中止为什么不是“失败后改规则”

在任何 factor solver 或 factor truth metric 打开前，正式 runner 三次 fail closed：

1. production calibration mean 的 canonical shape 与 adapter flatten 假设不一致；
2. MPS batch 与 singleton graph replay 的浮点差被 front-F1 阈值放大；
3. 原 2,744 connectivity 来自 `A+D`，真正 A-only 数据耦合数是 2,322。

每次都满足：factor trajectories `0`、factor solver calls `0`、factor metric rows `0`、
正式结果目录未创建。修订只处理接口、非绑定 graph traceability 和正确 A-only
活动域；样本、checkpoint、更新式、调用预算、评分公式和八项决策阈值均未改。
V4 冻结后才第一次产生 factor 性能行。

## 4. 正式结果

| K=32 方法 | mean field-L2 | mean gradient-L2 | mean front top-10% F1 |
|---|---:|---:|---:|
| graph-PCGLS | 0.421042 | 0.545508 | 0.744330 |
| scalar A-only PDHG | 0.996031 | 0.995745 | 0.363073 |
| view-block A-only PDHG | 0.995241 | 0.994920 | 0.368902 |
| voxel-factor A-only PDHG | 0.982876 | 0.982061 | 0.136564 |

voxel-factor 相对 scalar 的 paired field gain 在 15/16 样本为正，两 replicate 的
mean gain 分别为 1.324% 与 1.318%，最坏样本仅 -0.00274%。这说明 factor 并非完全
随机：它确实改变了有限步收敛方向，而且信号可重复。

但是“稳定的小改善”不等于有论文价值：

- mean gain vs scalar：1.321%，门槛 25%，失败；
- mean gain vs view block：1.242%，门槛 3%，失败；
- mean error gap vs graph：133.439%，上限 20%，失败；
- K=4→32 mean field-L2 单调下降、15/16 正样本、两 replicate 为正、worst 与
  2.403× wall-time 门通过。

最终八门通过 5 项、失败 3 项，判 `NO-GO`。运行时间可接受不能抵消准确度门失败。

## 5. 物理与数值上学到了什么

### 5.1 局部异质尺度是真的，但不是主要瓶颈

voxel-factor 对 scalar 与 view-block 都有小正增益，支持“相机/voxel 耦合不均匀”
这个机制存在。但增益只有约 1%，说明静态对角尺度没有捕获主导的全局耦合。

### 5.2 graph-PCGLS 的优势主要来自全局 Krylov 方向

同样 32 对 `F/A^T` 调用，graph-PCGLS 已到 0.421，而三条 PDHG 仍接近零场。
factor 只重缩坐标；PCGLS 会利用历史残差构造共轭方向，近似全局谱信息。当前有限
预算里，后者远强于局部静态步长。

### 5.3 front 指标揭示了更严重的问题

voxel-factor 的 field 与 gradient 虽略改善，front-F1 却从约 0.36 降到 0.137。
这意味着当前更新优先移动了可降低总体 L2 的分量，却没有保住反应流最重要的薄层、
激波或火焰前缘。后续算法不能再只以 field-L2 选模。

### 5.4 422 个 A-null voxels 不是 epsilon 能解决的

这些自由度没有数据证据。将它们加 epsilon 留在 solver 中只会把任意 gauge 包装成
“可学习信号”。真实方法必须由明确的空间先验、时间演化、多模态观测或额外相机
补足，而不是由 factor certificate 偷渡。

## 6. 立即停止什么

1. 不把 FM-CG-PDNO 的 `beta=0` 基线写成“已改善”；Gate B 已阻断 neural smoke。
2. 不在同一 opened 数据上继续扫 factor exponent、`eta`、K 或阈值寻找过门结果。
3. 不加入 TV/Huber、graph warm start 后把收益归给 factor；那会同时改变目标和预算。
4. 不打开 fresh seed 来救一个已过不了 development mechanism gate 的分支。
5. 不宣称击败 DeepONet、FNO、FFNO、NeRIF 或 TDBOST；这些比较根本没有运行。

## 7. 下一步只保留三条有物理含义的分叉

### D0：根因诊断，不作为论文胜负实验

在 tiny/streaming opened development 上比较 exact-`|A|` 与 factor majorizer 的
row/column tightness，并画长时 `K=64/128` 轨迹。目的只回答“上界太松”还是“PDHG
递推本身在有限预算落后”，不得据此重开已关闭 Gate B。

### H2：RayKernel-DCO，当前最值得向师兄争取数据

把学习对象从 inverse solver 改成真实光学 forward discrepancy：有限孔径、景深、
cone/thin ray、曲线光路与标定漂移。需要两档光圈/焦平面、phantom 或 paired
low/high-fidelity renderer。先证明 discrepancy 大于 flow-off 噪声地板两倍，再让
小网络学习受约束 kernel correction；重建仍由强 graph-PCGLS/NeRIF baseline 完成。

### H3：TRAIL-4D，若师兄有连续高速序列则优先

用 timestamp、曝光积分与 dropout mask 建立时空 forward operator，让算子学习预测
时间演化或低秩时空系数，不替代每帧光学物理。主指标改为事件 p95、前缘时差、
topology birth/death 和换 cadence 泛化；单纯时间平滑必须作为强消融。

只有静态多视角且能永久留 `Q_audit` 相机时才重开 H1 near-nullspace 路线；此前
query-calibrated correction 已有 direct-view 强对照失败，优先级低于 H2/H3。

## 8. 现在发给何远哲的关键问题

1. 组内当前最痛的是有限孔径/标定 forward mismatch，还是 4D timestamp/缺帧？
2. 能否给两档光圈或焦平面、phantom、paired renderer 或已知密度场 CFD？
3. 是否保存每台相机未经平均的 flow-off/reference repeats、坏点与 confidence？
4. 是否有 50–200 帧连续序列及真实 timestamp、曝光和 dropout 日志？
5. 组内最终看重 field、front 位置/宽度、held-out displacement、PIV 补偿还是速度？
6. 能否固定一台相机和一个 session 永久不参与训练与选模？

## 9. 可复核入口

- [公开结果包](../demo_t16_operator/results/psu_b0_factor_pdhg_gate_b_public/README.md)
- [四联图 PDF](../demo_t16_operator/results/psu_b0_factor_pdhg_gate_b_public/factor_gate_b_no_go.pdf)
- [八项门禁 CSV](../demo_t16_operator/results/psu_b0_factor_pdhg_gate_b_public/decision_gates.csv)
- [独立验证 JSON](../demo_t16_operator/results/psu_b0_factor_pdhg_gate_b/validation_report.json)
- [公开分析脚本](../site_tools/analyze_psu_b0_factor_gate_b_no_go.py)

证据等级仍是 opened synthetic E2 mechanism diagnostic。fresh、真实 OERF、论文胜出
和 neural training 均未授权。
