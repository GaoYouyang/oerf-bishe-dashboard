# LGWO-A24 R2-F0 冻结前红队回应

- 日期：2026-07-21
- 状态：`HOLD_R2F0_PROTOCOL_NOT_READY_TO_FREEZE`
- 正式 case：`0`
- 科学结论：无
- 突破：无

这份文件逐项回应第一轮红队，不是结果报告。只有新版独立 validator 通过、第二轮红队不再发现 P0/P1、source closure 被提交并写入配置后，才可以讨论把配置从 `HOLD` 改为 `FROZEN`。即使未来 R2-F0 通过，它也只说明“固定可观察方向在六个已见 synthetic cases 上存在离线表示余量”，不等于可部署算法、真实 BOST 重建、泛化或论文成功。

## P0 回应

| 原问题 | 当前处理 | 证据 | 剩余边界 |
|---|---|---|---|
| C5-C1 把百分点错算成相对百分比 | 改为“候选相对 H1-20 的 gain”减“旧 P20 oracle 相对 H1-20 的 gain”，门槛为 3 个百分点 | runner 聚合键使用 `*_percent_points`；单测构造 C1=0.20、C5=0.19，只得到 1pp 并 fail | 仍需独立 validator 复算 aggregate |
| union synergy 门为 0 且未进入总门 | 门槛冻结为 1pp；RH/RE/EH/RHE 必须同时在 field、gradient 上超过最佳真子空间，并折回 `representation_gate_passed` | 零协同 synthetic test 必须 fail | 该门只防止“堆方向”，不证明物理新颖性 |
| MPS float32 承担关键 reduced algebra | projector/SVD、joint field-gradient objective、误差归一化、Gram/RHS、reduced coefficient、residual ray、field trust 全部在 CPU float64 计算；仅把最终方向/系数恢复到输入设备和 dtype | core tests 检查 coefficient 常驻 CPU float64，并检查 sufficient statistics 为 float64；MPS 测试在可用机器上运行 | 原始物理算子输出仍为 MPS float32，因此三档 floor 必须稳定 |
| DirectionPacket hash 不完整且在 oracle 后才落盘 | packet 绑定 operator/ray/lambda/spacing、support/weights、P20 field/residual/bases、raw/projected directions、projector coefficients、全部常数和 measured call ledgers；每 case 在 truth oracle 打开前 fsync、只读落盘并记录 monotonic 顺序 | `direction_packets/*.json` 预期 seal；oracle 后重新哈希；独立 manifest 预期复核 | 时间顺序是单进程文件证据，不是可信硬件证明 |
| F/A^T 只是手填常量 | 每个方向组件由 wrapper 前后 delta 实测；每个 family 的 standalone vector 由实测组件组成；40 方法 × 6 case 另存逐方法 ledger | runner 对实测 vector 与冻结 vector 逐项比较 | H1-21/22/23 是一次 H1-23 路径的 logical prefix，同时保留 parent 23/23 证据 |
| 配置可以误翻成 frozen | HOLD 要求 `frozen_date=null` 且 source bindings 为空；FROZEN 必须同时绑定完整本地 import closure | 状态翻转但 closure 为空的测试必须拒绝 | validator 完成和二次红队之前不会填 bindings |

## P1 回应

| 原问题 | 当前处理 | 当前状态 |
|---|---|---|
| projector 与 oracle floor 没有端到端耦合 | `1e-5/1e-6/1e-7` 各自重算 projected directions、方向 acceptance/fraction、span、oracle coefficients、safe ray、metrics、rank 和 passing-family set；NPZ 预期保存每档方向与 sufficient statistics | 生成端已实现；等待新版 validator 独立复算 |
| permutation 只检查 SVD | 对 R/H/E/Q 及 unions 的新方向排列，重新计算 unrestricted、residual-safe、residual+field-safe 三个 endpoint，再比较 field、safe fraction、field/gradient metrics | 相关双方向单测已覆盖；正式 case 尚未打开 |
| tiny raw direction 可凭比例过门 | 增加 `minimum_raw_field_norm=1e-10` 与 `minimum_raw_augmented_norm=1e-10` | `1e-12` 方向即使 outside fraction=1 也必须拒绝 |
| case 矩形和 finite 不完整 | metric 必须是精确 6×40 唯一矩形；direction 必须 6×4；三档 direction 必须 3×6×4；novelty 必须 6×8；任意 NaN/Inf 在 aggregate/CSV/NPZ 前触发 INVALID，coefficient padding 使用零而不是 NaN | duplicate-case 与 NaN tests 已覆盖 |
| 原子证据不足 | 临时目录维护 failure stage；prevalidation manifest 覆盖全部科学产物；validator 通过后 final manifest 再覆盖 validation、stage 和 prevalidation manifest，最后原子 rename | 等待 validator 对 prevalidation manifest 的独立篡改测试 |
| invalid notice 信息太少 | notice 记录 failure stage、completed case count、config hash、已有 artifact/hash、CSV 行数、source 与 geometry 复核 | 只用于失败审计，不能转化为科学 NO-GO |

## P2 回应

1. `truth_accessed` 拆出 `construction_truth_accessed` 与 `evaluation_truth_accessed`。H1 endpoint 构造不读 truth，但所有 field/gradient 指标评估都读 synthetic truth；oracle endpoint 构造和评估都读 truth。
2. Huber-vs-quadratic、edge-gated-vs-raw 的 attribution 不再以“均值略大于 0”成立；两项指标均至少高 1% 才进入表示门。
3. 报告不再写含混的 `family_selection_truth_free=false`，改为 `family_set_prebound_before_oracle=true` 和 `deployable_family_selection_performed=false`。

## 第二轮红队后的协议加固

第二轮审计开始时又找到两项 P0 和五项 P1。它们没有被解释成“以后再说”，而是继续在 `HOLD` 下逐项关闭：

1. `RHE` 的协同收益现在必须超过六个真子空间：`R/H/E/RH/RE/EH`。旧实现漏掉三个双方向，会让三方向联合与不完整对手比较。
2. Huber-vs-quadratic 与 edge-vs-raw attribution 严格使用冻结的 `1%` 门；`0.5%` 正增益必须失败。
3. 新方向不再只看 pairwise cosine。runner 用 PyTorch float64、validator 用 NumPy 独立复算 canonical correlation、最小主角、实际秩增量 `rank([P20,Q])-rank(P20)` 及三档 floor 稳定性。
4. 系数、安全 ray、field/gradient 指标和最终门限判决全部留在 CPU float64；只有送入物理算子的显式 endpoint 转到设备 dtype。封包同时保存 decision endpoint 与 physical endpoint 的数组及哈希。
5. 三档 rank floor 都重走投影、物理 endpoint、oracle、安全 ray 和指标链；方向排列测试也重走整条链，不再只比较 SVD 或 rank。
6. 状态、11 项全 false claim boundary、R2-E0/R2-D0/R2-D1 先验证据及 17 文件源码闭包均使用精确合同；过度成功状态、漏 claim、旧证据和源码篡改已有负测。
7. case 使用预绑定的 32 字符 opaque token，执行阶段另记 direction sealed、oracle opened、packet written；token 不是可信硬件，但不再从 case 序号可逆推导。
8. 所有连续候选门都有冻结的数值歧义 margin。任何候选太靠近门槛时，协议返回 `R2F0_NUMERICAL_AMBIGUITY_NO_DECISION`，不能借 CSV 容差或舍入翻成 PASS。
9. 正式包要求携带离散物理核证据。独立 validator 不导入 runner 或 PyTorch，而是用 NumPy 从三线性采样索引、权重、投影坐标和 ray scale 重建公开 PSU 的 `A/A^T`，复算 40 个主投影、三档 floor 的 24 个投影、伴随探针及内积恒等式。小型正例与投影/伴随篡改负例已经通过；正式 R2-F0 包仍未产生。

当前定向套件为 `77 passed`，Ruff、`py_compile`、JSON 状态检查和 `git diff --check` 均通过。这个数字只证明软件合同和篡改测试，不是 77 个实验，更不是算法性能证据。

## 冻结前剩余门

1. 第二轮只读红队必须重新检查 runner、core、config、validator 和测试；任何未关闭 P0/P1 都维持 HOLD。
2. source closure、commit HEAD、配置哈希和 geometry bundle 必须在同一提交中预绑定。
3. 仅做 `--preflight-only`；确认正式目录和 `.incomplete` 都不存在后，才允许一次正式 reused-case run。
4. 正式 run 无论 PASS、NO-GO、numerical ambiguity 或 INVALID，都不允许偷偷改同一 output ID 重跑。
5. 即便 reused-case representation audit 过门，真正面向论文的下一道门仍是何远哲师兄确认真实 callable、straight/curved residual 层级、JVP/VJP、标定、主痛点、认可基线、数据 split 与宿主合同；在此之前不能替实验室造真实结论。

## 给高阳同学的一句话

这轮没有让模型“更强”，而是把几种很容易看成创新的假象关掉了：truth 重配旧系数、百分比口径错误、多个高度相关方向叠加、float32 数值尾巴、重复 case、NaN 和手填调用成本。只有这些假象都被关掉，后面出现的正结果才值得继续投入训练；出现 NO-GO 也同样是有用证据。
