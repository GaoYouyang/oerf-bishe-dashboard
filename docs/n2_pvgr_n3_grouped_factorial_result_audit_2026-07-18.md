# N2-PVGR-N3 96 条件结果审计与研究决策

> 日期：2026-07-18
>
> 结论：`GROUPED_FACTORIAL_FAIL_NO_FORWARD_AUTHORIZATION`
>
> 边界：两个 synthetic development families、每 family 四个 field seeds。没有真实 OERF 数据，
> 没有三维重建，没有神经算子比较，也没有论文 claim。

## 1. 先说人话

这轮没有证明“我们的新算法成功”。它证明了三件更有价值的事：

1. OCBH 虽然在九格机制桥接里对离散 JVP 很准确，但扩到 96 个条件后不够稳，也不够快，不能直接
   当三维重建的部署 forward；
2. Picard-1 在 8/8 个独立场的整体误差上都明显优于 OCBH，而且本机墙钟更快、logical query
   几乎相同，是下一阶段更值得保留的物理基线；
3. 当前 H256 evaluator 在 16/96 条件上没有通过 H512 sentinel，导致某些“谁更准”的比较本身
   不可解释。必须先把参考解做稳，不能把 evaluator 的误差包装成算法创新。

因此本轮是一个严格的 no-go，但它成功排除了错误路线，并给出了下一轮应当解决的具体数值和尾部
问题。

## 2. 证据链

| 层 | 标识 | 作用 |
|---|---|---|
| 原预注册 | `676c23d1962c93f17e8c2d0c0a81332146268790` | 在结果前冻结 8/32/96 设计、阈值、统计、图和停止规则 |
| 原 attestation | `8a684e6` | 证明正式结果与 checkpoint 在原协议冻结时不存在 |
| 首次运行 | 96/96 checkpoint | 完成物理格后在 query schema 汇总处发生 KeyError，未发布结果 |
| 盲态恢复协议 | `d25889de3fe19b8177e72086ec998d2d31452751` | 看数值前冻结 Picard 查询字段的唯一等义映射 |
| 恢复 attestation | `3ac6f41` | 不解析 JSON，封存 96 文件 Merkle root `f8dbf07b...c9145c7f` |
| 最终 bundle | `demo_t16_operator/results/n2_pvgr_n3_grouped_factorial_v1/` | result、8 个 CSV、图、summary、manifest |
| 独立 validator | `site_tools/validate_n2_pvgr_n3_grouped_factorial.py` | 复核 41 个文件哈希、行数、判决算术、恢复披露和图像非空 |

恢复只把 Picard 的 `total_field_point_queries` 映射为预注册语义中的
`logical_scalar_grid_point_queries`。阈值、seed、cell、bootstrap、machine decision 均未改变，96 个
物理格没有重算。原 crash 必须在论文和网页中保留披露。

## 3. 总门结果

| gate | pass / required | 解释 |
|---|---:|---|
| OCBH primary | `73 / 96` | 23 格至少一个对称主门失败 |
| discrete JVP teacher | `96 / 96` | OCBH 仍正确复现当前离散 epsilon-JVP 机制 |
| H256/H512 sentinel | `80 / 96` | 16 格 evaluator 不足，关闭这些格的效果解释 |
| OCBH timing | `0 / 4` | 四套 geometry package 全未达到 `p90/H128-p10 <= 0.25` |
| OCBH logical query | `96 / 96` | 查询预算定义本身满足预注册上限 |

OCBH 主门失败计数会重叠：matched residual 11 格、absolute reference 6 格、global no-harm 8 格、
Q95 no-harm 8 格、variance 2 格、risk Spearman 1 格；base consistency 为 0 格失败。高应力是主要
压力点：smooth stress-10 有 9 格失败；wrinkled stress-10 有 9 格失败，但 wrinkled 低应力也不是
完全安全。

OCBH 四套 timing 的 `candidate p90 / H128 p10` 为 `0.318, 0.390, 0.339, 0.341`，均高于
冻结上限 `0.25`。这说明它可以作为机制对照，却不应被包装成当前实现上的高效 forward。

## 4. Evaluator 为什么先挡住了结论

sentinel 的 16 个失败分成两个不同模式：

1. smooth 的 10 格主要集中在 `orientation_58 + wide aperture`。H256-to-H512 output L2 已低于
   `1e-4`，但 matched-residual convergence 为 `0.0106–0.0267`，超过 `0.01`。总输出看似收敛，
   小残差仍没有收敛；这正是只看全场 L2 会漏掉的问题。
2. wrinkled seed `3163` 的 `orientation_22` 六格，H256-to-H512 output L2 为约
   `1.017e-4–1.023e-4`，刚超过 `1e-4`。这六格也正是所有方法 absolute-reference gate 失败的
   六格，因此更像 evaluator bias，不是 Picard 独有失败。

下一轮必须对这 16 格加 H1024，并加入已通过的 matched controls。只有 H512-to-H1024 和
H256-to-H512 显示一致收敛阶，才能决定 evaluator 用 H512 还是 H1024。不得直接放宽 `1e-4/1e-2`
阈值来“救回”结果。

## 5. Picard 的强信号与不能越过的门

| metric | Picard-1 | Picard-2 |
|---|---:|---:|
| 8/8 field units matched mean 更低 | yes | yes |
| grouped matched ratio / OCBH | `0.1983` | `0.1130` |
| family-blocked 95% interval | `[0.1507, 0.2645]` | `[0.0748, 0.1667]` |
| worst wall p90 / OCBH | `0.3151` | `0.4751` |
| max logical query / OCBH | `0.9961` | `1.4942` |
| symmetric gates | `90 / 96` | `90 / 96` |
| raw invalid | `0` | `0` |
| preregistered dominance | no | no |

Picard-1 的总体 matched error 很强，并且 logical query 与 OCBH 基本相同；Picard-2 虽更准，查询
成本是 OCBH 的 `1.494` 倍，已经失去严格同预算优势。

不能宣布 Picard-1 支配 OCBH，原因有三层：

1. 六个 symmetric/false-safe 失败都来自未通过 evaluator 的 wrinkled-3163/orientation-22，先要
   解决参考解；
2. worst matched ratio 是 `6.4517`，发生在同一 evaluator-failed 格：Picard-1 为 `0.001709`，
   OCBH 为 `0.0002649`。参考未稳前不能解释这个相对比；
3. 在 sentinel 已通过的 `wrinkled-s3001/orientation_58/narrow/stress_1`，Picard-1 的 Q95 为
   `9.5117e-8`，OCBH 为 `9.3418e-8`，相对差 `1.819%`，超过预注册 `1%` no-harm 门。绝对量很小，
   但门已冻结，所以它仍是一个真实尾部反例。

## 6. 对“开发新算法”的直接启发

### 6.1 保留 Picard-1，关闭 OCBH 部署角色

在当前实现中，OCBH 应降级为离散机制 teacher/diagnostic；三维重建 forward 的第一强基线改为
`full curved RK4` 与 `Picard-1`。这是研究路线选择，不是 N3 machine authorization。

### 6.2 新算法不应只追求平均 L2

真正待解决的目标已经变得具体：在不超过 Picard-1/OCBH 查询预算的前提下，保持 Picard-1 的整体
优势，同时修复少数 ray/case 的 Q95 尾部。候选方向是：

- `Picard-1 + learned residual operator`：只学习 `H - P1`，而不是从零学习 BOST forward；
- loss 同时约束 field L2、per-ray Q95、held-out view reprojection 和 forward/VJP dot test；
- neural correction 必须有 uncertainty/validity gate，失败时回到同配置 full curved forward；
- 路由特征、fallback 和 residual model 的查询/墙钟都计入端到端预算；
- 不能用 `P2-P1` 作免费路由特征，因为算出它时已经支付 Picard-2；
- 可探索“共享 Picard 内部状态的低秩尾部校正”，但先证明它不需要重复完整网格查询。

### 6.3 与算子学习、三维重建的结合

下一阶段不是直接宣称 DeepONet/FNO/FFNO 被击败，而是先建立同一个 field-to-observation operator 的
JVP/VJP：

```text
3D refractive-index field
  -> full curved / Picard-1 BOST forward
  -> field JVP and field VJP dot/FD gates
  -> 6-view reconstruction loss
  -> 2 sealed held-out views
  -> residual neural operator only after numerical/noise floor is known
```

神经算子最有说服力的角色不是“取代一切物理”，而是学习 P1 在复杂折射/有限孔径条件下的结构化
尾部误差，同时保留物理 forward、adjoint 闭合和 fail-closed fallback。这与何远哲的 BOST/NeRIF/
四维重建主线更接近，也比在无真实数据时盲目搭一个 FNO 更容易形成可审查的创新点。

## 7. 冻结的下一步顺序

1. **N4 evaluator audit**：16 个失败格 + 匹配 controls，H256/H512/H1024；必要时只对仍不收敛格
   开 H2048。输出收敛阶、matched residual 收敛和成本，不比较/调算法。
2. **Field JVP/VJP tiny gate**：实现 tensor-only central RK4 field operator；3 seeds x 2 states 的
   dot test `<=1e-8`、FD `<=1e-4`，先过小网格。
3. **8-view reconstruction smoke**：6 train + 2 held-out，full curved 与 P1 同预算；只验证闭环和
   数据隔离，不报算法优越。
4. **Residual target audit**：在 evaluator 稳定后重新量化 `H-P1` 是否高于 numerical sentinel 与未来
   实验 noise floor。若不高，停止训练 neural residual。
5. **Operator baselines**：同一数据合同下比较 voxel/L-BFGS、DeepONet、FNO/FFNO，以及物理预条件
   residual operator；参数量、训练样本、VJP 次数、wall、memory、held-out views 全部同表报告。
6. **真实 OERF 合同**：向师兄索要单位、view/frame layout、背景图与位移提取、mask、标定、ray model、
   train/held-out split 和最小调用链；拿到前不得写“真实泛化”。

## 8. 仍未解决的风险

- 只有两个 synthetic families、每 family `n=4`，不能作 population inference；
- ray crossing/caustic、discontinuous topology、peak RSS、host sync、独立 evaluator process 仍未实现；
- timing 是当前 Mac CPU/PyTorch 实现结果，不是硬件无关复杂度定理；
- H2/4D/TDBOST 尚无经师兄确认的数据合同；
- 没有真实实验噪声地板，暂时不知道 `H-P1` 是否值得学习；
- 本轮没有任何 neural operator 或三维重建结果。

当前最可靠的结论不是“有一篇论文了”，而是：**P1 是更强起点，尾部是可研究问题，evaluator 与
field adjoint 是现在必须先铺平的路。**
