# N2-PVGR-N3：96 次 field-condition 分组开发筛查预注册

> 状态：`PREREGISTRATION — NO FORMAL 96-CELL RESULT HAS BEEN VIEWED`
>
> 日期：2026-07-18
>
> 证据等级：synthetic development only；不是 reserved audit、三维重建、真实数据或论文授权。

## 1. 为什么必须先冻结

N2-PVGR-N2 的九格机制桥接解决了“程序到底在对哪个离散算子求导”的问题，但它只有两个场家族、
每家族一个 field seed，而且阈值、实现和九格结果处于同一个已曝光版本。它能证明当前实现自洽，
不能证明阈值先于结果，也不能把 rays 当作泛化样本。

本协议先冻结配置、runner、测试、图表、统计单位和停止规则并单独提交。随后生成第二个提交中的
attestation，记录第一个提交的完整 Git SHA 和每个关键文件的 SHA-256。只有 attestation 已提交、
正式结果目录与 checkpoint 目录在证明创建时均不存在、且所有哈希仍匹配，runner 才允许开始
96 次正式计算。

## 2. 正确的统计单位

常用简称“96-cell”容易误导。准确设计是：

```text
2 fixed field families
x 4 field seeds per family
= 8 independent development field units

每个 field unit 内重复：
2 rig orientations x 2 aperture packages x 3 stresses = 12 conditions

8 field units x 12 repeated conditions = 96 field-condition runs
```

因此：

- 独立 development repeat 是 `phantom_family x field_seed`，总数只有 `8`；
- orientation、aperture 和 stress 是同一场内的重复测量因子；
- 256 条 rays、二维输出分量、128/256/512 个步长和 20 次 timing repeat 都不是物理重复；
- 每个 family 只有 `n=4`。即使四个 seed 全部同向，双侧精确符号检验也不能给出确认性
  `p<0.05`；本轮只能作分组开发筛查，不能声称 population-level 泛化。

## 3. 冻结的 96 次设计

### 3.1 开放场家族与 seeds

| family id | generator | seeds |
|---|---|---|
| smooth | `smooth_plume` | `1729, 1871, 1987, 2131` |
| wrinkled | `wrinkled_density_interface` | `2718, 2851, 3001, 3163` |

失败后不得替换 seed。`oblique_compression_sheet` 与 `shock_expansion_pair` 继续保持 unopened。

### 3.2 正交 rig orientation packages

| id | angle | detector u/z | path half length |
|---|---:|---:|---:|
| orientation_22 | 22 deg | `0.05 / -0.04` | `0.62` |
| orientation_58 | 58 deg | `-0.07 / 0.06` | `0.62` |

### 3.3 正交 aperture packages

| id | aperture radius | cone u/z |
|---|---:|---:|
| narrow | `0.035` | `0.04 / 0.03` |
| wide | `0.075` | `0.07 / 0.05` |

每一 orientation 与每一 aperture 完全交叉，不再像原三 case 那样把 family、orientation 和 aperture
成组混在一起。

### 3.4 应力、rays 与参考

- refractivity stress multiplier：`1, 3, 10`；
- 每格：`256` 个 Sobol pupil rays；
- candidate/execution：`128` steps；
- evaluator：`H256`；
- convergence sentinel：`H512`；
- forward-JVP teacher：全部 96 格；
- N1、OCBH、Picard-1、Picard-2：全部 96 格。

## 4. 共同随机数合同

原 `_build_case_context` 以 `case["id"]` 生成 Sobol seed。若直接把 32 个几何 case id 送进去，
orientation/aperture 条件会看到不同 pupil samples，破坏 field 内配对。

N3 runner 在真正调用冻结的 N2 bundle 前，把 execution case id 临时改为 `field_unit_id`：同一个场的
12 个条件共享完全相同的 256-ray Sobol prefix；输出账本仍恢复唯一 condition id。正式结果还必须
写出 8 个 field grid SHA-256 和 8 个 Sobol-prefix SHA-256。这样几何差异不会和 ray resampling
混杂。

## 5. 每格对称指标与完整性

旧 N2 machine gate 只判 OCBH，Picard 行没有同一 gate schema。N3 对 N1、OCBH、Picard-1 和
Picard-2 对称计算：

| metric | development threshold |
|---|---:|
| matched residual relative-L2 | `<= 0.02` |
| corrected residual variance ratio | `<= 0.01` |
| per-ray risk Spearman | `>= 0.99` |
| raw valid-ray fraction | `= 1.0` |
| candidate-to-H256 relative-L2 | `<= 0.002` |
| global reference no-harm | `<= 1.10` |
| per-ray Q95 reference no-harm | `<= 1.10` |

若 H128-to-H256 global denominator 小于 `1e-12`，global ratio 记为不可评分并由 absolute gate 接管；
若 Q95 denominator 小于 `1e-12`，candidate Q95 必须直接小于 `1e-12`。不得用极小分母产生的比值
自动判安全。

OCBH 另外保留九格继承的机制门：teacher output/position/direction、base consistency、
timing 与 logical query。四种方法都对称要求 `risk Spearman >= 0.99`，但它仍只是
teacher/mechanism 描述，不能替代 router 安全。

所有 raw invalid 和 false-safe 格都必须保留。整格回退到 straight output 不能抹掉原始
`valid_ray_fraction < 1` 的事实。

## 6. Evaluator 门

全部 96 格均要求：

- `H256-to-H512 output relative-L2 <= 1e-4`；
- `matched residual 256-to-512 relative-L2 <= 1e-2`。

失败格不删除、不换 seed；它被保留为 evaluator failure，并关闭该格的前向效果解释。

## 7. Field-unit 汇总

每个 `family x seed x method` 必须保留 12 条原值并生成：

- 12 条 matched residual 的 geometric mean；
- worst matched residual；
- worst global no-harm；
- worst Q95 no-harm；
- minimum raw valid fraction；
- raw-invalid count；
- false-safe count；
- symmetric-gate pass count。

总体图不得把 96 行送入普通独立样本置信区间。配对 CI 使用固定 family-blocked bootstrap：每个
family 内从四个 field units 有放回抽四个，两个 family 保持平衡，seed `44519`，`20,000` 次，
percentile `95%` interval。它仍是 development uncertainty summary，不是确认性 population CI。

## 8. Picard 对 OCBH 的预注册 dominance

对 Picard-1 和 Picard-2 分别判定。某方法只有同时满足以下全部条件，才可关闭 OCBH 的当前
forward-candidate role：

1. `8/8` field units 的 12-condition geometric mean matched ratio 均 `< 1`；
2. family-blocked grouped geometric-mean matched ratio 的 `95% upper bound <= 0.75`；
3. 任一 paired condition 的 matched ratio 不高于 `0.95`；
4. 任一 paired condition 的 global/Q95 no-harm ratio 相对 OCBH 不高于 `1.01`；
5. minimum valid fraction 不低于 OCBH，raw invalid 与 false-safe 均为 `0`；
6. 96 格全部通过对称 method gates；
7. 四个 geometry packages 的 paired `p90 wall` 相对 OCBH最坏不高于 `0.90`；
8. logical point-query count 相对 OCBH 不高于 `1.00`。

这一定义刻意使用相对 OCBH 的 query gate，而不是只要求低于 H128。Picard-2 即使墙钟更快，若
logical queries 比 OCBH 多，也不能称作多维度严格 dominance；Picard-1 则可能在相同查询预算下
形成真正反证。

第 4 项不允许用机器零分母制造优势：若 OCBH 的 paired absolute reference error 低于
`1e-12`，分母固定为 `1e-12`，candidate 仍须通过自身的对称 absolute/Q95 门。结果同时报告
96 个 paired conditions 中实际可评分的数量。

## 9. 预注册 machine decisions

只有三个允许输出的判决：

1. `GROUPED_FACTORIAL_FAIL_NO_FORWARD_AUTHORIZATION`
2. `PICARD_DOMINATES_OCBH_FORWARD_ROLE_CLOSED_FIELD_VJP_GATE_NEXT`
3. `OCBH_NOT_DOMINATED_CONDITIONAL_FIELD_VJP_GATE_NEXT`

三者都不授权 reserved audit、三维重建、真实数据、神经算子优越性或论文结论。第二个判决也只指
当前 synthetic program、当前 Mac CPU/PyTorch implementation 和冻结成本账本，不是硬件无关定理。

## 10. 风险路由为什么不放进本轮主判决

- `||D||` 需要先支付 OCBH；
- `||P1-(M+D)||` 需要同时支付 OCBH 与 P1，逻辑 query 已约为 `0.8 x H128`；
- `||P2-P1||` 只有算完 P2 才可用，只能决定是否再升级 H128，不能声称节省 P2。

因此在 Picard-1 与 OCBH 的同预算比较完成前，给风险特征结果后挑 cutoff 很容易制造虚假收益。
本轮只保留各方法自身的 per-ray risk ranking，不训练 router、不从 96 格选择 cutoff。若未来另开
router 协议，必须在新数据前冻结 `0 false-safe`、每 family coverage 与包含特征计算的端到端成本。

## 11. 已知仍未实现的门

当前程序没有完成：ray-crossing/caustic certificate、间断 cell topology certificate、逐方法 peak
RSS、host scalar synchronization count、独立进程 evaluator isolation。这些缺口写入配置并保持
claim gates 关闭。96 格即使全过，也不能据此宣称“物理安全”“算法复杂度优势”或“可投稿”。

## 12. 中止与停止规则

### 12.1 允许的完整性中止

只允许因以下原因中止计算并保留 checkpoint：

- config/runner/test/protocol/依赖哈希漂移；
- 重复或缺失 seed、field hash 不一致、共同 Sobol prefix 不一致；
- 非有限输出、reserved family 出现；
- H256/H512 evaluator 失效；
- 机器进程异常退出。

不得因为中途结果难看或某方法暂时落后而提前停。
单格 checkpoint 必须同时匹配预注册哈希与 cell metadata 才允许续跑。全部结果、CSV、图和 manifest
先在 `formal_work_output/final_artifacts_staging` 完整生成，只在所有文件成功后原子发布到正式目录；
中断的 staging 不得被当成完整结果。

### 12.2 完成 96 次后的研究停止

- 若 Picard-1 或 Picard-2 满足 8/8 dominance，停止把 OCBH 当部署前向候选；
- 若 OCBH inherited gate 失败，不提升超出机制证据；
- 未证明 residual target 同时高于 numerical sentinel 与未来实验 noise floor 前，不训练 neural residual；
- 看过正式结果后若要改 threshold、seed、图表或方法，必须新建 N4 协议并保留 N3 结果。

## 13. 可复现入口

- 配置：`demo_t16_operator/configs/n2_pvgr_n3_grouped_factorial_preregistered_v1.json`
- runner：`demo_t16_operator/run_n2_pvgr_n3_grouped_factorial.py`
- tests：`demo_t16_operator/test_run_n2_pvgr_n3_grouped_factorial.py`
- field JVP/VJP 下一门设计：`docs/n2_pvgr_field_jvp_vjp_reconstruction_interface_design_2026-07-18.md`
- attestation builder：`site_tools/create_n2_pvgr_n3_preregistration_attestation.py`
- attestation：在本协议提交后生成，正式结果运行前单独提交。

本文件没有 96 格数字。任何声称本协议已经产生结果的文字，在正式结果目录和 manifest 存在前都为
错误。
