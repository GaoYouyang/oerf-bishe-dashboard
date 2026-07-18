# N2-PVGR N4.1 评估器收敛结果审计

日期：2026-07-18

## 一句话结论

N4.1 在 16 个 N3 sentinel 失败格和 16 个同场 matched controls 上完成 H256/H512/H1024，并对 9 格按预注册升级 H2048。最终 30/32 格取得有效逐格 reference，但两个 `smooth-s1871 / orientation_58 / narrow / stress 1,3` controls 的 H1024-H2048 matched-residual relative-L2 仍高于 `0.125%` 门，因此机器严格输出：

`FAIL_CLOSED_EVALUATOR_REMAINS_UNAUTHORIZED`

这不是“所有曲线射线都不收敛”。32/32 格的 detector output、finite、domain、stencil、direction norm、support topology 和 frustum 门都通过。当前窄瓶颈是：曲线减直线的物理残差比完整输出小约四个数量级，残差相对误差在两个窄孔径格上仍略高于预注册门。

## 运行合同

- 统计/物理格：16 个 N3 failures + 16 个 same-field/same-stress matched controls；不是 32 个独立场。
- common Sobol rays：每格 256 条。
- 基础层：H256、H512、H1024，共 96 个 level evaluations。
- 条件升级：9 个 H2048，总计 105 个 level checkpoints。
- logical scalar-grid point queries：`814,743,552`。
- checkpoint 内核 wall-time 合计：`334.13 s`；中位数 H256/H512/H1024/H2048 分别为 `1.157/2.257/4.501/8.925 s`。
- CPU float64；wall-time 只描述本机，不是可移植复杂度结论。

## 机器门计数

### H1024

| gate | 通过 |
|---|---:|
| output absolute | 32/32 |
| output contraction | 32/32 |
| matched-residual absolute | 26/32 |
| matched-residual contraction | 29/32 |
| parent N3 output reproduction | 32/32 |
| parent N3 matched-residual reproduction | 27/32 |
| finite/domain/stencil/direction | 32/32 |
| support/frustum topology | 32/32 |

H1024 共 23/32 全门通过。9 格升级 H2048，其中 7 格通过，2 格仍失败。H2048 唯一剩余失败门是 matched-residual absolute gate。

## 两个最终失败格

### `smooth-s1871-orientation_58-narrow__stress_1`

- H1024-H2048 output relative-L2：`6.686e-7`；输出收敛门通过。
- H1024-H2048 matched-residual relative-L2：`1.647e-3`，高于门 `1.25e-3`。
- matched-residual contraction ratio：`0.346`；收缩门通过。
- H2048 完整输出 L2 norm：`2.4807e-3`。
- H2048 matched residual L2 norm：`1.8279e-7`，仅为完整输出的 `7.37e-5`。
- H1024-H2048 matched-residual absolute L2 difference：`3.011e-10`，相对完整输出仅 `1.214e-7`。

### `smooth-s1871-orientation_58-narrow__stress_3`

- H1024-H2048 output relative-L2：`6.686e-7`；输出收敛门通过。
- H1024-H2048 matched-residual relative-L2：`1.392e-3`，高于门 `1.25e-3`。
- matched-residual contraction ratio：`0.327`；收缩门通过。
- H2048 完整输出 L2 norm：`7.4456e-3`。
- H2048 matched residual L2 norm：`1.6488e-6`，为完整输出的 `2.21e-4`。
- H1024-H2048 matched-residual absolute L2 difference：`2.296e-9`，相对完整输出 `3.083e-7`。

两格都在相邻加密后继续收缩，并非发散；但预注册 absolute-relative 门没有通过，所以不能事后用“只差一点”改判。

## matched wide 对照告诉了什么

同一 `smooth-s1871 / orientation_58 / stress 1,3` 的 wide aperture controls 在 H2048 matched-residual relative-L2 分别为 `0.0928%` 和 `0.0687%`，均通过 `0.125%` 门。narrow aperture 的 matched residual 更小，固定绝对离散差在更小分母下被放大。因此当前证据支持一个新的数值问题：

> 如何为 BOST 弱曲率 residual target 构造 cancellation-aware、noise-aware、仍可 fail closed 的 reference certificate？

这是推断出的机理假设，不是已证明原因。需要 H4096/paired quadrature/Richardson 与实验 noise floor 分开验证。

## 实现与恢复披露

N4 v1 首次执行发现 H2048 escalation 控制流缺陷；N4.1 只改执行顺序，科学配置不变，并使用新 work/output 与新 attestation，未复用 v1 的 6 个 checkpoint。

N4.1 数值完成后，冻结 base plot 把字典直接传入 Matplotlib category axis，artifact staging 在制图处退出。后处理恢复在解析 checkpoint 数值前封存 105 个文件、9 个 H2048 和 Merkle root：

`407ec681e0f8787c286531db222cc6cbeb8009b83717b35bb2aca83498cb242a`

恢复只把 bar x 输入改成 `list(counts.keys())`，没有重跑数值 level。原 N4.1 validator 与 recovery validator 均为 `valid: true`，并验证图像非空。

## 现在允许与禁止

允许：

- 说完整 detector output 已在 32 个 selected cells 上通过 H1024 门；
- 说 30/32 取得逐格 H1024/H2048 reference；
- 把两个 narrow controls 定义为 cancellation-aware reference 的下一研究问题；
- 用这些结果设计 N5，不反向修改 N4.1。

禁止：

- 声称 forward evaluator 已获得广泛授权；
- 启动八视角定量 reconstruction comparison；
- 训练并宣称 H-P1 residual operator 优于 DeepONet/FNO/FFNO；
- 把 30/32 写成真实 BOST、跨 rig、跨场族或论文成功；
- 删除两个 controls 或放宽 `0.125%` 门来改变判决。

## 下一步 N5

1. 在不训练网络的前提下，对两失败格与同场 wide controls 做 H4096/H8192 小范围 asymptotic probe。
2. 实现 direct paired residual quadrature：在共享节点上直接积 `curved integrand - straight integrand`，与“两个完整输出相减”对照。
3. 加入 float64 compensated summation 与 Richardson extrapolation，只作 reference mechanism baselines。
4. 将误差同时报告为 residual-relative、full-output-relative、per-ray Q95 和未来 flow-off detector noise units。
5. 向何远哲申请同一 rig 的 flow-off repeats、位移单位/像素尺度和 confidence/covariance，冻结实验 noise floor 后才决定残差是否值得学习。
6. 只有新预注册 N5 清除两格，才开放 tiny field-JVP/VJP gate；仍不直接开放神经算子训练。

## 可复核文件

- [逐格机器结果](../demo_t16_operator/results/n2_pvgr_n4_1_evaluator_convergence_v1/result.json)
- [逐层门与误差表](../demo_t16_operator/results/n2_pvgr_n4_1_evaluator_convergence_v1/metrics.csv)
- [matched-pair 诊断表](../demo_t16_operator/results/n2_pvgr_n4_1_evaluator_convergence_v1/pair_diagnostics.csv)
- [计算成本账本](../demo_t16_operator/results/n2_pvgr_n4_1_evaluator_convergence_v1/cost_ledger.csv)
- [N4.1 原验证报告](../demo_t16_operator/results/n2_pvgr_n4_1_evaluator_convergence_v1/validation_report.json)
- [不透明恢复验证报告](../demo_t16_operator/results/n2_pvgr_n4_1_evaluator_convergence_v1/recovery_validation_report.json)
