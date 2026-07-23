# GCT-KMix O1 SOC 数值缩放审计

## 发生了什么

结果前合同提交 `88e4364` 后，第一次完整 18-case 运行严格返回
`GCT_KMIX_FAIL_CLOSED_SOLVER_OR_CONTRACT_FAILURE`。17 个 main oracle 为
`Solved`；公开探索 OOD 的 `seed=307/single_interface` 为 `AlmostSolved`，其
relative primal-dual gap 为 `2.6047681456e-8`，高于冻结上限 `1e-8`。因此
attempt 1 没有被重命名为 NO-GO 或 headroom，也没有删除失败 case。

原始 attempt 1 包保存在
`demo_t16_operator/results/gct_kmix_oracle_ceiling_v1_attempt1_fail_closed/`。

## 允许的修复

对每条 ray 的 SOC

`||E_i w||_2 <= b_i`, `b_i > 0`

同时把左右两边除以 `b_i`：

`||(E_i / b_i) w||_2 <= 1`。

正数缩放不改变可行域、目标、checkpoint、truth、ray、安全锚点、调用预算
或任何通过门，只改善不同 ray block 的数值尺度。`b_i` 因冻结的正 slack
严格大于零；实现仍用 machine tiny 作防御性下界，并公开每 case 的最小/最大
scale。

## 没有做什么

- 没有把 `Solved` 放宽成 `AlmostSolved`；
- 没有把 gap 门从 `1e-8` 改大；
- 没有改变 5% field、3% H1、逐 ray/camera no-harm 或 12/12 调用门；
- 没有换 seed、删 case、改 k4/k12、打开 fresh/真实数据或训练 learner；
- 没有把 attempt 1 的约 2.17% development / 1.05% OOD field gain 写成成功。

修复代码与本说明必须先进入新的 source commit，第二次结果包再从该干净提交
生成。若第二次仍不能得到全 `Solved` 和 gap 门，继续 fail closed；若数值合同
通过但 headroom 门失败，则冻结 zero-start convex hull NO-GO。
