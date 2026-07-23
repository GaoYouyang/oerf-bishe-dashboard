# N1.7-D：四倍半径事后敏感性审计

- 日期：2026-07-18
- 状态：`POSTHOC_DIAGNOSTIC_NO_VERDICT_CHANGE`
- 审计配置提交：`551d42d`
- 证据：E1 synthetic、已经打开的 development；不是确认集
- learner / OOD / fresh / final / 真实 BOST：全部未打开

## 一句话结论

冻结 N1.7 的安全半径确实压低了结果，但它不是唯一瓶颈。把三个可见半径系数同时放大四倍后，
主 measurement projection 不再触边，field gain 从 `4.828%` 升到 `5.556%`；然而 exact headroom
retention 仍只有 `65.264%`，support-adjoint gain 仍只有 `28.364%`。因此冻结协议的 NO-GO 不变，
更严格的结论是：**当前 rank-4 Krylov span 与当前目标定义都需要改，不能只学习一个更大的幅值。**

## 1. 只改了什么

这是一项明确标注的 post-hoc mechanism audit，不是重新预注册，也不能改写正式判决。相对冻结
配置只改两件事：

```text
rho: min(0.50||d||, max(4||r||, 0.10||d||))
  -> min(2.00||d||, max(16||r||, 0.40||d||))

Powell maxfev per start: 96 -> 384
```

basis、12+2+10 调用日程、17 项门、数据 split、三种起点和所有 controls 保持不变。四倍半径
仍只依赖部署可见的 `||d||` 与 `||r||`，但这个倍数是在看过正式结果后选择的，所以不得用于
算法优越性或泛化声明。

## 2. 冻结结果与敏感性结果

| 路线 | field vs low | H1 vs low | field vs damping | adjoint gain | exact retention | 门 |
|---|---:|---:|---:|---:|---:|---:|
| 冻结 measurement projection | `+4.828%` | `+11.076%` | `+1.193%` | `+16.281%` | `56.717%` | `14/17` |
| 四倍半径 measurement projection | `+5.556%` | `+10.158%` | `+2.013%` | `+28.364%` | `65.264%` | `15/17` |
| 冻结 finite-K truth search | `+5.560%` | `+12.145%` | `+2.020%` | `+11.728%` | `65.313%` | `15/17` |
| 四倍半径 finite-K truth search | `+6.186%` | `+11.825%` | `+2.731%` | `+24.239%` | `72.669%` | `16/17` |

四倍半径下，measurement projection 的 `12/12` development cases 都不再触边，平均系数范数
只占新半径的 `62.59%`，最大为 `95.51%`。因此它可作为当前 measurement objective 下近似
unbounded rank-4 span ceiling：field 门已跨过，但 retention 与 adjoint 两门仍失败。

finite-K 的 `12/12` cases 也不触边，平均半径占用 `56.15%`，最大 `84.00%`。它通过 field、
retention、两类 family、逐 geometry 尾部和 harm 等 16 项，只剩 `P A^T` gain 的 50% 门失败。
这说明 span 中存在比 measurement projection 更好的 solver-aware 系数，但不能说明部署可见量能
预测它。

## 3. 为什么仍不能训练 learner

1. **主 deployable-shaped oracle 仍只过 15/17。** 未触边后仍丢掉约三分之一 exact field
   headroom，说明仅预测安全幅值不够。
2. **finite-K 读取 truth。** 它在每次函数评估后运行 CGLS-10 并读取真实 field error，不是
   训练时或部署时可见的标签生成器。
3. **搜索成本很高。** development 为 `74,010F/74,010A^T`，calibration 为
   `42,680F/42,680A^T`，完整结果包共 `116,690F/116,690A^T`；生产等价路径仍只有
   `25F/24A^T`，两者不得混写。
4. **仍不是全局上界。** development 的 36 个 Powell runs 有 35 个报告成功；数值稳定性比
   冻结 96-evaluation 搜索好，但 Powell 仍不提供全局最优证明。
5. **同一 development 已被打开。** 这批数据只能生成 N1.8 假设，不能继续选 basis、半径和
   loss 后再称独立验证。

## 4. 50% adjoint 门应怎样看

`P A^T` gain 是 solver-visible mismatch 的机制指标，不等同于最终 field-L2。它在预注册前已冻结，
所以本轮必须保留，不能因为 field 变好就事后删除。但下一轮需要请何远哲师兄确认：论文主终点
究竟是 field、界面位置、held-out reprojection，还是 forward mismatch 的物理忠实度。

若实验室真正关心有限步重建质量，N1.8 可以把 `P A^T` 从单一否决门改成预注册的联合安全指标；
若目标是校正 forward physics，则必须继续要求 correction 与真实 mismatch 对齐，不能把
truth-conditioned field optimum 冒充物理模型校正。

## 5. N1.8 的可执行设计

下一轮不再训练 N1.7 的四个系数，而是在新 geometry/session split 上比较三个同预算表示：

1. **Krylov-4 control：**冻结当前 `orth([d,r,Kd,Kr])`，作为负基线。
2. **camera/ray hybrid-4：**两个 centered global mismatch modes 加两个 per-case
   `A P A^T` modes；global mean 也计入总 correction norm，避免旧 PCA 范数不公平。
3. **camera-block rank-6：**按 camera 或 ray-coordinate 对 `d/r` 做可见调制，再用同样两次
   probes；若真实调用成本随 block RHS 增长，必须按 wall time 和实际 operator work 重新匹配。

每个表示先过无 learner oracle gate。只有新表示在新 split 上同时超过 damping、Krylov-4、
5% field、70% extra headroom、逐 geometry tail 与 harm 门，才训练 bounded coefficient learner。
learner 的损失应穿过固定 10 步 CGLS，并同时报告 field/H1、`P A^T`、held-out reprojection、
operator calls 和端到端时间。

## 6. 给师兄新增的三个问题

1. `P A^T mismatch gain >=50%` 是否符合组内对 forward correction 的物理要求，还是应以 held-out
   camera reprojection / field / flame-front 指标作为主门？
2. camera index、ray coordinate、aperture sample、optical-flow confidence 哪些在真实部署时可见，
   并且可以进入 basis 而不产生 truth leakage？
3. 同一 operator 对多 RHS 的实际成本如何计数；camera-block probe 是一次批处理还是按相机线性
   增长？

## 7. 复现

```bash
cd oerf-bishe-dashboard

.venv/bin/python site_tools/run_jacru_n1_7_geometry_krylov_oracle.py \
  --config demo_t16_operator/configs/jacru_n1_7_radius_4x_posthoc_audit_v1.json \
  --output-dir demo_t16_operator/results/jacru_n1_7_radius_4x_posthoc_audit_full1

cd demo_t16_operator/results/jacru_n1_7_radius_4x_posthoc_audit_full1
shasum -a 256 -c checksums.sha256
```

12 个登记文件全部通过 SHA-256。结果包同时明确保存 development、calibration 和完整 package 的
evaluator call ledger；旧名为 `finite_k_evaluator_total_*` 的兼容字段明确标注为
`development_only_legacy_field_name`。

## 8. 允许和禁止的表述

**允许：**“事后半径敏感性显示安全边界是重要瓶颈，但解除边界后 rank-4 measurement span 仍未
满足 retention 与 adjoint 门；finite-K 只说明存在 solver-aware 系数线索。”

**禁止：**“四倍半径算法成功”“已达到 6.19% 可部署提升”“16/17 等于基本发表成功”
“可以用同一 development 训练后宣称泛化”。
