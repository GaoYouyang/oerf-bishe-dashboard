# v244.2：Case 19 一次性同族公开门保持不确定，固定 K14 确认路线关闭

## 先说结论

结果前冻结的 Case 19 一次性同族公开门已经完成正式运行与完全独立第二实现。权威判决是：

`INCONCLUSIVE_INVALID_CASE19_CANONICAL_CAMERA_ACTUAL_K14_EXTERNAL_V244_2`

这不是通过，也不是可以改容差后包装成通过的“接近成功”。独立验证共 **26/29** 项通过，但 residual 重算相对差 **1.1060e-7** 高于冻结的 **1e-8**，逐单元 metric 最大绝对差 **1.0219e-8** 也略高于 **1e-8**，相机换序后的 metric 未达到逐值完全一致。按结果前合同，只要其中一项失配，整批就必须保持不确定。

## 工程输入与两次前置失效

Case 19 共取得 **37** 个公开数据对象，独立逐字节验证全部通过；这个阶段没有解析科学数值，只证明输入完整。

正式科学链前两次启动均保留为工程失效：第一次在任何数值网格或密度解析前发现源形状不兼容；第二次完成预处理后、预测与评分前发现状态有效性检查与已冻结的零均值边界约定冲突。两次都没有生成科学判决，也没有被复用或计作算法进展。

v244.2 只修正这两个执行矛盾，不改 Case 19 数据、33 帧、13 套 rig、六条 arms、K14 主候选、K16 reference、四个 controls、精度门或禁止调参规则。正式侧完成每条 arm 的 **429** 个预测与物理重放；正式有效性检查为 **19/20**，唯一失败同样是规范相机换序后的 metric 未逐值完全一致。

## 独立验证到底差在哪里

第二实现不导入正式数值求解 helper，并在生成和封存自己的预测后才读取正式科学数组。以下项目保持在冻结门内：

- 场的最大相对差：**1.3479e-9**，低于 **1e-8**；
- observation 最大相对差：**4.07e-16**；
- 汇总最大绝对差：**2.61e-10**；
- cache 与物理重放最大绝对差：**1.11e-15 / 5.13e-16**；
- 规范化后的直接 observation 最大绝对差：**0**。

但三个项目未过：相机换序 metric 必须完全一致、residual 相对差必须不超过 **1e-8**、逐单元 metric 绝对差必须不超过 **1e-8**。没有结果后放宽容差，也没有再次运行。

## 开封后诊断，不是替代判决

为定位为什么即便忽略数值失配也不能授权下一门，只对两边已经独立重建并封存的汇总做了事后读取：

| arm | 绝对安全单元 | 绝对完整 rig | K16 同精度完整 rig |
| --- | ---: | ---: | ---: |
| 固定 causal warm K14 primary | 428/429 | 12/13 | 13/13 |
| Zero geometry-Jacobi PCGLS K16 reference | 417/429 | 9/13 | 13/13 |
| BP geometry-Jacobi PCGLS K13 | 326/429 | 2/13 | 0/13 |
| BP CGLS K13 | 52/429 | 0/13 | 0/13 |
| Zero geometry-Jacobi PCGLS K14 | 313/429 | 0/13 | 0/13 |
| Zero CGLS K14 | 55/429 | 0/13 | 0/13 |

K16 reference 的绝对门只有 **9/13**，已经触发“参考不充分则不可解释候选”的 fail-closed 规则。主候选虽然相对这个 reference 达到 **13/13** matched，却在绝对门只有 **12/13**；唯一失败 rig 为 **32/33** 个严格安全单元，内部梯度 worst 为 **0.758223**，高于逐单元冻结门 **0.75**。

这些数字只解释已开封批次，不把不确定判决改成失败或通过。特别是，不能用“matched 13/13”跳过 reference 只有 9/13 的事实。

## 路线动作与证据边界

固定 v243 机制从 Case 7 原样迁移到 Case 19 的这条前瞻确认路线关闭：不重跑、不放宽容差、不调深度或 cache、不换 Case 13/18 补考，也不用 CNN、FNO 或 GPU 挽救。由于同精度外门没有成立，fresh wall/RSS 阶段不启动。

这不证明 C 路线不可能，也不否定已开封 Case 7 的机制证据；它只说明当前固定机制没有获得 Case 19 的可复算前瞻确认。下一项工作必须由新的物理信息或物理上真正不同、结果前唯一冻结且可证伪的机制触发。

`algorithm_breakthrough=false` · `paper_success=false` · `external_generalization=false` · `resource_speedup=false` · `real_bost=false`

---

# v244.2: the one-shot same-family Case 19 public gate remains inconclusive, closing the fixed K14 confirmation route

## Bottom line

The preregistered one-shot same-family Case 19 public gate has completed formal execution and a fully independent second implementation. Its authoritative decision is:

`INCONCLUSIVE_INVALID_CASE19_CANONICAL_CAMERA_ACTUAL_K14_EXTERNAL_V244_2`

This is not a pass and not a near-pass that may be rescued by relaxing a tolerance. Independent validation passes **26/29** checks. The residual recomputation relative difference, **1.1060e-7**, exceeds the frozen **1e-8** limit; the maximum cell-metric absolute difference, **1.0219e-8**, also slightly exceeds **1e-8**; and camera-permuted metrics are not value-exact. The preregistered contract requires the whole batch to remain inconclusive when any of these checks fails.

## Input engineering and two pre-result failures

All **37** public Case 19 objects were acquired and independently validated byte for byte. No scientific numeric value was parsed during that acquisition validation, so it establishes input completeness only.

The first two science-chain launches remain engineering failures. The first detects a source-shape incompatibility before any numeric grid or density parse. The second reaches preprocessing but fails before prediction or scoring because its state-validity check contradicts the frozen zero-mean, zero-boundary representation. Neither launch generates a scientific result, and neither output is reused or counted as algorithmic progress.

v244.2 repairs only those execution contradictions. It changes no Case 19 data, 33-frame roster, thirteen rigs, six arms, K14 primary, K16 reference, four controls, accuracy gate, or no-retuning rule. Formal execution completes **429** predictions and physical replays per arm. It passes **19/20** formal validity checks; the only failure is again value-exact metric agreement after canonical camera permutation.

## What independent validation disagrees on

The second implementation imports no formal numerical-solver helper and reads the formal scientific arrays only after independently generating and sealing its own predictions. These comparisons remain within their frozen gates:

- maximum field relative difference: **1.3479e-9**, below **1e-8**;
- maximum observation relative difference: **4.07e-16**;
- maximum summary absolute difference: **2.61e-10**;
- maximum cache and physical-replay absolute differences: **1.11e-15 / 5.13e-16**;
- maximum direct canonical-observation absolute difference: **0**.

The three failures are exact camera-permutation metric equality, residual relative agreement at **1e-8**, and cell-metric absolute agreement at **1e-8**. No tolerance is relaxed and no rerun is performed.

## Post-open diagnostic only

To diagnose why the resource gate would remain unauthorized even without the numerical mismatch, the already reconstructed and sealed summaries are read after opening:

| arm | Absolute-safe cells | Absolute complete rigs | K16-matched complete rigs |
| --- | ---: | ---: | ---: |
| Fixed causal warm K14 primary | 428/429 | 12/13 | 13/13 |
| Zero geometry-Jacobi PCGLS K16 reference | 417/429 | 9/13 | 13/13 |
| BP geometry-Jacobi PCGLS K13 | 326/429 | 2/13 | 0/13 |
| BP CGLS K13 | 52/429 | 0/13 | 0/13 |
| Zero geometry-Jacobi PCGLS K14 | 313/429 | 0/13 | 0/13 |
| Zero CGLS K14 | 55/429 | 0/13 | 0/13 |

The K16 reference reaches only **9/13** absolute complete rigs, triggering the fail-closed inadequate-reference rule. The primary reaches **13/13** relative to that reference but only **12/13** on the absolute gate. Its sole failing rig has **32/33** strict-safe cells; the worst interior-gradient error is **0.758223**, above the frozen cell limit **0.75**.

These values diagnose the already-opened batch; they do not replace the inconclusive decision with either failure or success. In particular, the matched **13/13** count cannot override a reference that is adequate in only 9/13 rigs.

## Route action and evidence boundary

The fixed v243 mechanism's prospective Case 7-to-Case 19 confirmation route closes: no rerun, tolerance relaxation, depth or cache retuning, Case 13/18 substitution, CNN/FNO rescue, or GPU rental is authorized. Because the matched-accuracy external gate is not established, fresh wall/RSS testing does not start.

This does not prove the C route impossible and does not erase the opened Case 7 mechanism evidence. It establishes only that the fixed mechanism did not obtain reproducible prospective confirmation on Case 19. Further work requires new physical information or one physically distinct, uniquely preregistered, falsifiable mechanism.

`algorithm_breakthrough=false` · `paper_success=false` · `external_generalization=false` · `resource_speedup=false` · `real_bost=false`
