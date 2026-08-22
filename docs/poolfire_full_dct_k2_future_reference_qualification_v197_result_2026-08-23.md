# v197：future-only 合格参考已经建立，但还没有新算法结果

## 为什么做

v196 的 full-DCT K2 在已开封 p22 完整开发轨迹上达到 `2626/2626`，但 v196 冻结的 Zero-K4 reference 本身是 `0/2626`。v196.1 又证明，这个 reference 在 v196 冻结前已经被旧证据判为不充分。因此，继续评价任何新候选前，必须先单独回答一个问题：能否把一个已经充分、不可替换的物理结果固定为**未来合同专用 reference**？

v197 在读取任何 v197 之后的候选结果前，固定选择已封存的 full-DCT K2。这个选择只对未来候选有效，不能回头替换 v196 的 reference，也不能改写 v196 的 `INCONCLUSIVE_REFERENCE_ZERO_K4_INADEQUATE_V196`。

## 实际做了什么

- 只读取 v196 已封存的 formal 与独立数组，不打开新轨迹、p14 或 test。
- 固定同一条 p22 开发轨迹、101 帧、13 套标定、五相机与九相机两臂，共 `2626` 个单元和 `26` 个完整标定组。
- reference 身份固定为稠密 full-DCT 起点加**恰好两步**未修改 strict CGLS；逻辑在线账为 `3A+2A^T`。
- 重新计算逐单元 field / gradient / observation 门、逐组 `p90(method=higher)` 与 worst 门、全部正裕量和每行调用账。
- 独立程序先从独立生成的数组逐元素循环重算，完成后才读取 formal 输出作事后比较。

## 结果

| 审计项 | 结果 | 要求 |
| --- | ---: | ---: |
| 严格单元 | **2626 / 2626** | 2626 / 2626 |
| 完整标定组 | **26 / 26** | 26 / 26 |
| `3A+2A^T` 调用行 | **2626 / 2626** | 2626 / 2626 |
| 最小逐单元裕量 | **0.004185** | `> 0` |
| 最小组 p90 裕量 | **0.081378** | `> 0` |
| 最小组 worst 裕量 | **0.234186** | `> 0` |

五相机的 field / gradient / observation 全局 p90 为 `0.363959 / 0.599450 / 0.098924`，九相机为 `0.249912 / 0.417821 / 0.088978`。两臂各自都是 `1313/1313` 严格单元和 `13/13` 完整标定组。

独立复算的 14 项有效性与 formal 比较检查全部为真；formal 与独立的资格结构完全一致，最大数值差为 `0`。正式科学判决为：

`PASS_FUTURE_ONLY_FULL_DCT_K2_REFERENCE_QUALIFICATION_V197`

## 成功在哪里，没成功在哪里

**成功：** 后续实验不再缺少可接受 reference。full-DCT K2 已经被固定成不可与 Zero-K4、full-DCT K1、更深 CGLS 或未来候选交换的 future-only reference。下一候选可以在一个可判别的合同下接受或拒绝。

**尚未成功：** v197 没有提出或评价新候选，没有训练参数，也没有证明紧凑表示、exact-call 减少、wall/RSS 加速、p14 泛化、真实 BOST 或论文成功。`26234` 次几何 cache setup projection 仍与逻辑在线账分开披露。

## 下一门

只允许另行、结果前冻结**一个物理上不同的候选**，继续使用同一绝对真值门、v197 reference、Zero/BP/CGLS/PCGLS/dual-ridge 便宜对照、完整调用账和独立第二实现。候选未冻结前，不读取结果；候选未通过前，不开 p14/test，不测 wall/RSS，不训练大模型，也不租 GPU。

`algorithm_breakthrough=false`。

---

# v197: a future-only adequate reference is established, but there is no new algorithm result yet

## Why this was needed

Full-DCT K2 reached `2626/2626` on the opened complete p22 development trajectory in v196, but the Zero-K4 reference frozen inside v196 reached `0/2626`. v196.1 further showed that this reference had already been known to be inadequate before v196 was frozen. Before evaluating another candidate, the project therefore needed to establish a non-exchangeable, already adequate physical result as a **future-contract-only reference**.

Before seeing any post-v197 candidate result, v197 fixes the already sealed full-DCT K2 endpoint. This choice applies only to future candidates. It cannot replace the reference inside v196 or revise `INCONCLUSIVE_REFERENCE_ZERO_K4_INADEQUATE_V196`.

## What was executed

- Only already sealed v196 formal and independent arrays were read; no new trajectory, p14 validation, or test was opened.
- The same opened p22 trajectory was retained: 101 frames, 13 calibrations, five-camera and all-nine arms, `2626` cells, and `26` complete groups.
- The reference identity is fixed to the dense full-DCT start followed by **exactly two** unchanged strict CGLS iterations, with a logical online ledger of `3A+2A^T`.
- Cellwise field / gradient / observation gates, complete-group `p90(method=higher)` and worst gates, positive margins, and every call row were recomputed.
- The independent program first looped over independently generated arrays and only then read the formal output for post-hoc comparison.

## Results

| Audit item | Result | Requirement |
| --- | ---: | ---: |
| Strict cells | **2626 / 2626** | 2626 / 2626 |
| Complete calibration groups | **26 / 26** | 26 / 26 |
| `3A+2A^T` call rows | **2626 / 2626** | 2626 / 2626 |
| Minimum cell margin | **0.004185** | `> 0` |
| Minimum group-p90 margin | **0.081378** | `> 0` |
| Minimum group-worst margin | **0.234186** | `> 0` |

Five-camera field / gradient / observation global p90 values are `0.363959 / 0.599450 / 0.098924`; all-nine values are `0.249912 / 0.417821 / 0.088978`. Each arm reaches `1313/1313` strict cells and `13/13` complete calibration groups.

All 14 independent validity and formal-comparison checks pass. The formal and independent qualification structures agree exactly, with a maximum numeric difference of `0`. The sealed scientific decision is:

`PASS_FUTURE_ONLY_FULL_DCT_K2_REFERENCE_QUALIFICATION_V197`

## What succeeded and what did not

**Succeeded:** future experiments no longer lack an adequate accepted reference. Full-DCT K2 is now fixed as a future-only reference that cannot be exchanged with Zero-K4, full-DCT K1, deeper CGLS, or a future candidate. A future candidate can therefore be accepted or rejected under an informative contract.

**Not yet succeeded:** v197 neither proposes nor evaluates a new candidate. It has no trainable parameters and does not establish a compact representation, exact-call reduction, wall/RSS speedup, p14 generalization, real BOST, or paper success. The `26234` geometry-cache setup projections remain disclosed separately from logical online calls.

## Next gate

Exactly one physically distinct candidate may now be frozen separately and before results. It must retain the same absolute truth gates, the v197 reference, cheap Zero/BP/CGLS/PCGLS/dual-ridge controls, complete call accounting, and an independent second implementation. Until that candidate passes, p14/tests, wall/RSS, large-model training, and GPU rental remain closed.

`algorithm_breakthrough=false`.
