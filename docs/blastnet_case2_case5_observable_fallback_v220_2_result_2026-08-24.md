# v220.2：可观测回退没有建立跨工况成功，验证合同保持 inconclusive

## 结论

v218.1 在已开封 BLASTNet Case 5 上发现，固定 Low-64 observation-only 起点加未修改 PCGLS K11 可以用 `12A+11A^T` 达到 K16 reference 的 matched accuracy。v220.2 没有训练模型，也没有改表示或深度，而是冻结一个只看观测残差的回退门：满足门时在 K11 停止，否则沿同一条 PCGLS 轨迹继续到 K16。

正式与独立程序都完成了 Case 5 与 Case 2 共 `1261` 个单元。两边的名义科学判决相同：

`FAIL_OBSERVABLE_FALLBACK_DEVELOPMENT_V220_2`

但预注册独立验证还有两个场级数值闭合门未通过，所以正式结果不能被写成“独立验证后的 FAIL”。最终可采用的科学状态只能是：

`INCONCLUSIVE_INVALID_OBSERVABLE_FALLBACK_V220_2`

## 为什么没有放宽门槛

两项越线都很小，但合同已经在看结果前固定为 `1e-8`：

| 独立验证项 | 实测最大相对差 | 冻结门 | 结果 |
|---|---:|---:|---|
| 正式场与独立场 | `1.50948e-8` | `1.0e-8` | 未通过 |
| 相机乱序场 | `1.14546e-8` | `1.0e-8` | 未通过 |

其余关键量高度一致：`1261/1261` 个单元重放完成，gate selections 与调用账逐项完全相同，feature、逐单元 metric、summary 的最大差分别为 `8.19e-16`、`4.29e-10`、`1.52e-10`。这些一致性说明两套程序看见的是同一个名义失败，但不能覆盖已经失败的场级验证合同。

因此没有事后把门改成 `2e-8` 或 `1e-7`，也没有重新跑一遍直到出现 PASS。第一次完整独立验证的 inconclusive 状态原样保留。

## 名义复算暴露了什么

即使只看两套程序一致的名义结果，回退机制也没有跨工况成立：

| 工况 | K11 接受数 | Matched 单元 | 完整几何 | 最大 matched ratio |
|---|---:|---:|---:|---:|
| Case 5 | `189/546` | `546/546` | `13/13` | `1.02043` |
| Case 2 | `447/715` | `629/715` | `0/13` | `1.50661` |

Case 2 的问题不是回退门漏掉了少量危险样本。固定 Low-64 起点即使继续到 K16，仍为 `0/13` 完整几何通过；零起点 geometry-Jacobi PCGLS K16 则为 `13/13`。这说明当前阻塞在 Low-64 起点及其后续轨迹的跨工况兼容性，而不是再调整同一观测阈值就能可靠解决。

候选在两工况合并后的平均逻辑账为 `14.478A+13.478A^T`，对比 reference 的 `16A+16A^T`，总调用账名义减少 `12.64%`。然而精度合同没有通过，且独立验证状态为 inconclusive，所以这不能写成 exact-call 成果，更不能进入 wall/RSS 资源门。

## 路线动作

1. 不放宽独立场级容差，也不重复启动 validator；
2. 关闭当前 Low-64 K11/K16 observation-only 回退机制；
3. 不用 CNN、FNO、UNO 或更大模型挽救该表示，不租 GPU；
4. 不打开 Case 4/Case 6，也不运行 fresh wall/RSS；
5. 后续只有物理上不同、结果前可证伪的机制或新的真实二维 BOST 位移数据，才重新打开下一门。

这不是算法突破、外部泛化、曲线光路或真实 BOST 结果。`algorithm_breakthrough=false`。

---

# v220.2: Observable Fallback Does Not Establish Cross-Condition Success and Remains Inconclusive

## Conclusion

v218.1 found on opened BLASTNet Case 5 that a fixed observation-only Low-64 initializer followed by unchanged PCGLS K11 could match the K16 reference at `12A+11A^T`. v220.2 trains no model and changes neither the representation nor the depths. It freezes an observation-residual fallback: stop at K11 when the observable gate accepts, otherwise continue the same PCGLS trajectory to K16.

The formal and independent programs both complete all `1261` Case 5 and Case 2 cells. Their nominal scientific decision agrees:

`FAIL_OBSERVABLE_FALLBACK_DEVELOPMENT_V220_2`

Two preregistered field-level numerical-closure checks nevertheless fail, so the formal result cannot be reported as an independently validated failure. The only admissible scientific status is:

`INCONCLUSIVE_INVALID_OBSERVABLE_FALLBACK_V220_2`

## Why the tolerance was not loosened

Both misses are small, but the contract fixes the tolerance at `1e-8` before results are seen:

| Independent check | Observed maximum relative difference | Frozen gate | Result |
|---|---:|---:|---|
| Formal versus independent fields | `1.50948e-8` | `1.0e-8` | Fail |
| Camera-permuted fields | `1.14546e-8` | `1.0e-8` | Fail |

The remaining evidence agrees closely: all `1261/1261` cells are replayed, gate selections and call ledgers match exactly, and maximum feature, cell-metric, and summary differences are `8.19e-16`, `4.29e-10`, and `1.52e-10`. This shows that the two implementations see the same nominal failure, but it does not override the failed field-level validation contract.

The gate was therefore not changed post hoc to `2e-8` or `1e-7`, and validation was not repeated until it passed. The first complete independent result remains preserved as inconclusive.

## What the nominal replay reveals

Even before applying the failed validation contract, the fallback does not hold across conditions:

| Condition | K11 accepted | Matched cells | Complete rigs | Maximum matched ratio |
|---|---:|---:|---:|---:|
| Case 5 | `189/546` | `546/546` | `13/13` | `1.02043` |
| Case 2 | `447/715` | `629/715` | `0/13` | `1.50661` |

Case 2 is not merely a threshold miss on a few unsafe cells. The fixed Low-64 initializer remains at `0/13` complete rigs even when continued to K16, whereas zero-start geometry-Jacobi PCGLS K16 reaches `13/13`. The blocker is therefore cross-condition compatibility of the Low-64 start and its subsequent trajectory, not something that can be made reliable by retuning the same observable threshold.

Across both conditions, the candidate's nominal mean ledger is `14.478A+13.478A^T`, versus `16A+16A^T` for the reference, a `12.64%` reduction in total logical calls. Accuracy does not pass and independent validation remains inconclusive, so this is not an exact-call result and cannot authorize a wall/RSS gate.

## Route action

1. do not loosen the independent field tolerance or rerun the validator;
2. close the current Low-64 K11/K16 observation-only fallback mechanism;
3. do not rescue the representation with a CNN, FNO, UNO, larger model, or rented GPU;
4. do not open Case 4/Case 6 or run fresh wall/RSS;
5. reopen a scientific gate only for a physically distinct preregistered mechanism or new paired real-BOST displacement data.

This is not an algorithm breakthrough, external generalization, curved-ray validation, or real BOST. `algorithm_breakthrough=false`.
