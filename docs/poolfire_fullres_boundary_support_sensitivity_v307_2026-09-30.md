# V307 Full-Resolution Boundary-Support Sensitivity

**Date / 日期:** 2026-09-30  
**Evidence role:** Post-open descriptive audit on prepared PoolFire training fields; not a reconstruction or algorithm test.

## Question and fixed calculation

V304.2 reported no effect from a one-layer support mask on a coarse inverse-proxy field whose outer shell had already been set to zero during construction. V307 asks a narrower representation question on the separately prepared full-resolution rho bridge: how much centered-field norm lies in one outer voxel layer if that layer is set to zero?

The audit uses 55 fixed frames (five per each of 11 already-opened training trajectories). For each 80 x 80 x 200 field, it subtracts the mean over the interior, then measures the outer one-voxel shell's share of centered L2 norm and energy, shell/interior RMS ratios, and exact-zero boundary count. It applies no pass threshold and makes no claim that this shell is the physical boundary.

## Result

Across the 55 frames, removing the shell would remove a median **19.08%** of the centered-field L2 norm (p90 **25.65%**, maximum **28.67%**). The corresponding energy share is median **3.64%**, p90 **6.58%**, maximum **8.22%**. Raw-rho shell/interior RMS has median **1.020**; none of the 55 fields has an exactly-zero boundary voxel.

A separate implementation recomputed 220 floating metrics and 110 integer checks. The maximum relative-scaled difference was **4.99e-12**; input receipts and formal outputs were unchanged.

## Companion result: V306 native-grid gradient-stencil sensitivity

V306 independently compared first- and second-order endpoint stencils on the native `80 x 80 x 200` rho grid, using the same five fixed frames from each of the same 11 already-opened training trajectories. Across 55 frames, the relative L2 difference between the two full gradient fields has p50/p90/worst **6.59% / 9.06% / 9.72%**. The strict-interior difference is exactly zero; within the one-voxel shell, relative L2 difference is **23.91% / 26.42% / 27.67%**. The shell contains p50/p90/worst **22.60% / 31.15% / 32.85%** of the first-order gradient norm and **26.94% / 36.38% / 38.56%** of the second-order gradient norm.

The independent implementation checked 715 floating metrics and 110 integer checks; its maximum relative-scaled difference was **9.36e-16**. This shows the endpoint-stencil sensitivity also appears on the native grid, not only on V304.2's coarse proxy. It does not establish the BOS observation error or which stencil matches the actual label-generation pipeline.

## Interpretation boundary

This corrects the scope of the earlier coarse-proxy observation: the earlier mask result applies to a field already masked at construction, while this audit finds that applying the same one-layer operation to the prepared full-resolution bridge is not numerically negligible. It does **not** identify whether the outer layer is a physical-domain edge, crop edge, or model-query edge; determine the correct exterior extension or derivative stencil; establish rho units; generate BOS observations; or test reconstruction accuracy. The bridge's upstream processing and physical units remain unverified.

Together with V306, the evidence separates two numerical effects: zeroing the shell changes the field representation, and endpoint derivative choice changes the gradient primarily on that shell. Neither resolves the physical boundary semantics.

V307 is descriptive representation sensitivity, not a new warm start, exact-call saving, speed/memory result, generalization result, or algorithmic breakthrough. The previously frozen strict learned-cost verdict is unchanged. Validation, test, and external sets were not accessed.

`algorithm_breakthrough=false`; `paper_success=false`; `resource_speedup=false`; `external_generalization=false`; `real_bost=false`.

## 中文摘要

### 问题与固定计算

V304.2 中“置零一层没有影响”针对的是生成时已经把外壳置零的低分辨率逆问题代理。V307 只问一个表示问题：对另行准备的全分辨率 PoolFire rho 桥接场，若先减去内部均值，再把当前网格最外一层置零，会去掉多少去中心场的范数？

本检查固定使用 55 帧（11 条已开封训练轨迹各 5 帧）。每帧空间尺寸为 80 x 80 x 200；计算外壳占去中心场 L2 范数与能量的比例、边界/内部 RMS 比及边界精确零值数。不设通过阈值，也不假定这层网格就是物理边界。

### 结果

55 帧中，置零外壳会移除去中心场 L2 范数的中位数 **19.08%**（p90 **25.65%**，最大 **28.67%**）；对应能量占比中位数 **3.64%**、p90 **6.58%**、最大 **8.22%**。原始 rho 的边界/内部 RMS 比中位数为 **1.020**；55 帧都没有精确为零的边界体素。

独立实现重新计算 220 个浮点指标和 110 个整数检查，最大相对缩放差为 **4.99e-12**；输入校验收据及正式输出保持不变。

### 补充：V306 原生网格梯度模板敏感性

V306 在相同的 11 条已开封训练轨迹、相同的每条 5 帧上，对原生 `80 x 80 x 200` 网格比较一阶与二阶端点差分。两种全域梯度场的相对 L2 差异 p50/p90/最大为 **6.59% / 9.06% / 9.72%**；严格内部差异为零；最外一层内的相对差异为 **23.91% / 26.42% / 27.67%**。外壳承载的梯度范数比例，一阶模板为 **22.60% / 31.15% / 32.85%**，二阶模板为 **26.94% / 36.38% / 38.56%**。独立实现核对 715 个浮点指标与 110 个整数检查，最大相对缩放差 **9.36e-16**。

这说明端点差分敏感性并非只在 V304.2 的粗代理上出现，且两模板的数值差异局限于网格外壳。它仍不说明哪种模板符合师兄的二维投影标签生成方式，也没有直接计算 BOS 投影误差。

### 解释边界

这澄清了旧结果的适用范围：旧结论只适用于构造时已置零的低分辨率代理；对当前准备态全分辨率桥接场施加同样的一层置零，在数值上不可忽略。但本检查无法判断外壳是物理计算域边界、裁剪边界还是模型查询边界，也不能确定域外延拓或差分模板、验证 rho 单位、生成 BOS 观测或测试重建精度。桥接数据的上游处理与物理单位仍未核实。

V307 是描述性表示敏感性，不是新 warm start、精确算子调用节省、速度/内存、泛化或算法突破；既有严格学习成本判决不变。没有访问 validation、test 或外部集合。

`algorithm_breakthrough=false`；`paper_success=false`；`resource_speedup=false`；`external_generalization=false`；`real_bost=false`。
