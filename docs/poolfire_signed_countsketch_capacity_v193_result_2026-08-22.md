# v193：保留全部弱符号贡献几乎补齐容量，但固定 CountSketch 仍未过门

## 讲人话结论

v192 只从完整逐相机 DCT 表示里补选 `271` 个坐标，五/九相机都停在 `40/52`。这留下一个明确问题：也许不是强坐标选得还不够准，而是许多单独很弱、合起来有用的丢弃坐标被彻底舍弃了。

v193 因此保持固定 `1009` 个 QDEIM 锚点和总计 `1280` 个紧凑通道不变，不再从剩余坐标中挑 `271` 个。它把**全部**非锚点逐相机 DCT 坐标，用结果前冻结的相机 ID 与 DCT 模式哈希聚合进 `271` 个桶；primary 使用固定正负号的 signed CountSketch，便宜 control 使用相同桶但全部取正号。哈希、符号和桶数不读 CFD 真值，也没有搜索 seed、桶数、归一化、门槛或回退；最后仍只运行一轮未修改的精确 CGLS K1。

结果出现了迄今最强的紧凑容量改善。signed primary 在五相机达到 `51/52`，九相机达到 `49/52`；unsigned control 为 `48/52 · 46/52`，v192 选列为 `40/52 · 40/52`，v190 固定子集为 `35/52 · 30/52`。因此可以有证据地说：分散在大量弱坐标里的符号信息确实重要，而且正负抵消不是可以随意丢掉的细节。

但冻结门要求两臂都必须 `52/52`。五相机仍有一个 gradient 失败，最坏值为 `0.755045831`，略高于 `0.75`；九相机仍有三个 observation-only 失败，最坏 observation 为 `0.212354655`，高于 `0.20`。完整标定为 `12/13 · 11/13`，完整时间层为 `3/4 · 2/4`。这已经很接近，但“接近”不能替代完整通过。

完全独立第二实现使用独立的哈希与桶循环、不同 SVD driver，并重新构造物理候选、未修改 K1、逐单元指标、调用账和相机换序审计。`19/19` 检查全真；普通数组最大相对差为 `3.72e-11`，近零数组最大绝对差为 `2.10e-14`；特征、响应、紧凑响应和坐标的相机换序误差均为 `0`，桶与符号的离散换序完全一致。

正式判决为 `FAIL_SIGNED_COUNTSKETCH_CAPACITY_V193`。

这条结果同时包含很强的正证据和严格的负判决：

- 正证据：保留全部弱符号贡献比挑选少量强坐标明显更有效；signed 又稳定优于 unsigned，说明弥散的正规方程信息和符号抵消都是真实机制。
- 负判决：当前冻结的相机-模式哈希、符号约定和 `271` 桶 CountSketch 在两档相机下都没有达到 `52/52`，因此仍不能进入预测器、资源或外部门。

因此关闭这一条精确 CountSketch 机制，不搜索哈希 seed、桶数、归一化、阈值或预算，也不用 CNN/FNO/UNO/DeepONet 或 GPU 把擦线失败包装成成功。后续只能结果前冻结一个物理或表示上真正不同的结果不可见机制，或等待新的成对真实二维双分量 BOST 位移数据。

`algorithm_breakthrough=false`、`paper_success=false`、`exact_call_reduction=false`、`resource_speedup=false`、`external_generalization=false`、`real_bost=false`。

# v193: Retaining all weak signed contributions nearly closes capacity, but the frozen CountSketch still fails

## Plain-language conclusion

v192 supplements the full camera-resolved DCT representation with only `271` selected coordinates and reaches `40/52` strict-safe cells under both five and all-nine cameras. That leaves one concrete alternative explanation: the failure may come not from imperfect ranking of strong coordinates, but from discarding many individually weak coordinates whose aggregate action matters.

v193 keeps the fixed `1009` QDEIM anchors and the same total `1280` compact channels. Instead of selecting `271` remaining coordinates, it aggregates **all** non-anchor camera-resolved DCT coordinates into `271` buckets using a preregistered hash of camera ID and DCT mode. The primary uses fixed signed CountSketch aggregation; a cheap control uses the same buckets with all signs positive. Neither the hash, signs, nor bucket count reads CFD truth, and there is no seed, bucket-count, normalization, threshold, or fallback search. One unchanged exact CGLS K1 step follows.

This produces the strongest compact-capacity improvement so far. The signed primary reaches `51/52` under five cameras and `49/52` under all nine. The unsigned control reaches `48/52 · 46/52`, v192 selected columns reach `40/52 · 40/52`, and the v190 fixed subset reaches `35/52 · 30/52`. The evidence therefore supports two mechanism claims: useful normal-equation information is diffuse across many weak coordinates, and signed cancellation materially matters.

The frozen gate still requires `52/52` in both arms. Five-camera retains one gradient failure, with a worst value of `0.755045831` above the `0.75` limit. All-nine retains three observation-only failures, with worst observation `0.212354655` above `0.20`. Complete calibrations are `12/13 · 11/13`, and complete time strata are `3/4 · 2/4`. Near-pass evidence is useful, but it is not a pass.

A fully independent second implementation uses a separate hash and explicit bucket loop, a different SVD driver, and independently rebuilds physical candidates, unchanged K1 replay, cell metrics, call accounting, and camera-permutation audits. All `19/19` checks pass. Maximum ordinary-array relative and near-zero-array absolute differences are `3.72e-11 / 2.10e-14`; camera reordering produces zero feature, response, compact-response, and coordinate error, with exact discrete agreement in bucket and sign permutation.

Decision: `FAIL_SIGNED_COUNTSKETCH_CAPACITY_V193`.

The result carries strong positive mechanism evidence and a strict negative gate:

- Positive evidence: aggregating every weak signed contribution materially outperforms selecting a small set of strong coordinates, while signed aggregation consistently beats its unsigned control.
- Negative gate: the exact frozen camera-mode hash, sign convention, and `271`-bucket CountSketch fails the required `52/52` capacity in both sensor arms, so no predictor, resource, or external gate is authorized.

Close this exact CountSketch mechanism without searching hash seeds, bucket counts, normalizations, thresholds, or budgets, and without CNN/FNO/UNO/DeepONet or GPU rescue. Continue only with one preregistered physically or representationally distinct result-blind mechanism or genuinely new paired real two-component BOS displacement data.

`algorithm_breakthrough=false`, `paper_success=false`, `exact_call_reduction=false`, `resource_speedup=false`, `external_generalization=false`, `real_bost=false`.
