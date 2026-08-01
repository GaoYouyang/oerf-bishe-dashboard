# v85：Case 4 全过，但 Case 6 暴露了内部梯度风险门的失效

## 先说结论

v85 是第一次把 Case 3 上开发完成的 warm initializer 原封不动放到两个按顺序冻结的外部反应流工况。模型、归一化、几何、阈值和 CGLS 回退在读取 Case 4、Case 6 真值后都没有更新。

- Case 4：`84/84` 个单元通过全部八门，三档几何各 `28/28`；`41` 个停在 K1，`43` 个继续到 K2，平均为 `2.5119A + 2.5119A^T`，相对 Zero-K4 的 `4A + 4A^T` 减少约 `37.2%`。
- Case 6：只有 `78/90` 个单元通过，三档几何分别为 `28/30`、`27/30`、`23/30`；冻结 residual gate 把 `90/90` 全部留在 K1，因此正式结论是 `FAIL_EXTERNAL_CASE6_ACCURACY_V85`。
- 联合门要求 `174/174`，所以 v85 总结论是 **FAIL**。不能启动正式 wall/RSS 资源门，也不能声称外部泛化、算法突破或论文成功。

这不是“整体重建崩了”。Case 6 的 field、整体 gradient 和 observation 对照门全部通过；12 个失败只出现在内部梯度：4 个违反相对 Zero-K4 的 no-harm 门，8 个违反相对同调用 Zero-K2 的优势门。

## 为什么这个结果仍然有价值

Case 4 是一个真实的零适配外部正结果，而 Case 6 给出了很清楚的反例。两者合在一起，把问题从“模型能不能迁移”缩小到更具体的一点：

> 当前只看全局 observation residual 大小的单标量安全门，识别不了三维内部局部梯度风险。

Case 6 的 12 个失败分数落在 `0.24797` 到 `0.35614`，而 78 个通过单元的分数覆盖 `0.21709` 到 `0.36313`，两类高度重叠。事后若仍坚持“分数低就停 K1”这一维规则，为了零漏判，最多只能安全接受 `11/90`。这个数只能用于解释失效机理，不能拿来改阈值后补考 Case 6。

## 实验怎样避免看答案调参

1. 所有可训练部分只使用已开封的 Case 3；Case 4 和 Case 6 的顺序在读取数值前冻结。
2. 每个工况先一次性把 rho 变成九视角 observation；预测进程只接收 observation、已知几何和冻结模型，不接收真值或评分路径。
3. 全部 K1/K2 分支、三维场、残差和实际 A/A^T 回执先封存，之后评分进程才读取真值。
4. 独立 validator 重建几何、重新预测、重新评分并核对每个调用回执。两例的预测场、残差、评分与调用账最大差均为 `0`。

## 下一步不是什么

不是立即换 FNO、UNO 或更大 U-Net，也不是用 Case 6 真值偷偷调低阈值后重新宣布通过。当前表示在 Case 6 的 field、整体 gradient 和 observation 上已经很好，真正缺的是一个部署可见、能感知局部内部梯度风险的选择信号。

下一项机制诊断应先回答：对同一个 warm K1，强制再走一步未修改 CGLS 后，12 个失败是否都能被救回。如果能，问题主要在安全门；如果仍不能，说明 warm 表示本身也缺局部梯度容量。这个诊断只能作为已开封 Case 6 的事后机理证据，任何新门控都必须在新的未打开工况上重新做一次性外部门。

## 证据边界

- `algorithm_breakthrough=false`
- `paper_success=false`
- `external_generalization=false`
- `resource_advantage_proven=false`
- `curved_ray=false`
- `real_BOST=false`

准确表述是：**Case 3-only、零适配的 observation-only warm policy 在 Case 4 全部通过，但在 Case 6 因内部梯度风险漏判而失败；当前 scalar residual gate 不足以支撑跨工况严格部署。**
