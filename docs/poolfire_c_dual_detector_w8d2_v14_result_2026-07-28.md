# PoolFire C v14：2,912 参数候选在五轨迹门上 4/5，严格失败

## 一句话结论

`w8d2` 把已通过的 `w16d2` 从 10,548 参数进一步压到 2,912 参数，
但五条完整 fit trajectory 留一评估只通过 4 条。唯一失败的
`p=45kw_size=05` 有 90/101 帧同时满足 field、gradient、observation
兼容包络，结果前合同要求至少 91/101，而且五条必须全部通过。因此正式状态是：

```text
FAIL_INDEPENDENT_STRICT_W8D2_CAPACITY_GATE_V14
full_fit_authorized=false
synthetic_resource_gate_authorized=false
algorithm_breakthrough=false
paper_success=false
```

这是接近门槛但明确的负结果。没有增加 epoch、seed 或候选宽度，也没有打开 fresh、
stopping validation 或 untouched test 来挽救它。

## 为什么做这次实验

v13 已经把 `w16d2 + exact A^T + alpha + K1` 的完整调用减半转化成明显的稳态
wall 优势，并把旧 native 实现的 RSS 降低 22.41 MB；但相对于公平的流式 Zero-K4，
严格 RSS ratio 仍以约 68 kB 失败。

继续调 allocator、线程栈或分位数不会改变科学判断。v14 因此只问一个更直接的问题：

> 在不改输入、物理提升、alpha、K1、训练损失和数据角色的情况下，能否用一个
> 只有 2,912 参数的模型保住五轨迹兼容性，从结构上继续降低推理资源？

结果前只冻结了一个候选：

```text
width = 8
dilations = (1, 2)
parameter count = 2,912
extra seeds = 0
epochs = 120
```

它保留与 `w16d2` 相同的多视角 packing、严格奇对称/齐次构造、全局 context、
post-K1 teacher-deficiency loss、exact `A^T`、observable alpha 和 strict CGLS K1。

## 真正运行了什么

- 五条已经开放的 fit trajectory，每条 101 帧；
- 五个完整 leave-one-trajectory-out fold；
- 每折只用另外四条轨迹训练 120 个 epoch；
- 所有五个 checkpoint 冻结前，不读取任何 held-out gauge truth；
- 五个 checkpoint 全部冻结后，才一次性读取五条 held-out truth 并评分；
- deployment 账为 505 帧、`1010A + 1010A^T`，即每帧 `2A + 2A^T`；
- 第二套 NumPy 路径重新加载五个 checkpoint，独立重算 proposal、
  `A^T -> alpha -> K1`、三类指标和调用账。

没有读取 `p45-s03` fresh、`p14`/`p22` historical development 或两条 untouched
test。

## 五条完整 LOTO 结果

| 留出轨迹 | joint matched | joint harm | severe | teacher-q p90 | 判决 |
|---|---:|---:|---:|---:|---|
| P14-S05 | 101/101 | 0/101 | 0 | 0.11680 | PASS |
| P22-S03 | 101/101 | 0/101 | 0 | 0.11186 | PASS |
| P33-S01 | 101/101 | 0/101 | 0 | 0.13158 | PASS |
| P45-S05 | **90/101** | 0/101 | 0 | 0.15859 | **FAIL** |
| P58-S03 | 101/101 | 0/101 | 0 | 0.12697 | PASS |

关键点不是平均 4/5，而是预注册的逐轨迹 no-harm/compatibility 门。任何一条失败，
都不能用另外四条抵消。

## P45 到底为什么失败

P45 的 11 个未匹配帧为：

```text
5, 27-32, 62-65
```

逐帧独立诊断得到：

```text
field failure frames       = 0
gradient failure frames    = 0
observation failure frames = 11
joint harm frames          = 0
severe harm frames         = 0
```

也就是说，2,912 参数模型没有把三维场或梯度整体做坏；它只在三个时间片段的
observation 兼容性上略微越界。11 帧正超界 margin 的中位数为 `0.000576`，
最坏为 `0.002712`。

这确实很接近，但门槛要求至少 91 帧匹配，而实际只有 90 帧。即使只差一帧，也不能
在看过结果后把 90% 写成“约 90%”或降低阈值。

同样 11 帧上，已通过的 `w16d2` observation margin 中位数为 `-0.007797`，
最靠近门槛的值仍为 `-0.003525`。因此这不是指标本身随机贴线：
更大的 `w16d2` 在相同帧上保有可见余量，而 `w8d2` 的容量压缩损失集中体现在
measurement consistency。

## 独立复算与红队

独立 NumPy replay 的最大科学数值差为：

```text
3.3306690738754696e-16
```

五个 checkpoint、505 帧、`1010A + 1010A^T`、truth barrier 和 score barrier
均重新核验。

红队没有发现数据角色越界、路径泄露、结果后改 seed/epoch、缺失执行依赖或
fresh/test 偷看，但指出一个真实命名风险：继承的 v11 报告把“4/5 且失败轨迹无
harm”称为 `PASS_FIT_LOTO_DETECTOR_CNN_SENTINEL`。这个标签只属于旧 sentinel
语义，不是 v14 的科学判决。

v14 结果前合同明确要求 5/5，因此权威制品是 `V14_GATE` 和最终独立验证：

```text
inherited report decision = PASS_FIT_LOTO_DETECTOR_CNN_SENTINEL
authoritative v14 gate     = FAIL_W8D2_CAPACITY_GATE_V14
independent final status   = FAIL_INDEPENDENT_STRICT_W8D2_CAPACITY_GATE_V14
```

后续源码已经把中间 replay 状态改为中性 `COMPLETE`，并显式要求恰好五个、无重复
trajectory row，避免旧标签被单独引用为成功。当前结果仍绑定原始运行提交，未被
事后重写。

## 成功、失败与突破边界

已成功：

- 单候选、单 seed、五轨迹完整 LOTO 真正执行完毕；
- 参数相对 `w16d2` 减少 72.39%；
- 4 条轨迹 101/101 compatibility，P45 也保持 0 harm、0 severe；
- 所有调用账、checkpoint 和科学指标通过独立复算；
- 失败机制被定位到 P45 的 11 个 observation-only 边界帧。

未成功：

- 没有达到五轨迹 5/5；
- 没有授权 full-fit checkpoint；
- 没有授权 v13 同口径的 wall/RSS 资源实验；
- 没有 fresh 泛化、真实 BOST、真实相机标定或实验噪声结果；
- 没有算法突破或论文完成。

## 科学结论

在当前 fixed packing、loss 和 strict K1 shell 下，`w16d2` 到 `w8d2` 的纯宽度
压缩已经越过跨工况 compatibility 的稳健下限。失败集中在 P45 的 observation
consistency，而不是 field/gradient harm，说明下一项有价值的改动应针对“如何以
部署可见信息维持测量一致性”，而不是继续盲目删参数或增加随机种子。

按照冻结停止规则，纯容量压缩路线到此结束。`w16d2` 仍是当前最小的已验证
5/5 候选；这是一条扎实的容量下界证据，不是新的算法成功。
