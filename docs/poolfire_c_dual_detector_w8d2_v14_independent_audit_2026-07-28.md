# v14 w8d2 独立审计：P0=0，正式 4/5 FAIL 不变

## 审计对象

- 单候选 `w8d2` 模型与 2,912 参数闭包；
- 结果前 JSON 合同；
- 五轨迹 LOTO runner；
- 独立 checkpoint/NumPy replay validator；
- 当前正式私有运行产生的脱敏科学结果。

## 审计结论

```text
P0 = 0
P1 = 2
P2 = 1
authoritative scientific decision = FAIL
algorithm_breakthrough = false
```

没有发现：

- fresh、historical validation 或 untouched test 被用于训练、选择或救场；
- 结果后增加宽度、seed、epoch 或改兼容性阈值；
- source binding 缺少实际执行的 v12 compact block、v11 runner、物理算子或几何依赖；
- checkpoint 在 truth barrier 后被替换；
- 本机绝对路径、私有 checkpoint 或数据摘要进入公开制品。

## 两个 P1

### P1-1：继承报告的 4/5 PASS 标签可能被误读

旧 v11 sentinel 允许“4/5 且失败轨迹无材料性 harm”写为
`PASS_FIT_LOTO_DETECTOR_CNN_SENTINEL`。v14 的结果前规则更严格，只接受 5/5。

这次实际结果恰好是 4/5，所以风险真实出现。处理方式不是改写原始报告，而是明确：

```text
旧继承 report = 数值诊断制品，不是 v14 权威科学判决
V14_GATE       = FAIL
最终独立验证   = FAIL
```

公开页面、机器摘要和学习日志只把最终 v14 gate 作为结论。

### P1-2：中间独立 replay 的 PASS 字样也可能被误读

原始中间状态 `PASS_INDEPENDENT_NUMERIC_REPLAY...` 表示“数值回放一致”，不表示
候选通过科学门，但脱离上下文后容易混淆。

后续源码已把该状态改为 `COMPLETE_INDEPENDENT_NUMERIC_REPLAY...`，最终状态继续
独立写为 `FAIL_INDEPENDENT_STRICT_W8D2_CAPACITY_GATE_V14`。正式已完成运行保持
原字节与 hash，不做事后重写。

## 一个 P2

原 strict gate 先把 trajectory rows 折叠为字典，再检查键集合。正常 runner 固定只
生成五行，但异常报告中的重复 ID 可能被覆盖。

后续源码增加了：

```text
len(rows) == 5
ordered trajectory IDs exactly equal frozen five-ID roster
no duplicate row can pass
```

新增测试同时验证“4/5 必须 FAIL”和“重复 trajectory row 必须 FAIL”。

## 独立数值证据

正式 validator 没有复用 Torch 下游 K1 或正式 compatibility helper。它重新：

1. 读取并核验五个 checkpoint；
2. 生成 505 帧 proposal；
3. 用 NumPy 重算 exact `A^T`、observable alpha 和 strict K1；
4. 重算 field、gradient、observation compatibility；
5. 复核 `1010A + 1010A^T` 调用账；
6. 检查全局 checkpoint-before-truth 和 score visibility barrier。

最大科学数值差为 `3.33e-16`。因此 P45 的 90/101 与最终 4/5 不是 runner
单路径的偶然输出。

## 证据边界

本审计只支持：

> 在五条已经开放的 PoolFire straight-ray proxy fit trajectory 上，结果前冻结的
> 2,912 参数 `w8d2` 候选严格 LOTO 为 4/5；失败集中于 P45 的 11 个
> observation-only 边界帧。

它不支持 fresh 泛化、真实 BOST、wall/RSS 优势、SOTA、算法突破或论文成功。
