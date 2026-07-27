# PoolFire C v16.1 独立资源结果审计

## 审计结论

只读审计重新计算了 36 条 resource records、36 份 worker report、`36×101` 条
逐帧调用 receipt、字段确定性、native 数值等价和全部配对资源统计。

```text
P0 = 0
P1 = 0
scientific decision =
  FAIL_INDEPENDENT_PUBLIC_DEVELOPMENT_RESOURCE_GATE_V16_1
algorithm_breakthrough = false
```

该判决有效。它是一项扎实的负资源结果，不是加速成功。

## 独立复核通过的部分

- 18×2 roster 完整，两个 arm 各先运行 9 次；
- progress、seal、READY 和 independent validation 一致；
- 两个 arm 各自 18 次 field 输出逐字节确定；
- candidate 调用账为 `202A + 202A^T`；
- Zero-K4 调用账为 `404A + 404A^T`；
- candidate 对正式 v16 field 的 p90/worst relative-L2 为
  `9.24e-8 / 1.04e-7`；
- reference 与正式 Zero-K4 逐值一致。

因此“native 忠实复现”和“完整算子对减少 50%”通过。

## 独立复核失败的部分

### Wall

```text
candidate median = 0.201 s
reference median = 0.258 s
paired median reduction = 22.0%
candidate faster = 16/18
```

但首个 candidate 为 `1.165 s`，同对 reference 为 `0.259 s`，paired harm
为 `350.3%`，远高于 5% 上限。另一个较慢配对 harm 为 `3.59%`，仍在门内；
因此 wall 失败由首个全局冷启动触发，不是统计实现错误。

首个 candidate 的额外时间主要位于非求解阶段：约 `0.987 s`，后续中位约
`0.038 s`。这指向 checkpoint/native library/context/page-cache 初始化，但现有
证据不能继续区分动态链接、页缓存、内存分配与调度。

### CPU

```text
candidate child CPU median = 0.303 s
reference child CPU median = 0.245 s
paired CPU ratio median = 1.231
```

candidate 最多 6 线程，reference 为 1 线程。典型 wall 收益以多 23.1% 总 CPU
工作换得，不能称计算成本降低或 equal-core speedup。

### RSS

```text
candidate peak-RSS p90 = 116.2 MiB
reference peak-RSS p90 = 109.8 MiB
ratio = 1.0581
gate = 1.05
```

虽然只超约 0.8 个百分点，冻结门仍必须判失败。

## 反事实检查

即使只为理解机制、事后删除首个冷启动配对：

```text
paired wall median reduction remains about 22.6%
paired CPU ratio median remains about 1.228
RSS p90 ratio remains about 1.058
```

因此整体负判不是只靠一个异常值；CPU 和 RSS 两门仍独立失败。该反事实不得替代
正式判决。

## 允许公开的结论

> 在一条已消费的公开 development trajectory、101 帧和 18 组配对 fresh-process
> 测量上，native Dual-K1 以约 `1e-7` 场 relative-L2 复现正式实现，并将完整
> `A/A^T` 调用从每帧 `4/4` 降到 `2/2`。其配对 wall 中位数降低 22.0%，但首个
> 冷启动出现 350.3% wall harm；peak-RSS p90 增加 5.8%，配对 CPU 时间中位数
> 增加 23.1%。因此预注册资源门失败。

## 禁止声明

- 不得称端到端稳定加速或冷启动加速；
- 不得称内存 no-harm、CPU 成本下降或 equal-core speedup；
- 不得把 50% 调用减少写成 50% wall 减少；
- 不得用删除首个配对后的 post-hoc 数字改判；
- 不得称 confirmatory test、跨工况泛化、真实 BOST 或物理同精度；
- 不得称 SOTA、论文成功或算法突破；
- 不得公开私有路径、哈希、checkpoint、release 或模型资产；
- 不得称 cryptographic one-time batch 已证明。
