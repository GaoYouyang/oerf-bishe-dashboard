# D0 exact-|A| 生产重放：MPS 同步基础设施修正

## 边界

本修正发生在性能数据打开之前。父协议提交为 `b58fafe`；第一次正式
运行在第一个样本的 exact-|A| basis streaming 阶段失败，没有生成结果目录、
trajectory row、tightness row、power row 或方法排名。

失败信息为：

```text
ValueError: streamed entrywise factor majorizer does not dominate the signed operator
```

## 诊断

父实现连续提交 signed forward、absolute forward 和 MPS-to-CPU 归约，但在
读取两份张量前没有显式同步。失败后只打开同一个第一个样本的基础设施
诊断，不读取任何方法性能：

- 固定 2,322 个父协议 `M`-active 坐标；
- 固定 batch size 128，共 19 批；
- 每批在 CPU 归约前调用 `torch.mps.synchronize()`；
- 19/19 批均为 finite；
- 最大逐元素 `max(|A|-M,0)` 为 0；
- 每批 exact-only entry 为 0；
- 2,322/2,322 列在 signed `A` 与 factor `M` 中均为非零。

因此，父运行的失败不能被解释为已经证实的结构性 majorizer 违例。它暴露了
MPS 异步归约路径缺少明确完成边界。修正只在 basis batch 的两次 forward 后、
任何 finite/dominance/CPU reduction 前增加一次同步。

## 不变项

- signed `A`、factor `M`、active coordinate、target、field、noise 和 geometry 不变；
- `eta=0.7`、`theta=1`、K 集合和方法集合不变；
- dominance、active-set、Schur、安全和 call-ledger 失败门不放宽；
- graph-PCGLS 仍是 full-support 非绑定 headroom，不进入同算子主判定；
- Gate B 仍为 NO-GO；本诊断不能重新打开 Gate B；
- 本修正不构成新算法或性能结果。

## v2 运行要求

v2 必须从包含本修正的新干净提交运行。正式运行前应连续两次重放第一个
样本的 exact/factor sums，并要求 content hash、active counts、dominance 和
row/column replay error 一致；任一失败则停止，不进入 solver trajectory。
