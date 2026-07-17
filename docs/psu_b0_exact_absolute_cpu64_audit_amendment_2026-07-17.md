# D0 exact-|A| 生产重放：CPU float64 证书后端修正

## 为什么 v2 仍然停止

v2 提交 `8667fda` 在每个 MPS basis batch 后显式同步，并要求第一个样本
连续重放两次。正式运行在比较两次 content hash 时停止，没有生成结果目录、
solver trajectory 或方法性能行。

后续只进行了基础设施诊断。相同 MPS 进程中的重复 streaming 会随机出现：

- CPU 归约后的 NaN；
- `M`-active / signed-`A`-nonzero coordinate count 漂移；
- row/column replay error 为 NaN；
- 偶发的虚假逐元素 dominance violation。

这些现象在单次显式同步 replay 中可以消失，却不能在重复 replay 中稳定复现，
所以 MPS 不再被允许承担 exact-|A| 或 factor-M 证书构造。

## v3 修正

v3 保留 MPS 作为已经使用过的 solver trajectory 后端，但从同一份冻结 MPS
runtime 深拷贝 operator 与 whitening 到 CPU，再提升为 `torch.float64`。以下
工作只在 CPU float64 完成：

- exact signed `A` basis streaming；
- factor `M` basis streaming；
- entrywise `M >= |A|` 检查；
- row/column sums 与 active-coordinate 对照；
- Schur 安全证书；
- power-iteration 非绑定压力估计。

同一 CPU64 首样本预检得到：

- `M`-active coordinate：2,322；
- signed-`A`-nonzero coordinate：2,322；
- `M`-active / `A`-zero coordinate：0；
- 最大逐元素 violation：0；
- factor row replay relative error：约 `9.14e-16`；
- factor column replay relative error：约 `2.27e-16`；
- 包含冻结 runtime 构造的总墙钟时间：约 3.8 秒。

该结果只证明 CPU64 审计路径可构造且首样本通过，不是最终 D0 性能结果。

## 审计边界

- v1 与 v2 的失败均保留在 Git 历史和本说明中；
- v3 必须继续要求首样本两次 CPU64 content hash 完全相同；
- CPU64 与 MPS 的 `M`-active coordinate indices 必须完全相同；
- safety tolerance 收紧到 CPU64 数值尺度，不允许用放宽阈值掩盖 MPS 问题；
- solver 的 signed `A`、target、K、method、eta、theta 和 opened sample 不变；
- graph-PCGLS 仍为 full-support 非绑定 headroom；
- Gate B 仍为 NO-GO；D0 仍不是新算法或实验泛化证据。
