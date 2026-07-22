# E73 Phase-0 24 指标 fixture scorer：先把“怎样算输赢”钉死

> 状态：`FIXTURE_SCORER_IMPLEMENTED_TESTED_PRIVATE_PRODUCTION_CLOSED`
>
> 这是评分器的接口证据，不是算法突破。它没有打开真实 16-unit 分数，没有训练 predictor，也没有授权任何 candidate 接管 `cgls_k4`。

## 1. 为什么先写 scorer

如果不先冻结评分规则，后面很容易在看到结果后改口径：

- 把 9 个 view slot 合并后，藏掉某一台相机或某个 rotation 的伤害；
- 只报 pooled L2，不报 worst view 和 p95 tail；
- 将 candidate 与不同的 baseline 比，把 harm 符号写反；
- 重复跑 CGLS 或漏算 forward，伪造速度优势；
- 把 fixture 上的正常计算写成真实 BOST 性能结果。

新实现 [e73_phase0_metric_scorer.py](../site_tools/e73_phase0_metric_scorer.py) 只做一件事：从同一个 in-memory foundation 取出 `x_proxy`、`y=A(x_proxy)` 和七个非零 checkpoint，先校验真值、观测与 checkpoint 的内部 bytes/hash 绑定，再对每个 checkpoint 只做一次 forward，生成固定的 24 指标与 6 组 harm。

## 2. 七个 action 与精确调用预算

评分顺序固定为：

`k = [1, 2, 3, 4, 6, 8, 12]`

其中 `k4` 是唯一 fallback，候选为 `k1/k2/k3/k6/k8/k12`。foundation 阶段已经消耗：

- 1 次 `A(x_proxy)`；
- 12 次 CGLS forward；
- 13 次 CGLS adjoint。

scorer 只允许再增加 7 次 forward，所以每个 unit 最终必须精确为：

`20 A + 13 A^T`

如果 scorer 入口看到的计数不是 `13/13`，或评分后不是 `20/13`，它会拒绝输出 bundle。

## 3. 24 个指标的精确定义

令真值投影为 `y`，checkpoint 投影为 `y_k=A(x_k)`，误差为 `e_k=y_k-y`。所有分母都有冻结 floor，当前为 `1e-12`；调用者传入任何其他数值都会被拒绝。p95 显式冻结为线性分位插值。

### 3.1 四个全局投影指标

1. `projection_vector_relative_l2`：`||e_k||_2 / max(||y||_2, floor)`。
2. `equal_view_slot_macro_relative_l2`：先独立算 9 个 view-relative-L2，再等权平均。
3. `worst_view_slot_relative_l2`：9 个 view-relative-L2 中的最大值。
4. `group_p95_error_over_signal_rms`：先对每条 ray 算二分量误差的向量模，取全部 ray 的 95% 分位，再除以真值投影向量模的 RMS。

### 3.2 十八个 view-tail 指标

对 view slot `0..8` 各保留：

- `view_i_relative_l2`；
- `view_i_p95_error_over_signal_rms`。

不允许将 camera 2/3/4 跨 rotation 合并，也不允许删除尾部最差的 slot。view id 必须严格是 `0..8`，九个 ray count 之和必须精确等于 operator 的总 ray count。

### 3.3 两个 field 指标

1. `field_relative_l2`：`||x_k-x||_2 / max(||x||_2, floor)`。
2. `gradient_relative_l2`：使用与 forward operator 同一 voxel spacing 和 finite-difference primitive，计算 `||grad(x_k)-grad(x)||_2 / max(||grad(x)||_2, floor)`。

当前 analytic proxy 不是 CFD truth，也不是折射率标定真值。因此 field/gradient 指标只能检查 schema 和符号，不能支持实验泛化主张。

## 4. harm 的唯一符号

对任何 candidate `a` 和指标 `m`：

`harm(a,m) = metric(a,m) - metric(cgls_k4,m)`

因为 24 个指标都是误差量，所以：

- `harm > 0`：candidate 比 `k4` 更差；
- `harm = 0`：与 `k4` 指标相同；
- `harm < 0`：candidate 在该指标上更好。

测试会将 `k12` 替换成精确 truth，验证 24 个误差全部为 0，且相对非零 `k4` 的 field/gradient harm 为负。这只是符号单元测试，不是模型胜出。

## 5. fixture 私有 bundle 的发布边界

scorer 会将 metric rows、harm rows、logical-call ledger 和 claim firewall 序列化为 strict JSON：

- `allow_nan=False`，不允许 `NaN/Infinity`；
- 输出目录必须是当前用户拥有的 `0700` 真实目录；
- 从文件系统根开始逐级 `O_NOFOLLOW` 打开目录，不跟随任何祖先或文件符号链接；
- 文件以 `0600` staging 完整写入并 `fsync`；
- 使用 no-replace hard-link 发布固定文件名，拒绝覆盖旧 bundle；
- 发布前后按 inode 重开并对比精确 bytes、owner、mode 和 link count；任何异常只删除与本次 staging inode 一致的残留。

当前这个 publisher 明确只允许 fixture bundle。它并未与真实 private foundation finalizer 集成，因此真实 16-unit 分数仍是关闭状态。

## 6. 当前验证

- scorer 定向测试：`19 passed`；
- runner + scorer + Phase-0 contract：`80 passed`；
- E73 聚焦集：`156 passed`；data-foundation/scorer 相关链：`109 passed`；bounded fast matrix：`533 passed`。
- 覆盖：24 指标轴与独立 NumPy 数值 oracle、9-view 恒等式、harm 符号、7-forward 调用预算、真值/观测/checkpoint/operator-instance 绑定、所有 checkpoint 在首次评分 forward 前的结构预检、非有限 checkpoint、view id/count 漂移、额外 operator call、非 fixture 状态、冻结 floor、strict JSON、owner-only 发布、首次 `fsync`、初始 `fstat` 和限制性 umask 失败清理、duplicate 和 symlink 拒绝。

上述数字均来自本轮文件冻结后的实际重跑，不是按新增测试数手工相加。

## 7. 下一道门

下一步不是训练 FNO，而是将 scorer 与 private foundation 放进同一个受控进程：

1. scorer 必须消费刚生成的 in-memory `y` 和 checkpoint，不得从 caller 接收伪造 metric；
2. metric/harm bundle 必须绑定 run/unit、config、scorer hash、foundation finalization 和外部 unit anchor；
3. 只在全部 7 次 forward 和 24 指标有限后才发布 metric finalization；
4. 任何半写、换名、重复 unit 或 call-ledger 漂移都 fail closed；
5. commit 后 private preflight 只查 attestation、source snapshot、dot product、空 score ledger 和无重复 unit，仍不打开分数。

这些门全部关闭且人工授权后，才可启动 16 个 analytic proxy 的 schema pilot。即使 16/16 通过，唯一允许状态仍然是：

`PHASE0_SCHEMA_GO_ANALYTIC_PROXY_ONLY_NO_PERFORMANCE_CLAIM`
