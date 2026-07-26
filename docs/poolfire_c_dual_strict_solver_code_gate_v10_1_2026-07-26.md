# PoolFire C 路线 v10.1：strict CGLS/PCGLS 代码门

> 日期：2026-07-26
> 状态：严格数值求解器代码门已实现；正式隔离执行仍未实现
> 训练、outer prediction、score：全部关闭
> 突破状态：`algorithm_breakthrough=false`

## 1. 为什么不能直接沿用旧求解器

旧通用 CGLS/PCGLS 在搜索方向退化时，会把 `alpha` 设成 0 后继续循环。这个行为适合
某些宽松的基线诊断，却不符合 v10 结果前冻结的规则：

```text
denominator 非有限、非正或低于 1e-30
-> 在 field / residual 更新前失败
-> 不允许伪装成一次正常的零步长迭代
```

因此本轮没有修改旧文件，而是新增独立的
`learning_labs/poolfire_c_dual_strict_solver_v10_1.py`。旧结果仍可复核，新方法和全部
正式 controls 未来必须统一走 strict 路径。

## 2. strict 路径现在保证什么

一次有效迭代按下面的不可交换顺序执行：

```text
检查 gamma
-> 尝试 A p，并先记调用
-> 检查 raw output dtype / shape / finite
-> 检查 ||A p||^2 > 1e-30
-> 检查 alpha finite
-> 在临时数组计算 candidate field / residual
-> 检查候选全为 finite
-> 最后一次性提交 field / residual
```

如果任一门失败，当前候选不会写回已提交状态。失败回执记录：

- 失败阶段与目标迭代；
- 已完整提交的迭代数；
- 本地状态机尝试的 `A/A^T` 次数；
- strict wrapper 看到的 `A/A^T` 次数；
- 最后已提交 field 的摘要；
- residual 是否已经真实形成；若不存在则明确为 `null`；
- `field_or_residual_update_after_failure=false`；
- `algorithm_breakthrough=false`。

## 3. 这轮补掉的五类漏洞

第一轮独立审计没有发现错误候选已经提交的 P0，但找到五个 P1。当前修订分别处理为：

| 审计问题 | 当前处理 |
|---|---|
| 初始化阶段自报调用数可能漏账 | strict solver 要求初始化账等于同一 strict wrapper 的完整生命周期快照 |
| 旧 wrapper 会先把 raw float32 转成 float64 | 新 strict wrapper 在任何 cast 前检查 raw dtype、shape 和 finite |
| `FixedDiagonalSPD` 的只读数组仍可能被重新打开篡改 | solver 入口重新检查 float64、finite、shape、全正，并复制为私有只读数组 |
| 所有异常都被叫作 numerical breakdown | dtype/shape 是 contract error，底层 Python 异常是 execution error，数值退化才生成 breakdown receipt |
| 初始 residual 不存在时曾用假数组填摘要 | 回执现在显式记录 `committed_residual_exists`，不存在时摘要为 `null` |

attempt ledger 也改为先完成保护性只读复制，再登记将要进入 wrapper 的调用。

## 4. 已做的反例与等价测试

定向测试从 15 项扩到 26 项，覆盖：

- 正常满秩问题上，strict CGLS 的 K1/K2/K4 与冻结旧实现逐 checkpoint 数值等价；
- 固定正对角 PCGLS 与旧实现数值等价；
- normalized BP 的一次性 projection cache 不重复计算；
- exact first-step 后的 rank breakdown 不产生第二次伪更新；
- 真正有限但小于 `1e-30` 的 denominator 在更新前被拒绝；
- raw forward/adjoint 返回 `NaN/Inf`；
- raw forward 返回 float32 或错误 shape；
- raw operator 尝试原地修改只读输入；
- 被重新打开并改成负数的 SPD diagonal；
- 伪造 initializer preparation 调用账；
- 同一个 operator 在求解前出现未申报调用；
- 同进程自证明 Python initializer 被 strict 路径拒绝；
- observation dtype、checkpoint 类型和 denominator floor 篡改；
- 超大有限 observation 导致 gamma overflow 时，回执仍绑定真实 residual。
- 外部 `np.errstate(over="raise")` 不能绕开统一 breakdown receipt；
- strict wrapper 不再公开 raw operator 句柄；
- 带 normalized-BP 准备成本的失败回执覆盖完整生命周期。

当前结果：

```text
strict focused: 26 passed
v10.1 model / controls / contract / legacy-negative joint: 99 passed
all PoolFire C related regression: 376 passed
final independent audit within this code-gate boundary: P0=0 / P1=0
```

这些是代码与合同证据，不是 PoolFire 重建性能结果。

## 5. 为什么现在仍不能写“2A+2A^T 已证明”

strict solver 只解决 refinement 的数值语义和 wrapper 内调用账。下面这些正式执行部件仍
不存在：

```text
y -> 数值 z 的只读 proposal artifact
一次性、不可重放的 proposal receipt
拿不到 A/A^T 的 inference worker
拿不到模型 callback 的 physics worker
pre-A^T gate 状态机
17 arms 的统一 runner / validator
102 outputs 的全局 barrier 与单次 score token
fresh process wall 与 whole-pipeline peak RSS
```

新 strict wrapper 已改为组合并移除公开 raw handle，但它仍是同进程 Python 代码门；
Python 反射和同进程权限不等于安全边界。完整生命周期绑定能拒绝 strict wrapper
可见的漏账，却不能替代进程能力隔离。因此当前准确口径是：

```text
strict_v10_numerical_solver_code_gate_implemented=true
formal_callback_free_physics_worker_implemented=false
formal_accepted_branch_call_reduction_proven=false
model_training_authorized=false
outer_performance_opened=false
algorithm_breakthrough=false
```

## 6. 下一有效门

下一轮不训练模型。先定义一个只含数值 proposal 的 artifact 与一次性 receipt，再让
物理 worker 只接受这个 artifact，而不是 Python callback。只有 inference worker 和
physics worker 的能力边界、完整调用 receipt 与失败回退都通过独立审计，才能把
DualRange 接受分支接到这套 strict CGLS K1 上。

即使未来正式账确认为 `2A+2A^T`，也只代表调用目标实现。是否成为论文结果，仍取决于
逐 trajectory 的 field/gradient/observation 非劣、harm、wall、RSS、fresh holdout
和真实 BOST 迁移。
