# D0.6 runner v9：把 terminal 端点精确绑定到冻结日程

> 证据等级：**合成工程机械验证**。本轮没有正式训练、三维重建、算子学习性能、真实 OERF 数据、泛化或算法优越性结论。

## 一句话判决

三种参数化可以在 9 个独立合成子进程中各完成一对受控的 `FORWARD -> VJP -> Adam`。v9 额外封存 9 组、4 个 LR 分支、每分支 260 事件的精确哈希索引；terminal 里的最后事件和下一事件必须逐字段命中该索引，且不能跨 LR 分支拼接。这证明的是“terminal 端点属于冻结日程”，**不是**“前面全部历史已被独立重放”。每个 `arm x seed` 所需的 260 事件正式训练器、真实 callable、四候选 LR 重置、S2 step-81 转换、broker 正式射线内容哈希和 checkpoint 内容审计仍未实现，所以终态故意是 `FAILED_SEALED`，科学结论授权保持 `false`。

## 为什么不能沿用 v3-v8

第一轮独立红队发现 v3 仍可能接受旧梯度或 forward 后的参数漂移，也没有把模型、Adam 和 journal 事件逐项绑定到冻结排程；几何缓存还存在协同篡改空间。v3 因此只保留为历史 happy path，不能继续作为当前证据。

修复后的第一次全新运行命名为 v4。它没有被覆盖或删除：九个 worker 都因废弃字段而失败。v5-v7 先后补了合同字段、日志哈希、额外文件拒绝、S2 行为哈希、model/Adam 跨事件连续性和 lifecycle barrier。第三轮审计又证明：v7 只检查 terminal 序号连续，`step=999` 等伪事件仍可封存；batch identity 缺字段时还泄漏原始 `KeyError`。v8 补了这两点，但 manifest 篡改时 CLI 仍输出 traceback，而非规范 `FAIL` JSON，因此再生成 v9。

这两次不利历史的意义是：失败可以推翻旧结论，而不是被新的绿色数字覆盖。

## v9 实际证明了什么

| 检查对象 | v9 当前证据 | 不能外推成什么 |
| --- | --- | --- |
| 模型连续性 | exact class/object、完整张量状态、`requires_grad` 与 S2 residual-active 模式被绑定 | 不能说明模型表达能力更强 |
| 梯度连续性 | forward 前拒绝旧梯度；受控 backward 同时计算 field VJP 与全部参数梯度并封存收据 | 不能说明优化会收敛 |
| Adam 连续性 | 绑定同一对象、参数引用、学习率、有限 moment、state hash；每参数 step 必须精确加一 | 不能说明正式 260 事件已执行 |
| 排程连续性 | worker journal 的已完成事件精确等于冻结 schedule；terminal 端点逐字段命中封存索引并保持同一 LR 分支 | terminal 层尚未重放全部前缀，也不说明预算设计最优 |
| 几何连续性 | 合成运行内的 ray identity 与数值几何 content hash 每次重算一致 | 正式 input identity 尚无 broker 签发的 geometry hash，不能说明真实标定正确 |
| 失败语义 | 当前进程会永久 poison，第二次更新被拒绝；磁盘为 `POISONED` 或 unresolved begin，都不是可恢复的 clean prefix | 不是 Python sandbox、跨进程锁、外部签名或跨 run 唯一性证明 |
| 合成监督 | 3 arms x 3 seeds，共 9 个独立进程；每个只完成 2/260 个事件 | 不是 9 次训练、9 个重建或模型排名 |
| 保存终态 | 9/9 tombstones，terminal package 和保存包复核均为 `PASS`；barrier 绑定 arm/step/sequence | 没有 checkpoint、audit score 或 GT 指标 |

## 现场验证数字

- 定向测试：`103 passed, 18 warnings`；警告只来自继承的 `torch.jit.script` 弃用提示。
- 静态检查：Ruff `PASS`。
- 证据验证器现场重跑上述 pytest 与 Ruff，而不是只相信合同里的旧数字。
- 重新核对：11 个源码哈希、9 个正式日程族、2 个复合批次、16 个证据合同字段、v9 summary/verification/terminal bundle/schedule manifest。
- v9 worker：9/9 exit code 0；每个只执行一对 `FORWARD/VJP/Adam`。
- v9 保存包：`failure_count=0`，最终状态 `FAILED_SEALED`，`audit_unlock_authorized=false`。
- 负测：额外 worker 文件、stderr 漂移、manifest 篡改和合同 worker 数修改均返回结构化 `FAIL`。
- 科学结果：formal training、checkpoint、reconstruction、operator learning、generalization、paper success 和 breakthrough 全为 `false`。

## 初学者先掌握六个概念

### 1. Forward、VJP 与 Adam 是三个生命周期

forward 生成观测预测；VJP 把观测损失沿算子传回三维场和模型参数；Adam 再使用这组梯度更新参数。三者必须分别记账，才能阻止同一梯度被重复更新。

### 2. 参数对象相同还不够

即便 Python 对象没换，forward 后也可能原地修改参数、buffer 或执行模式。因此 v9 同时绑定张量状态、参数 `requires_grad` 和 S2 residual-active 标志，并在 VJP 和 optimizer 前再次核对。

### 3. Adam 也是实验状态

Adam 的动量会记住历史梯度。只保持学习率和参数相同，却重新建立一个空 Adam，会改变算法；v9 把对象、参数引用、配置、有限 moment、状态哈希和逐参数 step 一起绑定。

### 4. 哈希链合法不等于事件合法

单看哈希链时，一个不在预注册日程里的事件也可能拥有自洽的新链。v9 对 worker journal 逐事件比较 schedule，并用独立封存的 event manifest 检查 terminal 端点。但 terminal 中的 prefix hash 仍是调用者收据，尚未在 state 层重放全部 1…N 事件。

### 5. ray ID 相同不等于物理几何相同

相同 view/pixel 编号在坐标轴、单位、标定或 origin/direction 改变后会代表另一条光线。v9 在合成包内同时绑定离散身份和数值几何，并给投影函数副本，避免原地污染缓存。正式数据必须由 broker 从独立标定 manifest 签发 content hash，不能把“同一运行自己计算的期望值”写成物理几何正确。

### 6. `FAILED_SEALED` 是正确的机械终点

它表示九个 worker 都留下不可被 checkpoint 覆盖的终止票据，audit 和 GT 没有打开。dry-run 的任务是证明“一对事件能执行，也能干净停止”，不是制造好看的 loss。

## 证据上限与残余风险

- terminal 端点已精确命中 schedule，但 state 层没有从 worker journal 重放完整 1…N 前缀；这是下一版合同，不是 v9 结论。
- projector capability 只是可靠调用者模型下的 Python 正确性防护，不是同一解释器中任意代码的权限边界。
- manifest 在每次写入前会重读、验哈希和比较内存快照，但没有跨进程锁或外部见证。
- 如果 `OPTIMIZER_COMMIT` 与 `POISON` 同时无法持久化，当前进程仍不可恢复，磁盘审计则显示 unresolved optimizer begin，不能称为“外部永久 poison”。
- 合成 geometry content hash 对比成立；正式 geometry 的可信冻结来源仍为 0。

## 仍缺的三道科学门

1. **真实接口门**：何远哲师兄确认 straight/curved forward 层级、最小 callable、ray 排序与单位、JVP/VJP 或 `A/A^T`、标定版本和真实主痛点。
2. **正式执行门**：broker 为正式 batch 签发 ray-content hash；另行实现 260 事件 trainer、4 个学习率筛选、S1/S2 lockstep、S2 第 81 步 optimizer 重建和 checkpoint 内容审计。
3. **科学评价门**：真实或至少多个独立三维场、跨 geometry/session、CGLS/H1/TV 与 neural-operator 强基线、field/front/逐 rig 尾部、噪声与标定漂移、A/A^T 调用和端到端成本。

## 发给何远哲师兄的七个最小问题

1. 当前主 forward callable 的函数签名和最小输入输出是什么？
2. straight-ray 与 curved-ray residual 分别在哪一层计算？
3. 是否已有可信 JVP/VJP 或矩阵无关 `A/A^T`，调用成本怎样记？
4. rays 的顺序、origin/direction、坐标轴、长度单位和标定版本怎样绑定？
5. 当前最痛的真实失败是少视角、曲光线失配、噪声/饱和、标定漂移、时序速度，还是 PIV 补偿？
6. 师兄认可的最低强基线、训练/验证切分和主指标是什么？
7. 能否先给一个不含敏感数据的 tiny callable fixture，只做一对真实 forward/VJP 合同测试？

## 当前允许的下一动作

独立复核已完成；下一个真实增量必须来自师兄确认的接口。拿到 tiny callable、可重放 journal snapshot 与正式 ray-content hash 后，只替换合成投影入口，先重做一对事件机械测试；仍不读 audit/GT。该门通过后才实现完整 trainer，再申请单个 opened field 的正式参数化筛选。

**突破监测：否。** 新增的是旧结论被推翻后重新建立的模型、梯度、Adam、排程、几何与终态连续性证据；算法效果、重建、泛化和论文结果仍完全未知。
