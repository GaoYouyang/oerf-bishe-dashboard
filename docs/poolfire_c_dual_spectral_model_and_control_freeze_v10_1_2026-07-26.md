# PoolFire C 路线 v10.1：最小双空间模型、线性对照与执行阻断

> 日期：2026-07-26
> 主线：何远哲师兄确认的 C 路线
> 当前状态：架构与 NumPy 推理冻结；正式训练、outer prediction、score 均关闭
> 突破状态：`algorithm_breakthrough=false`

## 1. 这轮到底完成了什么

研究目标没有改变：

```text
公开 PoolFire rho
-> 独立 straight-ray 密度梯度 proxy
-> 16x16x32 三维逆问题
-> observation-only warm initializer
-> 未修改 CGLS refinement
-> 在最终精度兼容时减少完整 A/A^T、wall time 和内存
```

这轮冻结了第一个可训练候选的**精确结构**，但没有训练它。

```text
输入 y：2072 维三视角 float64 observation
固定表示：每个视角、每个分量取 4x4 DCT，共 96 个系数
额外可见特征：3 个视角能量比例 + 1 个投影能量比例
网络：100 -> 32 -> 32 -> 96
激活：SiLU
输出偏置：无
参数：7,360 个，float64 共 58,880 bytes
输出：z = y + 有界 DCT 残差
```

去掉输出偏置不是为了省 96 个参数。奇对称化使用
`0.5[phi(u)-phi(-u)]`，输出偏置会被代数抵消，保留它只会产生永远为零的梯度。

## 2. 由结构保证什么

模型的 residual 使用奇对称 logits、逐帧 RMS 和固定 DCT，因此在浮点误差内满足：

```text
G(0) = 0
G(a y) = a G(y)，a 为任意有限实数
||G(y)-y||_2 <= 0.5 ||y||_2
模型内部 A/A^T 调用数 = 0
```

这些只保证缩放一致性、零输入行为和修正幅度上限。它们**不保证**：

- field / gradient / observation 最终非劣；
- CGLS K1 一定达到 Zero-CGLS K4 的精度；
- wall time 或内存更低；
- 跨工况、真实 BOST 或论文成功。

## 3. 训练目标已经在结果前锁死

每个 fit frame 的教师固定为同一 observation、同一 geometry 下的
zero-start CGLS K4。候选必须经过：

```text
z_theta(y)
-> h = A^T z_theta
-> alpha in [0, 2]
-> x0 = alpha h
-> unchanged CGLS K1
```

然后才计算相对 K4 teacher 的三个 deficiency：

```text
d_f = field deficiency
d_g = gradient deficiency
d_o = observation deficiency
q = max(d_f, d_g, d_o)
```

完整 trajectory 的损失是全部 101 帧 `q` 的均值，加上最差 11 帧 `q` 的均值。
不同 trajectory 等权，禁止随机 frame split 和 pooled-frame 伪重复。

五个 outer LOTO row 每步各取四条 fit trajectory 的一帧；development-p14 row
每步取五条 fit trajectory 的一帧。每个 epoch 固定 101 步，每帧恰好使用一次。

## 4. 三个必须同场比较的表示匹配对照

为了区分“DCT/RMS 表示有效”和“神经非线性有效”，已实现三个只读 observation 的
NumPy 推理对照：

| 对照 | 参数 | 作用 |
|---|---:|---|
| clipped identity DualRange | 0 | `z=y`，但 alpha 仍截断到 `[0,2]` |
| diagonal DCT dual filter | 96 | 每个 DCT 系数一个线性增益 |
| full DCT dual map | 9,216 | 96x96 全线性映射，固定 L2 正则 |

它们与 MLP 共用逐帧 RMS、固定 DCT、修正上限、`A^T` 提升、alpha 和 CGLS K1。
两种线性对照也必须优化同一个 K1 后 trajectory-tail loss；当前没有唯一的
observation-space `z` 教师，所以不得把 full linear map 写成“闭式 ridge 解”。

如果 MLP 不能在每个要求的 row 上同时打赢两种线性对照，就不能把收益归因于神经
非线性，也不允许换更大的 FNO/UNO/DeepONet 挽救。

## 5. 独立审计发现的两个 P0

### P0-A：旧 Python callback 能绕过调用账

旧接口把 `dual_initializer` 作为同进程 Python callback 传入。`AuditedLinearOperator`
又公开保存底层 operator。对抗测试证明，callback 可以直接调用底层 `A^T`：

```text
真实底层调用：1A + 2A^T
wrapper 正式记录：1A + 1A^T
旧函数仍接受结果
```

因此旧接口只能保留为机制原型，不得进入正式 runner。正式路径必须改成：

```text
纯推理 worker：y -> 数值 z + 一次性 receipt
物理 worker：验证 receipt 后才执行 A^T、A、alpha 与 strict CGLS
```

两个 worker 是受测量父进程启动的 sibling exec，均不得再创建子进程；模型 worker
拿不到 operator、truth、trajectory role 或评价 membership。

### P0-B：旧 solver 的 breakdown 行为不符合 v10

v10 要求 denominator 非有限或非正时在更新前 fail closed。旧通用 PCGLS/CGLS
实现会把 `alpha` 设为 0 后继续，因此不能直接复用到正式 v10 arms。

下一步必须实现一个 strict-v10 solver，并让 Zero K1/K2/K3/K4/K8、PCGLS、
identity、线性 controls 和 MLP 全部经过同一套 breakdown 语义。

## 6. 一个容易混淆但已修正的名字

v10 arm registry 里沿用了 `normalized_bp_cgls_k1` 这个历史 ID，但正式语义是
**clipped identity DualRange + CGLS K1**。它一般不等于旧 normalized BP。

反例 `A=0.5I`：

```text
legacy normalized BP scale = 4
DualRange unclipped alpha = 4
DualRange accepted alpha = 2
```

网页和论文中不能再把两者写成恒等方法；正式表格应同时保留 canonical display name
和历史 arm ID。

## 7. 当前验证

定向联合测试：

```text
61 passed
```

覆盖模型 DCT 正交性、参数预算、零映射、实数齐次性、修正上限、checkpoint digest、
严格 float64、线性对照、合同篡改、旧 callback 绕账反例、identity/BP 反例和 v10
外层合同。

当前环境没有 PyTorch，因此也没有训练实现、checkpoint、正式环境 receipt 或模型
输出。此时安装框架既不能跨过两个 P0，也会增加磁盘压力，所以没有先安装再宣称进度。

## 8. 下一步只做这四件事

1. 实现 strict-v10 CGLS/PCGLS，并与冻结 reference 做数值等价及 breakdown 反例测试。
2. 定义数值 proposal artifact、一次性 receipt 与 callback-free physics API。
3. 实现 capability-isolated inference/physics sibling worker 和 process-tree RSS 计量。
4. 完成所有便宜 controls 的 fit-only 拟合实现与独立 validator。

四项全部通过后，才重新讨论 PyTorch CPU 环境、训练授权和六个 trajectory row 的正式
checkpoint。`p45-s03` fresh 与两条 test 继续封存。

## 9. 对“独一无二”的准确结论

当前作品指纹仍是组合级假设：

```text
BOST-specific observation-space proposal
+ exact Range(A^T) lift
+ pre-A^T abstention
+ unchanged strict CGLS K1
+ matched-cost and three-metric trajectory non-inferiority
+ full call / wall / RSS accounting
+ later real-BOST transfer
```

截至当前有界检索，没有发现完全同构公开方法；但各组件都有强近邻，尤其 WB-IPM、
Learned ReSeSOp、learned warm start、learned backprojection 和 neural
operator + Krylov。没有 outer 结果、fresh holdout、真实 BOST 与组内 IP 核对前，
只能写 `global_uniqueness_proven=false` 和 `algorithm_breakthrough=false`。
