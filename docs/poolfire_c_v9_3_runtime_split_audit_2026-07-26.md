# PoolFire C 路线 v9.3：双运行时源算子与六模型独立审计

> 日期：2026-07-26  
> 证据增量：源数组、正式请求与六个 Cross14 拟合之间的数值谱系闭合。  
> 科学状态：`algorithm_breakthrough=false`，`outer_scoring_authorized=false`。

## 1. 这次到底解决了什么

v9 正式训练已经生成五个 outer 模型和一个单独的 p14 development 模型，但仍有
两个必须先回答的问题：

1. 正式请求里的 raw BP、geometry-equalized BP 和 K4 teacher，是否真的来自冻结的
   三视角 straight-ray 代理算子？
2. 六个模型是否真的按完整 trajectory nested LOTO、冻结 lambda 网格和 one-standard-
   error 规则拟合，而不是由输出文件自己给自己作证？

原 v9.2 单进程审计正确地 fail closed。原因不是物理源或模型失败，而是源 artifact
与正式拟合由两个不同的 Python/NumPy 运行时产生。一个运行时能逐位复现源算子，
另一个运行时能逐位复现岭回归；强行用同一进程同时要求两边逐位相等是不合理的。

我们没有在看到差异后增加浮点容差。提交 `9764ce3` 先冻结 v9.3 双角色协议，再由
提交 `fddb40b` 实现两个只读 validator。

## 2. 两个角色分别证明什么

### 2.1 Source role

运行时固定为 Python 3.13.9 / NumPy 2.3.5。它重新构造冻结几何，并独立计算：

- 六条开放轨迹、606 帧的 raw `A^T y`；
- `diag(A^T A)`、固定 geometry-only multiplier 和 equalized BP；
- 每帧从零初值运行四步 CGLS 得到 K4 teacher；
- gauge-centered source slice 与六个正式训练请求的逐元素对应关系。

正式结果：

```text
PASS_RUNTIME_BOUND_SOURCE_TO_REQUEST_V9_3
```

五类数组的最大绝对差全部为 `0.0`：

```text
raw BP          0.0
sensitivity     0.0
multiplier      0.0
equalized BP    0.0
K4 teacher      0.0
```

完整调用账与协议一致：

| 生成项 | A | A^T |
|---|---:|---:|
| 606 帧 raw BP | 0 | 606 |
| 606 帧 K4 teacher | 2424 | 2424 |

该角色没有读取 formal fit outputs，也没有运行 nested fit。

### 2.2 Fit role

运行时固定为 Python 3.11.5 / NumPy 2.4.6。它先验证 Source receipt 仍绑定当前未变化的
request tree，再只从正式训练请求独立重算：

- 六行 batch membership；
- 五个 outer heldout 恰好各出现一次；
- p14 development 模型单独报告；
- fold-local scaler；
- 每个完整 trajectory 的 inner loss；
- 完整冻结 lambda 网格；
- one-standard-error 选择；
- 最终 14 参数 ridge 模型。

正式结果：

```text
PASS_RUNTIME_BOUND_SOURCE_AND_NESTED_FIT_BATCH_V9_3
```

六个模型的选择与系数都逐元素精确复现，六行均选择：

```text
lambda = 1e-4
```

该角色没有重新计算源算子，也没有读取 outer 或 p14 的评价数组。

## 3. 为什么这不是“为了通过而拆测试”

两个角色必须同时满足：

1. 使用同一个 clean detached validator commit；
2. Source receipt 绑定的 request-tree digest 与 Fit role 当前读取的 request tree 完全相同；
3. Source role 不能读取模型输出或运行拟合；
4. Fit role 不能重新定义源算子或增加结果后容差；
5. 任一 receipt 缺失、过期、被替换或来自另一提交时，整个组合 fail closed。

定向测试共 `26 passed`，包含错误 parent、错误运行时、软链接 payload、过期 request
digest、错误 validator commit、伪造 PASS 字段和抢跑 breakthrough 声明等攻击。

## 4. 现在可以说什么

可以说：

- 六个正式训练请求来自独立复算且逐元素一致的冻结代理算子；
- 五个 outer 模型和一个 p14 development 模型均由对应训练数组按冻结 nested LOTO /
  one-SE 规则得到；
- 六个模型的选择与最终系数已经被独立重拟合；
- 现在只获准冻结下一阶段 outer prediction / score 协议。

不能说：

- Cross14 已经比 Zero、BP、PCGLS、dual ridge、FNO 或 DeepONet 更好；
- 已经达到 matched field / gradient / observation accuracy；
- 已经减少端到端 wall time 或峰值内存；
- 已经跨轨迹泛化；
- 已经得到真实实验 BOST 结果；
- 已经形成算法突破或论文成功。

## 5. 下一道真实科学门

下一步不是继续扩建审计基础设施，而是在任何 outer 数字打开前冻结：

1. 每个 outer fold 的 prediction request；
2. unchanged Cross14-K2、Zero-K3、Zero-K4、BP、PCGLS、dual ridge 与其他冻结 controls；
3. field、gradient、observation 三类 matched-accuracy tolerance；
4. 完整 `A/A^T`、端到端 wall 和 fresh whole-pipeline peak RSS 账；
5. 逐 trajectory p50、p90、worst 与 harm 判决；
6. p14 mandatory veto 的独立角色；
7. 结果发布模板和失败动作。

只有这道门通过，Cross14 才能从“可信 sentinel”升级为“存在跨轨迹 headroom 的低容量
候选”。即使通过，它也不是最终新算法；最终方法仍需解决可观测子空间约束和强近邻
新颖性问题。

