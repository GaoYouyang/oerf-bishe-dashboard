# v10.9 全视角线性 KRR：容量放大后仍是 0/5

## 先说结论

这一轮没有得到算法突破，但得到了一条足够可信、会改变下一步模型设计的负结果：

> 在五条 PoolFire fit trajectory 的完整留一评估中，冻结的九点、无截距、
> observation-RMS 归一化 full-linear kernel ridge 配置为 `0/5`。即使允许在外层
> held-out 结果上挑最有利的正则参数，仍然没有一条完整轨迹通过原
> field / gradient / observation compatibility 门。

这句话只否定本次明确测试的线性 KRR 配置，**不等于证明一切线性算子都不可能**。

## 为什么做这一轮

v10.8 已经证明：

- 完整 K3 dual certificate 作为目标在五条轨迹上 `5/5` 可行；
- 六通道增益、全频 DCT diagonal ridge 和最近邻都为 `0/5`；
- 单看 certificate relative-L2 会漏掉经过 `A^T` 和 K1 后危险的方向。

因此 v10.9 去掉“每个频率只能独立缩放”的限制，允许每个输出 detector 坐标依赖
全部 2072 个输入坐标。模型用样本空间 kernel 形式实现：

```text
z_hat(q) = q X^T (X X^T + lambda I)^-1 Z
```

它是完整跨 view、跨频率线性映射；每个外层折只用另外四条 trajectory 拟合。

## 我实际跑了什么

每个外层 held-out trajectory 内部又做一次完整 trajectory 留一，比较九个冻结的
相对正则值。设置了三个判决臂：

1. `target-selected`：内层 certificate relative-L2 最小；
2. `safety-selected`：内层 compatibility、severe harm、joint harm 等按冻结次序选择；
3. `outer oracle`：允许偷看外层结果挑正则，只用于判断这个有限模型族有没有
   headroom，不能部署。

三个臂使用同一个下游：

```text
predicted dual -> exact A^T -> observable alpha -> unchanged CGLS K1
```

部署调用账仍是每帧 `2A + 2A^T`，参考仍是 zero-start CGLS K4。

## 逐轨迹结果

下表给的是最宽松的 `outer oracle`，不是可部署选择器：

| held-out trajectory | 最有利 lambda | target p90 | joint match | observation harm | severe |
|---|---:|---:|---:|---:|---:|
| P14-S05 | 1e-2 | 0.4431 | 0% | 91.09% | 0 |
| P22-S03 | 1e-2 | 0.4001 | 0% | 86.14% | 0 |
| P33-S01 | 1e-4 | 0.5432 | 0% | 4.95% | 0 |
| P45-S05 | 1e-2 | 0.6084 | 0% | 100% | 15 |
| P58-S03 | 1e-2 | 0.4799 | 0% | 100% | 1 |

三个臂的总判决完全相同：

```text
target-selected: 0/5
safety-selected: 0/5
outer oracle:    0/5
```

## 为什么这不是“正则没选好”

`outer oracle` 已经对每条 held-out trajectory 分别看完九个 lambda，再按最终
compatibility 挑最有利者。它仍为 `0/5`，所以至少在冻结的九点网格内，失败不能
甩给 inner selector。

更值得注意的是，full linear KRR 的 target p90 为 `0.4001–0.6084`，反而比 v10.8
局部结构更强的 DCT diagonal 控制 `0.1775–0.3342` 更差。505 帧对于一个隐式
高秩、跨坐标线性映射仍太少；完全自由的跨频率耦合损失了 detector 局部和平移结构，
跨工况外推时明显过拟合。

## 独立复核

第二套验证器没有导入正式 runner、v10.9 KRR 实现、Torch K1 helper 或正式
compatibility helper。它重新实现并重跑：

- 四步 CGLS 与 K3 dual certificate；
- 九点 kernel ridge；
- 五层 outer/inner trajectory 划分；
- NumPy `A^T -> alpha -> K1`；
- field / gradient / observation compatibility。

全部数值叶子的最大差为 `4.44e-16`，判决仍是 `0/5`。独立红队同时确认：

```text
P0 = 0
P1 = 0
LOTO leakage = none found
fresh / validation / test reads = none
```

## 这对下一步意味着什么

现在有三条连续证据：

1. full K3 dual 目标本身 `5/5` 可行；
2. 局部 diagonal / gain / nearest 控制 `0/5`；
3. 冻结的完整线性 KRR 仍 `0/5`。

所以才授权一个、也只授权一个最小非线性 detector sentinel。它必须保留：

- 只看部署可见 observation；
- exact `A^T` range lift；
- observable alpha；
- 未修改 CGLS K1；
- 五条完整 trajectory LOTO；
- loss 直接约束 K1 后 field / gradient / observation，而不是只拟合 certificate。

v11 的 77,020 参数奇对称多视角 CNN 已在本机开始五折训练。它是否成功只能由完整
五折结果决定；在此之前：

```text
algorithm_breakthrough=false
fresh_generalization=false
real_BOST=false
paper_success=false
```
