# v26.3：外部组合上的曲折光线压力测试

## 一句话结论

冻结的 Observable Reduced Warm K1 没有重训、没有换门槛。它在三条
external-to-fit PoolFire 组合上，经场依赖 eikonal 曲折光线重新生成观测后，
在两个真正达到预注册压力强度的离散档 `beta=0.001`、`0.002` 上逐轨迹通过
field / gradient / observation 兼容门，并保持为最便宜的兼容方法。

这是一个扎实的正向科学增量，但不是算法突破，也不是真实 BOST 成功。

```text
PASS_EXTERNAL_TO_FIT_CURVED_RAY_STRESS_V26_3
algorithm_breakthrough=false
real_BOST=false
paper_success=false
```

## 为什么要做

v24 已经证明固定 warm initializer 能在三条新的功率/尺寸组合上复现，但其
训练、观测与逆算仍共享 straight-ray forward。这样仍可能有一种解释：

> 方法只是利用了训练和测试完全相同的线性正演，一旦光路受折射率场影响而弯曲，
> warm start 的优势就会消失。

v25 首次把 forward 换成场依赖曲折光线，但使用的是三条已经参与过
fit-morphology 开发的轨迹。v26.3 把同一压力施加到 v24 的三条
external-to-fit 组合：

```text
p=33kw_size=05
p=45kw_size=01
p=58kw_size=05
```

因此本轮检验的是两个变化同时出现时，固定方法是否仍稳定：

1. 轨迹组合不在模型拟合集合中；
2. 生成 observation 的 forward 从 straight ray 变为 field-dependent curved ray。

## 如何防止“看答案调实验”

正式运行前冻结：

- 三条轨迹、每条 101 帧；
- 五个离散 beta：`0`、`0.0001`、`0.0005`、`0.001`、`0.002`；
- 192-step 正式积分与 96-step 数值对照；
- 曲率差至少 1% 的每一帧都必须进入 96/192 对照；
- 每条轨迹至少要有 10 个曲率差至少 1% 的帧，才算形成有效压力；
- 同一套八方法、调用账、兼容包络与失败动作；
- robustness 与 cost dominance 分开判断；
- 每个离散 beta 独立判断，不声称连续 beta 区间。

独立审计在正式运行前两次阻止了不充分设计：

1. v26.1 的“压力大小”与“兼容性评分帧”没有绑定，任务在 truth 解码前停止；
2. v26.2 只抽查高曲率帧的数值收敛，并可能让措辞暗示连续区间，因此结果值未读，
   协议修订为 v26.3 后重新正式运行。

还修复了 validator 少绑定 `manifest` 和 payload hashes 的端到端缺陷，并把
v24 release root 限定为固定的本机私有规范路径，避免调用者替换整套输入。

## 曲率压力是否足够大

最高离散档 `beta=0.002`：

| 轨迹 | observation 改变 p50 | p90 | worst | 曲率差 >=1% 帧 |
|---|---:|---:|---:|---:|
| p33-s05 | 2.470% | 3.215% | 3.718% | 101 |
| p45-s01 | 9.708% | 12.335% | 16.360% | 101 |
| p58-s05 | 3.181% | 4.494% | 5.234% | 101 |

p45 的 p90 已达到 12.34%，最坏帧达到 16.36%，所以最高档不是接近零的装饰性
扰动。

`beta=0.0001` 与 `0.0005` 没有被判为算法失败。它们未进入正式通过集合，是因为
三条轨迹没有同时达到“至少 10 帧曲率差 >=1%”的压力门：

| beta | p33 非平凡帧 | p45 非平凡帧 | p58 非平凡帧 | 可作三轨迹压力结论 |
|---:|---:|---:|---:|---|
| 0.0001 | 0 | 0 | 0 | 否 |
| 0.0005 | 0 | 92 | 2 | 否 |
| 0.001 | 69 | 101 | 91 | 是 |
| 0.002 | 101 | 101 | 101 | 是 |

这避免了用近零扰动制造“全 beta 通过”的漂亮结论。

## 数值误差是否伪装成曲率效应

所有曲率差至少 1% 的帧，以及固定的每 10 帧检查点，都重新用 96 steps 计算并
与 192 steps 对照。最高档最坏相对差：

| 轨迹 | 96 vs 192 worst | 冻结上限 | 曲率 p90 / 数值 worst |
|---|---:|---:|---:|
| p33-s05 | 0.143% | 0.5% | 22.41 |
| p45-s01 | 0.450% | 0.5% | 27.44 |
| p58-s05 | 0.186% | 0.5% | 24.11 |

三条均过 0.5% 门，且曲率信号至少是最坏收敛差的 22 倍，因此观察到的 forward
mismatch 不能由积分步数误差解释。

## 重建与成本结果

在 `beta=0.001` 和 `0.002` 上，三条轨迹的共同判决相同：

| 方法 | A | A^T | 总调用 | 三轨迹共同判决 |
|---|---:|---:|---:|---|
| Normalized BP | 101 | 101 | 202 | FAIL |
| Zero CGLS K1 | 101 | 101 | 202 | FAIL |
| **Observable Reduced Warm K1** | **202** | **152** | **354** | **PASS** |
| Full parent Warm K1 | 202 | 202 | 404 | PASS |
| Zero CGLS K2 | 202 | 202 | 404 | FAIL |
| Geometry PCGLS K2 | 202 | 202 | 404 | FAIL |
| Zero CGLS K3 | 303 | 303 | 606 | FAIL |
| Zero CGLS K4 reference | 404 | 404 | 808 | PASS |

最高档的逐轨迹兼容性：

| 轨迹 | all-frame joint match | odd-frame joint match | harm | 结论 |
|---|---:|---:|---:|---|
| p33-s05 | 101/101 | 50/50 | 0 | PASS |
| p45-s01 | 101/101 | 50/50 | 0 | PASS |
| p58-s05 | 99/101 | 48/50 | 0 | PASS，最薄弱边界 |

p58 不是“每帧完全等于 Zero-K4”。它有两帧越过严格 match 线，但没有一帧越过
harm 线，且全帧、奇数帧、非平凡曲率帧三套集合都通过冻结兼容包络。这与 v24
已经公开的 p58 边界一致，不能把 98.02% 写成 100%。

## 独立复算

正式 validator 不接受 runner 自报，而是从原始 rho 重新计算：

```text
1515 个 192-step 正式曲折光线观测
 749 个 96-step 数值对照
 120 个方法判决
```

结果：

```text
maximum formal observation difference = 0
maximum convergence value difference = 0
PASS_INDEPENDENT_FULL_ARM_RECOMPUTATION_EXTERNAL_CURVED_RAY_STRESS_V26_3
```

这说明正式 runner 与独立路径在本机、同一冻结输入和数值实现下逐值一致。它仍不
等于跨代码库、跨硬件或真实实验复现。

## 是否成功

**成功的部分：**

- 排除了“external-to-fit 组合一遇到非线性曲折 forward 就失效”的解释；
- 没有重训、重新选 rank、调整门槛或挑 beta；
- 两个真正达到压力门的离散 beta 都逐轨迹通过；
- 更便宜的 202-call controls 仍未共同通过；
- 正式 forward 与全部方法判决经过独立全量复算。

**没有成功或没有证明的部分：**

- beta 未由组分、波长和 Gladstone-Dale 常数标定；
- 没有真实相机内外参、背景图渲染、位移提取、噪声与标定误差；
- 三条轨迹仍来自同一个公开 PoolFire 数据集；
- 这是 post-open physics stress，不是 official untouched test；
- v26 没有重新测 wall time 与 whole-pipeline peak RSS；
- 没有证明连续 beta 区间；
- 没有证明真实 BOST 同精度、泛化或论文完成。

## 突破性进展判定

```text
important_reproducible_physics_robustness_increment=true
algorithm_breakthrough=false
```

本轮确实是比 v25 更强的证据，因为它把“新组合”和“曲折 forward mismatch”
同时放进同一冻结实验，并通过独立全量复算。但它仍没有跨过真实光学成像链和
组内数据门，不能称为新算法已经在真实 BOST 上成功。

下一项真正可能改变论文等级的证据，不是继续添加 synthetic beta，而是把冻结
方法原样接入组内 forward / observation 合同，用重复测量和标定不确定度定义
真实“同精度”，再统一测完整 A/A^T、wall time 和 whole-pipeline peak RSS。
