# BLASTNet H2-air S2：第二残差伴随方向确实新增秩，但仍未守住梯度门

更新：2026-07-29

## 这轮实际检验什么

v50 已经表明：在原五方向的 2,304 个固定候选里，没有同时通过 field、gradient、
observation 三项门的点。但这不能证明连续五维空间无解，也不能判断“增加一个真正
的新方向”是否有用。

v51 因而只在已经打开的 BLASTNet H2-air `phi=0.5` S2 快照上回答一个更窄的问题：

> 沿现有第五方向解释一次 curved observation residual 后，再把剩余 residual
> 通过固定 Direct-K3 线性化的精确伴随算子送回三维场，能否得到旧五维 span 外的
> 第六方向，并在相同搜索预算下通过完整精度门？

这是开封后的机理诊断，不是外部泛化检验，也不是部署算法。

## 方向是怎样构造的

记 Direct-K3 场为 `x3`，冻结在 `x3` 的曲射线线性化为 `J3`，观测残差为
`r0 = y - F(x3)`。现有第五方向先解释一部分残差，得到 `r1`。随后构造：

```text
g1 = J3^T r1
d6 = P_coarse(g1) - projection_to_span(d1, ..., d5)
```

其中 `P_coarse` 是固定的粗网格投影。它只使用部署时可见的观测、冻结几何、
forward/JVP/VJP，不读取真值来构造方向。

准确名称是“固定线性化下的第二个粗网格投影曲射线残差伴随增广方向”。它与投影
Gauss-Newton / augmented Krylov 思路有关，但没有构造完整 Hessian，也不是完整
二阶优化器。

## 独立验证先回答：第六方向到底是不是真的

正式 runner 与独立流程重放器分开实现。重放器不导入 v51 runner，重新生成
`d5`、`d6`、S5/S6 搜索、参考指标和最终判门，并保证在 Stage A 重放完成后才加载
评分真值。两者仍共享冻结的 v44 curved forward/JVP/VJP、几何、metric 与 gate
数值内核，所以这不是外部团队的独立物理实现。

```text
独立验证状态
PASS_INDEPENDENT_RECOMPUTATION_SECOND_RESIDUAL_ADJOINT_V51

Stage-A 数组最大绝对差                  0
S5 场最大绝对差                         0
S5 prediction 最大绝对差                0
最终 score 最大数值差                    2.44e-15
```

方向审计结果：

```text
d5 与 v44 方向相对差                     4.56e-15
d6 在粗网格旧五维 span 外能量             30.84%
J3 d6 在观测旧五维 span 外能量            38.54%
观测切空间秩                              5 -> 6
选中的第六方向权重 w6                     0.244372
伴随恒等式最大相对误差                    5.10e-14
有限差分 JVP 最大相对误差                 4.08e-10
```

因此第六方向在当前固定线性化、gauge、归一化和 `[-2,3]^6` 参数域内不是局部数值
重复项，而且搜索确实使用了它。这不证明全局 forward-operator 秩、抗噪秩或物理
可辨识性。

## 相同预算下的实际结果

S5 是原五方向控制，S6 是加入第六方向后的实验。为了防止 S6 仅因多做搜索而占便宜，
两者的名义搜索调用账都补齐为：

```text
9 F + 48 JVP + 0 VJP
```

S5 的自然账是 `5 F + 48 JVP`，额外四次 F 只是 no-op padding，不增加搜索机会；
S6 的自然账就是 `9 F + 48 JVP`。方向准备成本另记为
`1 F + 1 JVP + 2 VJP`，不藏进搜索账。因此这不是端到端总成本相同或速度优势。

相对 Direct-K4 的误差比为：

| 方法 | field | gradient | observation | 完整门 |
|---|---:|---:|---:|---|
| S5 五方向控制 | 0.976447 | 1.031173 | 1.013825 | FAIL |
| S6 第六方向增广 | 0.975858 | 1.030041 | 1.008749 | FAIL |
| S6 但强制 `w6=0` | 0.976052 | 1.030495 | 1.015399 | FAIL |

门线是 `1.01`。S6 把 observation 从门外推到门内，field 也在门内；但 gradient
仍为 `1.030041`，比门线高约 `0.020041`。它还没有通过“不劣于 Zero”的梯度安全
要求。这里的 gradient 是 `16x16x32` 粗网格 proxy 上的有限差分指标，不是高分辨率、
含噪或真实 BOST 梯度证据。

正式结果是：

```text
NO_OBSERVATION_ONLY_WITNESS_S2_V51
```

## 成功了什么，失败了什么

成功的是机制判断：

1. 第六方向真实离开旧 span，观测切空间秩从 5 增到 6；
2. 搜索给了它非零权重；
3. 在相同搜索预算下，它把 observation 推进了 1.01 包络；
4. 独立实现完整复算出同一结果。

失败的是完整算法目标：

1. gradient 仍比 Direct-K4 门高约 2.0 个百分点；
2. 没有得到三指标同时等价的 warm start；
3. 没有资格扩到其他快照、训练神经网络或测速；
4. 没有 matched-accuracy、调用减少、wall/RSS 加速、真实 BOST 或论文成功。

这里的 `w6=0` 消融固定 S6 的前五个权重后清零第六权重，没有重新优化前五维；它与
独立 S5 控制共同支持“第六方向对这个最终点有贡献”，不能证明连续五维空间无法得到
同样 observation。

这说明“继续沿 residual 做伴随增广”能够补 observation 缺口，却没有解决当前最硬的
gradient-safety 矛盾。继续机械增加第三、第四条同类 residual-adjoint 方向，信息
价值已经很低。

## 下一步由什么证据决定

v51 失败不能推出整个六维 span 无解，因为当前搜索目标只看部署可见 observation。
下一项高价值诊断应当用冻结的六维 span 做一次有界、答案可见的可行性封顶检查：

1. 若六维 span 内存在三指标 witness，说明方向集合有 headroom，真正问题在
   observation-only selector，需要设计部署可见的 gradient-aware surrogate 或
   fail-closed gate；
2. 若冻结六维候选也没有 witness，则停止 residual-only basis chain，改为显式带
   梯度正则或结构先验的新方向；
3. 无论哪一种，都不先扩大网络，因为当前还没有可学习的合格标签。

```text
direction_is_real_and_used=true
observation_gate_crossed=true
complete_gate_pass=false
six_space_impossibility_proven=false
run_other_opened_snapshots_authorized=false
neural_training_authorized=false
matched_accuracy=false
speedup=false
external_generalization=false
real_bost=false
algorithm_breakthrough=false
paper_success=false
```
