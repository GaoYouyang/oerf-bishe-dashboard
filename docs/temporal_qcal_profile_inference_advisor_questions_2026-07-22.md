# 给何远哲师兄的最小确认单：联合三维场与 q_cal 推断

这份清单不是请师兄审核一个已经“成功”的算法。当前 synthetic 结果只定位到：
plug-in 区间欠覆盖，full-profile 迭代略微改善点估计，但是否能对应真实 BOST
仍取决于组内接口与独立重复。

## 建议先发的 8 个问题

1. **当前最需要联合估计的低维参数是什么？** 相机外参、背景板/标定漂移、
   折射路径修正、时间对齐、输运速度，还是别的参数？请按实际痛点排序。
2. **forward callable 能否输入一个三维折射率/密度场和一组参数 `q`，输出预测
   位移或投影？** 如果能，输入输出 shape、单位、坐标系与 mask 是什么？
3. **能否获得 `JVP(q,x,v)` 与 `VJP(q,x,w)`？** 若没有自动微分，现有代码是否
   能对几何参数做有限差分，单次 forward 大约多贵？
4. **straight-ray 与 curved-ray residual 的层级是什么？** 哪一层是组内主模型，
   哪一层可作为更高保真 evaluator？二者是否在相同相机坐标和单位下输出？
5. **真实数据中什么算独立重复？** 不同 shot、不同 acquisition、不同时间段、
   不同 rig，还是只有同一段高速序列？同一背景图/标定文件会跨多少样本共享？
6. **flow-off 或已知 target 的重复标定数据是否存在？** 它们能否估计像素、相机、
   帧和 acquisition 层的噪声 covariance，而不依赖重建真值？
7. **当前组内最强基线和最难失败例是什么？** 请给出 TDBOST/NeRIF/传统迭代法
   的代码入口、checkpoint、指标脚本，以及一两个已知失败 case。
8. **最终更关心哪个物理终点？** 折射率/密度 field-L2、梯度/front、积分量、
   时间一致性、重投影，还是下游温度/速度补偿？哪些指标有可信参考？

## 若师兄愿意给一个最小可运行包

不需要立刻给完整数据集。最小包可只含：

```text
one flow-off repeat group
one easy reacting-flow sequence
one known hard sequence
camera + timestamp + coordinate manifest
forward callable
JVP/VJP or finite-difference wrapper
current baseline command
metric command
```

请同时注明：数据能否公开、只能本地私用，还是只能在实验室服务器运行。受限数据
不会上传 GitHub；公开网页只放 schema、假数据例子和不含观测的汇总。

## 收到回复后的路线分叉

### A. 有独立 shot/session + JVP/VJP

优先做 physical-target orthogonal score、按 acquisition cluster 的 Godambe
covariance，以及真正的 cross-fit。随后才训练小型 operator correction，输出 warm
start、低维 bias correction 或 preconditioner，而不是直接替代 forward。

### B. 有 callable，但没有独立重复

先做可辨识性与 sensitivity audit、penalized profile curve、signed-axis/weak-direction
压力测试。可以研究重建精度，暂不主张频率学 coverage。

### C. 有数据，但没有 callable/JVP/VJP

转向 4D 表征/时空低秩/NeRIF/TDBOST 重建主线，用固定几何做严格 baseline；不要
把 residual-only 权重称为几何可观测性或自动标定。

### D. 只有现有模型输出和指标

先做误差分层、failure taxonomy 和公平 benchmark，等待接口。此时开发新算子模型
很容易只学到当前 baseline 的偏差，不宜直接写论文胜利结论。

## 一句话汇报模板

> 我们先在 synthetic 多帧 BOST 中发现：固定场的 plug-in `q_cal` 置信域明显
> 欠覆盖，完整 variable projection 只能小幅改善点估计；现在想确认组内真实
> callable、JVP/VJP、独立重复和主物理指标，再决定做物理目标正交 score，还是
> 回到固定几何的 4D 重建主线。能否请您帮我核对上面 8 个问题？
