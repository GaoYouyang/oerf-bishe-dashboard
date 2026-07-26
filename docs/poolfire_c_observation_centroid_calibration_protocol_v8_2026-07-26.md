# PoolFire C 路线 v8：观测可见低容量质心校准协议

## 先讲人话

v6 说明火焰结构确实会移动，但一次反投影 `A^T y` 给出的位置带有工况偏差。v7.1 用固定几何灵敏度做校正，对 p45 和 p14-s05 有帮助，却伤害了 p22，而且仍有五条轨迹没有达到整数位移一致率门槛。

因此 v8 不再继续调 Jacobi floor 或 clip。它只问一个更窄的问题：

> 能不能用一次观测和反投影中已经可见的 8 个位置与视角平衡统计量，训练一个只有 15 个参数的小映射，把 BP 质心校准成更接近 K4 teacher 的质心？

这不是神经算子，也不是最终重建算法。它只是决定“先平移再学表示”这条支线是否还有资格继续。

## 唯一候选

输入全部来自同一次 `BP=A^T y` 和冻结的几何点乘均衡，不增加新的 `A` 或 `A^T`：

1. raw BP 的 3 个归一化质心坐标；
2. equalized BP 与 raw BP 的 3 个质心差；
3. x/z 与 y/z 三视角单位像素能量的 2 个 log-ratio。

合计 8 个特征。视角能量在形成 `A^T y` 前就存在，不增加算子调用，并直接对应“视角不平衡造成定位偏差”的物理假设。协方差虽然可见，但在只有五条独立 fit trajectory 时容易成为工况指纹，所以明确禁止。绝对能量、帧序号、时间、功率、尺寸和 trajectory ID 同样不能进入模型。

模型是多输出仿射残差岭回归：

```text
predicted teacher centroid[j]
  = raw BP centroid[j]
  + half-extent[j] × [1, raw-position[j], equalization-response[j],
                       view-balance-x/z, view-balance-y/z] @ Theta[j]
```

每个轴只有 5 个系数，三个轴共 15 个 float64 参数；不同轴不能偷用其他轴的质心特征，三个轴共享同一个 `lambda`。常数项也参加岭惩罚，因此 `lambda` 趋于无穷时模型严格退回 raw BP 质心。没有隐藏层、特征搜索、输出裁剪或结果后回退。

## 如何防止轨迹泄漏

五条 fit 轨迹逐条做外层留一。每个外层模型只能看到另外四条轨迹的 teacher；正则系数还要在这四条内部再做 leave-one-trajectory-out 选择。被留出的整条轨迹在模型拟合和正则选择时都不存在。`lambda` 只按轨迹等权的连续质心 MSE 使用 one-standard-error rule 选择；T0、exact shift、p90、worst 和 harm 全部留到模型冻结后判决，不能反过来调正则。

`p=14kw_size=01` 仍只是已经看过的开发验证：最终模型用五条 fit 训练，正则只在五条 fit 内选择，然后一次评估 p14。它不能被称作独立验证、confirmatory 或泛化。

帧只是轨迹内部的相关观测，不能把 606 帧写成 606 个独立样本。模型选择和最终判决都以完整 trajectory 为单位、每条轨迹等权。

正式实现把每个 fold 再拆成两个新进程：fit worker 只收到训练轨迹的特征和 teacher，raw 质心由前三个冻结特征无损还原，目录中不存在额外 raw 文件或任何 heldout 输入；predict worker 只收到冻结模型、heldout 特征和 raw 质心，目录中不存在训练数据或任何 teacher。模型、预测和最终结果目录都用操作系统级 no-replace 原子发布，目标一旦存在就 fail-closed；预测文件发布后，主进程才允许用 heldout teacher 评分。独立验证器不导入正式模型、runner 或评分函数，会重新计算全部 331 次岭求解、6 个模型、6 条预测和 43 个数值数组，并要求最大绝对差严格等于 0。

每条 heldout 轨迹还会报告 8 个特征落在该 fold 训练最小值至最大值范围内的帧比例；五条外层 cross-fit 轨迹另报 fold-training canonical center 相对五条 fit 全量 center 的差值。这些都只是解释工况覆盖、参考中心差异与失败原因的诊断，不能参与 `lambda` 选择、T0 判决或结果后调参。

## 过关条件

六条轨迹必须逐条通过 v6 原始 T0：

- 质心 `L∞` 误差 p50 ≤ 0.5 voxel；
- p90 ≤ 1.0 voxel；
- worst ≤ 2.0 voxel；
- integer shift exact ≥ 75%；
- within-one ≥ 95%；
- 大于 2 voxel 的 shift 错误帧数为 0。

还要逐条通过相对 raw 与 equalized 两个控制中较优者的 harm 门：

- p90 不得高于较优控制的 p90 加 `max(0.05 voxel, 5%)`；
- exact fraction 不得低于两个控制中较高者 0.02；
- within-one fraction 不得低于两个控制中较高者 0.02。

任何一条不通过，都停止 shifted-POD 定位路线，回到单独预注册的 observation-to-full-field initializer。只有六条全部通过，才允许另写协议测试 zero-fill aligned containment T1。

## 成本与不能说的话

部署时每帧仍是 `0 A + 1 A^T`；额外只有 2072 个观测分量的视角能量累积、8192 次几何点乘、moments 和最多 15 次模型乘加。嵌套留一与最终模型一共需要 331 次很小的岭回归求解，它属于训练成本，不是部署成本。正式运行仍要测 fresh-process 端到端 wall time 与 peak RSS，但 v8 本身不允许宣称：

- `A/A^T` 调用减少；
- 最终场重建加速；
- 内存下降；
- 神经算子优势；
- 真实 BOST 泛化；
- 算法突破或论文成功。

完整机器可读合同：

`learning_labs/protocols/poolfire_c_observation_centroid_calibration_audit_v8.json`
