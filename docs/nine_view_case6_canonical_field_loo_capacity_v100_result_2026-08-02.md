# v100：把不同相机几何下的重建结果塞进同一个线性场空间，独立验证后失败

## 一句话结论

v99 已经证明三维标量场、梯度、射线与伴随可以按同一坐标变换正确输运。v100 继续问：**三种相机几何分别得到的 truth-aware 重建结果，是否已经位于一个可跨几何复用的公共物理场子空间？**

答案是否定的。直接把每个单元自己的目标场送入同一 K1 外壳可以 `90/90` 通过，说明求解外壳没有问题；但在完整留出一套相机几何和一个连续时间块后，由其余数据建立的 rank 0 至 40 场基最多只有 `2/90` 通过。

```text
FAIL_NO_CANONICAL_FIELD_LOO_CAPACITY_V100
PASS_INDEPENDENT_RECOMPUTATION_CANONICAL_FIELD_LOO_CAPACITY_V100
algorithm_breakthrough = false
gpu_training_authorized = false
```

## 实验怎样防止偷看

- 数据是已经开封的 BLASTNet H2-air Case 6，三套九视角几何、30 个物理时刻，共 90 个 geometry-frame 单元。
- 外折是 `3 套留出几何 × 5 个连续六帧时间块`，共 15 折。
- 每折完全删除留出几何的 30 个单元；在其余几何中继续删除同一六帧和相邻一帧 embargo。
- 只用折内的 truth-aware 最终重建场拟合 mean 与 thin SVD；候选秩在结果前固定为 `0/8/16/24/32/40`。
- 投影后的三维场作为 initializer，再运行完全相同的 strict CGLS K1；每次真实记录 `2A+2A^T`。
- 评价仍是 field、完整 gradient、内部 gradient、observation 相对 Zero-K2 的等调用非劣与相对 Zero-K4 的 no-harm 八门。

这是一项 truth-aware 表示容量诊断，不是部署模型。留出目标的投影系数来自真值，只用于回答“这个输出空间装不装得下”，没有假装它能由观测预测。

## 正式与独立复算结果

| 输出表示 | 通过数 | 投影 field-L2 p50 | p90-higher |
|---|---:|---:|---:|
| direct target + K1 ceiling | **90 / 90** | 0 | 0 |
| rank 0 | 0 / 90 | 0.8053 | 1.1344 |
| rank 8 | 1 / 90 | 0.6542 | 0.8259 |
| rank 16 | 2 / 90 | 0.6117 | 0.7781 |
| rank 24 | 2 / 90 | 0.5986 | 0.7603 |
| rank 32 | 2 / 90 | 0.5723 | 0.7495 |
| rank 40 | 2 / 90 | 0.5705 | 0.7397 |

增加秩只让场投影误差缓慢下降，没有让八门通过数继续增长。rank 40 在三套留出几何上分别只通过 `1/30`、`1/30`、`0/30`。因此这不是“再加几层网络”能合理修补的小误差。

独立 validator 没有导入正式 v100 runner 或容量 core。它重建全部 15 折、SVD 符号约定、630 个 initializer、630 次严格 K1、八门和汇总。正式与独立程序的 candidate-row 最大绝对差为 `2.21e-13`，场最大绝对差为 `2.43e-14`；输入与正式输出在复算前后字节不变。

## 这个负结果真正排除了什么

它排除的是下面这条具体路线：

```text
不同几何下的 truth-aware 优化器终点
  -> 直接视为同一坐标含义的三维输出样本
  -> 一个跨几何线性 PCA / FNO 输出空间
```

这些场虽然都存放在同一个 `32×16×16` 数组中，但它们包含几何相关的可辨识方向、正则化偏差和求解器选择。数组索引相同，不等于统计坐标已经规范化。

它没有排除：

- 相机无关的真实密度场参考域表示；
- 非线性坐标输运或可逆形变模型；
- 连续相机 pose/ray 条件化；
- 在真实 BOST 数据上成立的其他 warm start。

## 为什么这与微分同胚建议直接相关

[DNO](https://www.nature.com/articles/s42005-024-01911-3)、[Geo-FNO](https://www.jmlr.org/papers/v24/23-0064.html) 和 [DIMON](https://www.nature.com/articles/s43588-024-00732-2) 的共同思想，是先把不同域上的物理量拉回公共参考域，再学习参考域上的算子。v100 说明“几何相关重建结果”不是一个好的公共标量场目标。

因此下一门 v101 改用相机无关的物理密度真值作为参考域目标，并用部署可见的 detector-dual anchor 做固定安全混合。它仍先做 truth-aware 容量门；只有容量通过，才有资格训练 observation + continuous ray/camera descriptor 到参考域系数的小模型。

## 当前证据边界

```text
coordinate_transport_interface = passed in v99
geometry-specific endpoint linear LOO capacity = failed in v100
continuous pose prediction = not run
arbitrary diffeomorphism generalization = not proven
external generalization = not proven
real BOST = not run
paper success = false
```
