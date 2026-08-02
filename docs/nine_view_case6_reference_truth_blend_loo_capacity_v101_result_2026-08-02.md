# v101：公共物理参考域容量通过，但静态先验仍是必须击败的强基线

## 一句话结论

师兄提出的“用微分同胚原理增强坐标变化泛化”已经被拆成两层。v99 先证明标量场、梯度、射线、探测器基和伴随必须一致输运；v101 再证明：**相机无关的物理密度真值比几何相关的重建终点更适合作为公共参考域目标。**

在已开封 BLASTNet H2-air Case 6 的三套九视角几何、30 个物理时刻上，fold-local physical-reference rank 0、4、8、12、16、20 在固定 `beta=0.25` 时全部达到 `90/90`。对应的 geometry-specific endpoint rank 4 只有 `37/90`，observable anchor 单独使用只有 `20/90`。

```text
PASS_REFERENCE_TRUTH_BLEND_CAPACITY_V101
PASS_INDEPENDENT_RECOMPUTATION_REFERENCE_TRUTH_BLEND_CAPACITY_V101
algorithm_breakthrough = false
```

这是一项 truth-aware 表示容量正结果，不是可部署算法、未见坐标系泛化或真实 BOST 成功。

## 真正运行了什么

- 数据仍是已经开封的 Case 6，共 `3 × 30 = 90` 个 geometry-frame 单元。
- 外折仍为 `3 套留出几何 × 5 个连续六帧时间块 = 15 折`，并保留相邻一帧 embargo。
- 每折先去掉相机重复，只用独立物理时刻的三维真值拟合公共参考域 mean 与 SVD。
- 固定比较 rank `0/4/8/12/16/20`、beta `0.25/0.5/0.75`、完整真值上限、匹配的 v100 endpoint controls 和 anchor-only control，共 43 个臂。
- 每个 initializer 都进入同一个 strict CGLS K1；实际总账为 `3870` 次回放、`7740A+7740A^T`，没有 breakdown。
- 每个单元仍同时检查 field、完整 gradient、内部 gradient、observation 相对 Zero-K2 与 Zero-K4 的八个门。

真值只用于回答“这个参考域表示装不装得下”。它没有进入部署可见输入，因此不能把这里的系数称为已经学会。

## 最关键的正式结果

| beta 0.25 的候选 | 八门通过 |
|---|---:|
| observable anchor only | 20 / 90 |
| geometry-specific endpoint rank 4 | 37 / 90 |
| physical reference static rank 0 | **90 / 90** |
| physical reference rank 4 oracle | **90 / 90** |

physical-reference rank 4 的 maximum-gate p50 / p90-higher / worst 为 `-0.13252 / -0.10584 / -0.04735`；rank 0 为 `-0.11987 / -0.04158 / -0.03786`。两者最坏单元都严格位于门内。

增加非零模态确实存在平均收益：rank 4 在 field 和完整 gradient 上逐单元 `90/90` 优于 rank 0，在 observation 上 `78/90` 更好。但它在内部 gradient 上只 `56/90` 更好，按八门中最危险的 maximum-gate 比较也只有 `63/90` 更好、`27/90` 更差。因此 rank 4 尚未稳定支配更简单的静态先验。

## 独立复算怎样落锤

独立 validator 没有导入正式 v101 runner 或容量 core。它从原始 Case 6 真值重新做预处理，重建全部 15 折、unique-frame mean/SVD、43 个臂、3870 个 initializer、3870 次 strict K1、八门与真实调用账。

正式与独立程序在以下对象上的最大绝对差全部为 `0`：

- initializer 数组；
- 最终三维场与观测残差；
- candidate rows 与 fold rows；
- result 与 manifest；
- `7740A+7740A^T` 调用账。

正式输入与输出在验证前后保持不变。独立程序仍共享冻结的 pre-v101 forward/adjoint 与评分物理内核，所以这里证明的是独立实验逻辑复算，不是第二套端到端物理实现。

## 与微分同胚原理的准确关系

一般的“把不同域拉回公共参考域再学习”已经由 [DNO](https://www.nature.com/articles/s42005-024-01911-3)、[Geo-FNO](https://www.jmlr.org/papers/v24/23-0064.html) 和 [DIMON](https://www.nature.com/articles/s43588-024-00732-2) 等工作覆盖，不能把微分同胚本身写成原创。

本项目当前得到的窄结论是：

1. v99 已验证 BOST 代理中的 scalar pullback、gradient `J^{-T}`、camera ray/detector basis 与 forward/adjoint 必须同步输运；
2. v100 已排除直接把 geometry-specific optimizer endpoint 当作公共输出目标；
3. v101 已确认 camera-independent physical field 是有足够容量的参考域目标。

但 v101 没有训练形变场，也没有在未见连续相机位姿、任意微分同胚或固定物理相机下测试。因此只能称为“微分同胚一致的参考域设计获得开发容量证据”，不能称为“微分同胚泛化已成功”。

## 为什么 rank 0 改变了下一步

rank 0 不预测任何模态系数，只使用每个 fit fold 的公共物理场均值，再与部署可见 anchor 固定混合。它已经 `90/90`，比 rank 4 predictor 更简单、更容易解释，也更便宜。

所以下一门不能直接训练大 FNO 或消耗未开封外部工况。必须先在同一已开封 Case 6 上公平比较：

```text
static physical prior rank 0
vs
observation + known geometry -> rank 4 coefficients
```

rank 4 只有在仍保持 `90/90`，并以结果前冻结的 paired margin 稳定优于 rank 0 时，才有资格进入外部坐标变化门。否则就冻结 rank 0 作为更可信的简单方法。

## 当前证据边界

```text
physics-consistent coordinate transport = passed in v99
geometry-specific endpoint common target = failed in v100
camera-independent physical reference capacity = passed in v101
strict observation-only rank4 predictor = not run
continuous pose / arbitrary diffeomorphism generalization = not proven
fresh wall / RSS advantage = not run
real BOST = not run
GPU rental = not needed yet
paper success = false
```
