# 公开 Phantom 1：clean-room forward identity v0 失败与 v1 修复协议

日期：2026-07-23  
证据等级：`E1_PUBLIC_SINGLE_PHANTOM_POSTOPEN_IMPLEMENTATION_DIAGNOSTIC_ONLY`

## 1. v0 先失败，不能改写成“差一点通过”

v0 固定读取作者仓库 `a385cce...` 的 Phantom 1，自己实现 16 位 PNG 解码、相机 ray、
ray--AABB 相交、三维场梯度和 midpoint quadrature。没有 import 或执行作者 Python/CUDA，
也没有把图像、mask、体数据或作者源码复制进本站仓库。

v0 的 6144 条采样 ray 中，6021 条与声明的 ROI 相交，相交率 `97.998%`，低于原先
`99%` 门；CGLS-TV 的 `64 -> 128` quadrature 相对差为 `5.260%`，高于 `2%` 门。
所以正式状态是：

```text
D0_FORWARD_IDENTITY_NO_GO_FIX_GEOMETRY_OR_UNITS
```

这个状态永久保留。相机中心方向、固定 XYZ 轴和发布 CGLS-TV 重放门虽然通过，也不能
覆盖两个失败门。

## 2. 只读诊断看到了什么

v0 打开后只做两项定位，不训练模型、不换采样 ray：

1. 未与 ROI 相交的 123 条 ray，其观测向量 RMS 为 `0.01449`；相交 ray 为 `3.77856`，
   比值 `0.003834`。这支持把 ROI 外 ray 解释为“零场贡献”，而不是把它从分母删除。
2. 提高 quadrature 后，GT 的 `128 -> 256` 相对差为 `0.00570`，发布 CGLS-TV 为
   `0.00962`；因此 128 步不适合作主值，但可以作为 256 步的收敛比较。

这两项只解释 v0 为什么失败，不能把同一批已看过的数据叫 fresh。

## 3. v1 在运行前冻结的语义修复

机器合同：
[open_nir_bos_d0_forward_identity_v1_protocol.json](../learning_labs/open_nir_bos_d0_forward_identity_v1_protocol.json)

v1 保持相同 12 个 train views、每视角 512 条 ray、seed `20260723` 和同一组像素选择哈希。
只改两件具有物理语义的事：

- `outside_aabb_policy = zero`：不相交 ray 的预测固定为零并保留在误差分母中；
- 主 quadrature 固定 `256`，用 `128` 比较，差仍须 `<= 2%`。

新加的 ROI 外观测门是 `RMS(outside)/RMS(inside) <= 1%`。这不是把 99% 相交门降低到
98%，而是把“没有穿过未知场”从错误的无效样本语义改成可检验的零场语义。

## 4. 即使 v1 全过，能说什么

最高只允许：

> clean-room 直线 ray 算子在一个公开 synthetic phantom 上，能以冻结坐标、单位和积分
> 规则重放发布观测到足以启动 D0 matched-budget overfit smoke 的程度。

仍然不能说：

- 开发了新算法；
- 完成真实 BOST 或 OERF 三维重建；
- 跨 field、跨 geometry 或跨装置泛化；
- 优于 NeRIF、Neural Refractive Index Primitives、FNO 或 DeepONet；
- 论文成功或突破。

## 5. 对下一步算法设计的实际意义

若 v1 通过，下一步不是马上堆 hash/Fourier 混合网络，而是用同一个 operator shell 做极小
overfit 对照：发布 CGLS-TV、低分辨率 voxel、Fourier MLP，以及有界 residual proposal。
所有方法使用相同 ray、forward 调用、步数和端到端时间；由于只有一个 field，结果只用于
检查梯度、损失、成本和反例，不能承担泛化结论。

若 v1 仍失败，就停止表示学习，回到相机坐标、ROI 轴序、MATLAB `imresize3` 差异和
`uvtoeps` 单位链。不能用训练网络吸收接口错误后再宣称模型更强。

**突破监测：没有突破。** 这是把公开数据接进可审计算子的基础门。
