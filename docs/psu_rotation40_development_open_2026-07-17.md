# PSU rotation-40 真实观测开发集：已打开，尚未形成重投影证据

## 这一轮做了什么

我们按冻结协议只打开了 Penn State flight-body BOS 数据集中的
`rotation 40` 开发成员。最终审计用的 rotation 30、60、70、80 仍然封存。

开发文件的三重完整性锚点均匹配：

| 项目 | 已验证值 |
|---|---|
| archive SHA-256 | `91c17faa7804ed0c56d28e27fc22407b818dce3b00e960533b19210b986cef82` |
| member bytes | `587,535,524` |
| member SHA-256 | `59fcba260010f8cf38505d47df37447a6c8b564e2d1efaad095d6352bea1e934` |
| member CRC32 | `d27a7306` |

原始数组保留在 Git 忽略的私有目录中；公开仓库只保存哈希、形状、聚合统计和
结论边界，不镜像作者的 MAT、图像、掩膜或射线。

## 文件里实际有什么

该 MAT 不是完整的三维重建包，而是四个 `1 x 7` camera-cell 变量：

- `u_new`：作者处理得到的水平位移；
- `v_new`：按作者脚本取负后作为垂直位移；
- `typevector_free`：用于构造 ambient mask；
- `typevector_new`：与 ambient mask 组合后构造 active mask。

七台相机的数组尺寸均为 `2160 x 2560`。相机 2、3、4 已按同一作者变换生成私有开发
shard，供后续与九视角 support 重建保持相机身份一致。

## 真实观测告诉了我们什么

| 相机 | active vector RMS / px | ambient vector RMS / px |
|---:|---:|---:|
| 1 | 0.6154 | 0.2128 |
| 2 | 0.2462 | 0.1756 |
| 3 | 0.3143 | 0.1812 |
| 4 | 0.3257 | 0.2092 |
| 5 | 0.5155 | 0.2553 |
| 6 | 0.4904 | 0.3717 |
| 7 | 0.6109 | 0.2305 |

这里最重要的不是 active 数字大，而是 ambient 区域也有明显非零残差。它混合了光流
估计误差、背景系统偏差、标定误差和真实实验噪声，不能直接当成独立同分布高斯噪声。
当前 release 也没有逐帧重复观测，因此还不能从数据中估计可靠的时间协方差。

## 为什么现在还不能评分

该开发 MAT 不包含同一 rotation-40 对应的：

1. camera extrinsics；
2. background extrinsics；
3. per-pixel ray directions；
4. 与观测行严格绑定的 camera-system constants。

缺少这些量时，无法保证预测向量与实际像素行一一对应。现在强行计算 residual，得到的
数字可能只是行序、坐标系或符号错误。因此当前状态是：

> `ROTATION40_REAL_DISPLACEMENT_PAYLOAD_READY_GEOMETRY_AND_REPROJECTION_UNAVAILABLE`

这不是失败，而是一次成功的证据防火墙：真实观测已接通，最终集仍封存，错误评分尚未
被允许进入论文。

## 下一步

从官方 calibration archive 只读取 rotation-40 的背景标定成员，核验其哈希和变量结构；
随后按作者 `AEDC_pprocess.m` 的相机 2–4 行映射构建几何 shard。第一项合法实验是把冻结
的 support CGLS 场投影到 rotation-40，报告 held-out image-space residual。即使该 residual
改善，也只能说明与未见真实 BOS 观测更一致，不能替代独立三维场真值。

**2026-07-17 状态更新：**上述 calibration member、逐像素行绑定和冻结 B0 全 active
重投影均已完成。pooled vector relative-L2 为 `0.959591`，完整边界见
[rotation-40 真实重投影基线](psu_rotation40_real_reprojection_baseline_2026-07-17.md)。

公开机器可读证据：

- [development access summary](psu_rotation40_development_access_public_summary.json)
- [cell payload audit summary](psu_rotation40_cell_payload_public_summary.json)
