# 公开 NIR-BOS 代码：从“看得到”到“跑得动”的可复现门禁

审计日期：2026-07-22

证据级别：`PUBLIC_CODE_STATIC_READINESS_AND_ENCODING_SMOKE_ONLY`

机器状态：`FOURIER_MPS_KERNEL_PASS_FULL_RELEASE_BLOCKED`

## 1. 先说人话

这次发现的作者代码很重要，因为它第一次给了我们一个与三维火焰 BOST 直接相关、包含 Phantom 1 数据和训练入口的公开实现。它让“复现 neural refractive-index primitive”从纯文献阅读推进到真实工程审计。

但公开仓库不等于这台 Mac 能原样跑，也不等于可以复制进我们的仓库。现在真正通过的只有一个很窄的组件门：作者的 Fourier 坐标编码可以在 Apple MPS 上做前向、一阶导和二阶导。完整 CUDA renderer、训练和三维重建均未运行。

一级来源：

- [Neural Refractive Index Primitives 论文全文](https://arxiv.org/html/2605.11454)
- [作者公开代码仓库](https://github.com/Weihu22/Neural-Implicit-Reconstruction-for-BOS)
- [NeRIF 论文全文](https://arxiv.org/html/2409.14722v2)
- [TDBOST DOI](https://doi.org/10.1145/3809488)

## 2. 冻结了什么

作者仓库固定在 commit：

```text
a385cce83d88df24ed05dccfd6fde20e124f5604
```

机器审计记录 483 个 tracked files、2,574,218,471 bytes。约 2.32 GB 来自 62 个 tracked Visual Studio cache 文件；它们增加下载体积，但不是新的训练样本或科学证据。

根目录没有 `LICENSE`、`COPYING` 或 `NOTICE`。唯一检出的许可文件位于 `MATLAB/Third_party_tools/sec2hours/`，只属于该第三方工具。当前边界是：允许本机只读审计、引用论文和仓库、记录哈希，并独立写 clean-room 实现；不把作者 Python、MATLAB、CUDA、数据或二进制复制到本站和 Git 仓库。此处只是保守工程策略，不是法律意见。

## 3. 数据不是两个数据集

仓库中 Python 和 MATLAB 各有一个 `Phantom 1` 目录。机器审计逐相对路径、文件大小和 SHA-256 比较后得到：

| 项目 | MATLAB 目录 | Python 目录 | 判定 |
|---|---:|---:|---|
| 文件数 | 71 | 71 | 相同 |
| 字节数 | 92,846,449 | 92,846,449 | 相同 |
| tree SHA-256 | `84f583...f7bf` | `84f583...f7bf` | 相同 |

所以它们是同一份数据的镜像，独立数据集计数只能记为 1。

Python manifest 的视角拆分是：

| split | frame | loader 实际使用的路径引用 | 全部 manifest 路径引用 |
|---|---:|---:|---:|
| train | 12 | 36 | 48 |
| validation | 2 | 6 | 8 |
| test | 2 | 6 | 8 |

64 个路径都写成 Windows 反斜杠。在 POSIX 上原样命中 `0/64`，把 `\\` 规范化为 `/` 后才命中 `64/64`。loader 对缺文件会先 `continue`，随后再对列表 `np.stack`；因此在 Mac/Linux 上不修正路径就不是一个可信的数据加载实验。

还有一个更严重的 split 问题：validation 的 `2/2` 相机位姿和 test 的 `2/2` 相机位姿都分别复用了训练集前两个位姿；validation 与 test 两对位姿又完全相同。进一步解码图像、mask、`img_mask` 和 RI integral 后，validation/test 的 `8/8` 对像素数组完全相同。PNG 文件哈希可因编码元数据不同而不同，所以这里以解码像素为准。

因此这组 `12/2/2` 只能叫同一 phantom 的文件级拆分，不能叫 unseen-camera split；validation 和 test 也不是两个独立审计集。它不证明跨视角、跨 phantom、跨 geometry、跨 session、真实火焰或 OOD 泛化。新的评估相机必须从独立角度重新生成或重新采集，不能沿用当前 test 名称。

## 4. Mac 上究竟过了什么

审计器动态读取固定 commit 的 `encoding.py`，没有复制源码，也没有改作者仓库。固定输入为两个三维点，Fourier 配置是 `input_dim=3, multires=6`。

| 检查 | 结果 |
|---|---:|
| device | Apple `mps` |
| 声明/观测输出 | 39 / `2 x 39` |
| forward finite | 通过 |
| first derivative finite | 通过 |
| second derivative finite | 通过 |
| output L2 | 6.0834 |
| first-derivative L2 | 2.0078 |
| second-derivative L2 | 4.9001 |

它只回答：“Fourier 编码这个局部数学核能否在本机自动微分？”答案是能。它没有加载相机、没有射线积分、没有计算位移、没有优化折射率场，也没有产生一张三维重建。

## 5. 为什么完整入口仍是红灯

| blocker | 固定代码事实 | 后果 |
|---|---|---|
| CLI 被覆盖 | `main_BOS.py` 自行重写 `sys.argv` | 命令行参数不是真正可控配置 |
| CUDA 强制开启 | 固定参数含 `--fp16 --cuda_ray` | Apple MPS 不能原样进入 |
| device 不含 MPS | 入口只在 `cuda` 与 `cpu` 间选择 | 即使 MPS 可用也不会被选中 |
| renderer 无条件导入 | 顶层 `import raymarching` | non-CUDA 意图也先触发 CUDA 扩展 |
| CUDA JIT | backend 加载 `raymarching.cu` | 当前主机没有 NVCC/CUDA |
| hash device literal | `BOX_OFFSETS` 写死 `device='cuda'` | hash 不能原样迁移到 MPS |
| Windows 环境 | Conda 文件锁定 Windows/MSYS2/CUDA 11.8 | 不是跨平台 environment |
| MATLAB Mac 配置缺失 | compile 脚本引用未提供的 `mex_CUDA_maci64.xml` | MATLAB/CUDA 路线也未具备 Mac 合同 |
| 路径分隔符 | 64 个 manifest 路径原样均不存在 | loader 会静默跳过样本 |
| split 重叠 | val/test 均复用 train 前两个位姿，且 val/test 8/8 解码资产相同 | 不能充当 unseen-view 或独立 test |
| 无根许可证 | 没有项目级授权文本 | 不复制、不改作、不再分发 |

因此“删掉 `--cuda_ray`”或“把 encoding 改成 Fourier”都不够。可信复现需要把 loader、ray/AABB、renderer、device、依赖和许可证分别过门。

## 6. 这对我们的算法方向有什么新价值

2026 论文已经覆盖 single RI primitive、Fourier/hash、automatic/discrete/hybrid gradient、smoothstep、mask 和层级采样。这些单独拿出来都不是新意。可继续检验的是它暴露出的物理与数值矛盾：高频表示的导数对噪声敏感、hash 可能对实验噪声过拟合、温度反演会在边界附近饱和，而单一 held-out view 不能代替三维场真值。

### 候选 A：噪声与饱和感知的 Fourier-hash 残差混合

结构不是二选一换编码，而是让低频 Fourier base 承担稳定主体，让受限 hash residual 只解释被独立证据支持的薄前沿；离散梯度作为训练主驱动，AD 梯度只作一致性监测。gate 只读部署可见的噪声、模糊、视角几何与 residual 摘要，并在 support 外退回 Fourier base。

最小反例：若同一 phantom 上换噪声种子就崩，或收益只来自更多参数/更多 rays，这条路线立即关闭。

在训练混合模型以前，先做一个不冒充最终算法的 `GCS-Hash` 诊断：用固定审计 rays 上离散梯度与 AD 梯度的归一化失配、投影误差和饱和率共同决定是否开放更高 hash 层。如果这个 sentinel 不能在独立 corruption/field 上提前预测 hash 失败，就关闭“自适应解冻”分支，也不要把它包装成创新。

### 候选 B：饱和感知的 4D 时序修正

先逐帧得到稳定 base field，再只学习一个有界 temporal residual；在前沿出生、熄灭或拓扑改变时，时序门必须拒绝传播旧帧。它与 TDBOST 的关系是“强基线之上的失效保护”，不是重新声称 tensor decomposition 或 4D 网络首创。

最小反例：如果 temporal gain 在独立 sequence、不同采样率或 front birth/extinction 上消失，就不能称 4D 泛化。没有何远哲师兄的数据与时序合同前不启动此路线。

### 候选 C：support-gated proposal + exact reprojection correction

网络不直接宣判三维场，而是预测编码权重、warm start 或低秩修正；随后用真实 forward/JVP/VJP 做 1--2 次 exact correction。support、尾部或物理终点不过门时回到经典 CGLS/TV 或稳定 Fourier base。

最小反例：若同预算下不能超过 simple damping、固定插值和 classical warm start，就不升级成神经算子论文。

这三项目前都是 `UNVERIFIED_HYPOTHESIS`，不是算法成果。

## 7. 严格 benchmark 的最小形状

第一阶段必须先复现作者基线，再打开新方法：

1. CGLS/TV-CGLS；
2. Fourier + discrete gradient；
3. Fourier + hybrid gradient；
4. hash + discrete gradient；
5. hash + hybrid gradient；
6. 候选方法，参数量、ray samples 与 wall-clock 分账。

每组至少报告：折射率扰动 `delta n = n - n0` 的 synthetic field relative-L2、gradient/front error、真正新相机上的 displacement relative-L2、边界饱和比例、逐 view 尾部、训练/推理时间、峰值内存和 ray samples。直接对接近 1 的完整折射率算 relative-L2 会人为稀释误差。PSNR/SSIM 可以补充，但不能替代 field/gradient 与物理终点。

单 Phantom 1 只能做开发与实现校验。跨场泛化至少需要多个相互独立的 3D fields；跨装置泛化还需要不同 camera geometry、noise/blur 和 calibration session。真实火焰没有独立三维真值时，只能报告 held-out measurement 与明确的物理 proxy，不能把重投影自洽写成 field truth。

详细冻结合同见 [NIR-BOS 三维 benchmark 与算法候选合同](open_nir_bos_benchmark_contract_2026-07-22.md)。

## 8. 本机与服务器分工

当前 Mac 适合：

- clean-room loader 与路径测试；
- Fourier/小 MLP 的 MPS 单元测试；
- 低分辨率 analytic renderer；
- 指标、split、日志、可视化与消融脚手架；
- 小规模噪声/模糊反例筛选。

NVIDIA 服务器适合：

- 固定 commit 的作者 CUDA baseline；
- hash 与完整 ray marching；
- 140 x 294 x 140 全体积和 10k-iteration 训练；
- 多 seed、多 corruption、跨 phantom 的正式横向比较。

租卡前先要求服务器满足 Linux、NVIDIA CUDA、足够显存和可固定环境；先跑 50--100 steps 的数据/梯度/显存 smoke，再决定是否开启完整 10k steps。服务器不能弥补数据只有一个 phantom 的证据缺口。

## 9. 现在最值得问何远哲师兄的六个问题

1. OERF 当前希望复现的是 NeRIF、TDBOST，还是 2026 NIR-BOS primitive 中哪一个 baseline？
2. 可提供多少相互独立的 field、sequence、rig 与 session？同序列切帧不计独立样本。
3. 真实 callable 是 straight-ray 还是 curved-ray；是否有 forward、JVP/VJP 或 adjoint？
4. 最痛的是噪声、有限孔径、折射路径弯曲、标定漂移、温度饱和，还是计算成本？只能先选一个主问题。
5. 哪个物理终点可以独立验收：held-out camera、温度/密度标定、front 位置，还是 PIV velocity compensation？
6. 组内是否允许在 NVIDIA 机器运行作者仓库；作者代码与数据的许可是否能向作者书面确认？

## 10. 复现审计

外部 clone 保持在 Git 之外，命令只输出我们自己的元数据、哈希和图：

```bash
.venv/bin/python site_tools/audit_open_nir_bos_release.py \
  --repo <LOCAL_AUTHOR_CLONE>
```

公开结果：

- `learning_labs/results/open_nir_bos_release_audit_v1/report.json`
- `learning_labs/results/open_nir_bos_release_audit_v1/source_inventory.csv`
- `learning_labs/results/open_nir_bos_release_audit_v1/readiness_audit.png`
- `learning_labs/results/open_nir_bos_release_audit_v1/readiness_audit_mobile.png`
- `learning_labs/results/open_nir_bos_release_audit_v1/checksums.sha256`

**突破监测：没有突破。** 新增的是公开代码可复现门禁、一个真实 MPS 组件烟测和三个可证伪算法假设。作者 baseline、训练、三维重建、真实火焰、跨场泛化、算法优越与论文成功仍全部为 0。
