# 给何远哲师兄：N5-D5 第一次真实接口沟通单

> 目的：先确认真实物理接口与组内痛点，再决定算法。
> 第一次不讨论 Ed25519、nonce、沙箱或透明日志，也不要求师兄整理整套数据。

记录师兄口头回复时，打开 [真实接口回复工作台](../advisor_interface_intake.html)。工作台只使用当前浏览器的
`localStorage`，可以导出 `ADVISOR_REPLY_DRAFT` JSON，但不会上传、不会自动写 Git，也不会把回复直接当成
七门数据合同或训练授权。

## 一、第一次见面必须问的 9 件事

1. 目前组内真正使用的 NeRIF/BOST forward 是哪个仓库、版本和入口函数？
2. 第一版应接“直接优化三维场”，还是“decoder 参数 → 三维场 → detector”这条链？
3. 是否同时存在 curved 与 straight 两条真实路径？residual 是在同一 ray/sample/integrand 层原生形成，
   还是最后把两张 detector map 相减？
4. 后端能否接受运行时任意 `x/v/q`？能否通过 PyTorch/autograd 得到 JVP 与 VJP？如果目前只有 forward，
   也请明确。
5. forward 中是否有 hard mask、ray termination、occupancy pruning、自适应采样或其他会让导数跨分支失真的机制？
6. 可以提供的最小匿名例子是什么？优先 4–16 rays；如果程序不支持，就用最小合法 batch，加一个匿名
   `x0` 与简化 geometry。
7. 程序真实运行在哪里：Linux 服务器、容器还是本机？需要什么 GPU、PyTorch/CUDA、编译扩展或环境文件？
8. 师兄最希望我先解决哪一个真实问题：导数不稳定、curved/straight 相消、有限视角重建，还是同等精度下
   减少算子调用与重建时间？主指标和必须比较的基线是什么？
9. 哪些源码、权重、几何、标定、样本与日志绝不能离开实验室或公开？哪些匿名汇总图可以用于毕设和论文？

## 二、后续接线时再补的 3 件事

1. shape、spacing、axis order、units、dtype、wavelength、sampling/interpolation/boundary/termination 与标定版本。
2. 是否能记录 wall time、CPU/GPU 内存、`A/A^T` 调用数、active rays、sample evaluations、失败与重试。
3. 只有进入正式论文实验后，再讨论独立 observer、一次性授权账本和受保护 Linux 宿主。

## 三、希望得到的最小匿名 callable 包

```text
README / 入口函数位置
describe() -> shape、units、dtype、支持的路径与环境
forward(x, path) -> detector output
jvp(x, v, path)  -> 可选；没有就明确说明
vjp(x, q, path)  -> 可选；没有就明确说明
一个匿名 x0 + 最小合法 rays/geometry
environment.yml、requirements.txt 或 container 信息
```

关键点：需要的是 callable，不是提前算好的几个 `Jv/Jᵀq` 数组。若源码不能离开实验室，可以使用实验室
服务器上的远程 callable，或者只让师兄指出入口，由我编写 wrapper 后再交师兄审核。

## 四、可直接发送的微信消息

> 师兄好，我把毕设方向进一步收缩到“真实 BOST 三维前向与算子学习”。我想先接一个很小的真实代码接口，
> 确认物理路径和导数是否可靠，再决定设计哪种新模型，避免只在 toy 数据上得到好看的结果。
>
> 这次不需要您整理完整实验数据或打包整个项目，先口头帮我确认几件事就可以：目前真实 forward 的代码
> 入口在哪里；输入更适合用三维场还是 decoder 参数；是否有 curved/straight 两条路径，residual 是在
> ray/sample 层形成还是最后两张图相减；现有代码能否用 autograd 做 JVP/VJP；是否有 termination、hard mask
> 或自适应采样等分支；程序实际运行环境是什么；以及您最希望我优先解决导数稳定、残差相消、有限视角重建，
> 还是重建加速。
>
> 如果方便，希望后续给我一个最小匿名例子，例如最小合法 rays、一个匿名输入和简化几何。您只要指明函数
> 入口也可以，wrapper 我来写，再交给您审核。源码、参数和 raw trace 默认只留私有本地，不上传 Git；每一步
> 运行前我都会单独确认。还想请您告诉我哪些信息绝不能离开组内，以及最终更重视重建精度、跨工况稳定性，
> 还是同等精度下更少的算子调用和运行时间。

## 五、收到回复后的分支

| 师兄回答 | 第一条真实实验 |
| --- | --- |
| 有 forward + JVP/VJP，且 residual 在 ray/sample 层原生形成 | 做 residual-native 导数稳定性、相消误差和局部 Schur/adjoint 检查 |
| 有 curved/straight，但没有 native residual | 使用诚实双路径协议，不在 wrapper 末端相减冒充第三路 |
| 只有 forward | 先做 finite-difference/complex-step 可行性与 branch map，暂不开神经算子训练 |
| 主要痛点是有限视角 inverse | 冻结 SIRT/TV/PCGLS/NeRIF 基线，再设计 operator warm-start 或 residual correction |
| 主要痛点是时间/调用成本 | 按 `A/A^T`、ray/sample、wall、显存和失败率做同精度 Pareto，不只报网络推理时间 |
| 有连续高速序列 | 才考虑 4D 低秩/时序算子；先按 rig/session 留出，避免逐帧随机泄漏 |

停止边界：接口语义不清、输出含隐私、导数跨离散分支、结果不可重复或成本失控时先停下问师兄，不自动换
seed 重跑，也不把局部 synthetic 改善写成真实或泛化成功。
