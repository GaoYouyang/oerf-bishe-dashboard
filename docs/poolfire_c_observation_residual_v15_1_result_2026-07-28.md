# PoolFire C v15.1：没有发现可跨轨迹复用的单一观测残差模式

## 一句话结论

我把 v14 中 `w8d2` 相对 `w16d2` 的最终 observation residual 拆成三视角、
六分量和低/中/高频共 18 个模式，在五条 fit trajectory、505 帧上逐项比较。
结果没有任何模式同时满足：

1. 至少三条轨迹在同一个模式上既一致变差、又进入正超额 top-3；
2. 没有其他轨迹发生强反转；
3. 同一个模式还必须解释 P45 的 11 个冻结失败帧，并达到两项 1.25 倍富集门。

独立复算的权威状态是：

```text
VALIDATED_NEGATIVE_NO_SHARED_MODE_V15_1
shared_mode_ids=[]
selected_mode_id=null
single_correction_preregistration_authorized=false
algorithm_breakthrough=false
paper_success=false
```

因此当前证据不支持围绕某一个视角/分量/频带训练“通用残差修补器”。`w16d2`
仍保留为最小的五轨迹 5/5 开发候选；proxy 内继续做架构救援到此停止。

## 为什么做这次诊断

v14 已经证明 `w8d2` 的 P45 失败不是三维场整体崩溃：

```text
field failure frames       = 0
gradient failure frames    = 0
observation failure frames = 11
harm frames                = 0
severe frames              = 0
```

一个合理但必须验证的猜想是：这 11 帧是否都由某个跨工况稳定的 detector residual
模式造成。如果答案是肯定的，才有理由在不扩大网络的前提下，预注册一个极小的
observation-consistency correction 或 abstention gate。

这轮没有直接训练修正模型，而是先问更基础的问题：

> P45 失败所对应的残差模式，在另外四条轨迹上是否也以相同方向、相近重要性出现？

## 结果前冻结了什么

每个最终候选场先重新投影到 observation space，残差定义为：

```text
r = A x_K1 - y
```

三个视角分别为 `2×14×30`、`2×14×30` 和 `2×14×14`。每个分量做完整二维
正交 FFT，再按归一化半径固定分成：

```text
low  = [0, 1/3)
mid  = [1/3, 2/3)
high = [2/3, +inf)
```

每帧每个模式的能量除以整帧 observation 能量。因此 18 个模式之和严格等于
observation relative-L2 的平方；正式结果最大 Parseval 误差为：

```text
8.326672684688674e-17
```

共享模式必须满足：

- 同一条轨迹上，median `w8-w16` 能量为正；
- 超过 60% 帧为正；
- 同一个模式在该轨迹严格进入正超额 top-3；
- 至少三条轨迹同时满足以上三项；
- 不能有轨迹出现 median 为负且正帧比例不超过 40% 的强反转；
- 在 P45 的 11 个失败帧中也必须进入 top-3；
- P45 失败帧的 `w8/w16` 模式能量比至少 1.25；
- P45 失败帧相对其余 90 帧的 `w8` 模式能量比至少 1.25。

零正超额模式不能因为并列而获得 top-3 票。若多个模式通过，选择顺序也已提前冻结：
支持轨迹数降序、跨轨迹 median 正超额份额降序、mode ID 字典序。

## 两种现象被明确分开

| 模式 | 一致轨迹 | 同轨迹 support | 强反转 | P45 `w8/w16` | P45 fail/matched | P45 top-3 | 结论 |
|---|---:|---:|---:|---:|---:|---|---|
| `view_1_component_1_mid` | 5 | 3 | 0 | 1.019 | 0.940 | 否 | 跨轨迹缺口，但不解释 P45 |
| `view_1_component_0_low` | 4 | 1 | 1 | 1.322 | 1.526 | 是 | P45 特异，但不跨轨迹 |
| `view_2_component_0_low` | 3 | 3 | 2 | 1.275 | 1.217 | 是 | 两条轨迹反转，富集不足 |
| `view_2_component_1_low` | 4 | 3 | 1 | 1.186 | 0.959 | 是 | P58 反转，P45 不富集 |

### 跨轨迹存在的容量缺口，不是 P45 的失败机制

`view_1_component_1_mid` 在五条轨迹上的 median 超额都为正，正帧比例为
`75.25%-100%`，并在 P14、P22、P58 进入 top-3。它确实说明 `w8d2` 在第二视角、
第二分量的中频表示上普遍弱于 `w16d2`。

但它不解释 P45 的 11 个失败帧：

```text
P45 failure w8/w16 ratio       = 1.019
P45 failure/matched w8 ratio   = 0.940
P45 failure top-3              = false
```

失败帧上的这个模式反而没有比其余 90 帧更强。

### 真正富集于 P45 失败的模式，不具有跨轨迹稳定性

`view_1_component_0_low` 在 P45 失败帧上同时满足两项富集：

```text
P45 failure w8/w16 ratio       = 1.322
P45 failure/matched w8 ratio   = 1.526
P45 failure top-3              = true
```

但它只在 P45 这一条轨迹同时进入 top-3 与一致变差；P22 上 median 超额为负、
正帧比例只有 `30.69%`，属于强反转。因此它更像 P45 特有的工况响应，不是可以
直接推广到五条轨迹的通用修正方向。

第三视角的两个低频模式也有类似问题：它们在 P45 上较强，但分别在 P22/P58
发生反转，或者没有达到冻结的失败富集门。

## 真正运行了什么

- 只使用已经开放的五条 fit trajectory；
- 每条 101 帧，共 505 帧；
- 分别重新加载五个 `w8d2` LOTO checkpoint 和五个 `w16d2` LOTO checkpoint；
- 对两个模型重新执行 observation-only proposal、exact `A^T`、observable alpha
  和 unchanged strict CGLS K1；
- 每个模型每帧的原 shell 仍为 `2A + 2A^T`；
- 诊断为了得到最终 residual，额外离线执行一次 `A x_K1`，明确不计作部署优势；
- 没有请求 raw pair truth，没有打开 fresh、historical validation 或 untouched test。

需要如实限定：程序读取了旧报告中的 truth-derived compatibility 汇总来绑定历史
模型身份，因此只能声明“没有请求 raw pair truth”，不能声明
filesystem-wide truth nonaccess 已被证明。

## 独立复算与红队

独立 validator 不导入正式 runner 或正式残差分析模块，而是从 checkpoint 和
observation 重新生成两套 505 帧候选、最终投影、18 模式 FFT、逐轨迹摘要和最终
gate：

```text
maximum array difference          = 0
maximum nested summary difference = 0
maximum Parseval error             = 8.326672684688674e-17
formal evidence unchanged          = true
```

第一次预运行后，红队发现四个 P1：零能量并列会获得 top-3 假票、一致轨迹与 top-3
轨迹不要求重合、负科学结果仍带 `PASS` 前缀、源码/历史模型身份闭包不够强。那次目录
已明确标记无效，没有用于本结论。

v15.1 在重新运行前完成了以下修复：

- top-3 只从严格正超额模式中选择；
- support 必须在同一条轨迹同时满足一致性和 top-3；
- 验证完整性与科学正负分开表示；
- runner/validator 使用同一精确 20 文件源码闭包；
- 旧报告、READY、v14 权威 gate、两个独立验证 seal、checkpoint 与 observation
  输入全部绑定；
- 旧模型还必须复现公开 observation-tail 数值锚点；
- 多模式选择规则提前固定；
- raw-truth 声明降级为真实可证明范围。

复审结果为 `P0=0 / P1=0 / P2=1`。剩余 P2 是未来回归测试还没有自动覆盖所有
seal/anchor 篡改组合，不影响当前冻结判决；当前定向测试为 `8 passed`。

## 成功、失败与突破边界

已成功：

- 把 v14 的 observation-only 失败拆成了可复算的视角/分量/频带证据；
- 区分了“跨轨迹容量缺口”和“P45 特异失败模式”；
- 排除了一个最诱人的错误方向：用单一频带修补器解决所有轨迹；
- 全部数组与判决得到独立零差复算；
- 红队发现的全部 P1 在权威运行前关闭。

未成功：

- 没有发现共享 residual mode；
- 没有授权 correction、abstention 或新网络训练；
- 没有得到 fresh 泛化、真实 BOST、真实相机/噪声或真实实验同精度证据；
- 没有算法突破，也没有完成论文。

## 科学结论

当前 proxy 上，`w8d2` 的 P45 observation failure 不是一个跨工况稳定的单频带
缺陷。确实存在较普遍的中频容量缺口，但它不富集于 P45 失败；真正富集于 P45 的
低频模式又在其他轨迹上反转。

因此，继续在这五条 proxy fit 轨迹上设计单模式修正、换宽度或追加 seed，很可能只会
把 P45 的特例写进模型，而不能提高跨轨迹可靠性。当前最诚实且最有价值的停止结论是：

```text
retain w16d2 as the smallest current 5/5 development candidate
stop proxy-only architecture rescue
require independent-trajectory or real-BOST migration evidence next
algorithm_breakthrough=false
```
