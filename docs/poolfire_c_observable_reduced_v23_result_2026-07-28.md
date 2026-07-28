# Observable Reduced Residual Warm K1：真实结果与确认性失败

更新时间：2026-07-28

## 一句话结论

固定方法在两条已经打开的公开 PoolFire CFD 轨迹上都显示出稳定的资源收益：
相对完整父模型，fresh-process 中位 wall time 分别降低 **15.26%** 和
**10.80%**；最新 `p=58kw_size=01` 的 field / gradient / observation
兼容门全部通过，101 帧 joint match 为 **100%**、harm 为 **0**。独立数值
路径将候选场复算到最大绝对差 `1.63e-15`。

但是，两条官方 test 的首次执行都因流程缺陷失效，所以当前有效 untouched
confirmatory 轨迹数仍为 **0**。这是一条有价值的机制信号，不是确认性算法突破，
更不是论文成功。

## 做了什么

方法固定为：

1. 五条 fit trajectory 上训练好的 compact dual CNN 只处理偶数帧，共
   `51/101` 帧。
2. 对相邻奇数帧，不再调用 CNN，也不直接复制上一帧，而是在仅由 fit 数据建立的
   rank-199 场/观测子空间中求解可见 observation residual。
3. 偶数帧和奇数帧都经过相同的解析 observable alpha 与未修改 strict CGLS K1。
4. 比较对象是每帧都运行 CNN 的完整父模型，以及 Zero-CGLS K4。

在线完整算子账为：

| 方法 | A | A^T | A+A^T |
|---|---:|---:|---:|
| Observable Reduced Warm K1 | 202 | 152 | 354 |
| Full-parent Warm K1 | 202 | 202 | 404 |
| Zero-CGLS K4 | 404 | 404 | 808 |

因此，主方法相对父模型减少 `12.38%` 的完整算子调用，相对 Zero-K4 减少
`56.19%`。rank-199 package 的 `200A+200A^T` 是一次性离线 setup，未藏入
在线账。

## 为什么这样做

v18-v21 已经否定了直接复用、有限平移缩放和跨工况共享 lift span 等更简单的
50% 跳帧方案。它们的问题不是一定“崩掉”，而是在 p45/p58 上无法逐帧守住
field、gradient、observation 三个非劣门。

这一版把需要预测的量改成 **观测残差中可解释的低秩修正**：

- 输入只依赖部署时可见的 observation；
- 修正仍被限制在精确 `Range(A^T)` 生成的场子空间；
- 奇数帧省掉一次 CNN 和一次 `A^T`；
- 最后仍使用同一个 CGLS K1，不靠给候选多迭代获得优势。

## p58 支持性结果

### 精度

全部 101 帧：

| 指标 p90（越低越好） | Reduced Warm K1 | Zero-K4 | Reduced / Zero |
|---|---:|---:|---:|
| field relative-L2 | 0.708815 | 0.705558 | 1.0046 |
| gradient relative-L2 | 1.415498 | 1.423306 | 0.9945 |
| observation relative-L2 | 0.365610 | 0.366029 | 0.9989 |

主方法在 field p90 上比 Zero-K4 高约 `0.46%`，在 gradient 与 observation
p90 上分别低约 `0.55%` 和 `0.11%`。按结果前冻结的逐帧单侧兼容门：

- all-frame joint match：`101/101`
- skipped-frame joint match：`50/50`
- joint harm：`0`
- severe harm：`0`

这表示“落在冻结的同精度兼容包络内”，不表示三种误差都严格优于 Zero-K4，
也不等于真实实验中的物理同精度。

### 时间和内存

每个 arm 运行 101 个全新子进程，顺序交替：

| 指标 | Reduced Warm K1 | Full parent | 判决 |
|---|---:|---:|---|
| external wall median | 0.226388 s | 0.253793 s | 降低 10.80% |
| child peak RSS p90 | 120.09 MB | 117.85 MB | ratio 1.0190 |

wall 通过“至少快 10%”的门，RSS 通过“不超过 1.05 倍”的 no-harm 门。
这里的 RSS 是 `wait4` 记录的直接子进程峰值；worker 没有再创建子进程，但这仍
不能写成完整实验系统的 process-tree / whole-pipeline 内存。

### 独立复算

第二条数值路径没有读取正式 package 的 SVD 分解结果，而是从五条 fit
trajectory 重新执行：

1. sample-space 对称特征分解；
2. field basis QR；
3. measurement-space 正规方程 ridge；
4. 逐帧独立 K1；
5. 指标和 benchmark 统计重算。

结果：

```text
candidate max absolute difference = 1.6306400674181987e-15
metric max absolute difference = 4.440892098500626e-16
benchmark statistic max difference = 0.0
pair binding unchanged = true
supporting run unchanged = true
```

## 为什么确认性试验仍然失败

### 第一个 test：p22

旧 test-bundle reader 在最初 pair 生成路径中拒绝 `split=test`。失败发生在
正式预测与正式 truth score 之前，因此没有确认性科学结果。之后对已打开数据做的
accuracy / wall / RSS 只能算 post-open diagnostic。

### 第二个 test：p58

正式 runner 把 `observations[::2]` 直接传给要求 C-contiguous 的 native
wrapper。NumPy 步进切片是非连续视图，因此第一次 proposal 调用就失败：

```text
FusedStreamingError: fused streaming proposal input changed
```

失败时：

- 已生成确认样本对；
- 预测文件数量为 0；
- 没有读取 truth 评分；
- 没有性能结论。

即使修复只需固定增加一次 `np.ascontiguousarray`，也不能修完后把同一条已打开
轨迹重新称作 untouched test。修复后的 p58 结果因此被永久标记为
`post-open supporting diagnostic only`。

独立红队随后还指出旧 release root、开封前完整 preflight、pair checksum
消费链和 benchmark 顺序存在缺口。这些缺口没有证明数值结果错误，但足以否定
“确认性执行有效”这一更强主张。

## 成功、失败与突破判断

**成功的部分：**

- 新机制在 p22 与 p58 两条 post-open proxy 上都获得超过 10% 的 wall 降幅；
- p58 的冻结三指标兼容门、harm 门、A/A^T 账和 RSS 门同时通过；
- 独立数值路径在浮点误差量级复现候选、指标和 benchmark；
- 算法确实省去了 50 次 CNN 和 50 次 `A^T`，不是只在报告里删调用。

**失败的部分：**

- 两次官方 test 首次执行都被工程/协议错误判为 INVALID；
- 有效 untouched confirmatory 轨迹数是 0；
- 尚未接入真实 BOST 相机、标定、噪声、重复测量和组内 forward；
- 当前 proxy 的绝对 field / gradient 误差仍高，不能把“兼容 Zero-K4”写成高质量
  三维物理重建。

**突破性进展：否。**

更准确的表述是：已经找到一个跨两个公开 CFD 工况重复出现的、可独立复算的
**候选加速机制**，但确认性验证流程失败，真实 BOST 外部有效性也未建立。

## 当前科学边界

```text
algorithm_breakthrough=false
confirmatory_valid=false
valid_untouched_confirmatory_trajectory_count=0
real_BOST=false
physical_same_accuracy_proven=false
global_generalization_proven=false
paper_success=false
```

脱敏机器可读结果见
`docs/poolfire_c_observable_reduced_v23_public_summary.json`。
