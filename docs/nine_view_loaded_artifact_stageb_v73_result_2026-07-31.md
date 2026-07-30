# v73：加载后的几何工件通过完整 1,515 单元兼容性门

## 一句话结论

把三档已知九视角几何的 q8 factor 离线编译为只读工件以后，在线加载路径在
五条已打开 PoolFire fit 轨迹、每条 101 帧上完成了
`3 × 5 × 101 = 1,515` 个重建单元；`1,515/1,515` 守住冻结的
field、gradient、interior-gradient 和 observation 门，并逐单元复现 v70
canonical 结果。独立程序的指标最大差为 `0`。

准确状态是：

```text
PASS_LOADED_ARTIFACT_FULL_TRAJECTORY_COMPATIBILITY_V73
PASS_INDEPENDENT_RECOMPUTATION_LOADED_ARTIFACT_STAGE_B_V73
formal_multi_trajectory_online_stage_c_authorized=true
algorithm_breakthrough=false
```

## 为什么必须做这一步

v70 已证明低内存 q8-K1 在完整 1,515 单元上保持精度，但 v70.1 的 fresh worker
每次重新构建 factor，导致 wall 通过而 RSS 高尾失败。v72 改为“离线编译一次、
在线只读加载”，并在一条 101 帧开发 stream 上第一次同时通过 wall 与
worker-self RSS。

不过，单 stream 的资源正结果不能替代完整精度门。真正的风险是：序列化、加载、
dtype、数组顺序或只读映射可能让工件路径与 canonical factor 出现微小漂移。
所以 v73 没有换算法，也没有调门槛，只回答一个必要问题：

> 加载后的工件能否在全部三档几何、五条轨迹和 101 帧上，逐单元保留 v70 的
> 精度与调用账？

## 正式实验

### 固定范围

- 数据：五条已经打开的 PoolFire CFD fit trajectory。
- 几何：三档已知九视角 straight-ray proxy。
- 网格：`32 × 16 × 16`。
- 帧数：每条 101 帧，共 1,515 个 trajectory-by-geometry-by-frame 单元。
- 候选：只读 loaded q8 factor、四步 cheap detector solve、精确
  `A^T` lift、未修改 CGLS K1。
- 对照：Zero-CGLS K4 与同调用的 Zero-CGLS K2。
- 候选在线精确账：每个单元 `2A + 2A^T`。
- Zero-K4 精确账：每个单元 `4A + 4A^T`。

离线构造不隐藏，但不计入每帧在线精确调用；下一阶段会把加载、校验、worker
启动与整条序列计入 fresh wall/RSS。

### 工件等价

三档几何各有一份工件，每份包含 18 个 view-component block 和 54 个数组：

```text
artifacts                              3
arrays per artifact                   54
total arrays                          162
maximum loaded/canonical array diff   0
forward relative difference           0
adjoint relative difference           0
```

### 完整轨迹判决

```text
evaluated cells                       1515
compatibility PASS                    1515 / 1515
loaded/canonical metric match         1515 / 1515
failed cells                          0
maximum metric absolute difference    0
maximum metric relative difference    0
```

相对 Zero-CGLS K4，四项 p50 / p90 / worst error ratio 为：

| 指标 | p50 | p90 | worst | 冻结门 |
|---|---:|---:|---:|---:|
| field | 0.98464 | 0.98980 | 1.00252 | <= 1.01 |
| gradient | 0.99496 | 0.99724 | 0.99949 | <= 1.01 |
| interior gradient | 0.97891 | 0.99020 | 1.00281 | <= 1.01 |
| observation | 0.89338 | 0.91081 | 0.93356 | <= 1.01 |

相对同为 `2A + 2A^T` 的 Zero-CGLS K2，四项 worst ratio 分别为
`0.91067 / 0.98631 / 0.95529 / 0.64662`，全部不高于 `1.00`。这说明当前
候选不是仅靠多花精确算子调用换来的改善。

## 独立复算做了什么

独立 validator 没有导入正式 runner，也没有导入正式 artifact loader。它：

1. 手工解析 NPY magic、version 和 header，并只读映射 payload；
2. 拒绝 object、structured、Fortran-order、尾随字节和文件集合漂移；
3. 重新构造全部 1,515 个 observation 与候选输出；
4. 独立重算 field、gradient、interior-gradient、observation 指标与调用账；
5. 用 dense p0 reference 检查三档几何的 forward 与 adjoint。

结果：

```text
independent metric maximum difference     0
dense p0 forward relative difference      1.23e-15
dense p0 adjoint relative difference      1.24e-15
formal payload unchanged                  true
bound source closure unchanged            true
```

因此，这不是只靠正式程序“自己证明自己”的 PASS。

## 成功了什么

v73 真实关闭了 loaded-artifact 路径的完整精度风险：

- 三档几何工件全部逐数组复现 canonical factor；
- 五条轨迹全部 505 帧、合计 1,515 单元通过；
- `2A + 2A^T` 在线精确调用账保持不变；
- 正式结果被另一套解析和重算路径逐项复现；
- 多轨迹 loaded-artifact fresh Stage C 现在有资格运行。

这是一个扎实的阶段性兼容性里程碑。

## 尚未成功什么

本轮没有测多轨迹 fresh wall，也没有测 whole-pipeline peak RSS。因此不能说：

- 算法已经实现正式端到端加速；
- 工件方案已经在多轨迹上通过资源门；
- 已经跨反应流数据族泛化；
- 已经处理 curved ray、相机标定、噪声或真实 BOST；
- 已经得到神经算子结果或论文成功。

当前仍严格保持：

```text
fresh_wall_speedup=false
whole_pipeline_rss_advantage=false
external_family_transfer=false
real_bost_result=false
neural_operator_result=false
algorithm_breakthrough=false
paper_success=false
```

## 下一条唯一有效门

下一步只运行多轨迹 loaded-artifact fresh Stage C：

- 候选仍为 loaded q8-K1；
- 对照仍为 Zero-CGLS K4；
- 加载、校验、worker 启动和完整 101 帧 stream 都计入在线 wall；
- 逐轨迹报告 paired wall p50/p90/worst；
- 同时测 worker-self、process-tree 与 whole-pipeline peak RSS；
- wall 与所有 RSS 门必须一起通过。

若资源门通过，才进入独立公开反应流族外门；若失败，就记录具体瓶颈，不再用
扩大模型或降低阈值挽救。
