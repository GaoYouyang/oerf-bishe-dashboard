# PoolFire C 路线 v9：Cross14 无数据核心代码门

> 当前状态：`PASS_NO_DATA_CROSS14_CORE_CODE_GATE_ONLY`
>
> 核心实现提交：`fc97cd7`
>
> 数据访问：本门没有读取任何 PoolFire trajectory、`p45-s03`、`p22-s01`
> 或 untouched test payload。
>
> 科学结论：`algorithm_breakthrough=false`。

## 1. 这一门证明了什么

v9 结果前协议已经固定主候选：

```text
q = G(A^T y)
e = G(W q)
x0 = G(e + sr * Cross14(q / sq, e / se))
x2 = CGLS_K2(x0, y)
```

这次提交实现的是公式中的低容量数值核心和 exact-K2 成本入口，不是跨轨迹数字
实验。已机械验证：

1. `Cross14` 固定为 raw/equalized BP 两通道、每通道
   `center, -x, +x, -y, +y, -z, +z` 七个位置，共 14 个 float64 权重；
2. 边界是 reflect-without-edge-repeat，并与独立
   `numpy.pad(mode="reflect")` 全数组一致；
3. 所有输入先做逐帧 gauge centering，模型无 bias；
4. `sq/se/sr` 使用完整训练轨迹等权 RMS，不能让帧数更多的轨迹占更大权重；
5. 只累计 `14×14` Gram、14 维 feature-target cross 和 target 二阶矩，不在内存中
   展开全部轨迹的巨大 feature tensor；
6. 岭回归由 float64 对称特征分解求解，正式模型只接受冻结的五个 lambda；
7. 模型采用 canonical JSON 和数值 digest，拒绝 duplicate key、NaN、额外字段、
   非规范编码和参数篡改；
8. 预测对正比例缩放保持齐次，强正则的零修正极限退回 equalized BP；
9. runner 不再接受任意 equalizer callable，并在入口重新核对
   `median/max-floor` 解析公式和 geometry-only 报告；
10. Cross14-K2 的核心调用账被断言为 `3A+3A^T`。

## 2. 为什么调用账是 3A+3A^T

| 阶段 | A | A^T |
|---|---:|---:|
| `q=A^T y` 与固定逐体素均衡 | 0 | 1 |
| 非零初值的初始投影 | 1 | 0 |
| 两步 CGLS refinement | 2 | 2 |
| **总计** | **3** | **3** |

这与 Zero-CGLS K3 的 6 次完整调用同成本，比 Zero-CGLS K4 的 8 次少 2 次。
但目前只证明代码中的调用账没有漏记，**没有证明相同终点精度、wall-time 加速或内存
下降**。

## 3. 本轮验证

```text
pytest -q learning_labs/test_poolfire_c_full_field_low_capacity.py
31 passed

pytest -q \
  learning_labs/test_poolfire_c_full_field_low_capacity.py \
  site_tools/test_poolfire_c_baselines.py \
  site_tools/test_validate_poolfire_c_full_field_low_capacity_protocol_v9.py
69 passed

python -m ruff check \
  learning_labs/poolfire_c_full_field_low_capacity.py \
  learning_labs/test_poolfire_c_full_field_low_capacity.py
All checks passed
```

同时通过 `py_compile` 和 `git diff --check`。这些测试覆盖公式、边界、轨迹等权、
退化 Gram、EVD、序列化、正比例齐次、equalizer 伪造拒绝和 exact-K2 成本。

独立代码审计没有发现 P0。审计允许的状态只有：

`PASS_NO_DATA_CROSS14_CORE_CODE_GATE_ONLY`

## 4. 这一门没有证明什么

以下状态仍为 false：

```text
equalizer_provenance_bound=false
process_truth_free_proven=false
independent_noninterference_proven=false
trajectory_split_proven=false
algorithm_breakthrough=false
```

具体含义：

- runner 能证明 equalizer 数组符合冻结解析公式，却还不能证明该数组确实来自与本次
  observation 相同的正式 geometry/operator；
- 核心函数签名没有 heldout truth 通道，不等于操作系统级进程已经无法读取 truth；
- 当前两个核心文件不负责证明 trajectory ID 唯一、101 帧完整、nested LOTO 或
  fit/validation/test 角色；
- 还没有独立 validator 重算正式模型、预测与评分；
- 还没有 fresh-process whole-pipeline wall/RSS；
- 没有运行五条 outer heldout、p14 veto 或 `p45-s03`；
- 没有训练 3D U-Net、FNO、UNO、DeepONet，也没有真实 BOST 结果。

## 5. 下一道门

下一步不是打开 holdout，而是实现并冻结三类角色：

1. **fit worker**：只读当前 fold 的训练 raw/equalized BP 与 K4 teacher，输出
   scaler、Gram、lambda 选择证据和 canonical model；
2. **deployment worker**：只读冻结模型、绑定的 geometry equalizer 和 heldout
   observation，原子发布 Cross14-K2 prediction 与完整调用/时间/RSS 账；
3. **score worker**：只在 prediction 已发布后读取 heldout teacher/proxy truth，
   计算 field、gradient、observation、harm 和 matched-accuracy 指标。

外层 manifest 必须绑定 protocol、实现提交、trajectory 角色、几何、equalizer、
solver、runtime、模型和报告模板。独立 validator 不能导入正式 fit/predict/score
helper。

只有五条完整 outer LOTO、p14 mandatory veto、同成本 Zero-K3、Zero-K4 终点、
wall 和 RSS 全部按冻结门通过，才允许另行生成一次性 `p45-s03` release。现在仍不获取该 payload。

## 6. 讲人话

现在已经把“14 个局部权重怎样计算、怎样存、怎样接到两步物理迭代、到底花几次
大算子”写实并锁住了。下一步才是在隔离的数据进程里回答真正的问题：

> 这 14 个权重能不能在没见过的完整轨迹上，用和 Zero-K3 相同的 6 次物理调用，
> 更接近 Zero-K4 的终点？

答不出来或任一轨迹受损，就按协议记为失败；不会用平均值或更换 holdout 掩盖。
