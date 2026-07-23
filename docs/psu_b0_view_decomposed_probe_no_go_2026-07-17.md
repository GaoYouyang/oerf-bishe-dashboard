# PSU B0 VD0-A 逐视角伴随冲突探针判决

日期：2026-07-17

状态：`INTERFACE_PASS_REPRESENTATION_NO_GO_POSTOPEN`

## 一句话结论

逐视角伴随分解已经精确实现，但只用 18 个范数、夹角和抵消统计量，不能比
原来的 pooled 首伴随特征更安全地选择 PCGLS 频谱专家。因此现在不训练
DeepSets；只允许再补一轮尚未进入 VD0-A 的图像平面 front proxy 和相机姿态。

## 这一步到底在验证什么

原来的 RQ 路由只看所有相机求和后的

\[
g_0=A^\top W y.
\]

它在 field-L2 上有 `+3.321% / +2.907%` 的迁移信号，却会让相关噪声斜激波
的 front-F1 最坏下降 `30.876%`。VD0-A 因此先问一个最小问题：

> 如果保留每台相机自己的
> \(g_{0,v}=A_v^\top W_v y_v\)，相机间范数份额、夹角和抵消是否足以识别
> 哪个固定 SPD 专家安全？

这一步没有训练大网络，也没有使用实验三维真值。

## 接口门：通过

新接口在一次射线散射遍历中，把每条射线贡献写入对应相机槽，再分别执行
有限差分伴随。九个视角的输出求和必须回到 pooled 伴随。

- 最大 grouped-sum relative error：`1.7769e-7`
- 每个 batch 的 ray scatter traversal：`1`
- grouped 调用记录：一次
- grouped 与 pooled 等 FLOP：`false`
- 原因：保留九份体场会增加输出内存，并需要逐视角有限差分伴随后处理
- 定向实现与特征测试：`19 passed`

这里的“一次调用”不能被写成“和 pooled 伴随完全同成本”。公平口径是：
没有把九组射线重复散射九遍，但保留分组输出确实有额外内存与后处理成本。

## 特征和对照

同一有限动作库、同一 train-only OOF、同一四步 `4F+4A^T` 重建合同下比较：

1. `pooled_initial_normal`：原 44 维 pooled 首伴随频谱/空间统计；
2. `view_conflict`：18 维逐视角范数份额、两两 cosine、负相关比例、求和抵消；
3. `pooled_plus_view_conflict`：两者拼接。

训练集的 16 个有限动作结果直接复用旧 RQ 私有缓存，没有重新运行训练集
重建。整轮只新增逐视角特征和转移路线：

- wall time：`5.973 s`
- peak RSS：`435.9 MB`
- 新增 train action reconstruction：`0`
- grouped feature batches：`9`
- transfer routes：`6`

## Train OOF 结果

### Pooled 基线

- mean field gain：`+2.357%`
- coverage：`43.75%`
- overall harm：`2.08%`
- accepted harm：`4.76%`

### 逐视角冲突单独使用

没有候选通过严格 OOF 门，严格路线精确回退 baseline。放宽诊断路线虽然平均
`+1.235%`，但 p10 为 `-1.400%`，harm 为 `14.58%`。

### Pooled + 逐视角冲突

- mean field gain：`+1.519%`
- coverage：`56.25%`
- OOF harm：`0`
- p10：`0`

它在训练 OOF 上更谨慎，但已经把主要效用从 `2.357%` 压到 `1.519%`。

## Post-open transfer 结果

| 严格路线 | Validation field | Calibration field | Calibration harm | Calibration front mean |
|---|---:|---:|---:|---:|
| pooled | `+3.321%` | `+2.907%` | `3.33%` | `-0.261%` |
| view conflict | baseline fallback | baseline fallback | `0` | `0` |
| pooled + view | `+2.258%` | `+1.604%` | `6.67%` | `-1.376%` |

拼接路线没有在两个 transfer split 同时击败 pooled，也没有通过 development
gate。它在 validation 把 front p10 从 `-3.219%` 提到 `0`，但这个局部信号
没有迁移到 calibration；calibration front p10 为 `-5.006%`，最坏
`-12.100%`。

## 留一组压力审计

严格 pooled 路线：

- leave-one-family mean：`-0.999%`
- leave-one-noise-profile mean：`-2.217%`

严格 pooled + view 路线：

- leave-one-family mean：`+0.199%`
- leave-one-noise-profile mean：`-0.347%`

这说明逐视角冲突可能缓和一部分形态迁移，但还不能跨噪声机制安全工作。
这个 `+0.199%` 是下一特征设计的弱机制信号，不是方法通过。

## 为什么现在不训练 DeepSets

[Deep Sets](https://papers.nips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html)
给出了置换不变集合函数的结构依据；
[Set Transformer](https://proceedings.mlr.press/v97/lee19d.html)
允许显式建模集合元素之间的交互；
[Learned Primal-Dual](https://arxiv.org/abs/1707.06474)
说明测量域状态与物理 forward/backprojection 可以共同进入学习型迭代；
[Deep Back Projection](https://arxiv.org/abs/1807.02370)
直接展示了“每个视角先单独反投影、再由网络融合”的断层重建思路。

这些论文支持结构合理性，却没有证明 BOST 上的 48 个训练样本足以训练注意力
模型。当前显式特征没有通过迁移门，直接上 DeepSets 很可能只增加容量和
post-open 过拟合，不能增加证据等级。

## 下一步只允许做什么

VD0-B 只补两个当前确实缺失的物理输入：

1. **图像平面 front proxy：**逐视角白化位移图的高频份额、ridge/梯度能量
   集中度、方向各向异性和视角间谱差；
2. **相机姿态：**把 camera pose/projection basis 与对应视角特征成对输入，
   避免完全置换汇总后丢失“哪台相机从哪个方向看”的信息。

冻结停止规则：

- 若 VD0-B 在 leave-one-family 与 leave-one-noise 上不能同时减少 harm，
  停止集合编码主线；
- 若 validation/calibration 的 field mean 任一低于 `2%`，不生成 fresh；
- 若 calibration front mean 为负、front p10 越界或 field harm 超过 `5%`，
  不生成 fresh；
- 只有显式 VD0-B 过门，才训练小型 DeepSets；不直接使用 Set Transformer。

## 证据边界

- 真实 PSU support geometry：是
- 解析反应场形态：是
- 合成 camera noise：是
- 真实 PSU displacement：否
- OERF 实验数据：否
- 实验三维 truth：否
- validation/calibration：已打开，只作 post-open 机制诊断
- independent fresh：未授权
- 优于 DeepONet/FNO/NeRIF/TDBOST：没有证据

## 复现

```bash
.venv/bin/python -m site_tools.run_psu_b0_view_decomposed_probe \
  --development-report private_library/external_datasets/psu_bost_flight_body/b0_residual_risk_development_v1/private_report.json \
  --pooled-probe-private-report private_library/external_datasets/psu_bost_flight_body/b0_observable_morphology_probe_v1/private_report.json \
  --rq-private-report private_library/external_datasets/psu_bost_flight_body/b0_rq_ogse_pcgls_development_v1/private_report.json \
  --view-root private_library/external_datasets/psu_bost_flight_body/all_view_geometry_audit \
  --device mps \
  --private-output private_library/external_datasets/psu_bost_flight_body/b0_view_decomposed_probe_v1/private_report.json \
  --public-output docs/psu_b0_view_decomposed_probe_public_summary.json
```

公开摘要：
`docs/psu_b0_view_decomposed_probe_public_summary.json`

图：
`demo_t16_operator/results/psu_b0_view_decomposed_probe/psu_b0_view_decomposed_probe_figure.png`
