# N2-PVGR N5-D3 32 格自适应残差参考包结果审计

日期：2026-07-18

## 先说结论

D3 已经把 N4.1、D1 和 D2 的冻结数组组装成一个 32 格参考包，独立
validator 通过。机器判决是：

`D3_VALID_MIXED_RESIDUAL_REFERENCE_ONLY`

这句话的准确含义是：已有一个字节级可追溯的 synthetic residual reference pack，
但它是混合方法参考，不是 fresh set、统一 paired reference、三维重建成果或神经
算子结果。

## 包里到底有什么

| 来源 | 格数 | 步数 | residual 路由 |
|---|---:|---:|---|
| N4.1 已授权参考 | 23 | H1024 | curved/straight 分别累加后相减 |
| N4.1 已授权参考 | 7 | H2048 | curved/straight 分别累加后相减 |
| D2 尾部补齐 | 2 | H8192 | paired residual + Neumaier 累加 |
| **合计** | **32** | 自适应 | **mixed** |

每格是 `256 x 2` 的 float64 数组，整包 shape 为 `32 x 256 x 2`。所有格的 finite
fraction 均为 1.0。

## 为什么不能叫“统一 paired 参考”

D1 仅在 p04/p05 的四格上直接核对过 raw 与 paired 路由，覆盖率是 `4/32`。
剩余 28 格没有完整 curved/straight component array，不能事后凭语言将它们
重写为 paired 结果。因此：

- `mixed_reference_pack = true`；
- `uniform_paired_reference = false`；
- paired equivalence coverage 固定披露为 `4/32`。

这是一个方法覆盖缺口，不影响 D3 作为后续小规模导数实验的数值基础，
但会限制它能支持的论文措辞。

## 可复核证据

| 项目 | 值 |
|---|---|
| cell count / unique count | `32 / 32` |
| allocation | `23 / 7 / 2` |
| stacked array SHA-256 | `8d2bba156028e4b14385f5a563d4d7c18817bb17a70dc0856bfeb240e8e765ed` |
| cell-order SHA-256 | `80640caed7bad515bc7449993ea0e18c7556be03b83bf26e28d2f83ee7ffa79e` |
| N4 checkpoint Merkle root | `407ec681e0f8787c286531db222cc6cbeb8009b83717b35bb2aca83498cb242a` |
| source logical field-point queries | `583,532,544` |
| D3 assembly field queries | `0` |
| protocol commit | `bdf9afd466e004354732e2caf70898a578cc6986` |
| independent validator | `valid = true` |

`583,532,544` 是被选中参考数组的原始计算成本账本，不是 D3 再次计算了
5.8 亿次。D3 只读取已冻结数组、核对身份并组装，新增 field query 是 0。

## 独立 validator 重算了什么

validator 没有导入 D3 runner，而是独立完成下列检查：

1. 重算父级 N4.1、D1、D2 manifest 和 validation 链；
2. 重建 N4.1 的 105 个 checkpoint inventory 与 Merkle root；
3. 按 N4.1 原 cell order 核对 32 格身份、唯一性和 `23/7/2` 映射；
4. 从源文件解码数组，重算 source hash、float64 little-endian C-order hash 和整包 hash；
5. 重算查询成本账本；
6. 验证 mixed semantics 和所有 claim authorization 仍为 false；
7. 检查图像为 `2860 x 854`，pixel standard deviation 为 `51.53`，不是空白图。

## 这一步允许和不允许说什么

### 允许

- 32 格 selected synthetic residual 已有混合多保真度参考包；
- 数组身份、顺序、步数、成本和字节可追溯；
- 可以为下一个结果前预注册的 tiny-field JVP/VJP 实验设计提供参考身份。

### 不允许

- 不允许写成 fresh/independent 32 格成功；
- 不允许写成统一 paired-Neumaier reference；
- 不授权 field JVP/VJP 正确、三维重建成功或神经算子训练；
- 不授权优于 DeepONet/FNO/FFNO、真实流场、泛化或论文结论。

## 下一步：D4 导数门，而不是训练网络

D4 需要另行预注册，并把两个导数对象分开：

1. detector output 的 `J_y`；
2. curved-straight residual 的 `J_r = J_curved - J_straight`。

最小实验先用 CPU float64、小网格、固定 rig 和固定 ray topology，同时通过：

- JVP/VJP dot product test；
- 多个 `h` 的中心有限差分曲线，不接受只有一个偶然的好 `h`；
- `VJP_residual ≈ VJP_curved - VJP_straight` 结构恒等式；
- domain/stencil/topology/finiteness fail-closed 门；
- forward/JVP/VJP 调用与峰值内存账本。

只有 D4 通过，才进入 6-train-view / 2-held-out-view 的最小三维重建。

## 复核入口

- 预注册：`docs/n2_pvgr_n5_d3_adaptive_reference_preregistration_2026-07-18.md`
- 配置：`demo_t16_operator/configs/n2_pvgr_n5_d3_adaptive_reference_preregistered_v1.json`
- 证明：`demo_t16_operator/configs/n2_pvgr_n5_d3_adaptive_reference_attestation_v1.json`
- 参考包：`demo_t16_operator/results/n2_pvgr_n5_d3_adaptive_reference_v1/reference_pack.json`
- 逐格账本：`demo_t16_operator/results/n2_pvgr_n5_d3_adaptive_reference_v1/cell_ledger.csv`
- 独立校验：`demo_t16_operator/results/n2_pvgr_n5_d3_adaptive_reference_v1/validation_report.json`

复跑校验：

```bash
/opt/anaconda3/envs/deeponet_env/bin/python \
  site_tools/validate_n2_pvgr_n5_d3_adaptive_reference.py

/opt/anaconda3/envs/deeponet_env/bin/python -m pytest -q \
  demo_t16_operator/test_run_n2_pvgr_n5_d3_adaptive_reference.py
```
