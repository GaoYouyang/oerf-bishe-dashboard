# PSU-B0 Gate A mechanics 正式证明记录

日期：2026-07-17
状态：`FORMAL_GATE_A_ATTESTED_MECHANICS_ONLY / GATE_B_NOT_RUN / NO_FRESH_REAL_OR_WIN_CLAIM`

## 一句话结论

在 source commit `53f1bccb287744a6ab97d7d6c2a86d556515e34a` 上，一个
`view-local + single frozen scale + synthetic truth-free fixture` 的 signed-factor
majorizer mechanics 已通过正式生成、独立 NumPy/MPS 重算和第二次只读复核。
这只证明代码合同、伴随、majorization、步长、六步 recurrence、zero ledger 和调用
账本在该小实例上成立，不证明重建效果、一般尺度、真实数据或方法优势。

## 冻结身份

| 项目 | 冻结值 |
|---|---|
| source commit | `53f1bccb287744a6ab97d7d6c2a86d556515e34a` |
| attestation identity | `b72bebf5aeb492f51bfd7b57e2e09097bbce905b2dbb1b6b72bf3f98d7ed0575` |
| attestation file SHA-256 | `7e6aa629ddcc9a6926caecb30b79bc0d0f5b131ef3d2f132ff4df6482df27344` |
| E1 records | `13 / 13 PASS` |
| declared selectors | `20` |
| expanded cases | `34 passed, 0 skipped`；独立 replay 同样 `34 passed` |
| independent validator | `333` core checks |
| evidence scope | `VIEW_LOCAL_SINGLE_FROZEN_SCALE_MECHANICS_ONLY` |

## 独立数值结果

| 检查 | 结果 | 门限或解释 |
|---|---:|---|
| `max(M-|A|, N-|D|)` violation | `0.0` | 逐元素支配成立 |
| scaled operator norm squared | `0.47786387837228383` | `eta^2 = 0.49` |
| NumPy 六步最大状态相对误差 | `4.1298927220699203e-16` | 门限 `1e-10` |
| MPS 最终 field 相对差 | `5.751670809943163e-08` | 门限 `5e-4` |
| MPS 六步最大状态相对差 | `1.0374490912413573e-07` | 诊断值，低于 field 门数量级 |
| exact-zero 删除 data rows | `3` | 删除值与目标常数独立重算 |

独立 oracle 只从 JSON 原语重建 `E/G/P/W/D+` 和 dense `A/M/D/N`，不导入
production reconstruction、factor、fixture 或 solver 模块。validator 不相信报告里的
`PASS` 字段，而是重跑测试、NumPy recurrence、CPU/MPS parity 和发布校验。

## 已封住的假通过入口

1. 阈值键集合固定，config 只能更严，不能放宽到任意大数。
2. 20 个 selector 与 E1 映射有独立摘要，不能缩成一个无关通过测试。
3. 参数化 selector 必须展开成同一组 34 个具体 node；失败、错误或跳过均拒绝。
4. pytest 使用 `--noconftest`，禁用插件自动加载、用户 site、继承 addopts 和 MPS fallback。
5. config、输入、20 个 source files、测试 node、expanded node、环境和 clean commit 均入指纹。
6. setup 后修改 kernel、scale、support、tau、sigma、mask、active indices、voxel spacing
   或 device-side indices 都会在 solver/scorer 前拒绝。
7. setup、oracle audit、solve、scorer 的逻辑与底层物理调用账本分段精确核对。
8. attestation 有 canonical/raw 双哈希；validation report 与 release manifest 再次只读复核。

## 仍然没有证明什么

- `Gate B = NOT_RUN`：没有 scalar/block/factor/graph-PCGLS 同调用性能比较。
- 没有 fresh、真实 OERF 或 held-out session 结果。
- 没有独立 flow-off calibration scale；本 fixture 只用于 mechanics。
- 没有证明 FM-CG-PDNO、DeepONet、FNO、FFNO、NeRIF 或 TDBOST 谁更好。
- 执行环境记录了 Torch、NumPy、pytest 完整安装树，但仍是同一台本机，状态明确为
  `NOT_ESTABLISHED_LOCAL_HOST_FINGERPRINT_ONLY`；它不是隔离容器或供应链证明。

## 证据入口

- [正式 attestation JSON](../demo_t16_operator/results/psu_b0_gate_a_attestation/attestation.json)
- [独立 validation report](../demo_t16_operator/results/psu_b0_gate_a_attestation/validation_report.json)
- [发布 checksums](../demo_t16_operator/results/psu_b0_gate_a_attestation/release_checksums.sha256)
- [34 个具体测试 node](../demo_t16_operator/results/psu_b0_gate_a_attestation/collected_nodes.txt)
- [完整设计与 Gate B 门禁](covariance_majorized_pdhg_design_2026-07-17.md)

## 复核命令

```bash
.venv/bin/python site_tools/validate_psu_b0_gate_a_attestation.py \
  --evidence demo_t16_operator/results/psu_b0_gate_a_attestation \
  --no-write
```

复核通过只能写成“Gate A mechanics attested”。下一步若要回答算法有没有用，必须
另行冻结 Gate B 数据、基线、预算、指标和失败门；不能用本页数值替代性能实验。
