# PSU B0 PDHG scale smoke v2: infrastructure-only amendment

状态：`PREREGISTERED_NOT_RUN`

本文件不是新的科学协议。除下述 MPS 导出修复与相应审计字段外，父协议
`psu_b0_pdhg_preregistered_protocol_2026-07-17.md` 的数据、算法、候选、门槛、
计时和 claim boundary 全部继续生效。

## 1. v1 失败事实

v1 绑定 commit：
`04f0f8877dee500ad77bd184453166e3b1e325d1`。

父配置与协议：

- config：`demo_t16_operator/configs/psu_b0_pdhg_scale_smoke_v1.json`，
  SHA-256 `57a172e74cbab96173e463651d4420ef4306e4c245b3eb6bb5e4c8b03699fb01`；
- protocol：`docs/psu_b0_pdhg_preregistered_protocol_2026-07-17.md`，
  SHA-256 `73e98b9f8af615c86cdaa94579d537e6b8d05642d8041cd7f5dfe16ce6fd93e3`。

v1 原始失败工件只保存在本地私有审计库。其
`preflight_invalid.json` SHA-256 为
`12af605565568c744183858d52e93fe7095ac4773e465ef8e84c5238030f338b`，
`README.md` SHA-256 为
`62a73346209589d93e4b09a27a946585f342911bfa007d27e39b617c02938296`。
公开脱敏摘要为
`docs/psu_b0_pdhg_v1_preflight_invalid_public_summary_2026-07-17.json`，
SHA-256 `99b0b84b23f894a53c873077c60e3dfa32dcd95958edcfd206818ff653e26601`。

不可改写的事实：

- 状态为 `PDHG_PREFLIGHT_INVALID`，不是算法性能 `NO-GO`，也不是通过；
- 12/12 条 stress trajectory 完成了各自 32 次 solver iteration，随后在审计导出
  阶段以同一 `TypeError` 失败；
- 失败原因是 MPS tensor 被要求在一次 `.to(...)` 中同时搬到 CPU 并转换为
  `float64`；
- stress 更新不读取 truth，但其 observation 来自父协议已披露的
  truth-derived oracle-scale synthetic construction；
- truth-based performance metric row 数为 0，candidate ranking 未生成；
- v1 的任何中间 tensor、trajectory 或计时不得并入 v2。

公开仓库只保留脱敏失败摘要；原始失败 JSON、环境详情和私有 geometry 路径不上传。

## 2. 唯一允许的实现修复

原失败导出：

```python
value.detach().to("cpu", torch.float64).numpy()
```

v2 唯一允许的数值路径修复：

```python
value.detach().to(device="cpu").to(dtype=torch.float64).numpy()
```

即先以原 `float32` 搬离 MPS，再在 CPU 上转换为 `float64` 供 NumPy 审计。求解器、
operator、whitener、prox、history 和正式 metric 仍使用父协议固定的 MPS
`float32`。新增跨平台 fake-MPS 单元测试必须拒绝合并的 device+dtype 转换，并
验证上述两步调用顺序。

允许同时保留已在 v1 开封前完成的成功/失败 bundle 原子发布修复。这只影响工件
完整性，不改变数值路径。

## 3. 明确冻结不变的科学内容

下列内容相对 v1 必须逐字节或逐值不变：

1. replicate `[0, 8]` 与每个 replicate 的 8 个 reaction morphology；
2. geometry、support、grid、9 views、rays、finite-aperture samples 及全部 source
   SHA-256；
3. runtime geometry anchor 与 replicate 0/8 deterministic-refit whitener anchors；
4. clean-truth-derived synthetic `scale_by_view` 及“仅 oracle-scale diagnostic”的
   claim boundary；
5. `A0`、`b0`、`x=s*z`、measurement normalization 与 `n_D` 定义；
6. PDHG 更新、初始化、TV/Huber prox、`theta=1`、Huber `delta=0.5`、
   `eta={0.90,0.50}`；
7. `2 penalties x 4 alpha x 4 K = 32` 候选网格；
8. 17 controls、49 methods、784 paired metric rows；
9. 每 iteration `1F + 1A + 1D + 1D^T`，以及 scorer/norm/timing 的独立账本；
10. 12-run stress gate、全部阈值、512-row validity、ranking、decision gates 与
    AB/BA timing；
11. fresh seeds、held-out cameras、实验 flow truth 与 neural training 继续不可达。

禁止因 v1 失败而改变 step size、alpha、K、baseline、门槛、数据、noise、scale、
dtype 或 device；禁止删除失败 trajectory；禁止把 solver 搬到 CPU；禁止在看到 v2
性能后回写本文件或配置。

## 4. v2 执行顺序

1. 代码、测试、本文件与 v2 config 提交到新的 clean commit；
2. 使用同一个 `.venv/bin/python` 运行 v2 config 冻结的完整 E1 node set；
3. 在 geometry/truth 打开前校验 config、protocol、六个 source SHA 与 clean HEAD；
4. 从磁盘重新加载 geometry，并 exact-match runtime geometry anchor；
5. 对 replicate 0/8 从头重建 observation/whitener，并逐 buffer exact-match whitener
   anchors；不得复用 v1 进程状态；
6. 从头运行完整 12-run stress gate；任一失败即只发布
   `PDHG_PREFLIGHT_INVALID`；
7. 仅在 stress 全部通过后，才运行 32 candidates、17 controls、784 rows 与冻结
   AB/BA timing；
8. 成功或失败 bundle 均先写同父目录 staging，完整 checksum 后原子发布；
9. v2 输出目录固定为
   `demo_t16_operator/results/psu_b0_pdhg_scale_smoke_v2_infrastructure_amendment`。

## 5. 允许的结论

v2 最高仍是两个已见 replicate 的 E2、post-open、oracle-scale mechanism
diagnostic。即使门槛通过，也不得称为部署算法、泛化验证、SOTA 或可投稿最终结论。
若门槛失败，必须报告冻结 scalar-step PDHG 网格的具体 `NO-GO`；若基础设施再次
失败，必须保留新的 invalid bundle，不能继续修补后在同一协议名下重跑。
