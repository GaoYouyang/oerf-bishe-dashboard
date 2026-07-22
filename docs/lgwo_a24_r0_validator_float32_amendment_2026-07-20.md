# LGWO-A24 R0 独立验证器修正 VA1

修正 ID：`R0-VA1-FLOAT32-PROVENANCE-TOLERANCE`  
日期：2026-07-20  
正式运行 source commit：`9021ff9`  
正式结果目录：`demo_t16_operator/results/lgwo_a24_r0_semiconvergence_v1`

## 触发原因

正式 R0 运行完成后，独立 validator 在第一条 validation provenance 上停止：

```text
observed relative_noise = 0.019999999552965164
expected JSON value      = 0.02
```

source config 中的十进制噪声等级在 split 中按 `torch.float32` 保存，再写入 CSV。原 validator 对所有浮点值统一使用 `rtol=2e-9, atol=2e-10`，这个容差错误地要求 float32 配置回写达到接近 float64 的十进制一致性。

## 唯一修正

只对 `relative_noise` 的 **source-JSON 到 recorded-float32 provenance 比较** 使用：

```text
rtol = 1e-6
atol = 1e-8
```

以下内容全部不变：

- R0 config 与 canonical SHA；
- 正式 result package 与 checksums；
- 24-step trajectory 与所有底层 metric；
- validation-global checkpoint 选择；
- 八种 selector；
- field/gradient/front/residual 定义；
- 20,000 次 bootstrap、seed 与 paired unit；
- truth-headroom、diversity、heldout-B、数值和调用门；
- 状态词和 claim boundary。

## 审计绑定

修正后的 validator 必须在 tracked worktree clean 的提交上运行。validation report 同时记录：

- 正式运行 commit 及当时 validator SHA-256；
- 本修正提交及 amended validator SHA-256；
- 本文件 SHA-256；
- `scientific_gate_changed=false`；
- `selector_changed=false`；
- `bootstrap_changed=false`；
- `result_package_changed=false`。

这是一项机器表示容差修正，不允许据此改变或美化 R0 的 NO-GO 科学判决。
